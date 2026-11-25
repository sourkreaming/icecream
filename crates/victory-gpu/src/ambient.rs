//! A least amount of indirect light.
//!
//! Grand Theft Auto VI's indirect diffuse light comes from probes it fills
//! by tracing rays, which do not run here yet: its program that resolves
//! them (`CS_ScreenProbes_TemporalResolve_*`) leaves next to nothing, and
//! whatever the sun does not reach directly is black. Right after that
//! program, its output is replaced by a constant (with a subtle tint):
//! the title's own lighting then applies it with the surfaces' colors.
//!
//! `VICTORY_AMBIENT=<level>` sets it (`0.6` by default, `0` for black);
//! `off` keeps the title's own resolve output untouched.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// The programs it follows, by the start of their name.
pub(crate) const PROGRAM: &[u8] = b"CS_ScreenProbes_TemporalResolve";
/// Where they write the indirect diffuse light (R9G9B9E5 as 32-bit
/// integers): root parameter and place in its table.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
/// Where the resolve reads the scene's depth (reversed, D32 with stencil):
/// root parameter and place in its table.
pub(crate) const DEPTH: (u8, u32) = (0, 464);

/// 0.6 sets balanced indoor/shadowed indirect diffuse light without rays;
/// `VICTORY_AMBIENT=off` leaves what the title's resolve wrote.
const DEFAULT: f32 = 0.6;

/// The floor; `None` if off.
pub(crate) fn level() -> Option<f32> {
    static LEVEL: OnceLock<f32> = OnceLock::new();
    let level = *LEVEL.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT").unwrap_or_default();
        if set.trim() == "off" {
            return -1.0;
        }
        set.trim()
            .parse()
            .ok()
            .filter(|l: &f32| l.is_finite() && *l >= 0.0)
            .unwrap_or(DEFAULT)
    });
    (level >= 0.0).then_some(level)
}

/// The light's colour: warm white, or what `VICTORY_AMBIENT_TINT=r,g,b` gives.
fn tint() -> [f32; 3] {
    static SET: OnceLock<[f32; 3]> = OnceLock::new();
    *SET.get_or_init(|| {
        let set: Vec<f32> = std::env::var("VICTORY_AMBIENT_TINT")
            .unwrap_or_default()
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        match set[..] {
            [r, g, b] => [r, g, b],
            _ => [1.0, 0.98, 0.94],
        }
    })
}

/// Depth-based crease/contact occlusion `[strength, radius_m, near_m, tan_half_fov]`,
/// overridden via `VICTORY_AMBIENT_SHADE=s,r,n,t` (`0` strength disables).
fn shade() -> [f32; 4] {
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [0.5, 0.6, 0.1, 0.45];
        let set = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        shade
    })
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(UAV(u0), SRV(t0))"
cbuffer Floor : register(b0) {
    float level; float3 tint;
    float strength; float radius; float near; float tan_y;
};
RWTexture2D<uint> light : register(u0);
Texture2D<float> depth : register(t0);
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = near / max(d, 1e-7);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
float open_to_light(int2 p, float2 size) {
    float3 c = seen(p, size);
    if (c.z > 400.0) return 1.0;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, c) > 0.0) n = -n;
    float px = radius * size.y / (2.0 * tan_y * c.z);
    if (px < 1.5) return 1.0;
    px = min(px, 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    float sum = 0.0;
    const int N = 16;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + noise * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + int2(round(o)), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(most, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    light.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float open = strength > 0.0 ? open_to_light(int2(at.xy), float2(w, h)) : 1.0;
    float3 floor_light = tint * level * open;
    float3 existing = unpack(light[at.xy]);
    float peak = max(existing.r, max(existing.g, existing.b));
    // Reject NaN/Inf/spikes and uninitialized magenta probe garbage while keeping
    // valid brighter bounce (such as sky/water scatter).
    bool garbage = (peak != peak) || peak > 64.0
        || (existing.r > existing.g * 2.2 + 0.05 && existing.b > existing.g * 2.2 + 0.05);
    light[at.xy] = pack(garbage ? floor_light : max(existing, floor_light));
}
"#;

/// Views, used in turn: more than the passes of every list in flight.
const SLOTS: u32 = 1024;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("ambient"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    let text = |b: &ID3DBlob| {
        // SAFETY: the blob's own bytes.
        unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }
            .to_vec()
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            tracing::warn!("ambient light: {}", String::from_utf8_lossy(&text(&errors)));
        }
        return Err(e);
    }
    let code = text(&code.ok_or_else(windows::core::Error::from_thread)?);
    // SAFETY: valid compiled DXBC with embedded root signature on a live device.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&pso_desc)?;
        let views: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: SLOTS * 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })?;
        let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        Ok(Pass {
            root,
            pso,
            views,
            step,
            next: AtomicU32::new(0),
        })
    }
}

fn pass(device: &ID3D12Device) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| make(device).ok()).as_ref()
}

/// Fills indirect diffuse light (`light`, R9G9B9E5 stored as `R32_UINT`)
/// to `level()` with optional depth-based occlusion after
/// `CS_ScreenProbes_TemporalResolve*`; caller clears its bindings when `true`.
pub(crate) fn apply(list: &HostList, light: This, depth: Option<This>) -> bool {
    let Some(level) = level() else {
        return false;
    };
    let Some(host) = host() else {
        return false;
    };
    let Some(light) = crate::resource::host_resource(light) else {
        return false;
    };
    let Some(pass) = pass(&host.device) else {
        return false;
    };
    // SAFETY: live texture resource.
    let d = unsafe { light.GetDesc() };
    static ONCE: std::sync::Once = std::sync::Once::new();
    if d.Format != DXGI_FORMAT_R32_UINT {
        ONCE.call_once(|| tracing::warn!(format = d.Format.0, "no ambient light: another format"));
        return false;
    }
    let depth = depth.and_then(crate::resource::host_resource).filter(|t| {
        // SAFETY: a live resource.
        let t = unsafe { t.GetDesc() };
        t.Format == DXGI_FORMAT_R32G8X24_TYPELESS && (t.Width, t.Height) == (d.Width, d.Height)
    });
    if let Some(depth) = depth {
        list.keep(depth);
    }
    ONCE.call_once(|| {
        tracing::info!(level, width = d.Width, height = d.Height, shaded = depth.is_some(), "ambient light");
    });
    let slot = pass.next.fetch_add(1, Ordering::Relaxed) % SLOTS * 2;
    list.keep(light);
    let l = &list.list;
    let barrier = crate::host::uav_barrier(Some(light));
    // SAFETY: an open list; a live texture in UAV state kept alive by `list`.
    unsafe {
        l.ResourceBarrier(&[barrier]);
        let cpu = pass.views.GetCPUDescriptorHandleForHeapStart();
        let gpu = pass.views.GetGPUDescriptorHandleForHeapStart();
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot * pass.step) as usize,
        };
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device
            .CreateUnorderedAccessView(light, None::<&ID3D12Resource>, Some(&view), at);
        let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let beside = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: at.ptr + pass.step as usize,
        };
        host.device.CreateShaderResourceView(depth, Some(&seen), beside);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        let [r, g, b] = tint();
        let [strength, radius, near, tan_y] = shade();
        let strength = if depth.is_some() { strength } else { 0.0 };
        let constants = [level, r, g, b, strength, radius, near, tan_y];
        l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(slot * pass.step),
            },
        );
        l.Dispatch((d.Width as u32).div_ceil(8), d.Height.div_ceil(8), 1);
        l.ResourceBarrier(&[crate::host::uav_barrier(Some(light))]);
    }
    true
}

#[cfg(test)]
mod tests {
    /// The pass compiles with the system's compiler and the host takes it.
    #[test]
    fn ambient_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        super::make(&host.device).expect("the ambient pass");
    }
}
