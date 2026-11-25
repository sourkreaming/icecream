//! NVIDIA Streamline SDK (`v2.14.16`) integration for DLSS Super Resolution,
//! Reflex Low Latency, and DLSS Frame Generation (`sl.dlss`, `sl.reflex`,
//! `sl.dlss_g`).
//!
//! Opt-in via environment variables:
//! - `VICTORY_DLSS=balanced|quality|performance|ultra|dlaa|1|0`
//! - `VICTORY_DLSS_FG=on|1|2x|auto|dynamic|0`
//! - `VICTORY_REFLEX=on|boost|1|0` (forced on automatically when `VICTORY_DLSS_FG` is active)
//! - `VICTORY_STREAMLINE_DIR=<path>` (defaults to the directory of the current executable)
//!
//! Uses Streamline's manual-hooking path (`eUseManualHooking |
//! eUseFrameBasedResourceTagging | eDisableCLStateTracking`) so native D3D12
//! device and command lists run unproxied with zero overhead except for the
//! specific factory / command-queue / swap-chain creation and present hooks.

use std::ffi::{CStr, c_char, c_void};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use windows::Win32::Foundation::{FreeLibrary, HMODULE, LUID};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATES, ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::Win32::Graphics::Dxgi::IDXGIFactory2;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{Interface, PCSTR, PCWSTR};

const SL_SDK_VERSION: u64 = (2 << 48) | (14 << 32) | (16 << 16) | 0xfedc;

const FEATURE_DLSS: u32 = 0;
const FEATURE_REFLEX: u32 = 3;
const FEATURE_PCL: u32 = 4;
const FEATURE_DLSS_G: u32 = 1000;

const BUFFER_DEPTH: u32 = 0;
const BUFFER_MVEC: u32 = 1;
const BUFFER_HUDLESS: u32 = 2;
const BUFFER_SCALING_IN: u32 = 3;
const BUFFER_SCALING_OUT: u32 = 4;
const BUFFER_EXPOSURE: u32 = 13;
const BUFFER_UI_COLOR_ALPHA: u32 = 23;

const LIFECYCLE_ONLY_VALID_NOW: u32 = 0;
const LIFECYCLE_VALID_UNTIL_PRESENT: u32 = 1;

const PREF_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
const PREF_DISABLE_DEBUG_TEXT: u64 = 1 << 1;
const PREF_USE_MANUAL_HOOKING: u64 = 1 << 2;
const PREF_USE_FRAME_BASED_TAGGING: u64 = 1 << 7;

const LOG_LEVEL_DEFAULT: u32 = 1;
const ENGINE_CUSTOM: u32 = 0;
const RENDER_API_D3D12: u32 = 1;

const INVALID_FLOAT: f32 = f32::MAX;
const INVALID_UINT: u32 = u32::MAX;

const BOOL_FALSE: u8 = 0;
const BOOL_TRUE: u8 = 1;
const BOOL_INVALID: u8 = 2;

const PCL_SIMULATION_START: u32 = 0;
const PCL_RENDER_SUBMIT_START: u32 = 2;
const PCL_RENDER_SUBMIT_END: u32 = 3;
const PCL_PRESENT_START: u32 = 4;
const PCL_PRESENT_END: u32 = 5;

const DLSSG_FLAG_RETAIN_WHEN_OFF: u32 = 1 << 3;
const DLSSG_FLAG_MENU_DETECTION: u32 = 1 << 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct StructType {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BaseStructure {
    next: *mut BaseStructure,
    struct_type: StructType,
    struct_version: usize,
}

impl BaseStructure {
    const fn new(struct_type: StructType, struct_version: u32) -> Self {
        Self {
            next: std::ptr::null_mut(),
            struct_type,
            struct_version: struct_version as usize,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Extent {
    top: u32,
    left: u32,
    width: u32,
    height: u32,
}

#[repr(C)]
struct Preferences {
    base: BaseStructure,
    show_console: u8,
    log_level: u32,
    paths_to_plugins: *const *const u16,
    num_paths_to_plugins: u32,
    path_to_logs_and_data: *const u16,
    allocate_callback: *const c_void,
    release_callback: *const c_void,
    log_message_callback: Option<extern "C" fn(u32, *const c_char)>,
    flags: u64,
    features_to_load: *const u32,
    num_features_to_load: u32,
    application_id: u32,
    engine: u32,
    engine_version: *const c_char,
    project_id: *const c_char,
    render_api: u32,
}

#[repr(C)]
struct AdapterInfo {
    base: BaseStructure,
    device_luid: *mut u8,
    device_luid_size_in_bytes: u32,
    vk_physical_device: *mut c_void,
}

#[repr(C)]
struct Resource {
    base: BaseStructure,
    kind: u8,
    native: *mut c_void,
    memory: *mut c_void,
    view: *mut c_void,
    state: u32,
    width: u32,
    height: u32,
    native_format: u32,
    mip_levels: u32,
    array_layers: u32,
    gpu_virtual_address: u64,
    flags: u32,
    usage: u32,
    internal_flags: u16,
    reserved: u16,
}

impl Resource {
    fn tex2d(native: *mut c_void, state: u32, width: u32, height: u32, format: u32) -> Self {
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0x3a9d_70cf,
                    data2: 0x2418,
                    data3: 0x4b72,
                    data4: [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61],
                },
                1,
            ),
            kind: 0, // ResourceType::eTex2d
            native,
            memory: std::ptr::null_mut(),
            view: std::ptr::null_mut(),
            state,
            width,
            height,
            native_format: format,
            mip_levels: 1,
            array_layers: 1,
            gpu_virtual_address: 0,
            flags: 0,
            usage: 0,
            internal_flags: 0,
            reserved: 0,
        }
    }
}

#[repr(C)]
struct ResourceTag {
    base: BaseStructure,
    resource: *const Resource,
    kind: u32,
    lifecycle: u32,
    extent: Extent,
}

impl ResourceTag {
    fn new(resource: *const Resource, kind: u32, lifecycle: u32, extent: Extent) -> Self {
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0x4c6a_5aad,
                    data2: 0xb445,
                    data3: 0x496c,
                    data4: [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53],
                },
                1,
            ),
            resource,
            kind,
            lifecycle,
            extent,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ViewportHandle {
    base: BaseStructure,
    value: u32,
}

impl ViewportHandle {
    const fn new(value: u32) -> Self {
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0x171b_6435,
                    data2: 0x9b3c,
                    data3: 0x4fc8,
                    data4: [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4],
                },
                1,
            ),
            value,
        }
    }
}

#[repr(C)]
struct Constants {
    base: BaseStructure,
    camera_view_to_clip: [[f32; 4]; 4],
    clip_to_camera_view: [[f32; 4]; 4],
    clip_to_lens_clip: [[f32; 4]; 4],
    clip_to_prev_clip: [[f32; 4]; 4],
    prev_clip_to_clip: [[f32; 4]; 4],
    jitter_offset: [f32; 2],
    mvec_scale: [f32; 2],
    camera_pinhole_offset: [f32; 2],
    camera_pos: [f32; 3],
    camera_up: [f32; 3],
    camera_right: [f32; 3],
    camera_fwd: [f32; 3],
    camera_near: f32,
    camera_far: f32,
    camera_fov: f32,
    camera_aspect_ratio: f32,
    motion_vectors_invalid_value: f32,
    depth_inverted: u8,
    camera_motion_included: u8,
    motion_vectors_3d: u8,
    reset: u8,
    orthographic_projection: u8,
    motion_vectors_dilated: u8,
    motion_vectors_jittered: u8,
    min_relative_linear_depth_object_separation: f32,
}

impl Constants {
    fn new(aspect: f32) -> Self {
        let proj = perspective(1.0, aspect.max(0.1), 0.1, 10000.0);
        let inv_proj = inv_perspective(1.0, aspect.max(0.1), 0.1, 10000.0);
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0xdcd3_5ad7,
                    data2: 0x4e4a,
                    data3: 0x4bad,
                    data4: [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe],
                },
                2,
            ),
            camera_view_to_clip: proj,
            clip_to_camera_view: inv_proj,
            clip_to_lens_clip: IDENTITY,
            clip_to_prev_clip: IDENTITY,
            prev_clip_to_clip: IDENTITY,
            jitter_offset: [0.0, 0.0],
            mvec_scale: [1.0, 1.0],
            camera_pinhole_offset: [0.0, 0.0],
            camera_pos: [0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0],
            camera_right: [1.0, 0.0, 0.0],
            camera_fwd: [0.0, 0.0, 1.0],
            camera_near: 0.1,
            camera_far: 10000.0,
            camera_fov: 1.0,
            camera_aspect_ratio: aspect.max(0.1),
            motion_vectors_invalid_value: INVALID_FLOAT,
            depth_inverted: BOOL_TRUE,
            camera_motion_included: BOOL_TRUE,
            motion_vectors_3d: BOOL_FALSE,
            reset: BOOL_FALSE,
            orthographic_projection: BOOL_FALSE,
            motion_vectors_dilated: BOOL_FALSE,
            motion_vectors_jittered: BOOL_FALSE,
            min_relative_linear_depth_object_separation: 40.0,
        }
    }
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn perspective(fov_y: f32, aspect: f32, z_near: f32, z_far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let q = z_far / (z_far - z_near);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, q, 1.0],
        [0.0, 0.0, -z_near * q, 0.0],
    ]
}

fn inv_perspective(fov_y: f32, aspect: f32, z_near: f32, z_far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let q = z_far / (z_far - z_near);
    let inv_nq = -1.0 / (z_near * q);
    [
        [aspect / f, 0.0, 0.0, 0.0],
        [0.0, 1.0 / f, 0.0, 0.0],
        [0.0, 0.0, 0.0, inv_nq],
        [0.0, 0.0, 1.0, -q * inv_nq],
    ]
}

#[repr(C)]
struct DLSSOptions {
    base: BaseStructure,
    mode: u32,
    output_width: u32,
    output_height: u32,
    sharpness: f32,
    pre_exposure: f32,
    exposure_scale: f32,
    color_buffers_hdr: u8,
    indicator_invert_axis_x: u8,
    indicator_invert_axis_y: u8,
    dlaa_preset: u32,
    quality_preset: u32,
    balanced_preset: u32,
    performance_preset: u32,
    ultra_performance_preset: u32,
    ultra_quality_preset: u32,
    use_auto_exposure: u8,
    alpha_upscaling_enabled: u8,
}

impl Default for DLSSOptions {
    fn default() -> Self {
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0x6ac8_26e4,
                    data2: 0x4c61,
                    data3: 0x4101,
                    data4: [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8],
                },
                3,
            ),
            mode: 0,
            output_width: INVALID_UINT,
            output_height: INVALID_UINT,
            sharpness: 0.0,
            pre_exposure: 1.0,
            exposure_scale: 1.0,
            color_buffers_hdr: BOOL_TRUE,
            indicator_invert_axis_x: BOOL_FALSE,
            indicator_invert_axis_y: BOOL_FALSE,
            dlaa_preset: 0,
            quality_preset: 0,
            balanced_preset: 0,
            performance_preset: 0,
            ultra_performance_preset: 0,
            ultra_quality_preset: 0,
            use_auto_exposure: BOOL_FALSE,
            alpha_upscaling_enabled: BOOL_FALSE,
        }
    }
}

#[repr(C)]
struct DLSSGOptions {
    base: BaseStructure,
    mode: u32,
    num_frames_to_generate: u32,
    flags: u32,
    dynamic_res_width: u32,
    dynamic_res_height: u32,
    num_back_buffers: u32,
    mvec_depth_width: u32,
    mvec_depth_height: u32,
    color_width: u32,
    color_height: u32,
    color_buffer_format: u32,
    mvec_buffer_format: u32,
    depth_buffer_format: u32,
    hud_less_buffer_format: u32,
    ui_buffer_format: u32,
    on_error_callback: *const c_void,
    b_reserved15: u8,
    queue_parallelism_mode: u32,
    enable_ui_recomposition: u8,
    dynamic_target_frame_rate: f32,
}

impl Default for DLSSGOptions {
    fn default() -> Self {
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0xfac5_f1cb,
                    data2: 0x2dfd,
                    data3: 0x4f36,
                    data4: [0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5],
                },
                5,
            ),
            mode: 0,
            num_frames_to_generate: 1,
            flags: 0,
            dynamic_res_width: 0,
            dynamic_res_height: 0,
            num_back_buffers: 0,
            mvec_depth_width: 0,
            mvec_depth_height: 0,
            color_width: 0,
            color_height: 0,
            color_buffer_format: 0,
            mvec_buffer_format: 0,
            depth_buffer_format: 0,
            hud_less_buffer_format: 0,
            ui_buffer_format: 0,
            on_error_callback: std::ptr::null(),
            b_reserved15: BOOL_INVALID,
            queue_parallelism_mode: 0,
            enable_ui_recomposition: BOOL_FALSE,
            dynamic_target_frame_rate: 0.0,
        }
    }
}

#[repr(C)]
struct DLSSGState {
    base: BaseStructure,
    estimated_vram_usage_in_bytes: u64,
    status: u32,
    min_width_or_height: u32,
    num_frames_actually_presented: u32,
    num_frames_to_generate_max: u32,
    b_reserved4: u8,
    b_is_vsync_support_available: u8,
    inputs_processing_completion_fence: *mut c_void,
    last_present_inputs_processing_completion_fence_value: u64,
    b_is_dynamic_mfg_supported: u8,
}

impl Default for DLSSGState {
    fn default() -> Self {
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0xcc8a_c8e1,
                    data2: 0xa179,
                    data3: 0x44f5,
                    data4: [0x97, 0xfa, 0xe7, 0x41, 0x12, 0xf9, 0xbc, 0x61],
                },
                4,
            ),
            estimated_vram_usage_in_bytes: 0,
            status: 0,
            min_width_or_height: 0,
            num_frames_actually_presented: 0,
            num_frames_to_generate_max: 0,
            b_reserved4: 0,
            b_is_vsync_support_available: 0,
            inputs_processing_completion_fence: std::ptr::null_mut(),
            last_present_inputs_processing_completion_fence_value: 0,
            b_is_dynamic_mfg_supported: 0,
        }
    }
}

#[repr(C)]
struct ReflexOptions {
    base: BaseStructure,
    mode: u32,
    frame_limit_us: u32,
    use_markers_to_optimize: u8,
    virtual_key: u16,
    id_thread: u32,
}

impl Default for ReflexOptions {
    fn default() -> Self {
        Self {
            base: BaseStructure::new(
                StructType {
                    data1: 0xf03a_f81a,
                    data2: 0x6d0b,
                    data3: 0x4902,
                    data4: [0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34],
                },
                1,
            ),
            mode: 0,
            frame_limit_us: 0,
            use_markers_to_optimize: 0,
            virtual_key: 0,
            id_thread: 0,
        }
    }
}

type FnSlInit = unsafe extern "C" fn(*const Preferences, u64) -> i32;
type FnSlShutdown = unsafe extern "C" fn() -> i32;
type FnSlIsFeatureSupported = unsafe extern "C" fn(u32, *const AdapterInfo) -> i32;
type FnSlSetD3DDevice = unsafe extern "C" fn(*mut c_void) -> i32;
type FnSlUpgradeInterface = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type FnSlGetNativeInterface = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
type FnSlGetFeatureFunction =
    unsafe extern "C" fn(u32, *const c_char, *mut *mut c_void) -> i32;
type FnSlGetNewFrameToken = unsafe extern "C" fn(*mut *const BaseStructure, *const u32) -> i32;
type FnSlSetConstants =
    unsafe extern "C" fn(*const Constants, *const BaseStructure, *const ViewportHandle) -> i32;
type FnSlSetTagForFrame = unsafe extern "C" fn(
    *const BaseStructure,
    *const ViewportHandle,
    *const ResourceTag,
    u32,
    *mut c_void,
) -> i32;
type FnSlEvaluateFeature = unsafe extern "C" fn(
    u32,
    *const BaseStructure,
    *const *const BaseStructure,
    u32,
    *mut c_void,
) -> i32;
type FnSlDLSSSetOptions =
    unsafe extern "C" fn(*const ViewportHandle, *const DLSSOptions) -> i32;
type FnSlDLSSGSetOptions =
    unsafe extern "C" fn(*const ViewportHandle, *const DLSSGOptions) -> i32;
type FnSlDLSSGGetState =
    unsafe extern "C" fn(*const ViewportHandle, *mut DLSSGState, *const DLSSGOptions) -> i32;
type FnSlReflexSetOptions = unsafe extern "C" fn(*const ReflexOptions) -> i32;
type FnSlReflexSleep = unsafe extern "C" fn(*const BaseStructure) -> i32;
type FnSlPCLSetMarker = unsafe extern "C" fn(u32, *const BaseStructure) -> i32;

/// Parsed Streamline runtime configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    /// `sl::DLSSMode`: 0 off, 1 perf, 2 balanced, 3 quality, 4 ultra-perf, 6 DLAA.
    pub(crate) dlss_mode: Option<u32>,
    /// `sl::DLSSGMode`: 1 On, 2 Auto, 3 Dynamic.
    pub(crate) dlss_g_mode: Option<u32>,
    /// Number of interpolated frames requested (`1` = 2x, `2` = 3x, `3` = 4x).
    pub(crate) dlss_g_frames: u32,
    /// `sl::ReflexMode`: 0 off, 1 low latency, 2 low latency + boost.
    pub(crate) reflex_mode: Option<u32>,
}

impl Config {
    fn from_env() -> Self {
        let dlss = std::env::var("VICTORY_DLSS").ok();
        let fg = std::env::var("VICTORY_DLSS_FG").ok();
        let reflex = std::env::var("VICTORY_REFLEX").ok();
        Self::parse(dlss.as_deref(), fg.as_deref(), reflex.as_deref())
    }

    pub(crate) fn parse(
        dlss: Option<&str>,
        fg: Option<&str>,
        reflex: Option<&str>,
    ) -> Self {
        let dlss_mode = dlss.and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => None,
            "1" | "on" | "true" | "balanced" => Some(2),
            "performance" | "perf" => Some(1),
            "quality" | "max_quality" => Some(3),
            "ultra" | "ultra_performance" => Some(4),
            "ultra_quality" => Some(5),
            "dlaa" => Some(6),
            _ => Some(2),
        });
        let (dlss_g_mode, dlss_g_frames) = fg
            .map(|v| match v.trim().to_ascii_lowercase().as_str() {
                "" | "0" | "off" | "false" | "no" => (None, 1),
                "auto" => (Some(2), 1),
                "dynamic" => (Some(3), 1),
                "3x" => (Some(1), 2),
                "4x" => (Some(1), 3),
                _ => (Some(1), 1),
            })
            .unwrap_or((None, 1));
        let explicit_reflex = reflex.and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => None,
            "boost" | "2" => Some(2),
            _ => Some(1),
        });
        // DLSS-G requires sl.reflex to be active (`eFailReflexNotDetectedAtRuntime`).
        let reflex_mode = match (explicit_reflex, dlss_g_mode) {
            (Some(r), _) => Some(r),
            (None, Some(_)) => Some(1),
            (None, None) => None,
        };
        Self {
            dlss_mode,
            dlss_g_mode,
            dlss_g_frames,
            reflex_mode,
        }
    }

    pub(crate) fn any_enabled(self) -> bool {
        self.dlss_mode.is_some() || self.dlss_g_mode.is_some() || self.reflex_mode.is_some()
    }
}

struct Api {
    _module: HMODULE,
    _proxy_device: Mutex<Option<ID3D12Device>>,
    swapchain_queue: Mutex<Option<windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue>>,
    config: Config,
    dlss_supported: bool,
    dlss_g_supported: bool,
    reflex_supported: bool,
    dlss_g_max_frames: u32,
    dlss_g_dynamic_ok: bool,
    sl_shutdown: FnSlShutdown,
    sl_is_feature_supported: FnSlIsFeatureSupported,
    sl_set_d3d_device: FnSlSetD3DDevice,
    sl_upgrade_interface: FnSlUpgradeInterface,
    sl_get_native_interface: FnSlGetNativeInterface,
    sl_get_feature_function: FnSlGetFeatureFunction,
    sl_get_new_frame_token: FnSlGetNewFrameToken,
    sl_set_constants: FnSlSetConstants,
    sl_set_tag_for_frame: FnSlSetTagForFrame,
    sl_evaluate_feature: FnSlEvaluateFeature,
    sl_dlss_set_options: Option<FnSlDLSSSetOptions>,
    sl_dlssg_set_options: Option<FnSlDLSSGSetOptions>,
    sl_dlssg_get_state: Option<FnSlDLSSGGetState>,
    sl_reflex_set_options: Option<FnSlReflexSetOptions>,
    sl_reflex_sleep: Option<FnSlReflexSleep>,
    sl_pcl_set_marker: Option<FnSlPCLSetMarker>,
}

// SAFETY: Streamline API pointers and HMODULE remain valid for the process
// lifetime after `early_init` / `on_device_created`; mutable state is gated by
// `Mutex`.
unsafe impl Send for Api {}
// SAFETY: same as above.
unsafe impl Sync for Api {}

struct FrameSync {
    frame_index: u32,
    token: usize,
    constants_sent: bool,
    saw_upscale: bool,
    present_count: u64,
    last_fg_log: std::time::Instant,
    fg_presented_accum: u64,
    depth_format: u32,
    mvec_format: u32,
    depth_extent: [u32; 2],
}

impl FrameSync {
    fn new() -> Self {
        Self {
            frame_index: 0,
            token: 0,
            constants_sent: false,
            saw_upscale: false,
            present_count: 0,
            last_fg_log: std::time::Instant::now(),
            fg_presented_accum: 0,
            depth_format: 0,
            mvec_format: 0,
            depth_extent: [0, 0],
        }
    }
}

static STATE: OnceLock<Mutex<Option<Api>>> = OnceLock::new();
static FRAME_SYNC: OnceLock<Mutex<FrameSync>> = OnceLock::new();
const VIEWPORT: ViewportHandle = ViewportHandle::new(0);

fn config() -> Config {
    static CFG: OnceLock<Config> = OnceLock::new();
    *CFG.get_or_init(Config::from_env)
}

extern "C" fn log_callback(kind: u32, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    // SAFETY: NUL-terminated ASCII/UTF-8 log line passed by Streamline.
    let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    match kind {
        2 => tracing::error!(target: "streamline", "{trimmed}"),
        1 => tracing::warn!(target: "streamline", "{trimmed}"),
        _ => tracing::info!(target: "streamline", "{trimmed}"),
    }
}

unsafe fn proc_fn<T>(module: HMODULE, name: &[u8]) -> Option<T> {
    // SAFETY: `name` is NUL-terminated and `module` is a live `sl.interposer.dll`.
    let p = unsafe { GetProcAddress(module, PCSTR(name.as_ptr())) }?;
    // SAFETY: caller matches the exact exported C function signature.
    Some(unsafe { std::mem::transmute_copy(&p) })
}

fn plugin_search_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("VICTORY_STREAMLINE_DIR") {
        let p = PathBuf::from(dir);
        if p.join("sl.interposer.dll").exists() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && dir.join("sl.interposer.dll").exists()
    {
        return Some(dir.to_path_buf());
    }
    None
}

fn wide_nul(s: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    s.encode_wide().chain(std::iter::once(0)).collect()
}

/// Initializes NVIDIA Streamline (`slInit`) before any D3D12 device or DXGI
/// swap chain is created. Safe no-op unless `VICTORY_DLSS`, `VICTORY_DLSS_FG`,
/// or `VICTORY_REFLEX` enables a feature.
pub(crate) fn early_init() {
    let cfg = config();
    if !cfg.any_enabled() {
        return;
    }
    let cell = STATE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_some() {
        return;
    }
    let Some(dir) = plugin_search_dir() else {
        tracing::warn!(
            "Streamline enabled by env, but sl.interposer.dll was not found in the executable dir or VICTORY_STREAMLINE_DIR; falling back"
        );
        return;
    };
    let interposer = dir.join("sl.interposer.dll");
    let interposer_w = wide_nul(interposer.as_os_str());
    // SAFETY: loading a local DLL by full path.
    let Ok(module) = (unsafe { LoadLibraryW(PCWSTR(interposer_w.as_ptr())) }) else {
        tracing::warn!(path = %interposer.display(), "failed to load sl.interposer.dll");
        return;
    };

    // SAFETY: exported functions defined by `sl_core_api.h`.
    let Some((
        sl_init,
        sl_shutdown,
        sl_is_feature_supported,
        sl_set_d3d_device,
        sl_upgrade_interface,
        sl_get_native_interface,
        sl_get_feature_function,
        sl_get_new_frame_token,
        sl_set_constants,
        sl_set_tag_for_frame,
        sl_evaluate_feature,
    )) = (|| unsafe {
        Some((
            proc_fn::<FnSlInit>(module, b"slInit\0")?,
            proc_fn::<FnSlShutdown>(module, b"slShutdown\0")?,
            proc_fn::<FnSlIsFeatureSupported>(module, b"slIsFeatureSupported\0")?,
            proc_fn::<FnSlSetD3DDevice>(module, b"slSetD3DDevice\0")?,
            proc_fn::<FnSlUpgradeInterface>(module, b"slUpgradeInterface\0")?,
            proc_fn::<FnSlGetNativeInterface>(module, b"slGetNativeInterface\0")?,
            proc_fn::<FnSlGetFeatureFunction>(module, b"slGetFeatureFunction\0")?,
            proc_fn::<FnSlGetNewFrameToken>(module, b"slGetNewFrameToken\0")?,
            proc_fn::<FnSlSetConstants>(module, b"slSetConstants\0")?,
            proc_fn::<FnSlSetTagForFrame>(module, b"slSetTagForFrame\0")?,
            proc_fn::<FnSlEvaluateFeature>(module, b"slEvaluateFeature\0")?,
        ))
    })()
    else {
        tracing::warn!("sl.interposer.dll is missing core Streamline exports");
        // SAFETY: freeing the just-loaded DLL handle.
        let _ = unsafe { FreeLibrary(module) };
        return;
    };

    let mut features = Vec::new();
    if cfg.reflex_mode.is_some() {
        features.push(FEATURE_PCL);
        features.push(FEATURE_REFLEX);
    }
    if cfg.dlss_mode.is_some() {
        features.push(FEATURE_DLSS);
    }
    if cfg.dlss_g_mode.is_some() {
        features.push(FEATURE_DLSS_G);
    }
    let plugin_dir_w = wide_nul(dir.as_os_str());
    let plugin_ptrs = [plugin_dir_w.as_ptr()];
    let mut flags =
        PREF_DISABLE_CL_STATE_TRACKING | PREF_USE_MANUAL_HOOKING | PREF_USE_FRAME_BASED_TAGGING;
    if std::env::var_os("VICTORY_SL_DEBUG_TEXT").is_none_or(|v| v == "0") {
        flags |= PREF_DISABLE_DEBUG_TEXT;
    }
    let pref = Preferences {
        base: BaseStructure::new(
            StructType {
                data1: 0x1ca1_0965,
                data2: 0xbf8e,
                data3: 0x432b,
                data4: [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14],
            },
            1,
        ),
        show_console: 0,
        log_level: LOG_LEVEL_DEFAULT,
        paths_to_plugins: plugin_ptrs.as_ptr(),
        num_paths_to_plugins: plugin_ptrs.len() as u32,
        path_to_logs_and_data: std::ptr::null(),
        allocate_callback: std::ptr::null(),
        release_callback: std::ptr::null(),
        log_message_callback: Some(log_callback),
        flags,
        features_to_load: features.as_ptr(),
        num_features_to_load: features.len() as u32,
        application_id: std::env::var("VICTORY_SL_APP_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        engine: ENGINE_CUSTOM,
        engine_version: c"1.0.0".as_ptr(),
        project_id: c"a0f57b54-1daf-4934-90ae-c4035c19df04".as_ptr(),
        render_api: RENDER_API_D3D12,
    };

    // SAFETY: `pref` and referenced slices live for the duration of `slInit`.
    let res = unsafe { (sl_init)(&pref, SL_SDK_VERSION) };
    if res != 0 {
        tracing::warn!(
            result = res,
            dir = %dir.display(),
            "slInit failed; continuing without Streamline"
        );
        // SAFETY: balanced library free on failed init.
        let _ = unsafe { FreeLibrary(module) };
        return;
    }
    tracing::info!(
        dir = %dir.display(),
        dlss = ?cfg.dlss_mode,
        dlss_g = ?cfg.dlss_g_mode,
        dlss_g_frames = cfg.dlss_g_frames,
        reflex = ?cfg.reflex_mode,
        "Streamline v2.14.16 initialized (manual hooking)"
    );
    *guard = Some(Api {
        _module: module,
        _proxy_device: Mutex::new(None),
        swapchain_queue: Mutex::new(None),
        config: cfg,
        dlss_supported: false,
        dlss_g_supported: false,
        reflex_supported: false,
        dlss_g_max_frames: 1,
        dlss_g_dynamic_ok: false,
        sl_shutdown,
        sl_is_feature_supported,
        sl_set_d3d_device,
        sl_upgrade_interface,
        sl_get_native_interface,
        sl_get_feature_function,
        sl_get_new_frame_token,
        sl_set_constants,
        sl_set_tag_for_frame,
        sl_evaluate_feature,
        sl_dlss_set_options: None,
        sl_dlssg_set_options: None,
        sl_dlssg_get_state: None,
        sl_reflex_set_options: None,
        sl_reflex_sleep: None,
        sl_pcl_set_marker: None,
    });
}

/// Registers the host `ID3D12Device` with Streamline, probes feature support,
/// resolves per-feature entry points, configures Reflex, and returns a Streamline
/// proxy `ID3D12Device` to use for `CreateCommandQueue`.
pub(crate) fn on_device_created(device: &ID3D12Device) -> Option<ID3D12Device> {
    let cell = STATE.get()?;
    let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    let api = guard.as_mut()?;

    // SAFETY: passing the live native D3D12 device pointer.
    let set_res = unsafe { (api.sl_set_d3d_device)(device.as_raw()) };
    if set_res != 0 {
        tracing::warn!(result = set_res, "slSetD3DDevice failed; disabling Streamline");
        *guard = None;
        return None;
    }

    // SAFETY: `GetAdapterLuid` returns the device's LUID by value.
    let mut luid: LUID = unsafe { device.GetAdapterLuid() };
    let adapter_info = AdapterInfo {
        base: BaseStructure::new(
            StructType {
                data1: 0x0677_315f,
                data2: 0xa746,
                data3: 0x4492,
                data4: [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4],
            },
            1,
        ),
        device_luid: (&raw mut luid).cast::<u8>(),
        device_luid_size_in_bytes: size_of::<LUID>() as u32,
        vk_physical_device: std::ptr::null_mut(),
    };

    let check = |feature: u32| -> bool {
        // SAFETY: adapter_info points at valid stack LUID.
        let r = unsafe { (api.sl_is_feature_supported)(feature, &adapter_info) };
        if r != 0 {
            tracing::warn!(feature, result = r, "Streamline feature not supported on this GPU/OS");
        }
        r == 0
    };

    if api.config.reflex_mode.is_some() {
        api.reflex_supported = check(FEATURE_REFLEX);
    }
    if api.config.dlss_mode.is_some() {
        api.dlss_supported = check(FEATURE_DLSS);
    }
    if api.config.dlss_g_mode.is_some() {
        api.dlss_g_supported = check(FEATURE_DLSS_G);
    }

    unsafe fn feat_fn<T>(api: &Api, feature: u32, name: &[u8]) -> Option<T> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `name` is NUL-terminated and device was set above.
        let r = unsafe { (api.sl_get_feature_function)(feature, name.as_ptr().cast(), &mut ptr) };
        if r == 0 && !ptr.is_null() {
            // SAFETY: pointer matches the C feature API signature.
            Some(unsafe { std::mem::transmute_copy(&ptr) })
        } else {
            None
        }
    }

    // SAFETY: resolving feature functions after `slSetD3DDevice`.
    unsafe {
        if api.reflex_supported {
            api.sl_reflex_set_options = feat_fn(api, FEATURE_REFLEX, b"slReflexSetOptions\0");
            api.sl_reflex_sleep = feat_fn(api, FEATURE_REFLEX, b"slReflexSleep\0");
            api.sl_pcl_set_marker = feat_fn(api, FEATURE_PCL, b"slPCLSetMarker\0");
        }
        if api.dlss_supported {
            api.sl_dlss_set_options = feat_fn(api, FEATURE_DLSS, b"slDLSSSetOptions\0");
        }
        if api.dlss_g_supported {
            api.sl_dlssg_set_options = feat_fn(api, FEATURE_DLSS_G, b"slDLSSGSetOptions\0");
            api.sl_dlssg_get_state = feat_fn(api, FEATURE_DLSS_G, b"slDLSSGGetState\0");
        }
    }

    if let (Some(mode), Some(set_reflex)) = (api.config.reflex_mode, api.sl_reflex_set_options) {
        let opts = ReflexOptions {
            mode,
            ..Default::default()
        };
        // SAFETY: valid `ReflexOptions`.
        let r = unsafe { set_reflex(&opts) };
        tracing::info!(mode, result = r, "Streamline Reflex configured");
    }

    if api.dlss_g_supported
        && let Some(get_state) = api.sl_dlssg_get_state
    {
        let mut st = DLSSGState::default();
        // SAFETY: valid `ViewportHandle` and `DLSSGState`.
        let r = unsafe { get_state(&VIEWPORT, &mut st, std::ptr::null()) };
        if r == 0 {
            api.dlss_g_max_frames = st.num_frames_to_generate_max.max(1);
            api.dlss_g_dynamic_ok = st.b_is_dynamic_mfg_supported == BOOL_TRUE;
            tracing::info!(
                status = st.status,
                min_dim = st.min_width_or_height,
                max_generated = st.num_frames_to_generate_max,
                dynamic_mfg = api.dlss_g_dynamic_ok,
                vsync_ok = st.b_is_vsync_support_available == BOOL_TRUE,
                "Streamline DLSS-G capabilities"
            );
        } else {
            tracing::warn!(result = r, "slDLSSGGetState failed at init");
        }
    }

    if !api.dlss_g_supported {
        return None;
    }

    // Transfer an explicit ref into `slUpgradeInterface` so `device` stays untouched.
    let cloned = device.clone();
    let mut proxy_ptr = cloned.as_raw();
    std::mem::forget(cloned);
    // SAFETY: manual hooking upgrades the COM pointer in-place to an SL proxy
    // that intercepts `ID3D12Device::CreateCommandQueue`.
    let up = unsafe { (api.sl_upgrade_interface)(&mut proxy_ptr) };
    if up == 0 && !proxy_ptr.is_null() {
        // SAFETY: proxy implements `ID3D12Device` with 1 owned ref.
        let proxy = unsafe { ID3D12Device::from_raw(proxy_ptr) };
        *api._proxy_device.lock().unwrap_or_else(|e| e.into_inner()) = Some(proxy.clone());
        Some(proxy)
    } else {
        tracing::warn!(result = up, "slUpgradeInterface(ID3D12Device) failed");
        None
    }
}

/// Unwraps a command queue created via the SL proxy device: keeps the proxy
/// queue for `CreateSwapChainForHwnd` and returns the native queue for normal
/// `ExecuteCommandLists` / `Signal` calls if Streamline wrapped it.
pub(crate) fn unwrap_command_queue(
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
) -> windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue {
    use windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue;
    let Some(cell) = STATE.get() else {
        return queue;
    };
    let guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    let Some(api) = guard.as_ref() else {
        return queue;
    };
    if !api.dlss_g_supported {
        return queue;
    }
    *api.swapchain_queue.lock().unwrap_or_else(|e| e.into_inner()) = Some(queue.clone());
    let mut native_ptr = std::ptr::null_mut();
    // SAFETY: queries native interface if `queue` is an SL proxy.
    let r = unsafe { (api.sl_get_native_interface)(queue.as_raw(), &mut native_ptr) };
    if r == 0 && !native_ptr.is_null() && native_ptr != queue.as_raw() {
        // SAFETY: `slGetNativeInterface` AddRefs the underlying native queue, or
        // returns a non-owning pointer; AddRef explicitly via `from_raw_borrowed`.
        let native = unsafe { ID3D12CommandQueue::from_raw_borrowed(&native_ptr) }
            .expect("non-null checked")
            .clone();
        tracing::info!("resolved native ID3D12CommandQueue for Streamline manual hooking");
        native
    } else {
        queue
    }
}

/// Returns the proxy queue to pass to `CreateSwapChainForHwnd` (or `fallback`).
pub(crate) fn swapchain_queue(
    fallback: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
) -> windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue {
    STATE
        .get()
        .and_then(|c| {
            c.lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .and_then(|a| {
                    a.swapchain_queue
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                })
        })
        .unwrap_or_else(|| fallback.clone())
}

/// Upgrades an `IDXGIFactory2` in place so `CreateSwapChainForHwnd` creates a
/// Streamline-hooked proxy swap chain when `DLSS-G` is enabled.
pub(crate) fn upgrade_factory(factory: &mut IDXGIFactory2) {
    let Some(cell) = STATE.get() else {
        return;
    };
    let guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    let Some(api) = guard.as_ref() else {
        return;
    };
    if !api.dlss_g_supported {
        return;
    }
    let mut ptr = factory.as_raw();
    // SAFETY: `slUpgradeInterface` replaces `ptr` with an SL DXGI factory proxy.
    let r = unsafe { (api.sl_upgrade_interface)(&mut ptr) };
    if r == 0 && !ptr.is_null() && ptr != factory.as_raw() {
        let old = std::mem::replace(
            factory,
            // SAFETY: proxy implements `IDXGIFactory2`.
            unsafe { IDXGIFactory2::from_raw(ptr) },
        );
        // `slUpgradeInterface` adopts/wraps the base pointer ref rather than
        // incrementing both, so forget the original wrapper to stay balanced.
        std::mem::forget(old);
        tracing::info!("IDXGIFactory2 upgraded for Streamline swap-chain hooking");
    } else if r != 0 {
        tracing::warn!(result = r, "slUpgradeInterface(IDXGIFactory2) failed");
    }
}

pub(crate) fn active() -> bool {
    STATE
        .get()
        .is_some_and(|c| c.lock().unwrap_or_else(|e| e.into_inner()).is_some())
}

fn frame_sync() -> &'static Mutex<FrameSync> {
    FRAME_SYNC.get_or_init(|| Mutex::new(FrameSync::new()))
}

unsafe fn ensure_frame_token(api: &Api, fs: &mut FrameSync) -> *const BaseStructure {
    if fs.token == 0 {
        let mut tok: *const BaseStructure = std::ptr::null();
        let idx = fs.frame_index;
        // SAFETY: obtains SL frame token matching `idx`.
        let r = unsafe { (api.sl_get_new_frame_token)(&mut tok, &idx) };
        if r == 0 {
            fs.token = tok as usize;
        }
    }
    fs.token as *const BaseStructure
}

pub(crate) struct UpscaleInputs<'a> {
    pub(crate) list: &'a ID3D12GraphicsCommandList,
    pub(crate) colour: &'a ID3D12Resource,
    pub(crate) colour_state: u32,
    pub(crate) depth: Option<(&'a ID3D12Resource, u32)>,
    pub(crate) motion: Option<(&'a ID3D12Resource, u32)>,
    pub(crate) exposure: Option<(&'a ID3D12Resource, u32)>,
    pub(crate) output: &'a ID3D12Resource,
    pub(crate) output_state: u32,
    pub(crate) jitter: [f32; 2],
    pub(crate) motion_scale: [f32; 2],
    pub(crate) render: [u32; 2],
    pub(crate) upscale: [u32; 2],
}

/// Evaluates `kFeatureDLSS` and tags `Depth` / `MotionVectors` for the active frame
/// (used by both `DLSS-SR` and `DLSS-G`). Returns `true` only if `output` was
/// written by DLSS Super Resolution.
pub(crate) fn evaluate_dlss_or_tag(u: &UpscaleInputs<'_>) -> bool {
    let (
        list,
        colour,
        colour_state,
        depth,
        motion,
        exposure,
        output,
        output_state,
        jitter,
        motion_scale,
        render,
        upscale,
    ) = (
        u.list,
        u.colour,
        u.colour_state,
        u.depth,
        u.motion,
        u.exposure,
        u.output,
        u.output_state,
        u.jitter,
        u.motion_scale,
        u.render,
        u.upscale,
    );
    let Some(cell) = STATE.get() else {
        return false;
    };
    let guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    let Some(api) = guard.as_ref() else {
        return false;
    };
    if !api.dlss_supported && !api.dlss_g_supported {
        return false;
    }

    // SAFETY: live D3D12 resource descriptions.
    let c_desc = unsafe { colour.GetDesc() };
    let o_desc = unsafe { output.GetDesc() };
    let (Some((d_res, d_state)), Some((m_res, m_state))) = (depth, motion) else {
        return false;
    };
    // SAFETY: live depth/mvec descriptions.
    let (d_desc, m_desc) = unsafe { (d_res.GetDesc(), m_res.GetDesc()) };

    let rw = render[0].max(1);
    let rh = render[1].max(1);
    let ow = upscale[0].max(o_desc.Width as u32).max(1);
    let oh = upscale[1].max(o_desc.Height).max(1);

    let mut fs = frame_sync().lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: obtains valid frame token.
    let tok = unsafe { ensure_frame_token(api, &mut fs) };
    if tok.is_null() {
        return false;
    }

    fs.saw_upscale = true;
    fs.depth_format = d_desc.Format.0 as u32;
    fs.mvec_format = m_desc.Format.0 as u32;
    fs.depth_extent = [rw, rh];

    if let Some(pcl) = api.sl_pcl_set_marker
        && !fs.constants_sent
    {
        // SAFETY: active frame token.
        unsafe {
            let _ = pcl(PCL_RENDER_SUBMIT_START, tok);
        }
    }

    if !fs.constants_sent {
        let mut consts = Constants::new(ow as f32 / oh as f32);
        consts.jitter_offset = jitter;
        // FidelityFX `motion_scale` scales mvec to pixels, whereas Streamline
        // expects `mvecScale` to normalize buffer values to `[-1, 1]` UV space.
        consts.mvec_scale = [
            if motion_scale[0] != 0.0 {
                motion_scale[0] / rw as f32
            } else {
                1.0 / rw as f32
            },
            if motion_scale[1] != 0.0 {
                motion_scale[1] / rh as f32
            } else {
                1.0 / rh as f32
            },
        ];
        if let Ok(scale_override) = std::env::var("VICTORY_DLSS_MVEC_NORM")
            && let Ok(s) = scale_override.trim().parse::<f32>()
        {
            consts.mvec_scale = [s, s];
        }
        if let Ok(inv) = std::env::var("VICTORY_DLSS_DEPTH_INVERTED") {
            consts.depth_inverted = if inv == "0" { BOOL_FALSE } else { BOOL_TRUE };
        }
        // SAFETY: valid `Constants`, `FrameToken`, `ViewportHandle`.
        let r = unsafe { (api.sl_set_constants)(&consts, tok, &VIEWPORT) };
        if r != 0 {
            tracing::warn!(result = r, "slSetConstants failed");
        }
        fs.constants_sent = true;
    }

    // Tag depth & motion vectors once for the frame (`eOnlyValidNow` makes SL
    // snapshot transient render targets inside this open command list so they
    // remain intact for DLSS-G at Present time even if later passes reuse them).
    let zero_copy = std::env::var_os("VICTORY_SL_ZERO_COPY").is_some_and(|v| v != "0");
    let frame_lifecycle = if zero_copy {
        LIFECYCLE_VALID_UNTIL_PRESENT
    } else {
        LIFECYCLE_ONLY_VALID_NOW
    };
    let render_ext = Extent {
        top: 0,
        left: 0,
        width: rw,
        height: rh,
    };
    let output_ext = Extent {
        top: 0,
        left: 0,
        width: ow,
        height: oh,
    };

    let depth_sl = Resource::tex2d(
        d_res.as_raw(),
        d_state,
        d_desc.Width as u32,
        d_desc.Height,
        d_desc.Format.0 as u32,
    );
    let mvec_sl = Resource::tex2d(
        m_res.as_raw(),
        m_state,
        m_desc.Width as u32,
        m_desc.Height,
        m_desc.Format.0 as u32,
    );

    if api.dlss_g_supported {
        let frame_tags = [
            ResourceTag::new(&depth_sl, BUFFER_DEPTH, frame_lifecycle, render_ext),
            ResourceTag::new(&mvec_sl, BUFFER_MVEC, frame_lifecycle, render_ext),
        ];
        // SAFETY: open command list & live D3D12 resource handles.
        let r = unsafe {
            (api.sl_set_tag_for_frame)(
                tok,
                &VIEWPORT,
                frame_tags.as_ptr(),
                frame_tags.len() as u32,
                list.as_raw(),
            )
        };
        if r != 0 {
            tracing::warn!(result = r, "slSetTagForFrame(depth/mvec) failed");
        }
    }

    if !api.dlss_supported {
        return false;
    }
    let Some(set_dlss) = api.sl_dlss_set_options else {
        return false;
    };

    let mut opts = DLSSOptions {
        mode: api.config.dlss_mode.unwrap_or(2),
        output_width: ow,
        output_height: oh,
        // Use DLSS AutoExposure unless an explicit exposure texture was supplied.
        use_auto_exposure: if exposure.is_some() {
            BOOL_FALSE
        } else {
            BOOL_TRUE
        },
        // 16-bit float or R11G11B10F render targets are linear/HDR; 8-bit targets are LDR.
        color_buffers_hdr: if matches!(c_desc.Format.0, 2 | 10 | 26) {
            BOOL_TRUE
        } else {
            BOOL_FALSE
        },
        ..Default::default()
    };
    if let Ok(hdr) = std::env::var("VICTORY_DLSS_HDR") {
        opts.color_buffers_hdr = if hdr == "0" { BOOL_FALSE } else { BOOL_TRUE };
    }
    // SAFETY: valid viewport and DLSSOptions.
    let opt_res = unsafe { set_dlss(&VIEWPORT, &opts) };
    if opt_res != 0 {
        tracing::warn!(result = opt_res, "slDLSSSetOptions failed");
        return false;
    }

    let in_sl = Resource::tex2d(
        colour.as_raw(),
        colour_state,
        c_desc.Width as u32,
        c_desc.Height,
        c_desc.Format.0 as u32,
    );
    let out_sl = Resource::tex2d(
        output.as_raw(),
        output_state,
        o_desc.Width as u32,
        o_desc.Height,
        o_desc.Format.0 as u32,
    );
    let mut local_tags = vec![
        ResourceTag::new(&in_sl, BUFFER_SCALING_IN, LIFECYCLE_ONLY_VALID_NOW, render_ext),
        ResourceTag::new(&out_sl, BUFFER_SCALING_OUT, LIFECYCLE_ONLY_VALID_NOW, output_ext),
        ResourceTag::new(&depth_sl, BUFFER_DEPTH, LIFECYCLE_ONLY_VALID_NOW, render_ext),
        ResourceTag::new(&mvec_sl, BUFFER_MVEC, LIFECYCLE_ONLY_VALID_NOW, render_ext),
    ];
    let exp_holder;
    if let Some((e_res, e_state)) = exposure {
        // SAFETY: live exposure texture.
        let e_desc = unsafe { e_res.GetDesc() };
        exp_holder = Resource::tex2d(
            e_res.as_raw(),
            e_state,
            e_desc.Width as u32,
            e_desc.Height,
            e_desc.Format.0 as u32,
        );
        local_tags.push(ResourceTag::new(
            &exp_holder,
            BUFFER_EXPOSURE,
            LIFECYCLE_ONLY_VALID_NOW,
            Extent {
                top: 0,
                left: 0,
                width: e_desc.Width as u32,
                height: e_desc.Height,
            },
        ));
    }
    let mut inputs: Vec<*const BaseStructure> = Vec::with_capacity(1 + local_tags.len());
    inputs.push(&VIEWPORT.base);
    for t in &local_tags {
        inputs.push(&t.base);
    }

    // SAFETY: command list is open, resources are transitioned to declared states,
    // and caller (`list::upscale_stretch`) replays full host pipeline state right after.
    let eval_res = unsafe {
        (api.sl_evaluate_feature)(
            FEATURE_DLSS,
            tok,
            inputs.as_ptr(),
            inputs.len() as u32,
            list.as_raw(),
        )
    };
    if eval_res != 0 {
        tracing::warn!(result = eval_res, "slEvaluateFeature(kFeatureDLSS) failed; falling back to stretch");
        return false;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            render = ?(rw, rh),
            upscale = ?(ow, oh),
            hdr = opts.color_buffers_hdr == BOOL_TRUE,
            "NVIDIA DLSS Super Resolution active"
        );
    });
    true
}

/// Tags HUD-less color + UI plane before Present, sets `DLSSGOptions`, and emits
/// Reflex/PCL submit-end + present-start markers.
pub(crate) fn before_present(
    list: &ID3D12GraphicsCommandList,
    hudless: &ID3D12Resource,
    hudless_state: D3D12_RESOURCE_STATES,
    back_width: u32,
    back_height: u32,
    back_format: DXGI_FORMAT,
    over: Option<(&ID3D12Resource, D3D12_RESOURCE_STATES, DXGI_FORMAT)>,
) {
    let Some(cell) = STATE.get() else {
        return;
    };
    let guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    let Some(api) = guard.as_ref() else {
        return;
    };
    let mut fs = frame_sync().lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: ensures active token even on non-upscale frames (menus/loading).
    let tok = unsafe { ensure_frame_token(api, &mut fs) };
    if tok.is_null() {
        return;
    }

    if let Some(pcl) = api.sl_pcl_set_marker {
        // SAFETY: active frame token.
        unsafe {
            if fs.constants_sent {
                let _ = pcl(PCL_RENDER_SUBMIT_END, tok);
            }
            let _ = pcl(PCL_PRESENT_START, tok);
        }
    }

    if !api.dlss_g_supported {
        return;
    }
    let Some(set_fg) = api.sl_dlssg_set_options else {
        return;
    };

    // Only keep DLSS-G active on frames that actually rendered 3D depth/motion
    // (`saw_upscale`). During loading screens or full-screen menus we retain
    // allocations (`DLSSG_FLAG_RETAIN_WHEN_OFF`) and switch `mode = Off`.
    let can_fg = fs.saw_upscale && back_width >= 300 && back_height >= 300;
    let mut fg_opts = DLSSGOptions {
        flags: DLSSG_FLAG_RETAIN_WHEN_OFF,
        color_width: back_width,
        color_height: back_height,
        color_buffer_format: back_format.0 as u32,
        mvec_depth_width: fs.depth_extent[0],
        mvec_depth_height: fs.depth_extent[1],
        depth_buffer_format: fs.depth_format,
        mvec_buffer_format: fs.mvec_format,
        ..Default::default()
    };

    if can_fg {
        // SAFETY: live resource descriptions.
        let h_desc = unsafe { hudless.GetDesc() };
        let h_match = h_desc.Width as u32 == back_width && h_desc.Height == back_height;
        let full_ext = Extent {
            top: 0,
            left: 0,
            width: back_width,
            height: back_height,
        };
        let hud_sl = Resource::tex2d(
            hudless.as_raw(),
            hudless_state.0 as u32,
            h_desc.Width as u32,
            h_desc.Height,
            h_desc.Format.0 as u32,
        );
        let mut ui_sl: Option<Resource> = None;
        let mut tags = Vec::new();
        if h_match {
            fg_opts.hud_less_buffer_format = h_desc.Format.0 as u32;
            tags.push(ResourceTag::new(
                &hud_sl,
                BUFFER_HUDLESS,
                LIFECYCLE_VALID_UNTIL_PRESENT,
                full_ext,
            ));
        }
        if let Some((ui_res, ui_state, ui_fmt)) = over {
            // SAFETY: live resource description.
            let u_desc = unsafe { ui_res.GetDesc() };
            // Avoid 2-bit-alpha formats (`R10G10B10A2` = 24) per SL guide.
            if u_desc.Width as u32 == back_width
                && u_desc.Height == back_height
                && ui_fmt.0 != 24
            {
                let res = ui_sl.insert(Resource::tex2d(
                    ui_res.as_raw(),
                    ui_state.0 as u32,
                    u_desc.Width as u32,
                    u_desc.Height,
                    ui_fmt.0 as u32,
                ));
                fg_opts.ui_buffer_format = ui_fmt.0 as u32;
                fg_opts.flags |= DLSSG_FLAG_MENU_DETECTION;
                tags.push(ResourceTag::new(
                    res,
                    BUFFER_UI_COLOR_ALPHA,
                    LIFECYCLE_VALID_UNTIL_PRESENT,
                    full_ext,
                ));
                if h_match {
                    fg_opts.enable_ui_recomposition = BOOL_TRUE;
                }
            }
        }
        if !tags.is_empty() {
            // SAFETY: open command list & live resources valid until Present.
            let _ = unsafe {
                (api.sl_set_tag_for_frame)(
                    tok,
                    &VIEWPORT,
                    tags.as_ptr(),
                    tags.len() as u32,
                    list.as_raw(),
                )
            };
        }
        let requested_mode = api.config.dlss_g_mode.unwrap_or(1);
        fg_opts.mode = if requested_mode == 3 && !api.dlss_g_dynamic_ok {
            2 // Fallback to Auto when Dynamic MFG isn't available (e.g., Ada vs Blackwell)
        } else {
            requested_mode
        };
        fg_opts.num_frames_to_generate = api.config.dlss_g_frames.clamp(1, api.dlss_g_max_frames);
    } else {
        fg_opts.mode = 0; // Off (retained)
        // Clear tags cleanly when not rendering game frames.
        let clear = [
            ResourceTag::new(std::ptr::null(), BUFFER_DEPTH, 0, Extent::default()),
            ResourceTag::new(std::ptr::null(), BUFFER_MVEC, 0, Extent::default()),
            ResourceTag::new(std::ptr::null(), BUFFER_HUDLESS, 0, Extent::default()),
            ResourceTag::new(std::ptr::null(), BUFFER_UI_COLOR_ALPHA, 0, Extent::default()),
        ];
        // SAFETY: clears tags for this frame token.
        let _ = unsafe {
            (api.sl_set_tag_for_frame)(
                tok,
                &VIEWPORT,
                clear.as_ptr(),
                clear.len() as u32,
                std::ptr::null_mut(),
            )
        };
    }

    // SAFETY: valid viewport and DLSSGOptions on present thread.
    let r = unsafe { set_fg(&VIEWPORT, &fg_opts) };
    if r != 0 {
        tracing::warn!(result = r, "slDLSSGSetOptions failed");
    }
}

/// Completes the Present marker, logs DLSS-G presented-frame throughput once per
/// second, calls `slReflexSleep` for frame pacing, and advances the frame token.
pub(crate) fn after_present() {
    let Some(cell) = STATE.get() else {
        return;
    };
    let guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    let Some(api) = guard.as_ref() else {
        return;
    };
    let mut fs = frame_sync().lock().unwrap_or_else(|e| e.into_inner());
    let tok = fs.token as *const BaseStructure;
    if !tok.is_null() {
        if let Some(pcl) = api.sl_pcl_set_marker {
            // SAFETY: active frame token matching `PresentStart`.
            unsafe {
                let _ = pcl(PCL_PRESENT_END, tok);
            }
        }
        if api.dlss_g_supported
            && fs.saw_upscale
            && let Some(get_state) = api.sl_dlssg_get_state
        {
            let mut st = DLSSGState::default();
            // SAFETY: present-thread state poll without `eRequestVRAMEstimate`.
            if unsafe { get_state(&VIEWPORT, &mut st, std::ptr::null()) } == 0 {
                fs.fg_presented_accum += u64::from(st.num_frames_actually_presented);
                fs.present_count += 1;
                if st.status != 0 {
                    tracing::warn!(status = st.status, "DLSS-G runtime status non-OK");
                }
                let elapsed = fs.last_fg_log.elapsed();
                if elapsed >= std::time::Duration::from_secs(2) && fs.present_count > 0 {
                    let secs = elapsed.as_secs_f64();
                    let displayed_fps = fs.fg_presented_accum as f64 / secs;
                    let mult = fs.fg_presented_accum as f64 / fs.present_count as f64;
                    tracing::info!(
                        displayed_fps = format_args!("{displayed_fps:.1}"),
                        multiplier = format_args!("{mult:.2}x"),
                        presented = fs.fg_presented_accum,
                        app_frames = fs.present_count,
                        "DLSS-G output rate"
                    );
                    fs.last_fg_log = std::time::Instant::now();
                    fs.fg_presented_accum = 0;
                    fs.present_count = 0;
                }
            }
        }
    }

    // Advance frame token & trigger Reflex sleep + next simulation start marker.
    fs.frame_index = fs.frame_index.wrapping_add(1);
    fs.token = 0;
    fs.constants_sent = false;
    fs.saw_upscale = false;
    // SAFETY: acquires token for upcoming frame.
    let next_tok = unsafe { ensure_frame_token(api, &mut fs) };
    if !next_tok.is_null() {
        if let Some(sleep) = api.sl_reflex_sleep {
            // SAFETY: valid frame token.
            let _ = unsafe { sleep(next_tok) };
        }
        if let Some(pcl) = api.sl_pcl_set_marker {
            // SAFETY: valid frame token.
            let _ = unsafe { pcl(PCL_SIMULATION_START, next_tok) };
        }
    }
}

/// Gracefully shuts down Streamline when tearing down.
#[allow(dead_code)]
pub(crate) fn shutdown() {
    let Some(cell) = STATE.get() else {
        return;
    };
    let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(api) = guard.take() {
        // SAFETY: shuts down SL plugins before device destruction.
        let _ = unsafe { (api.sl_shutdown)() };
        // SAFETY: releases loaded `sl.interposer.dll`.
        let _ = unsafe { FreeLibrary(api._module) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn streamline_ffi_struct_layouts_match_cpp_sdk() {
        assert_eq!(size_of::<BaseStructure>(), 32);
        assert_eq!(size_of::<ViewportHandle>(), 40);
        assert_eq!(offset_of!(ViewportHandle, value), 32);
        assert_eq!(size_of::<AdapterInfo>(), 56);
        assert_eq!(size_of::<Resource>(), 112);
        assert_eq!(offset_of!(Resource, native), 40);
        assert_eq!(offset_of!(Resource, state), 64);
        assert_eq!(size_of::<ResourceTag>(), 64);
        assert_eq!(offset_of!(ResourceTag, extent), 48);
        assert_eq!(size_of::<Constants>(), 456);
        assert_eq!(offset_of!(Constants, jitter_offset), 352);
        assert_eq!(offset_of!(Constants, mvec_scale), 360);
        assert_eq!(offset_of!(Constants, depth_inverted), 444);
        assert_eq!(offset_of!(Constants, min_relative_linear_depth_object_separation), 452);
        assert_eq!(size_of::<DLSSOptions>(), 88);
        assert_eq!(size_of::<DLSSGOptions>(), 120);
        assert_eq!(offset_of!(DLSSGOptions, dynamic_target_frame_rate), 116);
        assert_eq!(size_of::<DLSSGState>(), 88);
        assert_eq!(size_of::<ReflexOptions>(), 48);
    }

    #[test]
    fn config_parsing_enforces_reflex_with_frame_generation() {
        let off = Config::parse(None, None, None);
        assert!(!off.any_enabled());

        let fg = Config::parse(Some("quality"), Some("1"), None);
        assert_eq!(fg.dlss_mode, Some(3));
        assert_eq!(fg.dlss_g_mode, Some(1));
        assert_eq!(fg.dlss_g_frames, 1);
        assert_eq!(fg.reflex_mode, Some(1));

        let boost = Config::parse(None, Some("auto"), Some("boost"));
        assert_eq!(boost.dlss_mode, None);
        assert_eq!(boost.dlss_g_mode, Some(2));
        assert_eq!(boost.reflex_mode, Some(2));
    }
}
