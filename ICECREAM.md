# icecream (VIctory NVIDIA RTX-4050 / Low-VRAM Fork)

Community fork based on [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (`origin/main`), [fxceless/victory-fork](https://github.com/fxceless/victory-fork), [vitutu1/victory-perf](https://github.com/vitutu1/victory-perf), [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD), and [dexnom/VIctory](https://github.com/dexnom/VIctory), tuned for low-VRAM NVIDIA D3D12 hosts (tested on RTX 4050 Laptop, 5,152 MB DXGI budget, Wave32, Windows 11).

## What Is Already Integrated Here

- **Shader fixes (`crates/victory-shader`)**:
  - Wave32-compatible compute lowering (`host_max_wave = 32`), all 5 wave32 compute shadow/RT pipelines recovered.
  - Pixel quad helper reads (`helpers_read`: `QuadReadLaneAt`), intra-quad `dpp8` check, and `CubeLOD` log2 + min-LOD clamp.
  - `rect_list_gs` + CPU `expand_rects` right-angle completion opposite longest side (`right_angle`).
  - `typed_views.rs` / `tiled_lighting_uint_srvs` scalar UINT SRV typing (`bufferLoad.i32` / `textureLoad.i32` / `textureGather.i32`) for `VS_TiledLighting` and eye-lighting PS variants.
  - `UAVS_EVERY_STAGE` flags on VS/GS/MS with root UAVs; uniform branch over immediate samplers (`image_with_sampler`); PSV0 vertex output interpolation + water VS signature ordering; wave64 split-scalar mask preservation.
  - `norm_quotient` / `exact_norm` (`SNORM` `+1.0` clamp fixed) gated behind `VICTORY_EXACT_NORM=1` (off by default to keep NVIDIA DXIL PSO compile times fast).
- **GPU & HLE fixes (`crates/victory-gpu`, `crates/victory-hle`)**:
  - `VICTORY_INDIRECT_GATHER=1` (enabled by default; brought GPU draw execution down to ~18 ms on boat/cutscene frames).
  - `raw_unaligned` own-byte buffer copies (`buffer views left null` = 0).
  - `early_depth` PSO warming priority (`screen_requested` > `requested` > `screen_early` > `early_depth` > `early`), depth-twin `prelude()` barrier ordering, and lock-free `AtomicU64` descriptor mirror tracking (`WALKS = 8`).
  - Coalesced memory remap notifications (`record_remap` / `merge_remaps`), fair new-vs-retry texture queue, and interior-page `mapped_bytes` detection for streamed textures.
  - Loader resolution levers (`VICTORY_OWN_UPSCALER=1`, `VICTORY_RENDER_SCALE=0.5`) and `VICTORY_NO_WATCHDOG=1`.

## Recommended Runtime Environment (6 GB / ~5.1 GB VRAM Budget)

```bat
set "VICTORY_NO_WATCHDOG=1"
set "VICTORY_CARD_BUFFERS_MIN_MB=1"
set "VICTORY_CARD_BUFFERS_MIN_KB=256"
set "VICTORY_CARD_BUFFERS_MB=384"
set "VICTORY_CARD_FETCH_INFLIGHT=1"
set "VICTORY_CARD_REHASH_GAP=4"
set "VICTORY_RELAY=2"
set "VICTORY_LATE_WAIT=3"
set "VICTORY_WIDE_BACK=1"
set "VICTORY_JOIN_COPIES=1"
set "VICTORY_COPY_PLAIN=1"
set "VICTORY_INDIRECT_WINDOW=1"
set "VICTORY_INDIRECT_MERGE=1"
set "VICTORY_INDIRECT_GATHER=1"
set "VICTORY_REBAR=0"
set "VICTORY_REBAR_MB=0"
set "VICTORY_AMBIENT=0.55"
set "VICTORY_AMBIENT_SHADE=0.4,0.6,0.1,0.45"
set "VICTORY_TEXTURE_DROP_MIPS=1"
set "VICTORY_LEVELS_ON_DEMAND=1"
set "VICTORY_TEXTURE_ROOM_PERCENT=92"
set "VICTORY_CARD_ROOM_FOR_TARGETS_MB=512"
set "VICTORY_TEXTURE_STAGE_MB=64"
set "VICTORY_TEXTURE_STAGE_MS=16"
set "VICTORY_WAVE_WIDE_ONLY="
set "VICTORY_BVH_BUILDER=0"
set "VICTORY_BVH_QUARANTINE=0"
set "VICTORY_OWN_UPSCALER=1"
set "VICTORY_RENDER_SCALE=0.5"
set "VICTORY_DLSS=0"
set "VICTORY_DLSS_FG=0"
set "VICTORY_PIPELINE_WAIT_MS=500"
```

Note: Keep `VICTORY_CARD_BUFFERS_MB=384` (not `768`) on 6 GB cards so VRAM doesn't exhaust before `PS_BakeDecorations` bakes character skin textures.

## Open Areas to Help Optimize / Fix

1. **Frame Pacing / CPU Preparation Overhead**: GPU execution is ~18 ms/frame, while `preparing` (`copies`, `descriptor heap sync`, `heap buffers`) and copy-backs push total frame time to ~240–550 ms (~2–5 FPS outdoors vs ~8.5 FPS on Radeon/DexNom indoors).
2. **Cutscene Facial / Mouth Animation**: Character colors and bodies render in cutscenes/boat scenes, while character mouths do not animate yet.
3. **Shader / PSO Warmth**: First-run cold PSO compilation for uncached mesh shaders on NVIDIA takes minutes; optimizing shader IR size or pre-warming pipelines helps dramatically.
