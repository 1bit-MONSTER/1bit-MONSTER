# Video LoRA

Video generation with LoRA support on Strix Halo. Supports **Wan2.2**, **LTX-Video**, **AnimateDiff**, and **CogVideoX** — pure C++ inference, no Python, no Zig.

> Vendored into [1bit.MONSTER](../../README.md) at `tools/video-lora/` from
> [bong-water-water-bong/video-lora](https://github.com/bong-water-water-bong/video-lora).
> CI runs from the repo root via `.github/workflows/video-lora-ci.yml` (C++ build + GPU selftest).

## Models & LoRAs

| Model | LoRA Support | Size | Notes |
|-------|-------------|------|-------|
| **Wan2.2-Fun** | Reward LoRAs, Camera Control LoRAs | 1.3B / 14B | Best open-source T2V |
| **LTX-Video** | IC LoRA detailer (in-context) | 13B | Video-to-video control |
| **AnimateDiff** | Motion + Style LoRAs | 1.5B | Largest community LoRA ecosystem |
| **CogVideoX** | Transformer LoRA | 2B / 5B | Good for coherent motion |
| **Stable Video Diff.** | UNet LoRA | 2.5B | Image-to-video |

## Backend

The inference backend is **pure C++** (`vulkan/`): a Vulkan compute engine
dispatching GLSL shaders (conv2d, group_norm, silu, elementwise, attention,
lora_merge) compiled to SPIR-V at build time. It is linked into the single
`zaya_server` binary. LoRA adapters are fused into weights with the
`lora_merge` kernel (`W' = W + α·(B@A)`).

See [`vulkan/README.md`](vulkan/README.md) for the backend, build, and
selftest instructions.

## Project Structure

```
tools/video-lora/
├── vulkan/
│   ├── CMakeLists.txt     # builds video_lora_vk (static lib) + CLI; glslc/glslangValidator shader compile
│   ├── include/
│   │   └── vl_engine.hpp  # Tensor, VlEngine public API
│   ├── src/
│   │   ├── vl_engine.cpp  # Vulkan context, pipelines, dispatch, ops
│   │   └── main.cpp       # CLI + CPU-reference selftest
│   └── shaders/           # GLSL compute kernels (.comp → .spv at build time)
└── README.md
```
