# video-lora — Vulkan Compute Backend (Pure C++)

Low-level Vulkan compute inference for SD1.5/AnimateDiff video generation.
**Pure C++ — no Python, no Zig, no runtime interpreter.** Built into the
single `zaya_server` binary ("One engine. Every model. Any chip.").

## Ops (GLSL compute → SPIR-V at build time)

| Op | Shader | Contract |
|----|--------|----------|
| Conv2d 3×3 (stride 1, pad 1) | `conv2d.comp` | NCHW, groups=1 |
| Group norm | `group_norm.comp` | one workgroup per (batch, group) |
| SiLU | `silu.comp` | in-place x/(1+e^-x) |
| Elementwise | `elementwise.comp` | 0=add, 1=mul, 2=scale |
| Scaled dot-product attention | `attention.comp` | head_dim=64, N≤256, softmax |
| LoRA weight fusion | `lora_merge.comp` | W' = W + α(B@A) |

Shaders are compiled with `glslc` at build time (CMake), same pattern as the
ZINC GPU engine. The C++ engine (`vl_engine.hpp/cpp`) builds a pipeline per
op with exactly the descriptor layout + push constants each shader declares.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires the Vulkan SDK (`glslc`, Vulkan headers) and a Vulkan-capable GPU.

## Verify

```bash
./build/video_lora_vk_cli --selftest --shaders build/shaders
```

The selftest runs every op on the GPU and checks each result against a naive
CPU reference (1e-3 tolerance) — a regression in a shader or in the C++
plumbing fails loudly. This is the CI gate.

## Architecture

```
include/vl_engine.hpp   public API (Tensor, VlEngine, ops)
src/vl_engine.cpp       Vulkan context, pipelines, dispatch, op impls
src/main.cpp            CLI + CPU-reference selftest
shaders/*.comp          GLSL compute kernels → .spv at build time
```
