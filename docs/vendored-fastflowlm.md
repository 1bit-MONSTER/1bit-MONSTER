# Vendored: ROCm/FastFlowLM (open-source FLM runtime)

Vendored from https://github.com/ROCm/FastFlowLM at tag
`v1.0.6` (commit `1a40ad9`, "docs: update for v1.0.6", 2026-09-18; MIT license).
Re-vendored 2026-09-19 from the previous pin `17a35cb` (v0.9.46, PR #659
repo-transfer).

> ⚠️ **v0.9.46 pin preserved as a backup tag.** The previous submodule HEAD is
> tagged `project-pin-v0.9.46-17a35cb` inside `third_party/FastFlowLM`'s own
> git repo (local tag — not pushed). The in-progress Qwen3.6-35B-A3B
> GateDeltaNet float32 repair depends on the v0.9.46 `LM_Config` header layout;
> v1.0.x changed that layout (header ABI mismatch) and has a bf16-GDN NaN
> regression with no v1.0.x fix. See `docs/research/qwen36-gdn-float32-repair-
> plan.md`. To get the old tree back: `git -C third_party/FastFlowLM checkout
> project-pin-v0.9.46-17a35cb`.

The runtime that serves the Q4NX models on the Ryzen AI NPU (the Q4NX pivot).
The installed package (/opt/fastflowlm, FLM v0.9.46) does **not** match this
tree anymore — it remains pinned to v0.9.46 on the dev box. This vendor is the
reference/archive copy — the build links the installed package (see CMakeLists
FLM_BINARY/FLM_CONFIG/FLM_XCLBINS discovery), so the vendored headers here
(v1.0.6) are NOT what the build compiles against.

What is open in this tree:
- server / CLI / runner / tokenizer / model pull + registry (full sources)
- whisper model (fully open .cpp)
- xclbins (NPU AIE kernels, binaries) + aiebu assembler headers
- companion converter: third_party/FLM_Q4NX_Converter (GGUF -> q4nx writer)

What is NOT yet open (prebuilt .lib/.dll in src/lib/): the NPU model engines
(qwen3_npu, llama_npu, ...) and the Q4NX file reader (npu_quantize_block).
Declarations in src/include/; re-vendor on upstream sync.
