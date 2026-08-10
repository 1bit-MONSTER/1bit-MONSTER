# Vendored: ROCm/FastFlowLM (open-source FLM runtime)

Vendored from https://github.com/ROCm/FastFlowLM at commit
`17a35cb` (Merge pull request #659 from ROCm/repo-transfer — repo moved from
FastFlowLM org to ROCm, 2026-08-07; MIT license).

The runtime that serves the Q4NX models on the Ryzen AI NPU (the Q4NX pivot).
The installed package (/opt/fastflowlm, FLM v0.9.46) matches this tree. This
vendor is the reference/archive copy — the build links the installed package
(see CMakeLists FLM_BINARY/FLM_CONFIG/FLM_XCLBINS discovery).

What is open in this tree:
- server / CLI / runner / tokenizer / model pull + registry (full sources)
- whisper model (fully open .cpp)
- xclbins (NPU AIE kernels, binaries) + aiebu assembler headers
- companion converter: third_party/FLM_Q4NX_Converter (GGUF -> q4nx writer)

What is NOT yet open (prebuilt .lib/.dll in src/lib/): the NPU model engines
(qwen3_npu, llama_npu, ...) and the Q4NX file reader (npu_quantize_block).
Declarations in src/include/; re-vendor on upstream sync.
