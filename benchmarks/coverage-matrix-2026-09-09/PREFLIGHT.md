# Pre-flight readiness (2026-09-09)

| Lane | Tool | Device report | Status |
|------|------|---------------|--------|
| native-npu | engine/npu/build/npu_engine | XRT /dev/accel/accel0 (NPU Strix Halo, XDNA 2) | ✅ ran Qwen3-0.6B full pipeline (prefill 64ms/tok, decode 537ms/tok legacy path), exit 0 |
| npu-flm | /opt/fastflowlm/bin/flm | /dev/accel/accel0, 8 cols, fw 1.1.2.65 | ✅ `flm validate` ready:true |
| hrx | llama-hrx-b59/bin/llama-cli | HRX0: Radeon 8060S (gfx1151), 114688 MiB | ✅ --list-devices |
| vulkan | llama-hrx-b59/bin/llama-cli | Vulkan0: Radeon 8060S (RADV STRIX_HALO), 123904 MiB | ✅ --list-devices |

## Notes
- `NPU_XCLBIN_DIR` in the shell env is STALE (`/home/bcloud/1bit-MONSTER-pi/...`, dir no longer exists).
  Correct value: `NPU_XCLBIN_DIR=/home/bcloud/1bit-MONSTER/engine/npu/xclbins`.
- Native NPU default run = legacy per-op path (~2 tok/s decode). Fast path is `NPU_RUNLIST=1` (94 tok/s on 0.6B, branch `goal/runlist-decode-wire`).
- The HRX bundle ships BOTH `HRX0` and `Vulkan0` devices — the same `llama-cli`/`llama-server` binary drives either lane by `--device`.
- FLM cached models (model.q4nx present): Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3.6-35B-A3B, Llama-3.2-1B.
