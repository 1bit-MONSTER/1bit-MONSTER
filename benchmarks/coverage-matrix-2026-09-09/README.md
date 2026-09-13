# Coverage matrix — 2026-09-09

Verification run for the thesis **"2 NPU engines + (HRX + Vulkan) = any model"**
on the Strix Halo box (Ryzen AI MAX+ 395).

Start at **MATRIX.md** for the model × lane matrix and verdict.

- ROSTER.md — complete local inventory (by format)
- PREFLIGHT.md — 4-lane readiness + env gotchas
- native-npu.md + native-npu.log — reverse-engineered XDNA 2 engine
- flm.md + flm.log — FastFlowLM NPU
- hrx.md + hrx.log — HRX bundle (HRX0) + hrx-v2 fork (HRX20, Q4NX)
- vulkan.md + vulkan.log — ggml-vulkan/RADV
- hrx2-fork-fixes.patch — the 3 fork fixes (committed as df58715ca) that closed the Q4NX gaps

Every cell is backed by a captured run (the "Paris." greedy check, or the
native-NPU corr-1.0 CPU reference check for zaya).

> Raw capture logs (`*.log`) are **not committed** — repo convention keeps
> `*.log` out of git (`.gitignore`); the tracked `.md` files above carry the
> extracts from those runs. A copy of the logs stays with the run's host.

## Result
**17/17 model rows run end-to-end with verified output on ≥1 lane.**
The three models that initially failed (MiniCPM4-8B, GLM-4.7-Flash,
Qwen3-Next-80B) are now covered on the HRX2 fork after three JIT-range/fusion
fixes. ZAYA1-74B is a VEK385-platform artifact (out of scope).
