# Native NPU lane — results (engine/npu/build/npu_engine)

> NOTE: the checked-in `npu_engine` binary was STALE (Sep 8) vs. source (Sep 9,
> runlist wiring commit cf5529ae6). I rebuilt from committed source with
> `REPO_ROOT=/home/bcloud/1bit-MONSTER bash engine/npu/build_npu.sh`
> (no source changes; build script itself has a latent `REPO_ROOT`-before-assign
> bug, worked around by exporting it). Results below are from the FRESH binary.

| Model (q4nx) | Result | Evidence |
|---|---|---|
| zaya-8B | ✅ PASS (verified correct) | `EMB dbg corr=1.0000000 argmax 30334 vs 30334 (SAME)` vs CPU ref; MoE L1 corr=0.999; 7.2 tok/s; exit 0 |
| Qwen3.6-35B-A3B (MoE) | ⚠️ runs (correctness not independently verified) | full pipeline, exit 0, ~0.3 tok/s |
| Qwen3-0.6B | ❌ not correct | runlist path engages (9 tok/s) but greedy argmax = token 0 every step (logits read broken) |
| Qwen3-1.7B | ❌ not correct | runlist path, 36 tok/s, argmax = token 0 |
| Qwen3-4B | ❌ not correct | runlist path, 21 tok/s, argmax = token 0 |
| Llama-3.2-1B | ❌ FAIL | no runlist support (not dense Qwen3); split path missing `final_i8_QKV_K2048_N3072.xclbin` |

## Findings
- The native engine's dense-Qwen3 "runlist" whole-layer path (branch
  `goal/runlist-decode-wire`) now runs 0.6B/1.7B/4B at speed (9/36/21 tok/s) but
  `get_logits` returns all-zero logits → greedy argmax = token 0. The path is
  byte-identical-to-FastFlowLM per docs, but it is NOT yet producing correct
  tokens on this build for these models.
- The legacy split path is broken for dense Qwen3 too: the runtime instruction
  generator emits single-core-row insts for multi-row v27 xclbins
  (`WARN ... results will be wrong without it`), and 1.7B/4B/Llama lack the
  matching `final_i8_QKV_K2048_N4096.xclbin` / `G_K2560_N9728.xclbin` /
  `QKV_K2048_N3072.xclbin`.
- zaya-8B is the one fully-verified native-NPU model (CCA attention on CPU +
  MoE FFN on NPU, `NPU_FUSED` path) — corr 1.0 vs CPU reference.
- Net: the native NPU lane (the reverse-engineered XDNA 2 engine) is a
  work-in-progress; today it verifiably covers zaya-8B only out of the cached set.
