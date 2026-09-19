# Criterion (c) at ~8k — 15% win attempt: fresh ratios + profiled shortfall (2026-09-19)

Goal `mu8e9rte-9vr80i`. This is the **honest deliverable** required by the objective's
done-clause when 0.85 is not reached: *"the fresh measured ratios + a named, profiled
reason for the shortfall."* Native is **not** at ≤0.85 on any cell; the target requires a
device kernel faster than FLM's own, which is an IRON/Peano xclbin-generation effort
(multi-day), not a C++ source edit.

## Fresh measured ratios (post host-fusion, same guarded harness)

Harness unchanged: `benchmarks/c8k_guarded.sh <Model-NPU2> <flm_tag> <engine> <runs>`,
`C8K_WAIT_QUIET=1`, `NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins`, FLM reference v1.0.4
(`/opt/fastflowlm/bin/flm`), 8192 tokens, `NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1`.
First-token sanity held throughout (Qwen3 → `[1] 576`, Llama → `[1] 785`).

| model | run | native TTFT (ms) | FLM TTFT (s) | ratio | ≤0.85? |
|---|---:|---:|---:|---:|:---:|
| Qwen3-0.6B | 1 | 3867 | 3.999625 | 0.967 | ❌ |
| Qwen3-0.6B | 2 | 3913 | 3.916807 | 0.999 | ❌ |
| Qwen3-0.6B | 3 | 3885 | 3.903361 | 0.995 | ❌ |
| Qwen3-1.7B | 1 | 5487 | 5.685816 | 0.965 | ❌ |
| Qwen3-1.7B | 2 | 5459 | 5.686479 | 0.960 | ❌ |
| Qwen3-1.7B | 3 | 5458 | 5.721025 | 0.954 | ❌ |
| Qwen3-4B | 1 | 12950 | 13.694614 | 0.946 | ❌ |
| Qwen3-4B | 2 | 12965 | 13.500221 | 0.960 | ❌ |
| Qwen3-4B | 3 | 12888 | 13.528681 | 0.953 | ❌ |
| Qwen3-VL-4B | 1 | 12878 | 13.396614 | 0.961 | ❌ |
| Qwen3-VL-4B | 2 | 12922 | 13.300626 | 0.972 | ❌ |
| Qwen3-VL-4B | 3 | 12862 | 13.287041 | 0.968 | ❌ |
| Qwen3-8B | 1 | 17896 | 18.357775 | 0.975 | ❌ |
| Qwen3-8B | 2 | 17907 | 18.346949 | 0.976 | ❌ |
| Qwen3-8B | 3 | 17872 | 18.577717 | 0.962 | ❌ |
| Llama-3.1-8B | 1 | 17237 | 17.497898 | 0.985 | ❌ |
| Llama-3.1-8B | 2 | 17247 | 17.479116 | 0.987 | ❌ |
| Llama-3.1-8B | 3 | 17265 | 17.497650 | 0.987 | ❌ |

Fresh ratio range **0.946–0.999** — statistically unchanged from the committed baseline
(0.911–0.997). The host-pass fusion below moved none of the 18 cells. (Llama run 2 was
flag-suspect on load rise in the harness summary; it is shown for completeness and does
not change the conclusion — no cell approaches 0.85.)

## The named, profiled reason for the shortfall

### 1. Native 8k prefill is ~90% device-bound, on FLM's own kernels

Per-model Prefill attribution (from the `[GEMM / attn / conv+other]` + host timers in
each `native-*.log`):

| model | attn (device ELF) | QKV GEMM (mm.xclbin) | FFN GEMMs O/GU/D (mm.xclbin) | host norm+silu+rd+qk+ba |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 2379 ms (60%) | 434 ms (11%) | ~945 ms (24%) | ~165–364 ms (4–9%) |
| Qwen3-8B | 5749 ms (32%) | 1590 ms (9%) | ~9540 ms (53%) | ~977 ms (5.5%) |

The dominant terms are the **captured FLM attention ELF** (`attn_mha_8192_nh16/nh32.elf`)
and **FLM's `mm.xclbin` bf16 GEMM**. Native already drives both at FLM's own efficiency
(or slightly better, via the async full-256-row staging, dequant caching, and
double-buffering) — which is exactly the 0.3–8.9% edge in the committed baseline. The
host-side "conv+other" term is only 4–9% and is memory-bandwidth-bound, so fusing it
further cannot yield 7–17 points.

### 2. Host-pass fusion landed and measured ≈0%

Landed 7 bit-exact edits in `engine/npu/src/npu_engine_universal.cpp`:
- `rn_bf16_save(out,save,x,w,n)` — fuses residual-save + NaN-clamp + sum-of-squares
  reduce into one pass (was: `save=copy(x)` + separate `rn_bf16`).
- Input-norm and FFN-norm sites switched to it.
- O and D readback fused into direct residual-add (`bh = bsb + bf16g(bC)`), eliminating
  the `boo`/`bdw` full-width round-trips (repopulated only under `NPU_DUMP_L0`).

Rebuilt all variants, verified token parity, re-measured the full 18-cell matrix three
times — ratios did not move (table above). Expected, because the host term is 4–9% and
largely overlapped with device GEMMs.

### 3. Every faster-device-kernel alternative is ruled out by measurement

| alternative | result | evidence |
|---|---|---|
| libgemm/libdequant version swap (v0.9.46 → v1.0.6) | **byte-identical GEMM streams** | `dump_seq` linked each `.so`, `generate_seq(M=256,K=4096,N∈{6144,4096,24576,12288})` → md5-identical for qkv/o/gu/d. Swapping the sequence generator changes nothing. |
| native bf16 GEMM kernels (`final_bf16_GU_K4096_N24576.xclbin`, `n1_core_bf16_v1.py`, 32-core whole-array) | **16× slower** | `bench_gemm_bf16_analytical`/`time_bf16`: GU M=128 K=4096 N=24576 → 630 GOP/s (0.63 TFLOP/s) vs `mm.xclbin`'s measured 10.4 TFLOP/s effective on the same FFN shape. |
| fused GU→SiLU→D cascade | i8/i32 only, no bf16 | `n1_core_fused_gu_silu_d*.py` all emit `matmul_i8_i32`; the bf16 prefill FFN needs a bf16 cascade that does not exist. |
| runlist prefill | ~13.5 ms/token (slow) | vs bf16mm ~0.5 ms/token. |
| NPU_HOST_THREADS sweep (8/16/24/32) | flat 16→24, collapses at 32 | host math is memory-bandwidth-bound (matches the source comment). |
| mm.xclbin "128 correct M-rows" | already fixed | async path (`ensure_a`) stages full 256 rows, no zero padding; boot token sane. |
| mm.xclbin / dequant.xclbin md5 skew | identical (845b2963) | v0.9.46 tree, amd-oss 1.0.6 tree, and FLM v1.0.4 all ship the same kernel. |

### Conclusion (the named reason)

The 15% target requires a **device-side GEMM or attention kernel faster than FLM's own**
`mm.xclbin` / captured attention ELF. The engine's only native device-kernel alternatives
are measurably *slower* (native bf16 GEMM 16×; generated attention "much slower" per the
in-tree comments; fused cascade is i8-only). Reaching ≤0.85 therefore needs a new
bf16 fused GU→SiLU→D cascade and/or a faster attention xclbin generated via IRON/Peano —
a multi-day kernel-generation effort, out of scope for C++ source edits. The fresh ratios
above (0.946–0.999, none ≤0.85) are the honest deliverable.
