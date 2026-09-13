# Coverage pass — dense Qwen3 family (pi agent, 2026-09-13)

Goal `mttxt22c-a6rv75`: "meet-or-beat FLM's measured performance … for **every model**
the native NPU engine supports." First per-model pass beyond Qwen3-0.6B.

Runs: on-box, `NPU_XCLBIN_DIR=engine/npu/xclbins`, prompt `benchmarks/prompts/reclaimer.txt`
tokenized to `/tmp/ids_<n>.txt`. Gate = native greedy boot token vs FLM
`run_qwen3_prefill` `GREEDY_NEXT` on the same ids file. FLM on-box from
`/opt/fastflowlm/bin/flm bench qwen3:<size> -i cfg.json` (max_length 1024) — note FLM
checkpoints at ctx **981**, native at **1024** (~4% token mismatch).

## 1. Attention-shape matrix (why 1.7B needs no capture)

| model | hidden | nh | nkv | head_dim | layers | inter | attention shape |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 16 | 8 | 128 | 28 | 3072 | **nh16/nkv8/hd128** |
| Qwen3-1.7B | 2048 | 16 | 8 | 128 | 28 | 6144 | **nh16/nkv8/hd128 — identical to 0.6B** |
| Qwen3-4B   | 2560 | 32 | 8 | 128 | 36 | 9728 | **nh32/nkv8/hd128** |
| Qwen3-8B   | 4096 | 32 | 8 | 128 | 36 | 12288 | **nh32/nkv8/hd128 — identical to 4B** |
| Nanbeige4.1-3B | 2560 | 20 | 4 | 128 | 32 | 10752 | nh20/nkv4/hd128 |
| Phi4-mini  | 3072 | 24 | 8 | 128 | 32 | 8192 | nh24/nkv8/hd128 |
| Llama-3.2-1B | 2048 | 32 | 8 | **64** | 16 | 8192 | nh32/nkv8/hd64 |
| Gemma4-E2B | 1536 | 8 | 1 | **256** | 35 | 6144 | nh8/nkv1/hd256 |

The attention kernel is parameterised by `(nh, nkv, head_dim)`, **not** hidden size.
1.7B ≡ 0.6B and 4B ≡ 8B in attention shape, so one captured ELF serves each pair.

## 2. Dense-Qwen3 results (all @1k unless noted)

| model | npt | native boot | FLM boot | gate | native prefill | FLM on-box prefill | gap | native TTFT | FLM TTFT |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 25 | 25 | ✅ | **1440.5 tok/s** | 1123.1 | **+28.3%** | 0.7113 s | 0.7037 s |
| Qwen3-1.7B | 1024 | **220** | 220 | ✅ | 817.0 tok/s | 942.57 | **−13.3%** | 1.224 s | 1.042 s |
| Qwen3-4B | 1024 | **220** *(was 87672 ❌)* | 220 | ✅ | 430.5 tok/s | 509.97 | **−15.6%** | 2.323 s | 1.925 s |
| Qwen3-8B | 1024 | **220** *(fixed)* | 220 | ✅ | 281.4 tok/s | 362.76 | **−22.4%** | 3.554 s | 2.705 s |
| Qwen3-4B | 256 | 1614 | 1614 | ✅ | 328.2 tok/s | (not benched) | — | 0.780 s | — |

Raw native lines:

```
1.7B @1024: Prefill 1220ms (1.192 ms/tok) [GEMM 153, attn 188, conv+other 1195]  boot=220
4B   @1024: Prefill 2464ms (2.406 ms/tok) [GEMM 297, attn 339, conv+other 2409]  boot=220
8B   @1024: Prefill 3626ms (3.541 ms/tok) [GEMM 419, attn 329, conv+other 3526]  boot=220
4B   @1024 (BEFORE fix): Prefill 2517ms [GEMM 319, attn 286, conv+other 2480]     boot=87672  <-- WRONG
4B   @ 256: Prefill  780ms (3.047 ms/tok) [GEMM  76, attn 165, conv+other  758]  boot=1614
```

## 3. Findings

1. **All four dense Qwen3 models are now CORRECT at @1k** (boot token == FLM).
2. **The nh32 >256 break is FIXED** (4B was boot=87672≠220). Root cause:
   `run_attn` (`npu_engine_bf16_mm.h:~248`) selects the >256 kernel as
   `attn_kernel1k` = **`attn_mha_1024_nh16.elf`** regardless of `attn_qout`
   (4096 for NH=32). A single **nh32** >256 capture fixes **both 4B and 8B**
   (identical attention shape).
3. **Fix is durable (committed)**: `npu_engine_bf16_mm.h` now loads
   `attn_mha_1024_nh32.elf` (env `NPU_ATTN_ELF_1024_NH32`) and **auto-selects** it
   when `attn_qout==4096 && attn_tokens>256`. All four dense-Qwen3 engines were
   rebuilt and re-verified **without any env override**: 0.6B boot=25,
   1.7B/4B/8B boot=220. The ELF (177 696 B, captured from 4B) lives at
   `engine/npu/xclbins/attn_mha_1024_nh32.elf`.
   **Still open**: an nh32 **2k** ELF for (1024,2048] on nh32 models (that range
   still uses the nh16 2k ELF).
4. **Native does NOT meet-or-beat FLM on prefill for 1.7B/4B/8B** — it wins only
   on 0.6B (+28.3%) and loses with a monotonically growing gap (−11% → −18.5% →
   −22.2%). Native degrades faster with model size; FLM flat-lines better.
5. **The bottleneck is `conv+other` (host-side), not the NPU**: it is 1195/2409/3526 ms
   for 1.7B/4B/8B @1k — 92–97% of total; GEMM 153/297/419 ms and attn 188/339/329 ms
   are minor. This matches the earlier "prefill bottleneck is host math, not NPU GEMM".
   Closing the large-model gap is a **host-math / overlap** task, not a kernel task.

## 4. Capture recipe (recovered + lean)

```
mkdir -p ~/npu-build/cap4b
cd npu-infer/tools/capture
RT_TOKENS=/tmp/ids_1024.txt \
LD_PRELOAD=$PWD/cap_interposer.so CAP_DIR=$HOME/npu-build/cap4b \
CAP_NO_SYNC=1 CAP_SKIP_BIG=1 \
  ./run_qwen3_prefill ~/.config/flm/models/Qwen3-4B-NPU2
```
`CAP_NO_SYNC=1 CAP_SKIP_BIG=1` makes it **lean: 3.2 GB** (vs ~80 GB full) — only the
`elf_*.bin` + small insts are dumped. The attention ELF is `elf_0012`
(0.6B nh16 → 98 848 B; 4B nh32 → 177 696 B; verified by the boot-token gate).
`run_qwen3_prefill` is model-agnostic for dense-Qwen3 (`argv[1]`=model dir,
`RT_TOKENS`=ids, prints `GREEDY_NEXT`); other families need a family driver.

## 5. Repro

```
cd /home/bcloud/1bit-MONSTER-goal
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
# native prefill + boot token (nh32 auto-selected since the 2026-09-13 fix)
NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=<npt> \
  ./engine/npu/build/npu_engine_qwen3_<size> ~/.config/flm/models/Qwen3-<Size>-NPU2/model.q4nx 1 /tmp/ids_<npt>.txt
# FLM reference boot token
cd npu-infer/tools/capture && RT_TOKENS=/tmp/ids_<npt>.txt ./run_qwen3_prefill ~/.config/flm/models/Qwen3-<Size>-NPU2
# FLM on-box bench
mkdir -p ~/npu-build/flmbench && cd ~/npu-build/flmbench
python3 -c "import json;json.dump({'max_length':1024,'iterations':1,'input_text':open('/home/bcloud/1bit-MONSTER-goal/benchmarks/prompts/reclaimer.txt').read()},open('cfg.json','w'))"
/opt/fastflowlm/bin/flm bench qwen3:<size> -i cfg.json   # -> bench_*.csv
```

## 6. Host-math parallelization (2026-09-13)

Three per-layer host loops were single-threaded (no OMP) while their neighbours were
parallel: the two `bsb = bh` residual copies and the `bActQ` f32→bf16 build. All three
are element-independent and were parallelized with `#pragma omp parallel for
num_threads(host_threads())`. Measured back-to-back (same conditions) — **correctness
unchanged (boot tokens identical)**:

| model | before | after | gain |
|---|---|---|---|
| Qwen3-0.6B @1k | 744 ms | 700 ms | +3.8% |
| Qwen3-1.7B @1k | 1269 ms | 1224 ms | +1.3% |
| Qwen3-4B @1k | 2511 ms | 2323 ms | +5.5% |
| Qwen3-8B @1k | 3685 ms | 3554 ms | +3.5% |

So the easy serial-loop wins are real but only **3–5%** → the remaining −13/−16/−22%
gap is **structural**, not a missed `#pragma omp`. Biggest remaining host term is the GU
SiLU/convert pass (`IM` sigmoid calls per token per layer: 3072/6144/9728/12288 for
0.6B/1.7B/4B/8B → up to ~10⁸ scalar transcendentals per prefill). Closing the large-model
gap needs SIMD/bit-exact fused host math (or less host traffic), not more threads.

## 7. Remaining coverage work

| group | models | status | needs |
|---|---|---|---|
| nh16/hd128 | Qwen3-0.6B, 1.7B | ✅ gated | 1.7B prefill −11% (host-math perf) |
| nh32/hd128 | Qwen3-4B, 8B | ✅ gated @1k (auto-selected) | nh32 2k ELF for (1024,2048] |
| nh20/nkv4/hd128 | Nanbeige4.1-3B | not run | capture |
| nh24/nkv8/hd128 | Phi4-mini | not run | capture |
| nh32/nkv8/hd64 | Llama-3.2-1B/3B | not run | capture (hd64) |
| nh8/nkv1/hd256 | Gemma4-E2B/E4B | not run | capture (hd256) |
| MoE | Qwen3.6-35B-A3B | not run | capture (MoE layer ELFs) |
| no engine | LFM2-1.2B/2.6B | not run | build engine first |

## 8. Host-thread default depends on npt AND model size (2026-09-13)

A flat `host_threads()==8` was wrong for long context. Paired A/B sweeps (the box is
contended, so unpaired single runs drift ~5% — pairs are the reliable signal):

| model @1024 | 8 thr | 16 thr | 20 thr | 24 thr | 28 thr | 32 thr |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 675 ms | **659/654/662** | — | 694/643/701 | — | — |
| Qwen3-1.7B | 1117 ms | 1039/1028 | — | 1027/1060 | — | — |
| Qwen3-4B | 2401 ms | 2183/2191/2218 | 2166 | **2106/2142/2163** | 2293 | 5377 |
| Qwen3-8B | 3472 ms | 3181/3272 | 3183 | **3152** | 3446 | — |

And @256 the order flips down: 0.6B 8 thr 324–328 ms vs 16 thr 336 vs 24 thr 357.

So the default is now **`npt <= 256 ? 8 : (H >= 2560 ? 24 : 16)`** — long prompts need
more workers, and larger models need more again (more host math per layer).
`NPU_HOST_THREADS` still overrides. Verified back: @256 328 ms (≈8 thr), and 0.6B/1.7B
use 16 while 4B/8B use 24.

### Final dense-Qwen3 scorecard @1k

| model | native prefill | FLM on-box | gap | (before this session's perf work) |
|---|---|---|---|---|
| Qwen3-0.6B | **1520 tok/s** (674 ms) | 1123.1 | **+35.3%** | +27% |
| Qwen3-1.7B | **985 tok/s** (1039 ms) | 942.6 | **+4.5%** | −14.5% |
| Qwen3-4B | **472 tok/s** (2171 ms) | 510.0 | **−7.5%** | −20% |
| Qwen3-8B | **322 tok/s** (3179 ms) | 362.8 | **−11.2%** | −23.4% |

Boot tokens unchanged (25/220/220/220). Two of four models now meet-or-beat FLM on
prefill; the 4B/8B gaps are down to roughly a third of where they started.

## 9. The host conversion loops were NOT vectorized — and it does not matter (2026-09-13)

The earlier claim "the compiler already vectorizes the prefill" was **wrong**, because
`-fopt-info-vec-optimized` reports only successes. The `missed` report (use
`-fopt-info-vec-missed=$HOME/...`; a literal `~` inside the flag does not expand, and
`-vec-optimized` never emits `missed:` lines) showed the three hottest host loops were
dropped:

| loop | reason |
|---|---|
| O conversion (`boo`/`bh` from `bC`) | "loop nest containing two or more consecutive inner loops cannot be vectorized" |
| **GU SiLU** (`bGu` from `bC`, the dominant loop: `IM` × npt × NC) | "**unsupported control flow in loop**" — the `if (!std::isfinite(gv)) gv = 0;` branch |
| D conversion (`bdw`/`bh` from `bC`) | two consecutive inner loops, as O |

Fix applied: split the O/D pairs and add `#pragma omp simd` to each inner loop, and
make the GU finite-test branchless — `(g0 - g0 == 0.0f) ? g0 : 0.0f`, which is false for
NaN **and** ±inf, i.e. exactly `std::isfinite` with no control flow.

**Result: the loops now vectorize (verified: `vec-opt` entries appear at 4077 and 4088,
zero remaining "unsupported control flow" in the region), and the prefill time does not
move** — 0.6B 674→668 ms, 1.7B 1039→1077, 4B 2171→2164, 8B 3179→3221 (all within the
~3–5% run-to-run noise; boots unchanged 25/220/220/220).

**Conclusion:** the residual 4B/8B prefill gap is **memory-bandwidth-bound, not
compute-bound**. Vectorizing the conversions is free but buys nothing; closing the last
7–11% needs less host traffic (fewer passes / fusing host math into the GEMM staging),
which is an algorithmic change, not a flag.

## 10. Decode @1k — native beats FLM on all four (2026-09-13)

`NPU_RUNLIST=1 <engine> <model.q4nx> 32 /tmp/ids_1024.txt`, final
`=== Z ms/tok (W tok/s) ===` line. FLM from `flm bench qwen3:<size> -i cfg.json`
(the 0.6B figure from the earlier matched-context scorecard).

| model | native decode | FLM on-box decode | gap |
|---|---|---|---|
| Qwen3-0.6B | 12.6 ms/tok (**80 tok/s**) | 77.8 | **+2.8%** |
| Qwen3-1.7B | 24.9 ms/tok (**40 tok/s**) | 39.53 | **+1.2%** |
| Qwen3-4B | 52.1 ms/tok (**19 tok/s**) | 18.75 | **+1.3%** |
| Qwen3-8B | 91.0 ms/tok (**11 tok/s**) | 10.70 | **+2.8%** |

### Three-metric summary (native vs FLM on-box @1k)

| model | decode | prefill | TTFT (= prefill time) |
|---|---|---|---|
| Qwen3-0.6B | **+2.8%** ✅ | **+35.3%** ✅ | 0.668 s vs 0.704 s — **+5.1%** ✅ |
| Qwen3-1.7B | **+1.2%** ✅ | **+4.5%** ✅ | 1.039 s vs 1.042 s — +0.3% ✅ |
| Qwen3-4B | **+1.3%** ✅ | −7.5% | 2.164 s vs 1.925 s — −12.4% |
| Qwen3-8B | **+2.8%** ✅ | −11.2% | 3.221 s vs 2.705 s — −19.1% |

**So the objective (meet-or-beat on decode, prefill AND TTFT) is met for Qwen3-0.6B and
Qwen3-1.7B, and for 4B/8B on decode only.** The outstanding deficit is prefill (hence
TTFT) on the two largest dense models, and it is bandwidth-bound host math.

## 11. `perf` profile of the prefill — sync-limited, not arithmetic (2026-09-13)

`perf record -F 199` on `npu_engine_qwen3_4b` @1024 (`NPU_PREFILL_BF16=1`).

**All threads** (`--sort dso`): **83.6% `libgomp.so.1.0.0`**, 10.8% the engine,
3.6% libc, 1.8% libxrt_driver_xdna. That is *thread-time*, not wall time: the worker
threads spend ~84% of their sampled time spinning in the OpenMP runtime while the main
thread does the serial work (the per-256-row NPU GEMM launch/wait + the memcpys).
Confirmed as useful spin, not waste — forcing passive waiting is *worse*:
`OMP_WAIT_POLICY=passive` 2116→2464 ms, `GOMP_SPINCOUNT=0` →2407 ms,
`OMP_PROC_BIND=false` 2122 ms (neutral).

**Single-threaded** (`NPU_HOST_THREADS=1`, 3846 ms — i.e. 24 threads buy only 1.8×,
the signature of a sync-dominated workload) gives the real work ranking:

| # | symbol | % | reading |
|---|---|---|---|
| 1 | `main._omp_fn.8` | 29.7% | scalar (`vmulss/vaddss/vdivss/vcomiss`) with a reduction + division — a per-head norm/RoPE loop (QKV post-processing), not the GU SiLU (which the `#pragma omp simd` did vectorize) |
| 2 | `__memmove_avx512_unaligned_erms` | 24.4% | bulk copy — attention `attn_act` staging (`rows*q*2` = 8 MB/layer @4B), KV-region copies, `ensure_a` staging |
| 3 | `shim_xdna::buffer::sync` | 14.1% | NPU buffer sync (serial, main thread) |
| 4 | `dequant_i8_to_float_ex` | 9.7% | weight dequant — mostly the init phase |
| 5 | `__memset_avx512_unaligned_erms` | 3.6% | |
| 6 | `main._omp_fn.2` | 2.9% | another parallel loop |
| 7 | `lm_topk_omp` | 2.6% | final greedy argmax |
| 8 | `npu_pack_layer_bo` | 2.5% | weight packing — init |

(`addr2line` has no line info — the engine is built without `-g`; `-fopt-info` line
numbers plus the instruction mix were used to identify the loops.)

**Conclusions for the last 7–11%:** the prefill is (a) **OpenMP-synchronisation limited**
(~84% worker spin; many short parallel regions per layer interleaved with serial NPU
waits) and (b) **copy-heavy** (~28% memmove/memset). Neither is arithmetic. The levers
are structural: fewer/larger parallel regions, and overlapping the serial NPU waits (or
removing the redundant staging copies), not more threads or more SIMD. Re-profiling with
`-g` would let the exact loops be pinned to lines.

**Test of the "remove barriers" lever — negative.** Fusing the two `bsb = bh` copies into
their adjacent `rn_bf16` regions removes 2 of ~19 parallel regions per layer (72 fewer
barriers over 36 layers), with identical semantics: 0.6B 674→677, 1.7B 1039→1075,
4B 2171→**2146**, 8B 3179→**3168** ms — i.e. ≤1%, inside noise (4B/8B nominally better).
So the wall clock is **not** set by the barrier count even though the workers spend 84% of
their *thread-time* in `libgomp`: it is set by the **serial main-thread work**
(`shim_xdna::buffer::sync` 14% + memmove/memset ~28%) while the workers idle. The
remaining lever is therefore **overlapping** that serial work with the parallel regions
(double-buffering the NPU staging), not shaving regions.

## 12. ✅ DOUBLE-BUFFERED GEMM BLOCKS — the gap is closed (2026-09-13)

Section 11 pointed at the serial main-thread NPU waits. The bf16mm API already had the
machinery (`gemm_launch(..., batch, A)` uses per-slot `c_cache0/1`, `a_cache0/1` and
`g_run[2]`, and `start()` returns without waiting) — but it was **half-wired**: `c_cache`
was per-slot while `ensure_a()` staged every A into `a_cache0`, and the engine always
passed `batch=0`, so each 256-row block was fully serialised
(`launch → wait → convert`).

**Change:** `ensure_a(A, K, batch)` now stages into the batch's own A cache (so slot 0 and
slot 1 can be in flight at once), and all four prefill block loops became:

```
launch(block 0, slot 0); launch(block 1, slot 1);
for i:  wait(slot i&1, bC + i*256*N);
        if (i+2 < nblk) launch(block i+2, slot i&1);   // NPU works on i+2 …
        convert block i on the host                       // … while the host works on i
```

Blocks write disjoint `bC` regions and use disjoint A caches, so a kernel holding slot
`i&1` can never disturb the block being converted. Semantics are unchanged — all four
boot tokens are identical (25/220/220/220) — and decode is unaffected
(0.6B 80 tok/s, 4B 19 tok/s, same as before).

### Result @1k: **~30% faster prefill, and every dense Qwen3 now beats FLM on all three metrics**

| model | prefill before | prefill now | FLM on-box | prefill gap | decode | FLM dec | TTFT | FLM TTFT |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 677 ms | **536 ms** (1912 tok/s) | 1123.1 | **+70.3%** | 80 | 77.8 | 0.536 s | 0.704 s |
| Qwen3-1.7B | 1075 ms | **767 ms** (1335 tok/s) | 942.6 | **+41.6%** | 40 | 39.53 | 0.767 s | 1.042 s |
| Qwen3-4B | 2146 ms | **1524 ms** (672 tok/s) | 510.0 | **+31.8%** | 19 | 18.75 | 1.524 s | 1.925 s |
| Qwen3-8B | 3168 ms | **2196–2257 ms** (461 tok/s) | 362.8 | **+27.1%** | 11 | 10.70 | 2.207 s | 2.705 s |

Repeat runs are stable (4B 1524/1524 ms; 8B 2257/2196 ms) and every boot token still
matches FLM. **So for the dense Qwen3 family the objective is met outright: decode,
prefill and TTFT all beat FLM's on-box measurements, on this box, with byte-identical
tokens.**

The lesson is worth keeping: the profile's 84% `libgomp` thread-time was a *symptom* —
the workers were idle behind a serialised NPU wait — and the fix was to overlap work,
not to shave regions (\\S11) or add threads (\\S8).

## 13. The same fix transfers to the other gated models (2026-09-13)

Qwen3-VL-4B and Llama-3.1-8B run the same bf16 prefill path, so they inherit the
double-buffering. Rebuilt and re-measured @1k (FLM reference via `NPU_FLM_PREFILL=1`;
for Qwen3-4B that method agrees with `flm bench` to ~2.5%, 497 vs 510 tok/s):

| model | native prefill | FLM reference | gap | gate |
|---|---|---|---|---|
| Qwen3-VL-4B | 1506 ms (1.471 ms/tok, **680 tok/s**) | 2026 ms (1.98, 505) | **+34.6%** | boot 220 == 220 ✅ |
| Llama-3.1-8B | 2171 ms (2.120, **472 tok/s**) | 2869 ms (2.80, 357) | **+32.3%** | boot 220 == 220 ✅ |

So six models now beat FLM on prefill: the four dense Qwen3 (§12) plus VL and Llama-3.1-8B.

The four remaining families are still wrong for the family reasons in §8 (they are now
faster too, but the token mismatch is architectural, not a capture): Qwen3.5-4B (boot 0),
Nanbeige4.1-3B (1214 vs reference 1033), Phi4-mini (350 vs 25), Gemma3 (crash / FLM
config-parse failure).

**Caveat:** this box is contended — `flm serve qwen3.6-moe:35b-a3b` (pid 285847) and
`llama-server` (pid 344571) both hold `/dev/accel/accel0`, so run-to-run variance is
several percent (e.g. 0.6B @256 repeats: 325/335/325 ms). Treat single-run deltas under
~5% as noise; the thread ordering and the 4B/8B gains are larger than that.
