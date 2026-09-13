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

## 8. npt-dependent host-thread default (2026-09-13)

A flat `host_threads()==8` was wrong for long context. Sweep @1024 (native bf16 prefill):

| model | 4 thr | 8 thr | **16 thr** | 32 thr |
|---|---|---|---|---|
| Qwen3-0.6B | — | 675 ms | **658 ms** | — |
| Qwen3-1.7B | — | 1117 ms | **1076 ms** | — |
| Qwen3-4B | 2616 ms | 2401 ms | **2211 ms** | 5319 ms |
| Qwen3-8B | — | 3472 ms | **3181 ms** | — |

But @256 the ordering flips (0.6B: 8 thr 324–328 ms vs 16 thr 336 ms). So
`host_threads()` now defaults to **16 for npt > 256, else 8** (env `NPU_HOST_THREADS`
still wins; measured back: default@256 = 325/335/325 ms ≈ t=8, default@1024 ≈ t=16).

### Updated dense-Qwen3 scorecard @1k (after loop parallelization + thread default)

| model | native prefill | FLM on-box | gap | (was, §2) |
|---|---|---|---|---|
| Qwen3-0.6B | **1536 tok/s** (667 ms) | 1123.1 | **+36.8%** | +28.3% |
| Qwen3-1.7B | **948 tok/s** (1081 ms) | 942.6 | **+0.5%** | −13.3% |
| Qwen3-4B | **465 tok/s** (2200 ms) | 510.0 | **−8.8%** | −15.6% |
| Qwen3-8B | **319 tok/s** (3207 ms) | 362.8 | **−12.0%** | −22.4% |

Boot tokens unchanged (25/220/220/220). So Qwen3-1.7B now **meets** FLM on prefill, and
the 4B/8B gaps roughly halved.

**Caveat:** this box is contended — `flm serve qwen3.6-moe:35b-a3b` (pid 285847) and
`llama-server` (pid 344571) both hold `/dev/accel/accel0`, so run-to-run variance is
several percent (e.g. 0.6B @256 repeats: 325/335/325 ms). Treat single-run deltas under
~5% as noise; the thread ordering and the 4B/8B gains are larger than that.
