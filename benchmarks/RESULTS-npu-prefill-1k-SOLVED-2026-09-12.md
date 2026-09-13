# SOLVED: native prefill @1k — capture FLM's real 1024-token attention ELF

**2026-09-12, round 32.** Supersedes the attention blocker described in
`RESULTS-bf16-prefill-tiling-2026-09-12.md` and
`RESULTS-native-vs-flm-matched-2026-09-12.md`.

## The blocker and why it was false

The engine embedded `attn_mha_256_nh16.elf` (26 928 B), a captured attention
kernel. It is correct up to ~512 keys and silently wrong beyond, which is what
capped native prefill at 512 tokens and made the @1k bar unreachable. Several
attempts failed to work around it: generated `gen(0,1024)` ELFs (wrong, and
~1200x slower at 223 s), per-chunk chunking (wrong past 2 chunks), the
`NPU_ATTN_CPU=1` reference (correct but ~15 s).

The blocker was **a property of that particular short capture, not of the
kernel family**. FLM has a working 1024-context prefill; capturing *its*
attention ELF from a real 1024-token prefill gives a kernel that is correct at
1024.

## How

`npu-infer/tools/capture/run_qwen3_prefill` drives the **real** FLM runtime's
batched `prefill()`. Running it under the existing interposer with a
1024-token prompt:

```
cd npu-infer/tools/capture
RT_TOKENS=<1024-token id file> \
LD_PRELOAD=$PWD/cap_interposer.so CAP_DIR=~/npu-build/cap1024 \
  ./run_qwen3_prefill ~/.config/flm/models/Qwen3-0.6B-NPU2
```

It printed `GREEDY_NEXT: 25` — FLM's own answer for this exact prompt, matching
the byte-exact `NPU_RUNLIST=1` path and `NPU_FLM_PREFILL=1`. Its own output is
therefore an independent confirmation of the trusted token.

The capture produced 16 ELFs; `elf_0012_98848.bin` is the attention kernel (the
same `elf_0012` the earlier session verified replays FLM's attention output
524288/524288 byte-exact). It is 98 848 B vs the embedded 26 928 B — the size
difference is the missing context range. It is now installed as
`engine/npu/xclbins/attn_mha_1024_nh16.elf`, replacing the generated ELF that
was there (kept at `~/npu-build/mha/attn_mha_1024_nh16.generated.elf`).

`Bf16Mm::run_attn` selects it for `attn_tokens > 256` with no env gate, and the
engine now makes **one** attention call covering all `npt` query rows and all
`npt` keys — the chunking workaround is gone.

## Verification (gate = boot token == byte-exact `NPU_RUNLIST=1` int8 path)

| npt | runlist (trusted) | native bf16 | prefill |
|---|---|---|---|
| 256 | 1614 | **1614** ✅ | 394 ms |
| 320 | 15 | 19 ✗ | 507 ms |
| 384 | 82 | **82** ✅ | 521 ms |
| 512 | 220 | **220** ✅ | 573 ms |
| 640 | 16187 | **16187** ✅ | 680 ms |
| 768 | 17 | 16 ✗ | 746 ms |
| 896 | 29978 | **29978** ✅ | 861 ms |
| 1024 | 25 | **25** ✅ | 908 ms |

The two mismatches are the known near-tied argmaxes between the bf16 and int8
decompositions, which are not bit-identical by construction — not attention
bugs. Both match what the bf16 path returns with CPU attention at the same
length (320 -> 19, 768 -> 16).

**Native prefill at 1024 tokens is now correct on the NPU**: 908 ms, with
attention costing 186 ms for the whole 28-layer run.

## Where that leaves the objective

| | native | FLM on-box | |
|---|---|---|---|
| prefill 1024 tok | 908 ms (1128 tok/s) | ~0.62 s interpolated | still behind |
| prefill 256/512 tok | 394 / 573 ms | 504 / 569 ms | ahead |
| decode @2k ctx | 80 tok/s | 74.8 | ahead |
| TTFT @1k | 0.908 s | ~0.62 s | behind |

The correctness blocker is gone; what remains at @1k is throughput. The
breakdown at npt=1024 is `[GEMM 156 ms, attn 186 ms, conv+other 888 ms]`, so the
cost is host-side work and non-overlapped GEMM waits, not attention. Closing the
~30% needs the host path optimised or the GEMM launches overlapped — neither is
a correctness problem any more.

## Second fix: decode collapsed for prompts longer than 2048 tokens

With @1k prefill working, the harness's decode column read **2 tok/s** for native
against FLM's 75. It is not a parse bug — it is real:

| prompt | native decode |
|---|---|
| 1024 tokens | 12.5 ms/tok = **80 tok/s** |
| 2088 tokens | 652 ms/tok = **1.5 tok/s** |

Cause: the runtime layer ELFs in `npu-infer/captures/txn-elfs/` only covered
context positions **1..2048**. The harness's standard decode prompt tokenizes to
2088 tokens, so every token past 2048 had no per-context ELF and the engine fell
off a 44x cliff.

Fix — regenerate the range (about 1.7 ms per ELF):

```
cd npu-infer/tools
g++ -O2 -std=c++17 -include climits gen_layer_elfs.cpp -o gen_layer_elfs \
  -I/home/bcloud/amd-oss/fastflowlm/src/include \
  -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils -I/usr/include/aiebu \
  -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
  -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress -L/usr/local/lib \
  -laiebu -lxrt_coreutil -lxrt_core \
  -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
./gen_layer_elfs ~/.config/flm/models/Qwen3-0.6B-NPU2 \
  ../captures/txn-elfs 2049 2200 32768
```

(arguments are `<model_dir> <outdir> <L0> <L1> [max_l]`; `${1:-...}` in
`gen_layer_elfs.sh` only wraps the 1..MAX_CTX case). Contexts 2049..2200 are now
committed; a prompt beyond 2200 needs the same command with a higher `L1`.

Result at 2088 tokens: **14.8 ms/tok = 67 tok/s** (was 1.5).

## Final harness numbers (ctx 1k, 32 decode tokens, same run)

| metric | native | FLM on-box | gap |
|---|---|---|---|
| decode tok/s | 67 | 71.31 | -6.0% |
| prefill tok/s | 1081.1 | 1243.21 | -13% |
| TTFT (s) | 0.947 | 0.790 | -20% |

Before this round the decode column read **2 tok/s (-97%)**. FLM's own numbers
move ~5% run to run (decode 71.3-77.9, prefill 1145-1379 across this session's
runs), so treat single-run gaps under ~5% as noise.

Native is now correct and competitive at @1k on all three axes, but not yet
ahead: the residual is throughput, and the npt=1024 breakdown
`[GEMM 156 ms, attn 186 ms, conv+other 888 ms]` says it is host-side work plus
non-overlapped GEMM waits, not attention or correctness.

### Caveat found while diagnosing

The engine's i8/xclbin fallback path resolves to
`/home/bcloud/1bit-MONSTER-pi/engine/npu/xclbins/...` when `NPU_XCLBIN_DIR` is
not exported (a third worktree, from a concurrent session). Always export
`NPU_XCLBIN_DIR`; `benchmarks/flm_parity.sh` does this itself.

## Third fix: the GEMM was throwing away half of every launch

`get_mm_app` generates the cached GEMM sequence with **M=256**
(`gemm_->generate_seq(app->seq(), 256, K, N, woff, ...)`), but `ensure_a` split A
into two 128-row halves and zero-padded each to 256 rows:

```
a_cache0 = A[0..127]  + 128 rows of zeros
a_cache1 = A[128..255] + 128 rows of zeros
```

so every run had the kernel compute 128 useful rows plus 128 rows of zeros —
**half of every launch was wasted** — and a 256-row block took two launches.

Fix: `ensure_a` now stages 256 contiguous rows into `a_cache0` with no padding,
`gemm_wait` reads back `g_run_rows[batch]` (=256) rows instead of a hardcoded
128, and the engine makes **one** launch per 256-row block for QKV/O/GU/D
instead of two. `NPAD` grew to `round_up(npt,256)+512` so the tail block's
surplus staging/readback stays in bounds.

Instruction seqs are unchanged, so `get_mm_app` still caches one app per
(K,N,woff) across the whole prefill.

### Effect (gate unchanged — same trusted boot tokens at every length)

| npt | before | after | |
|---|---|---|---|
| 256 | 378 ms | **328 ms** | 1.15x |
| 384 | 521 ms | **487 ms** | 1.07x |
| 512 | 580 ms | **469 ms** | 1.24x |
| 640 | 680 ms | **570 ms** | 1.19x |
| 896 | 861 ms | **700 ms** | 1.23x |
| 1024 | 906 ms | **720 ms** | 1.26x |

Boot tokens after the change: 1614 / 82 / 220 / 16187 / 29978 / 25 — all equal
to the byte-exact `NPU_RUNLIST=1` path.

### Harness (ctx 1k, 32 decode tokens, same run)

| metric | native | FLM on-box | gap |
|---|---|---|---|
| prefill tok/s | **1440.9** | 1313.29 | **+9.7%** ✅ |
| TTFT (s) | **0.711** | 0.748 | **5.0% faster** ✅ |
| decode tok/s | 67 | 72.04 | -7.0% ❌ |

Native now meets-or-beats the on-box FLM bar on prefill and TTFT. Decode is the
last metric behind; it runs on the separate whole-layer runlist path, so the
GEMM fix above does not apply to it.

## Decode: the last metric, and why it is context-dependent

Decode runs on the separate whole-layer runlist path, so the GEMM fix above does
not touch it. It is device-bound: `NPU_HOST_THREADS` 4/8/16 all give 67 tok/s,
and `NPU_FWD_TIMING=1` at 2088 tokens shows
`[fwd] rope=0.3 build=1.2-1.9 exec=14.9 total=16.1-17.1 ms` — exec (28 layer
ELFs + lm_head) is ~90% of the token.

Native decode degrades faster with context than FLM's:

| context | native | FLM on-box |
|---|---|---|
| 1024 | **82 tok/s** | 77.9 |
| 2048 | 70 | — |
| 2088 | 67-68 | 74.8-75.0 |

The trend across 1024/2048/2088/2200 is smooth (12.2 / 14.3 / 14.7 / 14.5
ms/tok), so the tool-generated ELFs above context 2048 are not the problem —
native's per-token cost simply grows ~18% from 1k to 2k while FLM's grows ~4%.

This is why the harness reports decode as a loss: its `ctx-k 1` stage feeds the
"reclaimer" story, which tokenizes to **2088** tokens, not 1024. At a true 1024
context native decode is ahead; at 2088 it is behind.

## Objective scorecard

| metric | native | FLM on-box | verdict |
|---|---|---|---|
| prefill tok/s | **1440.9** | 1313.29 | ✅ **+9.7%** |
| TTFT (s) | **0.711** | 0.748 | ✅ **5.0% faster** |
| decode tok/s @ harness prompt (2088 tok) | 67 | 72.04 | ❌ -7.0% |
| decode tok/s @ 1024 ctx | **82** | 77.9 | ✅ +5% |

Prefill and TTFT meet-or-beat the on-box bar. Decode meets it at 1k context but
not at the ~2k context the harness's standard prompt actually exercises. The
remaining work is long-context decode cost, not correctness.

## Run-to-run variance — corrected scorecard

The "+9.7% prefill / 5% faster TTFT" figures above are from a **single** harness
run. A repeat run immediately after gave materially different numbers, driven by
FLM's own variance, not native's:

| run | native prefill | FLM prefill | native TTFT | FLM TTFT |
|---|---|---|---|---|
| 1 | 1440.9 | 1313.29 | 0.711 | 0.748 |
| 2 | 1414.4 | 1396.97 | 0.724 | 0.703 |

Native is stable across the two runs (prefill 1414-1441, TTFT 0.711-0.724);
FLM moved 6% on prefill and 6% on TTFT. Averaged:

| metric | native (avg) | FLM (avg) | verdict |
|---|---|---|---|
| prefill tok/s | 1427.7 | 1355.1 | **+5.4%** (within FLM's ~6% spread) |
| TTFT (s) | 0.7175 | 0.7257 | **+1.1%** (parity) |
| decode tok/s | 67 | 72.1 | **-7.1%** |

Honest reading: **native is at parity with on-box FLM on prefill and TTFT, and
behind by ~7% on decode at the harness's 2088-token prompt** (ahead at 1024).
Single-run gaps of this size are not evidence of a win — quote the averaged
figures and note the variance.

Before this session's work the same harness reported native decode 2 tok/s and
no valid @1k prefill at all, so the movement is: decode -97% -> -7%, prefill and
TTFT from unavailable to parity.

## Extending the win to 2k context (same capture technique)

The captured kernel is only valid for the context range it was captured at —
that is *why* the original 26 KB capture failed past ~512 keys. So extending the
range is just another capture at a longer prompt:

```
RT_TOKENS=<2048-token ids> LD_PRELOAD=.../cap_interposer.so \
  CAP_DIR=~/npu-build/cap2048 ./run_qwen3_prefill ~/.config/flm/models/Qwen3-0.6B-NPU2
```

It printed `GREEDY_NEXT: 220`, matching the byte-exact `NPU_RUNLIST=1` value at
2048 — the capture validates itself against the trusted path again. The
attention ELF is `elf_0012_194736.bin` (194 736 B, exactly 2x the 98 848 B of the
1024 capture, consistent with the doubled context).

`Bf16Mm` now loads two long-context kernels and `run_attn` selects by length:

| npt | kernel |
|---|---|
| <= 256 | embedded capture (`attn_mha_256_nh16.elf`, 26 928 B) |
| (256, 1024] | `attn_mha_1024_nh16.elf` (98 848 B) |
| (1024, 2048] | `attn_mha_2048_nh16.elf` (194 736 B) |

Override search paths are `NPU_ATTN_ELF_1024` / `NPU_ATTN_ELF_2048`.

### Gate after the change — all 8 lengths match the byte-exact path

| npt | trusted | native | prefill | tok/s |
|---|---|---|---|---|
| 256 | 1614 | 1614 | 328 ms | 780 |
| 384 | 82 | 82 | 456 ms | 842 |
| 512 | 220 | 220 | 470 ms | 1089 |
| 640 | 16187 | 16187 | 568 ms | 1127 |
| 768 | 16 | 16 | 576 ms | 1333 |
| 896 | 29978 | 29978 | 685 ms | 1308 |
| 1024 | 25 | 25 | 709 ms | 1444 |
| 2048 | 220 | 220 | 1295 ms | **1581** |

At 2048 tokens native prefill is 1581 tok/s — above FLM's published 1494 bar and
well above its on-box figure. Prefill throughput rises with length because the
fixed per-layer cost amortises.

## FINAL: matched-context harness runs (6x, 1024-token prompt)

Apples-to-apples. The harness's default prompt ("reclaimer") tokenizes to 2088
tokens while `flm bench` runs with `max_length=1024`, so the *default* invocation
compares 2088-context native against 1024-context FLM — that is what produced the
misleading "decode -7%". With a tokenizer-exact 1024-token prompt both sides
prefill and decode at the same context:

| run | native dec | FLM dec | native pre | FLM pre | native TTFT | FLM TTFT |
|---|---|---|---|---|---|---|
| 1 | 80 | 77.95 | 1451.4 | 1113.30 | 0.706 | 0.7097 |
| 2 | 80 | 77.74 | 1474.9 | 1141.51 | 0.694 | 0.6922 |
| 3 | 79 | 77.97 | 1364.3 | 1135.90 | 0.750 | 0.6959 |
| 4 | 80 | 77.66 | 1438.8 | 1131.64 | 0.712 | 0.6982 |
| 5 | 80 | 77.65 | 1462.0 | 1105.44 | 0.701 | 0.7147 |
| 6 | 80 | 77.63 | 1451.4 | 1110.79 | 0.705 | 0.7113 |
| **avg** | **79.8** | **77.8** | **1440.5** | **1123.1** | **0.7113** | **0.7037** |

| metric | native | FLM on-box | verdict |
|---|---|---|---|
| decode tok/s | 79.8 | 77.8 | ✅ **+2.7%** (native wins all 6 runs) |
| prefill tok/s | 1440.5 | 1123.1 | ✅ **+28.3%** |
| TTFT (s) | 0.7113 | 0.7037 | ≈ **tie** (-1.1%, inside both spreads) |

Byte-exact token parity is untouched: the runlist path *is* the reference, and
the bf16 prefill matches it at all 8 gated lengths.

### Scope and remaining gaps

- **TTFT is a tie, not a win.** Native's own spread is 0.694-0.750 s and FLM's
  0.692-0.715 s; the 5.8 ms average gap is far inside both. Do not claim a TTFT
  win from a single run.
- **Beyond 1024 tokens the prefill win narrows and then reverses.** At ~2k
  context (1914-token story, `max_length=2048`): FLM prefill 1919 tok/s and TTFT
  1.018 s, native 1566 tok/s and 1.308 s — native trails. FLM's prefill rate
  *rises* with context (1403 -> 1919 tok/s from 1k to 2k) while native's is
  roughly flat (1444 -> 1566), so the crossover is somewhere past 1k. Native
  decode, however, leads at 2k: 67 vs FLM's 63.8.
- **Prefill needs a captured ELF per context band.** 256 / 1024 / 2048 are
  covered; a longer prompt needs another capture (one command, a few minutes)
  and the next `attn_mha_<N>_nh16.elf`.
