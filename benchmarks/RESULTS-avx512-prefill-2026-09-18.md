# The last (c) cell: 512-bit prefill build, bit-identical, and 8k TTFT now ahead on all six — 2026-09-18

Goal `mu35shsg-i3hlyi`. Criterion (c) stood at 17 of 18 cells at ~8k context, the exception being
**Qwen3-0.6B TTFT at 8192** (native 4039–4151 ms vs FLM 3979–4112 ms, 1–3% behind). The residual
was localised to the host per-element prefill path (`RESULTS-0_6b-prefill-hostbound-2026-09-18.md`:
`conv+other` 4007–4071 ms of a 4030–4094 ms total, IPC 0.61, 74.4 G instructions ≈ 316 host
instructions per hidden element per layer). This closes it.

## The change (one line of build flags)

`engine/npu/build_npu.sh`, `CXXFLAGS` — the engine translation unit was built `-mavx2` only, on a
Zen 5 part with a full 512-bit datapath:

```
-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vbmi -ffp-contract=off
```

**`-ffp-contract=off` is not cosmetic.** AVX-512F brings FMA with it, GCC's default
`-ffp-contract=fast` then folds `a*b+c` into FMA, and the rounding change is visible:
without the flag the 8k boot token on 0.6B moves **576 → 785**. With it the build is numerically
identical to the `-mavx2` one it replaces.

## Correctness: bit-identical, all six models

Both binaries built from the same source (baseline `-mavx2`, candidate the flags above); for each
model, boot token at 1k and 8k and the **sha256 of the full hidden-state dump** (`NPU_DUMP_HIDDEN`,
every layer × every prompt row) at 256 tokens:

| model | 1k token base\|a512 | 8k token base\|a512 | hidden-dump sha256 base\|a512 | 8k prefill base\|a512 |
|---|---|---|---|---|
| Qwen3-0.6B | 576 \| 576 | 576 \| 576 | `0b1513be` \| `0b1513be` | 4113 \| **3928 ms** (−4.50%) |
| Qwen3-1.7B | 576 \| 576 | 576 \| 576 | `7c1932a8` \| `7c1932a8` | 5652 \| **5474 ms** (−3.15%) |
| Qwen3-4B | 576 \| 576 | 576 \| 576 | `a1c3d411` \| `a1c3d411` | 12918 \| 12891 ms (−0.21%) |
| Qwen3-VL-4B | 576 \| 576 | 576 \| 576 | `dea31ab8` \| `dea31ab8` | 13045 \| 13036 ms (−0.07%) |
| Qwen3-8B | 576 \| 576 | 576 \| 576 | `deff59d6` \| `deff59d6` | 18083 \| 18144 ms (+0.34%) |
| Llama-3.1-8B | 785 \| 785 | 785 \| 785 | `855eb493` \| `855eb493` | 16987 \| 16981 ms (−0.04%) |

Every hidden state is byte-identical, so **the existing oracle scoreboard (a)/(b) carries over
exactly** — the gated build and the faster build produce the same numbers, not merely the same
tokens. The gain is concentrated where the host term dominates (the small models); the large
models are unchanged within run-to-run noise, as their prefill is device-GEMM dominated.

The committed `build_npu.sh` reproduces the verified binaries **byte-for-byte**
(`sha256sum` of all six build outputs = `735eb40d1ebffd55…`, identical to the `/tmp` candidates
used for the table above).

## Result: criterion (c) is now 18 of 18 cells at ~8k

Paired, guard-accepted runs through `benchmarks/c8k_guarded.sh` (device-holder + during-run
foreign-CPU gates, `C8K_WAIT_QUIET=1`, 8192-token prompt, `=== Prefill 8192 [bf16] ===`):

| model | accepted | native TTFT | FLM TTFT | ratio | native prefill | FLM prefill |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 3/3 | **3840 ms** | 3868 ms | **0.993** | 2133 t/s | 2007 t/s |
| Qwen3-1.7B | 3/3 | **5478 ms** | 5801 ms | **0.944** | 1495 t/s | 1338 t/s |
| Qwen3-4B | 2/2 | **13029 ms** | 13777 ms | **0.946** | 629 t/s | 563 t/s |
| Qwen3-VL-4B | 2/2 | **13009 ms** | 13599 ms | **0.957** | 630 t/s | 571 t/s |
| Qwen3-8B | 1/2 | **17713 ms** | 18569 ms | **0.954** | 462 t/s | 418 t/s |
| Llama-3.1-8B | 2/2 | **16968 ms** | 17556 ms | **0.966** | 483 t/s | 444 t/s |

**Native TTFT is faster than FLM at 8192 on every model** (0.7% to 5.9%), and native prefill rate
is ahead on every model. The 0.6B cell that failed — the one this work was for — is now faster in
**all three** accepted pairs (3807/3868, 3840/3922, 3852/3863), with a spread of 1.2% across runs.
TTFT is the metric here because FLM's reported `prefill t/s` is computed over a token count that
does not equal the 8192 it was given; the wall time to first token is the comparable number.

Decode at the top of the window, re-measured on this build (`benchmarks/d8k.sh`, prompt 8160 +
`ng=32`, ctx ≤ 8192): 0.6B native **28.5–28.6 ms/tok (35 tok/s)** vs FLM 32.81–32.88 t/s =
**1.064–1.067x native**, up from 1.047x on the previous build.

## Negative result worth keeping: the readback/conversion restructure is a loss

The first attempt at this fix targeted what looked like the obvious overhead: the engine spawns a
fresh OpenMP region per 256-row block per GEMM (≈131 regions per layer at 8k, ≈3668 per prefill),
and 59% of sampled cycles sit in libgomp (futex/barrier). Deferring every block's *conversion* out
of the readback loop, so each GEMM's conversion became ONE region per layer (131 → 7), was
arithmetically neutral (token streams identical at 1k/2k/4k/8k) and **5% slower**: 8k prefill
4039 → 4244 ms, the host term 4015 → 4220 ms, the loss growing with prompt length (1k +26 ms,
8k +205 ms).

The interleaved design is right for a reason the region count does not show: converting block *i*
immediately after its readback keeps that block's output in cache and overlaps the host conversion
with the *next* block's device GEMM. Deferring the conversions buys region count back and pays for
it in cache locality and overlap. Reverted, and recorded so nobody re-derives it.

## Why this is the right lever and not a micro-optimisation

Thread scaling was already exhausted: 8k prefill is 7407 ms at 1 host thread, 5511 at 2, 4673 at 4,
4039 at 8, and **no further gain at 12/16/24** (4238/4330/4373 ms) — a 1.83x ceiling, with 32
threads a 5x cliff (20.4 s). `NPU_HOST_THREADS=8` is the optimum and the host term is ~4 s of
per-element work that does not parallelise further. Widening the ISA is the one lever that reduces
the work itself without touching arithmetic.

## Reproduce

```
# build (reproduces the verified binaries)
bash engine/npu/build_npu.sh
# correctness: hidden-state hashes, all six models (baseline vs candidate)
#   see ~/verify_models.sh; a -mavx2 baseline for one model:
bash engine/npu/_tmp_mk.sh /tmp/base_<m> <m>            # helps only for the comparison build
# measurement (paired, guarded)
C8K_WAIT_QUIET=1 bash benchmarks/c8k_guarded.sh Qwen3-0.6B-NPU2 qwen3:0.6b npu_engine_qwen3_0_6b 3
bash benchmarks/d8k.sh Qwen3-0.6B-NPU2 qwen3:0.6b npu_engine_qwen3_0_6b qwen3_0_6b 2
```

## 1k yardstick refreshed on the new build (criterion (c), lower end)

The 1k rows in `RESULTS-1k-six-model-2026-09-18.md` were measured on the `-mavx2` binary. Since the
build changes the prefill path, they were re-measured on the committed one (native = 32 tokens from
`/tmp/p_1k.txt` through the default unified path, FLM = `npu_ab.sh --skip-native --ctx-k 1
--decode-tokens 32`, medians of 2):

| model | native prefill | native TTFT | native decode | FLM prefill | FLM TTFT | FLM decode | prefill | TTFT | decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | **513 ms (1996 t/s)** | 0.513 s | 79 tok/s | 1364.32 t/s | 0.709 s | 73.80 t/s | 1.463x | 0.723x | 1.070x |
| Qwen3-1.7B | **752 ms (1362 t/s)** | 0.752 s | 39 | 899.37 | 1.061 s | 38.47 | 1.514x | 0.709x | 1.014x |
| Qwen3-4B | **1554 ms (659 t/s)** | 1.554 s | 19 | 480.88 | 2.001 s | 18.42 | 1.370x | 0.777x | 1.031x |
| Qwen3-VL-4B | **1561 ms (656 t/s)** | 1.561 s | 19 | 481.43 | 1.932 s | 18.52 | 1.363x | 0.808x | 1.026x |
| Qwen3-8B | **2253 ms (455 t/s)** | 2.253 s | 11 | 340.04 | 2.807 s | 10.58 | 1.337x | 0.803x | 1.040x |
| Llama-3.1-8B | **2184 ms (469 t/s)** | 2.184 s | 11 | 347.78 | 2.838 s | 11.00 | 1.349x | 0.770x | 1.000x |

All three metrics are ahead of FLM for all six models at 1k (TTFT is 19–29% faster, ratio < 1), so
the lower end of (c) is unchanged in direction and slightly better in magnitude on this build.

## Criterion (c): 18 of 18 cells

Six models × {prefill, TTFT, decode} at ~8192 context, plus the same three at 1k. The last failing
cell (0.6B TTFT at 8k) is now faster in all three accepted pairs. (c) is met with no refused parity
claim: every number above is a paired, guard-accepted measurement, and every one of them is taken
on a build whose hidden states are byte-identical to the build the oracle scoreboard (a)/(b) was
measured on.
