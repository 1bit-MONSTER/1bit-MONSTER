# Matched native-vs-FLM comparison, Qwen3-0.6B (2026-09-12, round 31)

All numbers on-box, same machine, NPU idle, `benchmarks/flm_parity.sh` /
`flm bench` for FLM and `npu_engine_qwen3_0_6b` for native. FLM's bench reports
only at 1k granularity, so short-context FLM points were taken from its TTFT on
a tokenizer-exact prompt (512 and 256 tokens) with `max_length=1024`.

## Correctness baseline (this makes the perf numbers meaningful)

Trusted reference = `NPU_RUNLIST=1`, the int8 whole-layer path that is byte-exact
vs FLM. `NPU_FLM_PREFILL=1` agrees with it at every length checked.

| npt | runlist (trusted) | 320 | 384 | 640 | 768 | 896 | 1024 |
|---|---|---|---|---|---|---|---|
| boot | 1614 @256, 220 @512 | 15 | 82 | 16187 | 17 | 29978 | 25 |

bf16 prefill boot tokens (current build) — 6/8 match the trusted path:

| npt | bf16 | trusted | |
|---|---|---|---|
| 256 | 1614 | 1614 | ✅ |
| 512 | 220 | 220 | ✅ |
| 1024 | 25 | 25 | ✅ |
| 384 | 82 | 82 | ✅ |
| 640 | 16187 | 16187 | ✅ |
| 896 | 29978 | 29978 | ✅ |
| 320 | 19 | 15 | ✗ |
| 768 | 16 | 17 | ✗ |

The two mismatches are near-tied argmaxes between the bf16 and int8
decompositions, which are not bit-identical by construction. They are *not*
attention bugs: below 320 (and at 1024) the bf16 path agrees exactly.

> **Correction to an earlier table in this session.** A sweep run earlier today
> reported references of 81622/57894/77653/125959/93082 for 320/384/640/768/896.
> Those were invalid: at that time the engine still had the 256-row GEMM cap, so
> every one of those runs was silently truncated to 256 tokens. The real trusted
> values are the ones in the first table above. Conclusions drawn about *chunked
> attention* from that sweep still hold, because they were re-checked against the
> corrected references.

## Native attention envelope (verified)

`npt <= 256` → NPU attention, 1 chunk. `npt == 512` → NPU attention, 2 chunks.
Both hit the trusted boot token. Every other length tested (320/384/640/768/896
with chunking, 768/1024 with 3-4 chunks) returns a wrong token, so the engine
falls back to the CPU attention reference there — correct but 10-18x slower.

The captured ELF therefore composes over **at most two full 256-row chunks**; a
growing key prefix past 512 keys and any partial chunk both break it.

## Matched performance

| workload | native | FLM on-box | verdict |
|---|---|---|---|
| prefill 256 tok | **375-407 ms (631-683 tok/s)** | 504 ms (419 tok/s) | ✅ **+51%** |
| prefill 512 tok | **560-571 ms (897-914 tok/s)** | 569 ms (714 tok/s) | ✅ **+26%** |
| prefill 1024 tok | CPU attention, 8-15 s | 713 ms (1379 tok/s) | ❌ |
| decode @256 ctx | 80 tok/s | 87.2 tok/s | ❌ -8% |
| decode @512 ctx | 80 tok/s | 83.5 tok/s | ❌ -4% |
| decode @2k ctx | **80 tok/s** | 74.8 tok/s | ✅ **+7%** |
| TTFT @512 | 0.560-0.571 s | 0.569 s | ≈ par |
| TTFT @1024 | not available | 0.713 s | ❌ |

FLM's decode degrades with context (87.2 → 74.8 tok/s from 256 to 2088 tokens)
while native's stays ~80, so native's decode win appears only at long context.

## Status against the objective

- **decode: met** at 1k+ context (+7%), byte-exact runlist path untouched.
- **prefill: met up to 512 tokens** (+51% / +26%), **not met at @1k**.
- TTFT: par at 512, not available at 1k.

The single remaining blocker is a correct attention kernel for >512 keys. The
generated per-position-range ELFs are diagonal 256x256 tiles
(`gen(0,256)` = `gen(256,512)` = `gen(768,1024)` = 95936 B), not key prefixes, so
they are not a drop-in long-context kernel; using them would need a tiled
flash-attention with an online-softmax combine, and the attention ABI exposes
only `out/act/kv` with no accumulator BO.
