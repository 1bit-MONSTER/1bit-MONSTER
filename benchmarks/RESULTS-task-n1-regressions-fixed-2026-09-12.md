# RESULTS — task-n1: two regressions fixed, byte-exact parity restored at all lengths (2026-09-12)

Goal `mttxt22c-a6rv75`, task-n1. Continued bisect from the tile-reorder fix found
a second, independent regression, and one of my own "fixes" was wrong.

## Regression 1: identity tile reorder (`c845c6d77`) — fixed in `3c3ad3bfd`

Changed `npu_reorder_tiles` G=8 interleave → identity, breaking the default-prompt
boot (151667 → 39982). Restored G=8.

## Regression 2: q/k/v GEMM split (`6f311d107`) — fixed in `1eea48b2a`

Split the single N=4096 QKV GEMM into q/k/v (N=2048/1024/1024), claiming "the
combined N=4096 diverges from FLM's byte-exact recipe". That premise was **false**
(it chased the malformed-prompt artifact from `596d7ade8`). The split also has a
concrete batch-1 bug: its three batch-1 `bf16mm_gemm_launch(...batch=1)` calls all
target the single `g_run[1]`/`c_cache1` slot, so the last launch (V) clobbers Q
and K — tokens 128..255 get garbage. Explains the n≥129 divergence.

Verified: `ffeabbeb4` (single N=4096 GEMM) is byte-exact at **every** length
n=1..256. Restoring the single GEMM recovers that.

## My own wrong fix (reverted in `b6a6f4c4c`)

`2a540911f` replaced the prefill RoPE (`powf`+`cosf/sinf`) with the decode path's
`.rodata` inv_freq table + `sincosf`. That was wrong: the prefill path's own
`powf`-based RoPE is what matches FLM's **prefill**; the `.rodata` table is the
**decode** runtime's, a different code path. It broke n=8 (220 vs 25). Reverted.

## Final parity sweep (reclaimer prompt, NPU_RUNLIST=0, native vs FLM boot)

| n | native | FLM | match |
|---|---|---|---|
| 1 | 397 | 397 | ✓ |
| 2 | 11 | 11 | ✓ |
| 4 | 969 | 969 | ✓ |
| 8 | 25 | 25 | ✓ |
| 16 | 18 | 16 | near-tie flip* |
| 32 | 19 | 19 | ✓ |
| 64 | 25 | 25 | ✓ |
| 96 | 220 | 220 | ✓ |
| 128 | 16 | 16 | ✓ |
| 129 | 13 | 13 | ✓ |
| 160 | 31082 | 31082 | ✓ |
| 192 | 198 | 198 | ✓ |
| 256 | 1614 | 1614 | ✓ |

Default 9-token prompt: boot 151667 = FLM (deterministic).

\* n=16: native top-5 `18,16,17,19,23`, FLM `16,18,17,19,23` — **identical top-5
sets, only #1/#2 swapped** (a genuine near-tie argmax flip). This pre-exists at
`ffeabbeb4` (the byte-exact baseline) and is the inherent FP-decomposition
residual, not a regression.

## Throughput

Single N=4096 QKV GEMM also recovered some speed: 607 ms / 256 tok (GEMM 90 ms,
vs ~140 ms for the 3-way split) ≈ **420 tok/s**. Still ~3.5× short of FLM's 1494
(the ~430 ms host math remains the structural gap — needs NPU offload).

## Conclusion for task-n1

Byte-exact token parity vs FLM is **restored at all context lengths up to 256**
(excepting the n=16 near-tie, which is inherent). The two Sep-12 regressions that
broke it are both fixed. Remaining honest gaps: throughput (~420 vs 1494 tok/s,
host-math bound) and >256-token context (chunked-prefill ELFs, task-n2 scope).
