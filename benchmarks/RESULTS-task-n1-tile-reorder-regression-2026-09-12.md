# RESULTS — task-n1: tile-reorder regression fixed, byte-exact parity restored (2026-09-12)

Goal `mttxt22c-a6rv75`, task-n1. Bisected the native bf16 prefill's token-parity
break to a single commit and reverted the offending change.

## Bisect result

| commit | default 9-tok prompt boot | verdict |
|---|---|---|
| `ffeabbeb4` (Sep-11, before qkv split) | **151667** = FLM | byte-exact ✓ |
| `6f311d107` (q/k/v GEMM split) | **151667** = FLM | byte-exact ✓ |
| `c845c6d77` (identity tile reorder) | **39982** ≠ FLM | ❌ regression |

`c845c6d77` changed `npu_reorder_tiles` (model.c) from the G=8 interleave
`i = G*(o/G) + (o/2)%S + S*(o%2)` to **identity** (`i = o`). That commit was
chasing a *malformed-prompt* boot number (the OOV token-248044 artifact, only
discovered later in `596d7ade8`) and "fixed" a non-bug. The G=8 interleave is
the correct FLM runtime tile layout.

## Fix (committed `3c3ad3bfd`)

Restored the G=8 interleave in `npu_reorder_tiles`. The GU dequant mode=1/2
part of `c845c6d77` is retained — it is correct (byte-exact parity holds with it).

- Default 9-token prompt: **boot 151667 = FLM** (deterministic, 3/3 runs).
- Timing unchanged: ~649 ms / 256 tok.

## Re-characterized parity after the fix (reclaimer prompt, NPU_RUNLIST=0)

| n (tokens) | native boot | FLM boot | match |
|---|---|---|---|
| 1 | 397 | 397 | ✓ |
| 2 | 11 | 11 | ✓ |
| 4 | 969 | 969 | ✓ |
| 8 | 220 | 25 | near-tie flip |
| 16 | 16 | 16 | ✓ |
| 32 | 19 | 19 | ✓ |
| 64 | 25 | 25 | ✓ |
| 96 | 220 | 220 | ✓ |
| 128 | 16 | 16 | ✓ |
| 192 | 220 | 198 | flip |
| 256 | 16 | 1614 | flip |

Top-5 at the flips show **overlapping token sets**, not disjoint answers:

- n=8: native `220,25,17,18,16` vs FLM `25,56,220,18,711` (share 220/25/18 — a
  classic #1/#2 argmax near-tie).
- n=256: native `16,220,17,18,15` vs FLM `1614,14324,16,18,678` (share 16/18).

## Key correction to prior findings

The prior session's `f20f0f87b` "attention is a real divergence — NPU 91364 /
CPU 79362 / FLM 62865" (three *disjoint* answers) was itself a **symptom of the
broken tile reorder**. With correct weights, NPU attention and CPU attention now
**agree** (both give the same boot token at n=256), and native-vs-FLM is in
near-tie argmax-flip territory at long context — a "different FP decomposition"
regime, not a disjoint divergence.

## Remaining gap (separate, pre-existing)

At n ≥ ~192 the native top-5 grows "sticky" on the first token's neighbors
(`16,220,17,18,15` recurs at n=256), while FLM's prediction evolves. This is the
long-context attention issue (position-shifted ELF / chunked prefill, task-n2
scope), not the tile layout. It is now cleanly isolated from the regression.
