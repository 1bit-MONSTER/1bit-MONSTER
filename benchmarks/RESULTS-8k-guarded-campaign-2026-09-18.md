# First guard-accepted 8k campaign: 0.6B is parity, 1.7B could not be measured cleanly — 2026-09-18

Goal `mu35shsg-i3hlyi`, criterion (c). Method: `benchmarks/c8k_guarded.sh` with **both**
contamination axes gated — foreign `accel0` holders and 1-min host load (`C8K_MAX_LOAD=18`,
calibrated in `RESULTS-8k-contention-source-2026-09-18.md`). 8192-token prompt, native `ng=1`
(the 8k decode's second forward needs `ctx=8194` against the baked `MAX_L=8192` window), FLM
via `npu_ab.sh --skip-native` interleaved in the same campaign.

## Qwen3-0.6B — 5 runs, 3 accepted

| run | status | native | FLM prefill | per-run ratio | load |
|---:|---|---:|---:|---:|---|
| 1 | ACCEPT | 0.493 ms/tok (2029 t/s) | 1979.13 t/s | **1.03x** | 9.01 → 9.13 |
| 2 | ACCEPT | 0.584 (1712) | 1762.03 | **0.97x** | 6.63 → 7.35 |
| 3 | SUSPECT (load 12.27 → 22.83) | 1.889 (529) | 1802.75 | 0.29x | — |
| 4 | DISCARD (pre-holder 230198) | 0.600 (1667) | 1786.54 | 0.93x | — |
| 5 | ACCEPT | 0.643 (1555) | 1618.45 | **0.96x** | 10.62 → 10.89 |

Medians over accepted runs: native **0.584 ms/prompt-token (1712 t/s)**, FLM **1762 t/s** →
**0.97x**. Per-run ratios 0.96–1.03x. **0.6B at 8k is parity** — neither the 0.62–0.71x
inversion of the 2026-09-16 record nor a native win.

Note the residual spread even with the load gate: native 0.493–0.643 (1.30x) and FLM
1618–1979 (1.22x) across three accepted low-load runs. **That is the practical noise floor of
this comparison, and it is larger than any of the differences being claimed at 8k** — an 8k
parity statement here is meaningful to roughly ±35%.

## Qwen3-1.7B — 5 runs, 0 accepted

| run | status | load |
|---:|---|---|
| 1 | SUSPECT (26.67 → 33.40) | `pf` at 2466% |
| 2 | SUSPECT (37.00 → 57.38) | `pf` at 2097% |
| 3 | SUSPECT (49.68 → 59.34) | `clang-23` |
| 4 | SUSPECT (32.97, no rise) | `pf` at 2944% |
| 5 | DISCARD (pre-holder `259872:/tmp/bh … Bonsai-27B-Q1_0.1bp 32`) | — |

No 1.7B number is reported. Two separate causes: other lanes compiling/hashing on the host
(load 26.7 → 59.3, the highest seen) and **another lane's NPU job on `accel0`** — run 5's
pre-holder is the Prism/Bonsai lane's binary. Both are correctly caught; neither may be
averaged into a result.

## What this says about the 8k clause

- 8k **prefill/TTFT parity** is now measured under a guard for 0.6B (0.97x median, 0.96–1.03x
  per run) and remains *unmeasured* under the guard for the other five models.
- The 2026-09-16 inversion and my single-run "1.01–1.24x" table are both outside this
  measurement's noise floor in the same direction — they are environment readings as much as
  engine readings.
- **The 8k decode row still does not exist** and cannot: `ctx=8194` is past the baked
  `MAX_L=8192` per-ctx ELF window.
- A six-model 8k campaign needs a window where the host is quiet (no `pf`, no parallel
  compiles) *and* no other lane is on `accel0`. Both were present throughout this attempt; the
  guard's value is that it says so per run instead of leaving it to be inferred later.

## Addendum: 1.7B measured with a wait-for-quiet protocol (2/3 accepted)

The first 1.7B attempt was rejected 5/5 because the load rose 18 → 48 *inside* the campaign.
In a contended environment the right protocol is "wait for the window, then take one sample",
so `c8k_guarded.sh` gained `C8K_WAIT_QUIET=1` (bounded by `C8K_WAIT_MAX`, default 600 s): before
each run it polls until both axes are clear, then measures immediately.

| run | status | native | FLM prefill | per-run ratio | load |
|---:|---|---:|---:|---:|---|
| 1 | ACCEPT | 0.853 ms/tok (1172 t/s) | 1274.76 t/s | 0.92x | 17.18 → 14.84 |
| 2 | SUSPECT (load rose 9.75 → 22.01) | 2.090 (478) | 1165.06 | 0.41x | — |
| 3 | ACCEPT | 0.717 (1395) | 1087.33 | **1.28x** | 16.98 → 17.55 |

Medians over accepted runs: native **0.785 ms/prompt-token (1274 t/s)**, FLM **1181 t/s** →
**1.08x**, with per-run ratios 0.92x and 1.28x. **1.7B at 8k is parity** — inside the same
±35% noise floor measured for 0.6B, and consistent with the earlier load-uncontrolled runs
(5.56–7.16 s native vs 5.65–6.51 s FLM).

No quiet-window protocol can fix a *host-side* fairness problem, only mitigate it: run 2 was
still caught by the load-rise check because a `pf` instance started mid-run. The device-side
holder was clear in all three runs; what moved the numbers was CPU.
