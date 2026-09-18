# The default path now uses the fast prefill — a 23–41x TTFT change (2026-09-15)

Every round since the first has measured the engine's best configuration with
explicit flags, and every round has noted the same thing in its closing section:
**the default is the slow-prefill path.** A plain
`npu_engine_<model> model.q4nx <n> <ids>` ran the runlist path, whose prefill is
one whole-layer forward *per prompt token* — so its TTFT was 23x FastFlowLM's at
1k and 37x at 4k, while its decode was fine. That is the difference between
"native can beat FLM" and "native beats FLM".

This round makes the choice automatic, using the measured break-even rather than
a guess.

## The rule, derived from measurements

Both sides measured on Qwen3-0.6B, 2026-09-15:

| | bf16 prefill | decode |
|---|---:|---:|
| runlist path | 13.46 ms / prompt-token | 11.4 ms / token |
| unified path | 0.53 ms / prompt-token | 13.8 ms / token |

The two halves of the engine are fast at different things, and the prefill
difference is the large one. Break-even:

```
npt*13.46 + ng*11.4   >   npt*0.53 + ng*13.8
npt*12.93             >   ng*2.4
ng                    <   5.4 * npt
```

So the combination wins for every prompt above a handful of tokens, and the
**only** case that favours the runlist path is a short prompt with a long
generation. The code uses `4x` rather than `5.4x`, which keeps the margin on the
side of the path whose decode is verified token-for-token against FLM.

`ng` is known — it is `argv[2]` — and `npt` is counted from the ids file, so the
choice is exact, not heuristic. An explicit `NPU_RUNLIST` or `NPU_UNIFIED` always
wins over it, and a non-dense model is never affected.

## Measured, same box, same tokens

| npt | path | prefill | decode |
|---:|---|---:|---:|
| 1024 | **new default** | **591 ms** | 13.3 ms/tok (75) |
| 1024 | runlist (old default) | 13353 ms | 11.4 ms/tok (88) |
| 1024 | FLM's own runtime | 744 ms | 13.4 ms/tok (75) |
| 4095 | **new default** | **1700 ms** | **19.1 ms/tok (52)** |
| 4095 | runlist (old default) | 69109 ms | 17.6 ms/tok (57) |
| 4095 | FLM's own runtime | 1891 ms | 20.3 ms/tok (49) |

Out of the box the engine now **beats FLM on prefill and TTFT at both lengths and
on decode at 4k**, which is not something the default could claim before. TTFT
falls 13353 → 591 ms (22.6x) at 1k and 69109 → 1700 ms (40.6x) at 4k.

The cost is real and is the reason for the `4x` bound: unified decode is **13.3
against the runlist's 11.4 ms/token, ~17% slower**. A short prompt with a long
output keeps the runlist path.

## A silent truncation found on the way

Selecting the unified path exposed a trap that was already reachable by hand. The
bf16 prefill caps the prompt at `NPU_PREFILL_MAX`, **defaulting to 256** ("the
historical 256"), and the unified session initialises inside that block. So:

```
NPU_UNIFIED=1 ... ids1024.txt     ->  Prefill 256 [bf16]   (a 1024-token prompt)
```

It returned the 256-token answer for a 1024-token prompt and said nothing. Since
round 4 made `NPU_UNIFIED=1` imply the bf16 prefill, that was reachable by anyone
who followed the documentation. The flag now carries the prompt length with it:

```
[unified] NPU_PREFILL_MAX defaulted to the prompt length (1024); the bf16 path
caps at 256 otherwise and would answer from a truncated context
```

## Failure behaviour

The automatic choice is conservative about failing:

- unified session init fails, **auto-selected** → falls back to the runlist path,
  which is where execution would have gone anyway;
- that fails too → clears the unified flag and continues, so the bf16 prefill
  completes and reports its boot token (`rc=0`) instead of dying at
  `runlist KV write L0 failed` after a full prefill. Verified with a nonexistent
  `NPU_LAYER_ELF_DIR`: `rc=0`, `Prefill 1024 [bf16]`, `boot=25`;
- **explicit** `NPU_UNIFIED=1` is still a hard error, because the user asked for
  that specific combination.

## Verified

Default invocation, no environment: `[auto]` selects bf16+unified, `Prefill 1024
[bf16]`, tokens `25 220 16 13` (the unified trajectory from
`RESULTS-token-disagreements-are-ties`), 591 ms. Short prompt with a long output
(npt=10, ng=100) keeps the runlist path. `NPU_RUNLIST=1` keeps the runlist path.
`NPU_UNIFIED=1` now runs the full 1024-token prefill. A non-dense model (Llama)
is untouched. All on the rebuilt binary.

## What this does not fix

- **Context > 4095** is still capped, and the 8192 capture is still not wireable
  (`RESULTS-ctx8192-blocked-2026-09-15.md`).
- The unified decode's ~17% cost is real; the rule keeps it off the
  short-prompt/long-output case but a user who wants the runlist decode
  unconditionally should set `NPU_RUNLIST=1`.
- The families outside the dense-Qwen3 set do not take this path at all.
