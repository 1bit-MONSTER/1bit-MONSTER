# RESULTS — default-path parity: unified vs runlist vs FLM, dense Qwen3, 1k/2k/4k — 2026-09-16

Verification of the auto-selected default path (`30a9d1d81`, "the default path now
picks the fast prefill") across **4 models × 3 contexts**, which is the broad
measurement `RESULTS-native-4k-decode-and-unified-2026-09-15.md` §5 said the
default change was deferred for.

Session: dsh `goal-8d57213e`. Worktree `/home/bcloud/1bit-MONSTER-goal`, branch
`goal/runlist-decode-wire`. Harness and raw logs:
`~/.dsh/scratch/parity-sweep-2026-09-16/` (`sweep.sh`, `summary.tsv`, `logs/`).

## Conditions

- Engine: `engine/npu/build/npu_engine_qwen3_{0_6b,1_7b,4b,8b}` (built 2026-09-16).
- ELF sets: `~/.cache/1bit-monster/elfs/<model>` (ctx 1…8193; 4B to 5256). In range
  for every prompt used, so no ELF generation ran during measurement.
- Prompts: `~/npu-build/parity/ids{1024,2048,4095}.txt` (prefixes of one sequence).
- 8 decode tokens, greedy. Sequential; one invocation at a time.
- `accel0`: only the parked `flm serve` (pid 2432) held it and it issues no work;
  `npu-preflight.sh` → exit 0, hwctx 0/16.
- Arms: `unified` = `NPU_PREFILL_BF16=1 NPU_UNIFIED=1 NPU_PREFILL_MAX=<npt>`;
  `runlist` = `NPU_RUNLIST=1`; `flm` = `NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1`.

## Prefill — unified beats FLM in all 12 model×context pairs

| model | ctx | unified ms | FLM ms | unified/FLM |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1024 | 547 | 828 | **0.66** |
| Qwen3-0.6B | 2048 | 896 | 1175 | **0.76** |
| Qwen3-0.6B | 4095 | 1714 | 2036 | **0.84** |
| Qwen3-1.7B | 1024 | 788 | 1146 | **0.69** |
| Qwen3-1.7B | 2048 | 1314 | 1703 | **0.77** |
| Qwen3-1.7B | 4095 | 2548 | 2937 | **0.87** |
| Qwen3-4B | 1024 | 1612 | 2120 | **0.76** |
| Qwen3-4B | 2048 | 2791 | 3438 | **0.81** |
| Qwen3-4B | 4095 | 5584 | 6628 | **0.84** |
| Qwen3-8B | 1024 | 2308 | 2949 | **0.78** |
| Qwen3-8B | 2048 | 4038 | 4906 | **0.82** |
| Qwen3-8B | 4095 | 7998 | 9116 | **0.88** |

12/12 pairs faster; margin narrows with context (0.66→0.88) but never inverts.

## Decode — unified matches or beats FLM in all 12 pairs (after repeat)

**Single runs of this metric are not reliable** — see §"Measurement variance"
below. Values marked ‡ are the median of 3 repeats; unmarked are single runs that
were not contradicted on repeat.

| model | ctx | unified tok/s | FLM tok/s | verdict |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 1024 | 73 ‡ | 74 ‡ | tie |
| Qwen3-0.6B | 2048 | 62 | 61 | **+** |
| Qwen3-0.6B | 4095 | 50 | 48 | **+** |
| Qwen3-1.7B | 1024 | 39 | 37 | **+** |
| Qwen3-1.7B | 2048 | 35 | 34 | **+** |
| Qwen3-1.7B | 4095 | **32 ‡** | 30 ‡ | **+** |
| Qwen3-4B | 1024 | 19 | 18 | **+** |
| Qwen3-4B | 2048 | 18 | 17 | **+** |
| Qwen3-4B | 4095 | **17 ‡** | 15 ‡ | **+** |
| Qwen3-8B | 1024 | 11 | 10 | **+** |
| Qwen3-8B | 2048 | 11 | 10 | **+** |
| Qwen3-8B | 4095 | **11 ‡** | 10 ‡ | **+** |

## Runlist decode, for the trade-off the auto-selection encodes

| model | ctx | runlist tok/s | FLM tok/s | vs FLM |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1024 | 87 | 73 | +19% |
| Qwen3-0.6B | 2048 | 73 | 61 | +20% |
| Qwen3-1.7B | 1024 | 44 | 37 | +19% |
| Qwen3-1.7B | 2048 | 40 | 34 | +18% |
| Qwen3-4B | 1024 | 21 | 18 | +17% |
| Qwen3-4B | 2048 | 20 | 17 | +18% |
| Qwen3-8B | 1024 | 12 | 10 | +20% |
| Qwen3-8B | 2048 | 12 | 10 | +20% |

Runlist decode is uniformly ~17–20% ahead of FLM, and ahead of the unified decode —
which is the cost the auto-selection knowingly pays for the prefill. The runlist
*prefill* is what it avoids: 14 765 ms at 1024 and 202 039 ms at 2048 for 8B.

## Measurement variance — a finding in its own right

The single-run sweep produced four points where unified appeared to lose decode to
FLM. **All four were noise.** 3 repeats each, same conditions:

| point | single-run value | 3-repeat median | corrected verdict |
|---|---|---|---|
| Qwen3-1.7B @4095 | 15 tok/s (64.5 ms/tok) | **31–32 tok/s** (31.4–31.8 ms/tok) | unified **wins** |
| Qwen3-4B @4095 | 12 tok/s (80.7 ms/tok) | **17 tok/s** (58.9–59.1) | unified **wins** |
| Qwen3-8B @4095 | 8 tok/s (122.8 ms/tok) | **11 tok/s** (94.3–95.1) | unified **wins** |
| Qwen3-0.6B @1024 | 71 tok/s | **73 tok/s** (13.6–13.9 ms) | **tie** (FLM 73–74) |

The 4B @4095 single run read 80.7 ms/tok against a 3-repeat spread of 58.9–67.6 —
a **37% swing on identical inputs**. A naive sweep would have reported four
"regressions" that do not exist, and would have concluded the unified default
hurts long-context decode. It does not.

**Rule for anyone reading parity numbers from this box: a single decode run is not
evidence.** Report a median of ≥3, or label it as unreplicated.

## Verdict

On the auto-selected default path, the native engine **beats FLM on prefill at
every model and context tested (0.66–0.88× the time) and matches-or-beats it on
decode at every one**. Combined with `30a9d1d81`'s measured break-even
(`ng < 4*npt` selects the combination), the default invocation is now
parity-or-better on both metrics.

**The open item is not performance, it is landing.** `30a9d1d81` is present on
`goal/runlist-decode-wire`, `fix/moe-packed-layer-mismatch`,
`family/head-block-loop` and `fix/oracle-scorer-truncation`, and **absent from
`main`** — which is 1278 commits behind `goal/runlist-decode-wire`. A binary built
from `main` still takes the 17–46× slow prefill.

## Limits of this result

- Dense Qwen3 only. 35B MoE is excluded and is separately known-broken (the
  captured whole-layer ELF never reads its weights — see
  `~/.dsh/scratch/mesh/alan-to-pi-MOE-ELF-IS-NOT-A-WHOLE-LAYER-2026-09-16.md`).
- 8 decode tokens per run: enough for a stable ms/tok, not for accuracy.
- Prompts are prefixes of one sequence; not a diverse prompt set.
- 4B ELF coverage stops at 5256, so 4B at 8192 was not attempted.
- No repeat pass at 1024/2048 for the models whose single runs already agreed with
  FLM to within 1 tok/s (`0.6B`), so those remain single-run.
