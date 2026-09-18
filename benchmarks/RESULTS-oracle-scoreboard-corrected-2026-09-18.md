# Corrected six-model oracle scoreboard, and the measurement-condition break behind the old one — 2026-09-18

Goal `mu35shsg-i3hlyi`, criteria (a) and (b). Box: strixhalo, box otherwise quiet.
This supersedes the 2026-09-16 rows in `RESULTS-oracle-1_7b-2026-09-16.md` and
`RESULTS-oracle-4b-8b-2026-09-16.md`, and re-confirms `RESULTS-oracle-vl-llama-2026-09-15.md`
for VL-4B.

Two independent things were wrong with the 2026-09-16 numbers, and both had to be fixed
before any of them meant anything.

## 1. The harness scored a truncated window, a stale prompt, or both

`origin/main` had already fixed the shared harness (#2433/#2434: `e46ff96c3`,
`a923b5ba3`, `e2714c4b9`, `1e0cb43aa`): the scored window was the **first 200 characters**
(90 in the 0.6B harness), which cuts every `<think>`-style answer off before the answer;
the ids/raw paths were **fixed `/tmp` names**, so a failed tokenize silently scored the
*previous* prompt; and the chat-template call raised for the Qwen3 templates that need
`user_system_prompt`. This branch now adopts those fixes (`git checkout origin/main --
benchmarks/oracle_accuracy_model.sh benchmarks/oracle_accuracy_0_6b.sh`) with this lane's
**I1 same-bytes assertion** re-applied on top (`flm_nids` + `i1` columns; MISMATCH voids
the row, UNPARSED is not a pass).

## 2. The runlist arm had been silently reading a replaced `layer.xclbin`

The runlist resolves its session xclbin from
`/home/bcloud/amd-oss/fastflowlm/src/xclbins/<model>/layer.xclbin` — a tree other lanes
actively rebuild. On **2026-09-18 08:19** the Qwen3-0.6B file there was replaced:

| file | size | md5 |
|---|---:|---|
| `engine/npu/xclbins/flm_models/Qwen3-0.6B-NPU2/layer.xclbin` (pinned, 09-10) | 339980 | `57431faab8593fadbffb5b9d5a9a0735` |
| `~/.local/flm-v0946/xclbins/Qwen3-0.6B-NPU2/layer.xclbin` | 339980 | `57431faab8593fadbffb5b9d5a9a0735` |
| `amd-oss/.../Qwen3-0.6B-NPU2/layer.xclbin` (now) | **401980** | **`fa9f8df2f2b5618a560fd5470104aade`** |

The per-context ELFs are built for the pinned file, so every runlist forward produced
**garbage**, with no error anywhere:

```
# auto-selected (amd-oss) layer.xclbin, all 20 oracle rows:
FLM oracle self-check : 18/20
native arm            :  0/20          <- I1 OK=20 MISMATCH=0, same prompt bytes
native text           : "omedicalomedical年人omedical年人农业大学…"   (looping)

# same engine, same prompt, LAYER_XCLBIN pinned to the repo copy:
native                : 151667 198 32313 11 279 1196 374 10161 911 279 6722 315   (coherent)
```

This is a measurement-condition break, not an engine regression: three engine binaries
built at 09:00, 14:48 and 20:07 on 2026-09-16 all reproduce the garbage under the new
xclbin and all produce the coherent stream under the pinned one. **Any runlist number
taken on this box after 2026-09-18 08:19 is void unless it pinned `LAYER_XCLBIN`.**

Fix: both oracle harnesses now default `LAYER_XCLBIN` to
`engine/npu/xclbins/flm_models/<Model>/layer.xclbin` when it exists (all six supported
models have one) and print `[layer.xclbin] pinned: …`; set `LAYER_XCLBIN` to override.

## The corrected scoreboard

Same 20-prompt set (`benchmarks/prompts/qwen3_0_6b_oracle_set.txt`), greedy,
`NPU_RUNLIST=1`, ntok=128, pinned `layer.xclbin`, `ORACLE_TEXT_CHARS=4000`.
"answer-region" is the text after the last `</think>` — the stricter verdict.

| model | FLM oracle self-check | native | FLM answer-region | native answer-region | I1 |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | 17/20 | **19/20** | 17/20 | **19/20** | OK=20, MISMATCH=0 |
| Qwen3-1.7B | 20/20 | **20/20** | 20/20 | 19/20 | OK=20, MISMATCH=0 |
| Qwen3-4B | 20/20 | 19/20 | 20/20 | 19/20 | OK=20, MISMATCH=0 |
| Qwen3-8B | 20/20 | 19/20 | 20/20 | 18/20 | OK=20, MISMATCH=0 |
| Qwen3-VL-4B | 20/20 | **20/20** | 20/20 | **20/20** | OK=20, MISMATCH=0 |
| Llama-3.1-8B | — | *not measured* | — | — | — |

**Criterion (a)**: on the format-equalised 0.6B scoreboard the native arm **meets/beats
the FLM oracle's own self-check** (19 vs 17, and 19 vs 17 in the answer region), with the
residual content errors classified in
`RESULTS-content-error-classification-2026-09-15.md`.

**Criterion (b)**: the same scoreboard now covers five of the six models with the I1
same-bytes assertion passing on **every row of every arm** (OK=100, MISMATCH=0).

Every native miss, individually:

| model | row | verdict | note |
|---|---|---|---|
| 0.6B | `The opposite of day is` (want `night`) | native N/N; FLM N | both arms miss |
| 1.7B | `2 + 2 =` (want `4`) | native whole-text Y, answer-region N | answer stated inside the reasoning block |
| 4B | `How many days are in a week?` (want digit `7`) | native N/N | native says the word, the set wants the digit |
| 8B | `How many days are in a week?` | native N/N | as 4B |
| 8B | `The capital of Japan is` (want `tokyo`) | native whole-text Y, answer-region N | answer inside the reasoning block |

So the residual gap is **one row of genuine disagreement (0.6B) plus three
format/verbosity rows**, and the 4B/8B "below the oracle" reading recorded on 2026-09-16
is withdrawn — it was the 200-char window. This is consistent with `origin/main`'s
independent re-measurements (`d621482f2` 4B 20/20 vs 20/20, `22d897c63` 8B 18/20 vs 20/20
at 128 tokens with their harness).

## Llama-3.1-8B is NOT measured, and hangs

The corrected run completed row 1 (`The capital of France is` → `Paris`, I1 OK, 40 ids)
and then **hung on row 2** (`The capital of Japan is`, 40 ids): no output in 600 s, and
again in the harness with its 900 s per-invocation timeout. The old 20/20 in
`RESULTS-oracle-vl-llama-2026-09-15.md` is therefore **unverified under the pinned
configuration** and is not carried forward as a (b) row. This is a reproducible hang, not
a slow run: the same prompt with the same ids printed nothing at all.

## Measurement-contract note: accuracy runs must be SERIAL

Two NPU engines *do* run concurrently at full per-stream speed (2.05–3.00x aggregate,
`RESULTS-dual-engine-and-family-fixes-2026-09-16.md`), and `hwctx_limit` is 16. That is a
throughput property. It does not extend to accuracy runs: with two harnesses plus a third
engine on `accel0`, per-prompt wall time went from ~25 s to ≥900 s (TDR teardown → the slow
fallback path), so the scoreboard above was taken **one harness at a time** with the device
otherwise quiet.

## Commands (one model; `T` ∈ `0_6b,1_7b,4b,8b`)

```
cd ~/1bit-MONSTER-goal
ENGINE_ENV="NPU_RUNLIST=1" bash benchmarks/oracle_accuracy_model.sh \
  Qwen3-<X>-NPU2 npu_engine_qwen3_<T> qwen3:<tag> 128 \
  benchmarks/prompts/qwen3_0_6b_oracle_set.txt /tmp/oracle_acc_<T>.tsv

# VL-4B / Llama
ENGINE_ENV="NPU_RUNLIST=1" bash benchmarks/oracle_accuracy_model.sh \
  Qwen3-VL-4B-Instruct-NPU2 npu_engine_qwen3_vl_4b qwen3vl-it:4b 128 \
  benchmarks/prompts/qwen3_0_6b_oracle_set.txt /tmp/oracle_acc_vl4b.tsv
ENGINE_ENV="NPU_RUNLIST=1" bash benchmarks/oracle_accuracy_model.sh \
  Llama-3.1-8B-NPU2 npu_engine_llama llama3.1:8b 128 \
  benchmarks/prompts/qwen3_0_6b_oracle_set.txt /tmp/oracle_acc_llama.tsv
```

Per-row data: `/tmp/oracle_acc_{06,q17,q4,q8,vl4b}_fixed.tsv` on strixhalo at time of
writing; the harness prints the summary table for each run.

## Cross-reference

The same two defects are recorded independently on this branch by the co-lane agent:
`832dffbee` (CORRECTION amendment on `RESULTS-oracle-4b-8b-2026-09-16.md`, citing this
measurement) and `639444644` (LEVERS 6.5, the three measurement conditions) and
`f15cc984d` (LEVERS 6.4.1). The numbers agree: this document is the standalone
scoreboard; those two are the in-place supersession banners and the register entry.

Additional evidence for the `LAYER_XCLBIN` mechanism, not in either: **three engines built
at different times on 2026-09-16 — 09:00 (`wt/family-head-block`), 14:48
(`wt/acc-scorer-trunc`) and 20:07 (this worktree) — all reproduce the garbage under the
replaced xclbin and all produce the coherent stream under the pinned one.** So the break
is purely the xclbin the engine resolves, not any build of this branch.

## What is still open

- **Llama-3.1-8B: the runlist hang.** Reproducible on row 2 of the set. Until it is
  diagnosed, Llama has no (b) row, and any speed number for it is unaccompanied by a
  correctness gate.
- 1.7B/4B/8B are 1–2 rows apart from a 20/20 oracle on a 20-prompt substring test; that is
  not a capability difference at this resolution, but it is also not parity, and the
  answer-region verdict (19/19/18) is the honest stricter reading.
