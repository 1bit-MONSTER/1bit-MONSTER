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
| Llama-3.1-8B | 20/20 | **20/20** | 20/20 | **20/20** | OK=20, MISMATCH=0 |

**Criterion (a)**: on the format-equalised 0.6B scoreboard the native arm **meets/beats
the FLM oracle's own self-check** (19 vs 17, and 19 vs 17 in the answer region), with the
residual content errors classified in
`RESULTS-content-error-classification-2026-09-15.md`.

**Criterion (b)**: the same scoreboard now covers **all six** supported models with the I1
same-bytes assertion passing on **every row of every arm** (OK=120, MISMATCH=0).

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

## Llama-3.1-8B: 20/20 — and the "hang" I first reported was my misconfiguration

First attempt reported a hang: row 1 completed and row 2 printed nothing for 600 s, and a
separate 240 s probe printed nothing at all. **Both were wrong, and the cause was mine.**
The probe's output was piped through `tail`, which shows nothing until EOF, and the run was
not hanging — it was running the *dense fallback* at 14.7 s/token after 211 s of dense
weight packing, so a 20-prompt harness run cannot finish inside any reasonable timeout.

The fallback is silent and is selected by a gate in `npu_engine_universal.cpp` (~:844):

```cpp
const bool dense_qwen3 = cfg.NV == 151936 && !cfg.has_moe && (...);
const bool runlist_eligible = dense_qwen3 || (!cfg.has_moe && getenv("NPU_LAYER_ELF_DIR"));
```

Llama's vocabulary is **128256**, so `dense_qwen3` is false, and with `NPU_LAYER_ELF_DIR`
unset `runlist_eligible` is false too — the engine then uses the 112-launch dense path
without saying so. The same command with an ELF dir completes on the runlist:

```
NPU_LAYER_ELF_DIR=/tmp/llama-elfs NPU_RUNLIST=1 ... npu_engine_llama model.q4nx 8 ids
[runlist] LAYER_XCLBIN pinned to the in-repo copy: engine/npu/xclbins/flm_models/Llama-3.1-8B-NPU2/layer.xclbin
=== Prefill 40 [runlist] ===    Prefill: 3638ms (91 ms/tok)
=== 77.5 ms/tok (13 tok/s) | tokens=8 ===
```

With that configuration the full 20-prompt row is **20/20 vs FLM 20/20**, answer-region
20/20, I1 OK=20 — i.e. the 2026-09-15 result (`e060acc4e`/`dac5417f4`) **reproduces**, and it
is now gate-carrying.

The defect that remains is real but different from a hang: **`NPU_RUNLIST=1` silently
demotes any non-Qwen3, non-MoE model to the slow dense path unless the caller also passes
`NPU_LAYER_ELF_DIR`** — even though the bridge can create and populate that directory
itself (`ensure_elf_gen_env`). Widening `runlist_eligible` to attempt the runlist for any
non-MoE model is a routing-semantics change affecting several families, which the goal's
block rules reserve for the user; it is recorded, not made.

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
# non-Qwen3 vocab: the runlist gate needs an ELF dir or it silently takes the dense path
ENGINE_ENV="NPU_RUNLIST=1 NPU_LAYER_ELF_DIR=/tmp/llama-elfs" \
  bash benchmarks/oracle_accuracy_model.sh \
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

- **The silent runlist demotion for non-Qwen3 vocabularies** (above): `NPU_RUNLIST=1` +
  no `NPU_LAYER_ELF_DIR` = the 112-launch dense path, ~190x slower per token for Llama, with
  no message. Recorded for a routing-semantics decision rather than changed here.
- 1.7B/4B/8B are 1–2 rows apart from a 20/20 oracle on a 20-prompt substring test; that is
  not a capability difference at this resolution, but it is also not parity, and the
  answer-region verdict (19/19/18) is the honest stricter reading.
- This document carried a wrong "Llama hangs" claim for about an hour and it reached two
  commits by the co-lane agent (832dffbee, LEVERS 6.5c) before it was retracted; both are
  being corrected. The lesson is the one this goal keeps re-learning: a probe whose output
  is piped through a pager-like filter cannot distinguish "slow" from "hung", and a
  misconfiguration that silently selects a 190x-slower path presents exactly as a hang.
