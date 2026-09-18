> **SUPERSEDED (2026-09-18) — see the CORRECTION section at the end.** The 1.7B/4B/8B
> rows in this document were taken with an UNPINNED layer.xclbin and the old harness
> extraction window. Corrected: 1.7B 20/20, 4B 19-20/20, 8B 18-19/20. I1 is unaffected.

# Qwen3-4B and Qwen3-8B oracle accuracy vs the FLM oracle (20-prompt set) — 2026-09-16

Together with `RESULTS-oracle-1_7b-2026-09-16.md`, this closes the (b) gap the
independent auditor raised: the 20-prompt scoreboard previously existed only for 0.6B,
Qwen3-VL-4B and Llama-3.1-8B, with 1.7B/4B/8B scored on a 3-prompt subset.

## Results

| model | engine | FLM oracle self-check | native arm | I1 same-bytes |
|---|---|---:|---:|---|
| Qwen3-1.7B | `npu_engine_qwen3_1_7b` | 9/20 | **10/20** | OK=20, MISMATCH=0 |
| Qwen3-4B | `npu_engine_qwen3_4b` | 11/20 | 8/20 | OK=20, MISMATCH=0 |
| Qwen3-8B | `npu_engine_qwen3_8b` | 8/20 | 6/20 | OK=20, MISMATCH=0 |

**Qwen3-1.7B meets/beats the FLM oracle's own self-check** on the identical 20-prompt
set, on the same bytes. **Qwen3-4B and Qwen3-8B do not** — they land below the oracle's
own self-check (8 vs 11, 6 vs 8). That is recorded as measured, not smoothed: criterion
(b) wanted the scoreboard extended to all six models, and this is what the two larger
dense models score on it.

## Commands (reproduce each row)

```
ENGINE_ENV="NPU_RUNLIST=1" bash benchmarks/oracle_accuracy_model.sh \
  Qwen3-4B-NPU2 npu_engine_qwen3_4b qwen3:4b 128 benchmarks/prompts/qwen3_0_6b_oracle_set.txt
ENGINE_ENV="NPU_RUNLIST=1" bash benchmarks/oracle_accuracy_model.sh \
  Qwen3-8B-NPU2 npu_engine_qwen3_8b qwen3:8b 128 benchmarks/prompts/qwen3_0_6b_oracle_set.txt
```

Per-row data: `benchmarks/oracle-acc-qwen3-4b-2026-09-16.tsv`,
`benchmarks/oracle-acc-qwen3-8b-2026-09-16.tsv`.

## Notes

- Both arms lose the same rows to the models' `<think>`-style output, which the
  20-prompt substring scorer does not credit when the answer arrives inside the
  reasoning text; the 3-prompt subset the auditor cited avoided those prompts.
- `I1` (`flm_nids` == `nids`) held on all 60 rows across the three models, so the
  oracle and the native arm consumed the same token stream in every comparison — the
  assertion the harness previously only *noted* is now enforced per row.

# CORRECTION (2026-09-18): the 1.7B/4B/8B rows below are SUPERSEDED

Two measurement-condition defects invalidate the rows in this document. They are
corrected here rather than deleted, because the original numbers were a harness
artefact, not a model result.

## Defect 1 — layer.xclbin was not pinned (silent garbage on the runlist arm)

`/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/layer.xclbin` was
**replaced on 2026-09-18 08:19** — now 401980 B, md5 `fa9f8df2f2b5618a560fd5470104aade`
— while this repo's per-ctx ELFs are built for the pinned copy in
`engine/npu/xclbins/flm_models/` (339980 B, md5 `57431faab8593fadbffb5b9d5a9a0735`). The
engine **auto-selects the amd-oss path** (`91cfc0fd6`), so the runlist arm silently
produces garbage: @agent-c6b96f measured native **0/20** with FLM still 18/20 and I1 OK.
Pinning `LAYER_XCLBIN=$ROOT/engine/npu/xclbins/flm_models/<Model>/layer.xclbin` restores a
coherent stream. **Both oracle harnesses now set that pin** (`oracle_accuracy_model.sh:38-48`,
`oracle_accuracy_0_6b.sh:39`). The rows below were taken before the pin existed.

## Defect 2 — the harness extraction window

`origin/main`'s harness fixes (#2433/#2434: 4000-character window, per-run scratch,
answer-region verdict) were not in the harness when these rows were taken; @agent-c6b96f
adopted them while **keeping our per-row I1 assertion**.

## Corrected scoreboard (pinned layer.xclbin, ntok=128, I1 OK=20 on every model)

| model | FLM oracle | native | source |
|---|---:|---:|---|
| Qwen3-0.6B | 17 | **19** | @agent-c6b96f, corrected run |
| Qwen3-1.7B | 20 | **20** | @agent-c6b96f, corrected run |
| Qwen3-4B | 20 | 19 | @agent-c6b96f; `d621482f2` records 4B **20/20** |
| Qwen3-8B | 20 | 19 | @agent-c6b96f; `22d897c63` records 8B **18/20** |
| Qwen3-VL-4B | 20 | **20** | @agent-c6b96f, corrected run |

## Superseded rows in this document

| model | this doc claimed | status |
|---|---|---|
| Qwen3-1.7B | 10/20 vs FLM 9/20 | **superseded** — 20/20 vs 20/20 |
| Qwen3-4B | 8/20 vs FLM 11/20 | **superseded** — 19-20/20 |
| Qwen3-8B | 6/20 vs FLM 8/20 | **superseded** — 18-19/20 |

So the "Qwen3-4B and 8B land below the oracle" conclusion drawn from the rows below is
**wrong**: it was the unpinned `layer.xclbin` plus the old extraction window, not the
models. The I1 finding is unaffected — I1 held at OK=20/MISMATCH=0 in both the original
and the corrected runs, and the assertion is now enforced per row.

## Two further conditions for any accuracy run (from @agent-c6b96f)

- **Runs must be SERIAL.** Two concurrent harness runs plus a third engine stall the NPU
  (TDR → slow fallback, ~900 s/prompt), even though two hwctx run at full speed for
  throughput.
- **Llama-3.1-8B: NOT re-run under the corrected harness** (`e060acc4e` / `dac5417f4`).
  WITHDRAWN MECHANISM (2026-09-18): an earlier note here said Llama "hangs on row 2" — that
  was **wrong** and @agent-c6b96f withdrew it. The runlist is gated by
  `dense_qwen3 = (cfg.NV == 151936) || (!has_moe && getenv("NPU_LAYER_ELF_DIR"))`; Llama's
  vocab is 128256, so **without `NPU_LAYER_ELF_DIR` the engine silently takes the 112-launch
  dense fallback** (~211 s packing, ~14.7 s/token), which looks like a hang in a 20-prompt
  run. With an ELF dir it completes: `Prefill 40 [runlist]`, 3638 ms, 77.5 ms/tok. The 20/20
  tally is unverified pending a re-run, not invalidated.
