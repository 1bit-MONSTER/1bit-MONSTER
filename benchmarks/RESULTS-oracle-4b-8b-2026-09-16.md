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
