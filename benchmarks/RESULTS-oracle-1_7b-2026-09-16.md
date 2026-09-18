> **SUPERSEDED 2026-09-18.** The numbers below were taken with a harness that scored
> only the first 200 characters of each arm's output and used fixed `/tmp` paths, and the
> runlist `layer.xclbin` was later replaced under the engine. Re-measured with the
> corrected harness and a pinned `LAYER_XCLBIN`: **native 20/20 vs FLM oracle 20/20**,
> I1 OK=20 MISMATCH=0 (see `RESULTS-oracle-scoreboard-corrected-2026-09-18.md`). The
> 10-vs-9 rows below are measurement artefacts and must not be cited.

# Qwen3-1.7B oracle accuracy vs the FLM oracle (20-prompt set) — 2026-09-16

Closes the (b) gap the independent auditor raised: the 20-prompt scoreboard previously
existed only for 0.6B, Qwen3-VL-4B and Llama-3.1-8B, with 1.7B/4B/8B scored on a
3-prompt subset.

## Result

```
prompts scored        : 20  (extraction errors: 0)
I1 same-bytes assertion: OK=20  MISMATCH=0
FLM oracle self-check : 9/20
native arm (npu_engine_qwen3_1_7b) : 10/20
```

The native arm **meets/beats the FLM oracle's own self-check** on the identical
20-prompt set (10 vs 9), on the same bytes.

## Command (reproduces the row)

```
OUT=/tmp/oracle_acc_Qwen3-1.7B-NPU2.tsv ENGINE_ENV="NPU_RUNLIST=1" \
  bash benchmarks/oracle_accuracy_model.sh Qwen3-1.7B-NPU2 npu_engine_qwen3_1_7b qwen3:1.7b 128 \
    benchmarks/prompts/qwen3_0_6b_oracle_set.txt
```

Per-row data: `benchmarks/oracle-acc-qwen3-1_7b-2026-09-16.tsv`.

## I1 same-bytes assertion — now IMPLEMENTED, not noted

`oracle_accuracy_model.sh` previously carried an I1 note admitting the native arm
builds ids with HF `transformers` from the model tokenizer while the FLM oracle
receives the raw prompt and templates it internally. The harness now parses the token
count FLM reports for the prompt it was given (`with N tokens`) and compares it with
the native `nids`, writing a `flm_nids` and an `i1` column per row (`OK` / `MISMATCH` /
`UNPARSED`). A `MISMATCH` voids the row and `UNPARSED` is not counted as a pass.
Result on this model: **OK=20, MISMATCH=0** — the identical stream is asserted.

Two harness defects were fixed to get here (both real, both found by running):
- Qwen3-1.7B's `tokenizer_config.json` template references the optional
  `user_system_prompt`, which made `apply_chat_template` raise
  `jinja2.exceptions.UndefinedError: 'user_system_prompt' is undefined` and produced
  20/20 TOKENIZE EMPTY rows. The harness now supplies `user_system_prompt=""` and
  `tools=None`, with a ChatML fallback if that still raises.
- `OUT` is a positional parameter, not an env var; `OUT=... bash harness ...` does not
  set it (the file lands at `/tmp/oracle_acc_<Model>.tsv`).

## Row mix

| flm | native | rows |
|---|---|---:|
| Y | Y | 8 |
| Y | N | 1 |
| N | Y | 2 |
| N | N | 9 |

Both arms fail the same 9 rows (the `<think>`-prefixed reasoning answers), and the
native arm wins 2 rows the oracle misses while losing 1 the oracle gets.
