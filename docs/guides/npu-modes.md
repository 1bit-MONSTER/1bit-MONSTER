# NPU engine modes and their artifact requirements

Environment switches read by `engine/npu/src/npu_engine_universal.cpp` (the
native XDNA 2 engine). Defaults are unchanged; every mode below is opt-in.

## `NPU_BF16=1` — bf16 activation / v8bfp16ebs8 weight contexts

Runs the attention GEMMs through the bf16 context family
(`n1_core_bf16_v1.py`, `Bf16Ctx`) instead of the INT8 `I8Ctx` path. The engine
constructs the artifact names straight from the dimensions:

```
final_bf16_<T>_K<K>_N<N>.xclbin
insts_bf16_<T>_K<K>_N<N>.txt          T ∈ {QKV, O, G, U, GU, D}
```

**These tiles are required and there is no fallback.** Selecting `NPU_BF16=1`
without them fails initialization (`FAIL bf16 QKV`, exit 1).

### Where they come from

They are build outputs, produced by:

```bash
bash engine/npu/generators/build_bf16_xclbins.sh
```

The script names its output exactly what the engine asks for (its header records
the `final_bf16_<PROJ>_K<K>_N<N>.xclbin` convention). The committed set under
`engine/npu/xclbins/` is tracked again since #2602; a fresh checkout carries it.

### Why this page exists

These artifacts were deleted once (#2598's sweep / `a81662ab8`) and nothing
noticed, because no check mapped the families the engine *constructs* to the
artifacts the commit *carries*. Two guards now exist:

* `engine/npu/tests/check_artifact_families.py` — discovers the construction
  sites and fails any family with **zero tracked members** unless it is declared
  `build_only` in `engine/npu/xclbins/ARTIFACT_FAMILIES.json`. Run by
  `Testing/run_all.sh`; its fixtures live in
  `Testing/artifact_family_coverage_selfcheck.sh`.
* a non-`.q4nx` model container is rejected with a diagnostic instead of
  SIGSEGVing in the JSON scan (issue #2601).

## Other switches (for reference)

| Variable | Effect |
|---|---|
| `NPU_FUSED=1` | fused GU→SiLU→D MoE path |
| `NPU_FUSED_SPLIT=1` | int8 two-launch split (P1 GU→SiLU→h2, P2 D) |
| `NPU_FUSED_I4=1` | int4 GU split |
| `NPU_MOE_ALLOW_BAD=1` | continue past a `[MoEgate]` refusal (diagnostics only) |
| `NPU_MOE_MIN_CORR=<x>` | move the split-path `[MoE L1 fused dbg]` threshold (default 0.95) |
| `NPU_FUSED_I4_ALLOW_BAD=1` | continue past a `[C2gate]` refusal (diagnostics only) |
| `NPU_XCLBIN_DIR=<dir>` | override the xclbin directory |
