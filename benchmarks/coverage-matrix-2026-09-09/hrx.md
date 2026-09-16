# HRX lane — results (gfx1151 iGPU via IREE/Loom)

Two runtimes, same Radeon 8060S: shipped **bundles** (b59/b66, `HRX0`, standard
ggml-hrx) and the **hrx-v2 fork** (`HRX20`, Q4NX-capable). Full logs in hrx.log.

## Fixes applied to the fork (commit df58715ca, patch in hrx2-fork-fixes.patch)
1. MoE `nselected` JIT range 64→1024 in the Q4NX MUL_MAT_ID kernels
   (`mul_mat_q4nx_fused_f32` / `mul_mat_q4nx_fused_tbl_tiled` /
   `mul_mat_f32_f32_ggml_tbl_tiled`) and 128→1024 in `get_rows_moe_weights_f32`
   — unblocks Qwen3-Coder-30B (nselected=128) and Qwen3-Next-80B (nselected=512).
2. `pointwise_f32` `src0_row_stride`/`src1_row_stride`/`src1_ncols` 65536→1048576
   — unblocks GLM-4.7-Flash (large-stride ADD, src0_row_stride=131072).
3. `flash_attn_fa0` fusion guarded to query-heads H≤16 (its validated config);
   H=28/H=32 models fall back to non-fused attention — fixes MiniCPM4-8B
   degenerate `<h4/>` and qwen25-7b `A. Paris`.

## Standard GGUFs — bundle b59, `--device HRX0`
| Model | Result |
|---|---|
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M | ✅ " Paris. The capital of Belgium…" |
| Qwen3-0.6B / 1.7B / 4B Q4_K_M | ❌ GET_ROWS fail-closed (q6_K embedding) |
| Qwen3.6-35B-A3B-Q8_0 | ❌ abort (Q8_0 embedding) |
| zaya-Q4_K | ❌ `unknown model architecture: 'zaya'` |

## Q4NX-converted GGUFs — hrx-v2 fork (fixed), `--device HRX20`
| Model | Result |
|---|---|
| qwen3-06b-q4nx | ✅ " Paris. What is the capital of France in 20…" |
| qwen25-05b-q4nx | ✅ " Paris. It is the largest city in the European Union…" |
| qwen25-3b-q4nx | ✅ " Paris. The capital of Germany is Berlin…" |
| qwen25-7b-q4nx | ✅ " Paris. The capital of the United States is Washington D.C…" |
| minicpm5-1b-q4nx | ✅ " Paris, the capital of the Île-de-…" (weak 1B) |
| minicpm4-8b-q4nx | ✅ " Paris." (was `<h4/>` before FA0 fix) |
| glm47-flash-q4nx | ✅ " Paris. It is located in the north-central part…" (was JIT fail) |
| qwen3coder-30b-q4nx | ✅ " Paris. The population of France is 67 million." (was JIT fail) |
| qwen3next-80b-q4nx | ✅ " Paris. The capital of Italy is Rome…" (was JIT fail) |

## zaya GGUF (fork has zaya arch)
| Model | Result |
|---|---|
| zaya-Q4_K.gguf | ❌ `check_tensor_dims: token_embd.weight expected 2048×262147, got 2048×262272` |
| zaya1-8b-ft-q4nx.gguf | ❌ failed to load model |
(zaya is covered by the native-NPU lane instead.)
