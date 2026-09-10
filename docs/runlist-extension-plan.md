# Extension plan — runlist whole-layer path to Qwen3-1.7B / 4B (task-5)

Goal `mtunui03-ekarlt`. The 0.6B path is done (1 `xrt::runlist` submit/token,
94 tok/s, byte-identical). Extending to 1.7B/4B = generalizing the 0.6B-specific
packing + per-ctx ELF generation.

## Model dims (config.json; all NV=151936, HD=128, NKV=8, max_pos=40960)

| model | H | NC | NH | IM |
|---|---|---|---|---|
| 0.6B | 1024 | 28 | 16 | 3072 |
| 1.7B | 2048 | 28 | 16 | 6144 |
| 4B   | 2560 | 36 | 32 | 9728 |

## Packing geometry — VERIFIED byte-exact vs runtime captures (1.7B)

- **Reorder group G = K/128** (K = contraction dim): q/k/v/up/gate → H/128,
  o → NH·HD/128, down → IM/128. (0.6B: 8/8/8/16/8/8/24; 1.7B: 16/16/16/16/16/16/48.)
- **gate/up alternating chunk CH = H/16** (= 8·G_gateup). (0.6B: 64; 1.7B: 128.)
- **Tile offsets = cumulative tile counts** (q, k, v, o, up/gate, down).
- **lm_head reorder G = H/128** (0.6B: 8, 1.7B: 16). lm_head tiles = shape[0]
  (0.6B: 18992, 1.7B: 37984, 4B: 47480).
- **Norms**: the per-layer norm tensors (input/post_attention_layernorm,
  q/k_norm) carry their OWN data_offsets in the metadata (verified equal to the
  old 0.6B hardcodes). i5 = ILN(H) + PALN(H); i6 = [1.0×64][0×64][q_norm][k_norm]
  (kn/qn are HD=128 bf16). No hardcoded offsets / pipeline order needed.

## Code changes LANDED this session (all on branch `goal/runlist-decode-wire`)

- `npu-infer/src/model.c`: `npu_pack_layer_bo` (G=K/128, CH=H/16, cumulative
  offsets, dynamic total), `npu_pack_lmhead_bo` (G=H/128), new
  `npu_layer_bo_bytes()`. Removed `NPU_LAYER_TILES/BO_BYTES`.
- `npu-infer/src/runtime_layer.cpp`: dynamic weight BO + lm_head BO sizes;
  `build_norm_bos` + final-norm read from metadata (deleted `META_L0_ILN_OFF`,
  `META_FINAL_NORM_OFF`, `NORM_BLOCK_BYTES`, `LMHEAD_TILES/BO_BYTES`,
  `NORM_PIPELINE_ORDER`, `NORM_POS_OF_LAYER`).
- 0.6B regression re-verified: engine 94 tok/s, greedy "Paris…", byte-identical.

## Remaining (next session)

1. **1.7B lm_head ELF** — `gen_layer_elfs` only emits layer ELFs; the lm_head
   ELF (`elf_0002_lmhead.bin`) was captured via an `xrt::elf` ctor interposer
   (Round 32). Need to capture/generate the 1.7B lm_head ELF (37984 tiles).
2. **Per-model config + gate** — add 1.7B/4B `ModelConfig`s in the bridge and
   widen the `NPU_RUNLIST` gate in `npu_engine_universal.cpp` (now `NC==28 &&
   H==1024 && NV==151936`); set `LAYER_XCLBIN` + `NPU_LAYER_ELF_DIR` per model
   (`Qwen3-1.7B-NPU2/layer.xclbin` and `Qwen3-4B-NPU2/layer.xclbin` exist).
3. **Full per-ctx ELFs** — generate ctx 1..N for 1.7B (and 4B) via `gen_layer_elfs`.
4. **Verify** — decode 1.7B via `NPU_RUNLIST=1`; reference greedy-next for the
   runtime is token 25 (ctx-1, token 1000, from `run_qwen3_npu` capture); expect
   byte-identical to the runtime (weight BO 30 MB, lm_head BO 188 MB, norms).

## Captures (ground truth, /tmp/cap-1p7b)

- 28 × layer weight BO (31457280 B = 6144 tiles × 5120) — `bo_to_0057_*.bin` is layer 0.
- lm_head BO `bo_to_0141_197132288.bin` (188 MiB).
- 28 × KV BO (32 MiB), act/norm BOs (1 MiB).
- runtime greedy-next: token 1000 → **25** (ctx=1, 1.7B).
