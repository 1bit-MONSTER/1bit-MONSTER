# RESULTS — runlist decode extension to Qwen3.6-35B-A3B (MoE), 2026-09-10

Goal `mtusoiy1-cfdhqr`: extend the dense-Qwen3 single-launch whole-layer
per-ctx ELF runlist path (1 submit/token, 88 tok/s on Qwen3-0.6B) to the
35B-A3B MoE, lifting its ~0.7 tok/s native decode.

## Correctness (task-4) — PASSED

6 structural bugs fixed (see npu-infer/docs/35b-forward-integration.md R102-R110):
QKV K/V pointer-stride, reference emb reshape, reference l2norm per-row,
reference residual raw-vs-normed, rope_theta 1e7, STD k-norm per KV head.
Weights byte-identical (max|d|=0.0); CPU attention+MoE match to float noise.
Token 4329 is the self-consistent int8 output (float64 parity needs bf16 O/QKV).

## Measurement (task-5, this box, Strix Halo NPU)

Prompt [151644] (BOS), greedy, single-token sequential decode, 16 tokens.

| Path | launches/token | decode | per-layer breakdown |
|---|---|---|---|
| CPU MoE (default) | **80** (40 QKV + 40 O) | **1751 ms/tok (0.57 tok/s)** | QKV 9.1 + attn 4.7 + O 5.8 + FFN 26.6 ms |
| NPU MoE fused (`NPU_MOE=1 NPU_MOE_FUSED=1`, M=1) | 160 (QKV/O + MOE_GUSGU/MOE_DSD ×2) | **3082 ms/tok (0.32 tok/s)** | FFN 62.3 ms/layer (expert pack-miss ~64 ms dominates) |

Bottleneck: the MoE FFN (26.6 ms/layer CPU = 58% of the 46 ms/layer). The
expert Q4NX dequant + float32 FFN runs on the CPU; the NPU path is *slower* at
M=1 because the per-expert dequant+repack (cache miss) costs ~64 ms/expert.

## Gap vs the runlist class

| | Qwen3-0.6B (dense, goal mtunui03) | Qwen3.6-35B-A3B (this goal) |
|---|---|---|
| runlist whole-layer | 1 submit/token, 11.4 ms/tok (88 tok/s) | **infeasible**: per-ctx ELF lane NaN (R59), cross-xclbin runlist batching blocked by differing ERT group_ids (R98) |
| best working path | — | 80 launches/token, 1751 ms/tok (0.57 tok/s) CPU MoE |

The single-launch runlist path could not be extended to the 35B MoE: the
per-ctx layer ELFs produce NaNs, and the MoE xclbins use different ERT
group_ids so one `xrt::runlist` cannot batch across them. The engine's
`NPU_MOE` fused path (v28 MOE_GUSGU/MOE_DSD) is the on-box 2-launch/layer
mechanism but loses to the CPU FFN at M=1 decode.

---

# Addendum — goal mtusoiy1 continuation (2026-09-16): re-verification + new evidence

The goal was resumed after the first completion was disapproved. This addendum
records a fresh, evidence-based re-verification of the blockers (no completion
is claimed — the headline deliverable is still absent).

## 1. Region-B weight format is now precisely pinned (was the "last closed piece")

Target model = `/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx`
(23,235,412,536 B; the model `npu-infer/src/runtime_layer_moe.cpp` /
`tools/moe_smoke.cpp` actually load). Its JSON metadata (parsed this session):

| tensor | dtype | shape | bytes |
|---|---|---|---|
| `model.layer.L.mlp.up_exps_proj.weight`   | I8 | [4096, 8, **5120**] | 167,772,160 |
| `model.layer.L.mlp.gate_exps_proj.weight` | I8 | [4096, 8, **5120**] | 167,772,160 |
| `model.layer.L.mlp.down_exps_proj.weight` | I8 | [16384, 2, **5120**] | 167,772,160 |
| `model.layer.L.linear_attn.qkv_proj.weight` | I8 | [256, 8, **8704**] | 17,825,792 |
| `model.layer.L.linear_attn.ssm_out_proj.weight` | I8 | [64, 16, **8704**] | 8,912,896 |
| `model.layer.L.self_attn.gate_proj.weight` | I8 | [128, 8, **8704**] | 4,849,664 |
| `model.layer.L.mlp.share_{up,gate,down}_exps_proj.weight` | I8 | [n, 8, **8704**] | varies |

Byte inspection (this session):
- **Experts = 5120-B Q4NX tiles.** Row 0..2 of `up_exps` carry 256 bf16 scales at
  `(g*32+lr)*2` ≈ 0.0026–0.004 and zps at `512+…` ≈ −0.016 … −0.025 — exactly the
  layout in `npu-infer/src/model.c` (`W = (q − zp) * scale`). Consistent for every
  row (96/256 scales in (0,10) per row, same as the 0.60B path).
- **Region-B (qkv/share_*/ssm_out/gate_proj) = 8704-B Q8_0 rows.**
  `[0:512]` = 256 bf16 scales (`block b` at offset `2b`, 32 int8/block) — all 256
  in (0,10) (≈0.00026–0.00047 for qkv); `[512:8704]` = 8192 int8. There is **no**
  256-entry zp region at `[512:1024]` (the values there are 0 / non-finite
  garbage). So region-B is 8-bit and *must be re-quantized to 4-bit Q4NX* for the
  layer ELF's on-device dequant — R114 is correct, R79/R83's "pure reorder" is not.

### The engine's current region-B packer is definitely wrong
`npu_pack_moe_region_b()` (model.c, R83) reads source rows at `NPU_MOE_8704_ROW =
8704` and copies only `[0:4736]` of each row = `[512 scales][4224 int8]` — it keeps
only Q+K+128 of V and discards 3968 int8 values (48%) of every row. It can never
be numerically correct.

### `flm104`'s model is a *different* quantization (do not mix)
`/home/bcloud/flm104/models/models/Qwen3.6-35B-A3B-NPU2/model.q4nx` is
21,044,871,704 B and stores **every** I8 tensor as `[…, 4736]` rows (region-B
included). It is not the file R50/R114 verified and not the file the engine loads.

## 2. The layer ELF's region-B read map (decoded from the generated TXN)

`tools/decode_txn --decode-only` over
`captures/txn-elfs-moe35b/moe_layer_ctx0.txn` (24636 words, 1468 cmds) yields the
arg-0 (weight-BO) patch table:
- region A (arg_off < 0x1bc00000): 64 descriptors.
- **region B (arg_off ≥ 0x1bc00000): 480 descriptors, all `linear=1`, lengths
  18944 (= 4×4736) or 4736, every `arg_off−0x1bc00000` an exact multiple of 4736.**
- Region-B = 3456 rows: share_up 0–127, share_gate 128–255, share_down 256–383,
  qkv 384–2431, gate_proj 2432–3455; max offset `0xf89800` (= row 3440).
- The *static* descriptors touch only 864/3456 rows (4 of every 16, at 16-row
  strides) — i.e. the kernel re-issues each base descriptor in a loop (4 passes);
  the BO row order is the desc-logical tensor order.

## 3. FastFlowLM cannot serve the 35B on this box (both versions tested)

- **v1.0.4** (`/home/bcloud/flm104/bin/flm`) — `flm run qwen3.6-moe:35b-a3b`
  SIGSEGVs during `load_weights` (the documented ASLR-dependent `qwen3_6_reorder`
  bug; retry does not reliably help).
- **v1.0.5** (downloaded this session:
  `fastflowlm_1.0.5_linux.tar.gz`, self-contained `flm-real` + `lib/*.so` +
  `xclbins/Qwen3.6-35B-A3B-NPU2/*`) — loads the 23.2 GB model, then the **first
  forward fails**:
  ```
  [ERROR] Generation error: runlist failed execution (ERT_CMD_STATE_TIMEOUT)
  Kernel Instance: MLIR_AIE
  txn_op_idx = 0xFFFFFFFF
  ctx_pc = 0x28B060AD
  ```
  So `libqwen3_6_moe_npu.so`'s 35B whole-layer path is broken in v1.0.4 (NaN /
  segfault) *and* v1.0.5 (hang) — the same conclusion as R59/R68, now confirmed on
  a second, newer release.

## 4. The capture oracle is gone
`~/.cache/moe-cap-rb` (77 GB) and `~/.cache/moe-cap4` (73 GB) — the only
byte-oracles for the region-B BO — no longer exist. `~/.cache/moe-cap3` is empty.
Byte-exact derivation from captures is therefore impossible.

## 5. New lead (not yet exploited): the weight-prep function is EXPORTED
`nm -D libqwen3_6_moe_npu.so` shows
`qwen3_6_moe_desc::load_linear_weights(int, Q4NX&, buffer<uchar>&, buffer<bf16>&,
buffer<bf16>&)` at `0x7ed90` (~9.6 KB) — exported, so addressable/callable. This
contradicts the older "closed lib, no entry point" framing. Its body is mostly
memcpy + calls; the actual Q8_0→Q4NX re-quant lives in a non-exported helper, so
deriving the exact transform needs a focused disassembly/call effort.

## Verification verdict (honest)

| objective requirement | status |
|---|---|
| single-launch whole-layer per-ctx ELF runlist path **extended to the 35B MoE** | **NOT MET** — no working whole-layer ELF exists for the 35B: the runtime's own ELF is broken in v1.0.4/v1.0.5 (§3) and the engine's copy NaNs (R89/R92); `MoERuntimeLayerEngine` executes into all-NaN |
| decode lifted from ~0.7 tok/s toward the dense runlist class | **NOT MET** — measured alternatives are *below* baseline: CPU MoE 0.57 tok/s, NPU-fused M=1 0.32 tok/s (task-5) |
| via **one** `xrt::runlist` submit/token | **NOT MET** — engine path is 80 launches/token; cross-xclbin batching is impossible (R98, differing ERT group_ids); the runtime's own single runlist needs the broken whole-layer ELF |

Residual blocker (unchanged, recurred across turns): **correct region-B weight
content (Q8_0→Q4NX re-quant) is underived, and there is no working whole-layer
35B ELF to consume it.** With the captures deleted and no FastFlowLM source, this
is a research-scale effort (disassemble `load_linear_weights`'s re-quant helper,
or obtain upstream source) — not completable with on-box assets alone.
