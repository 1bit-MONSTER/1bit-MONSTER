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

---

# Addendum 2 — ON-BOX live capture (2026-09-16, per user: "you live on the strixhalo")

Correction to the prior "everything is off-box" framing: the **FastFlowLM source
tree IS on this box** at `/home/bcloud/amd-oss/fastflowlm/src` (include/models/
qwen3_6_moe/qwen3_6_moe_npu.hpp, common/AutoModel/modeling_qwen3_6_moe.cpp,
lib/xrt/libqwen3_6_moe_npu.so, 35B xclbins). The npu-side `*_npu.cpp` impl is not
in the tree, but the **.so exports the weight-prep entry points**, so the runtime
can be driven/inspected locally.

## What was captured live (gdb on the running runtime, this box)

`qwen3_6_moe_desc::load_linear_weights(int, Q4NX&, buffer<uchar>&, buffer<bf16>&,
buffer<bf16>&)` is exported at `0x7ed90` in
`/tmp/flm105/lib/libqwen3_6_moe_npu.so`. Breaking on it and dumping its buffer
args (3rd arg = `[rcx+0x10]`, per the prologue `mov r14,[rcx+0x10]`) yields, per
layer:
- **arg3 = the 512 MB expert pool BO** (`0x7fffa8000000`, `0x7fff88000000`, … one
  per layer). **R50's pool layout reproduces byte-exactly on-box**: spot-checked
  206/206 pool rows (up/gate 32-row blocks `j=base+8*(i%4)+i//4`, down 8-groups
  `[0,2,4,6,1,3,5,7]`) → `tools/verify_moe_current_layout.py`'s spec is CONFIRMED
  against the live v1.0.5 runtime, not just the deleted captures.
- arg3 rows ≥102624 (through 113359) are all-zero → **region-B (share_*/qkv/
  ssm_out/gate_proj) is NOT in the pool BO** (R50's "allocator slack" holds).
- arg4/arg5 = per-layer 5 MB buffers (the linear-attn BOs) — not region-B.

A process-wide memory scan for the raw `input_layernorm` bytes found no hit →
region-A is transformed too (consistent with the R14 note that `reorder_cpy`
dtype=8 chunks the BF16 small tensors).

## FastFlowLM v1.0.5 tested (downloaded this session)

`fastflowlm_1.0.5_linux.tar.gz` (portable `flm-real` + `lib/*.so` +
`xclbins/Qwen3.6-35B-A3B-NPU2/*`). It **loads** the 23.2 GB model and reaches the
prompt, then the first forward dies: `runlist failed execution
(ERT_CMD_STATE_TIMEOUT)`, `Kernel Instance: MLIR_AIE`, `txn_op_idx=0xFFFFFFFF`,
`ctx_pc=0x28B060AD`. So the 35B whole-layer path is broken in v1.0.4
(SIGSEGV/NaN) **and** v1.0.5 (hang) on this box.

## Status

Region-B's exact content is still not located (it is a buffer that is neither the
pool nor the 5 MB linear BOs; the per-layer ~460 MB region-A+B weight BO the
layer ELF reads is created/filled outside `load_linear_weights`' three args).
The next concrete on-box step is to hook the layer kernel's arg-0 BO at submit
(e.g. via `cap_interposer` on `xrt::bo` creation, or a breakpoint on the runlist
arg setup) and dump it — the runtime's own forward need not succeed for that.

## Addendum 3 — on-box BO hunt via the set_arg interposer (2026-09-16)

Ran the live v1.0.5 runtime under `tools/capture/cap_interposer.cpp`
(LD_PRELOAD; it hooks `xrt::run::set_arg_at_index`, `xrt::ext::bo::bo`,
`xrt::runlist::{add,execute,wait}`, `xrt::elf`) to find the layer kernel's
region-A/B weight BO.

Result (652 SETARGs captured before the interposer's own SIGSEGV):
idx -> size map for the runs reached before the ERT timeout:
  idx3 = 536,870,912 (expert pool, 80x) ; idx4 = 1 MB ; idx5 = 2 MB ;
  idx6 = 5 MB / 1 MB ; idx7 = 128 MB / 3 MB ; (one 542,113,792 full-attn pool).
A probe that dumped `[map()+0x1bc00000, +16 MB]` of the first 40 distinct
512 MB idx-3 BOs found **all 40 zero** at both offset 0 and 0x1bc00000 — i.e.
the 512 MB BOs bound at `set_arg` are fresh/zero, NOT the R50 pool (whose
0x1bc00000 region is live `down` data, per Addendum 2). So no ~460 MB or ~16 MB
region-A/B BO is bound before the runtime's first runlist dies.

Interpretation: the v1.0.5 forward fails at (or before) the first runlist
execute, so the LAYER-ELF run's args (arg0 = region-A/B) are never reached; and
`load_linear_weights`' three args (512 MB pool, 5 MB, 5 MB) do not include it
either (Addendum 2). The region-A/B BO is therefore created/filled on a path not
exercised by a failing forward — it needs either a forward that at least binds
the layer run (v1.0.4 crashes at load; v1.0.5 times out before it), or a
breakpoint on the BO-creation/resident-object path inside `Impl::load_weights`.

Net on-box assets now banked: FastFlowLM source tree (Addendum 2), the live
expert-pool BO (R50 verified 206/206), and the interposer idx/size run map.

## Addendum 4 — pre-exec BO probe (2026-09-16)

Extended the interposer's `runlist::execute` pre-dump with a small-identity probe
(`CAP_BIG_PROBE`) so big BOs emit 4 KB @0 + 4 KB @0x1bc00000 instead of 512 MB.
The run reached **82 `runlist::execute`** calls but the pre-exec big-BO dump
produced nothing (the `run_bo()` slot lookup returned no >3 MB BOs for the
recorded runs), and the interposer SIGSEGVs during the run. The synced BO set
for the layer includes **11,534,336 B (11 MB)** and **10,485,760 B (10 MB)**
dumps; the 11 MB one is plausible BF16 weight data (0.80, −0.005, 0.29, …) but
contains no raw bytes of any layer-0 qkv/share_*/gate_proj/layernorm/router
tensor, so it is not region-A/B in any raw form.

Conclusion unchanged: on this box the region-A/B weight BO is not reachable from
a failing forward, and the runtime's own 35B forward cannot complete
(v1.0.4 SIGSEGV at load, v1.0.5 ERT_CMD_STATE_TIMEOUT at the first runlist).

## Addendum 5 — REPAIR: region-B packer derived from the runtime's own reorder (2026-09-16)

`qwen3_6_reorder_cpy` in the on-box `libqwen3_6_moe_npu.so` is callable at
`(gen_layer_seq - 0x97ad0) + 0x68b80` (as `tools/verify_moe_reorder_qkv.cpp`
does). Calling it on the LIVE model's region-B tensors yields the ground truth
the engine's packer was missing.

Findings (byte-verified against the runtime function):
- The source windows are **4736-B slices at a 4736-B stride** from the Q8_0
  tensor — *not* row-aligned 8704-B trims (that convention matched 1/221).
- There is **NO re-quantisation**; the Q8_0 bytes (256 bf16 scales + int8) are
  preserved.
- The reorder is an A/B interleave inside `2H`-window blocks:
  `out[blk*2H + i] = in[blk*2H + i/2 + H*(i%2)]` with **H = n/256**, where `n`
  is the tensor's row count (the count passed to reorder_cpy).

Verified: qkv `n=2048 -> H=8` matches **2048/2048**; gate_proj `n=1024 -> H=4`
matches **1024/1024**; share_* (`n=128 -> H<1 -> identity`) confirmed via an
`n=256` call whose first 128 windows are the raw windows (128/128).

**Repair applied** to `npu-infer/src/model.c:npu_pack_8704_tiles`: the old code
read at the `NPU_MOE_8704_ROW` (8704) stride and trimmed rows; it now reads at
`NPU_MOE_ROW_BYTES` (4736) and applies the H=n/256 interleave. Rebuilt packer
output is byte-identical to the runtime's reorder for qkv (2048/2048) and
gate_proj (1024/1024). This removes the "wrong region-B weights" cause of the
engine layer's all-NaN output (R90/R92/R93/R114); the end-to-end NPU re-test
(`tools/moe_smoke`) is pending the device being free.

### Addendum 5b — end-to-end re-test with the repaired packer

`tools/moe_smoke` (single xrt::runlist: layer ELF + lm_head, repaired region-B):
```
forward(1): EXECUTED
logits: argmax=0 max=0.0000 NaN=0 (of 248320)
act (device-synced): 2048/2048 NaN
```
So the repaired (byte-exact) region-B **removed the all-NaN logits** (R90/R92 had
NaN logits) but the layer's hidden state is still all-NaN. The remaining cause is
NOT region-B: it is either (a) the region-A layout, which the engine packs
"best-effort" as a naive sequential copy (input_layernorm, post_attention_layer-
norm, ssm_conv1d/norm/a/dt.bias) while the runtime demonstrably *reorders* its
weight regions (region-B proof above), or (b) the lib's own 35B layer sequence
(R59's conclusion). Disambiguating needs region-A derived the same way region-B
was — i.e. locating the runtime's norm/router transform (the `reorder_cpy`
dtype=8 BF16 path is the candidate) — or an engine-side layer sequence.

## Addendum 6 — the lib's sequence path is unusable (localized on-device)

Ran the engine with the repaired region-B and dumped the act BOTH sides of the layer:
- pre-layer act (after embed): **nan=0, 2038/2048 nonzero** (values ~0.016, -0.022) — clean;
- post-layer act: **2048/2048 NaN** — so the NaN enters inside the layer computation,
  not the embedding.

Then generated lib sequences STAGE BY STAGE (new tool `gen_layer_stages_moe.cpp`,
driving the exported `qwen3_6_moe_npu_sequence` generators) and ran each through the
engine:
- `_send_hidden_states` alone (34 words, 768 B ELF) → act **all-NaN**;
- `_send_hidden_states`+`_send_rms_weights` (64 words) → all-NaN;
- hidden+rms+conv weights+conv1d (1656 words) → all-NaN;
- full `gen_layer_seq` linear layer (24636 words) → all-NaN.

Every subset NaNs, and a "send hidden states" cannot itself change the act — so the
lib's instruction streams are not a valid decomposition to truncate: the whole
sequence path (or its arg/BO contract with this engine) is broken for the 35B. This
substantiates R59 and means the objective's layer must be **rebuilt** with
engine-authored kernels (the engine's own mm/dequant xclbins), not reused from the
lib. That is the multi-kernel per-layer rebuild now underway.

### Addendum 6 CORRECTION (the earlier stage-subset claim was wrong)

The four stage-subset runs in Addendum 6 passed the model **directory** as argv[1]
instead of model.q4nx, so they all died at `model_load` (`mmap failed: No such
device`) and the act files read afterwards were STALE from the full-layer run. The
"even _send_hidden_states alone NaNs the act" conclusion is therefore VOID.

Re-ran correctly (full model path, ELF named `moe_layer_ctx1.elf` in the elf dir):

| sequence | words | result |
|---|---|---|
| `_send_hidden_states` | 34 | EXECUTED; act **unchanged** (nan=0, pre == post) |
| `_send_rms_weights` | 34 | EXECUTED; act unchanged |
| hidden + rms | 64 | EXECUTED; act unchanged |
| hidden + rms + conv-weights + `gen_seq_conv1d` | 1656 | **`ERT_CMD_STATE_TIMEOUT`** (runlist failed) |
| full `gen_layer_seq` linear layer | 24636 | EXECUTED; act **all-NaN** |

So the first real localisation is: the DMA-only prefix is harmless and does not
touch the act; the **conv1d stage is where the failure signature first appears**
(ERT timeout in isolation; the full layer gets past it but then NaNs the act).
That is the stage to rebuild first — not a blanket "the lib path is broken".

### Addendum 7 — the layer's arg map is now readable per stage

Decoding the generated per-stage sequences (corrected runs) gives the layer's
weight sources factually:

| generator | arg (0-based in ELF) | engine slot | read | len |
|---|---|---|---|---|
| `_send_hidden_states` | arg1 | slot4 = act | MM2S | 1024 |
| `_send_rms_weights` | arg2 | slot5 = **router BO** | MM2S @0 | **3072** |
| `_send_linear_conv_weights` | (see full decode) | slot5 | | |

So the RMSNorm weights are read from the **router BO at offset 0**, length 3072 B —
while `npu_pack_moe_router_bo()` writes `input_layernorm` (4096 B) at offset 0 there.
That is a concrete layout mismatch on the norm path and a prime NaN suspect for a
linear-attn layer. `gen_seq_conv1d` is NOT called by `_gen_linear_sequence`
(the label's conv is done inline with `npu_dma_memcpy_nd`), so the earlier "conv
stage timeouts" result came from calling that generator with invented arguments and
is not evidence about the linear layer.

## Addendum 8 — internal-BO dump localises the NaN to the layer compute

Important device fact (user): the NPU exposes `hwctx_limit = 16` (xrt-smi shows
multiple live hw contexts per PID) — **the box supports concurrent hw contexts**, so
runs do not need accel0 to be idle. All runs below were done concurrently.

Added `MoERuntimeLayerEngine::dump_bos()` (env `NPU_DUMP_BOS`) and ran one linear
layer, dumping every BO device-synced after the layer run:

| BO | bf16 result |
|---|---|
| `weightA` (region-A head, norms/smalls) | **clean** (1.10, 1.04, 1.05, 0.96 …) |
| `router` (RMS w + router + shared gate) | **clean** |
| `norms` (5 MB linear BO) | **clean** |
| `weightB` (region-B head) | NaN only where Q8_0 int8 bytes pattern as bf16-NaN (expected — raw int8) |
| `logits` | all zero (lm_head wrote nothing finite) |
| `kv` (128 MB state) | 1634/524288 **NaN** |
| `act` | **2048/2048 NaN** |

So the layer's *inputs* (norm weights, router, region-A) are finite and the NaN is
produced by the layer's **compute** (the attention/state path and hence the act).
Diagnostic: zeroing the region-B `[512:1024]` "zp" half (the Q4NX-vs-Q8_0 question)
did **not** change the NaN — so the zp interpretation is not the cause.

## Addendum 9 — the NaN is structural, not weight content; the engine is missing the per-forward setup

Zeroing the weights one BO at a time (diagnostics gated by env vars, since removed):

| zeroed | act result |
|---|---|
| region-B (all of share_*/qkv/gate_proj) | still 2048/2048 NaN |
| arg2 router BO | still all-NaN |
| arg3 5 MB norms BO | still all-NaN |

So **no weight content causes the NaN** — with every weight zeroed the layer still
NaN, which a finite RMSNorm/GEMM chain cannot produce. The ELF is also byte-identical
whether generated by the v1.0.4 or the v1.0.5 lib, so the generator version is out.

### Lead

The *working* dense engine (`runtime_layer.cpp`) performs per-forward setup the MoE
engine never does:

1. `apply_rope(ctx_len)` → `update_rope_i6(i6_bos_[L], pos)` writes the **RoPE table**
   into the i6 BO for every layer before each forward (theta from the model config;
   1e7 for this model);
2. i6 is built as `[1.0 x64][0 x64][q_norm 128][k_norm 128] + RoPE`.

`runtime_layer_moe.cpp` has no equivalent: its arg3 is packed as the 5 MB linear-attn
BO (ssm head), with no RoPE table and no q/k-norm slots. A missing/garbage rotary
table on a finite input is exactly the kind of op that yields NaN, and it matches the
"structural, weight-independent" signature above.

Next: implement the MoE layer's i6/norm-BO setup from the dense template (RoPE table
via the model's rope_theta + the q/k norm slots) and re-run the BO dump.

### Addendum 9b — the RoPE-table lead is also excluded

Wrote a correct identity RoPE table (pos 0: cos=1, sin=0, theta=1e7) into
`arg3[0:256]` (the MoE analog of the dense i6 head) before the runlist → the act is
**still all-NaN**. So the missing RoPE table is not the cause either.

With this, the NaN survives: region-B zeroed, router zeroed, 5 MB norms zeroed, RoPE
table written, and ELF identical across lib versions. Every input to the layer can be
made finite/known and it still NaNs ⇒ the defect is inside the lib's 35B layer
instruction sequence itself (or its kernel-side contract), matching R59's conclusion
that the runtime's 35B layer forward is non-functional. Reusing that sequence is a
dead end; the layer must be authored engine-side (the rebuild).

## Addendum 10 — the ENGINE'S OWN 35B path measured today: 19.4 s/tok (0.05 tok/s)

Ran the engine's own 35B entry point end-to-end on this box:

```
NPU_MOE=1 NPU_MOE_FUSED=1 NPU_GREEDY=1 \
  engine/npu/build/npu_engine_qwen3_6_moe_35b model.q4nx 4 /tmp/ids35b.txt
```
Result:
```
[ModelConfig] MoE: experts=256 top_k=8 im_exp=512 shared=1 gdn=1
NPU MoE enabled (MOE_GU/D/SGU/SD xclbins + fused v28 GUSGU/DSD)   <-- fused path IS active
small-M(_m0) xclbins absent; decode uses M=128 ctx
Prefill: 21741 ms
  [1] 154742  22792 ms
  [2] 16023   22938 ms
  [3] 58404   30951 ms
=== 19407.0 ms/tok (0 tok/s) | boot=36ms batches=3 tokens=4 ===
```

So:
- the engine's own fused MoE is engaged (2 launches/layer), yet **decode is 19.4 s/tok** —
  i.e. **0.05 tok/s**, ~27× SLOWER than the ~0.7 tok/s baseline the goal is meant to
  lift, and far from the R99 "1.5 tok/s" figure (which does not reproduce today);
- the bottleneck is the **M=128 QKV/O kernels** plus launch overhead, not the MoE;
- the engine's own comment (line 1477) records the small-M alternative as
  *"the _m1 kernel's weight contract differs from the M=128 path (garbage decode, no
  perf win — launch-bound)"* — so M-shrinking is measured not to help.

Combined with Addenda 6-9b (the lib's whole-layer ELF NaNs with every input forced
finite), **both routes to the objective are currently dead or far off**:
  - lib per-ctx ELF runlist path -> NaN (R59);
  - engine's own path -> 19.4 s/tok, launch-bound, small-M documented as no-win.
