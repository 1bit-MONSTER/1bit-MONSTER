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

### Addendum 10b — CORRECTION: M=1 decode is ~1 tok/s; 19.4 s/tok was the M=8 BATCH path

The `npu_engine_qwen3_6_moe_35b ... 4 <ids>` run goes through the "M=8 Batch Decode"
path; re-running with **1 token** (and NPU_TIMING=1) gives:

```
=== M=8 Batch Decode (1 tokens) ===
=== 867.8 ms/tok (1 tok/s) | boot=28ms batches=0 tokens=1 ===
```

So the engine's own 35B **single-token** decode is **867 ms/tok ≈ 1.0 tok/s** —
about 1.4× the ~0.7 tok/s baseline (and consistent with R99's 1.5 tok/s), while the
**batch path is ~22× slower** (19.4 s/tok) and is a separate regression.

Per-layer init timings recorded under NPU_TIMING (dequant/pack phase, not the token):
`[moe l=L pack] ~100-120 ms`, `gu ~115-145 ms`, `d ~150-180 ms`, `moe_ffn_npu
~150-180 ms` per layer.

So the honest current state of the two routes:
- **engine's own M=1 path: ~1.0 tok/s** (works; modest gain over baseline; the M=8
  batch path is broken/slow);
- **lib per-ctx ELF runlist path: NaN** (dead, Addenda 6-9b).
The objective's dense-class (~88 tok/s) runlist decode remains ~88× away on the
working route.

### Addendum 11 — the 35B runtime ERT timeout is SYSTEMIC (not the intermittent Llama kind)

@agent-afbeb7 found the shipped FLM runtime itself trips `ERT_CMD_STATE_TIMEOUT`
on llama3.1:8b **intermittently** (France/Japan ok, Italy failed), suggesting device
state rather than a pure binding bug. Tested the same hypothesis on the 35B:
22 consecutive `flm run qwen3.6-moe:35b-a3b` attempts on flap-v1.0.5 —
- 20/22 died in `load_weights` with the ASLR SIGSEGV,
- **2/22 loaded, and BOTH then hit `ERT_CMD_STATE_TIMEOUT` on the first runlist.**

So on the 35B the ERT timeout is **reproducible, not transient** (2/2 loads), unlike
the Llama case. The 35B runtime whole-layer path is reliably broken rather than
flaky, which is consistent with (and does not rescue) Addenda 6-9b.

### Addendum 10c — CORRECTION of 10b: M=1 decode is ~9.5-9.9 s/tok; FFN dominates

Addendum 10b was wrong: the `... 1 <ids>` run only executes the BOOT (the decode
`while` loop needs step<ng), so its "867.8 ms/tok" is not a decode rate. Re-ran with
3 tokens and NPU_TIMING, which prints the real per-layer breakdown:

```
=== M=8 Batch Decode (3 tokens) ===
[decode-stage] QKV=27.3ms attn=5.6ms O=25.4ms FFN=188.7ms misc=0.0ms per-layer
  [1] 154742   9911 ms/tok
[decode-stage] QKV=27.1ms attn=5.4ms O=24.1ms FFN=181.0ms misc=0.0ms per-layer
  [2] 16023    9533 ms/tok
=== 6747.8 ms/tok | batches=2 tokens=3 ===
```

So the engine's own, M=1 **sequential** 35B decode is **~9.5-9.9 s/tok (≈0.10 tok/s)**,
and the cost is dominated by the **MoE FFN at ~188 ms/layer (76% of the ~247 ms/layer)**;
QKV 27 ms + O 25 ms + attn 5.6 ms are the rest. That matches the engine's own comment
(v12 note): decode is launch/kernel-bound because the microkernel is **M=128-baked**
(~4 ms minimum per launch, and the fused GUSGU/DSD pair costs ~94 ms/launch here), and
"the fix is per-shape small-M xclbins … or fused layer streams — not a runtime regen".

Net: the working engine route is **~0.10 tok/s**, i.e. ~7× BELOW the ~0.7 tok/s
baseline, with a fully identified bottleneck (MoE FFN launches on an M=128-baked
kernel). The objective's dense-class target remains ~900× away.

### Addendum 12 — the 35B FFN bottleneck is HOST per-token expert dequant, not the NPU

Dug into the 188 ms/layer FFN from addendum 10c. Per-layer instrumentation shows TWO
packs (routed + shared) and the FFN total:

```
[pack l=38] 122.2 ms (miss=119.9 sync=0.3)     <- routed top-8: HOST dequant, cache MISS
[pack l=38]   2.8 ms (miss=  0.0 sync=0.4)     <- cache HIT
[moe_ffn_npu_batch l=38 M=1] 193.3 ms (U=8)
[pack l=39]  81.9 ms (miss= 79.7)
[pack l=39]   3.1 ms (miss=  0.0)
[moe_ffn_npu_batch l=39 M=1] 194.6 ms (U=8)
```

`EXP_CACHE_SZ = 256` (all experts fit), but the **routed** experts change per token, so
the cache misses and the host re-dequantises 8 experts × 3 matrices every layer
(~15 ms/expert → ~120 ms/layer → **~4.8 s of the ~9.9 s/token**). The NPU GUSGU/DSD
portion is the remainder (~70 ms/layer).

So the working route's breakdown at M=1 is approximately:
| stage | per layer | per token (40 layers) |
|---|---|---|
| MoE FFN host expert dequant (misses) | ~120 ms | ~4.8 s |
| MoE FFN NPU (fused GUSGU/DSD) | ~70 ms | ~2.8 s |
| QKV (M=128-baked kernel) | 27 ms | 1.1 s |
| O (M=128-baked kernel) | 25 ms | 1.0 s |
| attention (GDN) | 5.6 ms | 0.2 s |

Actionable lever: dequantise/bf16-preload the expert pool ONCE (init already spends
~50 s on "Dequant+pack") so the per-token pack is a memcpy instead of a scalar
int4→float→int8 pass. Best case that removes ~4.8 s of ~9.9 s — i.e. ~0.2 tok/s, still
~440× short of the dense class. Everything else (the M=128-baked kernels) is the
engine's documented "per-shape small-M xclbins or fused layer streams" problem.

### Addendum 13 — Addendum 11's "systemic 35B ERT" is NOT established; it is probably CONTENTION

@agent-baaa57 (dense-Qwen3 runlist lane) produced a clean A/B on this box: with TWO
engines on accel0 the 1.7B/8B runlist prefill dies `ERT_CMD_STATE_TIMEOUT`
(txn_op_idx=0xFFFFFFFF, ctx_pc=0x28B06005) at ctx=1551 / ctx=1; with accel0 **quiet**
the identical commands complete (1.7B 5 tok/s, 8B 12 tok/s). So the ERT is
contention-induced, and @agent-afbeb7's dmesg evidence (83 firmware-timeout dumps,
2-4 contexts per dump, driver `timeout_in_sec=2`) points at a TDR cliff rather than a
dead kernel.

That invalidates the conclusion of Addendum 11. My two 35B runtime attempts that
reached the forward BOTH ran while other agents held accel0 (the other 20 attempts
never loaded, dying in the ASLR SIGSEGV). So **my 35B runtime ERT may also be pure
contention**, and the correct test is the same A/B: run the 35B runtime with accel0
QUIET, and if it still ERTs, raise `timeout_in_sec` 2 -> 15 (writable, reversible,
system-wide) and retry.

If the 35B runtime forward completes when quiet, then (a) the "lib whole-layer path is
reliably broken" claim in Addenda 6-11 is overstated for the ERT half, and (b) it
becomes worth re-testing whether its forward still NaNs (R59) or produces real logits —
which is the only thing that would reopen the objective's mechanism.

### Addendum 14 — QUIET-device test: the 35B runtime ERT is STRUCTURAL (addendum 11 was right)

Ran the decisive A/B that @agent-baaa57's result implied for my lane: FastFlowLM v1.0.5
`run qwen3.6-moe:35b-a3b` with **accel0 completely quiet** (fuser empty, all other
agents stopped, verified free immediately before each attempt).

- 14 attempts: 13 died at load with the ASLR SIGSEGV; **1 loaded, and it ERT'd**:
```
[FLM]  Prefill chunk 1/1 with 13 tokens
[ERROR]  Generation error: runlist failed execution (ERT_CMD_STATE_TIMEOUT)
ELF UUID: 65c06f83-...   txn_op_idx = 0xFFFFFFFF   ctx_pc = 0x28B060AD
```
- ctx_pc `0x28B060AD` is the same value @agent-afbeb7 recorded for the 35B/runtime ERT.

Combined with the earlier 2 loads under load: **3/3 loads ERT, one of them with the
device quiet.** So unlike the dense-Qwen3 runlist ERT (which @agent-baaa57 showed is
contention/TDR-induced and completes when quiet), **the 35B runtime whole-layer ERT is
structural** — it is not the `timeout_in_sec=2` contention cliff.

=> Addendum 11's conclusion stands; addendum 13's retraction is itself retracted FOR
THE 35B (the contention explanation applies to the dense path, not this one). The 35B
whole-layer ELF path fails reliably and independently of device load, consistent with
Addenda 6-9b (the lib's 35B layer sequence is broken).

### Addendum 15 — toolchain hazards banked for any engine-side layer rebuild

From @agent-7f1cce (mtygjrxl, fused prefill), five mlir-aie/NPU failure modes that are
all "the toolchain tells you nothing". Three matter if the 35B layer is ever authored
engine-side (the only remaining route after Addenda 6-14):

1. `XAIE_INVALID_ELF` / `_XAie_LoadProgMemSection():231 Overflow of program memory` is
   normally the **per-core 16 KB `.text`** limit, not a corrupt ELF. Check with
   `llvm-readelf -S <core>.elf` vs `0x4000`. `.bss` is a separate budget.
2. Soft-`double` is very expensive *code* on a 32-bit AIE core (one kernel:
   `.text` 16,208 -> 10,976 B when double->float, and it got MORE accurate). Grep
   kernels for `double` before blaming shapes.
3. A core's fifo ratio is compile-time: per N-tile it consumes `n_k` A-tiles and
   `n_k` W-tiles and produces 1 C-tile. A phase streamed with a DIFFERENT `n_k`
   **stalls silently** — no error, no timeout, no XRT failure, buffers just stay
   zero, and it can look like plausible partial correctness. This is a live hazard
   for a NaN/zero hunt.
4. Verify phases are actually in the emitted MLIR (`grep -c scf.for`), since patches
   silently no-op when anchors move.
5. `aie.dma_bd` legality (AIEXDialect.cpp ~236, applied to the REVERSED stride
   array): the LAST stride may be anything; every other must satisfy
   `stride * elemWidth % 256 == 0` (bf16 -> even). Leading size-1 dims are NOT
   exempt; no BD can transpose.

Detail: `engine/npu/FUSED-RMSNORM-QKV-DESIGN.md` (branch `family/head-block-loop`).

### Addendum 16 — hwctx residency for the expert-pool lever (from @agent-7f1cce)

Measured on this box (not inferred):
- `xrt-smi examine -r platform` -> **Total Columns: 8**; params `hwctx_limit=16`,
  `context_limit=64`, `timeout_in_sec=15` (raised by @agent-afbeb7's TDR repair).
- **20 hw_contexts were created and stayed resident at once** against an all-8-column
  kernel — past the advertised 16. So context count is not the cap; the **8 columns**
  are. No kernels were run in them, so this is residency, not throughput.
- `load_xclbin` fails here (`load_axlf: Operation not supported`); `register_xclbin`
  works.

Relevance to addendum 12 (host expert packing = ~4.8 s of the ~9.9 s/token): weight
residency per expert IS achievable — 20 resident contexts means experts need not be
re-packed per token. But 20 contexts cannot each own 8 columns, so compute serialises
(one at a time). The shape is **N experts resident, one computing at a time** — useful
against weight reloading, not for parallel experts. Bounded next step if ever pursued:
extend the probe to EXECUTE in N contexts concurrently and measure the serialisation.

Also confirmed (via @agent-7f1cce): the only worktree on `goal/runlist-decode-wire` is
this one (`/home/bcloud/1bit-MONSTER-goal`), so the git churn traced back here; my
process is now path-scoped commits (`git commit <paths>`), index checked first.

### Addendum 17 — the expert-pack lever is ALREADY IMPLEMENTED (cache warms): real steady state ~3.5 s/tok

Before writing the prepack change proposed in addendum 12, tested whether the expert
cache simply needs to warm. Ran 8 decode tokens with NPU_TIMING:

```
[1] 4263 ms   [decode-stage] QKV=8.7 attn=5.7 O=5.4 FFN=86.2 ms/layer
[2] 4160 ms   QKV=8.7 attn=5.8 O=5.5 FFN=83.3
[3] 4105 ms   QKV=8.6 attn=5.5 O=5.3 FFN=82.6
[4] 3766 ms   QKV=8.7 attn=5.5 O=5.4 FFN=74.0
[5] 3571 ms   QKV=8.6 attn=5.8 O=5.4 FFN=68.7
[6] 3619 ms   QKV=8.6 attn=5.4 O=5.5 FFN=70.4
[7] 3668 ms   QKV=8.6 attn=5.8 O=5.3 FFN=71.3
=== 3485.9 ms/tok ===
```

So the per-layer cost falls and then plateaus: **FFN ~70 ms/layer, QKV ~8.6 ms, O ~5.4 ms,
attn ~5.6 ms**, i.e. a steady-state decode of **~3.5 s/tok (≈0.29 tok/s)** — 2.8× better
than the 9.5-9.9 s/tok I reported in addenda 10c/12, which were **cold-cache** numbers
(and measured while other agents were loading the device).

CONSEQUENCE: addendum 12's proposed lever (prepack the expert pool to remove host
dequant) is **already implemented** — `exp_cache[l]` (EXP_CACHE_SZ=256) is warm after a
few tokens and the pack drops from 122 ms (miss) to 2.8 ms (hit); misses are only
cold-start. The remaining bottleneck is the **FFN's NPU kernels (~70 ms/layer of
2 launches ≈ 35 ms/launch)** — the engine's documented M=128-baked-kernel problem
("per-shape small-M xclbins or fused layer streams"), not host packing.

Corrected steady-state accounting per token: FFN ~2.8 s (80%), QKV ~0.34 s, O ~0.22 s,
attn ~0.22 s => **~3.5 s/tok, ~0.29 tok/s**, i.e. ~2.4× below the ~0.7 baseline and
~300× from the dense class.

## Addendum 18 — M=1 (true 1-row) fused MoE xclbins BUILT and wired; GEMM ~3x faster but the FFN is pack-bound

### Built
Adapted the engine's own M=1 generator (`generators/n1_core_i8_m1.py`, the one behind
`build_qwen3_0_6b_m1.sh`) to the 35B fused MoE shapes and built:
```
engine/npu/xclbins/final_i8_MOE_GUSGU_qwen3_6_35b_a3b_m1.xclbin  (K=2048 N=9216, 38,106 B)
engine/npu/xclbins/final_i8_MOE_DSD_qwen3_6_35b_a3b_m1.xclbin    (K=4608 N=4096, 38,106 B)
+ matching insts_*.txt
```
Toolchain gotcha fixed en route: the Python `aie` API in `install_tmp` is 2026-08-07 but
`build_tmp/bin/aiecc` is 2026-09-12 — a 5-week drift, so the newer aiecc rejects the
older `aie.dma_bd` syntax (same error on the already-working 0.6B shapes, so it was
toolchain, not shapes). The **2026-07-12 `mlir-aie/install/bin/aiecc` parses it**; the
build script now uses that.

### Wired
`moe_ctx` in `npu_engine_universal.cpp` now honours `NPU_MOE_SMALL_M=1` and prefers the
`_m1` xclbin/insts pair with `MD=1` (opt-in; unchanged otherwise).

### Measured
```
NPU_MOE=1 NPU_MOE_FUSED=1 NPU_MOE_SMALL_M=1 ... 8 tokens
  moe_ctx MOE_GUSGU M=1 (m1 kernel) K=2048 N=9216
  moe_ctx MOE_DSD   M=1 (m1 kernel) K=4608 N=4096
  [1] 4617 ms   [2] 4222   [3] 3959   [4] 3667   [5] 3759   [6] 3569   [7] 3417
  === 3493.8 ms/tok ===
[decode-stage] FFN plateaus at ~65-73 ms/layer (M=128 was ~68-86)
```
The tokens are IDENTICAL to the M=128 path (154742, 16023, 136614, 25238, 32858, 248050,
184997) -> the m1 kernel is numerically equivalent (as its own header claims: int8 x int8
-> int32 is exact). And the GEMM portion did shrink (~67 -> ~20 ms/layer).

BUT end-to-end is unchanged (3493.8 vs 3485.9 ms/tok) because the FFN cost is the
**host pack on cache MISS (~70 ms/layer: 8 experts x ~8.5 ms)**, which persists: routed
experts change per token, so `exp_cache[l]` keeps missing. Addendum 17's "the pack lever
is already implemented" was therefore wrong for a *changing* routing set — the cache only
helps repeated experts.

### Real remaining lever
Prepack ALL 256 experts per layer at init (or SIMD the host dequant): with the pack
removed, FFN -> ~20 ms/layer, giving 40 x (8.7+5.6+5.5+20) ~= 1.6 s/tok ~= 0.6 tok/s.
Cost: ~30-41 GB RAM (the engine's own comment) and a long init. That is the same lever
addendum 12 identified; addendum 17 wrongly retired it.

### Addendum 19 — AMENDED: the host pack is NOT quant_slice-bound either (SIMD attempt neutral)

Addendum 18 attributed the ~70 ms/layer FFN cost to the host pack on cache miss, and the
engine's own comment blames `quant_slice` ("~32 ms/expert ... dominant cost of a cache
miss"). So I AVX2-vectorised `quant_slice`: contiguous `j` inner loop (the old one
strided by N), 8-wide max/scale/quantize, half-away rounding reproduced exactly via
`copysign(floor(|v|+0.5), v)`.

Result: **bit-identical output** (the 7 decode tokens were unchanged: 154742, 16023,
136614, 25238, 32858, 248050, 184997) but **no speed-up** — avg `q` 7.31 ms vs 6.81 ms
scalar, and e2e 3681 vs 3494 ms/tok. So `quant_slice` is not the wall either; the SIMD
change was reverted (dead complexity, slightly worse measured).

Per-expert miss accounting on the 35B (avg over 1649 misses): `deq` ~2.9-3.6 ms,
`q` ~6.8-7.3 ms => ~10.5 ms of the ~18 ms/expert; the remaining ~7 ms is the BO/concat
write and call overhead. The pack is therefore spread across dequant + quantize + BO
writes, with no single dominant vectorisable hot spot.

Honest position after addenda 17/18/19: the engine's ~3.5 s/tok (0.29 tok/s) is bound by
the aggregate host-side expert handling (~120 ms/layer of pack) plus ~20 ms of NPU GEMM
per layer, and none of the three levers tried (M=1 kernels, cache, SIMD quant) moved it
materially. The durable deliverables of this line of work are the M=1 kernels (built,
wired, bit-identical) and the M=128-baked-kernel finding.

## Addendum 20 — EXPERT PREPACK (all 256/layer): FFN 70 -> 29 ms/layer, decode 3.49 -> 1.83 s/tok

The lever addendum 12 identified and addendum 17 wrongly retired, now actually done — using
the engine's own existing warm path (`NPU_WARM_EXPERTS=<stats> NPU_WARM_TOP=N`, which packs
the top-N routed experts per layer at init through `moe_pack_experts`).

Generate an all-expert stats file (the format is `layer expert count`):
```
for l in 0..39: for e in 0..255: print(f"{l} {e} 1")   # 10240 lines
```
Run:
```
NPU_MOE=1 NPU_MOE_FUSED=1 NPU_MOE_SMALL_M=1 NPU_WARM_EXPERTS=/tmp/route_all256.txt NPU_WARM_TOP=256 \
  npu_engine_qwen3_6_moe_35b model.q4nx 8 /tmp/ids35b.txt
```
Result:
```
  warm: 10240 experts pre-packed in 148195 ms
  [1] 154742  [2] 16023  [3] 136614  [4] 25238  [5] 32858  [6] 248050  [7] 184997
  [decode-stage] QKV=8.4 attn=5.5 O=5.4 FFN=28.9 ms/layer
  === 1826.5 ms/tok (0.55 tok/s) ===
```

| metric | addendum 18 (M=1, cold) | addendum 20 (M=1 + warm 256) |
|---|---|---|
| FFN / layer | ~70 ms | **~29 ms** |
| decode | 3.49 s/tok (0.29 tok/s) | **1.83 s/tok (0.55 tok/s)** |

So: **~1.9x**, tokens bit-identical, at the cost of ~148 s init and ~30 GB RAM (10240
expert slices x ~3 MB). The MoE FFN is no longer the whole story: it is still the largest
per-layer term (~29 ms, of which ~20 ms is the now-M=1 GEMM and ~9 ms the concat-BO
memcpy/sync of 18.9 MB/layer), with QKV 8.4 + attn 5.5 + O 5.4 alongside.

Steady state is now **0.55 tok/s** — 1.9x the 0.29 of the M=128 baseline measured in
addendum 10c, and within ~1.3x of the ~0.7 tok/s figure the objective starts from (though
far below the dense-class target). Remaining levers, in order of size: the per-layer
18.9 MB concat-BO sync/memcpy, then the M=1 GEMM's own ~10 ms/launch.

### Addendum 21 — QKV/O M=1 kernels HURT (reverted); the MoE m1 is a small win; the warm is the real gain

Built `_m1` variants for QKV (K=2048 N=8192) and O (K=4096 N=2048) with the same generator,
and patched `init_i8` to prefer them with `MD=1`. Measured against the addendum-20 config
(warm all-256), 8 tokens, tokens bit-identical in every arm:

| config | QKV ms | O ms | FFN ms | decode ms/tok |
|---|---|---|---|---|
| warm, no small-M | 9.1 | 5.6 | 30.0 | **1872.7** |
| warm + MoE m1 (GUSGU/DSD) | 8.4 | 5.4 | 28.9 | **1826.5** |
| warm + MoE m1 **+ QKV/O m1** | 11.5 | 6.4 | 30.0 | **2004.7** |

So:
- **QKV/O `_m1` is ~9% SLOWER** (QKV 8.4 -> 11.5 ms, O 5.4 -> 6.4) -> reverted (`init_i8`
  patch removed; the `moe_ctx` small-M patch kept).
- the **MoE (GUSGU/DSD) `_m1` is a genuine but small win** (~2.5%: 1873 -> 1827).
- the **real gain remains the expert warm/prepack** (addendum 20): 3486 -> 1827 ms/tok.
- This vindicates the engine's own note "*no perf win*" for the small-M path (its
  "*garbage decode*" half was wrong — every arm is bit-identical).

FINAL steady state on this lane: **~1.83 s/tok = 0.55 tok/s**, 1.9x the 0.29 of the
M=128 cold baseline, tokens identical, cost ~2-2.5 min init (warm) and ~30 GB RAM.

### Addendum 22 — three more engine/NPU hazards banked (from @agent-7f1cce's fk-3 work)

1. **Same group_id in a DIFFERENT hw_context is a DIFFERENT buffer.** A multi-launch
   chain where launch B has its own A buffer (XRT group 3) must fill BOTH A's; nothing
   warns you, and the symptom is a legitimate-looking all-zero output (x=0 -> silu=0 ->
   D=0*W_D+h=0).
2. **Device-state starvation of a dataflow kernel.** Measured: launch A 63 ms healthy,
   launch B 792 ms -> zeros, and launch B ALONE (bench-style pseudo-random inputs,
   launch A's context never created) still 744 ms -> zeros. The bench runs on an idle
   device; the driver runs immediately after that engine's own NPU work (dequant +
   bf16mm + runlist contexts live). With many resident hw_contexts serialising at
   submission granularity and no fine preemption, a kernel waiting on its own
   objectfifos while other contexts' dispatches occupy the array makes no progress:
   enormous wall time, NO error, NO timeout, NO XRT failure, zeroed buffers — one
   mechanism explaining both the 60x slowdown and the zeros. Proposed fix: quiesce the
   device around the fused layer before submitting.
   NOTE for this lane: this does NOT explain the 35B runtime ERT (addendum 14 measured
   that with accel0 fully quiet and no other contexts) nor the 35B layer NaN (a NaN, not
   zeros) — those remain as recorded.
3. `-bf16out` is argparse `store_true` (BF16OUT=1), NOT positional; omitting it silently
   leaves an f32 C store. xclbin size discriminates: 37,210 B = bf16-out (correct),
   20,474 B = f32-out (wrong).

### Addendum 23 — contention theory REFUTED by @agent-7f1cce (banked), and a method lesson

@agent-7f1cce retracted the device-contention/starvation explanation they had floated for their
fused layer's all-zero output: after waiting 222 s for a verified-idle accel0 (no PID holding
/dev/accel/accel0) they re-ran and got numbers identical to the contended run to 3 sig figs
(launch A 63.43 -> 63.71 ms, launch B 792.43 -> 793.13 ms, CD still all zeros nonzero 0.000,
launch A's C and post-RoPE bQ both nonzero 1.000 with identical maxabs). So the behaviour is
DETERMINISTIC, and the 63/793 ms are simply what those kernels cost here — timing was a red
herring; only the zeros matter.

Consequence for THIS lane: my addendum-21 caveat that the absolute 1.83 s/tok "may be inflated
by contention" is now UNSUPPORTED — the one controlled idle-vs-contended measurement we have
found no contention sensitivity at all. A quiet re-run is still worth doing to confirm it for
the GEMM path specifically (this measurement is of a different, stalling kernel), but the
number should be treated as real until then, not as an upper bound.

Method lesson banked: verify pass/fail by counting `func.call` occurrences, never `func.func`
declarations, and never trust a TRUNCATED grep for a pass/fail verdict (their "NOQKV drops the
QKV phases" conclusion happened to hold, but their grep had been truncated at 24/41 matches;
proper call counts: nq_acc_mac=3, nq_acc_store_bf16=2, nq_acc_store_f32=1 -> exactly O-proj +
GU + D, a QKV phase would need a 4th mac and a 3rd bf16 store).

### Addendum 24 — NPU pre-flight recipe adopted; the lock GAP that matters for this lane

@agent-7f1cce supplied the cheap device pre-flight and flagged a real gap:

- pre-flight: `for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q accel0 && echo $p; done`
  (non-empty => any NPU number taken now is suspect). Wrapped as
  `benchmarks/npu-device-preflight.sh` (prints holders + their cmdline + lock state).
- **the gap: the engine takes /tmp/1bit-npu-device.lock itself, but STANDALONE BENCHES and the
  XCLBIN GENERATORS do NOT.** For this lane that is exactly `npu-infer/tools/moe_smoke` (the
  runlist harness) and the `aiecc` builds — those can silently overlap someone else's run.
  So: run the pre-flight before recording any number from `moe_smoke` or a generator build.
- They also confirmed the discriminator this lane rests on: **zeros-with-no-error was their
  starvation-shaped symptom and it turned out to be DETERMINISTIC**, so the 35B ERT
  reproducing on a VERIFIED-QUIET device plus a NaN (not zeros) is a genuinely different,
  stronger signal. Keep both on the books as non-contention.
- zr1/zaya1-8b has re-spawned at least twice in the last hour, so waiting for a
  `timeout 1400` run to expire is futile; the quiet window has to be caught, not waited for.

### Addendum 25 — CORRECTION to addendum 23: my retraction was too broad

@agent-7f1cce pushed back correctly, and addendum 23's framing was wrong in one direction:

- Their idle-vs-contended result answers a **BIT-EXACTNESS / pass-fail** question ("does the
  output stay zeros under load?"). It does NOT generalise to a **throughput** question.
- This lane's numbers ARE timings, and **timing is exactly what contention moves** — plausibly
  by a factor of several. So "load is not the variable" is TRUE for a pass/fail question like
  theirs and FALSE for a tok/s question like mine.
- Therefore the only thing addendum 23 should have withdrawn is the implication that my runs
  might explain THEIR zeros (they don't). The **contention-sensitivity of this lane's absolute
  number stays OPEN**, and my original instinct to re-measure on a quiet device was right.

Acting on that: zr1's `timeout 1400` expired, the /proc fd scan came back EMPTY (lock file
present but unheld), and the quiet best-arm run was started immediately (PID 700860,
`NPU_MOE=1 NPU_MOE_FUSED=1 NPU_MOE_SMALL_M=1` + all-256 warm, 8 tokens) with the device
verified busy under it. Compare target: contended **1826.5 ms/tok** (addendum 21).

Operational note: the engine takes `/tmp/1bit-npu-device.lock` ITSELF, so I deliberately did
NOT wrap it in another `flock` (two exclusive holders on one file is a self-inflicted stall).
The /proc fd scan is the entry condition that matters; the lock file merely existing does not
mean it is held.

### Addendum 26 — QUIET-DEVICE re-measurement SETTLED: contention was NOT inflating this lane

Took the first genuinely idle accel0 window (zr1's `timeout 1400` expired; /proc fd scan EMPTY,
lock file present but unheld — verified per addendum 24), ran the addendum-21 best arm
(`NPU_MOE=1 NPU_MOE_FUSED=1 NPU_MOE_SMALL_M=1` + all-256 expert warm, 8 tokens), twice:

| arm | decode ms/tok | tokens |
|---|---|---|
| contended (addendum 21, n=1) | **1826.5** | identical |
| QUIET sample 1 (n=1) | **1903.4** | identical |
| QUIET sample 2 (n=2) | **1901.8** | identical |
| quiet per-layer | QKV 8.3-8.7, attn 5.2-6.0, O 5.3-5.5, FFN 30-31 ms | |

Conclusions:
- **The quiet figure is the trustworthy one: ~1.90 s/tok = 0.53 tok/s**, and it is TIGHT
  (two independent quiet runs 1903.4 / 1901.8, i.e. +/-0.1%).
- **Contention was not inflating this lane.** If anything the contended run was FASTER (1826.5),
  so any load effect here is at or below run-to-run variance and is NOT a penalty. The
  addendum-21 caveat is therefore resolved in the *opposite* direction to my worry: the number
  was not an optimistic contention-inflated artifact.
- @agent-7f1cce's correction to addendum 23/25 stands as the reason this had to be measured
  rather than argued: their pass/fail insensitivity to load does not transfer to a timing
  question, so the only way to close it was to take a quiet window.
- Tokens are bit-identical in every arm (154742, 16023, 136614, 25238, 32858, 248050, 184997),
  so the quiet re-run also re-confirms correctness on a clean device.

FINAL for this lane: **~1.90 s/tok (0.53 tok/s) steady-state M=1 decode** on an idle Strix
Halo, from 3.49 s/tok cold -- a 1.84x gain, entirely attributable to the expert prepack/warm.
The objective's dense-class target (~88 tok/s) remains out of reach: the runtime's whole-layer
per-ctx ELF path is structurally broken (deterministic ERT + NaN).

### Addendum 27 — feasibility inventory for building the 35B whole-layer ELF from OUR OWN kernels

The objective's core work ("extend the single-launch whole-layer per-ctx ELF runlist path to
Qwen3.6-35B-A3B") is now BUILDABLE without the runtime. Reuse of the lib's ELF stays rejected
(addenda 6-14), but every piece needed to author our own exists:

| 35B layer op | kernel status |
|---|---|
| QKV (K=2048 N=8192) | M=1 built (addendum 21 artifacts) |
| O (K=4096 N=2048) | M=1 built |
| MoE GUSGU (K=2048 N=9216) | M=1 built, WIRED, bit-identical |
| MoE DSD (K=4608 N=4096) | M=1 built, WIRED, bit-identical |
| attention | xclbins present (17 variants) |
| SiLU/activation | present (8) |
| RMSNorm | MISSING for 35B -> generators exist in fk-3 (`n1_rms_norm.py`, `n1_fused_rmsnorm_qkv.py`, `rms_norm_*.cc`) |
| RoPE | not a kernel upstream (a table / host-side in fk-3); fold or keep host |

WHY THIS IS WORTH BUILDING — where the current ~50 ms/layer actually goes. The engine's hot path
is, per kernel: `quantize_async(A)` (HOST quantize + `sync(TO_DEVICE)`) -> launch ->
`sync(FROM_DEVICE)` readback (npu_engine_universal.cpp:1621-1646, 1651-1654). That host
quantize+2-sync cycle repeats ~7x per layer, and it -- not the arithmetic -- dominates: the M=1
QKV GEMM does ~33.5 MFLOP, i.e. single-digit MICROSECONDS of math, against a measured 8.7 ms
stage. Same story for FFN (30 ms) where the prepack already removed expert packing. So the
per-layer time is host-side per-kernel overhead, which is EXACTLY what one runlist submit/token
collapses. This is the objective's premise and it survives scrutiny.

NEXT ACTION for the next run: author the 35B whole-layer per-ctx ELF from the above (QKV/O/attn/
GUSGU/DSD M=1 + an RMSNorm for the 35B hidden size generated with fk-3's generator), drive it
from npu-infer/tools with ONE xrt::runlist per token, and validate against the engine's
bit-identical token stream (154742, 16023, 136614, 25238, 32858, 248050, 184997).
Pre-flight the device first: benchmarks/npu-device-preflight.sh (addendum 24).

### Addendum 28 — measurement discipline adopted from @agent-7f1cce's ninth retraction

Their rule, which I am adopting as a gate on this lane's work: **validate the instrument before
trusting its reading, and the validation must have an independently-derived expected answer.**
They built a black-box one-hot probe, got a plausible number, propagated it into two
conclusions, and only afterwards found its correlation with the correct row was 0.08 where
1.0000 is required — the instrument had never measured the thing it was named for. Nine
retractions in their lane share that one root.

Applied to MY instruments on this lane (audit, so far clean but not uniform):
- decode-stage timers: VALIDATED by construction -- the per-layer sum (QKV 8.3-8.7 + attn
  5.2-6.0 + O 5.3-5.5 + FFN 30-31 = ~50 ms) x 40 layers = ~2.0 s matches the independently
  measured end-to-end 1.90 s/tok. Two routes to the same number.
- the M=1 kernel arms: VALIDATED by the bit-identical token stream, i.e. against an expected
  answer derived from the pre-existing verified path, not from the new arm.
- the -bf16out trap (addendum 22) is the same failure class: a bench reading an f32 buffer as
  bf16, i.e. an instrument measuring something other than what it was named for.
RULE GOING FORWARD, binding on the addendum-27 ELF work: the new whole-layer ELF must be
checked against an INDEPENDENTLY-DERIVED reference (a CPU reference for the layer / the
engine's own bit-identical tokens), never against a statistic produced by the artifact under
test. A "looks plausible" number is not evidence.

Cross-confirmed between two lanes, independently: on this box a DETERMINISTIC failure on a
verified-quiet device is a real defect, not load (my ERT+NaN, their fused-path failure, both
bit-for-bit reproducible on idle accel0). The idle-device check is what makes a determinism
claim meaningful.
Operational note banked: their probe code is env-gated and inert by default, but
**NPU_GEMM_PROBE2 currently HANGS when enabled** (documented, unfixed) -- do not set it.
