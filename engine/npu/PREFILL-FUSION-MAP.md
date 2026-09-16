# PREFILL-FUSION-MAP — launch structure of the native bf16 prefill (dense Qwen3 0.6B)

Goal `mtygjrxl-9lbnet`, task ra-1. This is the grounding map for the
"re-architect the fused prefill onto the runlist/ELF mechanism" direction the
user selected. Line refs are to `engine/npu/src/npu_engine_universal.cpp`
(the bf16 prefill loop) unless noted.

## 1. The three prefill-relevant paths on this tree

| path | mechanism | launches/layer | RMSNorm/RoPE/SiLU | speed |
|---|---|---|---|---|
| **native bf16 per-op** (`NPU_PREFILL_BF16=1`) | FLM `mm.xclbin` GEMM + `dequant.xclbin` + captured `attn_mha_*.elf` | **5 device + 4 host passes** | **HOST** (f32→bf16 round-trips) | **1945 tok/s @1k** |
| **runlist/ELF whole-layer** (`NPU_RUNLIST=1`, decode) | FLM `layer.xclbin` + per-ctx ELF (`gen_layer_seq`) | ~1 runlist submit/token | **in-kernel** (kernel reads host-written cos/sin table) | 86 tok/s decode, ~71 tok/s prefill (M=1) |
| **FLM-orchestrated** (`NPU_FLM_PREFILL=1`) | `libqwen3_npu.so` `qwen3_npu::prefill` (closed) | batched whole-layer | in-kernel | FLM's published 1494 tok/s |

The objective's "~655 tok/s cap" premise is refuted (see commit `7f827f02d`):
the un-fused native bf16 path already measures 1945.5 tok/s @1k, ~30% above the
1494 published bar. The fusion is therefore a **structural** goal (in-kernel
ops + ~1 launch), not a speed-gap goal.

## 2. Precise native bf16 per-layer launch map (loop `for l in 0..NC`, :4656)

Per layer, in order:

1. **HOST input RMSNorm** — `rn_bf16(&bA[pi*H], &bh[pi*H], in_n[l], H)` (:4663).
   f32 RMSNorm of the residual stream, written to bf16 `bA`. No device launch.
2. **DEVICE QKV GEMM** — `bf16mm_gemm_launch(Wqkv[l], H, qkvn, …)` (:4709),
   one `mm.xclbin` GEMM (N=qkvn=4096), walked in 128-row blocks (nblk = ⌈npt/256⌉).
3. **HOST q/k-norm + RoPE** — `qk_norm_pi` lambda (:4685): per-head RMSNorm of
   Q/K, `ra()` RoPE on Q and K, K/V scatter into `bKv` + `kv_caches`. No device launch.
4. **DEVICE attention** — `bf16mm_attn(bA, bActQ, bKv)` (:4830), one captured
   `attn_mha_*.elf` launch covering all npt query rows.
5. **DEVICE O GEMM** — `bf16mm_gemm_launch(Wo[l], qout, H, …)` (:4897) + host residual
   `bh = bsb + boo` (:4903 region).
6. **HOST post-attn RMSNorm** — `rn_bf16(&bA[pi*H], &bh[pi*H], pa_n[l], H)` (:4930).
7. **DEVICE GU GEMM** — `bf16mm_gemm_launch(Wgu[l], H, 2*IM, …)` (:4951),
   gate+up in ONE GEMM (N=2·IM).
8. **HOST SiLU(gate)·up** — `gv * sigmoid_fast(gv) * up` → `bGu` (:4969).
   This is the gated SwiGLU, NOT a plain output activation.
9. **DEVICE D GEMM** — `bf16mm_gemm_launch(Wd[l], IM, H, …)` (:4983) + host residual
   `bh = bsb + bdw`.

So: **5 device launches** (QKV, attn, O, GU, D) + **4 host passes**
(input-norm, q/k-norm+RoPE, post-norm, SiLU) + 2 residual adds ≈ the objective's
"~9-launch/layer". At npt=256 the device side is exactly 5 launches; at npt=1024
each GEMM walks 4×128-row blocks so the raw launch count is ~17, but the *shape*
is 5 distinct ops.

Host-op cost is the `conv+other` term in the prefill printf (:5010):
`Prefill: … [GEMM tg, attn ta, conv+other tc]`.

## 3. What the fast GEMM/attn kernels are (and are not)

- `mm.xclbin` = FLM's pure bf16 GEMM `A×W→C`, closed-source. Driven via the
  **open** `Gemm::generate_seq(seq, M, K, N, woff, ADD_BIAS, OUTPUT_MODE, bias_off[, ooff])`
  (`third_party/FastFlowLM/src/include/modules/gemm.hpp`).
- `OUTPUT_MODE` supports **`NO_Activation=0`, `GeLU=1`, `SiLU=2`** — i.e. an
  in-kernel element-wise output activation exists, but it is **not** the gated
  `SiLU(gate)·up` the FFN needs, so the current GU path uses `NO_Activation`
  and does SiLU on the host (`npu_engine_bf16_mm.h` :596/:649/:736).
- `dequant.xclbin` = Q4NX→bf16 (`Dequant::generate_dequant_q4_1_seq`).
- `attn_mha_*.elf` = captured attention ELFs, expect **pre-RoPE'd** Q/K (the host
  applies q/k-norm + RoPE; the kernel does NOT — comment :4740).

There is **no open RMSNorm or RoPE module** in FLM's `modules/` set (only
`dequant, embedding, gemm, lm_head, sampler`). RMSNorm/RoPE-in-kernel therefore
requires either the closed `layer.xclbin` (whole-layer, M=1) or the open sequence
symbols listed below.

## 4. The open prefill sequence symbols (the runlist/ELF lever)

`npu-infer/src/flm_bridge.cpp` (kept as reference; engine integration was
removed 2026-08-15 per `engine/npu/CMakeLists.txt:32`) shows FLM's own batched
prefill is reconstructible from **open** exports of `libqwen3_npu.so`:

```
_send_rope_rms_weights  (Impl::_send_rope_rms_weights)  — RoPE cos/sin + RMSNorm weights to device
_send_rms_weights       (Impl::_send_rms_weights)       — RMSNorm weights to device
gen_dequant_seq         (qwen3_npu_sequence::gen_dequant_seq)
_send_x                 (Impl::_send_x)                 — activations
_move_weights           (Impl::_move_weights)
Gemm::generate_seq(…, activation_type, …)               — GEMM with fused output activation
gen_mha_engine_seq      (qwen3_npu_sequence::gen_mha_engine_seq(seq, L_begin, L_end))
```

FLM prefill pipeline order (from `full_pipeline.cpp`, reproduced in
`flm_bridge.cpp::gen_gemm_instrs`):
`send_rope_rms_ → send_rms_ → gen_dequant_ → send_x_ → move_weights_ →
gemm_generate_seq_(activation)` then `cmds2seq_`; attention via
`gen_mha_engine_seq(seq, L_begin, L_end)` over the token range.

This is the mechanism that makes the objective reachable **on the fast path**:
RMSNorm + RoPE + (element-wise) SiLU are pushed into the same instruction
sequence as the GEMM/attention, eliminating the host f32→bf16 round-trips and
the per-op launches — exactly what "~1 fused layer launch" requires, and it is
the path FLM's 1494 tok/s number comes from.

## 5. What the re-architecture work is (ra-2/3/4)

- **ra-2** — fuse the input RMSNorm into the QKV GEMM sequence (use
  `_send_rms_weights` + the norm-bearing prelude rather than `rn_bf16` host pass);
  byte-identical output vs the host-norm path.
- **ra-3** — fuse RoPE (q/k-norm + rotation, via `_send_rope_rms_weights` +
  `gen_mha_engine_seq` over the token range) and SiLU (gated FFN; note the
  element-wise `OUTPUT_MODE=SiLU` does NOT match gated SwiGLU — the gated form
  needs the whole-layer kernel or a host pass, to be established in ra-3).
- **ra-4** — batch the per-layer ops into ~1 `xrt::runlist` submit/layer and
  measure `flm_parity.sh --ctx-k 1024`; bar = 1494 tok/s (objective) with token
  parity preserved; the status-quo per-op path is 1945.5 tok/s.

Open question that ra-2 must settle first: whether the norm-bearing GEMM
sequence (`_send_rms_weights` + `Gemm::generate_seq`) is byte-equivalent to the
host `rn_bf16` at the 0.6B shapes, or whether FLM's norm sequence assumes the
whole-layer `layer.xclbin`'s kernel ABI rather than the standalone `mm.xclbin`.

## 6. RESOLVED (ra-2 feasibility): the xclbin designs are disjoint, not modes

`xclbinutil` on the FLM set (`amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/`)
shows every kernel — `mm.xclbin`, `layer.xclbin`, `dequant.xclbin`, `attn.xclbin`
— exposes the SAME host ABI (`MLIR_AIE:MLIRAIE`, 5 data BOs `bo0..bo4` +
`instr`/`ninstr`/`opcode`). The difference is entirely in the compiled AIE
**design** (AIE_PARTITION):

- `mm.xclbin`   = pure bf16 GEMM datapath (`A_bf16 × W_bf16 → C_bf16`), **no
  norm/rope/silu logic in the tiles**.
- `layer.xclbin` = whole-layer datapath (norm + QKV + RoPE + attention + O +
  norm + gated SiLU + D all in the tiles), **M=1 per token**.
- `attn.xclbin`  = MHA-engine (batched attention over a position range), driven
  by `gen_mha_engine_seq(seq, L_begin, L_end)`.
- `dequant.xclbin` = Q4NX→bf16.

The instruction streams are **design-specific**: `Gemm::generate_seq` (libgemm)
emits for `mm.xclbin`, `qwen3_npu_sequence::gen_layer_seq`/`gen_mha_engine_seq`
emit for `layer.xclbin`/`attn.xclbin`. `_send_rms_weights` is an
`Impl`-method of `qwen3_npu_sequence` — i.e. part of the **layer** sequence, not
the mm GEMM. So a norm-bearing `_send_rms_weights` + `Gemm::generate_seq` mix
(the `flm_bridge.cpp` sketch) would be an invalid mixed-design stream — which is
consistent with that bridge having been removed from the engine build
(`engine/npu/CMakeLists.txt:32`).

**Consequence for the objective**: "fuse RMSNorm/RoPE/SiLU in-kernel" is only
reachable via `layer.xclbin` (whole-layer, in-kernel everything) or FLM's closed
`qwen3_npu::prefill`. `layer.xclbin` is M=1, so a *fast batched* fused native
prefill requires reconstructing FLM's batched whole-layer orchestration from the
open symbols (`gen_layer_seq` + `gen_mha_engine_seq` + the BO/npu_app plumbing
that `RuntimeLayerEngine` already implements for decode) — a real but bounded
effort, NOT a flag flip on `mm.xclbin`. The per-op bf16 path's speed (1945 tok/s)
is the bar to preserve.

## 7. RESOLVED (ra-2/ra-3 gate): no fast fused-prefill kernel exists for dense Qwen3

Inventory of FLM's shipped xclbins (`amd-oss/fastflowlm/src/xclbins/`):

| family | xclbin set |
|---|---|
| Qwen3-0.6B / 1.7B / 4B / 8B | `attn.xclbin` + `layer.xclbin` + `mm.xclbin` (+`dequant.xclbin` on 0.6B) — **no prefill kernel** |
| Qwen3.5-*/Hy-MT2/GateDeltaNet (newer) | + `fused_prefill.xclbin` / `GateDeltaNet_prefill.xclbin` — a dedicated **batched fused prefill** kernel |

FLM only ships a `fused_prefill.xclbin` (the whole-layer batched prefill, in-kernel
norm/RoPE/SiLU) for the **newer** families. Dense Qwen3 (the objective's target)
never got one — its FLM prefill is the per-op `mm.xclbin` + `attn.xclbin` path
with the norms done in the closed `libqwen3_npu.so` (which carries AVX2 host
intrinsics: `<immintrin.h>` in `qwen3_npu.hpp`). So FLM itself did **not** move
prefill RMSNorm/RoPE/SiLU in-kernel for dense Qwen3.

**This closes the re-architecture gate**: there is no fast fused-prefill kernel
to "re-architect onto" for dense Qwen3-0.6B. The three candidate routes are all
blocked or non-winning:
1. `mm.xclbin` — pure GEMM, no norm/RoPE/SiLU tiles (section 6).
2. `layer.xclbin` — in-kernel everything but **M=1** (decode-only; `gen_layer_seq(seq,L)` is per-position).
3. Building a `fused_prefill.xclbin` for 0.6B via the open generators — this is
exactly fk-3's route, which hit the depth-2-objectfifo descriptor hard floor at
~56x slower than the per-op path (see `fk-3`/`fk-4` commit history).

The one remaining question before calling this gate closed is whether
`layer.xclbin` can be driven batched (M>1) by an instruction sequence FLM does
not expose as an open symbol — the batched attention `gen_mha_engine_seq` exists
for `attn.xclbin`, but no batched *layer* sequence symbol exists. That is the
single experiment ra-2 needs (capture `qwen3_npu::prefill` on-box and read its
runlist/xclbin usage) and it requires the device, which is currently busy with
`flm serve`/`flm run` processes from other lanes.

## 8. RESOLVED (disassembly): FLM's own dense-Qwen3 prefill is per-op + HOST norms

Disassembly of `libqwen3_npu.so` settles it without the device.
`qwen3_npu::Impl::prefill` → `_prefill_with_mm`, whose call graph is:

- `_rms_norm(...)`   — **HOST** RMSNorm (local fn in libqwen3_npu.so)
- `_rope_rms(...)`   — **HOST** RoPE + q/k-norm
- `Gemm::generate_seq(...)` — `mm.xclbin` batched GEMM
- `MHA::generate_mha_sequence(...)` + `MHA::get_chunk_size()` — `attn.xclbin` batched chunked attention
- `fill_kv_cache(...)` — host KV fill
- `xrt::runlist::add/execute/wait` — batches the per-op kernels
- `npu_app::operator()` / `create_run` — per-op kernel launch

`gen_layer_seq` (the `layer.xclbin` whole-layer sequence) is **not** called by
prefill — it is decode-only. So FLM's own fast prefill for dense Qwen3 is
**exactly the per-op `mm.xclbin` + `attn.xclbin` + host-norm/RoPE/SiLU
structure** that the native bf16 path (1945 tok/s) already reimplements. FLM
never moved prefill norm/RoPE/SiLU in-kernel for this family.

**The objective therefore asks for a kernel that does not exist in FLM's dense-Qwen3
set and that FLM itself never built.** Every route to "fused + fast" is closed:
`mm.xclbin` (no norm tiles), `layer.xclbin` (M=1 decode-only), open generators
(fk-3 depth-2-fifo hard floor, 56x slow), and `fused_prefill.xclbin` (only newer
families). The re-architecture direction is definitively blocked for dense
Qwen3-0.6B.

## 9. DECODE PATH — the runlist/ELF whole-layer is the fused ~1-launch path (ra-2)

User redirect (2026-09-16): apply the fusion effort to the **decode** path, where
the runlist/ELF whole-layer genuinely is the fast in-kernel ~1-launch structure.
Code refs are `npu-infer/src/runtime_layer.cpp` + `engine/npu/src/npu_runlist_bridge.cpp`.

**Launch structure (per token)** — `RuntimeLayerEngine::build_runlist` builds ONE
`xrt::runlist` containing 28 layer runs + 1 lm_head run (29 runs), executed as a
single submit (`execute_runlist`). So the decode is already **~1 launch/token**
for the WHOLE model, not per layer. Double-buffered slots (`build_runlist(sa/sb)`)
overlap the next token's runlist build against the current device exec.

**In-kernel vs host (per forward)** — the layer kernel (`layer.xclbin`) reads five
BOs and computes the whole layer in-kernel:

| BO | arg | contents |
|---|---|---|
| `bo_act_` | 3 | hidden state (bf16, H) |
| `weight_bos_[L]` | 4 | dequant'd per-layer weights (QKV/O/GU/D) |
| `i5_bos_[L]` | 5 | input + post-attn RMSNorm weights |
| `i6_bos_[L]` | 6 | RoPE cos/sin (host-written) + q/k-norm weights |
| `kv_bos_[L]` | 7 | per-layer KV cache |

So **RMSNorm, q/k-norm, RoPE rotation, gated SiLU and the residuals are all
in-kernel** in `layer.xclbin`. The only host passes per forward are: `embed`
(act write), `apply_rope` (cos/sin table → `i6[0:128]`, 256 B sync × 28 layers),
`build_runlist` (overlapped), and `argmax_logits`. The kernel does NOT compute
RoPE internally — it reads the host-written cos/sin table (comment in
`update_rope_i6`).

**Measured** (`benchmarks/RESULTS-runlist-true-native-dense-qwen3-2026-09-16.md`,
and addendum 147):

| context | native decode | FLM on-box | gap |
|---|---:|---:|---:|
| short prompt (~10 tok) | **86 tok/s** | ~75 | **+14.6%** |
| 2k-token prompt | 67 tok/s | 74.92 | **−10.6%** |

At short context the fused decode beats FLM; at 2k context it trails by 10.6%
because the per-token attention cost scales with the KV prefix length (2k keys
vs 10). That context-length scaling — not the fusion structure — is the open
decode gap (ra-3/ra-4).

## 10. DECODE HOST-OP QUANTIFICATION (ra-3)

Already measured (`benchmarks/RESULTS-npu-prefill-parity-2026-09-12.md` §"Decode
breakdown" and §"Precise forward() breakdown"): `NPU_FWD_TIMING=1` on 0.6B gives
`[fwd] rope=0.51 build=0.78 exec=12.53 total=13.87 ms` steady-state. Decomposed:

| piece | ms/tok | overlappable with current exec? |
|---|---:|---|
| device exec (29 runs, 1 runlist) | ~12.5 | — (it is the device) |
| runlist build (29×8 set_arg) | 0.8 | yes — **already overlapped** (double-buffered sa/sb slots) |
| RoPE writes (28 BO writes + 28 syncs) | 0.5 | yes, **needs a double-buffered i6** (not yet done) |
| argmax (logits sync 304 KB + scan) | ~1.5 | no (needs this token's logits) |
| embed (memcpy + sync) | ~0.5 | no (needs this token's argmax) |

Already done and recorded: argmax was vectorised (OpenMP over a monotonic bf16
key) and was **neutral** — the scan was not the bottleneck; the 304 KB logits
`sync` is. Build overlap via the double-buffered runlist slots is already in
`npu_runlist_bridge.cpp` (raised decode 62 → 67 tok/s).

**Remaining actionable fix** — overlap the RoPE write by double-buffering `i6_bos_`
(28 → 56 BOs, one per runlist slot): `apply_rope(next_ctx, slot)` writes the next
slot's table BEFORE `wait_runlist`, so the 28×256 B syncs overlap the current exec.
Expected ~0.5 ms/tok (≈3-4% decode), taking ~67 → ~69-70 tok/s. The argmax+embed
(~2 ms) sit in the strict `logits→argmax→embed→next-exec` chain and are inherent.

### 10b. RESULT (ra-5, landed 2026-09-16): the rope overlap is a NO-OP

The i6 double-buffer was implemented and committed (`48a5dd4b2`, token parity
preserved: 16 tokens byte-identical). Clean measurement (numpy process cleared,
only the idle 35B serve present): decode is **12.2 ms/tok = 82 tok/s @1k — exactly
the pre-change number**, not the ~84-85 expected. The ~0.4 ms/token rope write was
NOT actually on the critical path: `xrt::ext::bo::sync(XCL_BO_SYNC_BO_TO_DEVICE)`
queues behind the in-flight runlist on the device's command queue, so moving
`apply_rope` before `wait_runlist` does not hide it — the sync still waits for the
executing runlist. Conclusion: overlapping the rope write via double-buffering
buys nothing on this driver; the decode critical path is `exec → argmax → embed`,
and the rope sync is effectively serial with exec regardless of slot. The change
is correctness-neutral and kept, but it is not a speed win.

**Honest ceiling** (from the same doc): best case ≈ 14.7 ms ≈ **68-70 tok/s** vs
FLM's 73.58-74.92 — the residual is host logits-sync/embed plus device time that
FLM shares. The i6 double-buffer is a correctness-sensitive hot-path change (the
wrong slot silently corrupts decode), so it is implemented only when the device
is free to validate token parity.

## 11. DECODE MEASUREMENT — gap closed @1k, open @2k (ra-4)

Authoritative scorecard (`benchmarks/RESULTS-npu-prefill-1k-SOLVED-2026-09-12.md`
§"Objective scorecard", averaged over repeat runs for variance):

| metric | native | FLM on-box | verdict |
|---|---:|---:|---|
| decode tok/s @ 1024 ctx | **82** (79.3-82) | 77.9 | ✅ **+5%** |
| decode tok/s @ 2088 tok | 67 | 72.1 | ❌ −7.1% |

Root cause is already localised (same doc §"Native decode degrades faster…"):
native's per-token cost grows **~18%** from 1k→2k ctx (12.2 → 14.3 ms/tok) while
FLM's grows **~4%**. That is the `layer.xclbin` attention reading a longer KV
prefix per token — a **device-side / closed-source** cost, not the fusion
structure and not the host passes quantified in §10. The tool-generated ELFs are
exonerated (the 1024/2048/2088/2200 trend is smooth, not a step).

**Conclusion**: the fused decode path (1 runlist submit/token, in-kernel
norm/RoPE/SiLU) **closes the gap at the objective's @1k context (+5%)** and
meets FLM at prefill/TTFT; it trails by ~7% only at the ~2k context the
harness's standard prompt happens to tokenize to, and that residual is the
closed-source attention KV-scaling — not something the native host side can
remove (the i6 rope-overlap of §10 buys only ~0.5 ms/tok of the ~1.4 ms gap).

## 12. FRESH RE-MEASUREMENT (2026-09-16, current tree, device usable)

After the user noted the device is not actually hard-blocked (idle `flm serve`
holds one of 16 shared hw contexts), direct `NPU_RUNLIST=1 NPU_FWD_TIMING=1`
runs re-confirm the committed numbers on the current tree:

| ctx | exec ms | build ms | rope ms | decode tok/s |
|---|---:|---:|---:|---:|
| 1024 | **12.5** | 2.0 (overlapped) | 0.4 | **82** (matches committed 82) |
| 2088 | **15.3** | 2.1 (overlapped) | 0.4 | **69** (committed 67, within variance) |

Decisive split: **build and rope are context-constant; the entire 1024→2088
scaling (~2.8 ms) is in `exec`** — the `layer.xclbin` attention reading a longer
KV prefix. This closes ra-3's open question: the host passes (§10) are NOT the
@2k gap; the gap is device-side, and the native exec scales ~2.5x more steeply
than FLM's (12.5→15.3 vs 12.8→13.9 ms), which is the one remaining unexplained
delta (KV layout / per-ctx ELF vs FLM's internal sequence).

### 12b. ra-6 device-free narrowing: FLM's forward uses the SAME mechanism

Disassembly of `qwen3_npu::Impl::forward` (libqwen3_npu.so @0x42960) shows it
calls `xrt::runlist::execute/wait`, `npu_app::_setup_kernel`, `set_context_length`,
and per-run `set_arg_at_index` — the SAME per-ctx-kernel + runlist structure the
native's `RuntimeLayerEngine::build_runlist` replicates (28 layer runs + lm_head
in one runlist). `gen_layer_seq`/`gen_mha_engine_seq` are not called in forward
itself (the sequence is generated inside `_setup_kernel`, matching the native's
pre-generated per-ctx ELF). So the ~2.5x-steeper native exec scaling is **not a
different-mechanism effect** — it is either a KV-layout/BO-size difference or a
measurement-condition artifact (FLM's 12.8/13.9 ms were taken clean 2026-09-12,
the native's 12.5/15.3 ms today with the 35B serve loaded). Both need the clean
A/B of ra-6 to settle.

KV-size cross-check: the native allocates `npu_kv_cache_bo_size = 128 MB` per
layer (134217728 B, `npu-infer/include/common.h:32`) and writes only 32 MB
(4 regions × 8 MB stride, MAX_L=8192); FLM's Qwen3 is constructed at MAX_L=32768
(`flm_prefill_bridge.cpp`), i.e. a 128 MB KV cache (32768×8×128×4 B) — the
allocation sizes match, so a BO-size difference is unlikely to be the cause.
