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
