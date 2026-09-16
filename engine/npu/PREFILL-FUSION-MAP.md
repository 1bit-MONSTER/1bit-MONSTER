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
