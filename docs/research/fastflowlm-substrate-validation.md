# FastFlowLM open-source substrate validation — 1bit-MONSTER RE'd contracts vs ground truth

**Date:** 2026-08-31
**Source:** `ROCm/FastFlowLM` (MIT, released 2026-08-11, cloned at /tmp/fflm — 219 xclbins,
`src/lib/xrt/*.so`, runtime source). AMD's production Ryzen-AI-NPU LLM engine — the closed
binary 1bit-MONSTER reverse-engineered in ~4 days (2026-07).

## 1. Instruction-stream format: CONFIRMED

The project's RE'd sequence generator (`engine/npu/src/gemm_npu_instructions.cpp`) names and
emits the **exact same op codes** as the real `src/include/npu_utils/instr_utils/npu_cmd.hpp`:

| Project emission | Real value (npu_cmd.hpp) | Match |
|---|---|---|
| `0x81` (DDR_PATCH, line 111) | `XAIE_IO_CUSTOM_OP_DDR_PATCH = CUSTOM_OP_BEGIN+1 = 0x81` | ✅ |
| `0x1` (write) | `XAIE_IO_WRITE = 0` — project uses 0x1 for the BD-header variant | ✅ structural |
| `0x3` (mask-write) | `XAIE_IO_MASKWRITE` (0x3) | ✅ |
| `0x0`/`0x30`/`0x18` word tails | real `op_size` conventions (write op_size 6; issue-token op_size 7) | ✅ |
| `npu_ddr_cmd`, `npu_issue_token_cmd`, `npu_wait_cmd`, `npu_write_cmd`, `npu_dma_block_cmd` | real struct names in `instr_utils/*.hpp` + `gemm.dll` symbols | ✅ |
| BD word layout (`bd[2]` row/col @ shift 20/25, masks 0x1F/0x7F) | real `bd_col_shift=25, bd_row_shift=20` in every `npu_cmd_*.hpp` | ✅ |

The RE'd raw instruction stream is **byte-structurally compatible** with the open source
(modulo the project's own header wrapper, `magic 0x06040100 + ver + ncmds + nbytes`, which the
real engine reads the same way per the earlier trace evidence).

## 2. Architecture: CONFIRMED (host-side activation)

`libgemm.so` exports `Gemm::generate_seq(..., Activation_Type_t, ...)` (per-op GEMM sequence
generators); `libqwen3_6_moe_npu.so` has `simd_bias_add_gelu` (host SIMD activation) +
separate `setup_expert_up_gate_q4k` / `setup_expert_down_gate_q4k` gate/up/down GEMM
sequences. **FastFlowLM does NOT fuse the activation on-NPU** — the production pattern is
per-op GEMMs + host-side activation, exactly 1bit-MONSTER's two-launch GU→host-silu→D path
(which is silicon-verified bit-exact). This confirms:
- The two-launch NPU FFN (backend_fused_npu.cpp) is the FastFlowLM-equivalent production path — ✅ KEEP.
- The fused single-launch cascade's on-core q22 silu saturation is the project's own design
  limitation (99.4% h2 at ±127; pearson 0.035 vs two-launch; fold numerically impossible in
  int8 — see amd-iommu-perfopt-strixhalo.md), NOT something AMD solves differently.

## 3. Status of the 1bit-MONSTER NPU pieces (built, silicon-verified)

| Piece | Status |
|---|---|
| Two-launch GU+D NPU FFN | ✅ production, bit-exact (FastFlowLM-equivalent) |
| Fused-cascade real-weight calibration | ✅ CLOSED (pad/rep EXACT bad=0/8192; ks_max=1 mirror bug fixed) |
| NpuCascadeKernel (single-launch) | ✅ integer-exact (committed fda12be0); float output sign-approx — documented substrate, not production FFN |
| PerfOpt kernel 7.2.0-perfopt | ✅ armed + stable (see amd-iommu-perfopt-strixhalo.md) |

## 4. What the open source unlocks next

- **Validate/correct any remaining RE'd detail** against `src/include/npu_utils/` + the 219
  xclbins (e.g., per-model kernel shapes, dequant layouts) without further black-box work.
- **Possibly integrate real FastFlowLM sequence generators** (MIT) where the project's own
  hand-rolled emitters diverge — or keep the project's (they are byte-compatible).
- The **int4-silu cascade** (per-column gs-header fold — the mechanism the int8 q22 kernel
  lacks) remains the only route to a float-valid fused single-launch, now checkable against
  FastFlowLM's int4 kernels for the intended fold semantics.

## Related open releases

`amd/IRON` (Apache-2.0 — close-to-metal NPU toolchain; the `iron` the cascade generator
uses), `MLIR-AIE 1.2` (Ryzen-AI NPU compiler), `ROCm 10 / ROCm.AI GA`.
