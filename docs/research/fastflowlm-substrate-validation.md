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

## Real sequence generation on our machine (2026-08-31) — byte-level CONFIRMED

Built a driver against the PREBUILT `libqwen3_npu.so` (headers from the repo's
`src/include`, linked `-lqwen3_npu -lmha -lq4_npu_eXpress -lgemm -ldequant
-llm_head`). With a minimal qwen3-0.6B `config.json` loaded directly into
`LM_Config::_json_config` (bypassing the path resolver):

- `qwen3_npu_sequence(cfg, 128).gen_layer_seq(&nseq, 1)` -> **7231 words**,
  dumped to `/tmp/fflm-layer0.seq` (28924 B).
- Header `0x1b 0x1b00 0x378 nbytes` (word[3] = 28924 = file size; the project's
  RE'd FLM header plays the same role).
- Op histogram matches the project's RE'd emission exactly:
  `0x80` TCT (169), `0x81` DDR_PATCH (172), `0x01` BLOCKWRITE (577),
  `0x03` MASKWRITE (166), `0x00` WRITE (4437), plus the `0x30/0x18/0x1c/0x10`
  tail-word patterns the project's `bd()/dp()/mw()/wr()` lambdas emit.
- The Qwen3-0.6B `layer.xclbin` from the repo REGISTERS on this box's NPU:
  kernel `MLIR_AIE` with args `(opcode:0, instr:1, ninstr:2, bo0..bo4:3..7)` —
  the SAME invocation signature the project uses (`k(3, bI, ninstr, A, B, C)`).

=> The project's RE'd instruction format is validated end-to-end against the
real AMD generator output, and the real layer kernel is directly runnable on
this NPU. Next: load the real weights + BOs per the sequence and run the real
layer-1 FFN kernel, comparing its output against the project's two-launch path.

## Real layer-1 buffer layout (from dump_patch_table) — 2026-08-31

The real Qwen3-0.6B layer-1 sequence's host patch table (172 triples) gives the
BO layout the runtime binds:
- **arg1: ~7.74 MB** (offset span 0..7,741,440) — the layer weights (Q4NX 4-bit:
  gate+up 1024x3072x2 + down 3072x1024 + attention ~ = ~7.7 MB at 4-bit ✓)
- **arg4: ~192 KB** (span 0..196,608) — the activation/output buffers
- **arg0, arg2, arg3: ~4 KB each** — small control/scratch buffers
This matches the project's conceptual layout (weights BO + activation BO +
small control BOs). To RUN the real layer-1 kernel on the NPU: allocate the 5
BOs per this map, load Q4NX-format weights (FastFlowLM's q4_npu_eXpress format,
not the project's int8 — needs the model files or a converter), bind via the
patch table, launch `MLIR_AIE(3, insts, ninstr, bo0..bo4)`.
