# FastFlowLM layer kernel on strixhalo — 2026-08-31 findings

## What WORKS (proven this session)
- **The module flow (ERT_START_NPU) executes control codes**: a minimal
  cmds2seq TXN via `xrt::module` → `state=4 COMPLETED`, 0ms, no faults.
- The SVA/PASID path is functional: AMD-Vi fault events carry `domain=0x0001`
  which is the **PASID (1)** — the DPU DMA is PASID-tagged and translated; the
  faults were all on **unmapped VAs** (targets outside the BOs).
- **Truncated layer TXNs complete**: the real Qwen3-0.6B layer TXN
  (`fflm-layer0.seq`, corrected header `06040100 00000108 00000378 000070fc`)
  truncated to ≤4000 words executes and completes via the module flow.

## What BLOCKS the full layer TXN
- The full 7231-word layer TXN hangs (state=8) at the **first DMA-completion
  wait** — a burst of TCT (op 128) sync-ops at ~word 4211, right after a
  DDR_PATCH (arg1=weights @0x366000) + BD enable (MASKWRITE). The DPU waits for
  a shim-DMA completion token that never arrives.
- The hang is DATA-path (a DMA that never completes), not address translation:
  no IO_PAGE_FAULTs on the full-TXN runs.

## The xclbin/PDI problem (layer.xclbin blocks the DPU start)
- The real `layer.xclbin` (aie_partition 0x52198, PDI UUID eab7b277) prevents
  the DPU from starting **for any TXN** (even the working cascade TXN: state=8,
  no faults). The `final_cascade_fused.xclbin` (0xe618, PDI 132548f9) lets the
  DPU start (both TXNs run register writes and reach their data paths).
- Suspect: the layer.xclbin's PDI (column/CDO config) is rejected or not
  applied by the in-tree amdxdna 0.10.0 + firmware 1.1.2.65 stack, leaving the
  DPU unable to start.

## The 09:12 "calibration CLOSED" is suspect
- The cascade control code's DDR_PATCH offsets (0x6c000..0x2f4000 into arg1)
  exceed the probe's bB (0x8000 B) ~100× — the DPU's output writes land far
  outside bB → unmapped → faults. The probe as-built **cannot** have produced
  `bad=0/8192 EXACT MATCH`; the claim likely came from a stale/different
  artifact or a CPU-mirror-only comparison.
- The kernel/driver/firmware/inputs were verified byte-identical between the
  09:12 claim and the post-reboot runs; a fresh boot does NOT restore it.

## Next-step candidates
1. **Fix the layer.xclbin DPU-start block** (the highest-value target): compare
   the layer vs cascade PDIs, find the CDO/column config the firmware rejects,
   regenerate the layer.xclbin (or patch its PDI).
2. **Fix the DMA-completion hang** on the cascade xclbin: the DDR_PATCH ops
   re-patch BDs at firmware runtime; in the module flow the command payload
   carries only the opcode arg (3) — verify the patched BD values actually
   encode the intended BO addresses (aperture 0x4000000000000000 + host VA).
3. Re-verify the layer BO layout (weights 7.7MB, acts 200KB, in 4096 B etc.)
   against the TXN's DDR offsets (0x1e000..0x726000 weights; 0x0/0x20000 acts).

## ✅ RESOLVED (2026-08-31 14:51): THE REAL MODEL RUNS ON THE NPU

**The real FastFlowLM Qwen3-0.6B model runs on the XDNA2 NPU via the actual
ROCm/FastFlowLM runtime** (built from the vendored third_party/FastFlowLM,
model downloaded from huggingface.co/FastFlowLM/Qwen3-0.6B-NPU2):

- Prompt "Hello" → full think-tagged reply, 146 tokens:
  prefill 502.8 ms (27.8 tok/s), decode 1.366 s (96.6 tok/s).
- Zero IO_PAGE_FAULTs, zero TDR timeouts during the run.
- Configuration: kernel 7.2.0-perfopt (identity IOMMU group 26 — the upstream
  SVA-capable default), pristine amdxdna 0.10.0, firmware 1.1.2.65.

**Why my hand-rolled launcher deadlocked but the real runtime works**: the
runtime generates the layer sequence at runtime WITH THE REAL WEIGHTS LOADED
and allocates the BOs to match. My dumped `fflm-layer0.seq` (from a weight-less
`gen_layer_seq` dump) and my BO sizes did not match the runtime's actual layout,
so the shim-DMA sync tokens never arrived. The layer.xclbin, identity IOMMU,
module flow, and SVA were all fine all along.

Run script: `engine/npu/fflm-run/run_real_runtime.sh`.
Model cache: `~/.config/flm/models/Qwen3-0.6B-NPU2/` (model.q4nx 683 MB).
Xclbins: `/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2 -> vendored`.
