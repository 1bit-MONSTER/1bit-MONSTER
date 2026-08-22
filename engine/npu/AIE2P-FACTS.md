# AIE2P / XDNA 2 — hardware facts for the NPU path

Durable findings from the MAX+XDNA investigation (2026-08-13, Strix Halo).
Source of truth for verification; everything here was measured on this box.

## 1. Platform identity

- `xrt-smi examine`: `NPU Strix Halo [0000:c6:00.1]`, **architecture `aie2p`**,
  topology **6x8**. XDNA 2 = AIE2P in AMD's tooling (answers the "does XDNA2
  map to AIE2P" question definitively).
- amdxdna driver 0.1, XRT 2.25.37, NPU firmware 1.1.2.65, BIOS AMI 1.09.
- IRON (github.com/amd/iron) + mlir-aie + llvm-aie (Peano) compile and run
  kernels on this NPU: AXPY 20/20, elementwise add 20/20, GEMM 2048³ pass.

## 2. AIE2P bf16 add rounds toward −∞ (RNI), NOT RNE ⚠️

Measured: for `s = bf16_to_f32(a) + bf16_to_f32(b)` (exact in fp32), the
hardware bf16 result is

```
result = trunc16(s) + ((frac16(s) != 0) && (sign(s) == 1))
```

i.e. truncation with round-toward-negative-infinity on negatives. 0/4096
mismatches vs this rule; RNE / round-half-away / ties-to-odd / RZ all
mismatch ~24% of elements on random data. IRON's own elementwise-add test
only passes on `rel_tol=0.04`, which hides this.

Measured scope: exact for ALL finite bf16 pairs (0/4096 across 6 fresh seeds,
incl. negative/positive values). Edge behavior (documented, not covered by the
rule): `-0 + -0` → `+0` (sign canonicalized); any NaN result → fixed quiet-NaN
pattern `0x7f81` (input NaN payloads are dropped, `inf + -inf` also yields
`0x7f81`).

**Any bf16 kernel verification on this NPU must use the RNI reference.**
Reference recipe (numpy):

```python
def rni_bf16(x):  # x: fp32 ndarray -> uint16 bf16 values, RNI rounding
    bits = x.astype(np.float32).view(np.uint32)
    up = ((bits & 0xFFFF) != 0) & ((bits >> 31) & 1) == 1
    return ((bits >> 16) + up).astype(np.uint16)
```

## 3. Dispatch costs

- Per-call BO allocation (3× host_only BOs + sync + readback) costs ~7 ms —
  the delta between the engine's persistent-BO path (~49 ms for 2048³ GEMM)
  and a naive allocate-per-call shim (~56 ms, 1.14×). Persistent BOs are the
  right design; this quantifies why.
- Single-column 2048³ bf16 GEMM design: ~49 ms (0.35 TFLOPS). Six columns
  would be ~6× faster — the engine's multi-column xclbins are essential for
  real decode workloads.

## 3b. Decode launch-cost diagnosis (2026-08-13, measured)

- Per-launch breakdown (NPU_GO_STATS on the universal engine): quantize ~0.1ms,
  **sync+launch ~0.02-0.06ms, kernel wait ~3.4-4.9ms**, dequant ~0.01ms. Decode =
  112 launches/token (4 GEMMs × 28 layers) → ~460ms/token (2 tok/s).
- The ~4ms/launch is the kernel executing its **fixed M=128 stream** (the FLM
  mm.xclbin is baked for XM=128). The generated stream has the SAME word count
  for any M (M is baked into descriptor values, not the stream length).
- **M is always 128.** The generated stream is a pure function of (K, N):
  `gemm_generate_sequence_i8` voids its M argument (the v27 microkernel is
  M=128-baked: 4 × 32-row slices, BD-rotation schedule covers exactly 4 slices).
  Pre-rework generators wrote M into REG_M/descriptor offsets — regen_insts(M<XM)
  DEADLOCKED (~2048ms/launch, kernel never completes) because REG_M cannot
  resize the baked kernel's tiling (M=1 and M=8 both hung; issue #1761). The
  misleading `regen_insts(int M)` path was removed; smaller batches MUST reuse
  the M=128 stream with `am` = real row count (quantize_async zero-pads rows
  [am, 128)).
- Pre-compiled `_v` streams are WORSE: 99-150K words → ~154ms/launch → 16.4s/token
  (the npu_engine_cb path). The runtime generator's 32K-word streams are the
  good path (35x faster).
- The generated path's addressing: B-DMA offset includes K*M (packed A+B
  convention); the engine uses separate A/B BOs. Removing the K*M term is
  TOKEN-IDENTICAL — descriptor addressing doesn't affect this kernel's result.
- **The fix that works:** per-shape SMALL-M xclbins (build_xclbins.sh Peano
  path — the missing final_i8_*_K1024_N4096.xclbin files) so decode launches
  run M=1 streams (~50µs), or fused whole-layer streams (FLM-style, one launch
  per token). Estimated 460ms → 5-15ms/token with either.
- Correctness caveat: the generated path's output was never oracle-validated
  (the single-core-row vs multi-row WARN). Verify tokens vs the CPU engine
  before trusting any speedup.

## 4. XRT dispatch protocol (matches what engine/npu already does)

- Kernel name in mlir-aie xclbins: `MLIR_AIE`; metadata args
  `(opcode, instr, ninstr, bo0, bo1, bo2)`.
- Data BOs: `host_only`, group id **0**, created from `xrt::device` (not the
  hw_context), synced TO/FROM device around the run.
- Ctrlcode BO: `cacheable`, `group_id(kern.group_id(1))`.
- ERT-style run: `set_arg(0, 3)` (opcode), `(1, insts_bo)`, `(2, ninstr_bytes)`,
  then the data BOs; wait for `ERT_CMD_STATE_COMPLETED`.
