# qgemv handoff — quantized TileFuse GEMV kernel on XDNA2 (M2-3b)

Status 2026-08-08: toolchain ✅ (IRON live on this box), converter ✅, CPU
reference ✅. The AIE kernel itself is the remaining piece. This doc is the
complete spec so the task starts cold.

## Environment (already working)
```bash
source ~/iron-repo/setup.sh     # PATH: mlir_aie bin + Vitis 2025.2 xclbinutil; PYTHONPATH: pyxrt
cd ~/iron-repo && python -m pytest iron/operators/axpy/    # 96/96 on NPU
cd ~/iron-repo && python -m pytest iron/operators/gemv/    # 95/95 on NPU
```

## Template to copy
`iron/operators/gemv/` → new `iron/operators/qgemv/`:
- `op.py` — GEMV MLIROperator: tile dims, kernel link (aie_kernels/generic/mv.cc → new qgemv.cc), ObjectFIFO sizes
- `design.py` — my_matvec dataflow: A (weights) + B (vector) FIFOs → kernel → C (output) FIFO; **weights FIFO payload = one .tfb tile (4352 B: 4096 codes + 128 bf16 scales + 128 int8 zps×2)**
- `reference.py` — port `tools/tf_gemv.cpp` gemv_tile logic (unpack nibbles → (q−zp)·scale → dot)
- `test.py` — golden vs NPU (mirror axpy's run_test)

## .tfb tile layout (from tools/gguf_to_tilefuse.cpp — VERIFIED)
128×64 tile, per column c (0..63):
- codes: byte(r, c//2) = row r, cols 2·(c//2) (lo nibble) + 2·(c//2)+1 (hi nibble); 4096 B
- scales: 64×bf16 @ +4096
- zps: 64×int8 ×2 (duplicated for DMA 128B alignment) @ +4224
- dequant: (q − zp)·scale; constant column (lo==hi): scale=|lo| so (0−zp)·scale==lo
- GEMV: out[c] = Σ_r w[r,c]·x[r]; accumulate across the kt tile-rows

## Kernel notes (from the TileFuse paper §4.3)
- Unpack: load 32-bit, extract nibbles via masks/shifts → int8 lanes → convert
  to bf16 → (q−zp)·scale → local buffer → dot with the shared activation
  fragment (reuse the dequantized tile across activation slices)
- GEMV dataflow: shim → memory core distributes 4 bundled tiles → all 4×8
  compute rows (baseline uses 1 row); 64×8 weight block per iteration = 64
  outputs/iteration
- The stock `aie_kernels/aie2/mm.cc` i8 combos are the closest existing
  pattern (i8_i32_ONLY: int8×int8→int32, 4×8×8)

## Acceptance gate
`qgemv` test vs the CPU reference on the same .tfb tile data; then compare
against `tf_gemv --check` outputs (err/bound < 1.0 per column).
