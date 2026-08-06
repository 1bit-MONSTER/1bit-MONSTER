# NPU INT8 MLIR Generators

MLIR generators for the 1bit-systems NPU INT8 GEMM engine. Each generator produces a `.mlir` file for the NPU2 AIE array (Strix Halo), which is compiled to an `.xclbin` via the [MLIR-AIE toolchain](https://github.com/Xilinx/mlir-aie) (`aiecc`).

## Generator Versions

| File | Status | Description |
|------|--------|-------------|
| `n1_core_i8_v24.py` | ✅ **Current** | BD descriptor pipelining — K-iteration batching in groups of 6 |
| `../xclbins/n1_core_i8_v2.py` | ✅ Stable | Flat K-iteration loop (baseline, 16.3s/tok) |
| `../xclbins/n1_core_i8_i32_4row_v10.py` | ⚡ Experimental | INT32 accumulator, 4-row task pipelining |

## v24: BD Descriptor Pipelining (Issue #1075)

### The Problem

In v2, each K-iteration issues a separate DMA start/wait cycle through the
object_fifo acquire/release mechanism. For K = 1024 with k_tile = 64, this
means 16 sequential DMA cycles per M-tile. The NPU shim DMA engine spends
most of its time in start/wait overhead instead of moving data.

Result: **16.3 seconds per token** for the D projection (worst-case K=3072).

### The Fix: K-Iteration Batching

v24 batches K-iterations in groups of **6** per DMA round:

```
v2 (flat):  for each K-iteration { acquire A; acquire B; compute; release }
v24 (batched): for batch of 6 { acquire 6 A + 6 B; wait once; compute 6; release 6 }
```

### Why 6?

The NPU shim DMA engine allows **16 BD descriptors per tile**. Each FIFO buffer
requires one BD for the L2→L1 DMA. The budget:

| FIFO | Buffers | BDs |
|------|---------|-----|
| A_l2l1 | 7 (batch 6 + 1 in-flight) | 7 |
| B_l2l1 | 7 (batch 6 + 1 in-flight) | 7 |
| C_l1l2 | 1 | 1 |
| **Total** | | **15** |

15 BDs ≤ 16 ✓ — stays within the hardware limit.

The L2 tile size (`mtk`) changes from 512 (8 K-iterations) to **384** (exactly
6 K-iterations). This means each L2→L1 batch feeds exactly one core batch.

### Memory Budget

Memory tile (512KB L2 scratchpad):

| Buffer | Count | Size | Total |
|--------|-------|------|-------|
| A_l1 buffers | 7 | 2,048 B (32×64) | 14 KB |
| B_l1 buffers | 7 | 8,192 B (64×128) | 56 KB |
| C_l1 buffers | 1 | 8,192 B (32×128) | 8 KB |
| **Total** | | | **78 KB** |

78 KB ≪ 512 KB ✓ — ample room in the memory tile.

Compute tile (64KB local memory): holds only 1 buffer of each at a time
(consumed from FIFO, not pre-loaded).

### Performance Impact

For Qwen3-0.6B (D projection: M=128, K=3072, N=1024):

| Metric | v2 (flat) | v24 (batched) | Improvement |
|--------|-----------|---------------|-------------|
| K-iterations per M-tile | 3072/64 = 48 | 48 | (same) |
| DMA start/await cycles | 48 | ceil(48/6) = 8 | **6× fewer** |
| Estimated decode time | 16.3 s/tok | ~2.7 s/tok* | **~6× faster** |

*Theoretical estimate: 6× reduction in DMA overhead. Actual depends on
compute-to-DMA overlap ratio on hardware.

### Usage

```bash
# Generate v24 MLIR for QKV projection (M=128, K=1024, N=4096)
python3 engine/npu/generators/n1_core_i8_v24.py \
    -M 128 -K 1024 -N 4096 > mm_qkv_v24.mlir

# Generate v24 MLIR for D projection (M=128, K=3072, N=1024)
python3 engine/npu/generators/n1_core_i8_v24.py \
    -M 128 -K 3072 -N 1024 > mm_d_v24.mlir

# Custom batch size (e.g., 4 for smaller memory budget)
python3 engine/npu/generators/n1_core_i8_v24.py \
    -M 128 -K 1024 -N 4096 --batch-size 4 > mm_qkv_v24_b4.mlir

# Compile with aiecc (requires MLIR-AIE toolchain)
# See: engine/npu/build_xclbins.sh
aiecc mm_qkv_v24.mlir ...
```

## v23 → v24 Changelog

| Aspect | v23 (v2) | v24 |
|--------|----------|-----|
| **L2 K-tile size** (`mtk`) | 512 | 384 (6 × 64) |
| **A_l2l1 FIFO depth** | 2 | 7 |
| **B_l2l1 FIFO depth** | 2 | 7 |
| **Core K-loop** | Sequential acquire→compute→release | Batch acquire 6 → compute 6 → release 6 |
| **Remainder handling** | N/A | Partial batch for leftover K-iterations |
| **BD descriptors/tile** | 5 | 15 |
| **Kernel** | `mm_32x64x128.o` | `mm_32x64x128.o` (same) |
| **Dtype** | int8 / int16 | int8 / int16 (same) |

## Toolchain Requirements

### MLIR-AIE Toolchain

The `.mlir` → `.xclbin` compilation requires:

- **aiecc** (MLIR-AIE v0.3.x) — compiles MLIR to AIE instructions + PDI
- **Peano compiler** (LLVM-based) — kernel compilation for GEMM xclbins
- **LLVM 21** (LLVM-AIE fork) — `opt`/`llc` passes for MLIR lowering

Verified toolchain setup (2026-07-29):

```bash
export AIE_TOOLS_DIR=~/mlir-aie/install_tmp
export PATH=$AIE_TOOLS_DIR/bin:$PATH
export PYTHONPATH=$AIE_TOOLS_DIR/python:$PYTHONPATH
```

### Fix Toolchain Script

Use `fix_toolchain.sh` to resolve opaque-pointer LLVM version mismatches:

```bash
# Check current toolchain
./engine/npu/generators/fix_toolchain.sh --check

# Set up environment (source from build script)
source engine/npu/generators/fix_toolchain.sh --setup-env

# Fix opaque pointers in generated LLVM IR
./engine/npu/generators/fix_toolchain.sh --fix build/dir/

# Generate aiecc wrapper for persistent fix
./engine/npu/generators/fix_toolchain.sh --generate-wrapper
```

The fix script:
1. Routes LLVM IR through LLVM-AIE's LLVM 21 for `opt`/`llc` passes
2. Uses Peano's clang for kernel compilation (correct pointer mode)
3. Patches typed-pointer IR to opaque-pointer format automatically

### Known Issues

- **Opaque pointer mismatch** (LLVM 15+ vs Peano): The `fix_toolchain.sh`
  wrapper handles this by routing typed-pointer IR through LLVM-AIE's LLVM 21
  `opt`/`llc`.
- **Chess vs Peano PDI divergence**: xclbins built with different compilers
  produce different PDI binaries. Always use Peano for GEMM xclbins.
  See [#1076](https://github.com/1bit-systems/1bit-systems/issues/1076).
- **BD count limit**: If you increase batch_size beyond 6, verify total BDs
  stay under 16 per tile. See the table above for the formula.

## Building kernel variants (mm_32x64x128.o) — measured verdict

The GEMM core kernel is `mm_32x64x128.o` (DIM_M=32, DIM_K=64, DIM_N=128,
`matmul_i8_i32` / `zero_i32` exports, 8x8x8 vectorized mmul).  The original
compile command was not recorded anywhere — this is the verified reproduction
(2026-08-05), with clang++ from the working Peano
(`~/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie/bin/clang++`):

```
clang++ mm_kernel_reference.cc -c -o mm_32x64x128.o \
  -I ~/mlir-aie/third_party/aie_api/include -I ~/mlir-aie/aie_kernels/aie2p \
  -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
  --target=aie2p-none-unknown-elf \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY
```

(The `npu2_40_toolchain` checkout's iron/aiecc are internally inconsistent —
missing modules, stale binaries — do not route kernel builds through it.)

**DIM_K=128 variant — measured, negative.** `mm_32x128x128.o` (same 8x8x8
kernel, K=128 tiles) on the QKV shape: **1.943 ms / 552.6 GOP/s vs 1.591 ms /
674.8 GOP/s** for DIM_K=64 (~18% regression), both correctness passes clean.
Cause: the 16 KB B tile + 16 KB C tile pin core L1 at 64 KB, forcing fifo
depth 2 (batch=1); the k=64 build's depth-6 K-batching DMA pipelining is
worth more than the halved K-iteration count.  Tile shape is L1-bound and
effectively optimal for this kernel structure — further kernel work should
target operand-feed efficiency inside the mmul loop, not tile dims.

### Kernel profile (source + disassembly, 2026-08-05)

The k-reduction loop of `matmul_vectorized_2x2_mmul` is:

```
for i in k/s:              // colA = k/s = 8 iterations for DIM_K=64
    load A0, A1            // 2 vector loads
    load B0, B1            // 2 vector loads
    C00.mac(A0,B0) C01.mac(A0,B1) C10.mac(A1,B0) C11.mac(A1,B1)  // 4 vmac
```

**4 vector loads per 4 vmacs (1:1)**; the disassembly (.LBB0_2) shows the
compiler double-buffering the loads (bm*1/bm*2 register pairs) but the four
macs per step share A0/A1/B0/B1, so each step is one dependency group —
the macs cannot start until all four loads land, and the next step's macs
wait on this step's.  That is the concrete mechanism behind the ~750 GOP/s
(≈20-30% of the core's mmul issue capacity): **insufficient independent
macs to hide load/mac latency, not raw load count**.

The fix direction is a wider n-expansion (2x4 instead of 2x2): load A0/A1
once per k-step, load four B pairs, issue 8 macs — halves A traffic and,
more importantly, doubles the independent macs the pipeline can overlap.
That is a rewrite of the mmul loop in `mm_kernel_reference.cc` — the
generator cannot fix it, and the tile-shape experiments (DIM_K=128)
confirmed the L1 layout is already at its limit.
