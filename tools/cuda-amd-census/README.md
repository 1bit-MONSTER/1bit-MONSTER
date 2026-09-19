# CUDA-surface → AMD backend capability census

Linux-native equivalent of the evidence discipline in
[Speedstu/CUDA-for-AMD-Windows](https://github.com/Speedstu/CUDA-for-AMD-Windows)
`docs/COMPATIBILITY.md` — **without** porting its Windows/ZLUDA stack. On Linux the
open stacks are already native HIP, so what matters here is which CUDA-shaped
surface resolves to which AMD backend, on which arch, at what evidence level,
**proven by probe rather than asserted**.

Published results: [`okf` reference — CUDA-surface capability matrix](https://github.com/1bit-MONSTER/okf)
(`systems/1bit-monster/references/cuda-on-amd-capability-matrix.md`).

## Two backends, same binaries

| Lane | Backend | Why it is a separate column |
|---|---|---|
| `rocm` | ROCm / TheRock HIP + rocBLAS, rocFFT, rocSPARSE, hipSOLVER, MIOpen | the conventional stack |
| `hrx` | AMD **HRX** (`libhrx`) + **Loom** JIT (`libloomc`) | ships its own `libamdhip64.so` drop-in, so the *same* probe binaries run against it by changing only `LD_LIBRARY_PATH` |

HRX is **not** self-contained on a bare box: it needs the ROCr/HSA runtime
(`libhsa-runtime64.so.1`) or it silently falls back to the CPU accelerator. The census
records that as `UNSUPPORTED`, never as a pass — a CPU fallback is a failure, not a result.

## Usage

```bash
# local (e.g. ryzen / gfx1201)
EXPECT_GFX=gfx1201 ./run-census.sh --machine ryzen --out reports --timeout 60

# remote (e.g. strixhalo / gfx1151), TheRock pip-SDK layout
ROCM_DEVEL=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel \
EXPECT_GFX=gfx1151 PYBIN=/usr/bin/python3 \
  ./run-census.sh --machine strixhalo --out reports --timeout 75
```

Useful overrides: `ROCM_DEVEL`, `HRX_LIB`, `HRX_BIN`, `PYBIN`, `EXPECT_GFX`.
Output: `reports/<machine>.rows.tsv` (one row per surface) and
`reports/<machine>.json` (machine-readable, with counts).

## Evidence model

Classes, inherited from upstream's "Safe failure matters":

`detected` · `loadable` · **`PASS`** · **`UNSUPPORTED`** · **`INCORRECT`** · **`TIMEOUT`** · **`ERROR`**

Enforced rules:

- **A row is never promoted.** `detected` ≠ `loadable` ≠ functional. A header or
  library being present is recorded as `detected` and nothing more.
- **Success is not support.** Every functional row must match an *independent CPU
  reference*: naive CPU SGEMM, naive O(N²) DFT, hand-computed CSR spmv, a swept
  determinant, a direct CPU convolution. An API returning success proves nothing.
- **GPU execution is proven, not assumed.** Device-only witnesses: a `clock64()`-derived
  sentinel with launch geometry, device-side `arange`, or an AMD-only intrinsic.
- **Missing stack ≠ hardware failure.** An absent toolchain is `UNSUPPORTED`, not `ERROR`
  — and `UNSUPPORTED` is not evidence that the hardware cannot do it.
- **Fail closed.** An explicit `UNSUPPORTED` beats a plausible-looking wrong tensor.

## Two traps this harness encodes, both of which caught real mistakes

1. **A random-valued mismatch cannot tell a backend defect from a reference bug.**
   An early run reported MIOpen conv2d as `INCORRECT` (`rel=8.2e-01`). A follow-up
   all-ones test — where *any* layout convention yields the same tensor, so a
   disagreement there must be a genuine defect — **passed 18/18**. The `INCORRECT`
   verdict was therefore withdrawn: it was this harness's ordering assumption, not
   MIOpen. `probe/p_dnn_diag.cpp` keeps that two-stage discipline.
2. **A compile failure can be the evidence.** `probe/p_dot4.cpp` exists separately
   because every variable-operand form of the 6-arg `__builtin_amdgcn_sudot4` is
   rejected on amdclang 23 / HIP 7.16 (`must be a constant integer`). A build failure
   there is the *finding* for that row, not a harness bug — and it is recorded as an
   open discrepancy against the ISA notes rather than resolved by guessing.

## Layout

| Path | Role |
|---|---|
| `run-census.sh` | driver: discovery, inventory, build, run, classify, report |
| `probe/common.hpp` | evidence classes, CPU references, tolerances |
| `probe/p_core.cpp` | HIP runtime, bf16/fp16/atomics, CUDA graphs, hipRTC JIT |
| `probe/p_libs.cpp` | rocBLAS, rocFFT, rocSPARSE, hipSOLVER, MIOpen (all CPU-referenced) |
| `probe/p_dot4.cpp` | int8 dot4 reachability (compile failure = the evidence) |
| `probe/p_dnn_diag.cpp` | layout-independent all-ones conv disambiguation |
| `probe/p_torch.py` | framework surface; a `+cpu` wheel is `UNSUPPORTED`, not a pass |
| `probe/vk_compute.{c,comp}` | Mesa RADV Vulkan compute lane (no CUDA analogue) |
| `reports/` | captured per-machine TSV + JSON |

## Notes for future runs

- The `int8_dot4`, MIOpen-random-case and ryzen-HRX items are **open**, not settled;
  see the okf note's "Open items" for the specific resolution paths.
- `cublas/hrx` is **not** an HRX BLAS: HRX ships none, so that row is rocBLAS running on
  HRX's `libamdhip64`. It evidences the compat layer, not a rocBLAS replacement.
- No performance claims are made anywhere here, and no timing was compared.

## gfx1201 device-pack mode (goal mu7tnx28-9nb7un)

ryzen's gfx1201 math-library failures were traced to running the **wrong SDK device build**:
the installed SDK is `rocm_sdk_device_gfx1151`, whose rocBLAS ships Tensile kernels for gfx1151
only. Reproduce the fix without touching the real install:

```bash
D=~/.cache/pip/therock/lib/python3.14/site-packages/_rocm_sdk_devel
rm -rf ~/sdk-gfx1201-devel && mkdir -p ~/sdk-gfx1201-devel
cp -al $D/. ~/sdk-gfx1201-devel/          # hardlinks: instant, ~0 bytes, same fs required
# download rocm_sdk_device_gfx1201-*.whl from
#   https://nightly.repo.amd.com/rocm/whl-next/rocm-sdk-device-gfx1201/
# then overlay its _rocm_sdk_libraries/lib/* into ~/sdk-gfx1201-devel/lib/
ROCM_DEVEL=$HOME/sdk-gfx1201-devel EXPECT_GFX=gfx1201 \
  ./run-census.sh --machine ryzen-gfx1201pack --out reports --timeout 90
```

Result (committed in `reports/ryzen-gfx1201pack.*`): **cuBLAS `ERROR`→`PASS`** (`sgemm
max_abs=1.72e-05`, identical to gfx1151) and **cuDNN `ERROR`→`PASS`**. rocSPARSE and hipSOLVER do
**not** move — the device pack covers only rocblas/hipblaslt/miopen. Compare **per-surface
verdicts**, not totals: the pack run stages no HRX bundle, so `detected` differs for an unrelated
reason. See `okf` → `references/gfx1201-rocm-gap-status.md`.

## SPIR-V / ZCFS gate (`probe/p_spirv_zcfs.cpp`)

```bash
L=$D/lib/llvm/bin
$L/clang++ -x hip --offload-arch=amdgcnspirv --rocm-path=$D -O3 \
  -I$D/include -isystem $D/include probe/p_spirv_zcfs.cpp -o /tmp/gate -L$D/lib -lamdhip64
$L/llvm-objdump --offloading /tmp/gate   # must show hip-spirv64-...--amdgcnspirv, NO gfxXXXX
```

⚠️ **CORRECTION (2026-09-19).** An earlier version of this section warned that `hipcc`
"silently substitutes native detection". **That was wrong.** The command behind the claim omitted
`--offload-arch` entirely while its label claimed the flag was passed; hipcc then correctly fell
back to native detection. Verified four ways (`.cpp`/`.cu` × with/without `-x hip`) on clang 23 and
24: `hipcc --offload-arch=amdgcnspirv` **honours the flag** and emits
`hip-spirv64-amd-amdhsa--amdgcnspirv`. Raw `clang++` and `hipcc` both work; there is no defect.
Keep using `clang++` if you already are — it is not required as a workaround.
