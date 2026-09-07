#!/bin/bash
# run_chess_exec_probe.sh — minimal constant-write chess-vs-peano EXECUTION
# probe on the NPU (issue #1878 family, "chess cores don't execute").
#
# Same design as bench_compiler_ab.sh (n1_core_i8_v27.py, mm_32x64x128.o link)
# but the kernel body (probe_kernel_const.cc) writes a fixed 0x5A5A5A5A
# pattern into C instead of a GEMM, and the runner (probe_run.cpp) pre-fills
# DDR C with 0xCDCDCDCD so an untouched buffer is observable.
#
# This isolates EXECUTION + C writeback from GEMM dataflow / arg delivery:
#   peano arm EXPECTED: VERDICT EXECUTED  (validates probe plumbing)
#   chess arm: EXECUTED  -> chess cores DO run on silicon (bug is elsewhere)
#              NEVER-WROTE / NO-WRITEBACK -> chess cores don't execute/write
#
# Usage: ./run_chess_exec_probe.sh [--keep] [--iters N]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
GEN="$REPO/engine/npu/generators"
TESTS="$REPO/engine/npu/tests"

MLIR_AIE="${MLIR_AIE:-$HOME/mlir-aie}"
PEANO_CLANG="$MLIR_AIE/.venv/lib/python3.14/site-packages/llvm-aie/bin/clang++"
PEANO="$MLIR_AIE/.venv/lib/python3.14/site-packages/llvm-aie"
AIECC="$MLIR_AIE/build_tmp/bin/aiecc"
AIETOOLS="$MLIR_AIE/build_tmp"
PYTHON="$MLIR_AIE/.venv/bin/python3"
MLIR_AIE_INC="$MLIR_AIE/.venv/lib/python3.14/site-packages/mlir_aie/include"
AIE_KERNELS_INC="$MLIR_AIE/aie_kernels/aie2p"

XILINX="${XILINX:-$HOME/Xilinx/2026.1}"
XCHESS_BIN="$XILINX/Vitis/aietools/bin"

M=128; K=2048; N=8192
M_T=32; K_T=64; N_T=128
COLS=8; ROWS=4; BATCH=5
PROBE_SRC="$TESTS/probe_kernel_const.cc"
KERNEL_O="mm_32x64x128.o"     # name the v27 design hardcodes (link_with)
DIMS=(-DDIM_M="$M_T" -DDIM_K="$K_T" -DDIM_N="$N_T" -Di8_i32_ONLY)

KEEP=0; ITERS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --keep)   KEEP=1; shift;;
    --iters)  ITERS=$2; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

export PATH="$XCHESS_BIN:$MLIR_AIE/install/bin:/opt/xilinx/xrt/bin:$PATH"
export PYTHONPATH="$MLIR_AIE/install_tmp/python:$MLIR_AIE/.venv/lib/python3.14/site-packages"
export LD_LIBRARY_PATH="$MLIR_AIE/install_tmp/python/aie/_mlir_libs:/opt/xilinx/xrt/lib"
export XILINXD_LICENSE_FILE="$HOME/.Xilinx/Xilinx.lic"

for t in "$PEANO_CLANG" "$AIECC" "$PYTHON" "$XCHESS_BIN/xchesscc" "$PROBE_SRC" \
         "$GEN/n1_core_i8_v27.py" "$TESTS/probe_run.cpp"; do
  [ -e "$t" ] || { echo "ERROR: missing $t" >&2; exit 1; }
done

if [ "$KEEP" = 1 ]; then
  W="$TESTS/_probe_out"; rm -rf "$W"; mkdir -p "$W"
else
  W=$(mktemp -d /tmp/chess_probe.XXXXXX)
  trap 'rm -rf "$W"' EXIT
fi
P="$W/peano"; C="$W/chess"; mkdir -p "$P" "$C"

echo "== 1/4 compile probe kernel .o — both compilers =="
"$PEANO_CLANG" "$PROBE_SRC" -c -o "$P/$KERNEL_O" \
  -I "$MLIR_AIE_INC" -I "$AIE_KERNELS_INC" \
  -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
  --target=aie2p-none-unknown-elf "${DIMS[@]}"
echo "  [peano] OK: $P/$KERNEL_O"
xchesscc_wrapper aie2p -c \
  -I "$MLIR_AIE_INC" -I "$AIE_KERNELS_INC" \
  -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
  "${DIMS[@]}" "$PROBE_SRC" -o "$C/$KERNEL_O"
echo "  [chess] OK: $C/$KERNEL_O"

echo "== 2/4 generate design.mlir (one design, both arms) =="
( cd "$W" && "$PYTHON" "$GEN/n1_core_i8_v27.py" \
    -M $M -K $K -N $N -m $M_T -k $K_T -n $N_T -c $COLS -r $ROWS -b $BATCH \
    > design.mlir 2>/dev/null )

echo "== 3/4 aiecc -> xclbin (both arms) =="
cp "$W/design.mlir" "$P/"; cp "$W/design.mlir" "$C/"
( cd "$P" && "$AIECC" --peano="$PEANO" --aietools="$AIETOOLS" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="final_peano.xclbin" --npu-insts-name="insts_peano.txt" \
    design.mlir > aiecc_peano.log 2>&1 )
echo "  [peano] OK: $P/final_peano.xclbin ($(stat -c%s "$P/final_peano.xclbin") B)"
( cd "$C" && "$AIECC" --aietools="$XILINX/Vitis/aietools" \
    --alloc-scheme=basic-sequential --xchesscc --xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="final_chess.xclbin" --npu-insts-name="insts_chess.txt" \
    design.mlir > aiecc_chess.log 2>&1 )
echo "  [chess] OK: $C/final_chess.xclbin ($(stat -c%s "$C/final_chess.xclbin") B)"

echo "== 4/4 NPU probe run =="
g++ -std=gnu++17 -O2 "$TESTS/probe_run.cpp" \
  -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
  -Wl,-rpath,/opt/xilinx/xrt/lib -lxrt_coreutil -lxrt_core -o "$W/probe"
echo "── peano arm ──"
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib "$W/probe" \
  "$P/final_peano.xclbin" "$P/insts_peano.txt" $M $K $N "$ITERS"
echo "── chess arm ──"
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib "$W/probe" \
  "$C/final_chess.xclbin" "$C/insts_chess.txt" $M $K $N "$ITERS"
echo
echo "Artifacts kept in: $W"
