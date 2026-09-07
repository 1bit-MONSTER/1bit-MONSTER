#!/bin/bash
# run_poke_sim.sh — build ps.so + run the single-core CHESS poke design under
# aiesimulator 2025.2 (V-2024.06 sim). Answers goal task-2's core question on
# the smallest possible design: does the chess core execute in sim at all?
#
#   poke_aie.mlir.prj  = aiecc --aiesim workdir (chess core main_core_4_2.elf,
#                        aie_inc.cpp with XAie_LoadElf, locks 0/1, C1@1024)
#   poke_ps_main.cpp   = ps driver: prefill C1, load ELF, give input token,
#                        enable core, wait, report C1/locks/status
set -euo pipefail

VITIS="${VITIS:-$HOME/Xilinx2025/2025.2}"     # 2025.2 (V-2024.06) — 2026.1 sim segfaults (#1908)
AIETOOLS="$VITIS/Vitis/aietools"
PRJ_DIR="${PRJ_DIR:-$HOME/poke_aie.mlir.prj}"  # sim package to run (poke2 for the fresh chess build)
PRJ="$PRJ_DIR"
AISIM_DIR="$(cd "$(dirname "$0")" && pwd)"

MLIR_AIE="$HOME/mlir-aie"
TL_INC="$MLIR_AIE/.venv/lib/python3.14/site-packages/mlir_aie/runtime_lib/x86_64/test_lib/include"
TL_LIB="$MLIR_AIE/.venv/lib/python3.14/site-packages/mlir_aie/runtime_lib/x86_64/test_lib/lib"
XAIE_INC="$MLIR_AIE/install/runtime_lib/x86_64/xaiengine/include"
XAIE_LIB="$MLIR_AIE/install/runtime_lib/x86_64/xaiengine/lib"
GENWRAP="$MLIR_AIE/install/aie_runtime_lib/AIE2P/aiesim/genwrapper_for_ps.cpp"
WAIT_US="${WAIT_US:-500}"

for p in "$GENWRAP" "$PRJ/aie_inc.cpp" "$AIETOOLS/bin/aiesimulator" "$AISIM_DIR/poke_ps_main.cpp"; do
  [ -e "$p" ] || { echo "ERROR: missing $p" >&2; exit 1; }
done

export PATH="$AIETOOLS/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINXD_LICENSE_FILE="$HOME/.Xilinx/Xilinx.lic"
export LD_LIBRARY_PATH="$XAIE_LIB:$AIETOOLS/lib/lnx64.o:$AIETOOLS/lib/lnx64.o/Ubuntu:$AIETOOLS/data/osci_systemc/lib/lnx64:${LD_LIBRARY_PATH:-}"

echo "== 1/3 build ps.so =="
mkdir -p "$PRJ/sim/ps"
g++ -O2 -shared -fPIC -fpermissive \
  -Wno-deprecated-declarations -Wno-format-security \
  -DSC_INCLUDE_DYNAMIC_PROCESSES -D__AIESIM__ -D__PS_INIT_AIE__ \
  -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -D__AIEARCH__=21 \
  "-Dmain(...)=ps_main(...)" "-DWAIT_US=$WAIT_US" \
  -I "$PRJ" \
  -I "$AIETOOLS/include" \
  -I "$XAIE_INC" \
  -I "$AIETOOLS/data/osci_systemc/include" \
  -I "$AIETOOLS/include/xtlm/include" \
  -I "$AIETOOLS/include/common_cpp/common_cpp_v1_0/include" \
  -I "$TL_INC" \
  "$GENWRAP" "$AISIM_DIR/poke_ps_main.cpp" \
  "$MLIR_AIE/runtime_lib/test_lib/test_library.cpp" \
  "$TL_LIB/libmemory_allocator_sim_aie.a" \
  -L "$XAIE_LIB" -lxaienginecdo \
  -L "$AIETOOLS/lib/lnx64.o" -L "$AIETOOLS/lib/lnx64.o/Ubuntu" \
  -L "$AIETOOLS/data/osci_systemc/lib/lnx64" \
  -Wl,--as-needed -lsystemc -lxtlm \
  -o "$PRJ/sim/ps/ps.so" 2>&1 | tail -15
if [ ! -s "$PRJ/sim/ps/ps.so" ]; then
  echo "ERROR: ps.so not produced (see compiler diagnostics above)" >&2
  exit 1
fi
echo "  OK: $PRJ/sim/ps/ps.so ($(stat -c%s "$PRJ/sim/ps/ps.so") B)"

echo "== 2/3 run aiesimulator (pkg $PRJ/sim) =="
cd "$PRJ"
HANG_NS="${HANG_NS:-1000000}"
timeout "${SIM_TIMEOUT:-600}" aiesimulator --pkg-dir=sim --simulation-cycle-timeout=200000000 \
  --hang-detect-time="$HANG_NS" \
  --display-run-interval=0 2>&1 | grep -v -E "^\[INFO\]|^\[WARNING\]" | tail -"${SIM_TAIL:-60}"
echo "aiesimulator exit=$?"
