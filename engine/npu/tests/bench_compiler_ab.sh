#!/bin/bash
# bench_compiler_ab.sh — controlled Peano (llvm-aie clang) vs Chess (xchesscc) kernel A/B
#
# Answers: "is there a performance difference between the Peano compiler and
# xchesscc for the same AIE2P kernel?" — with everything else held fixed.
#
# Method (one variable changed at a time):
#   - SAME kernel source      : generators/mm_kernel_reference.cc (2x2 mmul, i8_i32, 32x64x128)
#   - SAME MLIR design        : generators/n1_core_i8_v27.py design.mlir generated ONCE,
#                               reused by both arms (identical DMA/tiling/core placement)
#   - SAME bench harness      : tests/bench_gemm_analytical.cpp (correctness gate + ms/launch + GOP/s)
#   - ONLY the kernel .o compiler differs:
#       arm A (peano)  : llvm-aie clang++ --target=aie2p-none-unknown-elf + aiecc --no-xchesscc --peano=...
#       arm B (chess)  : xchesscc_wrapper aie2p (license brokered via the Vitis launcher) + aiecc --xchesscc --xbridge
#
# The bench only times when both correctness passes report 0 wrong, so a broken
# arm yields a clean FAIL + no timing (see OKF log issue #1878: chess-compiled
# kernels have historically zeroed C1 on the NPU — arg-delivery defect in the
# aiecc xchesscc core; the v27 external-objFifo design may or may not hit it).
#
# Usage:
#   ./bench_compiler_ab.sh [--rounds N] [--iters N] [--keep] [--no-run]
#   --rounds N   interleaved measurement rounds (default 3)
#   --iters N    benchmark iterations per launch batch (default 200)
#   --keep       keep work dirs under engine/npu/tests/_ab_out (default: /tmp, cleaned)
#   --no-run     build both xclbins + harness but skip the NPU timing (compile/A-B check)
#
# Needs the Strix Halo NPU host (strixhalo) with XRT + mlir-aie + Xilinx Vitis aietools.
# NOTE: do NOT run while the zaya/1bit engine is serving — the NPU is single-device
# and concurrent access is documented flaky (AMD-Vi IO_PAGE_FAULT storms).
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
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

# Xilinx Vitis aietools (xchesscc). 2026.1 ships chesscc X-2025.06.
XILINX="${XILINX:-$HOME/Xilinx/2026.1}"
XCHESS_BIN="$XILINX/Vitis/aietools/bin"

# Generator / kernel / bench parameters (same as check_mm_kernel_2x4.sh)
M=128; K=2048; N=8192
M_T=32; K_T=64; N_T=128
COLS=8; ROWS=4; BATCH=5
KERNEL_SRC="$GEN/mm_kernel_reference.cc"
KERNEL_O="mm_32x64x128.o"
DIMS=(-DDIM_M="$M_T" -DDIM_K="$K_T" -DDIM_N="$N_T" -Di8_i32_ONLY)

ROUNDS=3; ITERS=200; KEEP=0; NO_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds) ROUNDS=$2; shift 2;;
    --iters)  ITERS=$2;  shift 2;;
    --keep)   KEEP=1; shift;;
    --no-run) NO_RUN=1; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

# ── Environment ──────────────────────────────────────────────────────────────
# Order matters: the Vitis aietools/bin (the xchesscc LAUNCHER that brokers the
# tct_chess_me license) must come BEFORE mlir-aie/install/bin — that dir has a
# raw xchesscc→chess-clang symlink that makes aiecc derive the wrong aietools
# root (then chess-llvm-link is not found and the .chesslinked.ll step is
# skipped). xchesscc_wrapper (real file, mlir-aie) stays on PATH after it.
export PATH="$XCHESS_BIN:$MLIR_AIE/install/bin:/opt/xilinx/xrt/bin:$PATH"
export PYTHONPATH="$MLIR_AIE/install_tmp/python:$MLIR_AIE/.venv/lib/python3.14/site-packages"
export LD_LIBRARY_PATH="$MLIR_AIE/install_tmp/python/aie/_mlir_libs:/opt/xilinx/xrt/lib"
export XILINXD_LICENSE_FILE="$HOME/.Xilinx/Xilinx.lic"

check_env() {
  for t in "$PEANO_CLANG" "$AIECC" "$PYTHON" "$XCHESS_BIN/xchesscc" "$KERNEL_SRC" \
           "$GEN/n1_core_i8_v27.py" "$TESTS/bench_gemm_analytical.cpp"; do
    [ -e "$t" ] || { echo "ERROR: missing $t" >&2; exit 1; }
  done
  [ -e /dev/accel/accel0 ] || echo "WARN: no /dev/accel/accel0 — runtime phase will fail"
}

# ── Compile kernel .o, arm A: Peano (llvm-aie clang) ─────────────────────────
build_kernel_peano() { # $1 = out dir
  "$PEANO_CLANG" "$KERNEL_SRC" -c -o "$1/$KERNEL_O" \
    -I "$MLIR_AIE_INC" -I "$AIE_KERNELS_INC" \
    -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
    --target=aie2p-none-unknown-elf "${DIMS[@]}"
}

# ── Compile kernel .o, arm B: Chess (xchesscc via the Vitis launcher) ────────
# xchesscc_wrapper resolves AIETOOLS from `which xchesscc`, so the Vitis
# aietools/bin must be on PATH (the launcher brokers the tct_chess_me license;
# raw chesscc hits the license wall — OKF log #1878).
# NOTE: chess rejects -std= entirely (Release_LLVM default applies).
build_kernel_chess() { # $1 = out dir
  xchesscc_wrapper aie2p -c \
    -I "$MLIR_AIE_INC" -I "$AIE_KERNELS_INC" \
    -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
    "${DIMS[@]}" "$KERNEL_SRC" -o "$1/$KERNEL_O"
}

# ── Generate the MLIR design ONCE (identical for both arms) ──────────────────
gen_design() { # $1 = out dir
  ( cd "$1" && "$PYTHON" "$GEN/n1_core_i8_v27.py" \
      -M $M -K $K -N $N -m $M_T -k $K_T -n $N_T -c $COLS -r $ROWS -b $BATCH \
      > design.mlir 2>/dev/null )
}

# ── aiecc → xclbin. Arm A: peano flow (as check_mm_kernel_2x4.sh) ────────────
# The MLIR design's external_funcs link_with "mm_32x64x128.o" from the CWD, so
# the kernel .o and design.mlir must sit in the SAME directory.
build_xclbin_peano() { # $1 = arm dir, $2 = design dir
  cp "$2/design.mlir" "$1/"
  ( cd "$1" && "$AIECC" --peano="$PEANO" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
      --aie-generate-npu-insts \
      --xclbin-name="final_peano.xclbin" --npu-insts-name="insts_peano.txt" \
      design.mlir > aiecc_peano.log 2>&1 )
}

# ── aiecc → xclbin. Arm B: chess flow (xchesscc + xbridge) ───────────────────
# --aietools MUST point at the Vitis aietools ROOT for the chess arm: aiecc
# locates chess-llvm-link at <aietools>/tps/lnx64/target_aie2p/bin/LNa64bin/
# (target_aie2p is symlinked to target_aie2ps). Pointing it at mlir-aie's own
# build_tmp makes that step silently skip and the .chesslinked.ll never appear.
VITIS_AIETOOLS="$XILINX/Vitis/aietools"
build_xclbin_chess() { # $1 = arm dir, $2 = design dir
  cp "$2/design.mlir" "$1/"
  ( cd "$1" && "$AIECC" --aietools="$VITIS_AIETOOLS" \
      --alloc-scheme=basic-sequential --xchesscc --xbridge \
      --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
      --aie-generate-npu-insts \
      --xclbin-name="final_chess.xclbin" --npu-insts-name="insts_chess.txt" \
      design.mlir > aiecc_chess.log 2>&1 )
}

# ── Host harness (built once) ────────────────────────────────────────────────
build_harness() { # $1 = out dir
  g++ -std=gnu++17 -O2 "$TESTS/bench_gemm_analytical.cpp" \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
    -Wl,-rpath,/opt/xilinx/xrt/lib -lxrt_coreutil -lxrt_core -o "$1/bench"
}

# ── Run one arm; echo "PASS <ms> <gops>" or "FAIL (<reason>)" ─────────────────
run_arm() { # $1 = bench binary, $2 = xclbin path, $3 = insts path, $4 = iters
  local out; out=$(LD_LIBRARY_PATH=/opt/xilinx/xrt/lib "$1" \
    "$2" "$3" $M $K $N "$4" 2>&1) || true
  if echo "$out" | grep -q "^PASS$"; then
    local line; line=$(echo "$out" | grep "ms/launch" | tail -1)
    echo "PASS $(echo "$line" | grep -o '[0-9.]* ms/launch' | tr -d ' ms/launch') \
$(echo "$line" | grep -o '[0-9.]* GOP/s' | tr -d ' GOP/s')"
  else
    # surface the first failure line (e.g. all-zeros on the chess arm)
    local reason; reason=$(echo "$out" | grep -E "wrong=|cannot|error|Error" | head -1)
    [ -n "$reason" ] && echo "FAIL ($reason)" || echo "FAIL"
  fi
}

# ── Structural comparison of the two .o files ────────────────────────────────
# AIE2P objects are not x86 ELF — use the llvm-aie objdump with the aie2p triple.
# Peano emits per-function .text.<fn> sections; chess emits anonymous .text
# sections + a large .symtab/.strtab/.rodata, so report TOTAL code bytes (.text*)
# as the honest apples-to-apples metric.
report_objects() { # $1 = peano dir, $2 = chess dir
  local OD="$MLIR_AIE/.venv/lib/python3.14/site-packages/llvm-aie/bin/llvm-objdump"
  echo "── object comparison ───────────────────────────────────────────"
  printf "  %-28s %10s %14s\n" "arm" "size(B)" "code(B)"
  for arm in peano chess; do
    local d; [ "$arm" = peano ] && d=$1 || d=$2
    local sz code
    sz=$(stat -c%s "$d/$KERNEL_O")
    # .text* sizes: llvm-objdump prints e.g. ' 2 .text 0000000000000f40 ...';
    # sum them in shell arithmetic (mawk has no strtonum — bench on strixhalo
    # hit 'function strtonum never defined', 2026-08-28).
    code=0
    # llvm-objdump line: ' 2 .text.matmul_scalar_i8_i32 000001f0 00000000 TEXT'
    # -> $1=Idx $2=Name $3=Size $4=VMA $5=Type (size is hex without 0x prefix)
    while read -r _idx _name _sz _vma _typ; do
      case "$_name" in
        .text*) code=$((code + 0x$_sz));;
      esac
    done < <("$OD" --section-headers --triple=aie2p "$d/$KERNEL_O" 2>/dev/null)
    printf "  %-28s %10s %14s\n" "$arm ($KERNEL_O)" "$sz" "$code"
  done
  echo "  (code = sum of .text* section sizes, hex→dec via shell arithmetic)"
}

# ── Main ─────────────────────────────────────────────────────────────────────
check_env

if [ "$KEEP" = 1 ]; then
  W="$TESTS/_ab_out"; rm -rf "$W"; mkdir -p "$W"
else
  W=$(mktemp -d /tmp/ab_compiler.XXXXXX)
  trap 'rm -rf "$W"' EXIT
fi
P="$W/peano"; C="$W/chess"; mkdir -p "$P" "$C"

echo "== 1/5 compile kernel .o (i8_i32, ${M_T}x${K_T}x${N_T}) — both compilers =="
echo "  [peano] $("$PEANO_CLANG" --version 2>/dev/null | head -1)"
build_kernel_peano "$P"
echo "  [chess] $(xchesscc_wrapper aie2p --version 2>&1 | head -1 || true)"
build_kernel_chess "$C"
echo "  OK: $P/$KERNEL_O  $C/$KERNEL_O"
report_objects "$P" "$C"

echo "== 2/5 generate design.mlir (one design, both arms) =="
gen_design "$W"
echo "  OK: $W/design.mlir ($(stat -c%s "$W/design.mlir") B)"

echo "== 3/5 aiecc → xclbin (peano arm) =="
if build_xclbin_peano "$P" "$W"; then
  echo "  OK: $P/final_peano.xclbin ($(stat -c%s "$P/final_peano.xclbin") B)"
else
  echo "  PEANO XCLBIN BUILD FAILED — see $P/aiecc_peano.log"; exit 1
fi

echo "== 4/5 aiecc → xclbin (chess arm, --xchesscc --xbridge) =="
if build_xclbin_chess "$C" "$W"; then
  echo "  OK: $C/final_chess.xclbin ($(stat -c%s "$C/final_chess.xclbin") B)"
else
  echo "  CHESS XCLBIN BUILD FAILED — see $C/aiecc_chess.log"
  [ "$NO_RUN" = 1 ] && exit 0
  exit 1
fi

build_harness "$W"

if [ "$NO_RUN" = 1 ]; then
  echo "== 5/5 skipped (--no-run): both xclbins built, no NPU timing =="
  exit 0
fi

echo "== 5/5 interleaved NPU timing ($ROUNDS rounds x $ITERS iters) =="
echo "  WARNING: NPU must be otherwise idle (no zaya/1bit server) for valid numbers."
declare -a P_RES C_RES
for r in $(seq 1 "$ROUNDS"); do
  p=$(run_arm "$W/bench" "$P/final_peano.xclbin" "$P/insts_peano.txt" "$ITERS")
  c=$(run_arm "$W/bench" "$C/final_chess.xclbin" "$C/insts_chess.txt" "$ITERS")
  echo "  round $r: peano [$p]  chess [$c]"
  P_RES+=("$p"); C_RES+=("$c")
done

echo
echo "── results ────────────────────────────────────────────────────────"
printf "  %-10s %-22s %-22s\n" "round" "peano (ms, GOP/s)" "chess (ms, GOP/s)"
for i in $(seq 0 $((ROUNDS-1))); do
  printf "  %-10s %-22s %-22s\n" "$((i+1))" "${P_RES[$i]}" "${C_RES[$i]}"
done

# means over PASS rounds
mean() { # "$1" = list of "PASS ms gops"
  local n=0 ms=0 gp=0 tok
  for tok in "$@"; do
    [ "${tok:0:4}" = PASS ] || continue
    set -- $tok
    ms=$(awk -v a="$ms" -v b="$2" 'BEGIN{print a+b}')
    gp=$(awk -v a="$gp" -v b="$3" 'BEGIN{print a+b}')
    n=$((n+1))
  done
  [ "$n" -gt 0 ] && awk -v ms="$ms" -v gp="$gp" -v n="$n" \
    'BEGIN{printf "%.3f ms, %.1f GOP/s (%d rounds)", ms/n, gp/n, n}' || echo "no PASS rounds"
}
P_MEAN=$(mean "${P_RES[@]}"); C_MEAN=$(mean "${C_RES[@]}")
echo "  ───────────────────────────────────────────────────────────────"
echo "  peano mean: $P_MEAN"
echo "  chess mean: $C_MEAN"
if echo "$P_MEAN" | grep -q GOP && echo "$C_MEAN" | grep -q GOP; then
  p_ms=${P_MEAN%% *}; c_ms=${C_MEAN%% *}
  echo "  chess/peano ms ratio: $(awk -v a="$c_ms" -v b="$p_ms" 'BEGIN{printf "%.3f", a/b}')  (1.00 = identical, >1 = chess slower)"
  echo "  peano/chess GOP/s ratio: $(awk -v a="${P_MEAN#*, }" -v b="${C_MEAN#*, }" 'BEGIN{split(a,x," "); split(b,y," "); printf "%.3f", x[1]/y[1]}')"
fi
echo
echo "Artifacts kept in: $W  (bench + both xclbins + insts + aiecc logs)"
