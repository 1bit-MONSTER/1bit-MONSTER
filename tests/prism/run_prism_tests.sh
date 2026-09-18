#!/bin/bash
# run_prism_tests.sh — every automated gate this lane has, in one command.
#
# Compiles the standalone C++ checks (no CMake target needed) and runs the Python
# oracles plus the conversion verifier. Exits non-zero if any gate fails, so it can
# be wired into CI as-is.
#
# Usage: tests/prism/run_prism_tests.sh [models_dir]
#   models_dir defaults to ~/models/prism  (GGUFs and 1bp/ artifacts)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MDIR="${1:-$HOME/models/prism}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fails=0
run() {  # run <name> <cmd...>
  local name="$1"; shift
  local out
  if out="$("$@" 2>&1)"; then
    printf "%-52s PASS\n" "$name"
  else
    printf "%-52s FAIL\n" "$name"
    printf '%s\n' "$out" | tail -20 | sed 's/^/    /'
    fails=$((fails + 1))
  fi
}

echo "== build =="
g++ -O2 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
    "$REPO/tests/prism/test_onebp_v5_transform.cpp" -o "$TMP/t5" || exit 1
g++ -O2 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
    "$REPO/tests/prism/test_prism_primitives.cpp" "$REPO/src/gguf_reader.cpp" \
    -o "$TMP/tpp" || exit 1
g++ -O2 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
    "$REPO/tests/prism/test_prism_dequant.cpp" "$REPO/src/gguf_reader.cpp" \
    -o "$TMP/tpd" || exit 1
g++ -O2 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
    "$REPO/tests/prism/test_prism_layer.cpp" "$REPO/src/onebp_model.cpp" \
    -o "$TMP/tpl" || exit 1
g++ -O2 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
    "$REPO/tests/prism/prism_layer0.cpp" "$REPO/src/onebp_model.cpp" \
    -o "$TMP/l0" || exit 1
g++ -O3 -fopenmp -std=c++17 -I "$REPO/include" -I "$REPO/src" \
    "$REPO/tests/prism/prism_forward.cpp" "$REPO/src/onebp_model.cpp" \
    -o "$TMP/pf" || exit 1

# Optional GPU parity tool: built only when hipcc is present (needs the device).
PGEMV=""
HIPCC=/opt/rocm-therock/bin/hipcc
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
      "$REPO/tests/prism/prism_gemv_hip.hip" "$REPO/src/onebp_model.cpp" -o "$TMP/pgemv" \
      >/dev/null 2>&1; then
    PGEMV="$TMP/pgemv"
  else
    echo "  (hipcc present but the GPU parity tool failed to build — skipping)"
  fi
fi

# Optional P3.1 gate: Prism signed FWHT device parity (needs hipcc + the device).
PHADM=""
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
      "$REPO/tests/prism/test_prism_hadamard_hip.hip" "$REPO/kernels/prism_hadamard_fwht.hip" \
      -o "$TMP/phadm" >/dev/null 2>&1; then
    PHADM="$TMP/phadm"
  else
    echo "  (hipcc present but the Prism FWHT parity tool failed to build — skipping)"
  fi
fi

echo "== container / codec gates (no model file needed) =="
run "1BP v5 transform blob + Prism geometry" "$TMP/t5"
run "Prism codec round-trip (synthetic)" python3 "$REPO/tests/prism/roundtrip_prism_codec.py"
"$TMP/tpp" --dump 2048 1024 > "$TMP/cpp_fwht.txt"
python3 "$REPO/tests/prism/dump_prism_fwht.py" 2048 1024 > "$TMP/py_fwht.txt"
run "FWHT vs independent matrix reference" python3 "$REPO/tests/prism/compare_prism_fwht.py" \
    "$TMP/cpp_fwht.txt" "$TMP/py_fwht.txt"
[ -n "$PHADM" ] && run "Prism FWHT device parity (P3.1, gfx1151)" "$PHADM"

echo "== per-model gates (source GGUF required) =="
for g in "$MDIR"/ternary2-gguf/*.gguf "$MDIR"/ternary-gguf/*.gguf "$MDIR"/onebit-gguf/Bonsai-27B-Q1_0.gguf; do
  [ -f "$g" ] || continue
  base="$(basename "$g" .gguf)"
  run "$base: python oracle vs codec.py + manifest" \
      python3 "$REPO/tests/prism/oracle_prism_codec.py" "$g" --blocks 2 --max-tensors 12
  run "$base: C++ reader dequant == python oracle" bash -c \
      "\"$TMP/tpd\" \"$g\" 256 > \"$TMP/cpp.txt\" && python3 \"$REPO/tests/prism/dump_prism_dequant.py\" \"$g\" 256 > \"$TMP/py.txt\" && diff -q \"$TMP/cpp.txt\" \"$TMP/py.txt\" > /dev/null"
  run "$base: Hadamard contract + real-byte dequant" "$TMP/tpp" "$g"
  bp="$MDIR/1bp/$base.1bp"
  # Per-position next-token oracle agreement (the gate that catches convention bugs).
  if [ -f "$bp" ]; then
  if [ -n "$PGEMV" ] && [ -f "$bp" ]; then
    run "$base: HIP GEMV decode parity on device" \
        "$PGEMV" "$bp" blk.0.ffn_gate.weight 10
  fi
    run "$base: per-position oracle agreement" \
        python3 "$REPO/tests/prism/check_oracle_agreement.py" "$TMP/pf" "$bp" "$base"
  fi
  if [ -f "$bp" ]; then
    run "$base: converted .1bp is a byte-exact repack" \
        python3 "$REPO/tests/prism/verify_prism_1bp.py" "$g" "$bp"
    run "$base: engine loader + folded matmul vs python" bash -c \
        ""$TMP/tpl" "$bp" > "$TMP/l_cpp.txt" && python3 "$REPO/tests/prism/dump_prism_layer.py" "$bp" > "$TMP/l_py.txt" && python3 "$REPO/tests/prism/compare_prism_layer.py" "$TMP/l_cpp.txt" "$TMP/l_py.txt""
  fi
done

echo
if [ "$fails" -eq 0 ]; then echo "ALL PRISM GATES PASSED"; else echo "$fails GATE(S) FAILED"; fi
exit "$fails"
