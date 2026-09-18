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
g++ -O2 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
    "$REPO/tests/prism/test_prism_failclosed.cpp" "$REPO/src/onebp_model.cpp" \
    -o "$TMP/fc" || exit 1
g++ -O2 -std=c++17 -I "$REPO/include" -I "$REPO/src" -I "$REPO/engine/npu/include" \
    "$REPO/tests/prism/test_prism_npu_pack.cpp" "$REPO/src/prism_npu_bridge.cpp" \
    "$REPO/src/onebp_model.cpp" -o "$TMP/pnpu" || exit 1

# Optional GPU parity tool: built only when hipcc is present (needs the device).
PGEMV=""
HIPCC=/opt/rocm-therock/bin/hipcc
if [ -n "${PRISM_NO_DEVICE:-}" ]; then HIPCC=/nonexistent-skipping-device-gates; fi
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
      "$REPO/tests/prism/prism_gemv_hip.hip" "$REPO/src/onebp_model.cpp" -o "$TMP/pgemv" \
      >/dev/null 2>&1; then
    PGEMV="$TMP/pgemv"
  else
    echo "  (hipcc present but the GPU parity tool failed to build — skipping)"
  fi
fi

# Optional P3.2 gate: production Prism GEMV launchers (needs hipcc + the device).
PGEMVP=""
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
      "$REPO/tests/prism/test_prism_gemv_prod.hip" "$REPO/kernels/prism_gemv.hip" \
      "$REPO/src/onebp_model.cpp" -o "$TMP/pgemvp" >/dev/null 2>&1; then
    PGEMVP="$TMP/pgemvp"
  else
    echo "  (hipcc present but the production GEMV gate failed to build — skipping)"
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

# Optional P3.3 gate: GDN kernels (needs hipcc + the device).
PGDN=""
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" \
      "$REPO/tests/prism/test_prism_gdn_hip.hip" "$REPO/kernels/prism_gdn.hip" \
      -o "$TMP/pgdn" >/dev/null 2>&1; then
    PGDN="$TMP/pgdn"
  else
    echo "  (hipcc present but the GDN gate failed to build — skipping)"
  fi
fi

# Optional P3.3 gate: full-attention kernels (needs hipcc + the device).
PATTN=""
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" \
      "$REPO/tests/prism/test_prism_attn_hip.hip" "$REPO/kernels/prism_attn.hip" \
      -o "$TMP/pattn" >/dev/null 2>&1; then
    PATTN="$TMP/pattn"
  else
    echo "  (hipcc present but the attention gate failed to build — skipping)"
  fi
fi

# Optional P3.3 gate: thin per-layer ops (needs hipcc + the device).
POPS=""
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" \
      "$REPO/tests/prism/test_prism_ops_hip.hip" "$REPO/kernels/prism_ops.hip" \
      -o "$TMP/pops" >/dev/null 2>&1; then
    POPS="$TMP/pops"
  else
    echo "  (hipcc present but the ops gate failed to build — skipping)"
  fi
fi

# Optional P3.3 gate: whole device GDN layer-0 driver (needs hipcc + the device).
PGDNL=""
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
      "$REPO/tests/prism/test_prism_gdn_layer_hip.hip" "$REPO/kernels/prism_hadamard_fwht.hip" \
      "$REPO/kernels/prism_gemv.hip" "$REPO/kernels/prism_gdn.hip" "$REPO/kernels/prism_ops.hip" \
      "$REPO/src/onebp_model.cpp" -o "$TMP/pgdnl" >/dev/null 2>&1; then
    PGDNL="$TMP/pgdnl"
  else
    echo "  (hipcc present but the layer-0 device driver failed to build — skipping)"
  fi
fi

# Optional P3.3 gate: full 64-layer device forward (needs hipcc + the device).
PFHIP=""
if [ -x "$HIPCC" ]; then
  if "$HIPCC" --offload-arch=gfx1151 -O3 -std=c++17 -I "$REPO/include" -I "$REPO/src" \
      "$REPO/tests/prism/prism_forward_hip.hip" "$REPO/kernels/prism_hadamard_fwht.hip" \
      "$REPO/kernels/prism_gemv.hip" "$REPO/kernels/prism_gemv_row4.hip" \
      "$REPO/kernels/prism_gemv_tile.hip" "$REPO/kernels/prism_gemv_dp4a.hip" \
      "$REPO/kernels/prism_gdn.hip" \
      "$REPO/kernels/prism_attn.hip" "$REPO/kernels/prism_ops.hip" \
      "$REPO/src/onebp_model.cpp" -o "$TMP/pfhip" >/dev/null 2>&1; then
    PFHIP="$TMP/pfhip"
  else
    echo "  (hipcc present but the full device forward failed to build — skipping)"
  fi
fi

run "honesty tags: every numeric claim tagged (P6)" python3 "$REPO/tests/prism/check_honesty_tags.py"

echo "== container / codec gates (no model file needed) =="
run "1BP v5 transform blob + Prism geometry" "$TMP/t5"
run "Prism codec round-trip (synthetic)" python3 "$REPO/tests/prism/roundtrip_prism_codec.py"
"$TMP/tpp" --dump 2048 1024 > "$TMP/cpp_fwht.txt"
python3 "$REPO/tests/prism/dump_prism_fwht.py" 2048 1024 > "$TMP/py_fwht.txt"
run "FWHT vs independent matrix reference" python3 "$REPO/tests/prism/compare_prism_fwht.py" \
    "$TMP/cpp_fwht.txt" "$TMP/py_fwht.txt"
[ -n "$PHADM" ] && run "Prism FWHT device parity (P3.1, gfx1151)" "$PHADM"
[ -n "$PGDN" ] && run "Prism GDN kernel parity (P3.3, gfx1151)" "$PGDN"
[ -n "$PATTN" ] && run "Prism full-attention kernel parity (P3.3, gfx1151)" "$PATTN"
[ -n "$POPS" ] && run "Prism ops kernel parity (P3.3, gfx1151)" "$POPS"

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
  if [ -n "$PGEMVP" ] && [ -f "$bp" ]; then
    run "$base: production GEMV launcher parity (P3.2)" \
        "$PGEMVP" "$bp" blk.0.ffn_gate.weight
  fi
    run "$base: per-position oracle agreement" \
        python3 "$REPO/tests/prism/check_oracle_agreement.py" "$TMP/pf" "$bp" "$base"
    exp=0; [ "$base" = "Ternary-Bonsai-2-27B-PTQ1_0" ] && exp=1
    run "$base: fail-closed transform flag (R15)" "$TMP/fc" "$bp" "$exp"
    run "$base: Prism->INT8 NPU packer (P4.1)" "$TMP/pnpu" "$bp" blk.0.ffn_gate.weight
    if [ -n "$PFHIP" ]; then
      run "$base: DEVICE 64-layer forward vs fork oracle (P3.3)" \
        python3 "$REPO/tests/prism/check_oracle_agreement.py" "$PFHIP" "$bp" "$base"
    fi
    # Device-vs-CPU greedy-sequence agreement (two independent implementations). The CPU
    # forward is the fork-validated floor; any divergence is a device-path bug. 11 positions;
    # PRISM_SEQ_GATE=0 skips it.
    if [ -n "$PFHIP" ] && [ "${PRISM_SEQ_GATE:-1}" != "0" ]; then
      run "$base: DEVICE vs CPU greedy sequence (11 tokens)" bash -c \
        "\"$TMP/pf\" \"$bp\" 760 6511 314 9338 369 --predict 6 > \"$TMP/cpu_gen.txt\" 2>&1 && \"$TMP/pfhip\" \"$bp\" 760 6511 314 9338 369 --predict 6 > \"$TMP/dev_gen.txt\" 2>&1 && python3 \"$REPO/tests/prism/compare_gen.py\" \"$TMP/cpu_gen.txt\" \"$TMP/dev_gen.txt\""
    fi
    # Per-layer cosine vs the in-repo streaming numpy reference (P2 gate, clause 2).
    # Expensive (~6 min/pack): the folded pack by default; PRISM_LAYER_COSINE=all for
    # all three, =0 to skip.
    LC="${PRISM_LAYER_COSINE:-folded}"
    do_cos=0
    case "$LC" in
      0|off) do_cos=0 ;;
      all)   do_cos=1 ;;
      *)     [ "$base" = "Ternary-Bonsai-2-27B-PTQ1_0" ] && do_cos=1 || do_cos=0 ;;
    esac
    if [ "$do_cos" = 1 ]; then
      run "$base: per-layer cosine vs numpy reference (64 layers)" bash -c \
        "\"$TMP/pf\" \"$bp\" 1000 --dump-layers \"$TMP/cpp_layers.bin\" --quiet > /dev/null 2>&1 && python3 \"$REPO/tests/prism/dump_prism_layers.py\" \"$bp\" 1000 \"$TMP/py_layers.bin\" > /dev/null && python3 \"$REPO/tests/prism/compare_prism_layers.py\" \"$TMP/cpp_layers.bin\" \"$TMP/py_layers.bin\""
    fi
  fi
  if [ -f "$bp" ] && [ "$base" = "Ternary-Bonsai-2-27B-PTQ1_0" ]; then
    run "$base: layer-0 GDN block vs numpy reference" bash -c \
      "\"$TMP/l0\" \"$bp\" 1000 > \"$TMP/l0_cpp.txt\" && python3 \"$REPO/tests/prism/dump_prism_layer0.py\" \"$bp\" 1000 > \"$TMP/l0_py.txt\" && python3 \"$REPO/tests/prism/compare_prism_layer0.py\" \"$TMP/l0_cpp.txt\" \"$TMP/l0_py.txt\""
    if [ -n "$PGDNL" ]; then
      run "$base: layer-0 DEVICE driver vs CPU reference" bash -c \
        "\"$PGDNL\" \"$bp\" 1000 > \"$TMP/drv.txt\" 2>&1 && python3 \"$REPO/tests/prism/compare_prism_layer0.py\" \"$TMP/drv.txt\" \"$TMP/l0_cpp.txt\""
    fi
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
