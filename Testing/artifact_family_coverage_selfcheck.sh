#!/usr/bin/env bash
# artifact_family_coverage_selfcheck.sh — check_artifact_families.py must flag an
# engine-constructible family with zero tracked members, and must accept a family
# that is declared build-only. Issue #2601: bf16 tiles were deleted (a81662ab8) and
# no check noticed, because nothing mapped engine construction sites to tracked
# artifacts. A gate that cannot fail is not a gate, so each case asserts the exit
# code AND a marker in the output.
#
# Run: bash Testing/artifact_family_coverage_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$REPO/engine/npu/tests/check_artifact_families.py"
[ -f "$TOOL" ] || { echo "FAIL: no $TOOL"; exit 1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok()   { echo "  ok   $1"; }
bad()  { echo "  FAIL $1"; fail=1; }

# ── fixture: an engine source with two construction tags and both prefixes ──
cat > "$T/engine.cpp" <<'EOF'
    auto xpb=[&](const char*t, int K, int N){return xd+"/final_bf16_"+t+"_K"+std::to_string(K)+"_N"+std::to_string(N)+".xclbin";};
    auto ipb=[&](const char*t, int K, int N){return xd+"/insts_bf16_"+t+"_K"+std::to_string(K)+"_N"+std::to_string(N)+".txt";};
    if(!init_bf16(bcq,"QKV",cfg.xclbin_qkv_k,cfg.xclbin_qkv_n)){return 1;}
    if(!init_bf16(bcd,"D",cfg.xclbin_d_k,cfg.xclbin_d_n)){return 1;}
EOF

# ── case 1: both families tracked → 0 ──
printf '%s\n' \
  'engine/npu/xclbins/final_bf16_QKV_K1024_N4096.xclbin' \
  'engine/npu/xclbins/insts_bf16_QKV_K1024_N4096.txt' \
  'engine/npu/xclbins/final_bf16_D_K6144_N2048.xclbin' \
  'engine/npu/xclbins/insts_bf16_D_K6144_N2048.txt' > "$T/tracked_all.txt"
out="$(python3 "$TOOL" --root "$T" --source "$T/engine.cpp" --decl "$T/none.json" --tracked-file "$T/tracked_all.txt" 2>&1)"; rc=$?
if [ $rc -eq 0 ] && grep -q "artifact-families: OK" <<<"$out"; then ok "tracked families -> rc=0"; else bad "tracked families rc=$rc"; echo "$out" | sed 's/^/      /'; fi

# ── case 2: D family dropped to zero, undeclared → 1 with the offending family ──
grep -v '_D_' "$T/tracked_all.txt" > "$T/tracked_noD.txt"
out="$(python3 "$TOOL" --root "$T" --source "$T/engine.cpp" --decl "$T/none.json" --tracked-file "$T/tracked_noD.txt" 2>&1)"; rc=$?
if [ $rc -eq 1 ] && grep -q "final_bf16_D" <<<"$out" && grep -q "ZERO TRACKED MEMBERS" <<<"$out"; then ok "undeclared zero-member family -> rc=1"; else bad "undeclared zero-family rc=$rc"; echo "$out" | sed 's/^/      /'; fi
if grep -q "insts_bf16_D" <<<"$out"; then ok "both prefixes of the dropped family are named"; else bad "insts_bf16_D not reported"; fi

# ── case 3: same family declared build-only → 0 ──
cat > "$T/decl.json" <<'EOF'
{"build_only": [["final_bf16_", "D"], ["insts_bf16_", "D"]]}
EOF
out="$(python3 "$TOOL" --root "$T" --source "$T/engine.cpp" --decl "$T/decl.json" --tracked-file "$T/tracked_noD.txt" 2>&1)"; rc=$?
if [ $rc -eq 0 ] && grep -q "build-only" <<<"$out"; then ok "declared build-only family -> rc=0"; else bad "declared build-only rc=$rc"; echo "$out" | sed 's/^/      /'; fi

# ── case 4: a blind source scan must not pass vacuously → 2 ──
echo 'int nothing_here;' > "$T/empty.cpp"
out="$(python3 "$TOOL" --root "$T" --source "$T/empty.cpp" --decl "$T/none.json" --tracked-file "$T/tracked_all.txt" 2>&1)"; rc=$?
if [ $rc -eq 2 ] && grep -q "blind" <<<"$out"; then
    ok "blind scan -> rc=2 (does not pass vacuously)"
else
    bad "blind-scan handling rc=$rc"; echo "$out" | sed 's/^/      /'
fi

if [ "$fail" -eq 0 ]; then echo "artifact_family_coverage_selfcheck: PASS"; else echo "artifact_family_coverage_selfcheck: FAIL"; exit 1; fi
