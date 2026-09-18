#!/usr/bin/env bash
# npu_xclbin_census_selfcheck.sh — the --pairing RISK branch must fire for the slots
# the engine actually loads (#2584).
#
# engine_slot_tokens() matched "([A-Z][A-Z0-9_]{2,})" — three characters or more —
# while five of the legacy path's six slots are "O", "G", "U", "D" and "GU". An
# artifact whose instruction file had gone missing therefore printed
#
#     note   final_i8_D_qwen3_4b.xclbin — no engine slot token matches this name
#
# and the summary line stayed at "0 of them belong to a slot the engine loads". That
# is the one row the mode exists to report: init_i8() falls back to the runtime
# generator when the .txt is absent, and that generator emits single-core-row
# instructions which, against a multi-row (v27) xclbin, compute the WRONG result
# rather than a slower one. So a false "note" here is a wrong-answer row reported as
# somebody else's artifact.
#
# The fixture is synthetic and every assertion reads the lines the real tool prints,
# so this needs no NPU, no xrt, and none of the repo's artifacts.
#
# CENSUS is the seam, for pointing the same assertions at a stub (case H) or at a
# pre-fix copy.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CENSUS="${CENSUS:-$REPO/Testing/npu_xclbin_census.py}"
[ -f "$CENSUS" ] || { echo "FAIL: no $CENSUS"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok()  { printf '  ok   %-54s %s\n' "$1" "$2"; }
bad() { printf '  FAIL %-54s %s\n' "$1" "$2"; fail=1; }
expect() { if [ "$2" = "$3" ]; then ok "$1" "$2"; else bad "$1" "$2 != $3"; fi; }

XD="$TMP/xclbins"
mkdir -p "$XD"
# A pair for every legacy slot, plus ATTN — which no engine source in the tree names,
# so it is the fixture's negative control (see check_xclbin_provenance.py:212-215,
# which recorded the same absence independently).
apair() { : > "$XD/final_i8_$1_fixture.xclbin"; : > "$XD/insts_i8_$1_fixture.txt"; }
for slot in QKV O G U D GU ATTN; do apair "$slot"; done
unpair() { rm -f "$XD/insts_i8_$1_fixture.txt"; }
repair() { : > "$XD/insts_i8_$1_fixture.txt"; }
audit() { python3 "$CENSUS" --pairing --xclbins "$XD" 2>&1; }
risks() { printf '%s\n' "$1" | grep -c '^  RISK   ' || true; }

# ── A: every xclbin paired — the mode must not invent a RISK row ────────────────
out="$(audit)"; rc=$?
expect "A fully paired fixture reports 0 risky" "$(risks "$out")" "0"
case "$out" in
  *"0 of them belong to a slot the engine loads"*) ok "A summary line says 0" "0 of them" ;;
  *) bad "A summary line says 0" "$(printf '%s' "$out" | tail -1)" ;;
esac
expect "A exits 0 (report-only, not a gate)" "$rc" "0"

# ── B: the five slots the old pattern could not see ─────────────────────────────
missing=0
rc=0
for slot in O G U D GU; do
  unpair "$slot"
  out="$(audit)"; rc=$?
  case "$out" in
    *"RISK   final_i8_${slot}_fixture.xclbin"*"slot '${slot}' IS loaded"*)
      ok "B ${slot}: an xclbin with no insts is RISK" "RISK ${slot}" ;;
    *) bad "B ${slot}: an xclbin with no insts is RISK" "$(printf '%s' "$out" | grep "final_i8_${slot}_fixture" | cut -c1-60)" ;;
  esac
  missing=$((missing+1))
done
expect "B counts every one of them in the summary" "$(risks "$out")" "$missing"
expect "B exits 0" "$rc" "0"

# ── C: the negative control — ATTN is named by no engine, so it stays a note ─────
for slot in O G U D GU; do repair "$slot"; done
unpair ATTN
out="$(audit)"
case "$out" in
  *"note   final_i8_ATTN_fixture.xclbin"*) ok "C ATTN stays a note" "note ATTN" ;;
  *) bad "C ATTN stays a note" "$(printf '%s' "$out" | grep ATTN | cut -c1-60)" ;;
esac
expect "C and is not counted as risky" "$(risks "$out")" "0"
repair ATTN

# ── D: the branch that already worked must keep working ────────────────────────
unpair QKV
out="$(audit)"
case "$out" in
  *"RISK   final_i8_QKV_fixture.xclbin"*) ok "D QKV is still RISK" "RISK QKV" ;;
  *) bad "D QKV is still RISK" "$(printf '%s' "$out" | grep QKV | cut -c1-60)" ;;
esac
repair QKV

# ── E: the classifier's own token set — the literal floor ───────────────────────
# Asserted against engine_slot_tokens() directly, so a regex or path-glob regression
# fails here even if the printed lines above happen to come out right.
toks="$(python3 - "$CENSUS" "$REPO" <<'PY' 2>&1
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("census", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print(" ".join(sorted(mod.engine_slot_tokens(Path(sys.argv[2])))))
PY
)"
for slot in QKV O G U D GU; do
  case " $toks " in
    *" $slot "*) ok "E token set contains '$slot'" "present" ;;
    *) bad "E token set contains '$slot'" "MISSING from: $(printf '%s' "$toks" | cut -c1-50)" ;;
  esac
done

# ── F: the token set is read from more than one engine ────────────────────────
# npu_engine_hybrid.cpp and npu_engine_cb.cpp hand whole paths to .init() — "KV" is
# named only there, so a single-file scan cannot produce it.
case " $toks " in
  *" KV "*) ok "F a slot named only outside npu_engine_universal" "KV present" ;;
  *) bad "F a slot named only outside npu_engine_universal" "KV missing" ;;
esac

# ── G: no false RISK — ATTN must not be in the token set ──────────────────────
case " $toks " in
  *" ATTN "*) bad "G nothing names ATTN" "ATTN present (a comment read as code?)" ;;
  *) ok "G nothing names ATTN" "absent" ;;
esac

# ── H: control — a classifier that cannot see the slots must FAIL case B ───────
# The stub is the transcript the real tool printed before the fix, on this fixture's
# shape: every unpaired xclbin a "note", summary 0. If the assertions in B and D pass
# against it, they are not measuring anything.
if [ "${SELFTEST_STUB:-0}" = "1" ]; then
  exit "$fail"
fi
cat > "$TMP/stub.py" <<'STUB'
#!/usr/bin/env python3
"""Stands in for any classifier that cannot name the legacy slots."""
import argparse, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--xclbins"); ap.add_argument("--pairing", action="store_true")
a = ap.parse_args()
xd = Path(a.xclbins)
xcl = {f.name for f in xd.glob("final_i8_*.xclbin")}
insts = {f.name for f in xd.glob("insts_i8_*.txt")}
missing = sorted(n for n in xcl if "insts_i8_" + n[len("final_i8_"):-len(".xclbin")] + ".txt" not in insts)
for n in missing:
    print(f"  note   {n} — no engine slot token matches this name")
print(f"\n  0 of them belong to a slot the engine loads.")
sys.exit(0)
STUB
if CENSUS="$TMP/stub.py" SELFTEST_STUB=1 bash "${BASH_SOURCE[0]}" >/dev/null 2>&1; then
  bad "H the pre-fix stub is caught" "the assertions passed against it"
else
  ok "H the pre-fix stub is caught" "assertions fail against it"
fi

if [ "$fail" -eq 0 ]; then
  echo "✓ npu_xclbin_census: RISK branch fires for every legacy slot"
else
  echo "✗ npu_xclbin_census"
fi
exit "$fail"
