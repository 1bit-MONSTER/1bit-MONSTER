#!/bin/bash
# check_aie_dialect.sh — catch a generator ↔ aiecc dialect mismatch BEFORE a build.
#
# Why this exists (measured 2026-09-15, strixhalo):
#
#   The generators emit the PRE-#3306 `aie.dma_bd` form, where sizes/strides/
#   offset/len are a literal attribute list — `[<size = 4, stride = 32768>, …]`.
#   Upstream #3306 (mlir-aie 398f7f704, "[dyn-seq] Convert aie.dma_bd
#   sizes/strides/offset/len to SSA operands (DynamicIndexList)") is what moved
#   them TO SSA operands, so a root that HAS that change rejects the generated
#   literal form at parse time with a bare
#
#       loc("d.mlir":1059:45): error: expected ')'
#       Error parsing MLIR file
#
#   at the dma_bd line. That reads like a generator bug. It is not — the generator
#   emits the older form and the root being probed is the newer parser.
#
#   THE MISMATCH IS BETWEEN THE BINDINGS THAT GENERATE AND THE BINARY THAT
#   PARSES, not simply between two installs. `build_xclbins.sh:267` couples them
#   (PYTHONPATH="${AIE_TOOLS_DIR}/python:..."), so the standard flow is
#   internally consistent — but mixing them is easy and the failure is not
#   self-explanatory. Concretely, with the mlir-aie venv's python generating and
#   three different aie-opt binaries parsing the SAME file:
#
#       ~/mlir-aie/install_tmp/bin/aie-opt  PARSES   <- pre-#3306 parser
#       ~/mlir-aie/build_tmp/bin/aie-opt    rejects: error: expected ')'  <- post-#3306
#       ~/mlir-aie/iron/bin/aie-opt         rejects: error: expected ')'  <- post-#3306
#
#   Which is the other way round from how this header read for a while: the root
#   that REJECTS is the one that postdates #3306, not the one that predates it. The
#   operational advice is unchanged either way (install_tmp is the root to use), but
#   the diagnostic used to point a reader at the wrong upgrade. This script's own
#   count is the check: the form it greps for ends in `[<`, which is the literal
#   attribute list and cannot match an SSA operand.
#
#   (Each root also ships its own `python/`; generated with THAT, an old aiecc
#   accepts its own older form — which is exactly why testing each root against
#   itself proves nothing. This script generates ONCE and parses with each.)
#
#   `engine/npu/build_xclbins.sh:19-22` already names the known-good root
#   ("export AIE_TOOLS_DIR=~/mlir-aie/install_tmp"), so the pairing is
#   documented; this script makes a wrong one fail fast and legibly instead of
#   inside a build.
#
# Usage:
#   ./check_aie_dialect.sh                 # probe the known candidate roots
#   ./check_aie_dialect.sh <root> [...]    # probe specific aietools roots
#   PYTHON=<interpreter> ./check_aie_dialect.sh    # choose the generating python
#
# Exit: 0 if at least one probed root parses the generated MLIR, 1 otherwise.

set -uo pipefail

GENERATORS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The generating interpreter matters as much as the aiecc: it must have the
# mlir-aie bindings, and ml_dtypes (the system python3 lacks it, which is a
# second, unrelated failure mode).
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for cand in "$HOME/mlir-aie/.venv/bin/python3" "$HOME/iron-venv/bin/python3" python3; do
        if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
    done
fi

GEN="$GENERATORS_DIR/n1_core_i8_v27.py"
[ -f "$GEN" ] || GEN="$GENERATORS_DIR/n1_core_i8_v24.py"
if [ ! -f "$GEN" ]; then
    echo "check_aie_dialect: no generator found next to $GENERATORS_DIR" >&2
    exit 2
fi

if [ "$#" -gt 0 ]; then
    ROOTS=("$@")
elif [ -n "${AIE_TOOLS_DIR:-}" ]; then
    ROOTS=("$AIE_TOOLS_DIR")
else
    ROOTS=("$HOME/mlir-aie/install_tmp" "$HOME/mlir-aie/build_tmp" "$HOME/mlir-aie/install" "$HOME/iron")
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

MLIR="$TMP/probe.mlir"
# The v27 recipe's own shape: it exercises the dynamic dma_bd form, and cols=8
# rows=4 keeps the generator's divisibility checks satisfied.
GEN_ARGS=(-M 128 -K 2048 -N 4096 -m 32 -k 64 -n 128 -c 8 -r 4 -b 5)

echo "check_aie_dialect: generator = $(basename "$GEN")"
echo "                   python    = $PYTHON"
echo "                   args      = ${GEN_ARGS[*]}"
echo

if ! timeout 300 "$PYTHON" "$GEN" "${GEN_ARGS[@]}" > "$MLIR" 2> "$TMP/gen.err"; then
    echo "check_aie_dialect: the generator itself failed:" >&2
    tail -3 "$TMP/gen.err" >&2
    exit 2
fi
n_dma=$(grep -c 'aie\.dma_bd' "$MLIR" 2>/dev/null || echo 0)
# The pattern ends in `[<` — the literal attribute list `[<size = …, stride = …>]`.
# That is the PRE-#3306 form; the label below used to call it the post-#3306
# "operand" form, which the pattern itself cannot match.
n_lit=$(grep -c 'aie\.dma_bd(.*, \[<' "$MLIR" 2>/dev/null || echo 0)
echo "  generated $(wc -c < "$MLIR") bytes: $n_dma aie.dma_bd, $n_lit in the literal-attribute (pre-#3306) form"
echo

passed=0
for root in "${ROOTS[@]}"; do
    short="$(basename "$root")"
    aie_opt="$root/bin/aie-opt"
    if [ ! -x "$aie_opt" ]; then
        printf '  %-14s SKIP    (no bin/aie-opt)\n' "$short"
        continue
    fi
    if timeout 300 "$aie_opt" "$MLIR" -o /dev/null > "$TMP/$short.log" 2>&1; then
        printf '  %-14s OK      (parses this MLIR)\n' "$short"
        passed=$((passed + 1))
    else
        printf '  %-14s REJECT  %s\n' "$short" "$(grep -m1 -o "error: .*" "$TMP/$short.log" | cut -c1-52)"
        if grep -q "dma_bd" "$TMP/$short.log"; then
            printf '  %-14s         ^ postdates upstream #3306 (SSA operands), so it rejects\n' ""
            printf '  %-14s           the literal dma_bd form this generator emits. See the header.\n' ""
        fi
    fi
done

echo
if [ "$passed" -gt 0 ]; then
    echo "RESULT: PASS — $passed probed root(s) parse the MLIR this generator produces"
    echo "  Known-good root on this box: export AIE_TOOLS_DIR=\$HOME/mlir-aie/install_tmp"
    exit 0
fi
echo "RESULT: FAIL — no probed aiecc parses the MLIR this generator produces" >&2
echo "  A build would fail with a bare \"expected ')'\" at the dma_bd line." >&2
echo "  Known-good root on this box: AIE_TOOLS_DIR=\$HOME/mlir-aie/install_tmp" >&2
exit 1
