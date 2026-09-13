#!/usr/bin/env bash
# Build the prompt fixtures used for the coverage residuals.
#
# TWO families, differing in exactly one token at every length that appears in both:
#
#   D-family ("degenerate", /tmp/M*.txt): first token 220 -- and 220 is ALSO a common
#       prediction, which is the whole problem.  5 of the 10 lengths measured with it
#       returned 220, i.e. the output equalled a token of the prompt.
#   N-family (/tmp/N*.txt): first token 58907.  NOT DESCRIBED AS "NON-DEGENERATE" -- it was,
#       and the peer lane's section 140 showed 58907 is itself a DEGENERATING first token for
#       the bf16 path (changing it to 220 made that path correct).  It is a SECOND PROBE, not a
#       control: "the reference varies with length" is not evidence the PATH is well-conditioned.
#       See build_token_sweep.sh for the sweep that replaces this family as a control.
#
# Every pair below shares its LAST token and differs ONLY in the first, so a single pair
# of runs is a controlled experiment for whether the output tracks the prompt's own ids:
#
#   length   D-family first/last   N-family first/last
#     32         220 / 17              58907 / 17
#     64         220 / 49891           58907 / 49891
#    128         220 / 220             58907 / 220
#
# If native(N) != native(D) the output is tracking the fixture and the D-family numbers are
# artifacts; if native(N) == native(D), the first token is irrelevant and the value is real.
#
# Rule this file exists to enforce: RECORD first AND last for every fixture, and assert them
# at run time.  A hardcoded first-token value in an analysis script is how this lane produced
# and then had to retract a finding.
set -euo pipefail
SRC=${1:-/tmp/ids_1024.txt}
[ -f "$SRC" ] || { echo "need a token file: $SRC" >&2; exit 1; }
for n in 16 32 48 64 128 192 256; do
  python3 - "$SRC" "$n" > "/tmp/N$n.txt" <<'PY'
import sys
toks = open(sys.argv[1]).read().split()
n = int(sys.argv[2])
out = toks[:n]
out[0] = '58907'          # fixed probe token, NOT a known-good one -- see header
print(' '.join(out))
PY
done
echo "wrote /tmp/N{16,32,48,64,128,192,256}.txt"
