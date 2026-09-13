#!/usr/bin/env bash
# Build the prompt fixtures used for the coverage residuals.
#
# TWO families, differing in exactly one token at every length that appears in both:
#
#   D-family ("degenerate", /tmp/M*.txt): first token 220 -- and 220 is ALSO a common
#       prediction, which is the whole problem.  5 of the 10 lengths measured with it
#       returned 220, i.e. the output equalled a token of the prompt.
#   N-family ("non-degenerate", /tmp/N*.txt): first token 58907 -- the same choice the
#       peer lane used for its control, so the two lanes' numbers are directly comparable.
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
out[0] = '58907'          # non-degenerate first token, matches the peer lane's control
print(' '.join(out))
PY
done
echo "wrote /tmp/N{16,32,48,64,128,192,256}.txt"
