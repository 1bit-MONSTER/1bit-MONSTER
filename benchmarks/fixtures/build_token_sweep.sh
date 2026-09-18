#!/usr/bin/env bash
# FIRST-TOKEN SWEEP -- the question §140 leaves open, and the one that unifies the two lanes.
#
# The peer lane's §140 showed, at npt=448, that the bf16 plateau moves when ONLY the first token
# changes, and that changing it 58907 -> 220 makes the path CORRECT. So a plateau is not a
# property of the LENGTH; it is a property of the FIRST TOKEN. And the token both lanes picked
# to escape the token-16 trap (58907) is itself one of the degenerating values.
#
# This builds FIXED-LENGTH fixtures whose tail is constant and whose FIRST token varies, at two
# lengths: 32 (this lane's plateau) and 448 (the other lane's). Running the same token list at
# both lengths separates two hypotheses that the single-length result cannot:
#   * the degenerating set is a property of the TOKEN  -> same tokens degenerate at both lengths
#   * it is a property of the (token, length) pair      -> the sets differ
set -euo pipefail
SRC32=${1:-/tmp/N32.txt}
cd "$(dirname "$0")"
python3 - "$SRC32" <<'PY'
import sys, os
src=sys.argv[1]
base=open(src).read().split()
tokens=['16','220','58907','100','1024','12345','4096','777']
for t in tokens:
    row=base[:]; row[0]=t
    open(f'/tmp/S32_{t}.txt','w').write(' '.join(row))
print(f"wrote /tmp/S32_{{ {' , '.join(tokens)} }}.txt  (length 32, tail fixed)")
PY
echo "for length 448, supply the peer lane's G448 fixture as \$2 and the same loop applies"
