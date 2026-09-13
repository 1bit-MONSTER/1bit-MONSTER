#!/usr/bin/env bash
# The 5 runs that close the nh24 residual. Device-free to prepare; ~1 min of NPU time.
#
#  A. CHECK 1 -- last-token-only pair, same length, same first token, ONE token differs:
#        native(P32a) vs native(P32b)   differ => the output consumes the last token
#                                       equal  => it does not (truncation-shaped)
#  B. CHECK 2 -- already answered device-free: npt=2/4/8 give 6304/198/683, NOT 220,
#     so "the computation is done at 128 regardless of input length" is refuted.
#  C. DISCRIMINATOR between "padded to 128" and "a constant": if padded, the answer at
#     npt=32 should equal FLM's answer AT npt=128 for the same prompt. Take FLM(N128).
set -uo pipefail
cd "$(dirname "$0")/../.."
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
B=./engine/npu/build/npu_engine_phi4_mini_4b
M=~/.config/flm/models/Phi4-mini-Instruct-NPU2
tok() { grep -aoE '\[0\] boot=[0-9]+|\[1\] [0-9]+' | head -1; }
echo "load: clang=$(ps -eo comm 2>/dev/null | grep -c clang-23) $(uptime | sed 's/.*load average: //')"
echo "A. native, last-token-only pairs (32; plus 8 and 128, which the peer's note puts at risk)"
for f in P8a P8b P32a P32b P128a P128b N32; do
  printf "   %-5s first=%-6s last=%-6s -> " "$f" \
    "$(tr -s ' \n' '\n' < /tmp/$f.txt | grep . | head -1)" \
    "$(tr -s ' \n' '\n' < /tmp/$f.txt | grep . | tail -1)"
  NPU_PREFILL_BF16=1 $B $M/model.q4nx 1 /tmp/$f.txt 2>/dev/null | tok
done
echo "C. FLM reference at npt=128 (same prefix) and at 32, 8"
for f in N128 N32 N8; do
  printf "   FLM(%-5s) -> " "$f"
  env -u NPU_PREFILL_BF16 NPU_FLM_PREFILL=1 $B $M/model.q4nx 1 /tmp/$f.txt 2>/dev/null | tok
done

# --- degeneracy-edge bisect (device-free to prepare) ---
# MY lane's edge lies in (128, 192]: 128 is invariant at 220, 192 is 85.  6 runs locate it.
for L in 130 144 160 176 192 256; do
  printf "   E%-4s -> " "$L"
  env -u NPU_PREFILL_BF16 NPU_PREFILL_BF16=1 $B $M/model.q4nx 1 /tmp/E$L.txt 2>/dev/null | tok
done
