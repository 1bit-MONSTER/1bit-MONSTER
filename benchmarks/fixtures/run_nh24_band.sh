#!/usr/bin/env bash
# THE nh24 BAND, in one place: provenance re-take + edge bisect + token sweep.
#
# PROVENANCE FIRST. The band [16,128] rests on the 285 sweep, which ran on an
# earlier build. Only npt=32 and 64 were re-measured on today's binary (both 220,
# two different first tokens). Every other row is older. The gates that were
# re-verified today cover the six SUPPORTED models, not Phi4 -- so the band's
# anchors get re-taken here rather than trusted.
#
# No flags beyond NPU_PREFILL_BF16=1. Load printed at both ends.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
B=./engine/npu/build/npu_engine_phi4_mini_4b
M=~/.config/flm/models/Phi4-mini-Instruct-NPU2
tok() { grep -aoE '\[0\] boot=[0-9]+|\[1\] [0-9]+' | head -1; }
show() { printf "   %-16s first=%-6s last=%-7s -> " "$1" \
    "$(tr -s ' \n' '\n' < /tmp/$1.txt | grep . | head -1)" \
    "$(tr -s ' \n' '\n' < /tmp/$1.txt | grep . | tail -1)"
  NPU_PREFILL_BF16=1 $B $M/model.q4nx 1 /tmp/$1.txt 2>/dev/null | tok; }
echo "load: clang=$(ps -eo comm 2>/dev/null | grep -c clang-23) $(uptime | sed 's/.*load average: //')"
echo "A. PROVENANCE re-take of the band's anchors (285 values in brackets)"
for f in M8 N16 N48 N128 N192 N256; do show $f; done
echo "      [285: 8->683  16->220  48->220  128->220  192->85  256->6573]"
echo "B. EDGE bisect in (128,192] -- the degeneracy should end somewhere here"
for f in E130 E144 E160 E176 E192; do show $f; done
echo "C. TOKEN sweep at fixed length 32 -- input-invariance should show as a flat column"
for t in 16 220 58907 100 1024 12345 4096 777; do show S32_$t; done
echo "load: clang=$(ps -eo comm 2>/dev/null | grep -c clang-23) $(uptime | sed 's/.*load average: //')"
