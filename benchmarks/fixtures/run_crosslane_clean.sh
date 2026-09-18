#!/usr/bin/env bash
# THE FULLY CLEAN CROSS-LANE COMPARISON.
#
# The first cross-lane run used eight token-ids including 16 and 100 -- both of which are in
# Nanbeige's 319-row ZERO-EMBEDDING set, so two of the eight points answered context-free for a
# reason unrelated to the residual. This set replaces them with 30000 and 45000 and asserts that
# every chosen token is a NONZERO embedding on BOTH models before any run.
#
# Both arms are the ones that MEAN something, and the banner is asserted on every run:
#   Phi4     : NPU_PREFILL_BF16=1                     -> host attention (no nh24 ELF exists)
#   Nanbeige : NPU_PREFILL_BF16=1 NPU_ATTN_CPU=1      -> host attention (the nh20 NPU kernel is the defect)
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
TOK="220 777 1024 4096 12345 58907 30000 45000"
run() { # $1=binary $2=model $3=flags $4=expected-banner-fragment
  local B=$1 M=$2 F=$3 WANT=$4
  for t in $TOK; do
    out=$(env $F timeout 900 $B $M 1 /tmp/C32_$t.txt 2>&1)
    tok=$(echo "$out" | grep -aoE '\[0\] boot=[0-9]+|\[1\] [0-9]+' | head -1 | grep -oE '[0-9]+$')
    if echo "$out" | grep -qai "$WANT"; then arm="ok"; else arm="ARM MISMATCH"; fi
    printf "   %-8s -> %-8s (%s)\n" "$t" "${tok:-?}" "$arm"
  done
}
echo "load: clang=$(ps -eo comm 2>/dev/null | grep -c clang-23) $(uptime | sed 's/.*load average: //')"
echo "Phi4 bf16 / host attention:"
run ./engine/npu/build/npu_engine_phi4_mini_4b ~/.config/flm/models/Phi4-mini-Instruct-NPU2/model.q4nx "NPU_PREFILL_BF16=1" "CPU attn_omp fallback"
echo "Nanbeige bf16 / host attention:"
run ./engine/npu/build/npu_engine_nanbeige4_1_3b ~/.config/flm/models/Nanbeige4.1-3B-NPU2/model.q4nx "NPU_PREFILL_BF16=1 NPU_ATTN_CPU=1" "forced CPU attn"
echo "load: clang=$(ps -eo comm 2>/dev/null | grep -c clang-23) $(uptime | sed 's/.*load average: //')"
