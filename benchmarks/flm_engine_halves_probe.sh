#!/bin/bash
# engine_halves.sh — run ONE engine instance on the 4-column h0 kernels (cols 0-3) and TWO at once
# with the second on h1 (cols 4-7), built with the upstream col-offset knob. This is the
# column-sliced co-schedule (issue #2128) applied to the dense-Qwen3 bf16 path.
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
D="$HOME/.config/flm/models/Qwen3-0.6B-NPU2"
E=./engine/npu/build/npu_engine_qwen3_0_6b
NG=${NG:-256}; P=${P:-/tmp/p_1k.txt}
rate() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }
pf()   { grep -aoE "Prefill: [0-9]+ms" "$1" | tail -1 | grep -oE "[0-9]+"; }
seqs() { grep -aoE "^  \[[0-9]+\] [0-9]+" "$1" | tr -d ' ' | tr '\n' ' '; }
run()  { timeout 900 env NPU_NO_DEVICE_LOCK=1 NPU_BF16=1 NPU_XCLBIN_DIR="$1" NPU_GREEDY=1 \
             NPU_PREFILL_MAX=2048 "$E" "$D/model.q4nx" "$NG" "$P" > "$2" 2>&1; }

echo "== halves: h0=cols 0-3 ($HOME/xclbins-h0), h1=cols 4-7 ($HOME/xclbins-h1) =="
run "$HOME/xclbins-h0" /tmp/h_s.log
s=$(rate /tmp/h_s.log)
echo "   single (h0): prefill=$(pf /tmp/h_s.log)ms decode=${s} tok/s"

run "$HOME/xclbins-h0" /tmp/h_a.log & a=$!
run "$HOME/xclbins-h1" /tmp/h_b.log & b=$!
wait $a $b
ra=$(rate /tmp/h_a.log); rb=$(rate /tmp/h_b.log); agg=$(( ${ra:-0} + ${rb:-0} ))
echo "   pair (h0+h1): A=${ra:-?} B=${rb:-?}  aggregate=${agg} tok/s  (prefills $(pf /tmp/h_a.log)/$(pf /tmp/h_b.log) ms)"
python3 -c "
s=float('${s:-0}'); agg=float('${agg:-0}')
print('   aggregate vs single = %.2fx   (2.0 = two disjoint halves, ~1.0 = sharing one array)' % (agg/s if s else 0))
print('   per-instance = %.0f%% / %.0f%% of solo' % (100*${ra:-0}/s if s else 0, 100*${rb:-0}/s if s else 0))
"
for f in a b; do
    [ "$(seqs /tmp/h_s.log)" = "$(seqs /tmp/h_$f.log)" ] && v=IDENTICAL || v=DIFFER
    echo "   instance $f tokens vs single: $v"
done
