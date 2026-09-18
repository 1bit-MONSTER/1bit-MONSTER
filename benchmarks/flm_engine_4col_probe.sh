#!/bin/bash
# engine4col.sh — does a 4-column kernel set give the ENGINE the same spatial 2x the IRON probe
# showed? Single instance vs two at once, for the 4-column set and the 8-column set (control).
# The engine serialises itself with an flock, so concurrent legs opt out with NPU_NO_DEVICE_LOCK=1.
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
D="$HOME/.config/flm/models/Qwen3-0.6B-NPU2"
E=./engine/npu/build/npu_engine_qwen3_0_6b
NG=${NG:-256}
P=${P:-/tmp/p_1k.txt}
rate() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }
tok()  { grep -acE "^  \[[0-9]+\] [0-9]+" "$1"; }
seq()  { grep -aoE "^  \[[0-9]+\] [0-9]+" "$1" | tr -d ' ' | tr '\n' ' '; }
run()  { timeout 900 env NPU_NO_DEVICE_LOCK=1 NPU_BF16=1 NPU_XCLBIN_DIR="$1" NPU_GREEDY=1 \
             NPU_PREFILL_MAX=2048 "$E" "$D/model.q4nx" "$NG" "$P" > "$2" 2>&1; }
pf()   { grep -aoE "Prefill: [0-9]+ms" "$1" | tail -1 | grep -oE "[0-9]+"; }

for set in "4col:$HOME/xclbins-4col-dev" "8col:$PWD/engine/npu/xclbins"; do
    lbl=${set%%:*}; dir=${set#*:}
    echo "== $lbl kernels ($dir)  prompt=$P ng=$NG =="
    run "$dir" /tmp/e_${lbl}_s.log
    s=$(rate /tmp/e_${lbl}_s.log); spf=$(pf /tmp/e_${lbl}_s.log)
    echo "   single: prefill=${spf}ms decode=${s} tok/s"
    run "$dir" /tmp/e_${lbl}_a.log & a=$!
    run "$dir" /tmp/e_${lbl}_b.log & b=$!
    wait $a $b
    ra=$(rate /tmp/e_${lbl}_a.log); rb=$(rate /tmp/e_${lbl}_b.log)
    agg=$(( ${ra:-0} + ${rb:-0} ))
    echo "   pair:   A=${ra:-?} B=${rb:-?}  aggregate=${agg} tok/s   (prefills $(pf /tmp/e_${lbl}_a.log)/$(pf /tmp/e_${lbl}_b.log) ms)"
    python3 -c "
s=float('${s:-0}'); a=float('${agg:-0}')
print('   aggregate vs single = %.2fx   (2.0 = spatial, ~1.5 = sharing one partition)' % (a/s if s else 0))
print('   per-instance share  = %.0f%% / %.0f%% of solo' % (100*${ra:-0}/s if s else 0, 100*${rb:-0}/s if s else 0))
"
    # correctness: both instances must equal the single run's stream
    for f in a b; do
        [ "$(seq /tmp/e_${lbl}_s.log)" = "$(seq /tmp/e_${lbl}_$f.log)" ] && v=IDENTICAL || v=DIFFER
        echo "   instance $f tokens vs single: $v ($(tok /tmp/e_${lbl}_$f.log) tokens)"
    done
    echo
done
