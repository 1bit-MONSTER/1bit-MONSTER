#!/bin/bash
# cosched2.sh — same spatial-partitioning measurement, but with enough iterations that device work
# dominates the fixed pytest/build overhead. For each device: one process alone, then two at once,
# reporting each process's own wall time (so we can see they really overlapped).
set -u
cd "$HOME/amd-oss/iron" || exit 1
PY=~/amd-oss/iron-venv/bin/python
ITERS=${ITERS:-20}

one() {  # $1 dev  $2 key  $3 log
    local t0; t0=$(date +%s.%N)
    COLDEV="$1" PYTHONPATH=/tmp timeout 2400 "$PY" -m pytest iron/operators/flm/gemm/test.py \
        -p colplug -k "$2" --iterations "$ITERS" -q > "$3" 2>&1
    local rc=$?
    python3 -c "print('%.1f' % ($(date +%s.%N)-$t0))"; return $rc
}

for spec in "npu2_4col:M_256-K_512-N_512" "npu2:M_256-K_512-N_1024"; do
    dev=${spec%%:*}; key=${spec##*:}
    echo "== $dev  '$key'  iterations=$ITERS =="
    one "$dev" "$key" /tmp/cs2_warm.log >/dev/null       # warm the build cache
    s=$(one "$dev" "$key" /tmp/cs2_single.log)
    echo "   single: ${s}s  ($(grep -oE '[0-9]+ passed' /tmp/cs2_single.log | head -1))"
    ( one "$dev" "$key" /tmp/cs2_a.log > /tmp/cs2_a.time ) &
    pa=$!
    ( one "$dev" "$key" /tmp/cs2_b.log > /tmp/cs2_b.time ) &
    pb=$!
    t0=$(date +%s.%N); wait $pa $pb; pair=$(python3 -c "print('%.1f' % ($(date +%s.%N)-$t0))")
    a=$(cat /tmp/cs2_a.time); b=$(cat /tmp/cs2_b.time)
    echo "   pair:   ${pair}s (process A ${a}s, B ${b}s)  ($(grep -oE '[0-9]+ passed' /tmp/cs2_a.log | head -1) / $(grep -oE '[0-9]+ passed' /tmp/cs2_b.log | head -1))"
    python3 -c "
s=float('$s'); p=float('$pair'); a=float('$a'); b=float('$b')
print('   ratio pair/single = %.2f' % (p/s))
print('   implied aggregate speedup = %.2fx  (2.0 = spatial, 1.0 = serialised)' % (2*s/p))
print('   per-process inflation = %.2fx / %.2fx' % (a/s, b/s))
"
    echo
done
