#!/usr/bin/env bash
# WS-00 benchmark runner — one command, every backend, JSON + summary.
#
# Runs all available bench binaries in build/, captures exit status + duration,
# writes benchmarks/run-<timestamp>.json and a summary table with honesty tags.
#
# Tag policy (project-wide):
#   raw       = number captured from binary output, NOT yet verified
#   validated = verified on hardware against a reference
#   broken    = binary fails or hangs
#
# Usage:
#   ./run_benchmarks.sh [--timeout N] [--only bench_kv_fd]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$ROOT/build"
OUTDIR="$ROOT/benchmarks"
TIMEOUT="${BENCH_TIMEOUT:-120}"
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"
TS="$(date +%Y%m%d-%H%M%S)"
JSON="$OUTDIR/run-$TS.json"
MD="$OUTDIR/run-$TS.md"

BENCHES=(bench_1bp_cpu bench_fused bench_fused_tq2_1024 bench_ternary_new bench_kv_fd \
         bench_kv_i8 bench_prefill_variants bench_hip_1bp bench_ggml_vk bench_mamba2_kernels \
         bench_rotor bench_sherry bench_bonsai_q1_1024 bench_bonsai_tq2_1024 bench_twla_28layer)

echo "# Benchmark run $TS" > "$MD"
echo "" >> "$MD"
echo "| bench | status | tag | time_s | note |" >> "$MD"
echo "|---|---|---|---|---|" >> "$MD"

RESULTS="[]"
for b in "${BENCHES[@]}"; do
  bin="$BUILD/$b"
  if [ -n "$ONLY" ] && [ "$b" != "$ONLY" ]; then continue; fi
  if [ ! -x "$bin" ]; then continue; fi
  echo "== $b"
  t0=$(date +%s%N)
  out="$(timeout "$TIMEOUT" "$bin" 2>&1 || true)"
  rc=$?
  t1=$(date +%s%N)
  dur=$(( (t1 - t0) / 1000000 ))
  if [ $rc -eq 0 ]; then
    tag="raw"
    status="ok"
    note="$(echo "$out" | grep -iE 'tok/s|tokens|GB/s|TFLOPS|ms/tok' | tail -2 | tr '\n' ' ' | cut -c1-120)"
  elif [ $rc -eq 124 ]; then
    tag="broken"; status="timeout"; note="exceeded ${TIMEOUT}s"
  else
    tag="broken"; status="rc=$rc"; note="$(echo "$out" | tail -2 | tr '\n' ' ' | cut -c1-120)"
  fi
  echo "| $b | $status | $tag | ${dur}ms | $note |" >> "$MD"
  RESULTS=$(python3 - "$RESULTS" "$b" "$status" "$tag" "$dur" "$note" <<'PYEOF'
import json, sys
results = json.loads(sys.argv[1] if sys.argv[1] else '[]')
results.append({"bench": sys.argv[2], "status": sys.argv[3], "tag": sys.argv[4],
                "time_ms": int(sys.argv[5]), "note": sys.argv[6]})
print(json.dumps(results))
PYEOF
  )
done

echo "$RESULTS" | python3 -c "
import json, sys
d = json.load(sys.stdin)
json.dump({'timestamp': '$TS', 'hardware': 'Strix Halo (gfx1151 + XDNA 2)', 'runs': d},
          open('$JSON', 'w'), indent=1)
print(f'{len(d)} benches -> $JSON')
"
cat "$MD"
echo ""
echo "summary: $MD"
