#!/usr/bin/env bash
# Post-reboot NPU validation — P0.1 fix check (amd_iommu=off restored 07-31).
# Run as: ./validate_npu_after_reboot.sh
# Expect (if the IOMMU fix worked):
#   - IO_PAGE_FAULT count: 0 (was ~200-800 per run)
#   - Prefill: ~2 s for 9 tokens (was 90 s)
#   - Decode: 5-40 tok/s class (was 0.1)
set -euo pipefail

echo "=== 1. boot config check ==="
grep -o "amd_iommu=[a-z]*" /proc/cmdline || echo "FAIL: amd_iommu=off NOT on cmdline"
lsmod | grep amdxdna || echo "WARN: amdxdna not loaded"

echo "=== 2. baseline fault count ==="
BEFORE=$(sudo dmesg | grep -c IO_PAGE_FAULT || true)
echo "faults in ring before run: $BEFORE"

echo "=== 3. engine run (32 tokens) ==="
cd ~/1bit-systems
stdbuf -o0 timeout 300 env OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/npu_engine_overlap_fd models/qwen3_0_6b.q4nx 32 2>&1 | tee /tmp/npu_validation_run.log | grep -E "Prefill|tok=" | head -12

echo "=== 4. fault delta ==="
AFTER=$(sudo dmesg | grep -c IO_PAGE_FAULT || true)
echo "faults in ring after run: $AFTER (delta: $((AFTER - BEFORE)))"
if [ $((AFTER - BEFORE)) -eq 0 ]; then echo "PASS: zero new faults"; else echo "CHECK: $((AFTER - BEFORE)) new faults"; fi

echo "=== 5. numbers ==="
grep "Prefill" /tmp/npu_validation_run.log | tail -1
echo "done — record results in research/ws01-npu-attention/FINDINGS.md"
