#!/bin/bash
# post-reboot-verify.sh — verify the NPU SVA path is functional after the reboot
# and run the REAL FastFlowLM Qwen3-0.6B layer kernel (module flow).
#
# Background (see docs/research/amd-iommu-perfopt-strixhalo.md):
#  - The amdxdna NPU (c6:00.1, IOMMU group 26) is identity-mode by upstream
#    default for PASID-capable devices; the AMD identity domain is SVA-capable
#    (GCR3) and the FastFlowLM/XRT flow DMAs to host VAs via the context PASID.
#  - On 2026-08-31 the cascade calibration ran EXACT on this config at 09:12,
#    then the SVA path broke mid-session (~10:29+): every DPU launch faulted
#    with AMD-Vi IO_PAGE_FAULT at host VAs (GCR3 no longer translating) and
#    timed out (state=8, DPU PC=0xffffffff). Root cause undetermined but
#    boot-time config was proven good, so a reboot should restore it.
#  - This script: (1) checks group type, (2) runs the cascade probe (the
#    silicon-pinned control), (3) if good, launches the real layer kernel.

set -u
LOG=/tmp/npu-post-reboot-$(date +%Y%m%d-%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "===== NPU post-reboot verify $(date) ====="
echo "kernel: $(uname -r)"

echo "--- IOMMU groups ---"
for g in 20 26; do
  echo "group $g ($(ls /sys/kernel/iommu_groups/$g/devices | tr '\n' ' ')): $(cat /sys/kernel/iommu_groups/$g/type)"
done

echo "--- amdxdna driver ---"
ls -la /sys/module/amdxdna/ 2>/dev/null | head -2
dmesg 2>/dev/null | grep -E "Load firmware|PerfOpt armed" | tail -3

XC=/home/bcloud/1bit-MONSTER/engine/npu/xclbins/final_cascade_fused.xclbin
INST=/home/bcloud/1bit-MONSTER/engine/npu/xclbins/q4nx/insts_bf16_8col_D_qwen3_0_6b.bin
W=/home/bcloud/1bit-MONSTER/models/Qwen3-0.6B-q8-q4nx.1bp

echo "--- cascade probe (silicon control) ---"
timeout 120 /usr/local/bin/cascade_rw_probe "$W" "$XC" "$INST" 2>&1 | tail -6
echo "--- cascade probe dmesg faults (expect none): ---"
dmesg 2>/dev/null | grep -c "IO_PAGE_FAULT" || true

echo "--- REAL FastFlowLM layer kernel (module flow) ---"
LAYER_XC=/tmp/fflm/src/xclbins/Qwen3-0.6B-NPU2/layer.xclbin
SEQ=/home/bcloud/1bit-MONSTER/engine/npu/fflm-run/fflm-layer0.seq
if [ -f "$LAYER_XC" ] && [ -f "$SEQ" ]; then
  timeout 120 /usr/local/bin/fflm_run4 "$LAYER_XC" "$SEQ" 2>&1 | tail -6
  echo "--- layer dmesg faults (expect none): ---"
  dmesg 2>/dev/null | grep -c "IO_PAGE_FAULT" || true
else
  echo "layer.xclbin or fflm-layer0.seq missing (need /tmp/fflm clone + seq)"
fi
echo "===== done: $LOG ====="
