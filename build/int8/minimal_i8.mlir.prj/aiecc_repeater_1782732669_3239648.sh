#!/bin/bash
set -e
# Repeater script for: NPU lowering
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
'aie.dma_bd' op Transfer size of 0 bytes falls below minimum hardware transfer unit of 4 bytes.
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='minimal_i8.mlir.prj/aiecc_failure_1782732669_3239648.mlir'
PASS_PIPELINE='any(aie-materialize-runtime-sequences,aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
