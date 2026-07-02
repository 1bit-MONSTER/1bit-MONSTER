#!/bin/bash
set -e
# Repeater script for: NPU lowering
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
'aiex.dma_await_task' op Cannot wait on a BD that is not configured to issue a token.
failed to legalize operation 'aiex.dma_await_task' that was explicitly marked illegal
'aiex.dma_configure_task' op Cannot lower while op still has uses.
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='KV_v14.mlir.prj/aiecc_failure_1782735695_1460673.mlir'
PASS_PIPELINE='any(aie-materialize-runtime-sequences,aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
