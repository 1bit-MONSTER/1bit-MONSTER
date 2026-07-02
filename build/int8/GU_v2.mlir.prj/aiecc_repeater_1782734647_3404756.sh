#!/bin/bash
set -e
# Repeater script for: resource allocation
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
'aie.dma_bd' op Specified stride(s) and size(s) result in out of bounds access in buffer, for index 65535 in memref of length 8192.
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='GU_v2.mlir.prj/aiecc_failure_1782734647_3404756.mlir'
PASS_PIPELINE='any(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},unknown<RedundantLoadStoreOptimizationPass>,unknown<ReorderOperationsPass>,unknown<{anonymous}::CopyRemovalPass>,unknown<VectorBroadcastLoweringPass>,test-canonicalize-vector-for-aievec{aie-target=aie2p target-backend=llvmir},test-lower-vector-to-aievec{aie-target=aie2p target-backend=llvmir},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},unknown<ExtendUPDOpsPass>,cse,unknown<SimplifyUPDOpsPass>,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},test-aievec-optimize{aie-target=aie2p target-backend=llvmir},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},aievec-convolution-analysis{print=false},test-aievec-convolution-optimize{aie-target=aie2p shift=0 target-backend=llvmir},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},loop-invariant-code-motion,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},lower-affine,aie-canonicalize-device,aie.device(aie-assign-lock-ids,aie-register-objectFifos,aie-objectFifo-stateful-transform{dynamic-objFifos=true packet-sw-objFifos=false},aie-assign-bd-ids,aie-lower-cascade-flows,aie-lower-broadcast-packet,aie-lower-multicast,aie-assign-tile-controller-ids{column-wise-unique-ids=true},aie-generate-column-control-overlay{route-shim-to-tct=shim-only route-shim-to-tile-ctrl=false},aie-assign-buffer-addresses{alloc-scheme=basic-sequential},aie-assign-core-link-files,aie-vector-transfer-lowering{max-transfer-rank=4294967295}),convert-scf-to-cf{allow-pattern-rollback=true})'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
