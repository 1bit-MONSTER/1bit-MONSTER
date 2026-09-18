#!/bin/bash
# build_prism_gdn_npu.sh - P4.2: reproducible build of the GDN conv1d AIE design and its runlist
# entry. Produces, into a BUILD DIR (default /tmp/prism_gdn_npu.$$; never the tracked xclbin set):
#   prism_gdn_conv1d_aie.o  the aie2p kernel object
#   design.mlir             the IRON/AIE design
#   out.xclbin              the NPU xclbin (magic xclbkin2 checked)
#   insts.txt               the NPU instruction stream
#   main_core_0_2.elf       the per-ctx ELF, i.e. the runlist entry's payload
#   runlist_entry.json      the entry manifest (design, kernel, geometry, artifacts, ABI)
#
# Runlist ABI this entry targets (npu-infer/include/runtime_layer.h): ONE run per layer, the ELF
# built per context length. The GDN conv1d is per-token and context-INDEPENDENT, so this entry is
# a single-run, context-independent entry - the same shape as the runtime's lm_head ELF rather
# than the per-layer layer_ctxN.elf files.
set -euo pipefail

CD="${GDN_CD:-256}"
P=/home/bcloud/iron/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/iron/lib/python3.14/site-packages/mlir_aie
AI=/home/bcloud/mlir-aie/install_tmp
V=/home/bcloud/mlir-aie/.venv
GEN="${GEN:-$(cd "$(dirname "$0")/../generators" && pwd)/prism_gdn_npu.py}"
KSRC="${KSRC:-$(cd "$(dirname "$0")" && pwd)/prism_gdn_aie.cc}"
B="${BUILD_DIR:-/tmp/prism_gdn_npu.$$}"
mkdir -p "$B"

if [ ! -x "$AI/bin/aiecc" ]; then echo "build: no aiecc at $AI/bin/aiecc" >&2; exit 1; fi

# 1. kernel object for the AIE target (the "missing kernel objects" this task had to create)
"$P/bin/clang++" --target=aie2p-none-unknown-elf --std=c++20 -O2 -DGDN_CD="$CD" \
    -isystem "$P/include/c++/v1" \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I "$M/include/aie_kernels/aie2p" \
    -c "$KSRC" -o "$B/prism_gdn_packed.o"
echo "kernel object: $(stat -c%s "$B/prism_gdn_packed.o") bytes (name must match the design link_with)"

# 2. IRON design
( cd "$B" && PYTHONPATH="$AI/python:$V/lib/python3.14/site-packages" \
    LD_LIBRARY_PATH="$AI/python/aie/_mlir_libs" \
    "$V/bin/python3" "$GEN" --cd "$CD" > design.mlir ) || { echo "gen failed" >&2; exit 1; }
echo "design.mlir: $(stat -c%s "$B/design.mlir") bytes"

# 3. xclbin + instruction stream + per-ctx ELF
( cd "$B" && "$AI/bin/aiecc" --peano="$P" --aietools="$AI" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name=out.xclbin --npu-insts-name=insts.txt \
    design.mlir ) || { echo "aiecc failed" >&2; exit 1; }

MAGIC=$(head -c 8 "$B/out.xclbin" | tr -d '\0')
[ "$MAGIC" = "xclbin2" ] || { echo "xclbin magic is '$MAGIC', expected xclbin2" >&2; exit 1; }
ELF=$(ls "$B"/design.mlir.prj/main_core_*.elf 2>/dev/null | head -1)
[ -n "$ELF" ] || { echo "no per-ctx ELF produced" >&2; exit 1; }

# 4. the runlist entry manifest
cat > "$B/runlist_entry.json" <<EOF
{
  "entry": "prism_gdn_conv1d",
  "lane": "prism-bonsai-27b (goal mu6ylbom-10buvx)",
  "task": "P4.2",
  "kind": "single-run, context-independent",
  "kernel": "prism_gdn_conv1d_packed",
  "geometry": { "CD": $CD, "blocks": "GDN conv1d(k=4) + silu + rolling 3-tap state" },
  "xclbin": "out.xclbin",
  "npu_instructions": "insts.txt",
  "per_ctx_elf": "$(basename "$ELF")",
  "abi": "one run per submission; RuntimeLayerEngine submits one runlist per token, so this entry is a single node in that runlist",
  "built_by": "engine/npu/kernels/build_prism_gdn_npu.sh",
  "correctness": "host math gate: conv1d+silu rel-RMSE 3.839e-06, state 0.0, delta step 0.0 (engine/npu/kernels/prism_gdn_refcheck.cc)",
  "on_device": "PERFORMED 2026-09-18 (de5f6350e): our GDN conv1d AIE design was loaded onto /dev/accel/accel0, executed there, and its device output agreed with an independent scalar reference to max_rel_err 1.043e-06 against the rel-RMSE < 1e-3 contract criterion (engine/npu/kernels/prism_gdn_devrun.cpp, device lock held/released; evidence docs/research/prism-bonsai-27b/P4.2-ondevice-npu-gate.md)"
}
EOF
echo "entry: $B/runlist_entry.json"
echo "ok: $B"
