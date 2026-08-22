#!/bin/bash
# Build the fused GU→SiLU→D xclbin (issue #1759) — ONE launch per Zaya MoE
# layer instead of two (GU then D). On-core fixed-point SiLU (256-entry LUT,
# see silu_quant.h) eliminates the GU→CPU→D round trip and halves the 40
# decode launches/token; the plan's ~6.2 → ~7.5 tok/s milestone.
#
# Shape: A = residual [1×2048], GU [2048×4096 interleaved], D [2048×2048].
# Single core row (r=1), M=8 1x4 vectorized mmul (bit-identical to M=16/128).
#
# Stale-design hardening (issue #1777): the design is written to a PID-unique
# /tmp path (no fixed path a co-tenant process can clobber between generation
# and aiecc), byte-compared against a fresh regeneration (the generator is
# deterministic), and content-checked for the fused-kernel markers + the
# exact h2-writeback [8K, K, 8, 1] and B-tile (ki·32 + n_tile)·8192 DMA
# signatures BEFORE aiecc consumes it. Any mismatch fails the build loudly
# instead of silently emitting a corrupt xclbin (the 08-22 incident: stale
# file → wrong B offsets / h2-writeback strides, corr 0.374).
#
# REQUIRES the mlir-aie toolchain (aiecc) + an NPU2 device for verification —
# this machine only has the CPU-side contract validation
# (engine/npu/tests/test_fused_silu.cpp).
#
# Usage: engine/npu/generators/build_zaya_fused.sh
set -euo pipefail

P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
AIETOOLS=/home/bcloud/mlir-aie/build_tmp
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
mkdir -p "$XCLBIN_DIR"

# 2. PID-unique workdir (issue #1777). The old build used fixed /tmp paths
#    (/tmp/design_fused_gu_silu_d.mlir AND /tmp/mm_32x64x128.o) that a
#    co-tenant process or leftover build could overwrite between generation
#    and aiecc, silently producing a corrupt xclbin (wrong B offsets /
#    h2-writeback strides, corr 0.374). $$ = this shell's PID, so no two
#    builds share a workdir; the trap removes it on exit so no stale /tmp
#    artifacts accumulate (design, <design>.prj/, kernel .o included).
workdir="/tmp/zaya_fused_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# 1. Compile the DIM_M=8 kernel (1x4 mmul + the fused silu_quant_i8_fused
#    entry from silu_quant.h — the on-core SiLU+quant step) INTO the workdir.
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_8x64x128_fused.o"

# The MLIR references the kernel object by the fixed name mm_32x64x128.o.
cp "$workdir/mm_8x64x128_fused.o" "$workdir/mm_32x64x128.o"

echo "═══ fused GU→SiLU→D  M=8 K=2048 N_GU=4096 N_D=2048 ═══"

design="$workdir/design_fused_gu_silu_d.mlir"
design_ref="$workdir/design_fused_gu_silu_d.ref.mlir"

gen_design() {  # $1 = output path
    $PYTHON "$GENERATOR_DIR/n1_core_fused_gu_silu_d.py" -M 8 -K 2048 \
        -N_GU 4096 -N_D 2048 -m 8 -k 64 -n 128 -c 8 -b 2 2>/dev/null > "$1"
}

gen_design "$design"
[ -s "$design" ] || { echo "ERROR: design generation produced an empty file" >&2; exit 1; }

# 3. Verify BEFORE aiecc (issue #1777 fix #1). The generator is deterministic
#    — a fresh run always yields the correct design — so:
#    (a) the design must be byte-identical to a fresh regeneration (catches a
#        stale/tampered file, the exact 08-22 failure mode);
#    (b) the design must carry the fused-kernel markers and the exact DMA
#        signatures the stale file got wrong (B offsets (ki·32+n_tile)·8192,
#        h2-writeback strides [8K, K, 8, 1] = [16384, 2048, 8, 1]).
gen_design "$design_ref"
if ! cmp -s "$design" "$design_ref"; then
    echo "ERROR: design differs from a fresh regeneration — stale/tampered design or" >&2
    echo "       nondeterministic generator. Refusing to build (issue #1777)." >&2
    diff -u "$design_ref" "$design" | head -40 >&2 || true
    exit 1
fi

"$PYTHON" - "$design" <<'PYEOF' || { echo "ERROR: design verification FAILED — refusing to build (issue #1777)" >&2; exit 1; }
import re, sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
K = 2048  # build_zaya_fused.sh fixed shape
errors = []

# Fused-design markers: a stale GU/D-only or moe design lacks these.
for marker in ("silu_quant_i8_fused", "mm_32x64x128.o"):
    if marker not in text:
        errors.append(f"missing marker {marker!r} — not the fused design?")

ops = []  # (fifo, offset, sizes, strides) — one entry per aie.dma_bd

# Format A — this repo's mlir-aie version (verified on strixhalo):
#   %N = aiex.dma_configure_task_for @FIFO {
#     aie.dma_bd(%mem : memref<16384xi8>, 0, 512, [<size = 1, stride = 16384>, ...]) {burst_length = 0 : i32}
#     ...
#   }
# A task may carry several aie.dma_bd descriptors; capture the whole
# brace-balanced task body and inspect EVERY descriptor (a wrong offset or
# stride hidden in a later BD must not slip through).
for m in re.finditer(r"aiex\.dma_configure_task_for @(\w+)\s*\{", text):
    fifo = m.group(1)
    start = m.end()
    depth, i = 1, start
    while depth and i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    task_body = text[start:i - 1] if depth == 0 else text[start:]
    for bdm in re.finditer(r"aie\.dma_bd\(([^)]*)\)", task_body):
        body = bdm.group(1)
        mm = re.match(r"%\w+\s*:\s*memref<[^>]*>,\s*(\d+),\s*(\d+),\s*\[(.*)\]", body)
        if not mm:
            continue
        offset = int(mm.group(1))
        sizes, strides = [], []
        for dm in re.finditer(r"<size\s*=\s*(\d+),\s*stride\s*=\s*(\d+)>", mm.group(3)):
            sizes.append(int(dm.group(1)))
            strides.append(int(dm.group(2)))
        ops.append((fifo, offset, sizes, strides))

# Format B — newer mlir-aie main fallback:
#   aiex.npu.dma_memcpy_nd (%mem[offsets][sizes][strides]) {metadata = @fifo} : memref<...>
if not ops:
    for m in re.finditer(r"aiex\.npu\.dma_memcpy_nd\s*\(([^)]*)\)\s*\{([^}]*)\}", text):
        body, attrs = m.group(1), m.group(2)
        fm = re.search(r"metadata\s*=\s*@(\w+)", attrs)
        fifo = fm.group(1) if fm else ""
        groups = re.findall(r"\[([^\]]*)\]", body)
        if len(groups) < 3:
            continue

        def nums(group):
            out = []
            for tok in group.split(","):
                tok = tok.strip()
                cm = re.match(r"%c(\d+)", tok)   # SSA constant ref, e.g. %c16384_i64
                if cm:
                    out.append(int(cm.group(1)))
                elif re.match(r"\d+$", tok):    # inline literal
                    out.append(int(tok))
                else:
                    out.append(None)
            return out

        offs, sizes, strides = (nums(g) for g in groups[-3:])
        if None in offs or None in sizes or None in strides:
            continue
        # B-tile offset may be spread over several offset dims; verify EVERY
        # dim is an 8192-multiple so a 512-step offset in any dim is caught.
        ops.append((fifo, offs, sizes, strides))

if not ops:
    errors.append("no DMA bd ops found — design empty or printer format changed")

h2_ok = a_ok = b_ok = False
n_b_off_bad = 0
seen = set()
for fifo, offset, sizes, strides in ops:
    if fifo.startswith("H2_S"):  # h2 writeback — the bug signature
        if strides == [8 * K, K, 8, 1]:
            h2_ok = True
        else:
            msg = f"H2_S tap strides {strides}, expected [8K, K, 8, 1] = {[8*K, K, 8, 1]}"
            if msg not in seen:
                errors.append(msg); seen.add(msg)
    elif fifo.startswith("A_C"):  # A broadcast tap
        if strides == [8 * K, 8, K, 1]:
            a_ok = True
        else:
            msg = f"A_C tap strides {strides}, expected [8K, 8, K, 1] = {[8*K, 8, K, 1]}"
            if msg not in seen:
                errors.append(msg); seen.add(msg)
    elif fifo.startswith("B_S"):  # linear B tiles: (ki*32 + n_tile)*8192
        if sizes == [1, 1, 1, 8192] and strides == [1, 1, 1, 1]:
            b_ok = True
            # Format A passes offset as a scalar (int); Format B as a list.
            # Normalize to a list and verify EVERY offset dim is an 8192-multiple.
            offs = offset if isinstance(offset, list) else [offset]
            bad = [o for o in offs if o % 8192 != 0]
            if bad:
                n_b_off_bad += len(bad)
                msg = f"B_S tap offset(s) {bad} not 8192-multiples (expected (ki*32+n_tile)*8192)"
                if msg not in seen:
                    errors.append(msg); seen.add(msg)
        else:
            msg = f"B_S tap sizes {sizes} strides {strides}, expected linear 8192-byte tile"
            if msg not in seen:
                errors.append(msg); seen.add(msg)

if not h2_ok:
    errors.append("no H2_S writeback with strides [8K, K, 8, 1] — the h2-writeback bug signature")
if not a_ok:
    errors.append("no A tap with strides [8K, 8, K, 1]")
if not b_ok:
    errors.append("no linear B tile (sizes [1,1,1,8192], strides [1,1,1,1])")
if n_b_off_bad:
    errors.append(f"{n_b_off_bad} B-tile offset(s) not 8192-multiples")

if errors:
    print("fused design verification FAILED:", file=sys.stderr)
    for e in errors:
        print("  - " + e, file=sys.stderr)
    sys.exit(1)
print(f"fused design verification OK: {len(ops)} DMA bd ops, h2 [8K,K,8,1] + A [8K,8,K,1] + B 8192-step tiles")
PYEOF

xclbin="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya.xclbin"
insts="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya.txt"
cd "$workdir"
$AIECC --peano="$P" --aietools="$AIETOOLS" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$xclbin" --npu-insts-name="$insts" \
    "$design" 2>&1 | tail -3
cd "$GENERATOR_DIR"
ls -la "$xclbin" "$insts"

cat <<'EOF'
═══ verification checklist (strixhalo) ═══
1. aiecc build: watch for (a) the produce-only C1 fifo lowering, (b) shim
   S2MM channel pressure (2 outbound streams per column), (c) the gs 512B tap.
2. NPU decode with NPU_FUSED=1 (zaya_decode.cpp fused mode):
   - corr of the layer-1 MoE probe vs the CPU float reference ≥ ~0.999
     (contract validated on x86: 0.9993–0.9996, argmax parity — the fused
     int8 path must match the two-launch NPU path, not float exactly).
   - token parity vs the two-launch M=16 path.
3. perf: 40 → 20 launches/token; expect ~6.2 → ~7.5 tok/s if the ~0.85 ms
   per-launch fixed overhead is the binding cost.
Fallbacks if the C1 produce-only fifo misbehaves:
   - Design J: write C1 to bo4 via the v27 C path and read it back for the
     SiLU phase (adds ~16 KB/token DDR traffic; needs a 4th outbound stream
     budget check).
EOF
