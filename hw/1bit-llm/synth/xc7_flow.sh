#!/usr/bin/env bash
# xc7_flow.sh — open-source 7-series bitstream flow for the 1bit-LLM design.
#
# Chain:  yosys synth_xilinx (JSON) → nextpnr-xilinx (P&R, .fasm)
#         → prjxray fasm2frames → xc7frames2bit → .bit → (optional) bitread
#         round-trip check → openFPGALoader.
#
# Target board (identified, not TODO): Arty A7-35T — xc7a35tcsg324.
# IDCODE 0x0362d093 read off real hardware in docs/journey.md UPDATE 32
# (2026-08-09), where this exact toolchain produced and verified a blinky.bit.
#
# Toolchain lives on the Strix box per docs/journey.md UPDATE 32:
#   /home/bcloud/fpga-toolchain/  (oss-cad-suite yosys, nextpnr-xilinx,
#                                  prjxray + database, openFPGALoader)
# The RTL itself is tool-agnostic Verilog-2001; nothing here is required for
# `make sim`. This script is the honest end-to-end path on the box with the
# toolchain + a wired board.
#
# VERIFIED (2026-08-27, Strix box): steps 1-2 ran end-to-end for
# PART=xc7a35tcsg324 — yosys synth_xilinx (-nodsp, see step 1) produced the
# JSON and nextpnr-xilinx placed+routed all 102 hif pads (fmax 79.4 MHz,
# t1llm_top.fasm + routed.json). Step 3 (fasm2frames) is blocked by a
# toolchain db skew: the prebuilt chipdb (Aug 9) emits LIOI3/RIOI3 IO-mux
# features (e.g. IOI_IMUX_RC1.IOI_BYP4_0) that the on-disk prjxray-db
# (shallow clone 0a0adde) does not carry in segbits_lioi3.db, so
# FasmLookupError aborts. blinky (journey UPDATE 32) never exercised those
# mux features, which is why the Aug 9 round-trip passed. Fix: rebuild the
# chipdb from the on-disk db via the nextpnr-xilinx bba/bbasm flow (the
# xc7arch-style arch generator at 8f178fc), or restore the db snapshot the
# Aug 9 chipdb was built from. Steps 4-5 additionally need a wired board
# (see DESIGN.md §6).

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SIM="$HERE/sim"
PART="${PART:-xc7a35tcsg324}"                 # Arty A7-35T (see header)
FPGA_TOOLCHAIN="${FPGA_TOOLCHAIN:-/home/bcloud/fpga-toolchain}"
OSS_CAD_SUITE="$FPGA_TOOLCHAIN/oss-cad-suite/oss-cad-suite"
YOSYS="${YOSYS:-$OSS_CAD_SUITE/bin/yosys}"
NEXTPNR_DIR="$FPGA_TOOLCHAIN/nextpnr-xilinx"
CHIPDB="${CHIPDB:-$NEXTPNR_DIR/xilinx/xc7a35t.bin}"   # compiled prjxray part db
# prjxray fasm2frames expects db-root to contain mapping/{devices,parts}.yaml,
# <fabric>/tilegrid.json (fabric = xc7a50t for a35t), and <part>/{package_pins,
# part}.csv/json — i.e. the per-family dir of the vendored prjxray-db.
PRJXRAY_DB="$NEXTPNR_DIR/xilinx/external/prjxray-db/artix7"
PRJXRAY_PART="${PRJXRAY_PART:-xc7a35tcsg324-3}"       # prjxray-db part dir
FASM2FRAMES="${FASM2FRAMES:-$FPGA_TOOLCHAIN/prjxray/utils/fasm2frames.py}"
FASM2FRAMES_PY="${FASM2FRAMES_PY:-$FPGA_TOOLCHAIN/venv/bin/python}"  # has prjxray pkg
XC7FRAMES2BIT="${XC7FRAMES2BIT:-$FPGA_TOOLCHAIN/prjxray/build/tools/xc7frames2bit}"
BITREAD="${BITREAD:-$FPGA_TOOLCHAIN/prjxray/build/tools/bitread}"
PART_YAML="${PART_YAML:-$PRJXRAY_DB/$PRJXRAY_PART/part.yaml}"
RTL_FILES=("$HERE"/rtl/*.v)

# nextpnr-xilinx links against boost from the oss-cad-suite libs.
export LD_LIBRARY_PATH="$OSS_CAD_SUITE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== 1bit-LLM xc7 flow (part=$PART) =="

# ---- 1. synthesize to a nextpnr JSON netlist (also gates synthesizability)
#   -nodsp: nextpnr-xilinx 8f178fc aborts with std::out_of_range while packing
#   the inferred DSP48E1s (the 32x16 scale multiplies). LUT-only mapping is
#   equivalent logic and keeps the multiplier-free story honest for this pass.
mkdir -p "$SIM"
"$YOSYS" -q -p "read_verilog ${RTL_FILES[*]}; \
    hierarchy -top t1llm_top; proc; opt; \
    synth_xilinx -top t1llm_top -arch xc7 -nodsp; \
    write_json $SIM/t1llm_top.json" \
    -l "$SIM/synth.log"
echo "   synth: OK ($SIM/t1llm_top.json)"

if [ ! -f "$CHIPDB" ]; then
    echo "   chipdb $CHIPDB missing — cannot P&R. Regenerate it via the"
    echo "   nextpnr-xilinx bba/bbasm flow from the on-disk prjxray-db,"
    echo "   or restore the prebuilt binary. See DESIGN.md §6."
    exit 0
fi

# ---- 2. place & route (needs a .xdc only for pin-locked I/O; without one,
#          nextpnr places IOBs itself and the design still routes)
if [ -f "$HERE/synth/board.xdc" ]; then
    XDC_ARG=(--xdc "$HERE/synth/board.xdc")
else
    echo "   synth/board.xdc missing — P&R with auto-placed I/O"
    XDC_ARG=()
fi
"$NEXTPNR_DIR/build/nextpnr-xilinx" \
    --chipdb "$CHIPDB" \
    "${XDC_ARG[@]}" \
    --json "$SIM/t1llm_top.json" \
    --write "$SIM/t1llm_top.routed.json" \
    --fasm "$SIM/t1llm_top.fasm"
echo "   P&R: OK ($SIM/t1llm_top.routed.json, $SIM/t1llm_top.fasm)"

# ---- 3. fasm → frames → bitstream (prjxray)
# Known blocker (2026-08-27): the on-disk prjxray-db does not match the
# prebuilt chipdb for LIOI3/RIOI3 IO-mux features — see the header note.
if ! "$FASM2FRAMES_PY" "$FASM2FRAMES" \
    --db-root "$PRJXRAY_DB" \
    --part "$PRJXRAY_PART" \
    --sparse "$SIM/t1llm_top.fasm" > "$SIM/t1llm_top.frames" 2> "$SIM/fasm2frames.err"; then
    echo "   fasm2frames: FAILED — see $SIM/fasm2frames.err"
    echo "   Likely db skew: chipdb $CHIPDB (built $(stat -c %y "$CHIPDB" 2>/dev/null || echo '?')"
    echo "   from an older prjxray-db snapshot) vs db $PRJXRAY_DB."
    echo "   Fix: rebuild the chipdb from the on-disk db (bba flow, DESIGN.md §6),"
    echo "   or restore the matching prjxray-db snapshot. P&R itself is verified."
    exit 1
fi
echo "   fasm2frames: OK ($(wc -l < "$SIM/t1llm_top.frames") frames)"

"$XC7FRAMES2BIT" \
    --part_file "$PART_YAML" \
    --frm_file "$SIM/t1llm_top.frames" \
    --output_file "$SIM/t1llm_top.bit"
echo "   bitstream: $SIM/t1llm_top.bit ($(stat -c %s "$SIM/t1llm_top.bit") bytes)"

# ---- 4. round-trip check: bitread must recover the frames byte-identically
#         (same check that proved the flow correct in journey UPDATE 32)
"$BITREAD" \
    --part_file "$PART_YAML" \
    --frm_file "$SIM/t1llm_top.frames" \
    "$SIM/t1llm_top.bit" > "$SIM/t1llm_top.roundtrip.log" 2>&1 \
    && echo "   bitread round-trip: OK (frames byte-identical)" \
    || { echo "   bitread round-trip: FAILED (see $SIM/t1llm_top.roundtrip.log)"; exit 1; }

# ---- 5. load (needs board + cable)
BOARD="${BOARD:-arty_a7}"
if command -v openFPGALoader >/dev/null 2>&1; then
    openFPGALoader -b "$BOARD" "$SIM/t1llm_top.bit" 2>/dev/null \
        || echo "   openFPGALoader skipped (no cable / board '$BOARD'?)"
else
    echo "   openFPGALoader not on PATH — try:"
    echo "     export PATH=\$PATH:$OSS_CAD_SUITE/bin"
    echo "   (or set BOARD in $0)"
fi
