#!/usr/bin/env bash
# xc7_flow.sh — open-source 7-series bitstream flow for the 1bit-LLM design.
#
# Chain: iverilog (sim, in Makefile) → yosys synth_xilinx → nextpnr-xilinx
#         → prjxray (fasm2frames → bitread check) → openFPGALoader.
#
# The toolchain lives on the Strix box (per docs/journey.md UPDATE 32):
#   /home/bcloud/fpga-toolchain/   (nextpnr-xilinx + prjxray + openFPGALoader)
# The RTL itself is tool-agnostic Verilog-2001; nothing here is required for
# `make sim`. This script is the honest end-to-end path once a board is wired.
#
# TODO(board): pick the concrete 7-series board + part, e.g.
#   - Arty A7-100T  (xc7a100tcsg324-1, 135k LUT / 4.86 Mb BRAM)
#   - Nexys Video  (xc7a200tfbg484-1, 215k LUT / 13 Mb BRAM)
#   A 1B ternary model needs external DRAM streaming (roadmap §8); this first
#   pass exercises the on-chip weight cache path only.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SIM="$HERE/sim"
PART="${PART:-xc7a100tcsg324-1}"          # TODO(board): set your part
FPGA_TOOLCHAIN="${FPGA_TOOLCHAIN:-/home/bcloud/fpga-toolchain}"
RTL_FILES=("$HERE"/rtl/*.v)

echo "== 1bit-LLM xc7 flow (part=$PART) =="

# ---- 1. synthesize (gate: proves the RTL is synthesizable, not just simulatable)
mkdir -p "$SIM"
yosys -q -p "read_verilog ${RTL_FILES[*]}; \
    hierarchy -top t1llm_top; proc; opt; \
    synth_xilinx -top t1llm_top -arch xc7; \
    write_verilog -noattr $SIM/t1llm_top_synth.v" \
    -l "$SIM/synth.log"
echo "   synth: OK"

if ! command -v nextpnr-xilinx >/dev/null 2>&1; then
    echo "   nextpnr-xilinx not on PATH — try:"
    echo "     export PATH=\$PATH:$FPGA_TOOLCHAIN/nextpnr-xilinx/nextpnr-xilinx/bin"
    echo "   P&R skipped (flow documented; run on the Strix box)."
    exit 0
fi

# ---- 2. place & route (needs a .xdc — TODO(board))
if [ -f "$HERE/synth/board.xdc" ]; then
    nextpnr-xilinx --chipdb "$FPGA_TOOLCHAIN"/prjxray-db/*-*.chipdb \
        --xdc "$HERE/synth/board.xdc" \
        --json "$SIM/t1llm_top.json" \
        --write "$SIM/t1llm_top.routed.json" \
        --fasm "$SIM/t1llm_top.fasm"
    echo "   P&R: OK"

    # ---- 3. fasm → bitstream
    "$FPGA_TOOLCHAIN"/prjxray/utils/fasm2frames.py \
        --db-root "$FPGA_TOOLCHAIN"/prjxray-db \
        --sparse "$SIM/t1llm_top.fasm" > "$SIM/t1llm_top.frames"
    "$FPGA_TOOLCHAIN"/prjxray/utils/frames2bit.py \
        --db-root "$FPGA_TOOLCHAIN"/prjxray-db \
        --part "$PART" --frm_file "$SIM/t1llm_top.frames" \
        > "$SIM/t1llm_top.bit"
    echo "   bitstream: $SIM/t1llm_top.bit"

    # ---- 4. load (TODO(board): cable present?)
    openFPGALoader -b "$BOARD" "$SIM/t1llm_top.bit" 2>/dev/null \
        || echo "   openFPGALoader skipped (no cable / board '$BOARD')"
else
    echo "   synth/board.xdc missing — P&R skipped (see TODO(board) at top)."
fi
