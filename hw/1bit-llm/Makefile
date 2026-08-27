# 1bit-LLM — build & verify (see DESIGN.md)
#
#   make gold   — generate golden vectors (python3 tools/gen_golden.py)
#   make sim    — iverilog simulation, must print "=== TB: ALL PASS ==="
#   make synth  — yosys synth_xilinx synthesizability gate (open-source 7-series)
#   make clean

IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3
YOSYS    ?= yosys

RTL     := $(wildcard rtl/*.v)
TB      := tb/tb_top.v
SIM_DIR := sim

.PHONY: all gold sim synth clean

all: sim

gold:
	$(PYTHON) tools/gen_golden.py 0
	$(PYTHON) tools/gen_golden.py 1

sim: gold
	@mkdir -p $(SIM_DIR)
	$(IVERILOG) -g2005-sv -Wall -o $(SIM_DIR)/tb_top.vvp $(RTL) $(TB)
	cd $(SIM_DIR) && $(VVP) tb_top.vvp

# Synthesizability gate: elaboration + generic synth (no P&R here; the full
# nextpnr-xilinx + prjxray flow lives in synth/xc7_flow.sh).
synth: gold
	@mkdir -p $(SIM_DIR)
	$(YOSYS) -q -p "read_verilog $(RTL); \
	    hierarchy -top t1llm_top; \
	    proc; opt; \
	    synth_xilinx -top t1llm_top; \
	    stat; \
	    write_verilog -noattr $(SIM_DIR)/t1llm_top_synth.v" \
	    -l $(SIM_DIR)/synth.log
	@echo "synth gate: OK (log in $(SIM_DIR)/synth.log)"

clean:
	rm -rf $(SIM_DIR)
