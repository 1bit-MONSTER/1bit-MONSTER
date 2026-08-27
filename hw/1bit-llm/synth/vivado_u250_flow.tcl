# vivado_u250_flow.tcl — Vivado implementation flow for the 1bit-LLM RTL on the
# Alveo U250 (xcu250-figd2104-2L-e). Proves the portable RTL drops into a
# Vivado project and closes place & route on the eventual box (journey UPDATE 32).
#
# Usage (on the box with Vivado 2026.1 + Alveo license):
#   source /home/bcloud/Xilinx/2026.1/Vivado/settings64.sh
#   # legacy libc (Ubuntu 24): Vivado needs libncurses.so.5/libtinfo.so.5
#   export LD_LIBRARY_PATH=/home/bcloud/FPGAs_AdaptiveSoCs_Unified_SDI_2026.1_0616_1700/lib/lnx64.o/Ubuntu/24
#   vivado -mode batch -nolog -nojournal -source vivado_u250_flow.tcl \
#       -tclargs xcu250-figd2104-2L-e
#
# Result (2026-08-26, Vivado v2026.1): synth -> opt -> place -> route all
# succeeded ("DONE_OK"), 0 errors. Utilization (routed):
#   LUTs 98123 | FFs 99666 | URAM 1 | DSP 10 | BRAM 0
#   (wmem/xbuf inferred as FF arrays; ybuf as URAM; scale_unit uses 8 DSPs)
# No clock constraint was supplied, so timing reports NA (flow is a P&R-closure
# gate, not a frequency target).

set part [lindex $argv 0]
puts "== 1bit-LLM Vivado flow: part=$part =="
read_verilog [glob [file dirname [info script]]/../rtl/*.v]
synth_design -top t1llm_top -part $part
opt_design
place_design
route_design
report_utilization -file /tmp/1bitllm_util.rpt -hierarchical
report_timing_summary -file /tmp/1bitllm_timing.rpt
puts "DONE_OK part=$part"
