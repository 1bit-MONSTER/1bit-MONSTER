`timescale 1ns/1ps
// ctrl_fsm — GEMV sequencer + status.
//
// Runs one GEMV:  y[n] = sat16( ( Σ_k act[k]·w[k][n] ) · scale_q15 >> shift )
// for n in 0..N-1, K activations, using the 4-lane gemv_core and scale_unit.
//
// State flow (see DESIGN.md §5.2):
//   IDLE → CHECK → (ADDR ⇄ MAC)×K → DRAIN1 → DRAIN2 → (next g | DONE) → IDLE
//
// Timing: BRAM reads are registered, so ADDR issues the address and MAC (one
// cycle later) consumes the data. clr_acc is asserted during ADDR when k==0 so
// the accumulators reset one cycle before the first MAC.
module ctrl_fsm #(
    parameter WMEM_DEPTH = 8192,
    parameter XBUF_DEPTH = 4096,
    parameter YBUF_DEPTH = 4096,
    parameter WMEM_AW    = 13,
    parameter XBUF_AW    = 12,
    parameter YBUF_AW    = 12
)(
    input  wire              clk,
    input  wire              rst_n,
    // control from hif
    input  wire              start_pulse,
    input  wire              soft_reset,
    input  wire              clr_status,
    input  wire [15:0]       cfg_k,
    input  wire [15:0]       cfg_n,
    input  wire [15:0]       scale_q15,
    input  wire [7:0]        scale_shift,
    // status to hif
    output reg               busy,
    output reg               done,
    output reg               err,
    // wmem port B
    output reg  [WMEM_AW-1:0] wb_addr,
    input  wire [7:0]        wb_rdata,
    // xbuf port B
    output reg  [XBUF_AW-1:0] xb_addr,
    input  wire [7:0]        xb_rdata,
    // gemv handshake
    output wire              clr_acc,
    output reg               mac_en,
    output reg  [7:0]        act_in,
    output reg  [7:0]        w_in,
    // scaled results (from scale_unit, combinational on gemv acc)
    input  wire [63:0]       ysat,
    // ybuf write
    output reg  [YBUF_AW-1:0] yw_addr,
    output reg  [31:0]       yw_data,
    output reg               yw_wen
);
    localparam S_IDLE   = 3'd0,
               S_CHECK  = 3'd1,
               S_ADDR   = 3'd2,
               S_MAC    = 3'd3,
               S_DRAIN1 = 3'd4,
               S_DRAIN2 = 3'd5,
               S_DONE   = 3'd6,
               S_ERR    = 3'd7;

    reg [2:0]  state;
    reg [15:0] k, g;
    reg [15:0] k_max, groups;

    // ---------------------------------------------------------------- FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy  <= 1'b0;
            done  <= 1'b0;
            err   <= 1'b0;
            k     <= 16'd0;
            g     <= 16'd0;
            k_max <= 16'd0;
            groups<= 16'd0;
        end else if (soft_reset) begin
            state <= S_IDLE;
            busy  <= 1'b0;
            done  <= 1'b0;
            err   <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start_pulse) begin
                        k_max  <= cfg_k;
                        groups <= cfg_n >> 2;
                        k      <= 16'd0;
                        g      <= 16'd0;
                        busy   <= 1'b1;
                        done   <= 1'b0;
                        err    <= 1'b0;
                        state  <= S_CHECK;
                    end else if (clr_status) begin
                        done <= 1'b0;
                        err  <= 1'b0;
                    end
                end
                S_CHECK: begin
                    if (k_max == 16'd0                     || // K >= 1
                        groups == 16'd0                    || // N >= 4
                        cfg_n[1:0] != 2'b00                || // N % 4 == 0
                        k_max > XBUF_DEPTH                 || // K fits xbuf
                        (cfg_n >> 1) > YBUF_DEPTH          || // N/2 fits ybuf
                        (k_max * groups) > WMEM_DEPTH      || // entries fit wmem
                        scale_shift > 8'd15) begin           // shift in 0..15 (scale_unit contract)
                        busy  <= 1'b0;
                        err   <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        state <= S_ADDR;
                    end
                end
                S_ADDR: begin
                    state <= S_MAC;
                end
                S_MAC: begin
                    if (k == k_max - 16'd1)
                        state <= S_DRAIN1;
                    else begin
                        k     <= k + 16'd1;
                        state <= S_ADDR;
                    end
                end
                S_DRAIN1: begin
                    state <= S_DRAIN2;
                end
                S_DRAIN2: begin
                    if (g + 16'd1 == groups) begin
                        state <= S_DONE;
                    end else begin
                        g     <= g + 16'd1;
                        k     <= 16'd0;
                        state <= S_ADDR;
                    end
                end
                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                S_ERR: begin
                    busy  <= 1'b0;
                    err   <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // --------------------------------------------------- combinational out
    assign clr_acc = (state == S_ADDR) && (k == 16'd0);

    always @(*) begin
        // defaults
        wb_addr = {WMEM_AW{1'b0}};
        xb_addr = {XBUF_AW{1'b0}};
        mac_en  = 1'b0;
        act_in  = xb_rdata;
        w_in    = wb_rdata;
        yw_wen  = 1'b0;
        yw_addr = {YBUF_AW{1'b0}};
        yw_data = 32'd0;
        case (state)
            S_ADDR: begin
                wb_addr = (g * k_max) + k;   // entry e = g*K + k  (validated ≤ WMEM_DEPTH-1)
                xb_addr = k;
            end
            S_MAC: begin
                mac_en  = 1'b1;
                act_in  = xb_rdata;          // data for the addr issued in ADDR (1-cycle BRAM read)
                w_in    = wb_rdata;
            end
            S_DRAIN1: begin
                yw_wen  = 1'b1;
                yw_addr = g << 1;                            // ybuf[2g]  (g bounded by CHECK: 2g+1 <= N/2-1 <= YBUF_DEPTH-1)
                yw_data = {ysat[31:16], ysat[15:0]};         // {y1, y0}
            end
            S_DRAIN2: begin
                yw_wen  = 1'b1;
                yw_addr = (g << 1) + 1'b1;                   // ybuf[2g+1]
                yw_data = {ysat[63:48], ysat[47:32]};        // {y3, y2}
            end
            default: ;
        endcase
    end
endmodule
