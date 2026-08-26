`timescale 1ns/1ps
// t1llm_top — top-level integration for the 1bit-LLM ternary GEMV engine.
//
// Wires: hif_slave (host bus + regs)  →  ctrl_fsm (sequencer)
//        →  gemv_core (4-lane ternary MAC) → scale_unit → ybuf,
// with wmem/xbuf dual-ported between host (port A) and datapath (port B).
// See DESIGN.md §2 for the wiring diagram.
module t1llm_top #(
    parameter WMEM_DEPTH = 8192,
    parameter XBUF_DEPTH = 4096,
    parameter YBUF_DEPTH = 4096
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        hif_wr_req,
    input  wire [15:0] hif_wr_addr,
    input  wire [31:0] hif_wr_data,
    input  wire        hif_rd_req,
    input  wire [15:0] hif_rd_addr,
    output wire [31:0] hif_rd_data,
    output wire        hif_rd_valid,
    output wire        hif_ready
);
    localparam WMEM_AW = $clog2(WMEM_DEPTH);
    localparam XBUF_AW = $clog2(XBUF_DEPTH);
    localparam YBUF_AW = $clog2(YBUF_DEPTH);

    // ---- hif_slave ↔ ctrl_fsm
    wire        start_pulse, soft_reset, clr_status;
    wire        ctrl_busy, ctrl_done, ctrl_err;
    wire [15:0] cfg_k, cfg_n, scale_q15;
    wire [7:0]  scale_shift;

    // ---- wmem port A (host) / port B (ctrl)
    wire [WMEM_AW-3:0] wmem_a_addr;
    wire               wmem_a_wen;
    wire [31:0]        wmem_a_wdata, wmem_a_rdata;
    wire [WMEM_AW-1:0] wb_addr;
    wire [7:0]         wb_rdata;

    // ---- xbuf port A (host) / port B (ctrl)
    wire [XBUF_AW-3:0] xbuf_a_addr;
    wire               xbuf_a_wen;
    wire [31:0]        xbuf_a_wdata, xbuf_a_rdata;
    wire [XBUF_AW-1:0] xb_addr;
    wire [7:0]         xb_rdata;

    // ---- ybuf port A (host read) / port B (ctrl write)
    wire [YBUF_AW-1:0] ybuf_a_addr;
    wire [31:0]        ybuf_a_rdata;
    wire [YBUF_AW-1:0] yw_addr;
    wire [31:0]        yw_data;
    wire               yw_wen;

    // ---- ctrl ↔ gemv ↔ scale
    wire        clr_acc, mac_en;
    wire [7:0]  act_in, w_in;
    wire [127:0] gemv_acc;
    wire [63:0] ysat;

    // ============================================================== host
    hif_slave #(
        .WMEM_AW (WMEM_AW),
        .XBUF_AW (XBUF_AW),
        .YBUF_AW (YBUF_AW)
    ) u_hif (
        .clk          (clk),
        .rst_n        (rst_n),
        .hif_wr_req   (hif_wr_req),
        .hif_wr_addr  (hif_wr_addr),
        .hif_wr_data  (hif_wr_data),
        .hif_rd_req   (hif_rd_req),
        .hif_rd_addr  (hif_rd_addr),
        .hif_rd_data  (hif_rd_data),
        .hif_rd_valid (hif_rd_valid),
        .hif_ready    (hif_ready),
        .start_pulse  (start_pulse),
        .soft_reset   (soft_reset),
        .clr_status   (clr_status),
        .cfg_k        (cfg_k),
        .cfg_n        (cfg_n),
        .scale_q15    (scale_q15),
        .scale_shift  (scale_shift),
        .ctrl_busy    (ctrl_busy),
        .ctrl_done    (ctrl_done),
        .ctrl_err     (ctrl_err),
        .wmem_a_addr  (wmem_a_addr),
        .wmem_a_wen   (wmem_a_wen),
        .wmem_a_wdata (wmem_a_wdata),
        .wmem_a_rdata (wmem_a_rdata),
        .xbuf_a_addr  (xbuf_a_addr),
        .xbuf_a_wen   (xbuf_a_wen),
        .xbuf_a_wdata (xbuf_a_wdata),
        .xbuf_a_rdata (xbuf_a_rdata),
        .ybuf_a_addr  (ybuf_a_addr),
        .ybuf_a_rdata (ybuf_a_rdata)
    );

    // ============================================================ control
    ctrl_fsm #(
        .WMEM_DEPTH (WMEM_DEPTH),
        .XBUF_DEPTH (XBUF_DEPTH),
        .YBUF_DEPTH (YBUF_DEPTH),
        .WMEM_AW    (WMEM_AW),
        .XBUF_AW    (XBUF_AW),
        .YBUF_AW    (YBUF_AW)
    ) u_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .start_pulse  (start_pulse),
        .soft_reset   (soft_reset),
        .clr_status   (clr_status),
        .cfg_k        (cfg_k),
        .cfg_n        (cfg_n),
        .scale_q15    (scale_q15),
        .scale_shift  (scale_shift),
        .busy         (ctrl_busy),
        .done         (ctrl_done),
        .err          (ctrl_err),
        .wb_addr      (wb_addr),
        .wb_rdata     (wb_rdata),
        .xb_addr      (xb_addr),
        .xb_rdata     (xb_rdata),
        .clr_acc      (clr_acc),
        .mac_en       (mac_en),
        .act_in       (act_in),
        .w_in         (w_in),
        .ysat         (ysat),
        .yw_addr      (yw_addr),
        .yw_data      (yw_data),
        .yw_wen       (yw_wen)
    );

    // =========================================================== datapath
    gemv_core u_gemv (
        .clk     (clk),
        .rst_n   (rst_n),
        .clr_acc (clr_acc),
        .mac_en  (mac_en),
        .act_in  (act_in),
        .w_in    (w_in),
        .acc     (gemv_acc)
    );

    scale_unit u_scale (
        .scale_q15   (scale_q15),
        .scale_shift (scale_shift),
        .acc_in      (gemv_acc),
        .y_out       (ysat)
    );

    // ============================================================ memories
    wmem #(
        .DEPTH (WMEM_DEPTH),
        .AW    (WMEM_AW)
    ) u_wmem (
        .clk     (clk),
        .a_addr  (wmem_a_addr),
        .a_wen   (wmem_a_wen),
        .a_wdata (wmem_a_wdata),
        .a_rdata (wmem_a_rdata),
        .b_addr  (wb_addr),
        .b_rdata (wb_rdata)
    );

    xbuf #(
        .DEPTH (XBUF_DEPTH),
        .AW    (XBUF_AW)
    ) u_xbuf (
        .clk     (clk),
        .a_addr  (xbuf_a_addr),
        .a_wen   (xbuf_a_wen),
        .a_wdata (xbuf_a_wdata),
        .a_rdata (xbuf_a_rdata),
        .b_addr  (xb_addr),
        .b_rdata (xb_rdata)
    );

    ybuf #(
        .DEPTH (YBUF_DEPTH),
        .AW    (YBUF_AW)
    ) u_ybuf (
        .clk     (clk),
        .a_addr  (ybuf_a_addr),
        .a_rdata (ybuf_a_rdata),
        .b_addr  (yw_addr),
        .b_wen   (yw_wen),
        .b_wdata (yw_data)
    );
endmodule
