`timescale 1ns/1ps
// hif_slave — host interface: register file, address decode, memory port A.
//
// Single-word synchronous bus, always ready (hif_ready = 1).
// Reads are 3 cycles: rd_req → (address applied) → rd_valid + rd_data.
// Writes are 1 cycle.
//
// Address map (word addresses, see DESIGN.md §3):
//   0x0000 CTRL   [0] start · [1] soft_reset · [2] clr_status   (write strobes)
//   0x0001 STATUS [0] busy · [1] done · [2] err                  (read-only)
//   0x0002 CFG_K  [15:0] rows (activation length)
//   0x0003 CFG_N  [15:0] columns (output length, multiple of 4)
//   0x0004 SCALE  [15:0] scale_q15 (signed) · [23:16] shift
//   0x1xxx wmem   word = 4 entries (entry 4a in low byte)
//   0x2xxx xbuf   word = 4 int8 activations
//   0x3xxx ybuf   word = 2 int16 outputs (read-only)
module hif_slave #(
    parameter WMEM_AW = 13,
    parameter XBUF_AW = 12,
    parameter YBUF_AW = 12
)(
    input  wire               clk,
    input  wire               rst_n,
    // host bus
    input  wire               hif_wr_req,
    input  wire [15:0]        hif_wr_addr,
    input  wire [31:0]        hif_wr_data,
    input  wire               hif_rd_req,
    input  wire [15:0]        hif_rd_addr,
    output reg  [31:0]        hif_rd_data,
    output reg                hif_rd_valid,
    output wire               hif_ready,
    // control to ctrl_fsm
    output reg                start_pulse,
    output reg                soft_reset,
    output reg                clr_status,
    output reg  [15:0]        cfg_k,
    output reg  [15:0]        cfg_n,
    output reg  [15:0]        scale_q15,
    output reg  [7:0]         scale_shift,
    // status from ctrl_fsm
    input  wire               ctrl_busy,
    input  wire               ctrl_done,
    input  wire               ctrl_err,
    // wmem port A
    output reg  [WMEM_AW-3:0] wmem_a_addr,
    output reg                wmem_a_wen,
    output reg  [31:0]        wmem_a_wdata,
    input  wire [31:0]        wmem_a_rdata,
    // xbuf port A
    output reg  [XBUF_AW-3:0] xbuf_a_addr,
    output reg                xbuf_a_wen,
    output reg  [31:0]        xbuf_a_wdata,
    input  wire [31:0]        xbuf_a_rdata,
    // ybuf port A (read)
    output reg  [YBUF_AW-1:0] ybuf_a_addr,
    input  wire [31:0]        ybuf_a_rdata
);
    // read pipeline (3 cycles): req → addr applied → data sampled
    reg        rd_pending1, rd_pending2;
    reg [15:0] rd_addr_q1,  rd_addr_q2;

    assign hif_ready = 1'b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pulse  <= 1'b0;
            soft_reset   <= 1'b0;
            clr_status   <= 1'b0;
            cfg_k        <= 16'd0;
            cfg_n        <= 16'd0;
            scale_q15    <= 16'd0;
            scale_shift  <= 8'd0;
            hif_rd_data  <= 32'd0;
            hif_rd_valid <= 1'b0;
            rd_pending1  <= 1'b0;
            rd_pending2  <= 1'b0;
            rd_addr_q1   <= 16'd0;
            rd_addr_q2   <= 16'd0;
            wmem_a_addr  <= {WMEM_AW-2{1'b0}};
            wmem_a_wen   <= 1'b0;
            wmem_a_wdata <= 32'd0;
            xbuf_a_addr  <= {XBUF_AW-2{1'b0}};
            xbuf_a_wen   <= 1'b0;
            xbuf_a_wdata <= 32'd0;
            ybuf_a_addr  <= {YBUF_AW{1'b0}};
        end else begin
            // one-cycle strobes default to 0
            start_pulse <= 1'b0;
            soft_reset  <= 1'b0;
            clr_status  <= 1'b0;
            wmem_a_wen  <= 1'b0;
            xbuf_a_wen  <= 1'b0;

            // ------------------------------------------------- writes
            if (hif_wr_req) begin
                case (hif_wr_addr[15:12])
                    4'h0: begin
                        if (hif_wr_addr[3:0] == 4'h0) begin
                            start_pulse <= hif_wr_data[0];
                            soft_reset  <= hif_wr_data[1];
                            clr_status  <= hif_wr_data[2];
                        end
                        if (hif_wr_addr[3:0] == 4'h2) cfg_k       <= hif_wr_data[15:0];
                        if (hif_wr_addr[3:0] == 4'h3) cfg_n       <= hif_wr_data[15:0];
                        if (hif_wr_addr[3:0] == 4'h4) begin
                            scale_q15   <= hif_wr_data[15:0];
                            scale_shift <= hif_wr_data[23:16];
                        end
                    end
                    4'h1: begin
                        wmem_a_wen   <= 1'b1;
                        wmem_a_addr  <= hif_wr_addr[10:0];
                        wmem_a_wdata <= hif_wr_data;
                    end
                    4'h2: begin
                        xbuf_a_wen   <= 1'b1;
                        xbuf_a_addr  <= hif_wr_addr[9:0];
                        xbuf_a_wdata <= hif_wr_data;
                    end
                    default: ;
                endcase
            end

            // -------------------------------------------------- reads
            if (hif_rd_req) begin
                rd_pending1 <= 1'b1;
                rd_addr_q1  <= hif_rd_addr;
                case (hif_rd_addr[15:12])
                    4'h1: wmem_a_addr <= hif_rd_addr[10:0];
                    4'h2: xbuf_a_addr <= hif_rd_addr[9:0];
                    4'h3: ybuf_a_addr <= hif_rd_addr[11:0];
                    default: ;
                endcase
            end

            if (rd_pending1) begin
                rd_pending1 <= 1'b0;
                rd_pending2 <= 1'b1;
                rd_addr_q2  <= rd_addr_q1;
            end

            if (rd_pending2) begin
                rd_pending2 <= 1'b0;
                hif_rd_valid <= 1'b1;
                case (rd_addr_q2[15:12])
                    4'h0: begin
                        case (rd_addr_q2[3:0])
                            4'h0: hif_rd_data <= {31'd0, soft_reset};
                            4'h1: hif_rd_data <= {29'd0, ctrl_err, ctrl_done, ctrl_busy};
                            4'h2: hif_rd_data <= {16'd0, cfg_k};
                            4'h3: hif_rd_data <= {16'd0, cfg_n};
                            4'h4: hif_rd_data <= {8'd0, scale_shift, scale_q15};
                            default: hif_rd_data <= 32'd0;
                        endcase
                    end
                    4'h1: hif_rd_data <= wmem_a_rdata;
                    4'h2: hif_rd_data <= xbuf_a_rdata;
                    4'h3: hif_rd_data <= ybuf_a_rdata;
                    default: hif_rd_data <= 32'd0;
                endcase
            end else begin
                hif_rd_valid <= 1'b0;
            end
        end
    end
endmodule
