`timescale 1ns/1ps
// xbuf — activation buffer (signed int8 per entry).
// Same dual-port shape as wmem: host writes 32-bit words (4 acts), gemv reads bytes.
module xbuf #(
    parameter DEPTH = 4096,   // entries (bytes)
    parameter AW    = 12      // $clog2(DEPTH)
)(
    input  wire             clk,
    // port A — host
    input  wire [AW-3:0]    a_addr,   // word index
    input  wire             a_wen,
    input  wire [31:0]      a_wdata,
    output reg  [31:0]      a_rdata,
    // port B — gemv
    input  wire [AW-1:0]    b_addr,   // entry index (== k)
    output reg  [7:0]       b_rdata
);
    reg [7:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (a_wen) begin
            mem[{a_addr,2'b00}] <= a_wdata[7:0];
            mem[{a_addr,2'b01}] <= a_wdata[15:8];
            mem[{a_addr,2'b10}] <= a_wdata[23:16];
            mem[{a_addr,2'b11}] <= a_wdata[31:24];
        end
        a_rdata <= {mem[{a_addr,2'b11}], mem[{a_addr,2'b10}],
                    mem[{a_addr,2'b01}], mem[{a_addr,2'b00}]};
        b_rdata <= mem[b_addr];
    end
endmodule
