`timescale 1ns/1ps
// ybuf — output buffer, one 32-bit word = two signed int16 outputs {y[2g+1], y[2g]}.
//   port A: host read
//   port B: gemv/ctrl write
module ybuf #(
    parameter DEPTH = 4096,   // words
    parameter AW    = 12      // $clog2(DEPTH)
)(
    input  wire          clk,
    // port A — host read
    input  wire [AW-1:0] a_addr,
    output reg  [31:0]   a_rdata,
    // port B — datapath write
    input  wire [AW-1:0] b_addr,
    input  wire          b_wen,
    input  wire [31:0]   b_wdata
);
    reg [31:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (b_wen)
            mem[b_addr] <= b_wdata;
        a_rdata <= mem[a_addr];
    end
endmodule
