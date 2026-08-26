`timescale 1ns/1ps
// gemv_core — 4-lane ternary MAC datapath with int32 accumulators.
//
// Per cycle (mac_en): for each lane l, acc[l] += act * w[l], where
// w[l] decodes from 2 bits {sign,nz} to {-1,0,+1} — so the "multiply" is
// an add, a subtract, or a skip of the activation. No multipliers.
//
//   clr_acc: pulse that resets all accumulators (group start).
//   mac_en : act_in/w_in are valid; accumulate.
//   acc    : {acc3,acc2,acc1,acc0} — lane l at [32l+31:32l].
module gemv_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clr_acc,
    input  wire         mac_en,
    input  wire [7:0]   act_in,   // signed int8 activation
    input  wire [7:0]   w_in,     // 4 packed ternary weights, lane l at [2l+1:2l]
    output wire [127:0] acc       // 4 × int32
);
    reg signed [31:0] acc0, acc1, acc2, acc3;

    wire [1:0] wv0 = w_in[1:0];
    wire [1:0] wv1 = w_in[3:2];
    wire [1:0] wv2 = w_in[5:4];
    wire [1:0] wv3 = w_in[7:6];

    wire signed [8:0] act9 = $signed(act_in);

    // contribution: nz ? (sign ? -act : act) : 0
    wire signed [8:0] c0 = wv0[0] ? (wv0[1] ? -act9 : act9) : 9'sd0;
    wire signed [8:0] c1 = wv1[0] ? (wv1[1] ? -act9 : act9) : 9'sd0;
    wire signed [8:0] c2 = wv2[0] ? (wv2[1] ? -act9 : act9) : 9'sd0;
    wire signed [8:0] c3 = wv3[0] ? (wv3[1] ? -act9 : act9) : 9'sd0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc0 <= 32'sd0;
            acc1 <= 32'sd0;
            acc2 <= 32'sd0;
            acc3 <= 32'sd0;
        end else if (clr_acc) begin
            acc0 <= 32'sd0;
            acc1 <= 32'sd0;
            acc2 <= 32'sd0;
            acc3 <= 32'sd0;
        end else if (mac_en) begin
            acc0 <= acc0 + {{23{c0[8]}}, c0};
            acc1 <= acc1 + {{23{c1[8]}}, c1};
            acc2 <= acc2 + {{23{c2[8]}}, c2};
            acc3 <= acc3 + {{23{c3[8]}}, c3};
        end
    end

    assign acc = {acc3, acc2, acc1, acc0};
endmodule
