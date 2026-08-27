`timescale 1ns/1ps
// scale_unit — 4× saturating integer scale stage.
//
// y[l] = sat16( (acc[l] * scale_q15 + rnd) >>> shift ),  rnd = shift ? 1<<(shift-1) : 0
//
// Purely combinational; mirrors tools/gen_golden.py bit-for-bit.
module scale_unit (
    input  wire [15:0]  scale_q15,   // signed
    input  wire [7:0]   scale_shift, // 0..15
    input  wire [127:0] acc_in,      // 4 × int32, lane l at [32l+31:32l]
    output wire [63:0]  y_out        // 4 × int16, lane l at [16l+15:16l]
);
    function automatic [15:0] scale_one;
        input [31:0] a;
        input [15:0] q15;
        input [7:0]  sh;
        reg signed [47:0] prod;
        reg signed [47:0] y;
        begin
            prod = $signed(a) * $signed(q15);
            y = prod;
            if (sh != 8'd0)
                y = prod + (48'sd1 << (sh - 1));
            y = y >>> sh;
            if (y > 48'sd32767)
                scale_one = 16'sd32767;
            else if (y < -48'sd32768)
                scale_one = -16'sd32768;
            else
                scale_one = y[15:0];
        end
    endfunction

    assign y_out[15:0]   = scale_one(acc_in[31:0],   scale_q15, scale_shift);
    assign y_out[31:16]  = scale_one(acc_in[63:32],  scale_q15, scale_shift);
    assign y_out[47:32]  = scale_one(acc_in[95:64],  scale_q15, scale_shift);
    assign y_out[63:48]  = scale_one(acc_in[127:96], scale_q15, scale_shift);
endmodule
