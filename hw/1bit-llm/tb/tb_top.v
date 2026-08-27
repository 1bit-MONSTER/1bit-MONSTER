// tb_top — host-interface-driven testbench for t1llm_top.
//
// Runs two randomized cases (golden vectors from tools/gen_golden.py):
//   case0: K=8,  N=16   · case1: K=16, N=8
// Every interaction with the DUT goes through the host bus: program CFG,
// write wmem/xbuf, start, poll STATUS, read ybuf — then compare bit-exact
// against the golden files. Run from hw/1bit-llm/sim (see Makefile).
`timescale 1ns/1ps

module tb_top;
    parameter CLK_PERIOD = 10;

    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- DUT
    reg         rst_n;
    reg         hif_wr_req, hif_rd_req;
    reg  [15:0] hif_wr_addr, hif_rd_addr;
    reg  [31:0] hif_wr_data;
    wire [31:0] hif_rd_data;
    wire        hif_rd_valid, hif_ready;

    t1llm_top dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .hif_wr_req  (hif_wr_req),
        .hif_wr_addr (hif_wr_addr),
        .hif_wr_data (hif_wr_data),
        .hif_rd_req  (hif_rd_req),
        .hif_rd_addr (hif_rd_addr),
        .hif_rd_data (hif_rd_data),
        .hif_rd_valid(hif_rd_valid),
        .hif_ready   (hif_ready)
    );

    // ---- golden memories (loaded per case; sized to the largest case so
    // $readmemh has no leftover-array warnings: case0 K=8 N=16 → wmem 32,
    // act 8, ygold 8; case1 K=16 N=8 → wmem 32, act 16, ygold 4)
    reg [7:0]  tb_wmem  [0:31];
    reg [7:0]  tb_act   [0:15];
    reg [31:0] tb_ygold [0:7];

    reg global_fail = 0;

    // ============================================================ bus tasks
    task hif_write;
        input [15:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            hif_wr_req  = 1;
            hif_wr_addr = addr;
            hif_wr_data = data;
            @(posedge clk);
            hif_wr_req  = 0;
        end
    endtask

    task hif_read;
        input  [15:0] addr;
        output [31:0] data;
        integer t;
        begin
            @(posedge clk);
            hif_rd_req  = 1;
            hif_rd_addr = addr;
            @(posedge clk);
            hif_rd_req  = 0;
            t = 0;
            while (!hif_rd_valid && t <= 16) begin
                @(posedge clk);
                t = t + 1;
            end
            if (t > 16) begin
                $display("FATAL: hif read timeout at addr %04h", addr);
                $finish;
            end
            data = hif_rd_data;
            @(posedge clk);
        end
    endtask

    // ============================================================ run a case
    task run_case;
        input integer cid;
        reg [15:0] K, N, q15;
        reg [7:0]  sh;
        integer fd, i, w_entries, mismatches, timeout, nread;
        reg [31:0] rdata;
        begin
            mismatches = 0;

            // fresh reset
            rst_n = 0;
            repeat (2) @(posedge clk);
            rst_n = 1;
            repeat (2) @(posedge clk);

            // ---- load golden data for this case (explicit ranges keep
            // $readmemh warning-free: file length == range length)
            if (cid == 0) begin
                $readmemh("case0_wmem.hex", tb_wmem, 0, 31);
                $readmemh("case0_act.hex",  tb_act,  0, 7);
                $readmemh("case0_ygold.hex",tb_ygold,0, 7);
                fd = $fopen("case0_cfg.txt", "r");
            end else begin
                $readmemh("case1_wmem.hex", tb_wmem, 0, 31);
                $readmemh("case1_act.hex",  tb_act,  0, 15);
                $readmemh("case1_ygold.hex",tb_ygold,0, 3);
                fd = $fopen("case1_cfg.txt", "r");
            end
            if (fd == 0) begin
                $display("FATAL: cannot open cfg file for case %0d", cid);
                global_fail = 1;
                $finish;
            end
            nread = $fscanf(fd, "%d %d %d %d", K, N, q15, sh);
            if (nread != 4) begin
                $display("FATAL: malformed cfg file for case %0d", cid);
                global_fail = 1;
                $finish;
            end
            $fclose(fd);

            // ---- program config
            hif_write(16'h0002, {16'd0, K});
            hif_write(16'h0003, {16'd0, N});
            hif_write(16'h0004, {8'd0, sh, q15});

            // ---- load wmem: w_entries = (N/4)*K bytes, 4 per word
            w_entries = (N / 4) * K;
            for (i = 0; i < w_entries; i = i + 4)
                hif_write(16'h1000 + i[15:2],
                          {tb_wmem[i+3], tb_wmem[i+2], tb_wmem[i+1], tb_wmem[i]});

            // ---- load activations: K bytes, 4 per word
            for (i = 0; i < K; i = i + 4)
                hif_write(16'h2000 + i[15:2],
                          {tb_act[i+3], tb_act[i+2], tb_act[i+1], tb_act[i]});

            // ---- go
            hif_write(16'h0000, 32'd1);            // CTRL.start

            // ---- wait for completion
            timeout = 0;
            rdata = 0;
            while (!rdata[1] && !rdata[2] && timeout <= 1000000) begin
                hif_read(16'h0001, rdata);          // STATUS
                timeout = timeout + 1;
            end
            if (rdata[2]) begin
                $display("case%0d: ERR status (K=%0d N=%0d)", cid, K, N);
                mismatches = 1;
            end
            if (timeout > 1000000) begin
                $display("case%0d: TIMEOUT waiting for done (K=%0d N=%0d)",
                         cid, K, N);
                mismatches = 1;
            end

            // ---- read back and compare
            if (!rdata[2] && mismatches == 0) begin
                for (i = 0; i < N / 2; i = i + 1) begin
                    hif_read(16'h3000 + i[11:0], rdata);
                    if (rdata !== tb_ygold[i]) begin
                        mismatches = mismatches + 1;
                        if (mismatches <= 8)
                            $display("case%0d: ybuf[%0d] got %08h exp %08h",
                                     cid, i, rdata, tb_ygold[i]);
                    end
                end
            end

            if (mismatches == 0)
                $display("case%0d: PASS  (K=%0d N=%0d scale_q15=%0d shift=%0d)",
                         cid, K, N, q15, sh);
            else begin
                $display("case%0d: FAIL  (%0d mismatches)", cid, mismatches);
                global_fail = 1;
            end
        end
    endtask

    // ======================================================= error path test
    // Invalid config (N not a multiple of 4) must be rejected with STATUS.err;
    // clr_status must then clear the error latch.
    task run_err_case;
        reg [31:0] rdata;
        integer timeout;
        begin
            rst_n = 0;
            repeat (2) @(posedge clk);
            rst_n = 1;
            repeat (2) @(posedge clk);

            hif_write(16'h0002, 32'd8);              // K = 8
            hif_write(16'h0003, 32'd6);              // N = 6  → invalid
            hif_write(16'h0004, 32'h00009000);       // scale_q15 = 0x9000
            hif_write(16'h0000, 32'd1);              // start

            timeout = 0;
            rdata = 0;
            while (!rdata[1] && !rdata[2] && timeout <= 10000) begin
                hif_read(16'h0001, rdata);           // STATUS
                timeout = timeout + 1;
            end
            if (rdata[2])
                $display("err-case: PASS (N=6 rejected, ERR latched)");
            else begin
                $display("err-case: FAIL (expected ERR; got busy=%0d done=%0d)",
                         rdata[0], rdata[1]);
                global_fail = 1;
            end

            hif_write(16'h0000, 32'h4);              // CTRL.clr_status
            hif_read(16'h0001, rdata);
            if (!rdata[2])
                $display("clr-status: PASS (ERR cleared)");
            else begin
                $display("clr-status: FAIL (ERR still set)");
                global_fail = 1;
            end

            // second invalid config: shift out of the 0..15 contract
            // (scale_unit's arithmetic-shift path is undefined past the
            // 48-bit accumulator width, so CHECK must reject it)
            hif_write(16'h0002, 32'd8);              // K = 8
            hif_write(16'h0003, 32'd8);              // N = 8  (valid)
            hif_write(16'h0004, 32'h10_9000);        // shift = 16 → invalid
            hif_write(16'h0000, 32'd1);              // start

            timeout = 0;
            rdata = 0;
            while (!rdata[1] && !rdata[2] && timeout <= 10000) begin
                hif_read(16'h0001, rdata);           // STATUS
                timeout = timeout + 1;
            end
            if (rdata[2])
                $display("shift-case: PASS (shift=16 rejected, ERR latched)");
            else begin
                $display("shift-case: FAIL (expected ERR; got busy=%0d done=%0d)",
                         rdata[0], rdata[1]);
                global_fail = 1;
            end

            hif_write(16'h0000, 32'h4);              // CTRL.clr_status
            hif_read(16'h0001, rdata);
            if (!rdata[2])
                $display("clr-status2: PASS (ERR cleared)");
            else begin
                $display("clr-status2: FAIL (ERR still set)");
                global_fail = 1;
            end
        end
    endtask

    // ================================================================ main
    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);

        rst_n       = 0;
        hif_wr_req  = 0;
        hif_rd_req  = 0;
        hif_wr_addr = 16'd0;
        hif_rd_addr = 16'd0;
        hif_wr_data = 32'd0;

        repeat (4) @(posedge clk);
        rst_n = 1;

        run_case(0);
        run_case(1);
        run_err_case();

        if (global_fail)
            $display("=== TB: FAIL ===");
        else
            $display("=== TB: ALL PASS ===");
        $finish;
    end
endmodule
