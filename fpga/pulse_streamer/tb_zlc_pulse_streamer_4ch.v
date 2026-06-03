`timescale 1ns / 1ps

module tb_zlc_pulse_streamer_4ch;
    reg clk = 1'b0;
    reg reset = 1'b1;
    reg start = 1'b0;
    reg prog_we = 1'b0;
    reg [3:0] prog_addr = 4'd0;
    reg [15:0] prog_tick = 16'd0;
    reg [3:0] prog_mask = 4'd0;
    reg [4:0] prog_count = 5'd0;
    wire [3:0] out;
    wire running;
    wire done;

    integer cycle;
    reg [3:0] expected;

    always #5 clk = ~clk;

    zlc_pulse_streamer #(
        .CHANNEL_COUNT(4),
        .EDGE_ADDR_WIDTH(4),
        .TICK_WIDTH(16)
    ) dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .prog_we(prog_we),
        .prog_addr(prog_addr),
        .prog_tick(prog_tick),
        .prog_mask(prog_mask),
        .prog_count(prog_count),
        .out(out),
        .running(running),
        .done(done)
    );

    task upload_edge;
        input [3:0] addr;
        input [15:0] tick;
        input [3:0] mask;
        begin
            @(negedge clk);
            prog_addr = addr;
            prog_tick = tick;
            prog_mask = mask;
            prog_we = 1'b1;
            repeat (3) @(negedge clk);
            prog_we = 1'b0;
            repeat (2) @(negedge clk);
        end
    endtask

    task fail;
        input [1023:0] message;
        begin
            $display("ZLC_SIM_FAIL: %0s", message);
            $finish;
        end
    endtask

    initial begin
        $display("ZLC_SIM_START: 4ch pulse streamer core");
        repeat (4) @(posedge clk);

        prog_count = 5'd6;
        upload_edge(4'd0, 16'd0,  4'b0011);
        upload_edge(4'd1, 16'd3,  4'b0001);
        upload_edge(4'd2, 16'd5,  4'b1101);
        upload_edge(4'd3, 16'd6,  4'b0101);
        upload_edge(4'd4, 16'd12, 4'b0001);
        upload_edge(4'd5, 16'd13, 4'b0000);

        repeat (4) @(posedge clk);
        reset = 1'b0;
        @(negedge clk);
        start = 1'b1;

        wait (running === 1'b1);
        @(negedge clk);
        start = 1'b0;
        if (done !== 1'b0) fail("done asserted while run is starting");

        for (cycle = 0; cycle <= 13; cycle = cycle + 1) begin
            @(posedge clk);
            #1;
            if (cycle < 3)
                expected = 4'b0011;
            else if (cycle < 5)
                expected = 4'b0001;
            else if (cycle < 6)
                expected = 4'b1101;
            else if (cycle < 12)
                expected = 4'b0101;
            else if (cycle < 13)
                expected = 4'b0001;
            else
                expected = 4'b0000;

            if (out !== expected) begin
                $display("cycle=%0d expected=%b out=%b running=%b done=%b", cycle, expected, out, running, done);
                fail("output mask mismatch");
            end
            if (cycle < 13 && running !== 1'b1) fail("running dropped early");
        end

        if (running !== 1'b0) fail("running did not drop after final tick");
        if (done !== 1'b1) fail("done did not assert after final tick");
        if (out !== 4'b0000) fail("outputs are not safe after completion");

        $display("ZLC_SIM_PASS: 4ch pulse streamer upload/start/timing/done verified");
        $finish;
    end
endmodule
