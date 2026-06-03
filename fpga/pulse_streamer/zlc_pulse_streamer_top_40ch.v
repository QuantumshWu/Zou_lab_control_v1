`timescale 1ns / 1ps
// 40-channel top wrapper for zlc_pulse_streamer.
//
// Complete zlc_pulse_streamer_40ch.xdc before building this top.

module zlc_pulse_streamer_top_40ch(
    input wire clk,
    output wire [39:0] ch,
    output wire zlc_running_led,
    output wire zlc_done_led
);

    wire zlc_reset;
    wire zlc_start;
    wire zlc_prog_we;
    wire [6:0] zlc_prog_addr;
    wire [31:0] zlc_prog_tick;
    wire [39:0] zlc_prog_mask;
    wire [7:0] zlc_prog_count;
    wire zlc_repeat_forever;
    wire [6:0] zlc_loop_start_addr;
    wire [31:0] zlc_loop_end_tick;
    wire [31:0] zlc_loop_count;
    wire [39:0] out;
    wire zlc_running;
    wire zlc_done;

    assign ch = out;
    assign zlc_running_led = zlc_running;
    assign zlc_done_led = zlc_done;

    zlc_pulse_streamer #(
        .CHANNEL_COUNT(40),
        .EDGE_ADDR_WIDTH(7),
        .TICK_WIDTH(32)
    ) zlc_streamer_i (
        .clk(clk),
        .reset(zlc_reset),
        .start(zlc_start),
        .prog_we(zlc_prog_we),
        .prog_addr(zlc_prog_addr),
        .prog_tick(zlc_prog_tick),
        .prog_mask(zlc_prog_mask),
        .prog_count(zlc_prog_count),
        .repeat_forever(zlc_repeat_forever),
        .loop_start_addr(zlc_loop_start_addr),
        .loop_end_tick(zlc_loop_end_tick),
        .loop_count(zlc_loop_count),
        .out(out),
        .running(zlc_running),
        .done(zlc_done)
    );

    vio_0 zlc_vio_i (
        .clk(clk),
        .probe_in0(zlc_running),
        .probe_in1(zlc_done),
        .probe_out0(zlc_reset),
        .probe_out1(zlc_start),
        .probe_out2(zlc_prog_we),
        .probe_out3(zlc_prog_addr),
        .probe_out4(zlc_prog_tick),
        .probe_out5(zlc_prog_mask),
        .probe_out6(zlc_prog_count),
        .probe_out7(zlc_repeat_forever),
        .probe_out8(zlc_loop_start_addr),
        .probe_out9(zlc_loop_end_tick),
        .probe_out10(zlc_loop_count)
    );
endmodule
