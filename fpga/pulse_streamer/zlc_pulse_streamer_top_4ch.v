`timescale 1ns / 1ps
// First-light 4-channel top wrapper for zlc_pulse_streamer.
//
// VIO probe contract:
//   probe_out0 zlc_reset      width 1
//   probe_out1 zlc_start      width 1
//   probe_out2 zlc_prog_we    width 1
//   probe_out3 zlc_prog_addr  width 10
//   probe_out4 zlc_prog_tick  width 32
//   probe_out5 zlc_prog_mask  width 4
//   probe_out6 zlc_prog_count width 11
//   probe_in0  zlc_running    width 1
//   probe_in1  zlc_done       width 1

module zlc_pulse_streamer_top_4ch(
    input wire clk,
    output wire trap,
    output wire cooling,
    output wire probe,
    output wire qcm_trigger,
    output wire zlc_running_led,
    output wire zlc_done_led
);

    wire zlc_reset;
    wire zlc_start;
    wire zlc_prog_we;
    wire [9:0] zlc_prog_addr;
    wire [31:0] zlc_prog_tick;
    wire [3:0] zlc_prog_mask;
    wire [10:0] zlc_prog_count;
    wire [3:0] out;
    wire zlc_running;
    wire zlc_done;

    assign trap = out[0];
    assign cooling = out[1];
    assign probe = out[2];
    assign qcm_trigger = out[3];
    assign zlc_running_led = zlc_running;
    assign zlc_done_led = zlc_done;

    zlc_pulse_streamer #(
        .CHANNEL_COUNT(4),
        .EDGE_ADDR_WIDTH(10),
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
        .probe_out6(zlc_prog_count)
    );
endmodule
