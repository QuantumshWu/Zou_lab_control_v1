`timescale 1ns / 1ps
// Address-switch pin-map top wrapper for zlc_pulse_streamer (N-slot scan).
//
// The file name is kept for backward compatibility with older scripts, but the
// hardware contract is inferred from the historical address_switch XDC:
// 62 controllable outputs plus tied-low GND helper pins and two status LEDs.
//
// NUM_SLOTS is fixed in the bitstream (4 here).  Per-edge scan coefficients and
// per-point slot values are packed little-endian into single wide VIO probes:
//   prog_tick_coeffs : NUM_SLOTS*COEFF_WIDTH = 4*16 = 64 bits
//   scan_prog_values : NUM_SLOTS*TICK_WIDTH  = 4*32 = 128 bits
//   loop_end_coeffs  : NUM_SLOTS*COEFF_WIDTH = 4*16 = 64 bits

module zlc_pulse_streamer_top_address_switch(
    input wire clk,
    output wire [1:0] led,
    output wire cooling,
    output wire cooling_pgc,
    output wire repump,
    output wire probe,
    output wire pushout,
    output wire state_pre,
    output wire trig,
    output wire coil,
    output wire grey_cooling,
    output wire trap,
    output wire UV,
    output wire emCCD,
    output wire microwave,
    output wire address,
    output wire GND1,
    output wire GND4,
    output wire GND5,
    output wire GND6,
    output wire GND7,
    output wire GND8,
    output wire GND9,
    output wire GND10,
    output wire GND11,
    output wire cooling_shutter,
    output wire GND12,
    output wire repump_shutter,
    output wire GND13,
    output wire probe_shutter,
    output wire GND14,
    output wire bias,
    output wire GND15,
    output wire [9:0] da_dipole,
    output wire da_clk0,
    output wire [9:0] da_bias_y,
    output wire da_clk1,
    output wire [9:0] da_bias_x,
    output wire da_clk2,
    output wire [9:0] da_bias_z,
    output wire da_clk3
);

    localparam integer CHANNEL_COUNT = 62;
    localparam integer NUM_SLOTS = 4;
    localparam integer COEFF_WIDTH = 16;
    localparam integer TICK_WIDTH = 32;

    wire zlc_reset;
    wire zlc_start;
    wire zlc_prog_we;
    wire [9:0] zlc_prog_addr;
    wire [31:0] zlc_prog_tick;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_prog_tick_coeffs;
    wire [CHANNEL_COUNT-1:0] zlc_prog_mask;
    wire [10:0] zlc_prog_count;
    wire zlc_repeat_forever;
    wire [9:0] zlc_loop_start_addr;
    wire [31:0] zlc_loop_end_tick;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_loop_end_coeffs;
    wire [31:0] zlc_loop_count;
    wire zlc_scan_enable;
    wire zlc_scan_prog_we;
    wire [9:0] zlc_scan_prog_addr;
    wire [NUM_SLOTS*TICK_WIDTH-1:0] zlc_scan_prog_values;
    wire [10:0] zlc_scan_count;
    wire zlc_bus_prog_we;
    wire [1:0] zlc_bus_prog_bus;
    wire [5:0] zlc_bus_prog_addr;
    wire [31:0] zlc_bus_prog_start_tick;
    wire [31:0] zlc_bus_prog_stop_tick;
    wire [9:0] zlc_bus_prog_start_value;
    wire [9:0] zlc_bus_prog_stop_value;
    wire [1:0] zlc_bus_prog_mode;
    wire [2:0] zlc_bus_prog_value_select;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_bus_prog_start_tick_coeffs;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_bus_prog_stop_tick_coeffs;
    wire [27:0] zlc_bus_counts;
    wire [CHANNEL_COUNT-1:0] out;
    wire [39:0] zlc_bus_out;
    wire zlc_running;
    wire zlc_done;

    assign led[0] = zlc_running;
    assign led[1] = zlc_done;

    assign cooling = out[0];
    assign cooling_pgc = out[1];
    assign repump = out[2];
    assign probe = out[3];
    assign pushout = out[4];
    assign state_pre = out[5];
    assign trig = out[6];
    assign coil = out[7];
    assign grey_cooling = out[8];
    assign trap = out[9];
    assign UV = out[10];
    assign emCCD = out[11];
    assign microwave = out[12];
    assign address = out[13];
    assign cooling_shutter = out[14];
    assign repump_shutter = out[15];
    assign probe_shutter = out[16];
    assign bias = out[17];
    assign da_dipole[0] = zlc_bus_out[0];
    assign da_dipole[1] = zlc_bus_out[1];
    assign da_dipole[2] = zlc_bus_out[2];
    assign da_dipole[3] = zlc_bus_out[3];
    assign da_dipole[4] = zlc_bus_out[4];
    assign da_dipole[5] = zlc_bus_out[5];
    assign da_dipole[6] = zlc_bus_out[6];
    assign da_dipole[7] = zlc_bus_out[7];
    assign da_dipole[8] = zlc_bus_out[8];
    assign da_dipole[9] = zlc_bus_out[9];
    assign da_clk0 = out[28];
    assign da_bias_y[0] = zlc_bus_out[10];
    assign da_bias_y[1] = zlc_bus_out[11];
    assign da_bias_y[2] = zlc_bus_out[12];
    assign da_bias_y[3] = zlc_bus_out[13];
    assign da_bias_y[4] = zlc_bus_out[14];
    assign da_bias_y[5] = zlc_bus_out[15];
    assign da_bias_y[6] = zlc_bus_out[16];
    assign da_bias_y[7] = zlc_bus_out[17];
    assign da_bias_y[8] = zlc_bus_out[18];
    assign da_bias_y[9] = zlc_bus_out[19];
    assign da_clk1 = out[39];
    assign da_bias_x[0] = zlc_bus_out[20];
    assign da_bias_x[1] = zlc_bus_out[21];
    assign da_bias_x[2] = zlc_bus_out[22];
    assign da_bias_x[3] = zlc_bus_out[23];
    assign da_bias_x[4] = zlc_bus_out[24];
    assign da_bias_x[5] = zlc_bus_out[25];
    assign da_bias_x[6] = zlc_bus_out[26];
    assign da_bias_x[7] = zlc_bus_out[27];
    assign da_bias_x[8] = zlc_bus_out[28];
    assign da_bias_x[9] = zlc_bus_out[29];
    assign da_clk2 = out[50];
    assign da_bias_z[0] = zlc_bus_out[30];
    assign da_bias_z[1] = zlc_bus_out[31];
    assign da_bias_z[2] = zlc_bus_out[32];
    assign da_bias_z[3] = zlc_bus_out[33];
    assign da_bias_z[4] = zlc_bus_out[34];
    assign da_bias_z[5] = zlc_bus_out[35];
    assign da_bias_z[6] = zlc_bus_out[36];
    assign da_bias_z[7] = zlc_bus_out[37];
    assign da_bias_z[8] = zlc_bus_out[38];
    assign da_bias_z[9] = zlc_bus_out[39];
    assign da_clk3 = out[61];

    assign GND1 = 1'b0;
    assign GND4 = 1'b0;
    assign GND5 = 1'b0;
    assign GND6 = 1'b0;
    assign GND7 = 1'b0;
    assign GND8 = 1'b0;
    assign GND9 = 1'b0;
    assign GND10 = 1'b0;
    assign GND11 = 1'b0;
    assign GND12 = 1'b0;
    assign GND13 = 1'b0;
    assign GND14 = 1'b0;
    assign GND15 = 1'b0;

    zlc_pulse_streamer #(
        .CHANNEL_COUNT(CHANNEL_COUNT),
        .EDGE_ADDR_WIDTH(10),
        .TICK_WIDTH(TICK_WIDTH),
        .SCAN_ADDR_WIDTH(10),
        .NUM_SLOTS(NUM_SLOTS),
        .COEFF_WIDTH(COEFF_WIDTH),
        .COEFF_FRAC_BITS(8),
        .BUS_COUNT(4),
        .BUS_INDEX_WIDTH(2),
        .BUS_WIDTH(10),
        .BUS_SEG_ADDR_WIDTH(6)
    ) zlc_streamer_i (
        .clk(clk),
        .reset(zlc_reset),
        .start(zlc_start),
        .prog_we(zlc_prog_we),
        .prog_addr(zlc_prog_addr),
        .prog_tick(zlc_prog_tick),
        .prog_tick_coeffs(zlc_prog_tick_coeffs),
        .prog_mask(zlc_prog_mask),
        .prog_count(zlc_prog_count),
        .repeat_forever(zlc_repeat_forever),
        .loop_start_addr(zlc_loop_start_addr),
        .loop_end_tick(zlc_loop_end_tick),
        .loop_end_coeffs(zlc_loop_end_coeffs),
        .loop_count(zlc_loop_count),
        .scan_enable(zlc_scan_enable),
        .scan_prog_we(zlc_scan_prog_we),
        .scan_prog_addr(zlc_scan_prog_addr),
        .scan_prog_values(zlc_scan_prog_values),
        .scan_count(zlc_scan_count),
        .bus_prog_we(zlc_bus_prog_we),
        .bus_prog_bus(zlc_bus_prog_bus),
        .bus_prog_addr(zlc_bus_prog_addr),
        .bus_prog_start_tick(zlc_bus_prog_start_tick),
        .bus_prog_stop_tick(zlc_bus_prog_stop_tick),
        .bus_prog_start_tick_coeffs(zlc_bus_prog_start_tick_coeffs),
        .bus_prog_stop_tick_coeffs(zlc_bus_prog_stop_tick_coeffs),
        .bus_prog_start_value(zlc_bus_prog_start_value),
        .bus_prog_stop_value(zlc_bus_prog_stop_value),
        .bus_prog_mode(zlc_bus_prog_mode),
        .bus_prog_value_select(zlc_bus_prog_value_select),
        .bus_counts(zlc_bus_counts),
        .out(out),
        .bus_out(zlc_bus_out),
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
        .probe_out10(zlc_loop_count),
        .probe_out11(zlc_prog_tick_coeffs),
        .probe_out12(zlc_scan_enable),
        .probe_out13(zlc_scan_prog_we),
        .probe_out14(zlc_scan_prog_addr),
        .probe_out15(zlc_scan_prog_values),
        .probe_out16(zlc_scan_count),
        .probe_out17(zlc_loop_end_coeffs),
        .probe_out18(zlc_bus_prog_we),
        .probe_out19(zlc_bus_prog_bus),
        .probe_out20(zlc_bus_prog_addr),
        .probe_out21(zlc_bus_prog_start_tick),
        .probe_out22(zlc_bus_prog_stop_tick),
        .probe_out23(zlc_bus_prog_start_value),
        .probe_out24(zlc_bus_prog_stop_value),
        .probe_out25(zlc_bus_prog_mode),
        .probe_out26(zlc_bus_counts),
        .probe_out27(zlc_bus_prog_value_select),
        .probe_out28(zlc_bus_prog_start_tick_coeffs),
        .probe_out29(zlc_bus_prog_stop_tick_coeffs)
    );
endmodule
