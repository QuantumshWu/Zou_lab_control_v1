`timescale 1ns / 1ps
// =============================================================================
// zlc_pulse_streamer_loader_top -- board top for the affine edge-table pulse
// streamer with a JTAG-to-AXI control path.
//
// This restores the VALIDATED, seamless edge-table engine (zlc_pulse_streamer)
// -- single global edge pointer, single-cycle hardware loop (repeat NOT
// unrolled), single-cycle scan-point advance, N-slot affine scan of
// delay/duration, and DAC-value scan via bus value_select -- and feeds it from
// BRAM instead of VIO.  An on-chip loader (zlc_axi_program_loader) copies the
// program image out of the AXI BRAM into the engine's prog_* ports, then pulses
// start.  Because the engine is unchanged, repeat-to-repeat AND
// scan-to-scan are gapless and tick-exact (the per-channel run-length engine,
// which could not be seamless on one BRAM port, is discarded).
//
// Pin contract: identical to the board's address_switch XDC -- 62 controllable
// outputs + tied-low GND helper pins + two status LEDs + 4x10-bit DAC buses.
//
// Control path:
//   jtag_axi_0 (AXI master, hw_axi over JTAG) -> axi_bram_ctrl_0 (AXI4-Lite) ->
//   blk_mem_gen_0 (true dual-port BRAM).  Port A = AXI image upload + command/
//   status mailbox; port B = loader (reads image, writes STATUS).  Image layout
//   = Zou_lab_control/neutral_atom/devices/edgetable_image.py; host driver =
//   devices/axi_session.py.
// =============================================================================

module zlc_pulse_streamer_loader_top(
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
    localparam integer EDGE_ADDR_WIDTH = 10;
    localparam integer SCAN_ADDR_WIDTH = 10;
    localparam integer TICK_WIDTH = 32;
    localparam integer NUM_SLOTS = 4;
    localparam integer COEFF_WIDTH = 16;
    localparam integer COEFF_FRAC_BITS = 8;
    localparam integer BUS_COUNT = 4;
    localparam integer BUS_INDEX_WIDTH = 2;
    localparam integer BUS_WIDTH = 10;
    localparam integer BUS_SEG_ADDR_WIDTH = 6;
    localparam integer BUS_SEL_WIDTH = 3;

    // Engine outputs -> board pins.
    wire [CHANNEL_COUNT-1:0] out;
    wire [BUS_COUNT*BUS_WIDTH-1:0] zlc_bus_out;
    wire zlc_running;
    wire zlc_done;

    // ---- Control path: JTAG-to-AXI master -> AXI4-Lite -> AXI BRAM ctrl ------
    wire        axi_clk = clk;
    wire        axi_resetn = 1'b1;

    wire [31:0] m_axi_awaddr;  wire [2:0] m_axi_awprot;  wire m_axi_awvalid; wire m_axi_awready;
    wire [31:0] m_axi_wdata;   wire [3:0] m_axi_wstrb;   wire m_axi_wvalid;  wire m_axi_wready;
    wire [1:0]  m_axi_bresp;   wire m_axi_bvalid;        wire m_axi_bready;
    wire [31:0] m_axi_araddr;  wire [2:0] m_axi_arprot;  wire m_axi_arvalid; wire m_axi_arready;
    wire [31:0] m_axi_rdata;   wire [1:0] m_axi_rresp;   wire m_axi_rvalid;  wire m_axi_rready;

    // axi_bram_ctrl BRAM port A (byte-addressed; 17-bit -> 32768x32b memory).
    wire        bram_clka;  wire bram_rsta;  wire bram_ena;
    wire [3:0]  bram_wea;   wire [16:0] bram_addra; wire [31:0] bram_dina; wire [31:0] bram_douta;

    // Loader <-> BRAM port B (15-bit WORD address into the 32768-word BRAM).
    wire        ldr_en;
    wire        ldr_we;
    wire [14:0] ldr_addr;   // word address
    wire [31:0] ldr_wdata;
    wire [31:0] ldr_rdata;

    // Loader -> engine program-write port.
    wire        prog_we;
    wire [EDGE_ADDR_WIDTH-1:0] prog_addr;
    wire [TICK_WIDTH-1:0] prog_tick;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] prog_tick_coeffs;
    wire [CHANNEL_COUNT-1:0] prog_mask;
    wire [EDGE_ADDR_WIDTH:0] prog_count;
    wire        repeat_forever;
    wire [EDGE_ADDR_WIDTH-1:0] loop_start_addr;
    wire [TICK_WIDTH-1:0] loop_end_tick;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] loop_end_coeffs;
    wire [31:0] loop_count;
    wire        scan_enable;
    wire        scan_prog_we;
    wire [SCAN_ADDR_WIDTH-1:0] scan_prog_addr;
    wire [NUM_SLOTS*TICK_WIDTH-1:0] scan_prog_values;
    wire [SCAN_ADDR_WIDTH:0] scan_count;
    wire        bus_prog_we;
    wire [BUS_INDEX_WIDTH-1:0] bus_prog_bus;
    wire [BUS_SEG_ADDR_WIDTH-1:0] bus_prog_addr;
    wire [TICK_WIDTH-1:0] bus_prog_start_tick;
    wire [TICK_WIDTH-1:0] bus_prog_stop_tick;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_start_tick_coeffs;
    wire [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_stop_tick_coeffs;
    wire [BUS_WIDTH-1:0] bus_prog_start_value;
    wire [BUS_WIDTH-1:0] bus_prog_stop_value;
    wire [1:0]  bus_prog_mode;
    wire [BUS_SEL_WIDTH-1:0] bus_prog_value_select;
    wire [BUS_COUNT*(BUS_SEG_ADDR_WIDTH+1)-1:0] bus_counts;

    wire        eng_reset;
    wire        eng_start;
    wire [4:0]  loader_state;

    assign led[0] = zlc_running;
    // DIAGNOSTIC (bring-up): led[1] = "any channel output is high" so the board
    // shows whether the engine actually drives outputs.
    assign led[1] = |out;

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

    // ---- JTAG-to-AXI master: driven from Vivado hw_axi over the JTAG cable ----
    jtag_axi_0 zlc_jtag_axi_i (
        .aclk(axi_clk),
        .aresetn(axi_resetn),
        .m_axi_awaddr(m_axi_awaddr),   .m_axi_awprot(m_axi_awprot),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),     .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wvalid(m_axi_wvalid),   .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),     .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_araddr(m_axi_araddr),   .m_axi_arprot(m_axi_arprot),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),     .m_axi_rresp(m_axi_rresp),
        .m_axi_rvalid(m_axi_rvalid),   .m_axi_rready(m_axi_rready)
    );

    // ---- AXI4-Lite -> BRAM controller ----------------------------------------
    axi_bram_ctrl_0 zlc_bram_ctrl_i (
        .s_axi_aclk(axi_clk), .s_axi_aresetn(axi_resetn),
        .s_axi_awaddr(m_axi_awaddr), .s_axi_awprot(m_axi_awprot),
        .s_axi_awvalid(m_axi_awvalid), .s_axi_awready(m_axi_awready),
        .s_axi_wdata(m_axi_wdata), .s_axi_wstrb(m_axi_wstrb),
        .s_axi_wvalid(m_axi_wvalid), .s_axi_wready(m_axi_wready),
        .s_axi_bresp(m_axi_bresp), .s_axi_bvalid(m_axi_bvalid), .s_axi_bready(m_axi_bready),
        .s_axi_araddr(m_axi_araddr), .s_axi_arprot(m_axi_arprot),
        .s_axi_arvalid(m_axi_arvalid), .s_axi_arready(m_axi_arready),
        .s_axi_rdata(m_axi_rdata), .s_axi_rresp(m_axi_rresp),
        .s_axi_rvalid(m_axi_rvalid), .s_axi_rready(m_axi_rready),
        .bram_rst_a(bram_rsta), .bram_clk_a(bram_clka), .bram_en_a(bram_ena),
        .bram_we_a(bram_wea), .bram_addr_a(bram_addra),
        .bram_wrdata_a(bram_dina), .bram_rddata_a(bram_douta)
    );

    // ---- Program BRAM (true dual-port): port A = AXI, port B = loader ---------
    blk_mem_gen_0 zlc_prog_bram_i (
        .clka(bram_clka), .ena(bram_ena), .wea(bram_wea),
        .addra(bram_addra[16:2]), .dina(bram_dina), .douta(bram_douta),
        .clkb(axi_clk), .enb(ldr_en), .web({4{ldr_we}}),
        .addrb(ldr_addr), .dinb(ldr_wdata), .doutb(ldr_rdata)
    );

    // ---- On-chip loader: BRAM image -> engine prog_* ports --------------------
    zlc_axi_program_loader #(
        .CHANNEL_COUNT(CHANNEL_COUNT),
        .EDGE_ADDR_WIDTH(EDGE_ADDR_WIDTH),
        .SCAN_ADDR_WIDTH(SCAN_ADDR_WIDTH),
        .TICK_WIDTH(TICK_WIDTH),
        .NUM_SLOTS(NUM_SLOTS),
        .COEFF_WIDTH(COEFF_WIDTH),
        .BUS_COUNT(BUS_COUNT),
        .BUS_INDEX_WIDTH(BUS_INDEX_WIDTH),
        .BUS_SEG_ADDR_WIDTH(BUS_SEG_ADDR_WIDTH),
        .BUS_WIDTH(BUS_WIDTH),
        .BUS_SEL_WIDTH(BUS_SEL_WIDTH),
        .MEM_ADDR_WIDTH(15)
    ) zlc_loader_i (
        .clk(axi_clk),
        .ext_reset(1'b0),
        .mem_addr(ldr_addr), .mem_en(ldr_en), .mem_we(ldr_we),
        .mem_wdata(ldr_wdata), .mem_rdata(ldr_rdata),
        .eng_reset(eng_reset), .eng_start(eng_start),
        .eng_running(zlc_running), .eng_done(zlc_done),
        .prog_we(prog_we), .prog_addr(prog_addr), .prog_tick(prog_tick),
        .prog_tick_coeffs(prog_tick_coeffs), .prog_mask(prog_mask),
        .prog_count(prog_count), .repeat_forever(repeat_forever),
        .loop_start_addr(loop_start_addr), .loop_end_tick(loop_end_tick),
        .loop_end_coeffs(loop_end_coeffs), .loop_count(loop_count),
        .scan_enable(scan_enable), .scan_prog_we(scan_prog_we),
        .scan_prog_addr(scan_prog_addr), .scan_prog_values(scan_prog_values),
        .scan_count(scan_count),
        .bus_prog_we(bus_prog_we), .bus_prog_bus(bus_prog_bus),
        .bus_prog_addr(bus_prog_addr), .bus_prog_start_tick(bus_prog_start_tick),
        .bus_prog_stop_tick(bus_prog_stop_tick),
        .bus_prog_start_tick_coeffs(bus_prog_start_tick_coeffs),
        .bus_prog_stop_tick_coeffs(bus_prog_stop_tick_coeffs),
        .bus_prog_start_value(bus_prog_start_value),
        .bus_prog_stop_value(bus_prog_stop_value),
        .bus_prog_mode(bus_prog_mode),
        .bus_prog_value_select(bus_prog_value_select),
        .bus_counts(bus_counts),
        .loader_state(loader_state)
    );

    // ---- Validated edge-table engine (seamless loop + affine scan + DAC) ------
    zlc_pulse_streamer #(
        .CHANNEL_COUNT(CHANNEL_COUNT),
        .EDGE_ADDR_WIDTH(EDGE_ADDR_WIDTH),
        .TICK_WIDTH(TICK_WIDTH),
        .SCAN_ADDR_WIDTH(SCAN_ADDR_WIDTH),
        .NUM_SLOTS(NUM_SLOTS),
        .COEFF_WIDTH(COEFF_WIDTH),
        .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
        .BUS_COUNT(BUS_COUNT),
        .BUS_INDEX_WIDTH(BUS_INDEX_WIDTH),
        .BUS_WIDTH(BUS_WIDTH),
        .BUS_SEG_ADDR_WIDTH(BUS_SEG_ADDR_WIDTH),
        .BUS_SEL_WIDTH(BUS_SEL_WIDTH)
    ) zlc_engine_i (
        .clk(axi_clk),
        .reset(eng_reset),
        .start(eng_start),
        .prog_we(prog_we), .prog_addr(prog_addr), .prog_tick(prog_tick),
        .prog_tick_coeffs(prog_tick_coeffs), .prog_mask(prog_mask),
        .prog_count(prog_count), .repeat_forever(repeat_forever),
        .loop_start_addr(loop_start_addr), .loop_end_tick(loop_end_tick),
        .loop_end_coeffs(loop_end_coeffs), .loop_count(loop_count),
        .scan_enable(scan_enable), .scan_prog_we(scan_prog_we),
        .scan_prog_addr(scan_prog_addr), .scan_prog_values(scan_prog_values),
        .scan_count(scan_count),
        .bus_prog_we(bus_prog_we), .bus_prog_bus(bus_prog_bus),
        .bus_prog_addr(bus_prog_addr), .bus_prog_start_tick(bus_prog_start_tick),
        .bus_prog_stop_tick(bus_prog_stop_tick),
        .bus_prog_start_tick_coeffs(bus_prog_start_tick_coeffs),
        .bus_prog_stop_tick_coeffs(bus_prog_stop_tick_coeffs),
        .bus_prog_start_value(bus_prog_start_value),
        .bus_prog_stop_value(bus_prog_stop_value),
        .bus_prog_mode(bus_prog_mode),
        .bus_prog_value_select(bus_prog_value_select),
        .bus_counts(bus_counts),
        .out(out),
        .bus_out(zlc_bus_out),
        .running(zlc_running),
        .done(zlc_done)
    );
endmodule
