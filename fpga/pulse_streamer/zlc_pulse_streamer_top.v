`timescale 1ns / 1ps
// =============================================================================
// zlc_pulse_streamer_top -- FINAL board top for the affine edge-table streamer.
//
// One clean design (no variants).  JTAG-to-AXI control; edge + scan tables in
// BLOCK RAM, bus tables in LUTRAM inside the engine.  Reaches 4096 edges + 4096
// resident scan points + UNBOUNDED streaming scan points at 78% of the 35T
// RAMB36 (host.image.solve_capacity, <=90% target).
//
// Control path (all behind ONE proven axi_bram_ctrl, so AXI handshakes are the
// vendor IP -- only a SIMPLE combinational write decoder is custom):
//   jtag_axi_0 -> axi_bram_ctrl_0 -> {bram_addr_a, bram_we_a, ...} -> decoder,
//   by word-address region (bases == host.image.region_bases, single source):
//     R_CTRL  regfile: scalars + COMMAND/STATUS mailbox + bus_counts + BANK_SIZE
//             + SLOT_COUNT + CURSOR(read-back) + BANK_READY(host-written)
//     R_TICK  edge tick BRAM   (32b/edge)   ]
//     R_COEFF edge coeff BRAM  (64b/edge)    } 3 PARALLEL edge BRAMs, read in
//     R_MASK  edge mask BRAM   (62b/edge)   ]  lockstep on edge_raddr -> whole
//                                              edge per access, no width padding
//     R_SCAN  scan BRAM (128b slot vector/point), 2*BANK_SIZE deep (ping-pong)
//     R_BUS   bus-image BRAM; the mini-loader copies it into the engine bus LUTRAM
//
// STREAMING: the engine plays scan point 0..N-1, addressing bank (idx/BANK_SIZE)
// %2.  It exposes scan_cursor (points consumed) -> R_CTRL[CURSOR]; the host polls
// it, rewrites the bank it has left in R_SCAN, and sets the matching bit of
// R_CTRL[BANK_READY].  A not-ready bank STALLS the engine (STATUS underflow,
// never a wrong point).  This makes the scan-point count UNBOUNDED.
//
// 1-TICK: the build tcl forces the 3 edge BRAMs to READ_LATENCY_B = 2 so the
// engine's RD_LAT=2 prefetch pipeline is deterministic and back-to-back 20 ns
// edges play one per clock (see zlc_edge_streamer.v + engine_model proofs).
//
// Geometry localparams are computed by the SAME formulas as host.image.region_bases
// (locked by test_final_top_regions_match_image); the create-project tcl derives
// the BRAM IP geometry from host.image too.
//
// *** Structurally complete + contract-tested; the engine + control FSM are
// proven by the Python cycle models, but the multi-BRAM AXI integration needs
// on-board bring-up (no Verilog simulator in this repo). ***
// =============================================================================

module zlc_pulse_streamer_top #(
    parameter integer CHANNEL_COUNT = 62,
    parameter integer EDGE_ADDR_WIDTH = 12,     // 4096 edges
    parameter integer BANK_SIZE = 2048,         // power of two; scan ping-pong bank
    parameter integer SCAN_ADDR_WIDTH = 12,     // addresses 2*BANK_SIZE points
    parameter integer SCAN_COUNT_WIDTH = 32,    // total scan points N (unbounded)
    parameter integer TICK_WIDTH = 32,
    parameter integer NUM_SLOTS = 4,
    parameter integer COEFF_WIDTH = 16,
    parameter integer COEFF_FRAC_BITS = 8,
    parameter integer BUS_COUNT = 4,
    parameter integer BUS_INDEX_WIDTH = 2,
    parameter integer BUS_WIDTH = 10,
    parameter integer BUS_SEG_ADDR_WIDTH = 6,
    parameter integer BUS_SEL_WIDTH = 3
)(
    input  wire clk,
    output wire [1:0] led,
    output wire cooling, output wire cooling_pgc, output wire repump, output wire probe,
    output wire pushout, output wire state_pre, output wire trig, output wire coil,
    output wire grey_cooling, output wire trap, output wire UV, output wire emCCD,
    output wire microwave, output wire address,
    output wire GND1, output wire GND4, output wire GND5, output wire GND6, output wire GND7,
    output wire GND8, output wire GND9, output wire GND10, output wire GND11,
    output wire cooling_shutter, output wire GND12, output wire repump_shutter, output wire GND13,
    output wire probe_shutter, output wire GND14, output wire bias, output wire GND15,
    output wire [9:0] da_dipole, output wire da_clk0,
    output wire [9:0] da_bias_y, output wire da_clk1,
    output wire [9:0] da_bias_x, output wire da_clk2,
    output wire [9:0] da_bias_z, output wire da_clk3
);

    localparam integer COEFF_BITS = NUM_SLOTS * COEFF_WIDTH;     // 64
    localparam integer SLOT_BITS = NUM_SLOTS * TICK_WIDTH;       // 128
    localparam integer COEFF_PORTB_BITS = 64;                    // 4x16 exact, 2x32
    localparam integer MASK_PORTB_BITS = 64;                     // 62 padded to 64, 2x32
    localparam integer SCAN_PORTB_BITS = SLOT_BITS;             // 128 = 4x32
    localparam integer COEFF_WORDS = COEFF_PORTB_BITS / 32;      // 2
    localparam integer MASK_WORDS = MASK_PORTB_BITS / 32;        // 2
    localparam integer SCAN_WORDS = SCAN_PORTB_BITS / 32;        // 4
    localparam integer MAX_EDGES = (1 << EDGE_ADDR_WIDTH);
    localparam integer SCAN_DEPTH = 2 * BANK_SIZE;
    localparam integer MAX_BUS_SEGMENTS = (1 << BUS_SEG_ADDR_WIDTH);
    localparam integer BUS_ROWS = BUS_COUNT * MAX_BUS_SEGMENTS;
    localparam integer BUS_WORDS = 2 + 2 * ((COEFF_BITS + 31) / 32) + 1;   // 7

    // --- word-address region bases (== host.image.region_bases) ---------------
    localparam integer R_CTRL_BASE = 0;
    localparam integer R_CTRL_WORDS = 64;
    localparam integer R_TICK_BASE  = R_CTRL_BASE + R_CTRL_WORDS;
    localparam integer R_COEFF_BASE = R_TICK_BASE  + MAX_EDGES * 1;
    localparam integer R_MASK_BASE  = R_COEFF_BASE + MAX_EDGES * COEFF_WORDS;
    localparam integer R_SCAN_BASE  = R_MASK_BASE  + MAX_EDGES * MASK_WORDS;
    localparam integer R_BUS_BASE   = R_SCAN_BASE  + SCAN_DEPTH * SCAN_WORDS;
    localparam integer R_TOTAL_WORDS = R_BUS_BASE + BUS_ROWS * BUS_WORDS;

    // CTRL regfile word offsets (== host.image.CtrlWords).
    localparam integer C_MAGIC = 0;
    localparam integer C_COMMAND = 1;   // bit0 LOAD bit1 FIRE bit2 RESET bit3 SAFE
    localparam integer C_STATUS = 2;    // bit0 LOADED bit1 RUNNING bit2 DONE bit3 UNDERFLOW
    localparam integer C_PROG_COUNT = 3;
    localparam integer C_SCAN_COUNT = 4;
    localparam integer C_SCAN_ENABLE = 5;
    localparam integer C_REPEAT_FOREVER = 6;
    localparam integer C_LOOP_START = 7;
    localparam integer C_LOOP_COUNT = 8;
    localparam integer C_LOOP_END_TICK = 9;
    localparam integer C_LOOP_END_LO = 10;
    localparam integer C_LOOP_END_HI = 11;
    localparam integer C_BUS_COUNTS = 12;
    localparam integer C_BANK_SIZE = 13;
    localparam integer C_SLOT_COUNT = 14;
    localparam integer C_CURSOR = 15;       // engine -> host (points consumed)
    localparam integer C_BANK_READY = 16;   // host -> engine (bit b: bank b loaded)
    localparam integer C_BANK0_CHUNK = 17;  // host -> engine: sweep chunk resident in bank 0
    localparam integer C_BANK1_CHUNK = 18;  // host -> engine: sweep chunk resident in bank 1

    // engine outputs
    wire [CHANNEL_COUNT-1:0] out;
    wire [BUS_COUNT*BUS_WIDTH-1:0] zlc_bus_out;
    wire zlc_running, zlc_done, zlc_underflow;
    wire [SCAN_COUNT_WIDTH-1:0] zlc_cursor;

    // --- JTAG-to-AXI master -> AXI4-Lite -> AXI BRAM controller ---------------
    wire axi_clk = clk;
    wire axi_resetn = 1'b1;
    wire [31:0] m_axi_awaddr;  wire [2:0] m_axi_awprot;  wire m_axi_awvalid; wire m_axi_awready;
    wire [31:0] m_axi_wdata;   wire [3:0] m_axi_wstrb;   wire m_axi_wvalid;  wire m_axi_wready;
    wire [1:0]  m_axi_bresp;   wire m_axi_bvalid;        wire m_axi_bready;
    wire [31:0] m_axi_araddr;  wire [2:0] m_axi_arprot;  wire m_axi_arvalid; wire m_axi_arready;
    wire [31:0] m_axi_rdata;   wire [1:0] m_axi_rresp;   wire m_axi_rvalid;  wire m_axi_rready;

    wire        bram_clka, bram_rsta, bram_ena;
    wire [3:0]  bram_wea;
    wire [31:0] bram_addra;          // byte address from axi_bram_ctrl
    wire [31:0] bram_dina;
    reg  [31:0] bram_douta;          // read mux back to AXI

    wire [29:0] word_addr = bram_addra[31:2];
    wire        wr = |bram_wea;

    // region selects (combinational decode of the word address)
    wire sel_ctrl  = (word_addr >= R_CTRL_BASE)  && (word_addr < R_TICK_BASE);
    wire sel_tick  = (word_addr >= R_TICK_BASE)  && (word_addr < R_COEFF_BASE);
    wire sel_coeff = (word_addr >= R_COEFF_BASE) && (word_addr < R_MASK_BASE);
    wire sel_mask  = (word_addr >= R_MASK_BASE)  && (word_addr < R_SCAN_BASE);
    wire sel_scan  = (word_addr >= R_SCAN_BASE)  && (word_addr < R_BUS_BASE);
    wire sel_bus   = (word_addr >= R_BUS_BASE)   && (word_addr < R_TOTAL_WORDS);
    wire [29:0] tick_word_off  = word_addr - R_TICK_BASE[29:0];
    wire [29:0] coeff_word_off = word_addr - R_COEFF_BASE[29:0];
    wire [29:0] mask_word_off  = word_addr - R_MASK_BASE[29:0];
    wire [29:0] scan_word_off  = word_addr - R_SCAN_BASE[29:0];
    wire [29:0] bus_word_off   = word_addr - R_BUS_BASE[29:0];

    // --- CTRL regfile ---------------------------------------------------------
    reg [31:0] ctrl_reg [0:R_CTRL_WORDS-1];
    integer ci;
    initial begin for (ci = 0; ci < R_CTRL_WORDS; ci = ci + 1) ctrl_reg[ci] = 32'b0; end

    // loader/engine-driven write-backs (separate from AXI host writes)
    reg ldr_status_we;
    reg [31:0] ldr_status_val;
    reg ldr_cmd_clear;          // loader acks a command by clearing C_COMMAND

    always @(posedge clk) begin
        if (bram_ena && wr && sel_ctrl) ctrl_reg[word_addr[5:0]] <= bram_dina;
        if (ldr_status_we) ctrl_reg[C_STATUS] <= ldr_status_val;
        if (ldr_cmd_clear) ctrl_reg[C_COMMAND] <= 32'b0;
        ctrl_reg[C_CURSOR] <= zlc_cursor;        // engine cursor visible to host
    end

    // --- read mux back to AXI -------------------------------------------------
    always @(*) begin
        if (sel_ctrl) bram_douta = ctrl_reg[word_addr[5:0]];
        else bram_douta = 32'b0;
    end

    // --- 3 PARALLEL edge BRAMs (tick 32b, coeff 64b, mask 62/64b) -------------
    // Forced READ_LATENCY_B = 2 by the build tcl; engine RD_LAT must match.
    wire [TICK_WIDTH-1:0]      edge_tick_rdata;
    wire [COEFF_PORTB_BITS-1:0] edge_coeff_rdata_w;
    wire [MASK_PORTB_BITS-1:0]  edge_mask_rdata_w;
    wire [EDGE_ADDR_WIDTH-1:0] edge_raddr;

    blk_mem_gen_edge_tick zlc_edge_tick_i (
        .clka(axi_clk), .ena(bram_ena && sel_tick), .wea(bram_wea),
        .addra(tick_word_off[EDGE_ADDR_WIDTH-1:0]), .dina(bram_dina), .douta(),
        .clkb(axi_clk), .enb(1'b1), .web(4'b0),
        .addrb(edge_raddr), .dinb(32'b0), .doutb(edge_tick_rdata)
    );
    blk_mem_gen_edge_coeff zlc_edge_coeff_i (
        .clka(axi_clk), .ena(bram_ena && sel_coeff), .wea(bram_wea),
        .addra(coeff_word_off[($clog2(MAX_EDGES*COEFF_WORDS))-1:0]), .dina(bram_dina), .douta(),
        .clkb(axi_clk), .enb(1'b1), .web({(COEFF_PORTB_BITS/8){1'b0}}),
        .addrb(edge_raddr), .dinb({COEFF_PORTB_BITS{1'b0}}), .doutb(edge_coeff_rdata_w)
    );
    blk_mem_gen_edge_mask zlc_edge_mask_i (
        .clka(axi_clk), .ena(bram_ena && sel_mask), .wea(bram_wea),
        .addra(mask_word_off[($clog2(MAX_EDGES*MASK_WORDS))-1:0]), .dina(bram_dina), .douta(),
        .clkb(axi_clk), .enb(1'b1), .web({(MASK_PORTB_BITS/8){1'b0}}),
        .addrb(edge_raddr), .dinb({MASK_PORTB_BITS{1'b0}}), .doutb(edge_mask_rdata_w)
    );

    // --- SCAN BRAM (port A 32b write, port B 128b read; 2*BANK_SIZE deep) ------
    wire [SCAN_PORTB_BITS-1:0] scan_rdata_w;
    wire [SCAN_ADDR_WIDTH-1:0] scan_raddr;
    blk_mem_gen_scan zlc_scan_bram_i (
        .clka(axi_clk), .ena(bram_ena && sel_scan), .wea(bram_wea),
        .addra(scan_word_off[($clog2(SCAN_DEPTH*SCAN_WORDS))-1:0]), .dina(bram_dina), .douta(),
        .clkb(axi_clk), .enb(1'b1), .web({(SCAN_PORTB_BITS/8){1'b0}}),
        .addrb(scan_raddr), .dinb({SCAN_PORTB_BITS{1'b0}}), .doutb(scan_rdata_w)
    );

    // --- BUS image BRAM (32b TDP; the mini-loader reads it into bus LUTRAM) ----
    wire [31:0] bus_img_doutb;
    reg  [($clog2(BUS_ROWS*BUS_WORDS))-1:0] bus_img_raddr;
    blk_mem_gen_busimg zlc_bus_img_i (
        .clka(axi_clk), .ena(bram_ena && sel_bus), .wea(bram_wea),
        .addra(bus_word_off[($clog2(BUS_ROWS*BUS_WORDS))-1:0]), .dina(bram_dina), .douta(),
        .clkb(axi_clk), .enb(1'b1), .web(4'b0),
        .addrb(bus_img_raddr), .dinb(32'b0), .doutb(bus_img_doutb)
    );

    // --- control / bus mini-loader FSM ----------------------------------------
    // On LOAD: hold engine reset, copy the bus image (R_BUS) into the engine bus
    // LUTRAM via bus_prog_*, set STATUS.LOADED.  On FIRE: release reset + pulse
    // start.  Edge/scan are NOT copied (the engine reads those BRAMs directly).
    // Bus rows are 7 words = [start_tick, stop_tick, sc_lo, sc_hi, ec_lo, ec_hi,
    // flags] (host.image bus layout).  Rising-edge-detected commands.
    localparam CMD_LOAD = 4'b0001, CMD_FIRE = 4'b0010, CMD_RESET = 4'b0100, CMD_SAFE = 4'b1000;
    // STATUS bit map MUST match host.image: LOADED=1 RUNNING=2 DONE=4 ERROR=8(host-only,
    // never set here) UNDERFLOW=16.  Underflow is bit4 (NOT bit3) so a transient
    // streaming STALL is never confused with the host's fatal ERROR bit.
    localparam [4:0] ST_LOADED = 5'd1, ST_RUNNING = 5'd2, ST_DONE = 5'd4, ST_UNDERFLOW = 5'd16;
    localparam integer CNT_W = BUS_SEG_ADDR_WIDTH + 1;

    reg eng_reset = 1'b1, eng_start = 1'b0;
    reg bus_prog_we = 1'b0;
    reg [BUS_INDEX_WIDTH-1:0] bus_prog_bus = {BUS_INDEX_WIDTH{1'b0}};
    reg [BUS_SEG_ADDR_WIDTH-1:0] bus_prog_addr = {BUS_SEG_ADDR_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] bus_prog_start_tick = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] bus_prog_stop_tick = {TICK_WIDTH{1'b0}};
    reg [COEFF_BITS-1:0] bus_prog_start_tick_coeffs = {COEFF_BITS{1'b0}};
    reg [COEFF_BITS-1:0] bus_prog_stop_tick_coeffs = {COEFF_BITS{1'b0}};
    reg [BUS_WIDTH-1:0] bus_prog_start_value = {BUS_WIDTH{1'b0}};
    reg [BUS_WIDTH-1:0] bus_prog_stop_value = {BUS_WIDTH{1'b0}};
    reg [1:0] bus_prog_mode = 2'b0;
    reg [BUS_SEL_WIDTH-1:0] bus_prog_value_select = {BUS_SEL_WIDTH{1'b0}};
    reg [BUS_SEL_WIDTH-1:0] bus_prog_stop_value_select = {BUS_SEL_WIDTH{1'b0}};

    localparam [3:0] L_IDLE=0, L_RD=1, L_CAP=2, L_EMIT=3, L_NEXT=4, L_FIRE=5, L_RUN=6;
    reg [3:0] lstate = L_IDLE;
    reg [2:0] wi;                       // word index within a bus row
    reg [31:0] cap [0:6];
    reg [BUS_INDEX_WIDTH:0] bcur;       // current bus
    reg [BUS_SEG_ADDR_WIDTH:0] baddr;   // segment within bus
    reg [BUS_SEG_ADDR_WIDTH:0] bcnt;    // count for current bus
    reg [1:0] settle;
    reg [3:0] cmd_seen;
    integer ic;
    initial begin for (ic=0; ic<7; ic=ic+1) cap[ic]=32'b0; wi=0; bcur=0; baddr=0; bcnt=0; settle=0; cmd_seen=0; bus_img_raddr=0; end

    wire [3:0] cmd_now = ctrl_reg[C_COMMAND][3:0];
    wire [3:0] cmd_edge = cmd_now & ~cmd_seen;

    function [CNT_W-1:0] bus_count_of; input integer b; begin
        bus_count_of = ctrl_reg[C_BUS_COUNTS][b*CNT_W +: CNT_W]; end endfunction
    function [($clog2(BUS_ROWS*BUS_WORDS))-1:0] R_relbus;
        input integer b; input integer a;
        begin R_relbus = (b * MAX_BUS_SEGMENTS + a) * BUS_WORDS; end
    endfunction

    always @(posedge clk) begin
        ldr_status_we <= 1'b0;
        ldr_cmd_clear <= 1'b0;
        eng_start <= 1'b0;
        case (lstate)
        L_IDLE: begin
            cmd_seen <= cmd_now;
            if (cmd_edge & CMD_RESET) begin eng_reset <= 1'b1; ldr_status_we <= 1'b1; ldr_status_val <= 32'b0; end
            else if (cmd_edge & CMD_SAFE) begin eng_reset <= 1'b1; ldr_status_we <= 1'b1; ldr_status_val <= 32'b0; end
            else if (cmd_edge & CMD_LOAD) begin
                eng_reset <= 1'b1; bcur <= 0; baddr <= 0; bcnt <= bus_count_of(0); wi <= 0; lstate <= L_NEXT;
            end else if ((cmd_edge & CMD_FIRE) && (ctrl_reg[C_STATUS][0])) begin
                lstate <= L_FIRE;
            end
        end
        L_NEXT: begin
            wi <= 0;
            if (baddr >= bcnt) begin
                if (bcur == BUS_COUNT-1) begin
                    ldr_status_we <= 1'b1; ldr_status_val <= {27'b0, ST_LOADED}; lstate <= L_IDLE;
                end else begin
                    bcur <= bcur + 1'b1; baddr <= 0; bcnt <= bus_count_of(bcur + 1'b1); lstate <= L_NEXT;
                end
            end else begin
                bus_img_raddr <= R_relbus(bcur, baddr);
                settle <= 2'd2; lstate <= L_RD;
            end
        end
        L_RD: begin
            bus_img_raddr <= R_relbus(bcur, baddr) + wi;
            settle <= 2'd2; lstate <= L_CAP;
        end
        L_CAP: begin
            if (settle == 0) begin
                cap[wi] <= bus_img_doutb;
                if (wi == BUS_WORDS-1) lstate <= L_EMIT;
                else begin wi <= wi + 1'b1; lstate <= L_RD; end
            end else settle <= settle - 1'b1;
        end
        L_EMIT: begin
            bus_prog_bus <= bcur[BUS_INDEX_WIDTH-1:0];
            bus_prog_addr <= baddr[BUS_SEG_ADDR_WIDTH-1:0];
            bus_prog_start_tick <= cap[0]; bus_prog_stop_tick <= cap[1];
            bus_prog_start_tick_coeffs <= {cap[3][COEFF_BITS-33:0], cap[2]};
            bus_prog_stop_tick_coeffs <= {cap[5][COEFF_BITS-33:0], cap[4]};
            bus_prog_start_value <= cap[6][BUS_WIDTH-1:0];
            bus_prog_stop_value <= cap[6][2*BUS_WIDTH-1:BUS_WIDTH];
            bus_prog_mode <= cap[6][2*BUS_WIDTH+1:2*BUS_WIDTH];
            bus_prog_value_select <= cap[6][2*BUS_WIDTH+2+BUS_SEL_WIDTH-1:2*BUS_WIDTH+2];
            bus_prog_stop_value_select <= cap[6][2*BUS_WIDTH+2+2*BUS_SEL_WIDTH-1:2*BUS_WIDTH+2+BUS_SEL_WIDTH];
            bus_prog_we <= ~bus_prog_we;          // toggle commits a segment write
            baddr <= baddr + 1'b1;
            settle <= 2'd2; lstate <= L_RUN;
        end
        L_RUN: begin
            if (settle == 0) lstate <= L_NEXT; else settle <= settle - 1'b1;
        end
        L_FIRE: begin
            eng_reset <= 1'b0;
            eng_start <= 1'b1;
            ldr_status_we <= 1'b1; ldr_status_val <= {27'b0, ST_RUNNING};
            cmd_seen <= cmd_now;
            lstate <= L_IDLE;
        end
        default: lstate <= L_IDLE;
        endcase
        // surface DONE / UNDERFLOW while running
        if (ctrl_reg[C_STATUS][1]) begin   // was RUNNING: refresh DONE/UNDERFLOW, clear RUNNING on finish
            ldr_status_we <= 1'b1;
            ldr_status_val <= {27'b0, ((zlc_done ? 5'b0 : ST_RUNNING) | (zlc_done ? ST_DONE : 5'b0) | (zlc_underflow ? ST_UNDERFLOW : 5'b0))};
        end
    end

    // --- the FINAL edge-table engine ------------------------------------------
    zlc_edge_streamer #(
        .CHANNEL_COUNT(CHANNEL_COUNT), .EDGE_ADDR_WIDTH(EDGE_ADDR_WIDTH),
        .SCAN_ADDR_WIDTH(SCAN_ADDR_WIDTH), .SCAN_COUNT_WIDTH(SCAN_COUNT_WIDTH), .BANK_SIZE(BANK_SIZE),
        .TICK_WIDTH(TICK_WIDTH), .NUM_SLOTS(NUM_SLOTS), .COEFF_WIDTH(COEFF_WIDTH), .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
        .BUS_COUNT(BUS_COUNT), .BUS_INDEX_WIDTH(BUS_INDEX_WIDTH), .BUS_WIDTH(BUS_WIDTH),
        .BUS_SEG_ADDR_WIDTH(BUS_SEG_ADDR_WIDTH), .BUS_SEL_WIDTH(BUS_SEL_WIDTH),
        .RD_LAT(2), .FIFO_DEPTH(3)
    ) zlc_engine_i (
        .clk(axi_clk), .reset(eng_reset), .start(eng_start),
        .prog_count(ctrl_reg[C_PROG_COUNT][EDGE_ADDR_WIDTH:0]),
        .repeat_forever(ctrl_reg[C_REPEAT_FOREVER][0]),
        .loop_start_addr(ctrl_reg[C_LOOP_START][EDGE_ADDR_WIDTH-1:0]),
        .loop_end_tick(ctrl_reg[C_LOOP_END_TICK][TICK_WIDTH-1:0]),
        .loop_end_coeffs({ctrl_reg[C_LOOP_END_HI][COEFF_BITS-33:0], ctrl_reg[C_LOOP_END_LO]}),
        .loop_count(ctrl_reg[C_LOOP_COUNT]),
        .scan_enable(ctrl_reg[C_SCAN_ENABLE][0]),
        .scan_count(ctrl_reg[C_SCAN_COUNT][SCAN_COUNT_WIDTH-1:0]),
        .edge_raddr(edge_raddr),
        .edge_tick_rdata(edge_tick_rdata),
        .edge_coeff_rdata(edge_coeff_rdata_w[COEFF_BITS-1:0]),
        .edge_mask_rdata(edge_mask_rdata_w[CHANNEL_COUNT-1:0]),
        .scan_raddr(scan_raddr), .scan_rdata(scan_rdata_w),
        .bank_ready(ctrl_reg[C_BANK_READY][1:0]),
        .bank_chunk0(ctrl_reg[C_BANK0_CHUNK][SCAN_COUNT_WIDTH-1:0]),
        .bank_chunk1(ctrl_reg[C_BANK1_CHUNK][SCAN_COUNT_WIDTH-1:0]),
        .scan_cursor(zlc_cursor), .underflow(zlc_underflow),
        .bus_prog_we(bus_prog_we), .bus_prog_bus(bus_prog_bus), .bus_prog_addr(bus_prog_addr),
        .bus_prog_start_tick(bus_prog_start_tick), .bus_prog_stop_tick(bus_prog_stop_tick),
        .bus_prog_start_tick_coeffs(bus_prog_start_tick_coeffs),
        .bus_prog_stop_tick_coeffs(bus_prog_stop_tick_coeffs),
        .bus_prog_start_value(bus_prog_start_value), .bus_prog_stop_value(bus_prog_stop_value),
        .bus_prog_mode(bus_prog_mode), .bus_prog_value_select(bus_prog_value_select),
        .bus_prog_stop_value_select(bus_prog_stop_value_select),
        .bus_counts(ctrl_reg[C_BUS_COUNTS][BUS_COUNT*(BUS_SEG_ADDR_WIDTH+1)-1:0]),
        .out(out), .bus_out(zlc_bus_out), .running(zlc_running), .done(zlc_done)
    );

    // ---- JTAG-to-AXI + AXI BRAM controller IP --------------------------------
    jtag_axi_0 zlc_jtag_axi_i (
        .aclk(axi_clk), .aresetn(axi_resetn),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awprot(m_axi_awprot),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arprot(m_axi_arprot),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );
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

    // ---- LEDs + 62-pin board map (identical to the validated board XDC) -------
    assign led[0] = zlc_running;
    assign led[1] = |out;
    assign cooling = out[0]; assign cooling_pgc = out[1]; assign repump = out[2]; assign probe = out[3];
    assign pushout = out[4]; assign state_pre = out[5]; assign trig = out[6]; assign coil = out[7];
    assign grey_cooling = out[8]; assign trap = out[9]; assign UV = out[10]; assign emCCD = out[11];
    assign microwave = out[12]; assign address = out[13];
    assign cooling_shutter = out[14]; assign repump_shutter = out[15]; assign probe_shutter = out[16];
    assign bias = out[17];
    assign da_dipole[0] = zlc_bus_out[0]; assign da_dipole[1] = zlc_bus_out[1];
    assign da_dipole[2] = zlc_bus_out[2]; assign da_dipole[3] = zlc_bus_out[3];
    assign da_dipole[4] = zlc_bus_out[4]; assign da_dipole[5] = zlc_bus_out[5];
    assign da_dipole[6] = zlc_bus_out[6]; assign da_dipole[7] = zlc_bus_out[7];
    assign da_dipole[8] = zlc_bus_out[8]; assign da_dipole[9] = zlc_bus_out[9];
    assign da_clk0 = out[28];
    assign da_bias_y[0] = zlc_bus_out[10]; assign da_bias_y[1] = zlc_bus_out[11];
    assign da_bias_y[2] = zlc_bus_out[12]; assign da_bias_y[3] = zlc_bus_out[13];
    assign da_bias_y[4] = zlc_bus_out[14]; assign da_bias_y[5] = zlc_bus_out[15];
    assign da_bias_y[6] = zlc_bus_out[16]; assign da_bias_y[7] = zlc_bus_out[17];
    assign da_bias_y[8] = zlc_bus_out[18]; assign da_bias_y[9] = zlc_bus_out[19];
    assign da_clk1 = out[39];
    assign da_bias_x[0] = zlc_bus_out[20]; assign da_bias_x[1] = zlc_bus_out[21];
    assign da_bias_x[2] = zlc_bus_out[22]; assign da_bias_x[3] = zlc_bus_out[23];
    assign da_bias_x[4] = zlc_bus_out[24]; assign da_bias_x[5] = zlc_bus_out[25];
    assign da_bias_x[6] = zlc_bus_out[26]; assign da_bias_x[7] = zlc_bus_out[27];
    assign da_bias_x[8] = zlc_bus_out[28]; assign da_bias_x[9] = zlc_bus_out[29];
    assign da_clk2 = out[50];
    assign da_bias_z[0] = zlc_bus_out[30]; assign da_bias_z[1] = zlc_bus_out[31];
    assign da_bias_z[2] = zlc_bus_out[32]; assign da_bias_z[3] = zlc_bus_out[33];
    assign da_bias_z[4] = zlc_bus_out[34]; assign da_bias_z[5] = zlc_bus_out[35];
    assign da_bias_z[6] = zlc_bus_out[36]; assign da_bias_z[7] = zlc_bus_out[37];
    assign da_bias_z[8] = zlc_bus_out[38]; assign da_bias_z[9] = zlc_bus_out[39];
    assign da_clk3 = out[61];
    assign GND1 = 1'b0; assign GND4 = 1'b0; assign GND5 = 1'b0; assign GND6 = 1'b0;
    assign GND7 = 1'b0; assign GND8 = 1'b0; assign GND9 = 1'b0; assign GND10 = 1'b0;
    assign GND11 = 1'b0; assign GND12 = 1'b0; assign GND13 = 1'b0; assign GND14 = 1'b0;
    assign GND15 = 1'b0;
endmodule
