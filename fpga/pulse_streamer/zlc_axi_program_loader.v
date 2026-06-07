`timescale 1ns / 1ps
// On-chip program loader for the affine edge-table pulse streamer.
//
// The validated, seamless engine ``zlc_pulse_streamer`` is fed through its
// VIO-style ``prog_*`` / ``scan_prog_*`` / ``bus_prog_*`` write ports.  Instead
// of a VIO, this loader copies a program *image* out of the AXI-addressable BRAM
// (port B) into those ports, then releases the engine reset and pulses start.
// Because the engine itself is unchanged, repeat-to-repeat and scan-point
// transitions stay single-cycle seamless and the affine scan stays tick-exact;
// the loader only delivers the same data a VIO upload used to deliver.
//
// Image layout = Zou_lab_control.neutral_atom.devices.edgetable_image (the
// host-side packer); the constants below MUST match it (a Python contract test
// asserts equality).  The loader walks the image exactly like
// ``edgetable_image.unpack_program``.
//
// Control mailbox (BRAM words, host writes COMMAND via AXI, polls STATUS):
//   COMMAND bit0=LOAD  bit1=FIRE  bit2=RESET  bit3=SAFE  (rising-edge detected)
//   STATUS  bit0=LOADED bit1=RUNNING bit2=DONE bit3=ERROR
//
// Writes to the engine are toggle-committed: the engine sees ``prog_we`` change
// (after a 2-FF synchroniser) as one write while its reset is asserted, and
// latches first/loop-start/final shadow registers from prog_count/loop_start_addr
// during those writes -- so the loader sets the scalars first and holds them, and
// holds each row's data stable for several cycles around the prog_we toggle.

module zlc_axi_program_loader #(
    parameter integer CHANNEL_COUNT = 62,
    parameter integer EDGE_ADDR_WIDTH = 10,
    parameter integer SCAN_ADDR_WIDTH = 10,
    parameter integer TICK_WIDTH = 32,
    parameter integer NUM_SLOTS = 4,
    parameter integer COEFF_WIDTH = 16,
    parameter integer BUS_COUNT = 4,
    parameter integer BUS_INDEX_WIDTH = 2,
    parameter integer BUS_SEG_ADDR_WIDTH = 6,
    parameter integer BUS_WIDTH = 10,
    parameter integer BUS_SEL_WIDTH = 3,
    parameter integer MEM_ADDR_WIDTH = 15,    // BRAM word-address width (32768 words)
    parameter integer RD_SETTLE = 4,          // cycles to wait per BRAM read (latency <=2 + margin)
    parameter integer WR_HOLD = 5             // cycles to hold a row stable around the prog_we toggle
)(
    input  wire clk,
    input  wire ext_reset,                    // synchronous loader reset (active high)

    // BRAM port B (word addressed).
    output reg  [MEM_ADDR_WIDTH-1:0] mem_addr = {MEM_ADDR_WIDTH{1'b0}},
    output reg  mem_en = 1'b1,
    output reg  mem_we = 1'b0,
    output reg  [31:0] mem_wdata = 32'b0,
    input  wire [31:0] mem_rdata,

    // Engine control.
    output reg  eng_reset = 1'b1,
    output reg  eng_start = 1'b0,
    input  wire eng_running,
    input  wire eng_done,

    // Engine program-write port (driven into zlc_pulse_streamer).
    output reg  prog_we = 1'b0,
    output reg  [EDGE_ADDR_WIDTH-1:0] prog_addr = {EDGE_ADDR_WIDTH{1'b0}},
    output reg  [TICK_WIDTH-1:0] prog_tick = {TICK_WIDTH{1'b0}},
    output reg  [NUM_SLOTS*COEFF_WIDTH-1:0] prog_tick_coeffs = {(NUM_SLOTS*COEFF_WIDTH){1'b0}},
    output reg  [CHANNEL_COUNT-1:0] prog_mask = {CHANNEL_COUNT{1'b0}},
    output reg  [EDGE_ADDR_WIDTH:0] prog_count = {(EDGE_ADDR_WIDTH+1){1'b0}},
    output reg  repeat_forever = 1'b0,
    output reg  [EDGE_ADDR_WIDTH-1:0] loop_start_addr = {EDGE_ADDR_WIDTH{1'b0}},
    output reg  [TICK_WIDTH-1:0] loop_end_tick = {TICK_WIDTH{1'b0}},
    output reg  [NUM_SLOTS*COEFF_WIDTH-1:0] loop_end_coeffs = {(NUM_SLOTS*COEFF_WIDTH){1'b0}},
    output reg  [31:0] loop_count = 32'd1,
    output reg  scan_enable = 1'b0,
    output reg  scan_prog_we = 1'b0,
    output reg  [SCAN_ADDR_WIDTH-1:0] scan_prog_addr = {SCAN_ADDR_WIDTH{1'b0}},
    output reg  [NUM_SLOTS*TICK_WIDTH-1:0] scan_prog_values = {(NUM_SLOTS*TICK_WIDTH){1'b0}},
    output reg  [SCAN_ADDR_WIDTH:0] scan_count = {(SCAN_ADDR_WIDTH+1){1'b0}},
    output reg  bus_prog_we = 1'b0,
    output reg  [BUS_INDEX_WIDTH-1:0] bus_prog_bus = {BUS_INDEX_WIDTH{1'b0}},
    output reg  [BUS_SEG_ADDR_WIDTH-1:0] bus_prog_addr = {BUS_SEG_ADDR_WIDTH{1'b0}},
    output reg  [TICK_WIDTH-1:0] bus_prog_start_tick = {TICK_WIDTH{1'b0}},
    output reg  [TICK_WIDTH-1:0] bus_prog_stop_tick = {TICK_WIDTH{1'b0}},
    output reg  [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_start_tick_coeffs = {(NUM_SLOTS*COEFF_WIDTH){1'b0}},
    output reg  [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_stop_tick_coeffs = {(NUM_SLOTS*COEFF_WIDTH){1'b0}},
    output reg  [BUS_WIDTH-1:0] bus_prog_start_value = {BUS_WIDTH{1'b0}},
    output reg  [BUS_WIDTH-1:0] bus_prog_stop_value = {BUS_WIDTH{1'b0}},
    output reg  [1:0] bus_prog_mode = 2'b0,
    output reg  [BUS_SEL_WIDTH-1:0] bus_prog_value_select = {BUS_SEL_WIDTH{1'b0}},
    output reg  [BUS_COUNT*(BUS_SEG_ADDR_WIDTH+1)-1:0] bus_counts = {(BUS_COUNT*(BUS_SEG_ADDR_WIDTH+1)){1'b0}},

    output reg  [4:0] loader_state = 5'd0      // diagnostic (LED / probe)
);

    // --- image geometry (MUST match edgetable_image.EdgeTableImageParams) -----
    localparam integer COEFF_BITS = NUM_SLOTS * COEFF_WIDTH;
    localparam integer SLOT_BITS = NUM_SLOTS * TICK_WIDTH;
    localparam integer COEFF_WORDS = (COEFF_BITS + 31) / 32;          // 2
    localparam integer MASK_WORDS = (CHANNEL_COUNT + 31) / 32;        // 2
    localparam integer EDGE_WORDS = 1 + COEFF_WORDS + MASK_WORDS;     // 5
    localparam integer SCAN_WORDS = NUM_SLOTS;                        // 4
    localparam integer BUS_WORDS = 2 + 2 * COEFF_WORDS + 1;           // 7
    localparam integer MAX_EDGES = (1 << EDGE_ADDR_WIDTH);            // 1024
    localparam integer MAX_SCAN_POINTS = (1 << SCAN_ADDR_WIDTH);      // 1024
    localparam integer MAX_BUS_SEGMENTS = (1 << BUS_SEG_ADDR_WIDTH);  // 64
    localparam integer CNT_WIDTH = BUS_SEG_ADDR_WIDTH + 1;            // 7

    localparam integer CTRL_WORDS = 32;
    localparam integer EDGE_BASE = CTRL_WORDS;
    localparam integer SCAN_BASE = EDGE_BASE + MAX_EDGES * EDGE_WORDS;
    localparam integer BUS_BASE = SCAN_BASE + MAX_SCAN_POINTS * SCAN_WORDS;

    // CTRL word offsets (== edgetable_image.CtrlWords)
    localparam integer C_MAGIC = 0;
    localparam integer C_COMMAND = 1;
    localparam integer C_STATUS = 2;
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
    localparam integer C_SLOT_COUNT = 13;
    localparam integer CTRL_SCALARS = 11;       // contiguous words 3..13

    localparam [31:0] IMAGE_MAGIC = 32'h5A4C4531;  // "ZLE1"

    localparam CMD_LOAD = 4'b0001;
    localparam CMD_FIRE = 4'b0010;
    localparam CMD_RESET = 4'b0100;
    localparam CMD_SAFE = 4'b1000;

    localparam STATUS_LOADED = 4'b0001;
    localparam STATUS_RUNNING = 4'b0010;
    localparam STATUS_DONE = 4'b0100;
    localparam STATUS_ERROR = 4'b1000;

    // --- FSM states -----------------------------------------------------------
    localparam [4:0]
        S_INIT      = 5'd0,
        S_IDLE_RD   = 5'd1,   // issue read of COMMAND
        S_IDLE_CAP  = 5'd2,   // capture COMMAND, edge-detect
        S_MAGIC_RD  = 5'd3,
        S_MAGIC_CAP = 5'd4,
        S_CTRL_RD   = 5'd5,
        S_CTRL_CAP  = 5'd6,
        S_ROW_RD    = 5'd7,   // generic row word read (issue)
        S_ROW_CAP   = 5'd8,   // capture word
        S_ROW_EMIT  = 5'd9,   // assemble + drive engine + toggle we
        S_ROW_HOLD  = 5'd10,  // hold data stable
        S_REGION_NX = 5'd11,  // advance row / region
        S_LOADED    = 5'd12,
        S_FIRE0     = 5'd13,  // release reset, wait
        S_FIRE1     = 5'd14,  // pulse start
        S_RUN       = 5'd15,
        S_PUB       = 5'd16,  // write STATUS to BRAM
        S_PUB_WAIT  = 5'd17,
        S_SAFE      = 5'd18,
        S_ERROR     = 5'd19;

    // region encoding for the generic row reader
    localparam [1:0] RG_EDGE = 2'd0, RG_SCAN = 2'd1, RG_BUS = 2'd2;

    reg [31:0] cap [0:6];               // captured row words (max BUS_WORDS=7)
    reg [2:0] wi;                       // word index within row
    reg [2:0] row_words;                // words in current row
    reg [1:0] region;                   // RG_*
    reg [EDGE_ADDR_WIDTH:0] row_idx;    // row index within region
    reg [EDGE_ADDR_WIDTH:0] row_total;  // rows in region (edges or scan points)
    reg [MEM_ADDR_WIDTH-1:0] row_base;  // BRAM word base of current row

    reg [EDGE_ADDR_WIDTH:0] emit_idx;   // row number actually being emitted (edge/scan)
    reg [BUS_SEG_ADDR_WIDTH:0] emit_bus_addr;

    reg [3:0] ctrl_idx;
    reg [BUS_INDEX_WIDTH:0] bus_cur;    // current bus during BUS region
    reg [BUS_SEG_ADDR_WIDTH:0] bus_addr_cur;
    reg [BUS_SEG_ADDR_WIDTH:0] bus_cur_count;

    reg [$clog2(RD_SETTLE+1)-1:0] rd_wait;
    reg [$clog2(WR_HOLD+1)-1:0] hold_cnt;
    reg [1:0] pub_wait;
    reg [4:0] pub_next;

    reg [3:0] status_r;
    reg [3:0] cmd_prev;
    reg [3:0] cmd_now;
    wire [3:0] cmd_edge = cmd_now & ~cmd_prev;

    reg [31:0] magic_r;

    integer k;
    initial begin
        for (k = 0; k < 7; k = k + 1) cap[k] = 32'b0;
        wi = 3'b0; row_words = 3'b0; region = 2'b0; row_idx = 0; row_total = 0;
        emit_idx = 0; emit_bus_addr = 0;
        row_base = 0; ctrl_idx = 0; bus_cur = 0; bus_addr_cur = 0; bus_cur_count = 0;
        rd_wait = 0; hold_cnt = 0; pub_wait = 0; pub_next = 0; status_r = 4'b0;
        cmd_prev = 4'b0; cmd_now = 4'b0; magic_r = 32'b0;
    end

    // per-bus segment count from the packed bus_counts scalar
    function [CNT_WIDTH-1:0] bus_count_of;
        input [BUS_COUNT*CNT_WIDTH-1:0] packed_counts;
        input integer b;
        begin
            bus_count_of = packed_counts[b*CNT_WIDTH +: CNT_WIDTH];
        end
    endfunction

    always @(posedge clk) begin
        // defaults (one-cycle strobes)
        mem_we <= 1'b0;
        if (ext_reset) begin
            eng_reset <= 1'b1;
            eng_start <= 1'b0;
            prog_we <= 1'b0;
            scan_prog_we <= 1'b0;
            bus_prog_we <= 1'b0;
            status_r <= 4'b0;
            cmd_prev <= 4'b0;
            loader_state <= S_INIT;
        end else begin
            loader_state <= loader_state;
            case (loader_state)
            // -------------------------------------------------------------- INIT
            S_INIT: begin
                eng_reset <= 1'b1;
                eng_start <= 1'b0;
                prog_we <= 1'b0; scan_prog_we <= 1'b0; bus_prog_we <= 1'b0;
                status_r <= 4'b0;
                cmd_prev <= 4'b0;
                loader_state <= S_IDLE_RD;
            end
            // ------------------------------------------------------------- IDLE
            S_IDLE_RD: begin
                mem_addr <= C_COMMAND[MEM_ADDR_WIDTH-1:0];
                mem_en <= 1'b1;
                rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                loader_state <= S_IDLE_CAP;
            end
            S_IDLE_CAP: begin
                if (rd_wait == 0) begin
                    cmd_now <= mem_rdata[3:0];
                    // Commands are rising-edge detected vs the previous poll (cmd_now).
                    // RESET/SAFE win over LOAD/FIRE.  While the engine is RUNNING we
                    // also watch eng_done -> DONE (a finite loop finishes; a
                    // repeat_forever program never asserts done, so it keeps polling
                    // commands here and can be stopped via SAFE/RESET).
                    if ((mem_rdata[3:0] & ~cmd_now & CMD_RESET) != 0) begin
                        loader_state <= S_INIT;
                    end else if ((mem_rdata[3:0] & ~cmd_now & CMD_SAFE) != 0) begin
                        loader_state <= S_SAFE;
                    end else if ((mem_rdata[3:0] & ~cmd_now & CMD_LOAD) != 0) begin
                        loader_state <= S_MAGIC_RD;
                    end else if (((mem_rdata[3:0] & ~cmd_now & CMD_FIRE) != 0) &&
                                 (status_r & STATUS_LOADED) != 0) begin
                        loader_state <= S_FIRE0;
                    end else if (((status_r & STATUS_RUNNING) != 0) && eng_done) begin
                        status_r <= (status_r | STATUS_DONE) & ~STATUS_RUNNING;
                        pub_next <= S_IDLE_RD;
                        loader_state <= S_PUB;
                    end else begin
                        loader_state <= S_IDLE_RD;
                    end
                end else begin
                    rd_wait <= rd_wait - 1'b1;
                end
            end
            // ------------------------------------------------------------ MAGIC
            S_MAGIC_RD: begin
                eng_reset <= 1'b1;                  // hold engine in reset for the whole load
                status_r <= 4'b0;
                mem_addr <= C_MAGIC[MEM_ADDR_WIDTH-1:0];
                rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                loader_state <= S_MAGIC_CAP;
            end
            S_MAGIC_CAP: begin
                if (rd_wait == 0) begin
                    magic_r <= mem_rdata;
                    if (mem_rdata != IMAGE_MAGIC) begin
                        status_r <= STATUS_ERROR;
                        pub_next <= S_IDLE_RD;
                        loader_state <= S_PUB;
                    end else begin
                        ctrl_idx <= 4'd0;
                        loader_state <= S_CTRL_RD;
                    end
                end else rd_wait <= rd_wait - 1'b1;
            end
            // ------------------------------------------------------------- CTRL
            S_CTRL_RD: begin
                mem_addr <= (C_PROG_COUNT + ctrl_idx);
                rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                loader_state <= S_CTRL_CAP;
            end
            S_CTRL_CAP: begin
                if (rd_wait == 0) begin
                    case (ctrl_idx)
                        4'd0: prog_count <= mem_rdata[EDGE_ADDR_WIDTH:0];
                        4'd1: scan_count <= mem_rdata[SCAN_ADDR_WIDTH:0];
                        4'd2: scan_enable <= mem_rdata[0];
                        4'd3: repeat_forever <= mem_rdata[0];
                        4'd4: loop_start_addr <= mem_rdata[EDGE_ADDR_WIDTH-1:0];
                        4'd5: loop_count <= mem_rdata;
                        4'd6: loop_end_tick <= mem_rdata[TICK_WIDTH-1:0];
                        4'd7: loop_end_coeffs[31:0] <= mem_rdata;
                        4'd8: loop_end_coeffs[COEFF_BITS-1:32] <= mem_rdata[COEFF_BITS-33:0];
                        4'd9: bus_counts <= mem_rdata[BUS_COUNT*CNT_WIDTH-1:0];
                        4'd10: ; // SLOT_COUNT: informational only
                        default: ;
                    endcase
                    if (ctrl_idx == (CTRL_SCALARS - 1)) begin
                        // begin EDGE region
                        region <= RG_EDGE;
                        row_words <= EDGE_WORDS[2:0];
                        row_idx <= 0;
                        // prog_count was just (ctrl_idx 0) latched; use mem path: read again from reg
                        row_total <= prog_count;          // prog_count already latched
                        row_base <= EDGE_BASE[MEM_ADDR_WIDTH-1:0];
                        wi <= 3'd0;
                        loader_state <= S_REGION_NX;       // REGION_NX decides go/skip
                    end else begin
                        ctrl_idx <= ctrl_idx + 1'b1;
                        loader_state <= S_CTRL_RD;
                    end
                end else rd_wait <= rd_wait - 1'b1;
            end
            // ----------------------------------------------- generic row reader
            S_ROW_RD: begin
                mem_addr <= row_base + wi;
                rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                loader_state <= S_ROW_CAP;
            end
            S_ROW_CAP: begin
                if (rd_wait == 0) begin
                    cap[wi] <= mem_rdata;
                    if (wi == (row_words - 1'b1)) begin
                        loader_state <= S_ROW_EMIT;
                    end else begin
                        wi <= wi + 1'b1;
                        loader_state <= S_ROW_RD;
                    end
                end else rd_wait <= rd_wait - 1'b1;
            end
            S_ROW_EMIT: begin
                // assemble fields + drive the proper engine write port, toggle we
                if (region == RG_EDGE) begin
                    prog_addr <= emit_idx[EDGE_ADDR_WIDTH-1:0];
                    prog_tick <= cap[0][TICK_WIDTH-1:0];
                    prog_tick_coeffs <= {cap[2][COEFF_BITS-33:0], cap[1]};
                    prog_mask <= {cap[4][CHANNEL_COUNT-33:0], cap[3]};
                    prog_we <= ~prog_we;
                end else if (region == RG_SCAN) begin
                    scan_prog_addr <= emit_idx[SCAN_ADDR_WIDTH-1:0];
                    scan_prog_values <= {cap[3], cap[2], cap[1], cap[0]};
                    scan_prog_we <= ~scan_prog_we;
                end else begin // RG_BUS
                    bus_prog_bus <= bus_cur[BUS_INDEX_WIDTH-1:0];
                    bus_prog_addr <= emit_bus_addr[BUS_SEG_ADDR_WIDTH-1:0];
                    bus_prog_start_tick <= cap[0][TICK_WIDTH-1:0];
                    bus_prog_stop_tick <= cap[1][TICK_WIDTH-1:0];
                    bus_prog_start_tick_coeffs <= {cap[3][COEFF_BITS-33:0], cap[2]};
                    bus_prog_stop_tick_coeffs <= {cap[5][COEFF_BITS-33:0], cap[4]};
                    bus_prog_start_value <= cap[6][BUS_WIDTH-1:0];
                    bus_prog_stop_value <= cap[6][2*BUS_WIDTH-1:BUS_WIDTH];
                    bus_prog_mode <= cap[6][2*BUS_WIDTH+1:2*BUS_WIDTH];
                    bus_prog_value_select <= cap[6][2*BUS_WIDTH+2+BUS_SEL_WIDTH-1:2*BUS_WIDTH+2];
                    bus_prog_we <= ~bus_prog_we;
                end
                hold_cnt <= WR_HOLD[$clog2(WR_HOLD+1)-1:0];
                loader_state <= S_ROW_HOLD;
            end
            S_ROW_HOLD: begin
                if (hold_cnt == 0) begin
                    loader_state <= S_REGION_NX;
                end else hold_cnt <= hold_cnt - 1'b1;
            end
            // ---------------------------------------------- region advance / skip
            S_REGION_NX: begin
                wi <= 3'd0;
                if (region == RG_EDGE) begin
                    if (row_idx >= row_total) begin
                        // EDGE done -> SCAN region
                        region <= RG_SCAN;
                        row_words <= SCAN_WORDS[2:0];
                        row_idx <= 0;
                        row_total <= scan_count;
                        row_base <= SCAN_BASE[MEM_ADDR_WIDTH-1:0];
                        loader_state <= S_REGION_NX;
                    end else begin
                        emit_idx <= row_idx;
                        row_base <= EDGE_BASE[MEM_ADDR_WIDTH-1:0] + row_idx * EDGE_WORDS;
                        row_idx <= row_idx + 1'b1;
                        loader_state <= S_ROW_RD;
                    end
                end else if (region == RG_SCAN) begin
                    if (row_idx >= row_total) begin
                        // SCAN done -> BUS region (start at bus 0)
                        region <= RG_BUS;
                        row_words <= BUS_WORDS[2:0];
                        bus_cur <= 0;
                        bus_addr_cur <= 0;
                        bus_cur_count <= bus_count_of(bus_counts, 0);
                        loader_state <= S_REGION_NX;
                    end else begin
                        row_base <= SCAN_BASE[MEM_ADDR_WIDTH-1:0] + row_idx * SCAN_WORDS;
                        row_idx <= row_idx + 1'b1;
                        loader_state <= S_ROW_RD;
                    end
                end else begin // RG_BUS
                    if (bus_addr_cur >= bus_cur_count) begin
                        if (bus_cur == (BUS_COUNT - 1)) begin
                            // all buses done -> loaded
                            status_r <= status_r | STATUS_LOADED;
                            pub_next <= S_LOADED;
                            loader_state <= S_PUB;
                        end else begin
                            bus_cur <= bus_cur + 1'b1;
                            bus_addr_cur <= 0;
                            bus_cur_count <= bus_count_of(bus_counts, bus_cur + 1'b1);
                            loader_state <= S_REGION_NX;
                        end
                    end else begin
                        emit_bus_addr <= bus_addr_cur;
                        row_base <= BUS_BASE[MEM_ADDR_WIDTH-1:0] +
                                    (bus_cur * MAX_BUS_SEGMENTS + bus_addr_cur) * BUS_WORDS;
                        bus_addr_cur <= bus_addr_cur + 1'b1;
                        loader_state <= S_ROW_RD;
                    end
                end
            end
            // ------------------------------------------------------------ LOADED
            S_LOADED: begin
                eng_reset <= 1'b1;     // keep engine held until FIRE; tables persist in LUTRAM
                loader_state <= S_IDLE_RD;
            end
            // ------------------------------------------------------------- FIRE
            S_FIRE0: begin
                eng_reset <= 1'b0;     // release; wait a few cycles for reset_sync to clear
                eng_start <= 1'b0;
                hold_cnt <= WR_HOLD[$clog2(WR_HOLD+1)-1:0];
                loader_state <= S_FIRE1;
            end
            S_FIRE1: begin
                if (hold_cnt == 0) begin
                    eng_start <= 1'b1;                 // rising edge -> engine start_event
                    status_r <= (status_r | STATUS_RUNNING) & ~STATUS_DONE;
                    pub_next <= S_RUN;
                    loader_state <= S_PUB;
                end else hold_cnt <= hold_cnt - 1'b1;
            end
            S_RUN: begin
                // start is a one-shot; drop it and return to the command poll, which
                // watches eng_done (finite loop) and still services SAFE/RESET/LOAD
                // (needed because a repeat_forever program never asserts done).
                eng_start <= 1'b0;
                loader_state <= S_IDLE_RD;
            end
            // ------------------------------------------------- publish STATUS
            S_PUB: begin
                mem_addr <= C_STATUS[MEM_ADDR_WIDTH-1:0];
                mem_wdata <= {28'b0, status_r};
                mem_we <= 1'b1;
                pub_wait <= 2'd2;
                loader_state <= S_PUB_WAIT;
            end
            S_PUB_WAIT: begin
                if (pub_wait == 0) loader_state <= pub_next;
                else pub_wait <= pub_wait - 1'b1;
            end
            // -------------------------------------------------------------- SAFE
            S_SAFE: begin
                eng_reset <= 1'b1;     // force engine reset -> outputs cleared
                eng_start <= 1'b0;
                status_r <= 4'b0;
                pub_next <= S_IDLE_RD;
                loader_state <= S_PUB;
            end
            S_ERROR: begin
                loader_state <= S_IDLE_RD;
            end
            default: loader_state <= S_INIT;
            endcase
        end
    end
endmodule
