`timescale 1ns / 1ps
// =============================================================================
// zlc_edge_streamer -- FINAL affine edge-table pulse streamer engine.
//
// Global edge-table playback with:
//   * edge + scan tables in BLOCK RAM (so thousands of edges + unbounded scan
//     points fit the device); bus segment tables in LUTRAM (the bus/ramp engine
//     reads them combinationally every tick, so they MUST stay async-read).
//   * a depth-FIFO_DEPTH continuous PREFETCH of the next edges (one BRAM read per
//     cycle, fixed RD_LAT-cycle latency) + first/second and loop-start/+1 SHADOWS
//     latched at arm time, so the fire stays SAME-CYCLE and back-to-back **1-tick
//     (20 ns) edges** play one per cycle -- and the four gapless reload sites
//     (start / loop-rewind / scan-advance / repeat) reseed the prefetch instantly.
//   * a 2-bank PING-PONG scan window: the engine plays scan point
//     scan_point_index = 0..N-1, addressing bank (idx/BANK_SIZE)%2; the host
//     refills the bank it just left (cursor + bank_ready handshake), so the total
//     number of scan points is UNBOUNDED.  A not-yet-refilled bank STALLS the
//     engine (holds, flags STATUS underflow) -- never emits a wrong point.
//
// PROVEN PRE-HARDWARE (no Verilog sim here): the algorithm is byte-identical to
// the combinatorial reference for every program shape at read latency 1 AND 2,
// 1-tick spacing included, and the streaming is gapless over the full N-point
// sweep / stall-not-corrupt when starved -- see fpga/pulse_streamer/host/
// engine_model.py (reference_play / prefetch_play / streaming_scan_play) and the
// test_final_engine_model_* tests.  An RTL-register-transfer mirror of THIS
// module's FIFO is also asserted == reference (test_edge_streamer_rtl_mirror_*).
//
// RD_LAT MUST equal the synthesised edge-BRAM read latency; the build tcl FORCES
// the edge BRAMs to READ_LATENCY_B = 2 (both output registers on) so RD_LAT=2 is
// deterministic.  BANK_SIZE is a power-of-two build constant from
// host.image.solve_capacity (the host uses the same value).
//
// *** WIP -- PREFETCH NOT YET 1-TICK-EXACT, DO NOT BUILD. ***  The RTL-register-
// transfer mirror (engine_model.rtl_mirror_play) caught a real bug in THIS code:
// seeding only 2 shadows (edges target, target+1) + issuing the first prefetch
// read one cycle too late cannot sustain back-to-back 1-tick edges right after a
// boundary -- the next edge's BRAM read + the arm-register append (+1 cycle) miss
// the deadline.  The fix being converged in the mirror: seed FIFO_DEPTH(=RD_LAT+1)
// shadows AND add a same-cycle BYPASS (fire directly from edge_*_rdata on the
// cycle a read lands when the FIFO head is empty), so a 1-tick edge that lands
// exactly when needed still fires that cycle.  This module will be rewritten to
// match the corrected mirror (test_edge_streamer_rtl_mirror_*) before any build.
//
// Tables are external (top-level BRAM): edge fields are 3 PARALLEL BRAMs read in
// lockstep (tick / coeffs / mask) so a whole edge arrives per access with no
// width padding; scan is one BRAM (slot vector per access).  Bus is written
// through bus_prog_* by the top's mini-loader.
// =============================================================================

module zlc_edge_streamer #(
    parameter integer CHANNEL_COUNT = 62,
    parameter integer EDGE_ADDR_WIDTH = 12,
    parameter integer SCAN_ADDR_WIDTH = 12,     // addresses 2*BANK_SIZE points
    parameter integer SCAN_COUNT_WIDTH = 32,    // total scan points N (unbounded)
    parameter integer BANK_SIZE = 2048,         // power of two; points per ping-pong bank
    parameter integer TICK_WIDTH = 32,
    parameter integer NUM_SLOTS = 4,
    parameter integer COEFF_WIDTH = 16,
    parameter integer COEFF_FRAC_BITS = 8,
    parameter integer BUS_COUNT = 4,
    parameter integer BUS_INDEX_WIDTH = 2,
    parameter integer BUS_WIDTH = 10,
    parameter integer BUS_SEG_ADDR_WIDTH = 6,
    parameter integer BUS_SEL_WIDTH = 3,
    parameter integer RD_LAT = 2,               // edge-BRAM read latency (forced)
    parameter integer FIFO_DEPTH = 3            // >= RD_LAT + 1 for 1-tick spacing
)(
    input  wire clk,
    input  wire reset,
    input  wire start,

    // held program scalars (top regfile)
    input  wire [EDGE_ADDR_WIDTH:0] prog_count,
    input  wire repeat_forever,
    input  wire [EDGE_ADDR_WIDTH-1:0] loop_start_addr,
    input  wire [TICK_WIDTH-1:0] loop_end_tick,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] loop_end_coeffs,
    input  wire [31:0] loop_count,
    input  wire scan_enable,
    input  wire [SCAN_COUNT_WIDTH-1:0] scan_count,   // total N points

    // edge BRAM read port (3 parallel BRAMs, forced latency RD_LAT)
    output reg  [EDGE_ADDR_WIDTH-1:0] edge_raddr,
    input  wire [TICK_WIDTH-1:0] edge_tick_rdata,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] edge_coeff_rdata,
    input  wire [CHANNEL_COUNT-1:0] edge_mask_rdata,

    // scan BRAM read port (2-bank window; latency RD_LAT)
    output reg  [SCAN_ADDR_WIDTH-1:0] scan_raddr,
    input  wire [NUM_SLOTS*TICK_WIDTH-1:0] scan_rdata,

    // streaming handshake with the host
    input  wire [1:0] bank_ready,               // bit b: bank b loaded
    output reg  [SCAN_COUNT_WIDTH-1:0] scan_cursor,  // points consumed (host refills behind)
    output reg  underflow,                      // a bank was not ready in time

    // bus segment table write port (LUTRAM inside this module)
    input  wire bus_prog_we,
    input  wire [BUS_INDEX_WIDTH-1:0] bus_prog_bus,
    input  wire [BUS_SEG_ADDR_WIDTH-1:0] bus_prog_addr,
    input  wire [TICK_WIDTH-1:0] bus_prog_start_tick,
    input  wire [TICK_WIDTH-1:0] bus_prog_stop_tick,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_start_tick_coeffs,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_stop_tick_coeffs,
    input  wire [BUS_WIDTH-1:0] bus_prog_start_value,
    input  wire [BUS_WIDTH-1:0] bus_prog_stop_value,
    input  wire [1:0] bus_prog_mode,
    input  wire [BUS_SEL_WIDTH-1:0] bus_prog_value_select,
    input  wire [BUS_COUNT*(BUS_SEG_ADDR_WIDTH+1)-1:0] bus_counts,

    output wire [CHANNEL_COUNT-1:0] out,
    output wire [BUS_COUNT*BUS_WIDTH-1:0] bus_out,
    output reg  running = 1'b0,
    output reg  done = 1'b0
);

    localparam integer MAX_BUS_SEGMENTS = (1 << BUS_SEG_ADDR_WIDTH);
    localparam integer MAX_BUS_SEGMENT_ROWS = BUS_COUNT * MAX_BUS_SEGMENTS;
    localparam integer COEFF_BITS = NUM_SLOTS * COEFF_WIDTH;
    localparam integer SLOT_BITS = NUM_SLOTS * TICK_WIDTH;
    localparam integer ACC_WIDTH = TICK_WIDTH + COEFF_WIDTH + 4;
    localparam integer BANK_BITS = $clog2(BANK_SIZE);
    localparam [1:0] BUS_MODE_RAMP = 2'd2;

    // ----- bus segment tables: LUTRAM (per-tick combinatorial read) -----------
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_start_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_stop_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_start_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_stop_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [1:0] bus_mode_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_SEL_WIDTH-1:0] bus_value_select_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_start_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_stop_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];

    // ----- engine state -------------------------------------------------------
    reg [CHANNEL_COUNT-1:0] state_mask = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] time_count = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] loop_end_active = {TICK_WIDTH{1'b0}};
    reg [EDGE_ADDR_WIDTH:0] edge_index = {(EDGE_ADDR_WIDTH+1){1'b0}};
    reg [EDGE_ADDR_WIDTH:0] active_count = {(EDGE_ADDR_WIDTH+1){1'b0}};
    reg repeat_forever_active = 1'b0;
    reg scan_enable_active = 1'b0;
    reg [SLOT_BITS-1:0] slot_active = {SLOT_BITS{1'b0}};
    reg [SCAN_COUNT_WIDTH-1:0] active_scan_count = {SCAN_COUNT_WIDTH{1'b0}};
    reg [SCAN_COUNT_WIDTH-1:0] scan_point_index = {SCAN_COUNT_WIDTH{1'b0}};
    reg [31:0] loop_count_active = 32'd1;
    reg [31:0] loops_remaining = 32'd1;

    // shadows latched at arm time (BRAM pre-reads while reset is asserted)
    reg [TICK_WIDTH-1:0]  sh_first_t, sh_second_t, sh_ls_t, sh_ls1_t, sh_final_t;
    reg [COEFF_BITS-1:0]  sh_first_c, sh_second_c, sh_ls_c, sh_ls1_c, sh_final_c;
    reg [CHANNEL_COUNT-1:0] sh_first_m, sh_second_m, sh_ls_m, sh_ls1_m;
    reg [SLOT_BITS-1:0] scan_first_values;

    // depth-FIFO_DEPTH edge prefetch (FIFO_DEPTH==3 used here)
    reg [TICK_WIDTH-1:0] arm_t [0:FIFO_DEPTH-1];
    reg [COEFF_BITS-1:0] arm_c [0:FIFO_DEPTH-1];
    reg [CHANNEL_COUNT-1:0] arm_m [0:FIFO_DEPTH-1];
    reg [2:0] arm_nv;                         // number of valid arm entries (0..FIFO_DEPTH)
    reg [EDGE_ADDR_WIDTH:0] fetch_idx;        // next edge index to read
    reg [RD_LAT-1:0] pend;                    // in-flight read markers (1 bit/latency-stage)

    // ----- bus runtime --------------------------------------------------------
    reg [BUS_WIDTH-1:0] bus_value_active [0:BUS_COUNT-1];
    reg [BUS_SEG_ADDR_WIDTH:0] bus_index_active [0:BUS_COUNT-1];
    reg [BUS_SEG_ADDR_WIDTH:0] bus_count_active [0:BUS_COUNT-1];
    reg bus_ramp_active [0:BUS_COUNT-1];
    reg bus_ramp_dir_up [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_start_tick [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_stop_tick [0:BUS_COUNT-1];
    reg [BUS_WIDTH-1:0] bus_ramp_target [0:BUS_COUNT-1];
    reg [BUS_WIDTH:0] bus_ramp_delta [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_denom [0:BUS_COUNT-1];
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_ramp_accum [0:BUS_COUNT-1];

    reg reset_meta = 1'b0, reset_sync = 1'b0;
    reg start_meta = 1'b0, start_sync = 1'b0, start_prev = 1'b0;
    reg bus_prog_we_meta = 1'b0, bus_prog_we_sync = 1'b0, bus_prog_we_prev = 1'b0;
    integer bus_loop, ii;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_prog_flat_addr, bus_runtime_addr;
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_accum_next;

    wire start_event = start_sync && !start_prev;
    wire bus_prog_we_event = bus_prog_we_sync != bus_prog_we_prev;

    assign out = state_mask;
    genvar gi;
    generate
        for (gi = 0; gi < BUS_COUNT; gi = gi + 1) begin : zlc_bus_out_assign
            assign bus_out[gi*BUS_WIDTH +: BUS_WIDTH] = bus_value_active[gi];
        end
    endgenerate

    function [TICK_WIDTH-1:0] zlc_effective_tick;
        input [TICK_WIDTH-1:0] base_tick;
        input [COEFF_BITS-1:0] coeffs;
        input [SLOT_BITS-1:0] slots;
        integer slot_i;
        reg signed [ACC_WIDTH-1:0] acc;
        reg signed [COEFF_WIDTH-1:0] coeff_i;
        reg signed [TICK_WIDTH-1:0] slot_value_i;
        reg signed [ACC_WIDTH-1:0] total;
        begin
            acc = {ACC_WIDTH{1'b0}};
            for (slot_i = 0; slot_i < NUM_SLOTS; slot_i = slot_i + 1) begin
                coeff_i = coeffs[slot_i*COEFF_WIDTH +: COEFF_WIDTH];
                slot_value_i = slots[slot_i*TICK_WIDTH +: TICK_WIDTH];
                acc = acc + ($signed(coeff_i) * $signed(slot_value_i));
            end
            total = $signed({1'b0, base_tick}) + (acc >>> COEFF_FRAC_BITS);
            zlc_effective_tick = total[TICK_WIDTH-1:0];
        end
    endfunction

    function [BUS_SEG_ADDR_WIDTH:0] zlc_bus_count_at;
        input integer bus_index;
        begin zlc_bus_count_at = bus_counts[bus_index*(BUS_SEG_ADDR_WIDTH+1) +: (BUS_SEG_ADDR_WIDTH+1)]; end
    endfunction

    // scan-window address for point idx (2 banks, BANK_SIZE pow2)
    function [SCAN_ADDR_WIDTH-1:0] scan_addr_of;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        reg bnk;
        begin
            bnk = idx[BANK_BITS];                          // (idx/BANK_SIZE) & 1
            scan_addr_of = {bnk, idx[BANK_BITS-1:0]};      // bank*BANK_SIZE + offset
        end
    endfunction
    function bank_of;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        begin bank_of = idx[BANK_BITS]; end
    endfunction

    task zlc_bus_clear_runtime;
        integer i;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                bus_value_active[i] <= {BUS_WIDTH{1'b0}};
                bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_count_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_ramp_active[i] <= 1'b0; bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_delta[i] <= {(BUS_WIDTH+1){1'b0}};
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
            end
        end
    endtask

    task zlc_bus_apply_segment;
        input integer i;
        input [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        input [SLOT_BITS-1:0] slot_vec;
        reg [TICK_WIDTH-1:0] span;
        reg [BUS_SEL_WIDTH-1:0] sel;
        reg [BUS_WIDTH-1:0] vstart, vstop;
        reg [TICK_WIDTH-1:0] tkstart, tkstop;
        begin
            sel = bus_value_select_mem[addr];
            if (sel != {BUS_SEL_WIDTH{1'b0}}) begin
                vstart = slot_vec[(sel - 1'b1)*TICK_WIDTH +: BUS_WIDTH]; vstop = vstart;
            end else begin
                vstart = bus_start_value_mem[addr]; vstop = bus_stop_value_mem[addr];
            end
            tkstart = zlc_effective_tick(bus_start_tick_mem[addr], bus_start_tick_coeff_mem[addr], slot_vec);
            tkstop = zlc_effective_tick(bus_stop_tick_mem[addr], bus_stop_tick_coeff_mem[addr], slot_vec);
            if (bus_mode_mem[addr] == BUS_MODE_RAMP && tkstop > tkstart) begin
                span = tkstop - tkstart;
                bus_value_active[i] <= vstart; bus_ramp_active[i] <= 1'b1;
                bus_ramp_start_tick[i] <= tkstart; bus_ramp_stop_tick[i] <= tkstop;
                bus_ramp_target[i] <= vstop; bus_ramp_denom[i] <= span;
                bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                if (vstop >= vstart) begin bus_ramp_dir_up[i] <= 1'b1; bus_ramp_delta[i] <= vstop - vstart; end
                else begin bus_ramp_dir_up[i] <= 1'b0; bus_ramp_delta[i] <= vstart - vstop; end
            end else begin
                bus_value_active[i] <= vstop; bus_ramp_active[i] <= 1'b0;
                bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
            end
        end
    endtask

    function [TICK_WIDTH-1:0] zlc_bus_seg_start;
        input [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        input [SLOT_BITS-1:0] slot_vec;
        begin zlc_bus_seg_start = zlc_effective_tick(bus_start_tick_mem[addr], bus_start_tick_coeff_mem[addr], slot_vec); end
    endfunction

    task zlc_bus_start_table;
        input [SLOT_BITS-1:0] slot_vec;
        integer i;
        reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        reg [BUS_SEG_ADDR_WIDTH:0] count;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                count = zlc_bus_count_at(i); addr = i * MAX_BUS_SEGMENTS;
                bus_count_active[i] <= count; bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_value_active[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_active[i] <= 1'b0; bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_delta[i] <= {(BUS_WIDTH+1){1'b0}};
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                if (count != 0 && zlc_bus_seg_start(addr, slot_vec) == {TICK_WIDTH{1'b0}}) begin
                    zlc_bus_apply_segment(i, addr, slot_vec);
                    bus_index_active[i] <= {{BUS_SEG_ADDR_WIDTH{1'b0}}, 1'b1};
                end
            end
        end
    endtask

    task zlc_bus_step;
        begin
            for (bus_loop = 0; bus_loop < BUS_COUNT; bus_loop = bus_loop + 1) begin
                if (bus_ramp_active[bus_loop]) begin
                    if (time_count >= bus_ramp_stop_tick[bus_loop]) begin
                        bus_value_active[bus_loop] <= bus_ramp_target[bus_loop];
                        bus_ramp_active[bus_loop] <= 1'b0;
                        bus_ramp_accum[bus_loop] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                        if (bus_index_active[bus_loop] < bus_count_active[bus_loop]) begin
                            bus_runtime_addr = (bus_loop * MAX_BUS_SEGMENTS) + bus_index_active[bus_loop][BUS_SEG_ADDR_WIDTH-1:0];
                            if (zlc_bus_seg_start(bus_runtime_addr, slot_active) <= time_count) begin
                                zlc_bus_apply_segment(bus_loop, bus_runtime_addr, slot_active);
                                bus_index_active[bus_loop] <= bus_index_active[bus_loop] + 1'b1;
                            end
                        end
                    end else if (time_count > bus_ramp_start_tick[bus_loop] && bus_ramp_denom[bus_loop] != 0) begin
                        bus_accum_next = bus_ramp_accum[bus_loop] + bus_ramp_delta[bus_loop];
                        if (bus_accum_next >= bus_ramp_denom[bus_loop]) begin
                            bus_ramp_accum[bus_loop] <= bus_accum_next - bus_ramp_denom[bus_loop];
                            if (bus_ramp_dir_up[bus_loop]) begin
                                if (bus_value_active[bus_loop] < bus_ramp_target[bus_loop]) bus_value_active[bus_loop] <= bus_value_active[bus_loop] + 1'b1;
                            end else begin
                                if (bus_value_active[bus_loop] > bus_ramp_target[bus_loop]) bus_value_active[bus_loop] <= bus_value_active[bus_loop] - 1'b1;
                            end
                        end else bus_ramp_accum[bus_loop] <= bus_accum_next;
                    end
                end else if (bus_index_active[bus_loop] < bus_count_active[bus_loop]) begin
                    bus_runtime_addr = (bus_loop * MAX_BUS_SEGMENTS) + bus_index_active[bus_loop][BUS_SEG_ADDR_WIDTH-1:0];
                    if (time_count >= zlc_bus_seg_start(bus_runtime_addr, slot_active)) begin
                        zlc_bus_apply_segment(bus_loop, bus_runtime_addr, slot_active);
                        bus_index_active[bus_loop] <= bus_index_active[bus_loop] + 1'b1;
                    end
                end
            end
        end
    endtask

    // ----- ARM sequencer: pre-read shadows + scan point 0 (while reset high) ---
    localparam [3:0] A_E0=0, A_E1=1, A_LS=2, A_LS1=3, A_FIN=4, A_SC0=5, A_READY=6;
    reg [3:0] arm_state = A_E0;
    reg [3:0] arm_wait;
    reg arm_kick;

    // ----- prefetch helpers (combinational next-state) -------------------------
    reg landed;
    reg do_fire;
    reg issue;
    reg [2:0] nv_after_fire;
    integer k;

    always @(posedge clk) begin
        reset_meta <= reset; reset_sync <= reset_meta;
        start_meta <= start; start_sync <= start_meta; start_prev <= start_sync;
        bus_prog_we_meta <= bus_prog_we; bus_prog_we_sync <= bus_prog_we_meta; bus_prog_we_prev <= bus_prog_we_sync;

        if (reset_sync && bus_prog_we_event && bus_prog_bus < BUS_COUNT) begin
            bus_prog_flat_addr = {bus_prog_bus, bus_prog_addr};
            bus_start_tick_mem[bus_prog_flat_addr] <= bus_prog_start_tick;
            bus_stop_tick_mem[bus_prog_flat_addr] <= bus_prog_stop_tick;
            bus_start_value_mem[bus_prog_flat_addr] <= bus_prog_start_value;
            bus_stop_value_mem[bus_prog_flat_addr] <= bus_prog_stop_value;
            bus_mode_mem[bus_prog_flat_addr] <= bus_prog_mode;
            bus_value_select_mem[bus_prog_flat_addr] <= bus_prog_value_select;
            bus_start_tick_coeff_mem[bus_prog_flat_addr] <= bus_prog_start_tick_coeffs;
            bus_stop_tick_coeff_mem[bus_prog_flat_addr] <= bus_prog_stop_tick_coeffs;
        end

        if (reset_sync) begin
            running <= 1'b0; done <= 1'b0; underflow <= 1'b0;
            state_mask <= {CHANNEL_COUNT{1'b0}};
            time_count <= {TICK_WIDTH{1'b0}};
            edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
            scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}};
            scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
            loops_remaining <= 32'd1;
            arm_nv <= 3'd0; pend <= {RD_LAT{1'b0}};
            zlc_bus_clear_runtime();
            case (arm_state)
                A_E0: begin
                    edge_raddr <= {EDGE_ADDR_WIDTH{1'b0}};
                    if (arm_kick) begin
                        if (arm_wait == 0) begin sh_first_t <= edge_tick_rdata; sh_first_c <= edge_coeff_rdata; sh_first_m <= edge_mask_rdata;
                            edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}},1'b1}; arm_wait <= RD_LAT[3:0]; arm_state <= A_E1; end
                        else arm_wait <= arm_wait - 1'b1;
                    end else begin arm_kick <= 1'b1; arm_wait <= RD_LAT[3:0]; end
                end
                A_E1: if (arm_wait==0) begin sh_second_t<=edge_tick_rdata; sh_second_c<=edge_coeff_rdata; sh_second_m<=edge_mask_rdata;
                        edge_raddr<=loop_start_addr; arm_wait<=RD_LAT[3:0]; arm_state<=A_LS; end else arm_wait<=arm_wait-1'b1;
                A_LS: if (arm_wait==0) begin sh_ls_t<=edge_tick_rdata; sh_ls_c<=edge_coeff_rdata; sh_ls_m<=edge_mask_rdata;
                        edge_raddr<=loop_start_addr+1'b1; arm_wait<=RD_LAT[3:0]; arm_state<=A_LS1; end else arm_wait<=arm_wait-1'b1;
                A_LS1: if (arm_wait==0) begin sh_ls1_t<=edge_tick_rdata; sh_ls1_c<=edge_coeff_rdata; sh_ls1_m<=edge_mask_rdata;
                        edge_raddr<=(prog_count==0)?{EDGE_ADDR_WIDTH{1'b0}}:(prog_count[EDGE_ADDR_WIDTH-1:0]-1'b1); arm_wait<=RD_LAT[3:0]; arm_state<=A_FIN; end else arm_wait<=arm_wait-1'b1;
                A_FIN: if (arm_wait==0) begin sh_final_t<=edge_tick_rdata; sh_final_c<=edge_coeff_rdata;
                        scan_raddr<={SCAN_ADDR_WIDTH{1'b0}}; arm_wait<=RD_LAT[3:0]; arm_state<=A_SC0; end else arm_wait<=arm_wait-1'b1;
                A_SC0: if (arm_wait==0) begin scan_first_values<=(scan_enable && scan_count!=0)?scan_rdata:{SLOT_BITS{1'b0}}; arm_state<=A_READY; end else arm_wait<=arm_wait-1'b1;
                A_READY: ;
                default: arm_state <= A_E0;
            endcase
        end else if (start_event && !running) begin
            running <= (prog_count != 0); done <= (prog_count == 0); underflow <= 1'b0;
            active_count <= prog_count; repeat_forever_active <= repeat_forever;
            scan_enable_active <= scan_enable && scan_count != 0; active_scan_count <= scan_count;
            slot_active <= scan_first_values; scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}};
            scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
            scan_raddr <= scan_addr_of({{(SCAN_COUNT_WIDTH-1){1'b0}},1'b1});      // pre-read point 1
            loop_count_active <= (loop_count==0)?32'd1:loop_count;
            loops_remaining <= (loop_count==0)?32'd1:loop_count;
            final_tick <= (prog_count==0)?{TICK_WIDTH{1'b0}}:zlc_effective_tick(sh_final_t, sh_final_c, scan_first_values);
            loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, scan_first_values);
            zlc_bus_start_table(scan_first_values);
            // seed prefetch FIFO from shadows (edges 0,1); fetch from edge 2.
            arm_t[0]<=sh_first_t; arm_c[0]<=sh_first_c; arm_m[0]<=sh_first_m;
            arm_t[1]<=sh_second_t; arm_c[1]<=sh_second_c; arm_m[1]<=sh_second_m;
            arm_nv <= (prog_count>1)?3'd2:((prog_count==1)?3'd1:3'd0);
            fetch_idx <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'b10}; pend <= {RD_LAT{1'b0}};
            edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'b10};
            if (prog_count!=0 && zlc_effective_tick(sh_first_t,sh_first_c,scan_first_values)=={TICK_WIDTH{1'b0}}) begin
                // edge 0 fires immediately -> pop it; arm[0] becomes edge 1
                state_mask <= sh_first_m; time_count <= {{(TICK_WIDTH-1){1'b0}},1'b1};
                edge_index <= {{EDGE_ADDR_WIDTH{1'b0}},1'b1};
                arm_t[0]<=sh_second_t; arm_c[0]<=sh_second_c; arm_m[0]<=sh_second_m;
                arm_nv <= (prog_count>1)?3'd1:3'd0;
            end else begin
                state_mask <= {CHANNEL_COUNT{1'b0}}; time_count <= {TICK_WIDTH{1'b0}};
                edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
            end
        end else if (running) begin
            // -------- edge prefetch read pipeline (issue 1/cycle, land RD_LAT later)
            landed = pend[RD_LAT-1];
            // shift pend; new issue decided below
            // (defaults; do_fire / reseed branches set state)
            do_fire = 1'b0;

            if (loop_count_active>32'd1 && loops_remaining>32'd1 && time_count>=loop_end_active) begin
                state_mask <= sh_ls_m; time_count <= zlc_effective_tick(sh_ls_t,sh_ls_c,slot_active)+1'b1;
                edge_index <= {1'b0,loop_start_addr}+1'b1; loops_remaining <= loops_remaining-1'b1;
                arm_t[0]<=sh_ls1_t; arm_c[0]<=sh_ls1_c; arm_m[0]<=sh_ls1_m;
                arm_nv <= ((loop_start_addr+1'b1) < prog_count[EDGE_ADDR_WIDTH-1:0]) ? 3'd1 : 3'd0;
                fetch_idx <= {1'b0,loop_start_addr}+2'd2; edge_raddr <= loop_start_addr+2'd2;
                pend <= {RD_LAT{1'b0}};
                zlc_bus_start_table(slot_active);
            end else if (time_count >= final_tick) begin
                if (scan_enable_active && (scan_point_index+1'b1) < active_scan_count) begin
                    // need the next point resident (its bank ready)
                    if (!bank_ready[bank_of(scan_point_index+1'b1)]) begin
                        underflow <= 1'b1;          // STALL: hold, re-check next cycle
                    end else begin
                        underflow <= 1'b0;
                        scan_point_index <= scan_point_index+1'b1;
                        scan_cursor <= scan_point_index+1'b1;
                        scan_raddr <= scan_addr_of(scan_point_index+2'd2);   // pre-read the following point
                        slot_active <= scan_rdata;
                        final_tick <= zlc_effective_tick(sh_final_t,sh_final_c,scan_rdata);
                        loop_end_active <= zlc_effective_tick(loop_end_tick,loop_end_coeffs,scan_rdata);
                        loops_remaining <= loop_count_active;
                        // reseed FIFO from shadows for the new point
                        arm_t[0]<=sh_first_t; arm_c[0]<=sh_first_c; arm_m[0]<=sh_first_m;
                        arm_t[1]<=sh_second_t; arm_c[1]<=sh_second_c; arm_m[1]<=sh_second_m;
                        fetch_idx <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'b10}; edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'b10};
                        pend <= {RD_LAT{1'b0}};
                        if (zlc_effective_tick(sh_first_t,sh_first_c,scan_rdata)=={TICK_WIDTH{1'b0}}) begin
                            state_mask <= sh_first_m; time_count <= {{(TICK_WIDTH-1){1'b0}},1'b1}; edge_index <= {{EDGE_ADDR_WIDTH{1'b0}},1'b1};
                            arm_t[0]<=sh_second_t; arm_c[0]<=sh_second_c; arm_m[0]<=sh_second_m;
                            arm_nv <= (active_count>1)?3'd1:3'd0;
                        end else begin
                            state_mask <= {CHANNEL_COUNT{1'b0}}; time_count <= {TICK_WIDTH{1'b0}}; edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
                            arm_nv <= (active_count>1)?3'd2:((active_count==1)?3'd1:3'd0);
                        end
                    end
                end else if (repeat_forever_active) begin
                    slot_active <= scan_first_values; scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}}; scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
                    scan_raddr <= scan_addr_of({{(SCAN_COUNT_WIDTH-1){1'b0}},1'b1});
                    final_tick <= zlc_effective_tick(sh_final_t,sh_final_c,scan_first_values);
                    loop_end_active <= zlc_effective_tick(loop_end_tick,loop_end_coeffs,scan_first_values);
                    loops_remaining <= loop_count_active;
                    arm_t[0]<=sh_first_t; arm_c[0]<=sh_first_c; arm_m[0]<=sh_first_m;
                    arm_t[1]<=sh_second_t; arm_c[1]<=sh_second_c; arm_m[1]<=sh_second_m;
                    fetch_idx <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'b10}; edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'b10}; pend <= {RD_LAT{1'b0}};
                    if (zlc_effective_tick(sh_first_t,sh_first_c,scan_first_values)=={TICK_WIDTH{1'b0}}) begin
                        state_mask <= sh_first_m; time_count <= {{(TICK_WIDTH-1){1'b0}},1'b1}; edge_index <= {{EDGE_ADDR_WIDTH{1'b0}},1'b1};
                        arm_t[0]<=sh_second_t; arm_c[0]<=sh_second_c; arm_m[0]<=sh_second_m; arm_nv <= (active_count>1)?3'd1:3'd0;
                    end else begin
                        state_mask <= {CHANNEL_COUNT{1'b0}}; time_count <= {TICK_WIDTH{1'b0}}; edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
                        arm_nv <= (active_count>1)?3'd2:((active_count==1)?3'd1:3'd0);
                    end
                end else begin
                    running <= 1'b0; done <= 1'b1; state_mask <= {CHANNEL_COUNT{1'b0}}; zlc_bus_clear_runtime();
                end
            end else begin
                zlc_bus_step();
                do_fire = (edge_index < active_count) && (arm_nv != 0) && (time_count == zlc_effective_tick(arm_t[0],arm_c[0],slot_active));
                if (do_fire) begin
                    state_mask <= arm_m[0];
                    edge_index <= edge_index + 1'b1;
                end
                time_count <= time_count + 1'b1;
                // --- FIFO update: shift on fire, append on land (this cycle) ---
                nv_after_fire = do_fire ? (arm_nv - 1'b1) : arm_nv;
                if (do_fire) begin
                    for (k = 0; k < FIFO_DEPTH-1; k = k + 1) begin
                        arm_t[k] <= arm_t[k+1]; arm_c[k] <= arm_c[k+1]; arm_m[k] <= arm_m[k+1];
                    end
                end
                if (landed) begin
                    arm_t[nv_after_fire] <= edge_tick_rdata;
                    arm_c[nv_after_fire] <= edge_coeff_rdata;
                    arm_m[nv_after_fire] <= edge_mask_rdata;
                    arm_nv <= nv_after_fire + 1'b1;
                end else begin
                    arm_nv <= nv_after_fire;
                end
                // --- issue a read to keep (resident + in-flight) at FIFO_DEPTH ---
                // After this cycle: resident = nv_after_fire + landed; the only
                // still-in-flight read (RD_LAT=2) is pend[0] (pend[RD_LAT-1] landed
                // this cycle).  Issue iff that total is below FIFO_DEPTH.
                issue = ((nv_after_fire + (landed ? 1'b1 : 1'b0) + pend[0]) < FIFO_DEPTH[2:0])
                        && (fetch_idx < active_count);
                if (issue) begin edge_raddr <= fetch_idx; fetch_idx <= fetch_idx + 1'b1; end
                pend <= {pend[RD_LAT-2:0], issue};
            end
        end
    end
endmodule
