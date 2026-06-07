`timescale 1ns / 1ps
// =============================================================================
// zlc_edge_streamer -- FINAL affine edge-table pulse streamer engine.
//
// Global edge-table playback with:
//   * edge + scan tables in BLOCK RAM (thousands of edges + unbounded scan
//     points); bus segment tables in LUTRAM (the bus/ramp engine reads them
//     combinationally every tick, so they MUST stay async-read).
//   * a depth-FIFO_DEPTH continuous PREFETCH of the next edges (one BRAM read per
//     cycle, fixed RD_LAT-cycle latency) + FIFO_DEPTH(=RD_LAT+1) edge SHADOWS
//     latched at arm time per boundary, so the four gapless reload sites
//     (start / loop-rewind / scan-advance / repeat) reseed instantly and
//     back-to-back **1-tick (20 ns) edges** play one per cycle.
//   * a 2-bank PING-PONG scan window: the engine plays scan point 0..N-1,
//     addressing bank (idx/BANK_SIZE)%2; the host refills the bank it just left
//     (cursor + bank_ready handshake), so total scan points are UNBOUNDED.  A
//     not-yet-refilled bank STALLS the engine (holds, flags STATUS underflow) --
//     never emits a wrong point.
//
// PROVEN PRE-HARDWARE (no Verilog sim in repo): this module's exact register
// transfers are mirrored cycle-for-cycle by engine_model.rtl_mirror_play, which
// is byte-identical to the combinatorial reference_play for every program shape
// (1-tick spacing included) at read latency 1, 2 AND 3, over hand cases
// (b2b1/scan1tick/loop1tick) + 400 fuzz programs.  The streaming ping-pong is
// proven by streaming_scan_play.  See test_final_engine_model_* /
// test_edge_streamer_rtl_mirror_*.
//
// SEED INVARIANT (the subtle part the mirror forced out): at every boundary, seed
// FIFO_DEPTH resident shadows starting at the FIRST not-yet-output edge, and
// issue NO read at the boundary (occupancy == #shadows <= depth).  The first
// PREFETCHED edge is issued only when the head fires and frees a slot; with
// FIFO_DEPTH = RD_LAT+1 that read lands and registers into arm exactly in time
// for a 1-tick successor.  A 2-shadow seed is one cycle short and drops edges.
//
// RD_LAT MUST equal the synthesised edge-BRAM read latency; the build tcl FORCES
// the edge BRAMs to READ_LATENCY_B = 2 so RD_LAT=2 is deterministic.  BANK_SIZE
// is a power-of-two build constant from host.image.solve_capacity.
//
// Edge fields are 3 PARALLEL BRAMs read in lockstep (tick / coeffs / mask) so a
// whole edge arrives per access with no width padding; scan is one BRAM.
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
    parameter integer NUM_LANES = 4,            // disjoint-bit scanned-delay lane players
    parameter integer MAX_LANE_EDGES = 64,      // per-lane affine rise/fall edges (LUTRAM)
    parameter integer LANE_SEG_ADDR_WIDTH = 6,  // $clog2(MAX_LANE_EDGES)
    parameter integer CHANNEL_BIT_WIDTH = 6,    // $clog2(CHANNEL_COUNT)
    parameter integer RD_LAT = 2,               // edge-BRAM read latency (forced)
    parameter integer FIFO_DEPTH = 3,           // == RD_LAT + 1 for 1-tick spacing
    parameter integer ARM_SETTLE = 4            // generous one-time arm read settle
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
    // When set, repeat_forever rewinds to loop_start_addr (the steady-state frame of an
    // additive-delay program) instead of edge 0 -- so the real-startup preamble plays
    // ONCE.  The host points loop_start_addr at the steady frame (loop_count is 1, so
    // the finite-bracket rewind is unused and its shadows are reused for free).
    input  wire repeat_from_loop_start,
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
    input  wire [SCAN_COUNT_WIDTH-1:0] bank_chunk0,  // chunk index resident in bank 0
    input  wire [SCAN_COUNT_WIDTH-1:0] bank_chunk1,  // chunk index resident in bank 1
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
    input  wire [BUS_SEL_WIDTH-1:0] bus_prog_stop_value_select,
    input  wire [BUS_COUNT*(BUS_SEG_ADDR_WIDTH+1)-1:0] bus_counts,

    // delay-lane table write port (LUTRAM inside this module; mirrors bus_prog_*).
    // A lane is a 1-bit affine sub-player on a DISJOINT output bit -- a scanned digital
    // DELAY pulled out of the global sorted table so its reordering edges never disturb
    // the main stream.  Structurally the per-bus DAC engine specialised to 1 bit.
    input  wire lane_prog_we,
    input  wire [$clog2(NUM_LANES>1?NUM_LANES:2)-1:0] lane_prog_lane,
    input  wire [LANE_SEG_ADDR_WIDTH-1:0] lane_prog_addr,
    input  wire [TICK_WIDTH-1:0] lane_prog_tick,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] lane_prog_coeffs,
    input  wire lane_prog_value,
    input  wire [NUM_LANES*(LANE_SEG_ADDR_WIDTH+1)-1:0] lane_counts,  // packed per-lane edge count
    input  wire [NUM_LANES*CHANNEL_BIT_WIDTH-1:0] lane_channel_bits,  // packed per-lane output bit
    input  wire [$clog2(NUM_LANES+1)-1:0] lane_count,                 // number of active lanes

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
    // Affine-MAC slot operand width.  The per-slot scan VALUE is multiplied by a
    // 16-bit coeff; narrowing the slot operand to <=25 bits makes each product fit
    // a single DSP48E1 (25x18) instead of two (16x32), which is what lets the
    // engine's affine evaluators fit the 35T DSP budget.  This does NOT shrink the
    // sequence: base_tick stays full TICK_WIDTH (32b) and the coeff still scales
    // the slot, so the resulting tick OFFSET still spans the full 32b range -- only
    // the raw per-slot scan value is bounded to +/-2^24 ticks (~+/-335 ms at 20 ns),
    // far beyond any real scan.  host.image validates slot values against this.
    localparam integer SLOT_MUL_WIDTH = 25;
    localparam integer BANK_BITS = $clog2(BANK_SIZE);
    localparam [1:0] BUS_MODE_RAMP = 2'd2;

    // ----- bus segment tables: LUTRAM (per-tick combinatorial read) -----------
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_start_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_stop_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_start_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_stop_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [1:0] bus_mode_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_SEL_WIDTH-1:0] bus_value_select_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_SEL_WIDTH-1:0] bus_stop_value_select_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_start_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_stop_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];

    // ----- delay-lane tables: LUTRAM (per-tick combinatorial read) -------------
    // The lane PLAY mem is engine-internal distributed RAM (NEVER BRAM -- BRAM would
    // bust the 35T).  Each lane edge: affine base tick + slot coeffs + 1-bit value.
    localparam integer MAX_LANE_ROWS = NUM_LANES * MAX_LANE_EDGES;
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] lane_tick_mem [0:MAX_LANE_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] lane_coeff_mem [0:MAX_LANE_ROWS-1];
    (* ram_style = "distributed" *) reg lane_value_mem [0:MAX_LANE_ROWS-1];

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
    reg [TICK_WIDTH-1:0]  sh_e0_t, sh_e1_t, sh_e2_t, sh_e3_t;
    reg [COEFF_BITS-1:0]  sh_e0_c, sh_e1_c, sh_e2_c, sh_e3_c;
    reg [CHANNEL_COUNT-1:0] sh_e0_m, sh_e1_m, sh_e2_m, sh_e3_m;
    reg [TICK_WIDTH-1:0]  sh_ls0_t, sh_ls1_t, sh_ls2_t, sh_ls3_t;
    reg [COEFF_BITS-1:0]  sh_ls0_c, sh_ls1_c, sh_ls2_c, sh_ls3_c;
    reg [CHANNEL_COUNT-1:0] sh_ls0_m, sh_ls1_m, sh_ls2_m, sh_ls3_m;
    reg [TICK_WIDTH-1:0]  sh_final_t;
    reg [COEFF_BITS-1:0]  sh_final_c;
    reg [SLOT_BITS-1:0] scan_first_values;

    // depth-FIFO_DEPTH edge prefetch
    reg [TICK_WIDTH-1:0] arm_t [0:FIFO_DEPTH-1];
    reg [COEFF_BITS-1:0] arm_c [0:FIFO_DEPTH-1];
    reg [CHANNEL_COUNT-1:0] arm_m [0:FIFO_DEPTH-1];
    reg [2:0] arm_nv;                         // valid arm entries (0..FIFO_DEPTH)
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

    // ----- lane runtime -------------------------------------------------------
    // lane_time (llt) is the lane's OWN frame-tick counter (NOT time_count): reset at
    // every gapless boundary, +1 each output tick.  lane_idx[k]/lane_bit[k] are the
    // 1-bit affine sub-player state; lane_count_active[k]/lane_bit_pos[k] are latched
    // build constants from the host.
    reg [TICK_WIDTH-1:0] lane_time = {TICK_WIDTH{1'b0}};
    reg [LANE_SEG_ADDR_WIDTH:0] lane_idx [0:NUM_LANES-1];
    reg lane_bit [0:NUM_LANES-1];
    reg [LANE_SEG_ADDR_WIDTH:0] lane_count_active [0:NUM_LANES-1];
    reg [CHANNEL_BIT_WIDTH-1:0] lane_bit_pos [0:NUM_LANES-1];
    reg [NUM_LANES-1:0] lane_active = {NUM_LANES{1'b0}};   // lane k carries a program
    integer lane_i;
    reg [(NUM_LANES>1?$clog2(NUM_LANES):1)+LANE_SEG_ADDR_WIDTH-1:0] lane_flat_addr;

    reg lane_prog_we_meta = 1'b0, lane_prog_we_sync = 1'b0, lane_prog_we_prev = 1'b0;

    reg reset_meta = 1'b0, reset_sync = 1'b0;
    reg start_meta = 1'b0, start_sync = 1'b0, start_prev = 1'b0;
    reg bus_prog_we_meta = 1'b0, bus_prog_we_sync = 1'b0, bus_prog_we_prev = 1'b0;
    integer bus_loop;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_prog_flat_addr, bus_runtime_addr;
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_accum_next;

    wire start_event = start_sync && !start_prev;
    wire bus_prog_we_event = bus_prog_we_sync != bus_prog_we_prev;
    wire lane_prog_we_event = lane_prog_we_sync != lane_prog_we_prev;

    // Disjoint-bit merge: each delay lane owns its own output bit, so the lane bits
    // are simply OR'd in over the (lane-cleared) main mask -- no global re-sort.  The
    // lane sub-player state (lane_bit/lane_bit_pos/lane_active) is registered below; the
    // merge itself is combinational (no extra latency vs the registered lane_bit).
    reg [CHANNEL_COUNT-1:0] lane_mask;   // bit b set iff some active lane drives bit b
    reg [CHANNEL_COUNT-1:0] lane_out;    // OR of lane_bit[k] << lane_bit_pos[k]
    integer lane_m;
    always @(*) begin
        lane_mask = {CHANNEL_COUNT{1'b0}};
        lane_out  = {CHANNEL_COUNT{1'b0}};
        for (lane_m = 0; lane_m < NUM_LANES; lane_m = lane_m + 1) begin
            if (lane_active[lane_m]) begin
                lane_mask[lane_bit_pos[lane_m]] = 1'b1;
                lane_out[lane_bit_pos[lane_m]]  = lane_out[lane_bit_pos[lane_m]] | lane_bit[lane_m];
            end
        end
    end
    assign out = (state_mask & ~lane_mask) | lane_out;
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
        reg signed [SLOT_MUL_WIDTH-1:0] slot_value_i;   // low 25b of the slot, signed
        reg signed [ACC_WIDTH-1:0] total;
        begin
            acc = {ACC_WIDTH{1'b0}};
            for (slot_i = 0; slot_i < NUM_SLOTS; slot_i = slot_i + 1) begin
                coeff_i = coeffs[slot_i*COEFF_WIDTH +: COEFF_WIDTH];
                // single-DSP 16x25 product (slot bounded to +/-2^24; see SLOT_MUL_WIDTH)
                slot_value_i = slots[slot_i*TICK_WIDTH +: SLOT_MUL_WIDTH];
                acc = acc + (coeff_i * slot_value_i);
            end
            total = $signed({1'b0, base_tick}) + (acc >>> COEFF_FRAC_BITS);
            zlc_effective_tick = total[TICK_WIDTH-1:0];
        end
    endfunction

    function [2:0] clamp3;                     // min(FIFO_DEPTH, available)
        input [EDGE_ADDR_WIDTH:0] avail;
        begin clamp3 = (avail >= FIFO_DEPTH) ? FIFO_DEPTH[2:0] : avail[2:0]; end
    endfunction

    function [BUS_SEG_ADDR_WIDTH:0] zlc_bus_count_at;
        input integer bus_index;
        begin zlc_bus_count_at = bus_counts[bus_index*(BUS_SEG_ADDR_WIDTH+1) +: (BUS_SEG_ADDR_WIDTH+1)]; end
    endfunction

    // scan-window address for point idx (2 banks, BANK_SIZE pow2)
    function [SCAN_ADDR_WIDTH-1:0] scan_addr_of;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        begin scan_addr_of = {idx[BANK_BITS], idx[BANK_BITS-1:0]}; end   // bank*BANK_SIZE + offset
    endfunction
    function bank_of;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        begin bank_of = idx[BANK_BITS]; end
    endfunction
    function [SCAN_COUNT_WIDTH-1:0] chunk_of;        // which sweep chunk a point belongs to
        input [SCAN_COUNT_WIDTH-1:0] idx;
        begin chunk_of = idx >> BANK_BITS; end
    endfunction
    // bank b is usable for point idx only if it is armed AND actually holds idx's
    // chunk (host writes bank_chunk{0,1} when it loads a chunk).  This is the proven
    // streaming_scan_play handshake -- it makes a late/cyclic refill STALL, never a
    // wrong/stale point, and lets repeat_forever re-sweep a streamed scan.
    function scan_point_resident;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        reg b;
        begin
            b = idx[BANK_BITS];
            scan_point_resident = bank_ready[b] && ((b ? bank_chunk1 : bank_chunk0) == chunk_of(idx));
        end
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

    // ------------------------------------------------------------------ delay lanes
    function [LANE_SEG_ADDR_WIDTH:0] zlc_lane_count_at;
        input integer lane_index;
        begin zlc_lane_count_at = lane_counts[lane_index*(LANE_SEG_ADDR_WIDTH+1) +: (LANE_SEG_ADDR_WIDTH+1)]; end
    endfunction
    function [CHANNEL_BIT_WIDTH-1:0] zlc_lane_bit_at;
        input integer lane_index;
        begin zlc_lane_bit_at = lane_channel_bits[lane_index*CHANNEL_BIT_WIDTH +: CHANNEL_BIT_WIDTH]; end
    endfunction

    task zlc_lane_clear_runtime;
        integer i;
        begin
            for (i = 0; i < NUM_LANES; i = i + 1) begin
                lane_idx[i] <= {(LANE_SEG_ADDR_WIDTH+1){1'b0}};
                lane_bit[i] <= 1'b0;
            end
        end
    endtask

    // Unified lane player (mirrors zlc_bus_tick): reinit==1 reseeds the lane to the
    // boundary lane-time llt_value; reinit==0 advances ONE edge to llt_value (the
    // running step).  Both read exactly ONE edge and call the SHARED zlc_effective_tick
    // MAC ONCE per lane (no unrolled walk -> no replicated multipliers).
    //
    // Why one edge is enough for reinit: the host reseeds lanes ONLY to llt==0 -- every
    // lane program has loop_start_index==0, loop_count==1, repeat_from_index==0
    // (the reordering-delay scan is single-frame; validate ENFORCES this, rejecting an
    // inner bracket / additive preamble on a lane channel).  At llt==0 the lane edges
    // are strictly increasing >= 0, so at most edge 0 can satisfy eff<=0; checking edge 0
    // alone reproduces the model's reinit.  llt rises by 1/cycle and lane edges are >=1
    // tick apart, so the running step also fires at most one edge/cycle.  Byte-identical
    // to engine_model._lane_bits at llt==0 (the no-sim proof: _rtl_lane_realization_play).
    task zlc_lane_tick;
        input reinit;
        input [TICK_WIDTH-1:0] llt_value;
        input [SLOT_BITS-1:0] slot_vec;
        reg [LANE_SEG_ADDR_WIDTH:0] cnt;
        reg [LANE_SEG_ADDR_WIDTH:0] cur;
        reg [TICK_WIDTH-1:0] lane_eff;
        reg lane_due;
        begin
            // Guard on the HELD CTRL scalars (lane_count / lane_counts), not the
            // registered lane_active/lane_count_active, so the start-cycle seed runs
            // correctly even though those registers latch this same cycle (NBA).
            for (lane_i = 0; lane_i < NUM_LANES; lane_i = lane_i + 1) begin
                cnt = zlc_lane_count_at(lane_i);
                if ((lane_i < lane_count) && (cnt != {(LANE_SEG_ADDR_WIDTH+1){1'b0}})) begin
                    // edge under inspection: edge 0 on reinit (llt==0), else the next edge.
                    // ONE effective_tick eval per lane per cycle (the single shared MAC site
                    // counted in host.image.solve_capacity) -- no unrolled walk.
                    cur = reinit ? {(LANE_SEG_ADDR_WIDTH+1){1'b0}} : lane_idx[lane_i];
                    lane_flat_addr = (lane_i * MAX_LANE_EDGES) + cur[LANE_SEG_ADDR_WIDTH-1:0];
                    lane_eff = zlc_effective_tick(lane_tick_mem[lane_flat_addr], lane_coeff_mem[lane_flat_addr], slot_vec);
                    lane_due = (cur < cnt) && (lane_eff <= llt_value);
                    if (reinit) begin
                        // at llt==0 only edge 0 can be due; otherwise the lane idles low
                        lane_bit[lane_i] <= lane_due ? lane_value_mem[lane_flat_addr] : 1'b0;
                        lane_idx[lane_i] <= lane_due ? {{(LANE_SEG_ADDR_WIDTH){1'b0}}, 1'b1}
                                                     : {(LANE_SEG_ADDR_WIDTH+1){1'b0}};
                    end else if (lane_due) begin
                        lane_bit[lane_i] <= lane_value_mem[lane_flat_addr];
                        lane_idx[lane_i] <= cur + 1'b1;
                    end
                end
            end
        end
    endtask

    // Apply a segment given its ALREADY-COMPUTED effective start/stop ticks.  The
    // caller computes tkstart/tkstop once per bus per cycle (zlc_effective_tick is
    // expensive: a 4-slot affine MAC), and shares them with the advance checks, so
    // the whole engine evaluates only ~2 affine ticks per bus per cycle instead of
    // recomputing the same segment 3x in each branch.  Values + cycle timing are
    // identical to recomputing in-line (this is a pure resource dedup).
    task zlc_bus_apply_segment;
        input integer i;
        input [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        input [SLOT_BITS-1:0] slot_vec;
        input [TICK_WIDTH-1:0] tkstart;
        input [TICK_WIDTH-1:0] tkstop;
        reg [TICK_WIDTH-1:0] span;
        reg [BUS_SEL_WIDTH-1:0] start_sel, stop_sel;
        reg [BUS_WIDTH-1:0] vstart, vstop;
        begin
            // Independent start/stop value selects: each endpoint either takes its
            // literal value or reads its own scan slot.  A ramp can therefore go
            // from a scanned start level to a scanned stop level; an edge/hold
            // segment has start_sel == stop_sel so vstart == vstop.
            start_sel = bus_value_select_mem[addr];
            stop_sel  = bus_stop_value_select_mem[addr];
            vstart = (start_sel != {BUS_SEL_WIDTH{1'b0}})
                     ? slot_vec[(start_sel - 1'b1)*TICK_WIDTH +: BUS_WIDTH] : bus_start_value_mem[addr];
            vstop  = (stop_sel  != {BUS_SEL_WIDTH{1'b0}})
                     ? slot_vec[(stop_sel  - 1'b1)*TICK_WIDTH +: BUS_WIDTH] : bus_stop_value_mem[addr];
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

    // Unified bus engine: reinit==1 (re)starts the segment table at seg-0 (was
    // zlc_bus_start_table, used at the 4 gapless boundaries); reinit==0 advances the
    // active segment / steps the ramp (was zlc_bus_step, used every running tick).
    // The two are mutually exclusive each cycle, so MERGING them makes the engine's
    // bus affine multipliers a SINGLE shared set of 2-per-bus (s_eff/e_eff) instead
    // of one set per call site -- the dominant DSP/LUT saving.  Values + cycle timing
    // are byte-identical to the two old tasks (a pure resource dedup).
    task zlc_bus_tick;
        input reinit;
        input [SLOT_BITS-1:0] slot_vec;
        integer i;
        reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        reg [BUS_SEG_ADDR_WIDTH:0] idx, count;
        reg [TICK_WIDTH-1:0] s_eff, e_eff;          // the ONLY bus affine evals: 2 per bus
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                idx  = reinit ? {(BUS_SEG_ADDR_WIDTH+1){1'b0}} : bus_index_active[i];
                addr = (i * MAX_BUS_SEGMENTS) + idx[BUS_SEG_ADDR_WIDTH-1:0];
                s_eff = zlc_effective_tick(bus_start_tick_mem[addr], bus_start_tick_coeff_mem[addr], slot_vec);
                e_eff = zlc_effective_tick(bus_stop_tick_mem[addr],  bus_stop_tick_coeff_mem[addr],  slot_vec);
                if (reinit) begin
                    count = zlc_bus_count_at(i);
                    bus_count_active[i] <= count; bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                    bus_value_active[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_active[i] <= 1'b0; bus_ramp_dir_up[i] <= 1'b0;
                    bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                    bus_ramp_target[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_delta[i] <= {(BUS_WIDTH+1){1'b0}};
                    bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                    if (count != 0 && s_eff == {TICK_WIDTH{1'b0}}) begin
                        zlc_bus_apply_segment(i, addr, slot_vec, s_eff, e_eff);
                        bus_index_active[i] <= {{BUS_SEG_ADDR_WIDTH{1'b0}}, 1'b1};
                    end
                end else if (bus_ramp_active[i]) begin
                    if (time_count >= bus_ramp_stop_tick[i]) begin
                        bus_value_active[i] <= bus_ramp_target[i];
                        bus_ramp_active[i] <= 1'b0;
                        bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                        if (bus_index_active[i] < bus_count_active[i]) begin
                            if (s_eff <= time_count) begin
                                zlc_bus_apply_segment(i, addr, slot_vec, s_eff, e_eff);
                                bus_index_active[i] <= bus_index_active[i] + 1'b1;
                            end
                        end
                    end else if (time_count > bus_ramp_start_tick[i] && bus_ramp_denom[i] != 0) begin
                        bus_accum_next = bus_ramp_accum[i] + bus_ramp_delta[i];
                        if (bus_accum_next >= bus_ramp_denom[i]) begin
                            bus_ramp_accum[i] <= bus_accum_next - bus_ramp_denom[i];
                            if (bus_ramp_dir_up[i]) begin
                                if (bus_value_active[i] < bus_ramp_target[i]) bus_value_active[i] <= bus_value_active[i] + 1'b1;
                            end else begin
                                if (bus_value_active[i] > bus_ramp_target[i]) bus_value_active[i] <= bus_value_active[i] - 1'b1;
                            end
                        end else bus_ramp_accum[i] <= bus_accum_next;
                    end
                end else if (bus_index_active[i] < bus_count_active[i]) begin
                    if (time_count >= s_eff) begin
                        zlc_bus_apply_segment(i, addr, slot_vec, s_eff, e_eff);
                        bus_index_active[i] <= bus_index_active[i] + 1'b1;
                    end
                end
            end
        end
    endtask

    // ---- seed the prefetch from edge-0 shadows for slot vector sv (start/scan/repeat) ----
    // Mirrors engine_model.boundary_to: output edge0 directly iff eff(edge0)==0,
    // then seed FIFO_DEPTH(=3) resident shadows from the first not-yet-output edge.
    task seed_from_edge0;
        input [SLOT_BITS-1:0] sv;
        begin
            pend <= {RD_LAT{1'b0}};
            if (active_count != 0 && zlc_effective_tick(sh_e0_t, sh_e0_c, sv) == {TICK_WIDTH{1'b0}}) begin
                state_mask <= sh_e0_m; time_count <= {{(TICK_WIDTH-1){1'b0}},1'b1};
                edge_index <= {{EDGE_ADDR_WIDTH{1'b0}},1'b1};
                arm_t[0]<=sh_e1_t; arm_c[0]<=sh_e1_c; arm_m[0]<=sh_e1_m;
                arm_t[1]<=sh_e2_t; arm_c[1]<=sh_e2_c; arm_m[1]<=sh_e2_m;
                arm_t[2]<=sh_e3_t; arm_c[2]<=sh_e3_c; arm_m[2]<=sh_e3_m;
                arm_nv <= clamp3(active_count - 1'b1);
                fetch_idx <= {{(EDGE_ADDR_WIDTH-2){1'b0}},3'd4}; edge_raddr <= {{(EDGE_ADDR_WIDTH-2){1'b0}},3'd4};
            end else begin
                state_mask <= {CHANNEL_COUNT{1'b0}}; time_count <= {TICK_WIDTH{1'b0}};
                edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
                arm_t[0]<=sh_e0_t; arm_c[0]<=sh_e0_c; arm_m[0]<=sh_e0_m;
                arm_t[1]<=sh_e1_t; arm_c[1]<=sh_e1_c; arm_m[1]<=sh_e1_m;
                arm_t[2]<=sh_e2_t; arm_c[2]<=sh_e2_c; arm_m[2]<=sh_e2_m;
                arm_nv <= clamp3(active_count);
                fetch_idx <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'd3}; edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}},2'd3};
            end
        end
    endtask

    // ----- one-time ARM sequencer: pre-read 8 edge shadows + final + scan0 -----
    // step 0..7 -> e0,e1,e2,e3,ls0,ls1,ls2,ls3 ; 8 -> final ; 9 -> scan point 0.
    reg [3:0] arm_step;
    reg [3:0] arm_wait;
    reg arm_kicked;

    // combinational temps for the normal prefetch cycle
    reg landed;
    reg do_fire;
    reg issue;
    reg [2:0] nv_after_fire;
    integer k;

    // Boundary work-request flags (blocking, set inside the playback if-else chain,
    // consumed ONCE after it).  This makes the expensive affine tasks -- bus tick,
    // final/loop_end recompute, edge-0 seed -- appear in exactly ONE textual place
    // each (instead of being inlined at all 4 gapless boundaries), so they
    // synthesise to a single shared affine MAC set.  Pure resource dedup: the tasks
    // run on the same cycles with the same slot vectors as before.
    reg bnd_bus_tick;          // run the bus engine this cycle
    reg bnd_bus_reinit;        // ...as a segment-table (re)start (vs. a normal step)
    reg bnd_seed;              // reseed the edge prefetch from edge-0 shadows
    reg bnd_recompute_final;   // recompute final_tick / loop_end_active
    reg [SLOT_BITS-1:0] bnd_slots;
    // delay-lane boundary work (same dispatch as the bus/edge boundary work, so the
    // lane affine MAC is the SAME shared evaluator -- no per-lane DSP).
    reg bnd_lane_tick;         // run the lane players this cycle
    reg bnd_lane_reinit;       // ...as a gapless reseed to bnd_lane_llt (vs. a normal step)
    reg [TICK_WIDTH-1:0] bnd_lane_llt;   // the lane-time (llt) to advance/seed to

    always @(posedge clk) begin
        reset_meta <= reset; reset_sync <= reset_meta;
        start_meta <= start; start_sync <= start_meta; start_prev <= start_sync;
        bus_prog_we_meta <= bus_prog_we; bus_prog_we_sync <= bus_prog_we_meta; bus_prog_we_prev <= bus_prog_we_sync;
        lane_prog_we_meta <= lane_prog_we; lane_prog_we_sync <= lane_prog_we_meta; lane_prog_we_prev <= lane_prog_we_sync;

        if (reset_sync && bus_prog_we_event && bus_prog_bus < BUS_COUNT) begin
            bus_prog_flat_addr = {bus_prog_bus, bus_prog_addr};
            bus_start_tick_mem[bus_prog_flat_addr] <= bus_prog_start_tick;
            bus_stop_tick_mem[bus_prog_flat_addr] <= bus_prog_stop_tick;
            bus_start_value_mem[bus_prog_flat_addr] <= bus_prog_start_value;
            bus_stop_value_mem[bus_prog_flat_addr] <= bus_prog_stop_value;
            bus_mode_mem[bus_prog_flat_addr] <= bus_prog_mode;
            bus_value_select_mem[bus_prog_flat_addr] <= bus_prog_value_select;
            bus_stop_value_select_mem[bus_prog_flat_addr] <= bus_prog_stop_value_select;
            bus_start_tick_coeff_mem[bus_prog_flat_addr] <= bus_prog_start_tick_coeffs;
            bus_stop_tick_coeff_mem[bus_prog_flat_addr] <= bus_prog_stop_tick_coeffs;
        end

        // delay-lane LUTRAM write (toggled prog_we, while reset held -- mirrors bus_prog_*)
        if (reset_sync && lane_prog_we_event && lane_prog_lane < NUM_LANES) begin
            lane_flat_addr = (lane_prog_lane * MAX_LANE_EDGES) + lane_prog_addr;
            lane_tick_mem[lane_flat_addr] <= lane_prog_tick;
            lane_coeff_mem[lane_flat_addr] <= lane_prog_coeffs;
            lane_value_mem[lane_flat_addr] <= lane_prog_value;
        end

        // boundary work-request defaults (consumed once, after the state chain)
        bnd_bus_tick = 1'b0; bnd_bus_reinit = 1'b0; bnd_seed = 1'b0;
        bnd_recompute_final = 1'b0; bnd_slots = slot_active;
        bnd_lane_tick = 1'b0; bnd_lane_reinit = 1'b0; bnd_lane_llt = lane_time;

        if (reset_sync) begin
            running <= 1'b0; done <= 1'b0; underflow <= 1'b0;
            state_mask <= {CHANNEL_COUNT{1'b0}};
            arm_nv <= 3'd0; pend <= {RD_LAT{1'b0}};
            zlc_bus_clear_runtime();
            zlc_lane_clear_runtime();
            lane_time <= {TICK_WIDTH{1'b0}}; lane_active <= {NUM_LANES{1'b0}};
            // --- settle-based ARM read sequence (no timing pressure) ---
            // CONTINUOUSLY re-runs (steps 0..9 then wraps to 0) for as long as reset
            // is held.  Critical for real hardware: the host holds the engine in reset
            // (CMD_SAFE/CMD_LOAD) WHILE it uploads the edge BRAM, then releases reset on
            // CMD_FIRE.  A one-shot arm would latch the shadows from whatever was in the
            // edge BRAM at power-up (empty!) and never re-read the uploaded program.  By
            // looping while reset is held, the shadows always reflect the most-recent
            // (i.e. freshly-uploaded, and stable by the time FIRE releases reset) edge
            // table.  The loop is ~10*ARM_SETTLE cycles (<1 us); the host's
            // upload->LOAD->FIRE gap is milliseconds, so many full loops complete on the
            // final program before reset releases.
            if (!arm_kicked) begin
                arm_kicked <= 1'b1; arm_step <= 4'd0; arm_wait <= ARM_SETTLE[3:0];
                edge_raddr <= {EDGE_ADDR_WIDTH{1'b0}};
            end else if (arm_wait != 0) begin
                arm_wait <= arm_wait - 1'b1;
            end else begin
                case (arm_step)
                    4'd0: begin sh_e0_t<=edge_tick_rdata; sh_e0_c<=edge_coeff_rdata; sh_e0_m<=edge_mask_rdata; edge_raddr<={{(EDGE_ADDR_WIDTH-1){1'b0}},1'b1}; end
                    4'd1: begin sh_e1_t<=edge_tick_rdata; sh_e1_c<=edge_coeff_rdata; sh_e1_m<=edge_mask_rdata; edge_raddr<={{(EDGE_ADDR_WIDTH-2){1'b0}},2'd2}; end
                    4'd2: begin sh_e2_t<=edge_tick_rdata; sh_e2_c<=edge_coeff_rdata; sh_e2_m<=edge_mask_rdata; edge_raddr<={{(EDGE_ADDR_WIDTH-2){1'b0}},2'd3}; end
                    4'd3: begin sh_e3_t<=edge_tick_rdata; sh_e3_c<=edge_coeff_rdata; sh_e3_m<=edge_mask_rdata; edge_raddr<=loop_start_addr; end
                    4'd4: begin sh_ls0_t<=edge_tick_rdata; sh_ls0_c<=edge_coeff_rdata; sh_ls0_m<=edge_mask_rdata; edge_raddr<=loop_start_addr+1'b1; end
                    4'd5: begin sh_ls1_t<=edge_tick_rdata; sh_ls1_c<=edge_coeff_rdata; sh_ls1_m<=edge_mask_rdata; edge_raddr<=loop_start_addr+2'd2; end
                    4'd6: begin sh_ls2_t<=edge_tick_rdata; sh_ls2_c<=edge_coeff_rdata; sh_ls2_m<=edge_mask_rdata; edge_raddr<=loop_start_addr+2'd3; end
                    4'd7: begin sh_ls3_t<=edge_tick_rdata; sh_ls3_c<=edge_coeff_rdata; sh_ls3_m<=edge_mask_rdata;
                                edge_raddr<=(prog_count==0)?{EDGE_ADDR_WIDTH{1'b0}}:(prog_count[EDGE_ADDR_WIDTH-1:0]-1'b1); end
                    4'd8: begin sh_final_t<=edge_tick_rdata; sh_final_c<=edge_coeff_rdata; scan_raddr<={SCAN_ADDR_WIDTH{1'b0}}; end
                    4'd9: begin scan_first_values<=(scan_enable && scan_count!=0)?scan_rdata:{SLOT_BITS{1'b0}}; end
                    default: ;
                endcase
                if (arm_step < 4'd9) begin arm_step <= arm_step + 1'b1; arm_wait <= ARM_SETTLE[3:0]; end
                else begin   // wrap: re-arm continuously while reset is held (see above)
                    arm_step <= 4'd0; edge_raddr <= {EDGE_ADDR_WIDTH{1'b0}}; arm_wait <= ARM_SETTLE[3:0];
                end
            end
        end else if (start_event && !running) begin
            running <= (prog_count != 0); done <= (prog_count == 0); underflow <= 1'b0;
            active_count <= prog_count; repeat_forever_active <= repeat_forever;
            scan_enable_active <= scan_enable && scan_count != 0; active_scan_count <= scan_count;
            slot_active <= scan_first_values; scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}};
            scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
            scan_raddr <= scan_addr_of({{(SCAN_COUNT_WIDTH-1){1'b0}},1'b1});      // pre-read point 1
            loop_count_active <= (loop_count==0)?32'd1:loop_count;
            loops_remaining <= (loop_count==0)?32'd1:loop_count;
            // latch the per-lane build constants (count / output channel_bit / active)
            // from the held CTRL scalars; seed each lane to llt=0 below.
            for (lane_i = 0; lane_i < NUM_LANES; lane_i = lane_i + 1) begin
                lane_count_active[lane_i] <= zlc_lane_count_at(lane_i);
                lane_bit_pos[lane_i] <= zlc_lane_bit_at(lane_i);
                lane_active[lane_i] <= (lane_i < lane_count) && (zlc_lane_count_at(lane_i) != {(LANE_SEG_ADDR_WIDTH+1){1'b0}});
            end
            lane_time <= {TICK_WIDTH{1'b0}};
            // heavy affine work (final/loop_end recompute, bus (re)start, edge-0 seed)
            // is dispatched ONCE after the chain via these flags (see SLOT_MUL_WIDTH /
            // bnd_* notes): same cycle, same slots -> identical behavior, far fewer MACs.
            bnd_slots = scan_first_values; bnd_recompute_final = 1'b1;
            bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_seed = 1'b1;
            bnd_lane_tick = 1'b1; bnd_lane_reinit = 1'b1; bnd_lane_llt = {TICK_WIDTH{1'b0}};
        end else if (running) begin
            landed = pend[RD_LAT-1];
            if (loop_count_active>32'd1 && loops_remaining>32'd1 && time_count>=loop_end_active) begin
                // loop rewind: output loop_start mask, seed arm from loop_start+1
                state_mask <= sh_ls0_m; time_count <= zlc_effective_tick(sh_ls0_t,sh_ls0_c,slot_active)+1'b1;
                edge_index <= {1'b0,loop_start_addr}+1'b1; loops_remaining <= loops_remaining-1'b1;
                arm_t[0]<=sh_ls1_t; arm_c[0]<=sh_ls1_c; arm_m[0]<=sh_ls1_m;
                arm_t[1]<=sh_ls2_t; arm_c[1]<=sh_ls2_c; arm_m[1]<=sh_ls2_m;
                arm_t[2]<=sh_ls3_t; arm_c[2]<=sh_ls3_c; arm_m[2]<=sh_ls3_m;
                arm_nv <= clamp3(active_count - ({1'b0,loop_start_addr}+1'b1));
                fetch_idx <= {1'b0,loop_start_addr}+3'd4; edge_raddr <= loop_start_addr+3'd4;
                pend <= {RD_LAT{1'b0}};
                bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_slots = slot_active;  // re(start) bus, keep slots
                // lane loop-rewind: llt = eff(loop_start), reseed (engine_model loop branch)
                bnd_lane_tick = 1'b1; bnd_lane_reinit = 1'b1;
                lane_time <= zlc_effective_tick(sh_ls0_t,sh_ls0_c,slot_active);
                bnd_lane_llt = zlc_effective_tick(sh_ls0_t,sh_ls0_c,slot_active);
            end else if (time_count >= final_tick) begin
                if (scan_enable_active && (scan_point_index+1'b1) < active_scan_count) begin
                    if (!scan_point_resident(scan_point_index+1'b1)) begin
                        underflow <= 1'b1;          // STALL: next chunk not (yet) resident
                    end else begin
                        underflow <= 1'b0;
                        scan_point_index <= scan_point_index+1'b1;
                        scan_cursor <= scan_point_index+1'b1;
                        scan_raddr <= scan_addr_of(scan_point_index+2'd2);   // pre-read following point
                        slot_active <= scan_rdata;
                        loops_remaining <= loop_count_active;
                        bnd_slots = scan_rdata; bnd_recompute_final = 1'b1;
                        bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_seed = 1'b1;
                        // lane scan-advance: llt = 0, reseed with the new slot vector
                        bnd_lane_tick = 1'b1; bnd_lane_reinit = 1'b1; bnd_lane_llt = {TICK_WIDTH{1'b0}};
                        lane_time <= {TICK_WIDTH{1'b0}};
                    end
                end else if (repeat_forever_active) begin
                    if (repeat_from_loop_start && !scan_enable_active) begin
                        // ADDITIVE-DELAY repeat: rewind to the STEADY frame (loop_start
                        // shadows), NOT edge 0, so the real-startup preamble plays once.
                        // Same gapless reseed as the finite-bracket rewind, but
                        // loops_remaining is untouched (this repeat is infinite).
                        underflow <= 1'b0;
                        state_mask <= sh_ls0_m; time_count <= zlc_effective_tick(sh_ls0_t,sh_ls0_c,slot_active)+1'b1;
                        edge_index <= {1'b0,loop_start_addr}+1'b1;
                        arm_t[0]<=sh_ls1_t; arm_c[0]<=sh_ls1_c; arm_m[0]<=sh_ls1_m;
                        arm_t[1]<=sh_ls2_t; arm_c[1]<=sh_ls2_c; arm_m[1]<=sh_ls2_m;
                        arm_t[2]<=sh_ls3_t; arm_c[2]<=sh_ls3_c; arm_m[2]<=sh_ls3_m;
                        arm_nv <= clamp3(active_count - ({1'b0,loop_start_addr}+1'b1));
                        fetch_idx <= {1'b0,loop_start_addr}+3'd4; edge_raddr <= loop_start_addr+3'd4;
                        pend <= {RD_LAT{1'b0}};
                        bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_slots = slot_active;
                        // lane additive-delay repeat: llt = eff(loop_start), reseed
                        bnd_lane_tick = 1'b1; bnd_lane_reinit = 1'b1;
                        lane_time <= zlc_effective_tick(sh_ls0_t,sh_ls0_c,slot_active);
                        bnd_lane_llt = zlc_effective_tick(sh_ls0_t,sh_ls0_c,slot_active);
                    // Otherwise restart the sweep at point 0.  For a STREAMING scan the
                    // host has overwritten bank 0 with a later chunk, so wait until it
                    // reloads chunk 0 (bank_chunk0==0) before wrapping -- the re-sweep is
                    // then seamless (resident scans pass instantly; streamed ones stall
                    // only at the seam, never emit a wrong point).
                    end else if (scan_enable_active && !(bank_ready[1'b0] && bank_chunk0 == {SCAN_COUNT_WIDTH{1'b0}})) begin
                        underflow <= 1'b1;
                    end else begin
                        underflow <= 1'b0;
                        slot_active <= scan_first_values; scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}}; scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
                        scan_raddr <= scan_addr_of({{(SCAN_COUNT_WIDTH-1){1'b0}},1'b1});
                        loops_remaining <= loop_count_active;
                        bnd_slots = scan_first_values; bnd_recompute_final = 1'b1;
                        bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_seed = 1'b1;
                        // lane re-sweep / repeat-from-0: llt = 0, reseed with point-0 slots
                        bnd_lane_tick = 1'b1; bnd_lane_reinit = 1'b1; bnd_lane_llt = {TICK_WIDTH{1'b0}};
                        lane_time <= {TICK_WIDTH{1'b0}};
                    end
                end else begin
                    running <= 1'b0; done <= 1'b1; state_mask <= {CHANNEL_COUNT{1'b0}}; zlc_bus_clear_runtime();
                end
            end else begin
                bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b0; bnd_slots = slot_active;  // normal bus step
                // lane normal step: advance lane_time by 1 and step the lane players
                // (engine_model: out at this tick, then llt += 1 and step for next tick).
                bnd_lane_tick = 1'b1; bnd_lane_reinit = 1'b0;
                lane_time <= lane_time + 1'b1; bnd_lane_llt = lane_time + 1'b1;
                do_fire = (edge_index < active_count) && (arm_nv != 0) && (time_count == zlc_effective_tick(arm_t[0],arm_c[0],slot_active));
                if (do_fire) begin
                    state_mask <= arm_m[0];
                    edge_index <= edge_index + 1'b1;
                end
                time_count <= time_count + 1'b1;
                // --- FIFO update: shift down on fire, append landed at tail ---
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
                // --- issue a read iff resident + still-in-flight is below depth ---
                // resident-after = nv_after_fire + landed ; still-in-flight (RD_LAT=2) = pend[0].
                issue = ((nv_after_fire + (landed ? 1'b1 : 1'b0) + pend[0]) < FIFO_DEPTH[2:0])
                        && (fetch_idx < active_count);
                if (issue) begin edge_raddr <= fetch_idx; fetch_idx <= fetch_idx + 1'b1; end
                pend <= {pend[RD_LAT-2:0], issue};
            end
        end

        // ---- dispatch the boundary's heavy affine work ONCE (a single shared MAC
        // set), driven by the flags set in the chain above.  Same cycle + same slot
        // vector as the old in-line calls -> behavior is byte-identical, but the
        // affine evaluators are no longer replicated at every boundary site.
        if (bnd_recompute_final) begin
            final_tick <= zlc_effective_tick(sh_final_t, sh_final_c, bnd_slots);
            loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, bnd_slots);
        end
        if (bnd_bus_tick) zlc_bus_tick(bnd_bus_reinit, bnd_slots);
        if (bnd_seed) seed_from_edge0(bnd_slots);
        // dispatch the lane players once, on the SAME shared affine MAC + same slot
        // vector as the rest of the boundary work (no per-lane DSP).
        if (bnd_lane_tick) zlc_lane_tick(bnd_lane_reinit, bnd_lane_llt, bnd_slots);
    end

    initial begin arm_kicked = 1'b0; arm_step = 4'd0; arm_wait = 4'd0; end
endmodule
