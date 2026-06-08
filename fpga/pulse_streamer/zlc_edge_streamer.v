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
    parameter integer NUM_DELAYS = 8,           // per-channel OUTPUT delay players
    parameter integer MAX_DELAY_INTERVALS = 8,  // ON intervals stored per delayed channel (LUTRAM)
    parameter integer SKIP_WIDTH = 32,          // whole-period skip count = floor(d/T) (unbounded delay)
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

    // PHYSICAL per-channel OUTPUT DELAY -- UNBOUNDED membership player (NO buffer of any kind).
    // A channel delay is NOT baked into the edges -- it is applied to the engine OUTPUT:
    // out_delayed[t] = out_undelayed[t - d], 0 before fire.  d = skip*T + off, where
    // T = the (gapless) frame period.  Instead of BUFFERING the signal, each delayed channel
    // stores its OWN undelayed ON intervals [a_i, b_i) over [0, T) in a tiny per-channel
    // LUTRAM and EVALUATES membership at the shifted phase ``shifted = (time_count - off) mod
    // T`` -- no buffer, so T (and the delay) are UNBOUNDED (off is now full TICK_WIDTH, the
    // whole phase).  skip = floor(d/T) whole periods are suppressed by a startup counter (so
    // the delay LENGTH is unbounded too), and the first frame is REAL (silent until t == d).
    // Proven == engine_model.membership_delay_play == phase_offset_play == delay_line_reference
    // for ANY off, ANY T, ANY d (arbitrarily large/small), zero, and -- via the host's folded
    // global shift -- negative.
    //   * delay_count : number of delayed channels (<= NUM_DELAYS)
    //   * delay_bits  : output channel bit per delayed channel
    //   * delay_off   : off = d mod T (full TICK_WIDTH -- the phase shift; per scan point
    //                   when T is scanned, latched from scan_rdata; else held CTRL)
    //   * delay_skip  : skip = floor(d/T) whole periods (per scan point / held CTRL)
    //   * delay_iv_counts : number of ON intervals per delayed channel
    // The ON-interval table (a_i, b_i + affine coeffs) is written into LUTRAM via the
    // delay_prog_* port below (a prog_we toggle, exactly like the bus segment loader).
    input  wire [$clog2(NUM_DELAYS+1)-1:0] delay_count,            // number of delayed channels
    input  wire [NUM_DELAYS*CHANNEL_BIT_WIDTH-1:0] delay_bits,     // output bit per delayed channel
    input  wire [NUM_DELAYS*TICK_WIDTH-1:0] delay_off,             // off = d mod T (phase shift)
    input  wire [NUM_DELAYS*SKIP_WIDTH-1:0] delay_skip,            // skip = floor(d/T) (whole periods)
    input  wire [NUM_DELAYS*(($clog2(MAX_DELAY_INTERVALS)>0?$clog2(MAX_DELAY_INTERVALS):1)+1)-1:0] delay_iv_counts,

    // per-channel ON-interval table write port (LUTRAM inside this module, like the bus
    // tables).  A toggle on delay_prog_we commits one [a_i, b_i) interval (with affine
    // tick coeffs so a scanned DURATION moves the interval) for player delay_prog_player.
    input  wire delay_prog_we,
    input  wire [($clog2(NUM_DELAYS)>0?$clog2(NUM_DELAYS):1)-1:0] delay_prog_player,
    input  wire [($clog2(MAX_DELAY_INTERVALS)>0?$clog2(MAX_DELAY_INTERVALS):1)-1:0] delay_prog_addr,
    input  wire [TICK_WIDTH-1:0] delay_prog_start_tick,
    input  wire [TICK_WIDTH-1:0] delay_prog_stop_tick,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] delay_prog_start_coeffs,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] delay_prog_stop_coeffs,

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
    // per-channel ON-interval LUTRAM geometry (the UNBOUNDED delay player tables)
    localparam integer DELAY_IV_AW = ($clog2(MAX_DELAY_INTERVALS) > 0) ? $clog2(MAX_DELAY_INTERVALS) : 1;
    localparam integer DELAY_IV_CNT_W = DELAY_IV_AW + 1;       // holds 0..MAX_DELAY_INTERVALS
    localparam integer DELAY_PLAYER_AW = ($clog2(NUM_DELAYS) > 0) ? $clog2(NUM_DELAYS) : 1;
    localparam integer MAX_DELAY_IV_ROWS = NUM_DELAYS * MAX_DELAY_INTERVALS;

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

    // ----- per-channel OUTPUT DELAY runtime (UNBOUNDED membership player) ------
    // One delay player per delayed channel (k = 0..NUM_DELAYS-1).  There is NO buffer of any
    // kind.  Each player stores its channel's OWN undelayed ON intervals [a_i, b_i) over [0, T)
    // in a tiny LUTRAM (del_iv_*_mem, indexed k*MAX_DELAY_INTERVALS + i, with affine tick coeffs
    // so a scanned DURATION moves the interval).  The delayed bit is produced COMBINATIONALLY
    // each tick by EVALUATING membership at the shifted phase ``shifted = (time_count - off) mod
    // T`` (off < T < 2^TICK_WIDTH -> the modulo is a single conditional +T; NO buffer -> T and
    // the delay are UNBOUNDED).  A combinational startup gate ((frame_idx, time_count) vs
    // (skip, off)) suppresses the output until exactly t == d, so the first frame is REAL
    // (silent, never a wrapped-in tail).
    //   * del_active[k]    : player k carries a delayed channel
    //   * del_bit_pos[k]   : its output channel bit
    //   * del_off[k]       : off = d mod T (full TICK_WIDTH phase shift -- the whole phase)
    //   * del_iv_count[k]  : number of resolved ON intervals for this channel
    //   * del_skip[k]      : skip = floor(d/T) -- the TARGET frame index at which the gate opens
    //   * del_frame_idx[k] : frames elapsed since FIRE (increments at each gapless seam).  The
    //                        startup gate is purely COMBINATIONAL from (del_frame_idx, time_count):
    //                        started = (frame_idx > skip) || (frame_idx == skip && time_count >= off)
    //                        == ``t >= d`` with NO decrementing counter and NO off-by-one.
    reg [NUM_DELAYS-1:0] del_active = {NUM_DELAYS{1'b0}};
    reg [CHANNEL_BIT_WIDTH-1:0] del_bit_pos [0:NUM_DELAYS-1];
    reg [TICK_WIDTH-1:0] del_off [0:NUM_DELAYS-1];
    reg [DELAY_IV_CNT_W-1:0] del_iv_count [0:NUM_DELAYS-1];
    reg [SKIP_WIDTH-1:0] del_skip [0:NUM_DELAYS-1];
    reg [SKIP_WIDTH-1:0] del_frame_idx [0:NUM_DELAYS-1];
    // per-channel ON-interval table: a_i, b_i + affine tick coeffs (LUTRAM, async read).
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] del_iv_start_mem [0:MAX_DELAY_IV_ROWS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] del_iv_stop_mem  [0:MAX_DELAY_IV_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] del_iv_start_coeff_mem [0:MAX_DELAY_IV_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] del_iv_stop_coeff_mem  [0:MAX_DELAY_IV_ROWS-1];
    integer del_i;

    reg reset_meta = 1'b0, reset_sync = 1'b0;
    reg start_meta = 1'b0, start_sync = 1'b0, start_prev = 1'b0;
    reg bus_prog_we_meta = 1'b0, bus_prog_we_sync = 1'b0, bus_prog_we_prev = 1'b0;
    reg delay_prog_we_meta = 1'b0, delay_prog_we_sync = 1'b0, delay_prog_we_prev = 1'b0;
    integer bus_loop;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_prog_flat_addr, bus_runtime_addr;
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_accum_next;

    wire start_event = start_sync && !start_prev;
    wire bus_prog_we_event = bus_prog_we_sync != bus_prog_we_prev;
    wire delay_prog_we_event = delay_prog_we_sync != delay_prog_we_prev;

    // ---- per-delayed-channel interval-count input slice ----
    function [DELAY_IV_CNT_W-1:0] zlc_delay_iv_count_at;
        input integer k;
        begin zlc_delay_iv_count_at = delay_iv_counts[k*DELAY_IV_CNT_W +: DELAY_IV_CNT_W]; end
    endfunction
    function [TICK_WIDTH-1:0] zlc_delay_off_at;
        input integer k;
        begin zlc_delay_off_at = delay_off[k*TICK_WIDTH +: TICK_WIDTH]; end
    endfunction
    function [SKIP_WIDTH-1:0] zlc_delay_skip_at;
        input integer k;
        begin zlc_delay_skip_at = delay_skip[k*SKIP_WIDTH +: SKIP_WIDTH]; end
    endfunction
    function [CHANNEL_BIT_WIDTH-1:0] zlc_delay_bit_at;
        input integer k;
        begin zlc_delay_bit_at = delay_bits[k*CHANNEL_BIT_WIDTH +: CHANNEL_BIT_WIDTH]; end
    endfunction

    // Per-channel OUTPUT-delay merge (combinational, NO buffer).  delayed_mask[b] marks the
    // bits an active delay player owns (cleared from the undelayed state_mask); delayed_out[b]
    // is that player's delayed value, found by evaluating its stored ON intervals at the shifted
    // phase.  The "started" gate is the COMBINATIONAL frame-index compare
    //   del_started_eff = (del_frame_idx > del_skip) || (del_frame_idx == del_skip && time_count >= off)
    // == ``t >= d`` -- no decrementing counter, no off-by-one, opens at EXACTLY t == d.
    //   shifted = time_count - off ; if (shifted[TICK_WIDTH-1]) shifted += loop_end_active (T)
    //   bit = OR_i (a_i <= shifted < b_i)  over the resolved (affine) intervals
    // loop_end_active holds the active per-point frame T (recomputed every boundary), so the
    // modulo uses the SAME T the host used for off = d mod T.
    reg [CHANNEL_COUNT-1:0] delayed_mask;   // bit b set iff some active delay drives bit b
    reg [CHANNEL_COUNT-1:0] delayed_out;    // delayed value per owned bit
    reg del_started_eff;
    reg del_member;
    reg signed [TICK_WIDTH:0] del_shifted;  // time_count - off (one extra bit for the sign)
    reg [TICK_WIDTH-1:0] del_phase;
    reg [TICK_WIDTH-1:0] del_iv_a, del_iv_b;
    integer del_m, del_j, del_row;
    always @(*) begin
        delayed_mask = {CHANNEL_COUNT{1'b0}};
        delayed_out  = {CHANNEL_COUNT{1'b0}};
        for (del_m = 0; del_m < NUM_DELAYS; del_m = del_m + 1) begin
            if (del_active[del_m]) begin
                delayed_mask[del_bit_pos[del_m]] = 1'b1;
                // startup gate (combinational, NO counter): out is silent until exactly t == d
                //   started = (frame_idx > skip) || (frame_idx == skip && time_count >= off)
                // i.e. ``t >= d = skip*T + off`` expressed from the bounded (frame_idx, time_count).
                del_started_eff = (del_frame_idx[del_m] > del_skip[del_m])
                                  || ((del_frame_idx[del_m] == del_skip[del_m])
                                      && (time_count >= del_off[del_m]));
                // shifted = (time_count - off) mod T (T = loop_end_active); single +T fixes sign.
                del_shifted = $signed({1'b0, time_count}) - $signed({1'b0, del_off[del_m]});
                if (del_shifted < 0)
                    del_phase = del_shifted[TICK_WIDTH-1:0] + loop_end_active;
                else
                    del_phase = del_shifted[TICK_WIDTH-1:0];
                // membership: OR over this player's resolved ON intervals (affine in slots)
                del_member = 1'b0;
                for (del_j = 0; del_j < MAX_DELAY_INTERVALS; del_j = del_j + 1) begin
                    if (del_j < del_iv_count[del_m]) begin
                        del_row = del_m * MAX_DELAY_INTERVALS + del_j;
                        del_iv_a = zlc_effective_tick(del_iv_start_mem[del_row], del_iv_start_coeff_mem[del_row], slot_active);
                        del_iv_b = zlc_effective_tick(del_iv_stop_mem[del_row],  del_iv_stop_coeff_mem[del_row],  slot_active);
                        if ((del_phase >= del_iv_a) && (del_phase < del_iv_b))
                            del_member = 1'b1;
                    end
                end
                delayed_out[del_bit_pos[del_m]] =
                    delayed_out[del_bit_pos[del_m]] | (del_started_eff ? del_member : 1'b0);
            end
        end
    end
    assign out = (state_mask & ~delayed_mask) | delayed_out;
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

    // Clear the delay-player runtime (used on reset).  The ON-interval LUTRAM is written
    // separately by the delay_prog_* loader (host fills it while reset is held), so this
    // only clears the small scalar player state.
    task zlc_delay_clear_runtime;
        integer i;
        begin
            del_active <= {NUM_DELAYS{1'b0}};
            for (i = 0; i < NUM_DELAYS; i = i + 1) begin
                del_bit_pos[i] <= {CHANNEL_BIT_WIDTH{1'b0}};
                del_off[i] <= {TICK_WIDTH{1'b0}};
                del_iv_count[i] <= {DELAY_IV_CNT_W{1'b0}};
                del_skip[i] <= {SKIP_WIDTH{1'b0}};
                del_frame_idx[i] <= {SKIP_WIDTH{1'b0}};
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
    // Per-channel OUTPUT-delay boundary work (set in the playback if-else chain, consumed
    // once after it).  bnd_delay_step marks a RUNNING tick; bnd_delay_boundary additionally
    // flags a FRAME boundary (the gapless reseeds: loop-rewind, scan-advance, repeat) on which
    // del_frame_idx increments.  There is no buffer to maintain across the seam -- the delayed
    // bit is re-evaluated from the stored intervals each tick.
    reg bnd_delay_step;        // a running output tick happened
    reg bnd_delay_boundary;    // ...and it was a frame boundary: increment del_frame_idx
    reg [DELAY_PLAYER_AW+DELAY_IV_AW-1:0] delay_prog_flat_addr;

    always @(posedge clk) begin
        reset_meta <= reset; reset_sync <= reset_meta;
        start_meta <= start; start_sync <= start_meta; start_prev <= start_sync;
        bus_prog_we_meta <= bus_prog_we; bus_prog_we_sync <= bus_prog_we_meta; bus_prog_we_prev <= bus_prog_we_sync;
        delay_prog_we_meta <= delay_prog_we; delay_prog_we_sync <= delay_prog_we_meta; delay_prog_we_prev <= delay_prog_we_sync;

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

        // Per-channel delay ON-interval LUTRAM loader (same prog_we-toggle pattern as the bus
        // tables): the host writes each [a_i, b_i) interval (+ affine tick coeffs) for player
        // delay_prog_player while reset is held, before FIRE.
        if (reset_sync && delay_prog_we_event && delay_prog_player < NUM_DELAYS) begin
            delay_prog_flat_addr = (delay_prog_player * MAX_DELAY_INTERVALS) + delay_prog_addr;
            del_iv_start_mem[delay_prog_flat_addr] <= delay_prog_start_tick;
            del_iv_stop_mem[delay_prog_flat_addr]  <= delay_prog_stop_tick;
            del_iv_start_coeff_mem[delay_prog_flat_addr] <= delay_prog_start_coeffs;
            del_iv_stop_coeff_mem[delay_prog_flat_addr]  <= delay_prog_stop_coeffs;
        end

        // boundary work-request defaults (consumed once, after the state chain)
        bnd_bus_tick = 1'b0; bnd_bus_reinit = 1'b0; bnd_seed = 1'b0;
        bnd_recompute_final = 1'b0; bnd_slots = slot_active;
        bnd_delay_step = 1'b0; bnd_delay_boundary = 1'b0;

        if (reset_sync) begin
            running <= 1'b0; done <= 1'b0; underflow <= 1'b0;
            state_mask <= {CHANNEL_COUNT{1'b0}};
            arm_nv <= 3'd0; pend <= {RD_LAT{1'b0}};
            zlc_bus_clear_runtime();
            zlc_delay_clear_runtime();
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
            // AT FIRE (step 7): seed each per-channel OUTPUT-delay player from the held CTRL
            // scalars (off/skip/iv_count/bit) and zero its frame index.  The startup gate is
            // purely combinational from (del_frame_idx, time_count) vs (del_skip, del_off), so
            // there is no counter to seed and NO buffer to clear -- the ON intervals live in
            // del_iv_*_mem (filled by the host before FIRE) and are EVALUATED at the shifted
            // phase, so the first frame is REAL (silent until t == d) from the gate alone.
            for (del_i = 0; del_i < NUM_DELAYS; del_i = del_i + 1) begin
                del_active[del_i]    <= (del_i < delay_count);
                del_bit_pos[del_i]   <= zlc_delay_bit_at(del_i);
                del_off[del_i]       <= zlc_delay_off_at(del_i);
                del_iv_count[del_i]  <= zlc_delay_iv_count_at(del_i);
                del_skip[del_i]      <= zlc_delay_skip_at(del_i);
                del_frame_idx[del_i] <= {SKIP_WIDTH{1'b0}};
            end
            // heavy affine work (final/loop_end recompute, bus (re)start, edge-0 seed)
            // is dispatched ONCE after the chain via these flags (see SLOT_MUL_WIDTH /
            // bnd_* notes): same cycle, same slots -> identical behavior, far fewer MACs.
            bnd_slots = scan_first_values; bnd_recompute_final = 1'b1;
            bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_seed = 1'b1;
        end else if (running) begin
            landed = pend[RD_LAT-1];
            // every RUNNING output tick steps the delay players' startup gate (no buffer to
            // push -- the delayed bit is re-evaluated from the stored intervals).  The
            // frame-boundary seams below additionally raise bnd_delay_boundary to decrement
            // del_skip_cnt; a normal step / stall-hold leaves it 0 (no skip decrement).
            bnd_delay_step = 1'b1;
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
                // delay players: frame boundary (the gapless loop-rewind seam) -- increment
                // del_frame_idx at this seam (the startup gate uses it; no buffer to maintain).
                bnd_delay_boundary = 1'b1;
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
                        // delay players: frame boundary (scan-advance seam) -- decrement skip.
                        bnd_delay_boundary = 1'b1;
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
                        // delay players: frame boundary (additive-delay repeat seam -- the
                        // steady frame rewind) -- decrement skip.
                        bnd_delay_boundary = 1'b1;
                    // Otherwise restart the sweep at point 0.  For a STREAMING scan the
                    // host has overwritten bank 0 with a later chunk, so wait until it
                    // reloads chunk 0 (bank_chunk0==0) before wrapping -- the re-sweep is
                    // then seamless (resident scans pass instantly; streamed ones stall
                    // only at the seam, never emit a wrong point).
                    end else if (scan_enable_active && !(bank_ready[1'b0] && bank_chunk0 == {SCAN_COUNT_WIDTH{1'b0}})) begin
                        // STREAMED re-sweep seam: bank 0 still holds a later chunk, so we must
                        // wait for the host to reload chunk 0.  Publish scan_cursor = N (the full
                        // count) so the host's refill loop (which reloads chunk 0 only when
                        // CURSOR >= N) actually fires -- otherwise the cursor would stay at N-1,
                        // the host would never reload, and the engine would stall here forever
                        // (the scan stops after exactly one sweep).
                        underflow <= 1'b1;
                        scan_cursor <= active_scan_count;
                    end else begin
                        underflow <= 1'b0;
                        slot_active <= scan_first_values; scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}}; scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
                        scan_raddr <= scan_addr_of({{(SCAN_COUNT_WIDTH-1){1'b0}},1'b1});
                        loops_remaining <= loop_count_active;
                        bnd_slots = scan_first_values; bnd_recompute_final = 1'b1;
                        bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_seed = 1'b1;
                        // delay players: frame boundary (re-sweep / repeat-from-0 seam).
                        bnd_delay_boundary = 1'b1;
                    end
                end else begin
                    running <= 1'b0; done <= 1'b1; state_mask <= {CHANNEL_COUNT{1'b0}}; zlc_bus_clear_runtime();
                end
            end else begin
                bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b0; bnd_slots = slot_active;  // normal bus step
                // delay players: a normal running step -- bnd_delay_step is already set
                // (default for the running branch); bnd_delay_boundary stays 0 (no seam).
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
        // ---- per-channel OUTPUT DELAY frame-index advance (one frame seam) ------------
        // There is NO buffer and NO startup counter -- the delayed bit is produced purely by
        // EVALUATING the stored ON intervals at the shifted phase, and the startup gate is the
        // combinational compare (frame_idx, time_count) vs (skip, off) in the merge above.  So
        // the ONLY registered delay state to maintain is del_frame_idx: increment it at every
        // gapless frame seam (bnd_delay_boundary), so frame_idx == frames elapsed since fire.
        // (bnd_delay_step alone, a normal running tick, leaves frame_idx unchanged.)
        if (bnd_delay_step && bnd_delay_boundary) begin
            for (del_i = 0; del_i < NUM_DELAYS; del_i = del_i + 1) begin
                if (del_active[del_i])
                    del_frame_idx[del_i] <= del_frame_idx[del_i] + 1'b1;
            end
        end
    end

    initial begin arm_kicked = 1'b0; arm_step = 4'd0; arm_wait = 4'd0; end
endmodule
