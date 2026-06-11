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
//
// OUTPUT DELAY -- a LITERAL delay line:
//   * TTL: a per-channel EVENT SCHEDULER -- when the undelayed bit toggles at tick t the
//     engine pushes {t + d_ch - 1, level} into that channel's small event FIFO and pops it
//     against a free-running global counter, so out_delayed[t] = out_undelayed[t-d] with the
//     storage scaling in TOGGLES IN FLIGHT (<= EVT_DEPTH, host-validated), not delay length:
//     delays up to TTL_DELAY_WIDTH (32b, ~85.9 s) at ~2k LUTs for all 62 channels (the old
//     per-tick SRL lines cost ~4.1k LUTs and capped d at DELAY_DEPTH ~41 us).
//   * DAC: ONE BUS_WIDTH(10)-wide ring per bus (a 2D word array Vivado DOES infer as 3D RAM; one
//     delay shared by all 10 bits), read at (del_wptr - d_bus).
//   d=0 is exact passthrough (the non-delayed bits/buses bypass the line entirely).  The
//   delay is BOUNDED to DELAY_DEPTH ticks (~40 us @ 20 ns; the host validates d <= DELAY_DEPTH).
//   Proven cycle-exact by engine_model.rtl_delay_line_mirror / rtl_bus_delay_line_mirror.
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
    // IDLE/SAFE DAC code.  The DAC driver is bipolar OFFSET-BINARY: code 0 = NEGATIVE full
    // scale, code 2^(B-1) (=512 for 10 bits) = true 0 V.  Every "rest" value of a bus --
    // power-up, reset/CMD_SAFE, FIRE re-init, the delayed-read gate before the ring fills,
    // and after done -- uses THIS mid-scale code so an idle DAC outputs 0 V, not -FS.
    parameter integer BUS_SAFE_VALUE = (1 << (BUS_WIDTH - 1)),
    parameter integer DELAY_DEPTH = 2048,       // delay-line buffer depth in ticks (~40us @ 20ns);
                                                // bounded cap, covers +/-15us after the global shift G
    parameter integer RD_LAT = 2,               // edge-BRAM read latency (forced)
    // The PREFETCH pipeline from `issue` to data-valid is RD_LAT + 1 (the extra cycle is
    // the registered `edge_raddr`: an issued read only reaches the BRAM address port the
    // NEXT cycle, then the BRAM adds RD_LAT).  For sustained 1-tick (20 ns) playback the
    // FIFO must hold a resident head PLUS one in-flight read per pipeline stage, i.e.
    // FIFO_DEPTH = (RD_LAT+1) + 1 = RD_LAT + 2.  (The earlier value RD_LAT+1 under-counted
    // the edge_raddr stage: `landed` fired a cycle early, the append latched a stale bus
    // word, and a streamed edge was dropped -- the emCCD "40 ms / e7 vanished" bug.)
    parameter integer FIFO_DEPTH = RD_LAT + 2,
    parameter integer ARM_SETTLE = 4,           // generous one-time arm read settle
    // delay-tick field width; DECLARED HERE (not as a body localparam) so the port
    // declarations below can use it.  A body localparam is referenced-before-declaration,
    // which the Vivado synth frontend only warns about but STRICT tools (xsim, other FPGA
    // flows) reject -- a portability hazard across "different FPGA" builds.
    parameter integer DELAY_TICK_WIDTH = $clog2(DELAY_DEPTH + 1),
    // ----- TTL EVENT-SCHEDULER delay geometry ------------------------------------------
    // TTL channel delays are NO LONGER bounded by DELAY_DEPTH: each channel schedules its
    // output TOGGLES (time + level) in a small event FIFO against a free-running global
    // tick counter, so the storage scales with the number of IN-FLIGHT TOGGLES (validated
    // <= EVT_DEPTH by the host), not with the delay length.  TTL_DELAY_WIDTH bounds one
    // delay (32b = ~85.9 s at 20 ns); GTIME_WIDTH bounds one RUN (48b = ~65 days).
    parameter integer TTL_DELAY_WIDTH = 32,
    parameter integer EVT_DEPTH = 256,
    parameter integer GTIME_WIDTH = 48,
    // Event-FIFO COMPACTION.  Only channels that can carry a TTL delay (the real
    // outputs -- NOT the bus-member bits, whose pins are driven by bus_out and whose
    // engine `out` bit is always 0) get an EVT_DEPTH-deep event FIFO.  At depth 256 a
    // FIFO for every one of CHANNEL_COUNT=62 bits would need 62*256*49b ~ 760 Kb of
    // distributed RAM (the part has 400 Kb), so the top instantiates only NUM_DELAY_CH
    // FIFOs and slot s serves channel DELAY_CH_MAP[s*DELAY_CH_IDX_W +: DELAY_CH_IDX_W].
    // Default (DELAY_COMPACT=0): one FIFO per channel, slot == channel -- standalone /
    // testbench use, where the distributed-RAM cap is irrelevant.
    parameter integer DELAY_COMPACT = 0,
    parameter integer NUM_DELAY_CH = CHANNEL_COUNT,
    parameter integer DELAY_CH_IDX_W = 6,                 // bits/slot in the map (>= clog2(CHANNEL_COUNT))
    parameter [NUM_DELAY_CH*DELAY_CH_IDX_W-1:0] DELAY_CH_MAP = {(NUM_DELAY_CH*DELAY_CH_IDX_W){1'b0}}
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

    // PHYSICAL per-bus DAC DELAY -- a LITERAL per-bus delay line (a 10-bit-wide circular buffer
    // of depth DELAY_DEPTH).  The DAC value stream is NOT baked into the segment ticks: each
    // tick the engine's UNDELAYED bus value (bus_value_active) is pushed into the ring, and the
    // delayed bus value = the value pushed d_bus ticks ago (one delay shared by all 10 bits).
    // d=0 is exact passthrough; before the buffer fills (t<d) the read slot is still its FIRE-time
    // 0 -> the bus is silent until t>=d, for free.  Bounded: d <= DELAY_DEPTH (validated by the
    // host).  Proven == engine_model.rtl_bus_delay_line_mirror == bus_delay_line_reference.
    //   * bus_delay_ticks : per-bus delay d in ticks (0 = no delay = passthrough)
    input  wire [BUS_COUNT*DELAY_TICK_WIDTH-1:0] bus_delay_ticks,

    // PHYSICAL per-channel OUTPUT DELAY -- a LITERAL delay line (a per-channel variable-tap SHIFT
    // REGISTER of depth DELAY_DEPTH, the SRL primitive).  A channel delay is NOT baked into the
    // edges -- it is applied to the engine OUTPUT: out_delayed[t] = out_undelayed[t - d], 0 before
    // fire.  Each running tick the channel's undelayed bit is shifted into its OWN shift register;
    // the value pushed d_ch ticks ago is the tap at index d_ch-1, so ALL 62 channels are
    // independently delayable.  d=0 is exact passthrough; before the SR fills (t<d) the gated tap
    // returns its FIRE-time 0 -> the channel is silent until t>=d, for free.
    // Bounded: d <= DELAY_DEPTH (validated by the host).  Proven == engine_model.
    // rtl_delay_line_mirror == delay_line_reference for ANY d in [0, DELAY_DEPTH], zero, and --
    // via the host's folded global shift G -- negative.
    //   * delay_ticks : per-channel delay d in ticks (0 = no delay), one TTL_DELAY_WIDTH (32b)
    //     slice per channel -- the event scheduler supports delays far beyond the bus ring depth
    input  wire [CHANNEL_COUNT*TTL_DELAY_WIDTH-1:0] delay_ticks,

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
    // ----- LITERAL delay-line geometry --------------------------------------------------
    // The ring has DELAY_DEPTH+1 slots so a delay of EXACTLY DELAY_DEPTH is representable (the
    // slot written d ticks ago must not be the slot we overwrite this tick).  DELAY_TICK_WIDTH
    // holds a delay in [0, DELAY_DEPTH]; DELAY_ADDR_WIDTH indexes the ring.
    localparam integer DELAY_SLOTS = DELAY_DEPTH + 1;
    localparam integer DELAY_ADDR_WIDTH = $clog2(DELAY_SLOTS);
    // DELAY_TICK_WIDTH is now a module parameter (declared before the ports) so the port
    // widths can use it without a referenced-before-declaration error on strict tools.

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
    // CONTINUOUS CYCLIC PING-PONG bank parity: the bank a chunk lives in is
    // (chunk[0] ^ scan_bank_base), and scan_bank_base toggles by (n_chunks & 1) at each sweep
    // WRAP -- so the wrap is just another chunk boundary and the host feeds chunk 0 into the
    // alternating bank one-ahead (seamless re-sweep for ANY N, incl. odd chunk counts).  For a
    // RESIDENT scan (<=2 chunks) the toggle is 0 -> base stays 0 -> identical to chunk[0].
    reg scan_bank_base = 1'b0;
    reg [31:0] loop_count_active = 32'd1;
    reg [31:0] loops_remaining = 32'd1;

    // shadows latched at arm time (BRAM pre-reads while reset is asserted)
    // FIFO_DEPTH(=4) resident shadows are seeded at every boundary, so we pre-read that
    // many edges (e0..e4) and loop-start edges (ls0..ls4) -- one more than the old 4.
    reg [TICK_WIDTH-1:0]  sh_e0_t, sh_e1_t, sh_e2_t, sh_e3_t, sh_e4_t;
    reg [COEFF_BITS-1:0]  sh_e0_c, sh_e1_c, sh_e2_c, sh_e3_c, sh_e4_c;
    reg [CHANNEL_COUNT-1:0] sh_e0_m, sh_e1_m, sh_e2_m, sh_e3_m, sh_e4_m;
    reg [TICK_WIDTH-1:0]  sh_ls0_t, sh_ls1_t, sh_ls2_t, sh_ls3_t, sh_ls4_t;
    reg [COEFF_BITS-1:0]  sh_ls0_c, sh_ls1_c, sh_ls2_c, sh_ls3_c, sh_ls4_c;
    reg [CHANNEL_COUNT-1:0] sh_ls0_m, sh_ls1_m, sh_ls2_m, sh_ls3_m, sh_ls4_m;
    reg [TICK_WIDTH-1:0]  sh_final_t;
    reg [COEFF_BITS-1:0]  sh_final_c;
    reg [SLOT_BITS-1:0] scan_first_values;

    // depth-FIFO_DEPTH edge prefetch
    reg [TICK_WIDTH-1:0] arm_t [0:FIFO_DEPTH-1];
    reg [COEFF_BITS-1:0] arm_c [0:FIFO_DEPTH-1];
    reg [CHANNEL_COUNT-1:0] arm_m [0:FIFO_DEPTH-1];
    reg [2:0] arm_nv;                         // valid arm entries (0..FIFO_DEPTH)
    reg [EDGE_ADDR_WIDTH:0] fetch_idx;        // next edge index to read
    // In-flight read markers, ONE BIT PER PIPELINE STAGE from `issue` to data-valid.  The
    // true latency is RD_LAT+1, NOT RD_LAT: `edge_raddr` is a REGISTER (issue decides at
    // cycle T, the address only reaches the BRAM at T+1), then the BRAM adds RD_LAT.  So a
    // read issued at T lands (edge_*_rdata valid) at T+RD_LAT+1.  Tracking only RD_LAT made
    // `landed` fire ONE CYCLE EARLY, so when two reads landed back-to-back the append
    // latched the STALE (previous) bus word twice and silently dropped the next edge -- the
    // "second emCCD pulse is 40 ms / e7 vanished" hardware bug (reproduced in real-IP xsim).
    localparam integer PIPE = RD_LAT + 1;     // issue -> data-valid latency (incl. edge_raddr reg)
    reg [PIPE-1:0] pend;                       // in-flight read markers (1 bit/pipeline-stage)

    // ----- bus runtime --------------------------------------------------------
    reg [BUS_WIDTH-1:0] bus_value_active [0:BUS_COUNT-1];
    // POWER-UP: Xilinx registers take their initial value at configuration, so the DAC
    // pins sit at the SAFE mid-scale code (0 V) from the first clock -- not at code 0
    // (negative full scale) -- even before the host's first CMD_SAFE/reset.
    integer bus_pu;
    initial for (bus_pu = 0; bus_pu < BUS_COUNT; bus_pu = bus_pu + 1)
        bus_value_active[bus_pu] = BUS_SAFE_VALUE[BUS_WIDTH-1:0];
    reg [BUS_SEG_ADDR_WIDTH:0] bus_index_active [0:BUS_COUNT-1];
    reg [BUS_SEG_ADDR_WIDTH:0] bus_count_active [0:BUS_COUNT-1];
    reg bus_ramp_active [0:BUS_COUNT-1];
    reg bus_ramp_dir_up [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_start_tick [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_stop_tick [0:BUS_COUNT-1];
    reg [BUS_WIDTH-1:0] bus_ramp_target [0:BUS_COUNT-1];
    // Bresenham ramp stepping: value(k) = vstart +/- floor(k*delta/span).  Per tick the
    // value moves by step (= delta/span, >1 for STEEP ramps) plus 1 on a remainder-
    // accumulator carry, so ANY slope tracks the ideal line exactly and lands on the
    // target at stop_tick (no 1-LSB/tick crawl + end snap).
    reg [BUS_WIDTH:0] bus_ramp_step [0:BUS_COUNT-1];
    reg [BUS_WIDTH:0] bus_ramp_rem [0:BUS_COUNT-1];
    // The d/span divmod for a STEEP ramp is DEFERRED from segment apply to the first
    // stepping tick (>= 1 full cycle later): the divider then reads REGISTERED operands
    // (rem temporarily holds d), keeping the LUTRAM-read/endpoint-mux logic off its
    // timing path, and lives at ONE site per bus instead of one per apply call site.
    reg bus_ramp_steep [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_denom [0:BUS_COUNT-1];
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_ramp_accum [0:BUS_COUNT-1];

    // ----- delay runtime (per-channel TTL EVENT SCHEDULER + per-bus DAC ring) -----------------
    // TTL: see the EVENT SCHEDULER declarations below -- toggles are queued against a global
    // counter instead of shifting one bit per tick, so the TTL delay bound is TTL_DELAY_WIDTH
    // (~85.9 s) at a fraction of the old SRL cost.
    // DAC: each bus's UNDELAYED 10-bit value history is a BUS_WIDTH-wide ring bus_ring[bus] (a 2D
    // array of 10-bit words -- Vivado DOES recognise this as a 3D RAM and infers it; it is read at
    // (del_wptr - d_bus)), left exactly as is.  d_ch / d_bus are held CTRL (a delay is constant,
    // never scanned), latched into del_ch_ticks / del_bus_ticks at FIRE.  del_fill gates both reads
    // to 0 before the line has filled d deep, so a read before fill returns 0 -> silent until t == d
    // (real startup, for free; no bulk clear needed).
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_ring [0:BUS_COUNT-1][0:DELAY_SLOTS-1];
    reg [DELAY_ADDR_WIDTH-1:0] del_wptr = {DELAY_ADDR_WIDTH{1'b0}};   // write pointer (bus_ring)
    reg [TTL_DELAY_WIDTH-1:0]  del_ch_ticks  [0:CHANNEL_COUNT-1];  // per-channel d (0 = passthrough)
    reg [DELAY_TICK_WIDTH-1:0] del_bus_ticks [0:BUS_COUNT-1];      // per-bus d (0 = passthrough)
    // ----- TTL EVENT SCHEDULER (replaces the 62 per-channel SRL delay lines) -----------
    // The old SRLs stored ONE BIT PER TICK of delay (DELAY_DEPTH=2048 -> 65 SRL32 + tap
    // mux per channel = ~4.1k LUTs, delay capped at ~41 us).  A TTL waveform is toggle-
    // sparse, so store the TOGGLES instead: when the engine's undelayed bit for channel
    // ch flips at global tick t, push {t + d_ch - 1, new_level} into ch's EVT_DEPTH-deep
    // event FIFO; when the free-running g_time reaches the head's time, pop it into the
    // output register (visible the NEXT cycle -> level appears exactly at t + d_ch, i.e.
    // out[t] = in[t-d], identical to the SRL/ring semantics, 0 before the first event).
    // d == 1 cannot use the queue (the entry would have to pop the cycle it is pushed),
    // so it is served by the prev_undelayed register (a 1-tick delay IS one register).
    // Cost ~ EVT_DEPTH x (GTIME_WIDTH+1)b LUTRAM + one GTIME_WIDTH comparator per channel
    // (~2k LUTs total) and the delay bound becomes TTL_DELAY_WIDTH (32b = ~85.9 s) -- the
    // host validates <= EVT_DEPTH toggles in flight per channel inside any d-window.
    reg [GTIME_WIDTH-1:0] g_time = {GTIME_WIDTH{1'b0}};         // free-running ticks since FIRE
    reg [CHANNEL_COUNT-1:0] prev_undelayed = {CHANNEL_COUNT{1'b0}};
    // Event FIFOs are stored by DELAY SLOT (0..NUM_DELAY_CH-1), not by channel: slot s
    // serves channel evt_ch_of(s).  Only pin-driving channels get a slot, so the deep
    // (EVT_DEPTH) distributed RAM is not paid for the bus-member bits.
    // Each slot is its OWN 2D distributed-RAM FIFO, instantiated in the g_evtfifo generate
    // loop near the runtime block below.  A single 3D reg array (evt_mem[slot][depth]) does
    // NOT infer as distributed RAM here: every slot has an INDEPENDENT wr/rd pointer (unlike
    // bus_ring, whose banks share one write pointer), so Vivado's 3D-RAM inference bails and
    // implements the whole array in flip-flops -- at EVT_DEPTH=256 that is 18*256*49 = 226k
    // FF + 256:1 read muxes and does not fit.  Per-slot 2D arrays each map to one simple-
    // dual-port LUTRAM (1 sync write @wr + 1 async read @rd).
    localparam integer EVT_ADDR = $clog2(EVT_DEPTH);
    // Each slot drives ONLY its one owned channel bit (obit << evt_ch_of(slot)); evt_out is
    // their OR, so un-served channels read 0 (the un-driven / before-first-event level).
    wire [CHANNEL_COUNT-1:0] evt_out_contrib [0:NUM_DELAY_CH-1];
    reg  [CHANNEL_COUNT-1:0] evt_out;                          // scheduled (delayed) levels, by channel
    integer evt_ob;
    always @(*) begin
        evt_out = {CHANNEL_COUNT{1'b0}};
        for (evt_ob = 0; evt_ob < NUM_DELAY_CH; evt_ob = evt_ob + 1)
            evt_out = evt_out | evt_out_contrib[evt_ob];
    end
    // slot s -> channel: identity when not compacted, else the packed map.
    function integer evt_ch_of;
        input integer s;
        begin
            evt_ch_of = (DELAY_COMPACT != 0)
                ? DELAY_CH_MAP[s*DELAY_CH_IDX_W +: DELAY_CH_IDX_W]
                : s;
        end
    endfunction
    // Channels served by a FIFO (1 = delay-eligible).  Used to GATE the delay merge so a
    // stray delay on a non-eligible channel (a host bug) plays the channel UNDELAYED
    // instead of sticking it at the un-driven evt_out (0).  Elaboration-time constant.
    reg [CHANNEL_COUNT-1:0] evt_eligible_mask;
    integer em_s;
    initial begin
        evt_eligible_mask = {CHANNEL_COUNT{1'b0}};
        for (em_s = 0; em_s < NUM_DELAY_CH; em_s = em_s + 1)
            evt_eligible_mask[evt_ch_of(em_s)] = 1'b1;
    end
    // del_fill = number of UNDELAYED samples pushed BEFORE this tick (== the running tick index t
    // since FIRE), saturating at DELAY_DEPTH.  A read at distance d is valid (returns the value
    // pushed d ticks ago) once del_fill >= d, i.e. t >= d -- the slot d ago was actually written
    // this run.  Before that (t < d) the read returns 0: out[t]=in[t-d], 0 before fire.  ONE shared
    // counter -- it gives the "0 before fire" startup for the distributed-RAM ring WITHOUT a
    // (non-synthesizable) bulk RAM clear, exactly == delay_line_reference's 0-init startup.
    reg [DELAY_TICK_WIDTH-1:0] del_fill = {DELAY_TICK_WIDTH{1'b0}};
    integer del_i;

    reg reset_meta = 1'b0, reset_sync = 1'b0;
    reg start_meta = 1'b0, start_sync = 1'b0, start_prev = 1'b0;
    reg bus_prog_we_meta = 1'b0, bus_prog_we_sync = 1'b0, bus_prog_we_prev = 1'b0;
    integer bus_loop;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_prog_flat_addr, bus_runtime_addr;
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_accum_next;
    reg [BUS_WIDTH:0] bus_inc;        // this tick's ramp movement: step or step+1
    reg [BUS_WIDTH:0] bus_v_next;     // widened value+inc for target saturation
    reg [2*BUS_WIDTH+1:0] bus_qr;     // {step, rem} from the deferred ramp divmod

    wire start_event = start_sync && !start_prev;
    wire bus_prog_we_event = bus_prog_we_sync != bus_prog_we_prev;

    // ---- per-channel / per-bus held delay slices ----
    function [TTL_DELAY_WIDTH-1:0] zlc_delay_ch_at;
        input integer ch;
        begin zlc_delay_ch_at = delay_ticks[ch*TTL_DELAY_WIDTH +: TTL_DELAY_WIDTH]; end
    endfunction
    function [DELAY_TICK_WIDTH-1:0] zlc_delay_bus_at;
        input integer b;
        begin zlc_delay_bus_at = bus_delay_ticks[b*DELAY_TICK_WIDTH +: DELAY_TICK_WIDTH]; end
    endfunction
    // ring read index "d writes ago" with one conditional +DELAY_SLOTS (no divider).
    function [DELAY_ADDR_WIDTH-1:0] zlc_delay_rd;
        input [DELAY_TICK_WIDTH-1:0] d;
        reg signed [DELAY_ADDR_WIDTH:0] idx;
        begin
            idx = $signed({1'b0, del_wptr}) - $signed({1'b0, d});
            if (idx < 0) idx = idx + DELAY_SLOTS;
            zlc_delay_rd = idx[DELAY_ADDR_WIDTH-1:0];
        end
    endfunction

    // Per-channel OUTPUT-delay merge.  delayed_mask[b] marks the bits a delayed channel owns
    // (cleared from the undelayed state_mask); delayed_out[b] is the SCHEDULED level for that
    // channel: evt_out (event scheduler, d >= 2) or prev_undelayed (a register IS a 1-tick
    // delay, d == 1).  Both are 0 until the first scheduled toggle -> out[t] = in[t-d], 0
    // before t = d, exactly the proven delay_line_reference semantics.
    // out = (state_mask & ~delayed_mask) | delayed_out -- a non-delayed channel passes straight
    // through; a delay never touches another channel.  d_ch == 0 never reaches here.
    reg [CHANNEL_COUNT-1:0] delayed_mask;   // bit b set iff channel b is delayed (d_ch != 0)
    reg [CHANNEL_COUNT-1:0] delayed_out;    // delayed value per owned bit
    integer del_m;
    always @(*) begin
        delayed_mask = {CHANNEL_COUNT{1'b0}};
        delayed_out  = {CHANNEL_COUNT{1'b0}};
        for (del_m = 0; del_m < CHANNEL_COUNT; del_m = del_m + 1) begin
            if (del_ch_ticks[del_m] != {TTL_DELAY_WIDTH{1'b0}} && evt_eligible_mask[del_m]) begin
                delayed_mask[del_m] = 1'b1;
                // d == 1: a single register IS a 1-tick delay (the event queue cannot
                // pop an entry the same cycle it is pushed).  d >= 2: the scheduled
                // level register (out[t] = in[t-d], 0 before the first scheduled event).
                delayed_out[del_m] = (del_ch_ticks[del_m] == {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1})
                                     ? prev_undelayed[del_m] : evt_out[del_m];
            end
        end
    end
    assign out = (state_mask & ~delayed_mask) | delayed_out;

    // ----- per-bus OUTPUT merge: LITERAL per-bus delay line (combinational ring read) -----
    // A NOT-delayed bus (d_bus == 0) passes the live UNDELAYED bus_value_active straight through.
    // A DELAYED bus reads its 10-bit value d_bus writes ago from bus_ring -- gated by (del_fill >=
    // d_bus) so it holds the SAFE mid-scale code (BUS_SAFE_VALUE = 0 V on the offset-binary
    // driver) until t >= d_bus (silent until t == d).
    reg [BUS_WIDTH-1:0] bus_out_merged [0:BUS_COUNT-1];
    integer bus_om;
    always @(*) begin
        for (bus_om = 0; bus_om < BUS_COUNT; bus_om = bus_om + 1) begin
            if (del_bus_ticks[bus_om] != {DELAY_TICK_WIDTH{1'b0}})
                bus_out_merged[bus_om] = (del_fill >= del_bus_ticks[bus_om])
                                         ? bus_ring[bus_om][zlc_delay_rd(del_bus_ticks[bus_om])]
                                         : BUS_SAFE_VALUE[BUS_WIDTH-1:0];
            else
                bus_out_merged[bus_om] = bus_value_active[bus_om];
        end
    end
    genvar gi;
    generate
        for (gi = 0; gi < BUS_COUNT; gi = gi + 1) begin : zlc_bus_out_assign
            assign bus_out[gi*BUS_WIDTH +: BUS_WIDTH] = bus_out_merged[gi];
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
    // PARAMETERIZATION GUARD: this concatenation assumes SCAN_ADDR_WIDTH == BANK_BITS+1
    // (i.e. BANK_SIZE is a power of two and the scan window is exactly 2 banks).  A
    // mismatched geometry would silently alias both banks onto one window -- the host
    // (image.check_rtl_assumptions) rejects such configs at pack time.
    // bank = (chunk[0]) ^ scan_bank_base -- cyclic ping-pong parity (see scan_bank_base).
    function [SCAN_ADDR_WIDTH-1:0] scan_addr_of;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        begin scan_addr_of = {idx[BANK_BITS] ^ scan_bank_base, idx[BANK_BITS-1:0]}; end
    endfunction
    function bank_of;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        begin bank_of = idx[BANK_BITS] ^ scan_bank_base; end
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
            b = idx[BANK_BITS] ^ scan_bank_base;
            scan_point_resident = bank_ready[b] && ((b ? bank_chunk1 : bank_chunk0) == chunk_of(idx));
        end
    endfunction

    // ---- cyclic re-sweep helpers (combinational, off active_scan_count) ----
    // n_chunks = ceil(N / BANK_SIZE); STREAMED iff > 2 banks (else resident: never overwritten).
    wire [SCAN_COUNT_WIDTH-1:0] scan_n_chunks =
        (active_scan_count + (BANK_SIZE[SCAN_COUNT_WIDTH-1:0] - 1'b1)) >> BANK_BITS;
    wire scan_streamed       = scan_n_chunks > {{(SCAN_COUNT_WIDTH-2){1'b0}}, 2'd2};
    // toggle base by n_chunks parity ONLY when streamed; resident scans keep base = 0.
    wire scan_wrap_toggle    = scan_streamed ? scan_n_chunks[0] : 1'b0;
    wire scan_wrap_base_next = scan_bank_base ^ scan_wrap_toggle;
    // point 0 (chunk 0) lives in bank scan_wrap_base_next after the wrap; resident iff that
    // bank is armed and actually holds chunk 0 (the host fed it one-ahead).
    wire scan_point0_ready_next =
        bank_ready[scan_wrap_base_next] &&
        ((scan_wrap_base_next ? bank_chunk1 : bank_chunk0) == {SCAN_COUNT_WIDTH{1'b0}});

    task zlc_bus_clear_runtime;
        integer i;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                bus_value_active[i] <= BUS_SAFE_VALUE[BUS_WIDTH-1:0];   // idle DAC = mid-scale = 0 V
                bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_count_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_ramp_active[i] <= 1'b0; bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_step[i] <= {(BUS_WIDTH+1){1'b0}}; bus_ramp_rem[i] <= {(BUS_WIDTH+1){1'b0}}; bus_ramp_steep[i] <= 1'b0;
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
            end
        end
    endtask

    // Clear the LITERAL delay-line runtime (used on reset/FIRE): zero the per-channel + per-bus
    // delay amounts, the write pointer, and the fill counter.  The ring RAM is NOT bulk-cleared
    // (distributed RAM has no synchronous bulk reset); del_fill gates the read to 0 until the
    // ring has filled d deep this run, which IS the FIRE-time-0 startup (silent until t == d).
    task zlc_delay_clear_runtime;
        integer i;
        begin
            del_wptr <= {DELAY_ADDR_WIDTH{1'b0}};
            del_fill <= {DELAY_TICK_WIDTH{1'b0}};
            for (i = 0; i < CHANNEL_COUNT; i = i + 1) del_ch_ticks[i] <= {TTL_DELAY_WIDTH{1'b0}};
            for (i = 0; i < BUS_COUNT; i = i + 1) del_bus_ticks[i] <= {DELAY_TICK_WIDTH{1'b0}};
        end
    endtask

    // Apply a segment given its ALREADY-COMPUTED effective start/stop ticks.  The
    // caller computes tkstart/tkstop once per bus per cycle (zlc_effective_tick is
    // expensive: a 4-slot affine MAC), and shares them with the advance checks, so
    // the whole engine evaluates only ~2 affine ticks per bus per cycle instead of
    // recomputing the same segment 3x in each branch.  Values + cycle timing are
    // identical to recomputing in-line (this is a pure resource dedup).
    // Quotient + remainder of delta/span for a STEEP ramp (span < delta <= 2^BUS_WIDTH-1,
    // so both operands fit BUS_WIDTH+1 bits).  Restoring division, fully combinational:
    // BUS_WIDTH+1 subtract/compare stages, evaluated once per segment APPLY (not per tick)
    // and comfortably within a 20 ns cycle.  For gentle ramps (delta <= span) the caller
    // skips it (step = 0, rem = delta -- the historic 0/1-steps-per-tick behaviour).
    function [2*BUS_WIDTH+1:0] zlc_bus_ramp_divmod;
        input [BUS_WIDTH:0] num;
        input [BUS_WIDTH:0] den;
        reg [BUS_WIDTH:0] q, r;
        integer k;
        begin
            q = {(BUS_WIDTH+1){1'b0}};
            r = {(BUS_WIDTH+1){1'b0}};
            for (k = BUS_WIDTH; k >= 0; k = k - 1) begin
                r = {r[BUS_WIDTH-1:0], num[k]};
                if (r >= den) begin
                    r = r - den;
                    q[k] = 1'b1;
                end
            end
            zlc_bus_ramp_divmod = {q, r};
        end
    endfunction

    task zlc_bus_apply_segment;
        input integer i;
        input [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        input [SLOT_BITS-1:0] slot_vec;
        input [TICK_WIDTH-1:0] tkstart;
        input [TICK_WIDTH-1:0] tkstop;
        reg [TICK_WIDTH-1:0] span;
        reg [BUS_SEL_WIDTH-1:0] start_sel, stop_sel;
        reg [BUS_WIDTH-1:0] vstart, vstop;
        reg [BUS_WIDTH:0] d;
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
                if (vstop >= vstart) begin bus_ramp_dir_up[i] <= 1'b1; d = vstop - vstart; end
                else begin bus_ramp_dir_up[i] <= 1'b0; d = vstart - vstop; end
                // Bresenham split: per-tick base step = d/span, remainder feeds the carry
                // accumulator.  GENTLE (d <= span) is final as-is; STEEP defers the d/span
                // divmod to the first stepping tick (see bus_ramp_steep), with rem
                // temporarily holding d.  Steep => span < d <= 2^BUS_WIDTH-1, so span
                // fits the divider's BUS_WIDTH+1 bits.
                bus_ramp_step[i] <= {(BUS_WIDTH+1){1'b0}};
                bus_ramp_rem[i] <= d;
                bus_ramp_steep[i] <= (span < d);
                bus_value_active[i] <= vstart; bus_ramp_active[i] <= 1'b1;
                bus_ramp_start_tick[i] <= tkstart; bus_ramp_stop_tick[i] <= tkstop;
                bus_ramp_target[i] <= vstop; bus_ramp_denom[i] <= span;
                bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
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
                    bus_value_active[i] <= BUS_SAFE_VALUE[BUS_WIDTH-1:0]; bus_ramp_active[i] <= 1'b0; bus_ramp_dir_up[i] <= 1'b0;
                    bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                    bus_ramp_target[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_step[i] <= {(BUS_WIDTH+1){1'b0}}; bus_ramp_rem[i] <= {(BUS_WIDTH+1){1'b0}}; bus_ramp_steep[i] <= 1'b0;
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
                        if (bus_ramp_steep[i]) begin
                            // First stepping tick of a STEEP ramp: split d (parked in rem)
                            // into step + remainder from REGISTERED operands.  accum is
                            // still 0 and rem < span by construction, so this tick can
                            // never carry: inc is exactly the new step -- identical to
                            // having divided at apply, but with a register-fed divider.
                            bus_qr = zlc_bus_ramp_divmod(bus_ramp_rem[i], bus_ramp_denom[i][BUS_WIDTH:0]);
                            bus_ramp_step[i] <= bus_qr[2*BUS_WIDTH+1:BUS_WIDTH+1];
                            bus_ramp_rem[i] <= bus_qr[BUS_WIDTH:0];
                            bus_ramp_steep[i] <= 1'b0;
                            bus_ramp_accum[i] <= {{(TICK_WIDTH){1'b0}}, bus_qr[BUS_WIDTH:0]};
                            bus_inc = bus_qr[2*BUS_WIDTH+1:BUS_WIDTH+1];
                        end else begin
                            bus_accum_next = bus_ramp_accum[i] + bus_ramp_rem[i];
                            if (bus_accum_next >= bus_ramp_denom[i]) begin
                                bus_ramp_accum[i] <= bus_accum_next - bus_ramp_denom[i];
                                bus_inc = bus_ramp_step[i] + 1'b1;
                            end else begin
                                bus_ramp_accum[i] <= bus_accum_next;
                                bus_inc = bus_ramp_step[i];
                            end
                        end
                        // Move by the full Bresenham increment, saturating AT the target
                        // (widened compares cannot overflow: value+inc <= 2^(W+1)-1).
                        if (bus_inc != {(BUS_WIDTH+1){1'b0}}) begin
                            if (bus_ramp_dir_up[i]) begin
                                bus_v_next = {1'b0, bus_value_active[i]} + bus_inc;
                                bus_value_active[i] <= (bus_v_next >= {1'b0, bus_ramp_target[i]})
                                                       ? bus_ramp_target[i] : bus_v_next[BUS_WIDTH-1:0];
                            end else begin
                                if ({1'b0, bus_value_active[i]} <= {1'b0, bus_ramp_target[i]} + bus_inc)
                                    bus_value_active[i] <= bus_ramp_target[i];
                                else
                                    bus_value_active[i] <= bus_value_active[i] - bus_inc[BUS_WIDTH-1:0];
                            end
                        end
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
    // ``cnt`` is the program's edge count, passed in EXPLICITLY (not read from the
    // active_count REG).  At FIRE, active_count <= prog_count is a non-blocking write
    // that has NOT committed when this task runs the same cycle, so reading the reg
    // here would see the PREVIOUS program's count (0 right after a fresh bitstream).
    // That stale count truncated arm_nv -> the resident shadows beyond it were marked
    // invalid and overwritten by prefetch, permanently dropping the first frame's tail
    // edges (a 3-edge prior program dropped edge3 -> "second pulse never appears"; a
    // 2-edge prior program dropped edges 2,3 -> "the OFF never fires, stuck high").
    // The caller threads prog_count at FIRE and active_count at the (mid-run, already
    // committed) frame/scan boundaries.
    task seed_from_edge0;
        input [SLOT_BITS-1:0] sv;
        input [EDGE_ADDR_WIDTH:0] cnt;
        begin
            pend <= {PIPE{1'b0}};
            if (cnt != 0 && zlc_effective_tick(sh_e0_t, sh_e0_c, sv) == {TICK_WIDTH{1'b0}}) begin
                state_mask <= sh_e0_m; time_count <= {{(TICK_WIDTH-1){1'b0}},1'b1};
                edge_index <= {{EDGE_ADDR_WIDTH{1'b0}},1'b1};
                arm_t[0]<=sh_e1_t; arm_c[0]<=sh_e1_c; arm_m[0]<=sh_e1_m;
                arm_t[1]<=sh_e2_t; arm_c[1]<=sh_e2_c; arm_m[1]<=sh_e2_m;
                arm_t[2]<=sh_e3_t; arm_c[2]<=sh_e3_c; arm_m[2]<=sh_e3_m;
                arm_t[3]<=sh_e4_t; arm_c[3]<=sh_e4_c; arm_m[3]<=sh_e4_m;
                arm_nv <= clamp3(cnt - 1'b1);
                fetch_idx <= {{(EDGE_ADDR_WIDTH-2){1'b0}},3'd5}; edge_raddr <= {{(EDGE_ADDR_WIDTH-2){1'b0}},3'd5};
            end else begin
                state_mask <= {CHANNEL_COUNT{1'b0}}; time_count <= {TICK_WIDTH{1'b0}};
                edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
                arm_t[0]<=sh_e0_t; arm_c[0]<=sh_e0_c; arm_m[0]<=sh_e0_m;
                arm_t[1]<=sh_e1_t; arm_c[1]<=sh_e1_c; arm_m[1]<=sh_e1_m;
                arm_t[2]<=sh_e2_t; arm_c[2]<=sh_e2_c; arm_m[2]<=sh_e2_m;
                arm_t[3]<=sh_e3_t; arm_c[3]<=sh_e3_c; arm_m[3]<=sh_e3_m;
                arm_nv <= clamp3(cnt);
                fetch_idx <= {{(EDGE_ADDR_WIDTH-2){1'b0}},3'd4}; edge_raddr <= {{(EDGE_ADDR_WIDTH-2){1'b0}},3'd4};
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
    reg [2:0] inflight;        // popcount(pend): reads owned by the read pipeline this cycle
    integer k;
    integer pk;

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
    reg [EDGE_ADDR_WIDTH:0] bnd_count;   // edge count to seed with: prog_count at FIRE,
                                          // active_count at the (committed) frame/scan seams
    // LITERAL delay-line boundary work: push the undelayed state_mask / bus values into the
    // rings and advance the write pointer.  Set on EVERY tick once the engine has fired (running
    // OR done-but-emitting), so the ring shifts the WHOLE output stream -- exactly the post-play
    // delay_line_reference shift.  No frame seam / skip counter -- a circular buffer needs none.
    reg bnd_delay_advance;     // push state_mask + bus values into the rings this tick

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
            bus_stop_value_select_mem[bus_prog_flat_addr] <= bus_prog_stop_value_select;
            bus_start_tick_coeff_mem[bus_prog_flat_addr] <= bus_prog_start_tick_coeffs;
            bus_stop_tick_coeff_mem[bus_prog_flat_addr] <= bus_prog_stop_tick_coeffs;
        end

        // boundary work-request defaults (consumed once, after the state chain)
        bnd_bus_tick = 1'b0; bnd_bus_reinit = 1'b0; bnd_seed = 1'b0;
        bnd_recompute_final = 1'b0; bnd_slots = slot_active; bnd_count = active_count;
        bnd_delay_advance = 1'b0;

        if (reset_sync) begin
            running <= 1'b0; done <= 1'b0; underflow <= 1'b0;
            state_mask <= {CHANNEL_COUNT{1'b0}};
            arm_nv <= 3'd0; pend <= {PIPE{1'b0}};
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
                // Pre-read FIFO_DEPTH(=5 shadows: e0..e4) edges + the loop-start window
                // (ls0..ls4) + final + scan0.  ARM_SETTLE (>= PIPE) cycles between reads, so
                // these latch the SETTLED bus -- the streaming-prefetch off-by-one cannot
                // touch the seed shadows.  Steps: 0-4 e0..e4, 5-9 ls0..ls4, 10 final, 11 scan0.
                case (arm_step)
                    4'd0: begin sh_e0_t<=edge_tick_rdata; sh_e0_c<=edge_coeff_rdata; sh_e0_m<=edge_mask_rdata; edge_raddr<={{(EDGE_ADDR_WIDTH-1){1'b0}},1'b1}; end
                    4'd1: begin sh_e1_t<=edge_tick_rdata; sh_e1_c<=edge_coeff_rdata; sh_e1_m<=edge_mask_rdata; edge_raddr<={{(EDGE_ADDR_WIDTH-2){1'b0}},2'd2}; end
                    4'd2: begin sh_e2_t<=edge_tick_rdata; sh_e2_c<=edge_coeff_rdata; sh_e2_m<=edge_mask_rdata; edge_raddr<={{(EDGE_ADDR_WIDTH-2){1'b0}},2'd3}; end
                    4'd3: begin sh_e3_t<=edge_tick_rdata; sh_e3_c<=edge_coeff_rdata; sh_e3_m<=edge_mask_rdata; edge_raddr<={{(EDGE_ADDR_WIDTH-3){1'b0}},3'd4}; end
                    4'd4: begin sh_e4_t<=edge_tick_rdata; sh_e4_c<=edge_coeff_rdata; sh_e4_m<=edge_mask_rdata; edge_raddr<=loop_start_addr; end
                    4'd5: begin sh_ls0_t<=edge_tick_rdata; sh_ls0_c<=edge_coeff_rdata; sh_ls0_m<=edge_mask_rdata; edge_raddr<=loop_start_addr+1'b1; end
                    4'd6: begin sh_ls1_t<=edge_tick_rdata; sh_ls1_c<=edge_coeff_rdata; sh_ls1_m<=edge_mask_rdata; edge_raddr<=loop_start_addr+2'd2; end
                    4'd7: begin sh_ls2_t<=edge_tick_rdata; sh_ls2_c<=edge_coeff_rdata; sh_ls2_m<=edge_mask_rdata; edge_raddr<=loop_start_addr+2'd3; end
                    4'd8: begin sh_ls3_t<=edge_tick_rdata; sh_ls3_c<=edge_coeff_rdata; sh_ls3_m<=edge_mask_rdata; edge_raddr<=loop_start_addr+3'd4; end
                    4'd9: begin sh_ls4_t<=edge_tick_rdata; sh_ls4_c<=edge_coeff_rdata; sh_ls4_m<=edge_mask_rdata;
                                edge_raddr<=(prog_count==0)?{EDGE_ADDR_WIDTH{1'b0}}:(prog_count[EDGE_ADDR_WIDTH-1:0]-1'b1); end
                    4'd10: begin sh_final_t<=edge_tick_rdata; sh_final_c<=edge_coeff_rdata; scan_raddr<={SCAN_ADDR_WIDTH{1'b0}}; end
                    4'd11: begin scan_first_values<=(scan_enable && scan_count!=0)?scan_rdata:{SLOT_BITS{1'b0}}; end
                    default: ;
                endcase
                if (arm_step < 4'd11) begin arm_step <= arm_step + 1'b1; arm_wait <= ARM_SETTLE[3:0]; end
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
            scan_bank_base <= 1'b0;                                                // sweep 0: chunk c in bank c[0]
            scan_raddr <= {{(SCAN_ADDR_WIDTH-1){1'b0}}, 1'b1};                     // pre-read point 1 (bank 0)
            loop_count_active <= (loop_count==0)?32'd1:loop_count;
            loops_remaining <= (loop_count==0)?32'd1:loop_count;
            // AT FIRE: latch the LITERAL delay-line amounts from the held CTRL words and reset the
            // write pointer + fill counter (del_fill gates the read to 0 until the ring fills d
            // deep -> silent until t == d, real startup -- no bulk RAM clear).  A delay is constant
            // (never scanned), so the per-channel d_ch and per-bus d_bus are plain held values; ALL
            // 62 channels and ALL 4 buses are independently delayable (d == 0 => passthrough).
            del_wptr <= {DELAY_ADDR_WIDTH{1'b0}};
            del_fill <= {DELAY_TICK_WIDTH{1'b0}};
            for (del_i = 0; del_i < CHANNEL_COUNT; del_i = del_i + 1)
                del_ch_ticks[del_i] <= zlc_delay_ch_at(del_i);
            for (del_i = 0; del_i < BUS_COUNT; del_i = del_i + 1)
                del_bus_ticks[del_i] <= zlc_delay_bus_at(del_i);
            // heavy affine work (final/loop_end recompute, bus (re)start, edge-0 seed)
            // is dispatched ONCE after the chain via these flags (see SLOT_MUL_WIDTH /
            // bnd_* notes): same cycle, same slots -> identical behavior, far fewer MACs.
            bnd_slots = scan_first_values; bnd_recompute_final = 1'b1;
            bnd_count = prog_count;   // FIRE: active_count <= prog_count not yet committed
            bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_seed = 1'b1;
        end else if (running) begin
            landed = pend[PIPE-1];   // data valid PIPE (=RD_LAT+1) cycles after issue
            // every RUNNING output tick pushes the undelayed state_mask + bus values into the
            // delay rings (the literal delay line); the ring read d ticks ago IS the delayed
            // output.  A circular buffer needs no frame seam / skip counter -- it shifts the
            // whole stream uniformly, exactly the post-play delay_line_reference shift.
            bnd_delay_advance = 1'b1;
            if (loop_count_active>32'd1 && loops_remaining>32'd1 && time_count>=loop_end_active) begin
                // loop rewind: output loop_start mask, seed arm from loop_start+1
                state_mask <= sh_ls0_m; time_count <= zlc_effective_tick(sh_ls0_t,sh_ls0_c,slot_active)+1'b1;
                edge_index <= {1'b0,loop_start_addr}+1'b1; loops_remaining <= loops_remaining-1'b1;
                arm_t[0]<=sh_ls1_t; arm_c[0]<=sh_ls1_c; arm_m[0]<=sh_ls1_m;
                arm_t[1]<=sh_ls2_t; arm_c[1]<=sh_ls2_c; arm_m[1]<=sh_ls2_m;
                arm_t[2]<=sh_ls3_t; arm_c[2]<=sh_ls3_c; arm_m[2]<=sh_ls3_m;
                arm_t[3]<=sh_ls4_t; arm_c[3]<=sh_ls4_c; arm_m[3]<=sh_ls4_m;
                arm_nv <= clamp3(active_count - ({1'b0,loop_start_addr}+1'b1));
                fetch_idx <= {1'b0,loop_start_addr}+3'd5; edge_raddr <= loop_start_addr+3'd5;
                pend <= {PIPE{1'b0}};
                bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_slots = slot_active;  // re(start) bus, keep slots
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
                        arm_t[3]<=sh_ls4_t; arm_c[3]<=sh_ls4_c; arm_m[3]<=sh_ls4_m;
                        arm_nv <= clamp3(active_count - ({1'b0,loop_start_addr}+1'b1));
                        fetch_idx <= {1'b0,loop_start_addr}+3'd5; edge_raddr <= loop_start_addr+3'd5;
                        pend <= {PIPE{1'b0}};
                        bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_slots = slot_active;
                    // CONTINUOUS CYCLIC PING-PONG re-sweep: the wrap is just another chunk
                    // boundary.  scan_bank_base toggles by (n_chunks & 1) so chunk 0 lands in the
                    // bank the host fed it ONE-AHEAD (bank scan_wrap_base_next, NOT necessarily
                    // bank 0); for a RESIDENT scan (<=2 chunks) the toggle is 0 so chunk 0 stays
                    // in bank 0 (identical to before).  Proceed the instant point 0 is resident
                    // in that bank -- seamless; STALL (safe hold) only if the host is genuinely
                    // behind.  scan_cursor is published = N so a late host still gets the signal.
                    end else if (scan_enable_active && !scan_point0_ready_next) begin
                        underflow <= 1'b1;
                        scan_cursor <= active_scan_count;
                    end else begin
                        underflow <= 1'b0;
                        scan_bank_base <= scan_wrap_base_next;     // cyclic bank flip (0 if resident)
                        slot_active <= scan_first_values; scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}}; scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
                        // pre-read point 1 (chunk 0) from the NEW base's bank
                        scan_raddr <= {scan_wrap_base_next, {(SCAN_ADDR_WIDTH-2){1'b0}}, 1'b1};
                        loops_remaining <= loop_count_active;
                        bnd_slots = scan_first_values; bnd_recompute_final = 1'b1;
                        bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b1; bnd_seed = 1'b1;
                    end
                end else begin
                    running <= 1'b0; done <= 1'b1; state_mask <= {CHANNEL_COUNT{1'b0}}; zlc_bus_clear_runtime();
                end
            end else begin
                bnd_bus_tick = 1'b1; bnd_bus_reinit = 1'b0; bnd_slots = slot_active;  // normal bus step
                // >= (not ==): if the head edge's effective tick was ever passed without
                // firing (timing slip / upstream bug), fire it LATE instead of stalling the
                // whole frame.  Ticks are strictly increasing per scan point (host validator),
                // so on a correct run this is exactly == ; it only ever self-heals.
                do_fire = (edge_index < active_count) && (arm_nv != 0) && (time_count >= zlc_effective_tick(arm_t[0],arm_c[0],slot_active));
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
                // --- issue a read iff the pipeline+FIFO would not overflow ---
                // Every read the pipeline OWNS (resident shadows + ALL in-flight reads that
                // will still append) must have a landing slot: owned_next = nv_after_fire +
                // popcount(pend) + issue <= FIFO_DEPTH.  popcount(pend) spans the full PIPE
                // stages (the previous code summed only landed+pend[0], which silently
                // assumed PIPE==2 and under-counted in-flight reads once PIPE grew to 3).
                inflight = {3{1'b0}};
                for (pk = 0; pk < PIPE; pk = pk + 1) inflight = inflight + {2'b0, pend[pk]};
                issue = ((nv_after_fire + inflight) < FIFO_DEPTH[2:0])
                        && (fetch_idx < active_count);
                if (issue) begin edge_raddr <= fetch_idx; fetch_idx <= fetch_idx + 1'b1; end
                pend <= {pend[PIPE-2:0], issue};
            end
        end else if (done) begin
            // DONE-but-emitting: keep shifting the delay rings after the final tick so a
            // DELAYED channel/bus flushes its remaining tail (up to delay_depth ticks) and
            // then settles at its REST value -- state_mask was cleared to 0 and
            // bus_value_active to BUS_SAFE_VALUE (mid code = 0 V) at done, so the pushes are
            // those rest values and out[t] = in[t-d] holds for the WHOLE stream, exactly
            // the delay_line_reference / rtl_mirror_play contract.  Without this the rings
            // FREEZE at done and a delayed channel holds a STALE (possibly HIGH) tap value
            // until the host reacts (ms over JTAG).  A new FIRE takes the start_event branch
            // above (resets del_wptr/del_fill), so this free-running shift is never harmful.
            bnd_delay_advance = 1'b1;
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
        if (bnd_seed) seed_from_edge0(bnd_slots, bnd_count);
        // ---- DAC delay ring: push this tick's UNDELAYED bus values ------------------------------
        // (TTL no longer uses a per-tick line: see the EVENT SCHEDULER block below.)  Write each
        // bus's current undelayed value into its ring slot del_wptr, advance del_wptr + the fill
        // counter; the combinational ring read at (del_wptr - d) with the del_fill gate is the
        // proven out[t] = in[t-d] / safe-before-fill behaviour.
        if (bnd_delay_advance) begin
            for (del_i = 0; del_i < BUS_COUNT; del_i = del_i + 1)
                bus_ring[del_i][del_wptr] <= bus_value_active[del_i];
            del_wptr <= (del_wptr == DELAY_SLOTS-1) ? {DELAY_ADDR_WIDTH{1'b0}} : (del_wptr + 1'b1);
            if (del_fill < DELAY_DEPTH[DELAY_TICK_WIDTH-1:0])   // saturate at the max valid d
                del_fill <= del_fill + 1'b1;
        end
    end

    // ----- TTL EVENT SCHEDULER runtime -------------------------------------------------------
    // Self-contained block (its own reset/FIRE handling so the main playback chain is
    // untouched).  Timeline (g_time == running tick t, both reset at FIRE):
    //   * cycle t: the undelayed state_mask differs from prev_undelayed on channel ch
    //     (the toggle that happened AT t) -> push {g_time + d_ch - 1, new_level} into ch's
    //     event FIFO (d_ch >= 2 here; d == 1 is the prev_undelayed register, d == 0 bypass).
    //   * cycle u == t + d - 1: the head matches g_time -> evt_out[ch] <= level (NBA),
    //     visible during cycle t + d  ==>  out[t] = in[t-d] exactly, 0 before the first event.
    //   * g_time keeps counting after done so a long delayed tail drains; CMD_SAFE (reset)
    //     clears the queues and drops the outputs immediately (same as the old line).
    // Per-channel pushes are strictly time-ordered (one toggle per cycle per channel), so a
    // plain FIFO + equality compare is exact.  The host validates that no more than
    // EVT_DEPTH toggles are in flight inside any d-window; the guard below additionally
    // drops (rather than corrupts) on an impossible overflow.
    // Shared free-running time + the previous-undelayed snapshot (read by every slot FIFO).
    always @(posedge clk) begin
        if (reset_sync) begin
            g_time <= {GTIME_WIDTH{1'b0}};
            prev_undelayed <= {CHANNEL_COUNT{1'b0}};
        end else begin
            g_time <= g_time + 1'b1;
            prev_undelayed <= state_mask;
        end
    end
    // One independent FIFO per delay slot.  Each is its own 2D distributed-RAM (sync write @wr,
    // async read @rd) so the 226k-FF 3D fallback is gone; behaviour is bit-identical to the old
    // shared loop (xsim tb_delay_sched / tb_delay_compact).
    genvar gevs;
    generate
    for (gevs = 0; gevs < NUM_DELAY_CH; gevs = gevs + 1) begin : g_evtfifo
        localparam integer GEVC = (DELAY_COMPACT != 0)
            ? DELAY_CH_MAP[gevs*DELAY_CH_IDX_W +: DELAY_CH_IDX_W]
            : gevs;                                            // channel this slot serves (constant)
        (* ram_style = "distributed" *) reg [GTIME_WIDTH:0] fifo [0:EVT_DEPTH-1];
        reg [EVT_ADDR-1:0] wr  = {EVT_ADDR{1'b0}};
        reg [EVT_ADDR-1:0] rd  = {EVT_ADDR{1'b0}};
        reg [EVT_ADDR:0]   cnt = {(EVT_ADDR+1){1'b0}};
        reg                obit = 1'b0;                        // this channel's scheduled (delayed) level
        reg pushf, popf;
        wire [GTIME_WIDTH:0] headw = fifo[rd];                 // async-read FIFO head (LUTRAM read port)
        // obit -> bit GEVC; un-served channels contribute 0 (GEVC is a per-instance constant)
        assign evt_out_contrib[gevs] = {{(CHANNEL_COUNT-1){1'b0}}, obit} << GEVC;
        always @(posedge clk) begin
            if (reset_sync) begin
                wr   <= {EVT_ADDR{1'b0}};
                rd   <= {EVT_ADDR{1'b0}};
                cnt  <= {(EVT_ADDR+1){1'b0}};
                obit <= 1'b0;
            end else begin
                pushf = (state_mask[GEVC] != prev_undelayed[GEVC])
                        && (del_ch_ticks[GEVC] > {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1})
                        && (cnt != EVT_DEPTH[EVT_ADDR:0]);
                popf  = (cnt != {(EVT_ADDR+1){1'b0}})
                        && (headw[GTIME_WIDTH:1] == g_time);
                if (pushf) begin
                    // zero-EXTEND the 32b delay to the 48b time base (an out-of-range
                    // part-select here would X-poison the entry and kill the compare)
                    fifo[wr] <= {
                        g_time + {{(GTIME_WIDTH-TTL_DELAY_WIDTH){1'b0}}, del_ch_ticks[GEVC]} - 1'b1,
                        state_mask[GEVC] };
                    wr <= wr + 1'b1;
                end
                if (popf) begin
                    obit <= headw[0];
                    rd <= rd + 1'b1;
                end
                case ({pushf, popf})
                    2'b10: cnt <= cnt + 1'b1;
                    2'b01: cnt <= cnt - 1'b1;
                    default: ;
                endcase
            end
        end
    end
    endgenerate

    initial begin arm_kicked = 1'b0; arm_step = 4'd0; arm_wait = 4'd0; end
endmodule
