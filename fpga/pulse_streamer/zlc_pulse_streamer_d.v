`timescale 1ns / 1ps
// =============================================================================
// zlc_pulse_streamer_d -- Architecture-D affine edge-table pulse streamer.
//
// Same seamless behaviour as the validated zlc_pulse_streamer.v (global edge
// pointer, single-cycle hardware loop, single-cycle scan-point advance, N-slot
// affine scan of delay/duration, DAC-value scan via bus value_select) but the
// EDGE table and SCAN-point table live in BLOCK RAM (so 2048 edges + thousands
// of scan points fit the 35T at <=75%) instead of LUTRAM.  The analog-BUS
// segment tables STAY in LUTRAM, because the bus/ramp engine reads them
// combinationally every tick -- moving them to BRAM would break that.  This is
// the "D" choice (adversarially reviewed as the optimal one).
//
// BRAM read is synchronous, so the same-cycle fire is fed by a DEPTH-1 prefetch:
// `cur` is a register holding the current edge -- edge 0 comes from the
// first-edge SHADOW (latched at arm time, instant), every later edge is read
// from BRAM into `pre_*` and becomes valid `RD_SETTLE` cycles after edge_index
// advances.  Because a read is issued the moment an edge fires and the next edge
// is >= the host-enforced MINIMUM EDGE SPACING away, `pre_*` is always valid
// before its effective tick, so the fire stays same-cycle / gapless.  The
// gapless reload sites (start / loop-rewind / scan-advance / repeat-forever)
// already reload from shadow registers (first/loop_start/final), so they incur
// ZERO BRAM latency -- exactly as the validated engine.
//
// PROVEN PRE-HARDWARE (no Verilog sim in repo): the depth-1 algorithm is
// byte-identical to the validated combinatorial engine for every program shape
// whose min edge spacing >= RD_SETTLE+1, and STALLS otherwise -- see
// Zou_lab_control/neutral_atom/devices/edgetable_engine_model.py:prefetch_d1_play
// and test_edgetable_d1_prefetch_matches_reference_with_min_spacing.  The host
// (compile + validate) enforces min edge spacing >= RD_SETTLE+1 (= 3 ticks =
// 60 ns at 50 MHz) with a clear error, so the stall corner never occurs.
//
// Edge/scan tables are external (top-level blk_mem_gen, port A = AXI write, port
// B = engine read here).  An EDGE row is read packed as
//   {mask[CHANNEL_COUNT], coeffs[NUM_SLOTS*COEFF_WIDTH], base_tick[TICK_WIDTH]}
// (low->high).  A SCAN row is the NUM_SLOTS*TICK_WIDTH slot vector.  Bus tables
// are written through bus_prog_* (a small top loader copies them into LUTRAM).
// =============================================================================

module zlc_pulse_streamer_d #(
    parameter integer CHANNEL_COUNT = 62,
    parameter integer EDGE_ADDR_WIDTH = 11,
    parameter integer TICK_WIDTH = 32,
    parameter integer SCAN_ADDR_WIDTH = 12,
    parameter integer NUM_SLOTS = 4,
    parameter integer COEFF_WIDTH = 16,
    parameter integer COEFF_FRAC_BITS = 8,
    parameter integer BUS_COUNT = 4,
    parameter integer BUS_INDEX_WIDTH = 2,
    parameter integer BUS_WIDTH = 10,
    parameter integer BUS_SEG_ADDR_WIDTH = 6,
    parameter integer BUS_SEL_WIDTH = 3,
    parameter integer RD_SETTLE = 2              // tolerate BRAM read latency 1 OR 2
)(
    input  wire clk,
    input  wire reset,
    input  wire start,

    // Held program scalars (driven by the top's AXI register file).
    input  wire [EDGE_ADDR_WIDTH:0] prog_count,
    input  wire repeat_forever,
    input  wire [EDGE_ADDR_WIDTH-1:0] loop_start_addr,
    input  wire [TICK_WIDTH-1:0] loop_end_tick,
    input  wire [NUM_SLOTS*COEFF_WIDTH-1:0] loop_end_coeffs,
    input  wire [31:0] loop_count,
    input  wire scan_enable,
    input  wire [SCAN_ADDR_WIDTH:0] scan_count,

    // EDGE table BRAM read port (port B of the top-level edge BRAM).
    output reg  [EDGE_ADDR_WIDTH-1:0] edge_raddr,
    input  wire [TICK_WIDTH + NUM_SLOTS*COEFF_WIDTH + CHANNEL_COUNT - 1:0] edge_rdata,

    // SCAN table BRAM read port (port B of the top-level scan BRAM).
    output reg  [SCAN_ADDR_WIDTH-1:0] scan_raddr,
    input  wire [NUM_SLOTS*TICK_WIDTH-1:0] scan_rdata,

    // Bus segment table write port (LUTRAM inside this module).
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
    localparam integer EDGE_BITS = TICK_WIDTH + COEFF_BITS + CHANNEL_COUNT;
    localparam integer ACC_WIDTH = TICK_WIDTH + COEFF_WIDTH + 4;
    localparam [1:0] BUS_MODE_EDGE = 2'd1;
    localparam [1:0] BUS_MODE_RAMP = 2'd2;

    // --- BUS segment tables stay in LUTRAM (per-tick combinatorial read) -------
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_start_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_stop_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_start_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_stop_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [1:0] bus_mode_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_SEL_WIDTH-1:0] bus_value_select_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_start_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_stop_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];

    // --- engine state ----------------------------------------------------------
    reg [CHANNEL_COUNT-1:0] state_mask = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] time_count = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] loop_end_active = {TICK_WIDTH{1'b0}};
    reg [EDGE_ADDR_WIDTH:0] edge_index = {(EDGE_ADDR_WIDTH+1){1'b0}};
    reg [EDGE_ADDR_WIDTH:0] active_count = {(EDGE_ADDR_WIDTH+1){1'b0}};
    reg repeat_forever_active = 1'b0;
    reg scan_enable_active = 1'b0;
    reg [SLOT_BITS-1:0] slot_active = {SLOT_BITS{1'b0}};
    reg [SCAN_ADDR_WIDTH:0] active_scan_count = {(SCAN_ADDR_WIDTH+1){1'b0}};
    reg [SCAN_ADDR_WIDTH:0] scan_point_index = {(SCAN_ADDR_WIDTH+1){1'b0}};
    reg [31:0] loop_count_active = 32'd1;
    reg [31:0] loops_remaining = 32'd1;

    // shadows latched at arm time (BRAM pre-reads while reset is asserted)
    reg [TICK_WIDTH-1:0] first_tick_shadow;
    reg [COEFF_BITS-1:0] first_coeffs_shadow;
    reg [CHANNEL_COUNT-1:0] first_mask_shadow;
    reg [TICK_WIDTH-1:0] loop_start_tick_shadow;
    reg [COEFF_BITS-1:0] loop_start_coeffs_shadow;
    reg [CHANNEL_COUNT-1:0] loop_start_mask_shadow;
    reg [TICK_WIDTH-1:0] final_tick_shadow;
    reg [COEFF_BITS-1:0] final_coeffs_shadow;
    reg [SLOT_BITS-1:0] scan_first_values;

    // depth-1 prefetch: pre_* holds edge_index's row when edge_index >= 1.
    reg [TICK_WIDTH-1:0] pre_tick;
    reg [COEFF_BITS-1:0] pre_coeffs;
    reg [CHANNEL_COUNT-1:0] pre_mask;
    reg pre_valid;
    reg [$clog2(RD_SETTLE+1)-1:0] rd_wait;

    // bus runtime
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
    integer bus_loop;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_prog_flat_addr;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_runtime_addr;
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_accum_next;

    wire start_event = start_sync && !start_prev;
    wire bus_prog_we_event = bus_prog_we_sync != bus_prog_we_prev;
    wire [EDGE_ADDR_WIDTH-1:0] edge_addr = edge_index[EDGE_ADDR_WIDTH-1:0];

    // current edge = edge-0 shadow (instant) when edge_index==0, else prefetch reg.
    wire is_edge0 = (edge_index == {(EDGE_ADDR_WIDTH+1){1'b0}});
    wire [TICK_WIDTH-1:0] cur_base = is_edge0 ? first_tick_shadow : pre_tick;
    wire [COEFF_BITS-1:0] cur_coeffs = is_edge0 ? first_coeffs_shadow : pre_coeffs;
    wire [CHANNEL_COUNT-1:0] cur_mask = is_edge0 ? first_mask_shadow : pre_mask;
    wire cur_avail = is_edge0 ? 1'b1 : pre_valid;

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

    function [TICK_WIDTH-1:0] row_base;  input [EDGE_BITS-1:0] r; begin row_base = r[TICK_WIDTH-1:0]; end endfunction
    function [COEFF_BITS-1:0] row_coeffs; input [EDGE_BITS-1:0] r; begin row_coeffs = r[TICK_WIDTH +: COEFF_BITS]; end endfunction
    function [CHANNEL_COUNT-1:0] row_mask; input [EDGE_BITS-1:0] r; begin row_mask = r[TICK_WIDTH+COEFF_BITS +: CHANNEL_COUNT]; end endfunction

    wire [TICK_WIDTH-1:0] cur_eff = zlc_effective_tick(cur_base, cur_coeffs, slot_active);

    function [BUS_SEG_ADDR_WIDTH:0] zlc_bus_count_at;
        input integer bus_index;
        begin
            zlc_bus_count_at = bus_counts[bus_index*(BUS_SEG_ADDR_WIDTH+1) +: (BUS_SEG_ADDR_WIDTH+1)];
        end
    endfunction

    task zlc_bus_clear_runtime;
        integer i;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                bus_value_active[i] <= {BUS_WIDTH{1'b0}};
                bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_count_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_ramp_active[i] <= 1'b0;
                bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}};
                bus_ramp_delta[i] <= {(BUS_WIDTH+1){1'b0}};
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
            end
        end
    endtask

    task zlc_bus_apply_segment;
        input integer i;
        input [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        input [SLOT_BITS-1:0] slot_vec;
        reg [TICK_WIDTH-1:0] span;
        reg [BUS_SEL_WIDTH-1:0] sel;
        reg [BUS_WIDTH-1:0] eff_val_start;
        reg [BUS_WIDTH-1:0] eff_val_stop;
        reg [TICK_WIDTH-1:0] eff_tk_start;
        reg [TICK_WIDTH-1:0] eff_tk_stop;
        begin
            sel = bus_value_select_mem[addr];
            if (sel != {BUS_SEL_WIDTH{1'b0}}) begin
                eff_val_start = slot_vec[(sel - 1'b1)*TICK_WIDTH +: BUS_WIDTH];
                eff_val_stop = eff_val_start;
            end else begin
                eff_val_start = bus_start_value_mem[addr];
                eff_val_stop = bus_stop_value_mem[addr];
            end
            eff_tk_start = zlc_effective_tick(bus_start_tick_mem[addr], bus_start_tick_coeff_mem[addr], slot_vec);
            eff_tk_stop = zlc_effective_tick(bus_stop_tick_mem[addr], bus_stop_tick_coeff_mem[addr], slot_vec);
            if (bus_mode_mem[addr] == BUS_MODE_RAMP && eff_tk_stop > eff_tk_start) begin
                span = eff_tk_stop - eff_tk_start;
                bus_value_active[i] <= eff_val_start;
                bus_ramp_active[i] <= 1'b1;
                bus_ramp_start_tick[i] <= eff_tk_start;
                bus_ramp_stop_tick[i] <= eff_tk_stop;
                bus_ramp_target[i] <= eff_val_stop;
                bus_ramp_denom[i] <= span;
                bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                if (eff_val_stop >= eff_val_start) begin
                    bus_ramp_dir_up[i] <= 1'b1;
                    bus_ramp_delta[i] <= eff_val_stop - eff_val_start;
                end else begin
                    bus_ramp_dir_up[i] <= 1'b0;
                    bus_ramp_delta[i] <= eff_val_start - eff_val_stop;
                end
            end else begin
                bus_value_active[i] <= eff_val_stop;
                bus_ramp_active[i] <= 1'b0;
                bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
            end
        end
    endtask

    function [TICK_WIDTH-1:0] zlc_bus_seg_start;
        input [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        input [SLOT_BITS-1:0] slot_vec;
        begin
            zlc_bus_seg_start = zlc_effective_tick(bus_start_tick_mem[addr], bus_start_tick_coeff_mem[addr], slot_vec);
        end
    endfunction

    task zlc_bus_start_table;
        input [SLOT_BITS-1:0] slot_vec;
        integer i;
        reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        reg [BUS_SEG_ADDR_WIDTH:0] count;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                count = zlc_bus_count_at(i);
                addr = i * MAX_BUS_SEGMENTS;
                bus_count_active[i] <= count;
                bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH+1){1'b0}};
                bus_value_active[i] <= {BUS_WIDTH{1'b0}};
                bus_ramp_active[i] <= 1'b0;
                bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}};
                bus_ramp_delta[i] <= {(BUS_WIDTH+1){1'b0}};
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
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
                                if (bus_value_active[bus_loop] < bus_ramp_target[bus_loop])
                                    bus_value_active[bus_loop] <= bus_value_active[bus_loop] + 1'b1;
                            end else begin
                                if (bus_value_active[bus_loop] > bus_ramp_target[bus_loop])
                                    bus_value_active[bus_loop] <= bus_value_active[bus_loop] - 1'b1;
                            end
                        end else begin
                            bus_ramp_accum[bus_loop] <= bus_accum_next;
                        end
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

    // ARM sequencer: pre-read shadow edges (0, loop_start, prog_count-1) + scan
    // point 0 from BRAM into shadow regs while reset is asserted, so the gapless
    // reload sites need no table read.
    localparam [2:0]
        A_E0 = 3'd0, A_LS = 3'd1, A_FIN = 3'd2, A_SC0 = 3'd3, A_READY = 3'd4;
    reg [2:0] arm_state = A_E0;
    reg [$clog2(RD_SETTLE+1)-1:0] arm_wait;
    reg arm_kicked;

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
            running <= 1'b0;
            done <= 1'b0;
            state_mask <= {CHANNEL_COUNT{1'b0}};
            time_count <= {TICK_WIDTH{1'b0}};
            edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
            scan_point_index <= {(SCAN_ADDR_WIDTH+1){1'b0}};
            loops_remaining <= 32'd1;
            pre_valid <= 1'b0;
            zlc_bus_clear_runtime();
            // ARM: walk the shadow pre-reads, one BRAM read per state.
            case (arm_state)
                A_E0: begin
                    edge_raddr <= {EDGE_ADDR_WIDTH{1'b0}};
                    if (arm_kicked) begin
                        if (arm_wait == 0) begin
                            first_tick_shadow <= row_base(edge_rdata);
                            first_coeffs_shadow <= row_coeffs(edge_rdata);
                            first_mask_shadow <= row_mask(edge_rdata);
                            edge_raddr <= loop_start_addr; arm_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0]; arm_state <= A_LS;
                        end else arm_wait <= arm_wait - 1'b1;
                    end else begin
                        arm_kicked <= 1'b1; arm_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                    end
                end
                A_LS: if (arm_wait == 0) begin
                    loop_start_tick_shadow <= row_base(edge_rdata);
                    loop_start_coeffs_shadow <= row_coeffs(edge_rdata);
                    loop_start_mask_shadow <= row_mask(edge_rdata);
                    edge_raddr <= (prog_count == 0) ? {EDGE_ADDR_WIDTH{1'b0}} : (prog_count[EDGE_ADDR_WIDTH-1:0] - 1'b1);
                    arm_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0]; arm_state <= A_FIN;
                end else arm_wait <= arm_wait - 1'b1;
                A_FIN: if (arm_wait == 0) begin
                    final_tick_shadow <= row_base(edge_rdata);
                    final_coeffs_shadow <= row_coeffs(edge_rdata);
                    scan_raddr <= {SCAN_ADDR_WIDTH{1'b0}};
                    arm_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0]; arm_state <= A_SC0;
                end else arm_wait <= arm_wait - 1'b1;
                A_SC0: if (arm_wait == 0) begin
                    scan_first_values <= (scan_enable && scan_count != 0) ? scan_rdata : {SLOT_BITS{1'b0}};
                    arm_state <= A_READY;
                end else arm_wait <= arm_wait - 1'b1;
                A_READY: ; // armed; hold until reset releases
                default: arm_state <= A_E0;
            endcase
        end else if (start_event && !running) begin
            running <= (prog_count != 0);
            done <= (prog_count == 0);
            active_count <= prog_count;
            repeat_forever_active <= repeat_forever;
            scan_enable_active <= scan_enable && scan_count != 0;
            active_scan_count <= scan_count;
            slot_active <= scan_first_values;
            scan_point_index <= {(SCAN_ADDR_WIDTH+1){1'b0}};
            scan_raddr <= {{(SCAN_ADDR_WIDTH-1){1'b0}}, 1'b1};   // pre-read scan point 1
            loop_count_active <= (loop_count == 0) ? 32'd1 : loop_count;
            loops_remaining <= (loop_count == 0) ? 32'd1 : loop_count;
            final_tick <= (prog_count == 0) ? {TICK_WIDTH{1'b0}} : zlc_effective_tick(final_tick_shadow, final_coeffs_shadow, scan_first_values);
            loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, scan_first_values);
            zlc_bus_start_table(scan_first_values);
            if ((prog_count != 0) && zlc_effective_tick(first_tick_shadow, first_coeffs_shadow, scan_first_values) == {TICK_WIDTH{1'b0}}) begin
                state_mask <= first_mask_shadow;
                time_count <= {{(TICK_WIDTH-1){1'b0}}, 1'b1};
                edge_index <= {{EDGE_ADDR_WIDTH{1'b0}}, 1'b1};
                edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}}, 1'b1};   // read edge 1
                pre_valid <= 1'b0; rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
            end else begin
                state_mask <= {CHANNEL_COUNT{1'b0}};
                time_count <= {TICK_WIDTH{1'b0}};
                edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};        // edge 0 -> shadow
                pre_valid <= 1'b0;
            end
        end else if (running) begin
            // --- depth-1 read service (default; fire/boundary below may override) ---
            if (!pre_valid) begin
                if (rd_wait == 0) begin
                    pre_tick <= row_base(edge_rdata);
                    pre_coeffs <= row_coeffs(edge_rdata);
                    pre_mask <= row_mask(edge_rdata);
                    pre_valid <= 1'b1;
                end else rd_wait <= rd_wait - 1'b1;
            end

            if (loop_count_active > 32'd1 && loops_remaining > 32'd1 && time_count >= loop_end_active) begin
                state_mask <= loop_start_mask_shadow;
                time_count <= zlc_effective_tick(loop_start_tick_shadow, loop_start_coeffs_shadow, slot_active) + 1'b1;
                edge_index <= {1'b0, loop_start_addr} + 1'b1;
                loops_remaining <= loops_remaining - 1'b1;
                edge_raddr <= loop_start_addr + 1'b1;
                pre_valid <= 1'b0; rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                zlc_bus_start_table(slot_active);
            end else if (time_count >= final_tick) begin
                if (scan_enable_active && (scan_point_index + 1'b1) < active_scan_count) begin
                    scan_point_index <= scan_point_index + 1'b1;
                    scan_raddr <= scan_point_index[SCAN_ADDR_WIDTH-1:0] + 2'd2;  // pre-read the following point
                    slot_active <= scan_rdata;     // point (scan_point_index+1), pre-read
                    final_tick <= zlc_effective_tick(final_tick_shadow, final_coeffs_shadow, scan_rdata);
                    loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, scan_rdata);
                    loops_remaining <= loop_count_active;
                    done <= 1'b0;
                    zlc_bus_start_table(scan_rdata);
                    if (zlc_effective_tick(first_tick_shadow, first_coeffs_shadow, scan_rdata) == {TICK_WIDTH{1'b0}}) begin
                        state_mask <= first_mask_shadow;
                        time_count <= {{(TICK_WIDTH-1){1'b0}}, 1'b1};
                        edge_index <= {{EDGE_ADDR_WIDTH{1'b0}}, 1'b1};
                        edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}}, 1'b1};
                        pre_valid <= 1'b0; rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                    end else begin
                        state_mask <= {CHANNEL_COUNT{1'b0}};
                        time_count <= {TICK_WIDTH{1'b0}};
                        edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
                        pre_valid <= 1'b0;
                    end
                end else if (repeat_forever_active) begin
                    slot_active <= scan_first_values;
                    scan_point_index <= {(SCAN_ADDR_WIDTH+1){1'b0}};
                    scan_raddr <= {{(SCAN_ADDR_WIDTH-1){1'b0}}, 1'b1};
                    final_tick <= zlc_effective_tick(final_tick_shadow, final_coeffs_shadow, scan_first_values);
                    loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, scan_first_values);
                    loops_remaining <= loop_count_active;
                    done <= 1'b0;
                    zlc_bus_start_table(scan_first_values);
                    if (zlc_effective_tick(first_tick_shadow, first_coeffs_shadow, scan_first_values) == {TICK_WIDTH{1'b0}}) begin
                        state_mask <= first_mask_shadow;
                        time_count <= {{(TICK_WIDTH-1){1'b0}}, 1'b1};
                        edge_index <= {{EDGE_ADDR_WIDTH{1'b0}}, 1'b1};
                        edge_raddr <= {{(EDGE_ADDR_WIDTH-1){1'b0}}, 1'b1};
                        pre_valid <= 1'b0; rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                    end else begin
                        state_mask <= {CHANNEL_COUNT{1'b0}};
                        time_count <= {TICK_WIDTH{1'b0}};
                        edge_index <= {(EDGE_ADDR_WIDTH+1){1'b0}};
                        pre_valid <= 1'b0;
                    end
                end else begin
                    running <= 1'b0;
                    done <= 1'b1;
                    state_mask <= {CHANNEL_COUNT{1'b0}};
                    zlc_bus_clear_runtime();
                end
            end else begin
                zlc_bus_step();
                if (edge_index < active_count && cur_avail && time_count == cur_eff) begin
                    state_mask <= cur_mask;
                    edge_index <= edge_index + 1'b1;
                    // issue the depth-1 read for the new current edge (>=1).
                    edge_raddr <= edge_index[EDGE_ADDR_WIDTH-1:0] + 1'b1;
                    pre_valid <= 1'b0;
                    rd_wait <= RD_SETTLE[$clog2(RD_SETTLE+1)-1:0];
                end
                time_count <= time_count + 1'b1;
            end
        end
    end
endmodule
