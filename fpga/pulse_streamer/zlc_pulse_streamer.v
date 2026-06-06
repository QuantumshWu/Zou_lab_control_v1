`timescale 1ns / 1ps
// Runtime-programmable edge-table pulse streamer with N-slot affine scanning.
//
// Each edge stores an absolute base tick plus one signed coefficient per scan
// slot.  At runtime the effective tick of an edge is
//
//     effective_tick = base_tick + (sum_j coeff_j * slot_j) >>> COEFF_FRAC_BITS
//
// where slot_j is the j-th 32-bit value of the current scan point.  Coefficients
// and per-point slot values are packed little-endian into single wide probes so
// one VIO write programs a whole edge / scan point.  With NUM_SLOTS=2 this is
// exactly the old (x, y) affine engine; NUM_SLOTS is sized for the few fields
// scanned at once (durations / delays), not for the channel count.
//
// Host upload contract:
//   1. Hold reset high.
//   2. Set prog_count to the number of active edges.
//   3. For each edge set prog_addr/prog_tick/prog_tick_coeffs/prog_mask and
//      toggle prog_we (edge-triggered write while reset is high).
//   4. Optionally write the scan-point table through scan_prog_* and the analog
//      bus segments through bus_prog_*.  A bus segment whose bus_prog_value_select
//      is j+1 takes its DAC code from scan slot j at runtime instead of the
//      uploaded literal, which is how analog (DAC) values are scanned seamlessly.
//   5. Set repeat_forever / loop_* / scan_enable / scan_count.
//   6. Release reset, then pulse start.

module zlc_pulse_streamer #(
    parameter integer CHANNEL_COUNT = 4,
    parameter integer EDGE_ADDR_WIDTH = 10,
    parameter integer TICK_WIDTH = 32,
    parameter integer SCAN_ADDR_WIDTH = 10,
    parameter integer NUM_SLOTS = 4,
    parameter integer COEFF_WIDTH = 16,
    parameter integer COEFF_FRAC_BITS = 8,
    parameter integer BUS_COUNT = 4,
    parameter integer BUS_INDEX_WIDTH = 2,
    parameter integer BUS_WIDTH = 10,
    parameter integer BUS_SEG_ADDR_WIDTH = 6,
    parameter integer BUS_SEL_WIDTH = 3
)(
    input wire clk,
    input wire reset,
    input wire start,
    input wire prog_we,
    input wire [EDGE_ADDR_WIDTH-1:0] prog_addr,
    input wire [TICK_WIDTH-1:0] prog_tick,
    input wire [NUM_SLOTS*COEFF_WIDTH-1:0] prog_tick_coeffs,
    input wire [CHANNEL_COUNT-1:0] prog_mask,
    input wire [EDGE_ADDR_WIDTH:0] prog_count,
    input wire repeat_forever,
    input wire [EDGE_ADDR_WIDTH-1:0] loop_start_addr,
    input wire [TICK_WIDTH-1:0] loop_end_tick,
    input wire [NUM_SLOTS*COEFF_WIDTH-1:0] loop_end_coeffs,
    input wire [31:0] loop_count,
    input wire scan_enable,
    input wire scan_prog_we,
    input wire [SCAN_ADDR_WIDTH-1:0] scan_prog_addr,
    input wire [NUM_SLOTS*TICK_WIDTH-1:0] scan_prog_values,
    input wire [SCAN_ADDR_WIDTH:0] scan_count,
    input wire bus_prog_we,
    input wire [BUS_INDEX_WIDTH-1:0] bus_prog_bus,
    input wire [BUS_SEG_ADDR_WIDTH-1:0] bus_prog_addr,
    input wire [TICK_WIDTH-1:0] bus_prog_start_tick,
    input wire [TICK_WIDTH-1:0] bus_prog_stop_tick,
    // Per-segment affine tick coefficients (one signed COEFF_WIDTH per slot,
    // packed little-endian).  effective_tick = base + (sum coeff_j*slot_j)>>FRAC,
    // so a scanned duration/delay moves the bus segment in lockstep with the
    // digital edges -- this is what lets DA + duration + delay scan together.
    input wire [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_start_tick_coeffs,
    input wire [NUM_SLOTS*COEFF_WIDTH-1:0] bus_prog_stop_tick_coeffs,
    input wire [BUS_WIDTH-1:0] bus_prog_start_value,
    input wire [BUS_WIDTH-1:0] bus_prog_stop_value,
    input wire [1:0] bus_prog_mode,
    // Per-segment value source: 0 = use the literal start/stop value above;
    // j+1 = take the DAC code from scan slot j (slot_active[j]) so a bus level
    // tracks a scan point seamlessly (the DAC-value scan path).
    input wire [BUS_SEL_WIDTH-1:0] bus_prog_value_select,
    input wire [BUS_COUNT*(BUS_SEG_ADDR_WIDTH+1)-1:0] bus_counts,
    output wire [CHANNEL_COUNT-1:0] out,
    output wire [BUS_COUNT*BUS_WIDTH-1:0] bus_out,
    output reg running = 1'b0,
    output reg done = 1'b0
);

    localparam integer MAX_EDGES = (1 << EDGE_ADDR_WIDTH);
    localparam integer MAX_SCAN_POINTS = (1 << SCAN_ADDR_WIDTH);
    localparam integer MAX_BUS_SEGMENTS = (1 << BUS_SEG_ADDR_WIDTH);
    localparam integer MAX_BUS_SEGMENT_ROWS = BUS_COUNT * MAX_BUS_SEGMENTS;
    localparam integer COEFF_BITS = NUM_SLOTS * COEFF_WIDTH;
    localparam integer SLOT_BITS = NUM_SLOTS * TICK_WIDTH;
    localparam integer ACC_WIDTH = TICK_WIDTH + COEFF_WIDTH + 4;
    localparam [1:0] BUS_MODE_EDGE = 2'd1;
    localparam [1:0] BUS_MODE_RAMP = 2'd2;

    // Keep the runtime read side to one table address.  Vivado otherwise has
    // to synthesize a multi-read-port memory, which is expensive on Artix-7.
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] tick_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] coeff_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg [CHANNEL_COUNT-1:0] mask_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg [SLOT_BITS-1:0] scan_value_mem [0:MAX_SCAN_POINTS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_start_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_stop_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_start_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_stop_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [1:0] bus_mode_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_SEL_WIDTH-1:0] bus_value_select_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_start_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [COEFF_BITS-1:0] bus_stop_tick_coeff_mem [0:MAX_BUS_SEGMENT_ROWS-1];

    reg [CHANNEL_COUNT-1:0] state_mask = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] time_count = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick_shadow = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] first_tick_shadow = {TICK_WIDTH{1'b0}};
    reg [COEFF_BITS-1:0] first_coeffs_shadow = {COEFF_BITS{1'b0}};
    reg [CHANNEL_COUNT-1:0] first_mask_shadow = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] loop_start_tick_shadow = {TICK_WIDTH{1'b0}};
    reg [COEFF_BITS-1:0] loop_start_coeffs_shadow = {COEFF_BITS{1'b0}};
    reg [CHANNEL_COUNT-1:0] loop_start_mask_shadow = {CHANNEL_COUNT{1'b0}};
    reg [COEFF_BITS-1:0] final_coeffs_shadow = {COEFF_BITS{1'b0}};
    reg [EDGE_ADDR_WIDTH:0] edge_index = {(EDGE_ADDR_WIDTH + 1){1'b0}};
    reg [EDGE_ADDR_WIDTH:0] active_count = {(EDGE_ADDR_WIDTH + 1){1'b0}};
    reg repeat_forever_active = 1'b0;
    reg scan_enable_active = 1'b0;
    reg [EDGE_ADDR_WIDTH-1:0] loop_start_active = {EDGE_ADDR_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] loop_end_active = {TICK_WIDTH{1'b0}};
    reg [SLOT_BITS-1:0] slot_active = {SLOT_BITS{1'b0}};
    reg [SCAN_ADDR_WIDTH:0] active_scan_count = {(SCAN_ADDR_WIDTH + 1){1'b0}};
    reg [SCAN_ADDR_WIDTH:0] scan_point_index = {(SCAN_ADDR_WIDTH + 1){1'b0}};
    reg [31:0] loop_count_active = 32'd1;
    reg [31:0] loops_remaining = 32'd1;
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

    reg reset_meta = 1'b0;
    reg reset_sync = 1'b0;
    reg start_meta = 1'b0;
    reg start_sync = 1'b0;
    reg start_prev = 1'b0;
    reg prog_we_meta = 1'b0;
    reg prog_we_sync = 1'b0;
    reg prog_we_prev = 1'b0;
    reg scan_prog_we_meta = 1'b0;
    reg scan_prog_we_sync = 1'b0;
    reg scan_prog_we_prev = 1'b0;
    reg bus_prog_we_meta = 1'b0;
    reg bus_prog_we_sync = 1'b0;
    reg bus_prog_we_prev = 1'b0;

    integer bus_loop;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_prog_flat_addr;
    reg [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] bus_runtime_addr;
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_accum_next;

    wire start_event = start_sync && !start_prev;
    wire prog_we_event = prog_we_sync != prog_we_prev;
    wire scan_prog_we_event = scan_prog_we_sync != scan_prog_we_prev;
    wire bus_prog_we_event = bus_prog_we_sync != bus_prog_we_prev;
    wire [EDGE_ADDR_WIDTH-1:0] edge_addr = edge_index[EDGE_ADDR_WIDTH-1:0];
    wire [SCAN_ADDR_WIDTH:0] next_scan_point_index = scan_point_index + 1'b1;
    wire [SCAN_ADDR_WIDTH-1:0] next_scan_addr = next_scan_point_index[SCAN_ADDR_WIDTH-1:0];
    wire [SLOT_BITS-1:0] scan_first_values =
        (scan_enable && scan_count != 0) ? scan_value_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {SLOT_BITS{1'b0}};
    wire [TICK_WIDTH-1:0] current_edge_tick = zlc_effective_tick(
        tick_mem[edge_addr],
        coeff_mem[edge_addr],
        slot_active
    );

    assign out = state_mask;

    genvar bus_out_index;
    generate
        for (bus_out_index = 0; bus_out_index < BUS_COUNT; bus_out_index = bus_out_index + 1) begin : zlc_bus_out_assign
            assign bus_out[bus_out_index*BUS_WIDTH +: BUS_WIDTH] = bus_value_active[bus_out_index];
        end
    endgenerate

    // effective_tick = base + (sum_j coeff_j * slot_j) >>> COEFF_FRAC_BITS
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
        begin
            case (bus_index)
                0: zlc_bus_count_at = bus_counts[BUS_SEG_ADDR_WIDTH:0];
                1: zlc_bus_count_at = bus_counts[(2*(BUS_SEG_ADDR_WIDTH+1))-1:(BUS_SEG_ADDR_WIDTH+1)];
                2: zlc_bus_count_at = bus_counts[(3*(BUS_SEG_ADDR_WIDTH+1))-1:(2*(BUS_SEG_ADDR_WIDTH+1))];
                3: zlc_bus_count_at = bus_counts[(4*(BUS_SEG_ADDR_WIDTH+1))-1:(3*(BUS_SEG_ADDR_WIDTH+1))];
                default: zlc_bus_count_at = {(BUS_SEG_ADDR_WIDTH + 1){1'b0}};
            endcase
        end
    endfunction

    task zlc_bus_clear_runtime;
        integer i;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                bus_value_active[i] <= {BUS_WIDTH{1'b0}};
                bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH + 1){1'b0}};
                bus_count_active[i] <= {(BUS_SEG_ADDR_WIDTH + 1){1'b0}};
                bus_ramp_active[i] <= 1'b0;
                bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}};
                bus_ramp_delta[i] <= {(BUS_WIDTH + 1){1'b0}};
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_accum[i] <= {(TICK_WIDTH + BUS_WIDTH + 1){1'b0}};
            end
        end
    endtask

    // Apply one segment to bus i.  ``slot_vec`` is the slot vector that is in
    // force for the current scan point; it is passed explicitly (rather than read
    // from slot_active) so the value is correct on the very cycle a scan point
    // is entered, before the slot_active register has latched the new point.
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
                // Scanned DAC level: take the low BUS_WIDTH bits of slot (sel-1).
                eff_val_start = slot_vec[(sel - 1'b1)*TICK_WIDTH +: BUS_WIDTH];
                eff_val_stop = eff_val_start;
            end else begin
                eff_val_start = bus_start_value_mem[addr];
                eff_val_stop = bus_stop_value_mem[addr];
            end
            // Affine segment ticks: move with any scanned duration/delay.
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
                bus_ramp_accum[i] <= {(TICK_WIDTH + BUS_WIDTH + 1){1'b0}};
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
                bus_ramp_accum[i] <= {(TICK_WIDTH + BUS_WIDTH + 1){1'b0}};
            end
        end
    endtask

    // Effective (scan-point-resolved) start tick of bus segment ``addr``.
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
                bus_index_active[i] <= {(BUS_SEG_ADDR_WIDTH + 1){1'b0}};
                bus_value_active[i] <= {BUS_WIDTH{1'b0}};
                bus_ramp_active[i] <= 1'b0;
                bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_start_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_stop_tick[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}};
                bus_ramp_delta[i] <= {(BUS_WIDTH + 1){1'b0}};
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_accum[i] <= {(TICK_WIDTH + BUS_WIDTH + 1){1'b0}};
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
                        bus_ramp_accum[bus_loop] <= {(TICK_WIDTH + BUS_WIDTH + 1){1'b0}};
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
                                if (bus_value_active[bus_loop] < bus_ramp_target[bus_loop]) begin
                                    bus_value_active[bus_loop] <= bus_value_active[bus_loop] + 1'b1;
                                end
                            end else begin
                                if (bus_value_active[bus_loop] > bus_ramp_target[bus_loop]) begin
                                    bus_value_active[bus_loop] <= bus_value_active[bus_loop] - 1'b1;
                                end
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

    always @(posedge clk) begin
        reset_meta <= reset;
        reset_sync <= reset_meta;
        start_meta <= start;
        start_sync <= start_meta;
        start_prev <= start_sync;
        prog_we_meta <= prog_we;
        prog_we_sync <= prog_we_meta;
        prog_we_prev <= prog_we_sync;
        scan_prog_we_meta <= scan_prog_we;
        scan_prog_we_sync <= scan_prog_we_meta;
        scan_prog_we_prev <= scan_prog_we_sync;
        bus_prog_we_meta <= bus_prog_we;
        bus_prog_we_sync <= bus_prog_we_meta;
        bus_prog_we_prev <= bus_prog_we_sync;

        if (reset_sync && prog_we_event) begin
            tick_mem[prog_addr] <= prog_tick;
            coeff_mem[prog_addr] <= prog_tick_coeffs;
            mask_mem[prog_addr] <= prog_mask;
            if (prog_addr == {EDGE_ADDR_WIDTH{1'b0}}) begin
                first_tick_shadow <= prog_tick;
                first_coeffs_shadow <= prog_tick_coeffs;
                first_mask_shadow <= prog_mask;
            end
            if (prog_addr == loop_start_addr) begin
                loop_start_tick_shadow <= prog_tick;
                loop_start_coeffs_shadow <= prog_tick_coeffs;
                loop_start_mask_shadow <= prog_mask;
            end
            if (prog_count != 0 && prog_addr == (prog_count[EDGE_ADDR_WIDTH-1:0] - 1'b1)) begin
                final_tick_shadow <= prog_tick;
                final_coeffs_shadow <= prog_tick_coeffs;
            end
        end
        if (reset_sync && scan_prog_we_event) begin
            scan_value_mem[scan_prog_addr] <= scan_prog_values;
        end
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
            final_tick <= {TICK_WIDTH{1'b0}};
            edge_index <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
            active_count <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
            repeat_forever_active <= 1'b0;
            scan_enable_active <= 1'b0;
            loop_start_active <= {EDGE_ADDR_WIDTH{1'b0}};
            loop_end_active <= {TICK_WIDTH{1'b0}};
            slot_active <= {SLOT_BITS{1'b0}};
            active_scan_count <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
            scan_point_index <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
            loop_count_active <= 32'd1;
            loops_remaining <= 32'd1;
            zlc_bus_clear_runtime();
        end else if (start_event && !running) begin
            running <= (prog_count != 0);
            done <= (prog_count == 0);
            zlc_bus_start_table(scan_first_values);
            slot_active <= scan_first_values;
            final_tick <= (prog_count == 0) ? {TICK_WIDTH{1'b0}} : zlc_effective_tick(
                final_tick_shadow,
                final_coeffs_shadow,
                scan_first_values
            );
            active_count <= prog_count;
            repeat_forever_active <= repeat_forever;
            scan_enable_active <= scan_enable && scan_count != 0;
            active_scan_count <= scan_count;
            scan_point_index <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
            loop_start_active <= loop_start_addr;
            loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, scan_first_values);
            loop_count_active <= (loop_count == 0) ? 32'd1 : loop_count;
            loops_remaining <= (loop_count == 0) ? 32'd1 : loop_count;
            if (prog_count != 0 && zlc_effective_tick(first_tick_shadow, first_coeffs_shadow, scan_first_values) == {TICK_WIDTH{1'b0}}) begin
                state_mask <= first_mask_shadow;
                time_count <= {{(TICK_WIDTH-1){1'b0}}, 1'b1};
                edge_index <= {{EDGE_ADDR_WIDTH{1'b0}}, 1'b1};
            end else begin
                state_mask <= {CHANNEL_COUNT{1'b0}};
                time_count <= {TICK_WIDTH{1'b0}};
                edge_index <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
            end
        end else if (running) begin
            if (loop_count_active > 32'd1 && loops_remaining > 32'd1 && time_count >= loop_end_active) begin
                state_mask <= loop_start_mask_shadow;
                time_count <= zlc_effective_tick(loop_start_tick_shadow, loop_start_coeffs_shadow, slot_active) + 1'b1;
                edge_index <= {1'b0, loop_start_active} + 1'b1;
                loops_remaining <= loops_remaining - 1'b1;
                zlc_bus_start_table(slot_active);
            end else if (time_count >= final_tick) begin
                if (scan_enable_active && (scan_point_index + 1'b1) < active_scan_count) begin
                    zlc_bus_start_table(scan_value_mem[next_scan_addr]);
                    scan_point_index <= scan_point_index + 1'b1;
                    slot_active <= scan_value_mem[next_scan_addr];
                    final_tick <= zlc_effective_tick(final_tick_shadow, final_coeffs_shadow, scan_value_mem[next_scan_addr]);
                    loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, scan_value_mem[next_scan_addr]);
                    if (zlc_effective_tick(first_tick_shadow, first_coeffs_shadow, scan_value_mem[next_scan_addr]) == {TICK_WIDTH{1'b0}}) begin
                        state_mask <= first_mask_shadow;
                        time_count <= {{(TICK_WIDTH-1){1'b0}}, 1'b1};
                        edge_index <= {{EDGE_ADDR_WIDTH{1'b0}}, 1'b1};
                    end else begin
                        state_mask <= {CHANNEL_COUNT{1'b0}};
                        time_count <= {TICK_WIDTH{1'b0}};
                        edge_index <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
                    end
                    loops_remaining <= loop_count_active;
                    done <= 1'b0;
                end else if (repeat_forever_active) begin
                    zlc_bus_start_table(scan_enable_active ? scan_value_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {SLOT_BITS{1'b0}});
                    scan_point_index <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
                    slot_active <= scan_enable_active ? scan_value_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {SLOT_BITS{1'b0}};
                    final_tick <= zlc_effective_tick(final_tick_shadow, final_coeffs_shadow, scan_enable_active ? scan_value_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {SLOT_BITS{1'b0}});
                    loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_coeffs, scan_enable_active ? scan_value_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {SLOT_BITS{1'b0}});
                    if (zlc_effective_tick(first_tick_shadow, first_coeffs_shadow, scan_enable_active ? scan_value_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {SLOT_BITS{1'b0}}) == {TICK_WIDTH{1'b0}}) begin
                        state_mask <= first_mask_shadow;
                        time_count <= {{(TICK_WIDTH-1){1'b0}}, 1'b1};
                        edge_index <= {{EDGE_ADDR_WIDTH{1'b0}}, 1'b1};
                    end else begin
                        state_mask <= {CHANNEL_COUNT{1'b0}};
                        time_count <= {TICK_WIDTH{1'b0}};
                        edge_index <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
                    end
                    loops_remaining <= loop_count_active;
                    done <= 1'b0;
                end else begin
                    running <= 1'b0;
                    done <= 1'b1;
                    state_mask <= {CHANNEL_COUNT{1'b0}};
                    zlc_bus_clear_runtime();
                end
            end else begin
                zlc_bus_step();
                if (edge_index < active_count && time_count == current_edge_tick) begin
                    state_mask <= mask_mem[edge_addr];
                    edge_index <= edge_index + 1'b1;
                end
                time_count <= time_count + 1'b1;
            end
        end
    end
endmodule
