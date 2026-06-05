`timescale 1ns / 1ps
// Runtime-programmable edge-table pulse streamer.
//
// Host upload contract:
//   1. Hold reset high.
//   2. Set prog_count to the number of active edges.
//   3. For each edge, set prog_addr/prog_tick/prog_mask and toggle prog_we.
//      The core writes once per synchronized prog_we transition while reset is
//      high. This avoids level-write corruption while VIO updates addr/data.
//   4. Optionally upload per-scan analog-bus values through scan_prog_*.
//   5. Optionally upload non-scan analog-bus segments through bus_prog_* probes.
//   6. Set repeat_forever plus optional loop_start_addr/loop_end_tick/loop_count.
//   7. Release reset.
//   8. Pulse start.  The FPGA may repeat the table forever and/or loop a
//      sub-table in hardware without receiving expanded edges from Vivado.

module zlc_pulse_streamer #(
    parameter integer CHANNEL_COUNT = 4,
    parameter integer EDGE_ADDR_WIDTH = 9,
    parameter integer TICK_WIDTH = 32,
    parameter integer SCAN_ADDR_WIDTH = 8,
    parameter integer COEFF_WIDTH = 16,
    parameter integer COEFF_FRAC_BITS = 8,
    parameter integer BUS_COUNT = 4,
    parameter integer BUS_INDEX_WIDTH = 2,
    parameter integer BUS_WIDTH = 10,
    parameter integer BUS_SEG_ADDR_WIDTH = 6
)(
    input wire clk,
    input wire reset,
    input wire start,
    input wire prog_we,
    input wire [EDGE_ADDR_WIDTH-1:0] prog_addr,
    input wire [TICK_WIDTH-1:0] prog_tick,
    input wire signed [COEFF_WIDTH-1:0] prog_tick_x_coeff,
    input wire signed [COEFF_WIDTH-1:0] prog_tick_y_coeff,
    input wire [CHANNEL_COUNT-1:0] prog_mask,
    input wire [EDGE_ADDR_WIDTH:0] prog_count,
    input wire repeat_forever,
    input wire [EDGE_ADDR_WIDTH-1:0] loop_start_addr,
    input wire [TICK_WIDTH-1:0] loop_end_tick,
    input wire signed [COEFF_WIDTH-1:0] loop_end_x_coeff,
    input wire signed [COEFF_WIDTH-1:0] loop_end_y_coeff,
    input wire [31:0] loop_count,
    input wire scan_enable,
    input wire scan_prog_we,
    input wire [SCAN_ADDR_WIDTH-1:0] scan_prog_addr,
    input wire signed [TICK_WIDTH-1:0] scan_prog_x,
    input wire signed [TICK_WIDTH-1:0] scan_prog_y,
    input wire [BUS_COUNT*BUS_WIDTH-1:0] scan_prog_bus_values,
    input wire [SCAN_ADDR_WIDTH:0] scan_count,
    input wire bus_prog_we,
    input wire [BUS_INDEX_WIDTH-1:0] bus_prog_bus,
    input wire [BUS_SEG_ADDR_WIDTH-1:0] bus_prog_addr,
    input wire [TICK_WIDTH-1:0] bus_prog_start_tick,
    input wire [TICK_WIDTH-1:0] bus_prog_stop_tick,
    input wire [BUS_WIDTH-1:0] bus_prog_start_value,
    input wire [BUS_WIDTH-1:0] bus_prog_stop_value,
    input wire [1:0] bus_prog_mode,
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
    localparam [1:0] BUS_MODE_EDGE = 2'd1;
    localparam [1:0] BUS_MODE_RAMP = 2'd2;

    // Keep the runtime read side to one table address.  Vivado otherwise has
    // to synthesize a multi-read-port memory, which is expensive on Artix-7.
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] tick_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg signed [COEFF_WIDTH-1:0] x_coeff_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg signed [COEFF_WIDTH-1:0] y_coeff_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg [CHANNEL_COUNT-1:0] mask_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg signed [TICK_WIDTH-1:0] scan_x_mem [0:MAX_SCAN_POINTS-1];
    (* ram_style = "distributed" *) reg signed [TICK_WIDTH-1:0] scan_y_mem [0:MAX_SCAN_POINTS-1];
    (* ram_style = "distributed" *) reg [BUS_COUNT*BUS_WIDTH-1:0] scan_bus_value_mem [0:MAX_SCAN_POINTS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_start_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] bus_stop_tick_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_start_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [BUS_WIDTH-1:0] bus_stop_value_mem [0:MAX_BUS_SEGMENT_ROWS-1];
    (* ram_style = "distributed" *) reg [1:0] bus_mode_mem [0:MAX_BUS_SEGMENT_ROWS-1];

    reg [CHANNEL_COUNT-1:0] state_mask = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] time_count = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick_shadow = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] first_tick_shadow = {TICK_WIDTH{1'b0}};
    reg signed [COEFF_WIDTH-1:0] first_x_coeff_shadow = {COEFF_WIDTH{1'b0}};
    reg signed [COEFF_WIDTH-1:0] first_y_coeff_shadow = {COEFF_WIDTH{1'b0}};
    reg [CHANNEL_COUNT-1:0] first_mask_shadow = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] loop_start_tick_shadow = {TICK_WIDTH{1'b0}};
    reg signed [COEFF_WIDTH-1:0] loop_start_x_coeff_shadow = {COEFF_WIDTH{1'b0}};
    reg signed [COEFF_WIDTH-1:0] loop_start_y_coeff_shadow = {COEFF_WIDTH{1'b0}};
    reg [CHANNEL_COUNT-1:0] loop_start_mask_shadow = {CHANNEL_COUNT{1'b0}};
    reg signed [COEFF_WIDTH-1:0] final_x_coeff_shadow = {COEFF_WIDTH{1'b0}};
    reg signed [COEFF_WIDTH-1:0] final_y_coeff_shadow = {COEFF_WIDTH{1'b0}};
    reg [EDGE_ADDR_WIDTH:0] edge_index = {(EDGE_ADDR_WIDTH + 1){1'b0}};
    reg [EDGE_ADDR_WIDTH:0] active_count = {(EDGE_ADDR_WIDTH + 1){1'b0}};
    reg repeat_forever_active = 1'b0;
    reg scan_enable_active = 1'b0;
    reg [EDGE_ADDR_WIDTH-1:0] loop_start_active = {EDGE_ADDR_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] loop_end_active = {TICK_WIDTH{1'b0}};
    reg signed [TICK_WIDTH-1:0] scan_x_active = {TICK_WIDTH{1'b0}};
    reg signed [TICK_WIDTH-1:0] scan_y_active = {TICK_WIDTH{1'b0}};
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
    wire [TICK_WIDTH-1:0] current_edge_tick = zlc_effective_tick(
        tick_mem[edge_addr],
        x_coeff_mem[edge_addr],
        y_coeff_mem[edge_addr],
        scan_x_active,
        scan_y_active
    );

    assign out = state_mask;

    genvar bus_out_index;
    generate
        for (bus_out_index = 0; bus_out_index < BUS_COUNT; bus_out_index = bus_out_index + 1) begin : zlc_bus_out_assign
            assign bus_out[bus_out_index*BUS_WIDTH +: BUS_WIDTH] = bus_value_active[bus_out_index];
        end
    endgenerate

    function [TICK_WIDTH-1:0] zlc_effective_tick;
        input [TICK_WIDTH-1:0] base_tick;
        input signed [COEFF_WIDTH-1:0] x_coeff;
        input signed [COEFF_WIDTH-1:0] y_coeff;
        input signed [TICK_WIDTH-1:0] x_value;
        input signed [TICK_WIDTH-1:0] y_value;
        reg signed [TICK_WIDTH+COEFF_WIDTH:0] delta;
        reg signed [TICK_WIDTH+COEFF_WIDTH:0] total;
        begin
            delta = (($signed(x_coeff) * $signed(x_value)) + ($signed(y_coeff) * $signed(y_value))) >>> COEFF_FRAC_BITS;
            total = $signed({1'b0, base_tick}) + delta;
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

    task zlc_bus_apply_segment;
        input integer i;
        input [BUS_INDEX_WIDTH+BUS_SEG_ADDR_WIDTH-1:0] addr;
        reg [TICK_WIDTH-1:0] span;
        begin
            if (bus_mode_mem[addr] == BUS_MODE_RAMP && bus_stop_tick_mem[addr] > bus_start_tick_mem[addr]) begin
                span = bus_stop_tick_mem[addr] - bus_start_tick_mem[addr];
                bus_value_active[i] <= bus_start_value_mem[addr];
                bus_ramp_active[i] <= 1'b1;
                bus_ramp_start_tick[i] <= bus_start_tick_mem[addr];
                bus_ramp_stop_tick[i] <= bus_stop_tick_mem[addr];
                bus_ramp_target[i] <= bus_stop_value_mem[addr];
                bus_ramp_denom[i] <= span;
                bus_ramp_accum[i] <= {(TICK_WIDTH + BUS_WIDTH + 1){1'b0}};
                if (bus_stop_value_mem[addr] >= bus_start_value_mem[addr]) begin
                    bus_ramp_dir_up[i] <= 1'b1;
                    bus_ramp_delta[i] <= bus_stop_value_mem[addr] - bus_start_value_mem[addr];
                end else begin
                    bus_ramp_dir_up[i] <= 1'b0;
                    bus_ramp_delta[i] <= bus_start_value_mem[addr] - bus_stop_value_mem[addr];
                end
            end else begin
                bus_value_active[i] <= bus_stop_value_mem[addr];
                bus_ramp_active[i] <= 1'b0;
                bus_ramp_accum[i] <= {(TICK_WIDTH + BUS_WIDTH + 1){1'b0}};
            end
        end
    endtask

    task zlc_bus_start_table;
        input scan_active;
        input [SCAN_ADDR_WIDTH-1:0] scan_addr;
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
                if (scan_active) begin
                    bus_value_active[i] <= scan_bus_value_mem[scan_addr][i*BUS_WIDTH +: BUS_WIDTH];
                end else if (count != 0 && bus_start_tick_mem[addr] == {TICK_WIDTH{1'b0}}) begin
                    zlc_bus_apply_segment(i, addr);
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
                            if (bus_start_tick_mem[bus_runtime_addr] <= time_count) begin
                                zlc_bus_apply_segment(bus_loop, bus_runtime_addr);
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
                    if (time_count >= bus_start_tick_mem[bus_runtime_addr]) begin
                        zlc_bus_apply_segment(bus_loop, bus_runtime_addr);
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
            x_coeff_mem[prog_addr] <= prog_tick_x_coeff;
            y_coeff_mem[prog_addr] <= prog_tick_y_coeff;
            mask_mem[prog_addr] <= prog_mask;
            if (prog_addr == {EDGE_ADDR_WIDTH{1'b0}}) begin
                first_tick_shadow <= prog_tick;
                first_x_coeff_shadow <= prog_tick_x_coeff;
                first_y_coeff_shadow <= prog_tick_y_coeff;
                first_mask_shadow <= prog_mask;
            end
            if (prog_addr == loop_start_addr) begin
                loop_start_tick_shadow <= prog_tick;
                loop_start_x_coeff_shadow <= prog_tick_x_coeff;
                loop_start_y_coeff_shadow <= prog_tick_y_coeff;
                loop_start_mask_shadow <= prog_mask;
            end
            if (prog_count != 0 && prog_addr == (prog_count[EDGE_ADDR_WIDTH-1:0] - 1'b1)) begin
                final_tick_shadow <= prog_tick;
                final_x_coeff_shadow <= prog_tick_x_coeff;
                final_y_coeff_shadow <= prog_tick_y_coeff;
            end
        end
        if (reset_sync && scan_prog_we_event) begin
            scan_x_mem[scan_prog_addr] <= scan_prog_x;
            scan_y_mem[scan_prog_addr] <= scan_prog_y;
            scan_bus_value_mem[scan_prog_addr] <= scan_prog_bus_values;
        end
        if (reset_sync && bus_prog_we_event && bus_prog_bus < BUS_COUNT) begin
            bus_prog_flat_addr = {bus_prog_bus, bus_prog_addr};
            bus_start_tick_mem[bus_prog_flat_addr] <= bus_prog_start_tick;
            bus_stop_tick_mem[bus_prog_flat_addr] <= bus_prog_stop_tick;
            bus_start_value_mem[bus_prog_flat_addr] <= bus_prog_start_value;
            bus_stop_value_mem[bus_prog_flat_addr] <= bus_prog_stop_value;
            bus_mode_mem[bus_prog_flat_addr] <= bus_prog_mode;
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
            scan_x_active <= {TICK_WIDTH{1'b0}};
            scan_y_active <= {TICK_WIDTH{1'b0}};
            active_scan_count <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
            scan_point_index <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
            loop_count_active <= 32'd1;
            loops_remaining <= 32'd1;
            zlc_bus_clear_runtime();
        end else if (start_event && !running) begin
            running <= (prog_count != 0);
            done <= (prog_count == 0);
            zlc_bus_start_table(scan_enable && scan_count != 0, {SCAN_ADDR_WIDTH{1'b0}});
            scan_x_active <= (scan_enable && scan_count != 0) ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}};
            scan_y_active <= (scan_enable && scan_count != 0) ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}};
            final_tick <= (prog_count == 0) ? {TICK_WIDTH{1'b0}} : zlc_effective_tick(
                final_tick_shadow,
                final_x_coeff_shadow,
                final_y_coeff_shadow,
                (scan_enable && scan_count != 0) ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}},
                (scan_enable && scan_count != 0) ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}
            );
            active_count <= prog_count;
            repeat_forever_active <= repeat_forever;
            scan_enable_active <= scan_enable && scan_count != 0;
            active_scan_count <= scan_count;
            scan_point_index <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
            loop_start_active <= loop_start_addr;
            loop_end_active <= zlc_effective_tick(
                loop_end_tick,
                loop_end_x_coeff,
                loop_end_y_coeff,
                (scan_enable && scan_count != 0) ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}},
                (scan_enable && scan_count != 0) ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}
            );
            loop_count_active <= (loop_count == 0) ? 32'd1 : loop_count;
            loops_remaining <= (loop_count == 0) ? 32'd1 : loop_count;
            if (prog_count != 0 && zlc_effective_tick(first_tick_shadow, first_x_coeff_shadow, first_y_coeff_shadow, (scan_enable && scan_count != 0) ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}, (scan_enable && scan_count != 0) ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}) == {TICK_WIDTH{1'b0}}) begin
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
                time_count <= zlc_effective_tick(loop_start_tick_shadow, loop_start_x_coeff_shadow, loop_start_y_coeff_shadow, scan_x_active, scan_y_active) + 1'b1;
                edge_index <= {1'b0, loop_start_active} + 1'b1;
                loops_remaining <= loops_remaining - 1'b1;
                zlc_bus_start_table(scan_enable_active, scan_point_index[SCAN_ADDR_WIDTH-1:0]);
            end else if (time_count >= final_tick) begin
                if (scan_enable_active && (scan_point_index + 1'b1) < active_scan_count) begin
                    zlc_bus_start_table(1'b1, next_scan_addr);
                    scan_point_index <= scan_point_index + 1'b1;
                    scan_x_active <= scan_x_mem[next_scan_addr];
                    scan_y_active <= scan_y_mem[next_scan_addr];
                    final_tick <= zlc_effective_tick(final_tick_shadow, final_x_coeff_shadow, final_y_coeff_shadow, scan_x_mem[next_scan_addr], scan_y_mem[next_scan_addr]);
                    loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_x_coeff, loop_end_y_coeff, scan_x_mem[next_scan_addr], scan_y_mem[next_scan_addr]);
                    if (zlc_effective_tick(first_tick_shadow, first_x_coeff_shadow, first_y_coeff_shadow, scan_x_mem[next_scan_addr], scan_y_mem[next_scan_addr]) == {TICK_WIDTH{1'b0}}) begin
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
                    zlc_bus_start_table(scan_enable_active, {SCAN_ADDR_WIDTH{1'b0}});
                    scan_point_index <= {(SCAN_ADDR_WIDTH + 1){1'b0}};
                    scan_x_active <= scan_enable_active ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}};
                    scan_y_active <= scan_enable_active ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}};
                    final_tick <= zlc_effective_tick(final_tick_shadow, final_x_coeff_shadow, final_y_coeff_shadow, scan_enable_active ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}, scan_enable_active ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}});
                    loop_end_active <= zlc_effective_tick(loop_end_tick, loop_end_x_coeff, loop_end_y_coeff, scan_enable_active ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}, scan_enable_active ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}});
                    if (zlc_effective_tick(first_tick_shadow, first_x_coeff_shadow, first_y_coeff_shadow, scan_enable_active ? scan_x_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}, scan_enable_active ? scan_y_mem[{SCAN_ADDR_WIDTH{1'b0}}] : {TICK_WIDTH{1'b0}}) == {TICK_WIDTH{1'b0}}) begin
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
