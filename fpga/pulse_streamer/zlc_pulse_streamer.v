`timescale 1ns / 1ps
// Runtime-programmable edge-table pulse streamer.
//
// Host upload contract:
//   1. Hold reset high.
//   2. Set prog_count to the number of active edges.
//   3. For each edge, set prog_addr/prog_tick/prog_mask and toggle prog_we.
//      The core writes once per synchronized prog_we transition while reset is
//      high. This avoids level-write corruption while VIO updates addr/data.
//   4. Set repeat_forever plus optional loop_start_addr/loop_end_tick/loop_count.
//   5. Release reset.
//   6. Pulse start.  The FPGA may repeat the table forever and/or loop a
//      sub-table in hardware without receiving expanded edges from Vivado.

module zlc_pulse_streamer #(
    parameter integer CHANNEL_COUNT = 4,
    parameter integer EDGE_ADDR_WIDTH = 10,
    parameter integer TICK_WIDTH = 32
)(
    input wire clk,
    input wire reset,
    input wire start,
    input wire prog_we,
    input wire [EDGE_ADDR_WIDTH-1:0] prog_addr,
    input wire [TICK_WIDTH-1:0] prog_tick,
    input wire [CHANNEL_COUNT-1:0] prog_mask,
    input wire [EDGE_ADDR_WIDTH:0] prog_count,
    input wire repeat_forever,
    input wire [EDGE_ADDR_WIDTH-1:0] loop_start_addr,
    input wire [TICK_WIDTH-1:0] loop_end_tick,
    input wire [31:0] loop_count,
    output wire [CHANNEL_COUNT-1:0] out,
    output reg running = 1'b0,
    output reg done = 1'b0
);

    localparam integer MAX_EDGES = (1 << EDGE_ADDR_WIDTH);

    // Keep the runtime read side to one table address.  Vivado otherwise has
    // to synthesize a multi-read-port memory, which is expensive on Artix-7.
    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] tick_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg [CHANNEL_COUNT-1:0] mask_mem [0:MAX_EDGES-1];

    reg [CHANNEL_COUNT-1:0] state_mask = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] time_count = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick_shadow = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] first_tick_shadow = {TICK_WIDTH{1'b0}};
    reg [CHANNEL_COUNT-1:0] first_mask_shadow = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] loop_start_tick_shadow = {TICK_WIDTH{1'b0}};
    reg [CHANNEL_COUNT-1:0] loop_start_mask_shadow = {CHANNEL_COUNT{1'b0}};
    reg [EDGE_ADDR_WIDTH:0] edge_index = {(EDGE_ADDR_WIDTH + 1){1'b0}};
    reg [EDGE_ADDR_WIDTH:0] active_count = {(EDGE_ADDR_WIDTH + 1){1'b0}};
    reg repeat_forever_active = 1'b0;
    reg [EDGE_ADDR_WIDTH-1:0] loop_start_active = {EDGE_ADDR_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] loop_end_active = {TICK_WIDTH{1'b0}};
    reg [31:0] loop_count_active = 32'd1;
    reg [31:0] loops_remaining = 32'd1;

    reg reset_meta = 1'b0;
    reg reset_sync = 1'b0;
    reg start_meta = 1'b0;
    reg start_sync = 1'b0;
    reg start_prev = 1'b0;
    reg prog_we_meta = 1'b0;
    reg prog_we_sync = 1'b0;
    reg prog_we_prev = 1'b0;

    wire start_event = start_sync && !start_prev;
    wire prog_we_event = prog_we_sync != prog_we_prev;
    wire [EDGE_ADDR_WIDTH-1:0] edge_addr = edge_index[EDGE_ADDR_WIDTH-1:0];

    assign out = state_mask;

    always @(posedge clk) begin
        reset_meta <= reset;
        reset_sync <= reset_meta;
        start_meta <= start;
        start_sync <= start_meta;
        start_prev <= start_sync;
        prog_we_meta <= prog_we;
        prog_we_sync <= prog_we_meta;
        prog_we_prev <= prog_we_sync;

        if (reset_sync && prog_we_event) begin
            tick_mem[prog_addr] <= prog_tick;
            mask_mem[prog_addr] <= prog_mask;
            if (prog_addr == {EDGE_ADDR_WIDTH{1'b0}}) begin
                first_tick_shadow <= prog_tick;
                first_mask_shadow <= prog_mask;
            end
            if (prog_addr == loop_start_addr) begin
                loop_start_tick_shadow <= prog_tick;
                loop_start_mask_shadow <= prog_mask;
            end
            if (prog_count != 0 && prog_addr == (prog_count[EDGE_ADDR_WIDTH-1:0] - 1'b1)) begin
                final_tick_shadow <= prog_tick;
            end
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
            loop_start_active <= {EDGE_ADDR_WIDTH{1'b0}};
            loop_end_active <= {TICK_WIDTH{1'b0}};
            loop_count_active <= 32'd1;
            loops_remaining <= 32'd1;
        end else if (start_event && !running) begin
            running <= (prog_count != 0);
            done <= (prog_count == 0);
            final_tick <= (prog_count == 0) ? {TICK_WIDTH{1'b0}} : final_tick_shadow;
            active_count <= prog_count;
            repeat_forever_active <= repeat_forever;
            loop_start_active <= loop_start_addr;
            loop_end_active <= loop_end_tick;
            loop_count_active <= (loop_count == 0) ? 32'd1 : loop_count;
            loops_remaining <= (loop_count == 0) ? 32'd1 : loop_count;
            if (prog_count != 0 && first_tick_shadow == {TICK_WIDTH{1'b0}}) begin
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
                time_count <= loop_start_tick_shadow + 1'b1;
                edge_index <= {1'b0, loop_start_active} + 1'b1;
                loops_remaining <= loops_remaining - 1'b1;
            end else if (time_count >= final_tick) begin
                if (repeat_forever_active) begin
                    if (first_tick_shadow == {TICK_WIDTH{1'b0}}) begin
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
                end
            end else begin
                if (edge_index < active_count && time_count == tick_mem[edge_addr]) begin
                    state_mask <= mask_mem[edge_addr];
                    edge_index <= edge_index + 1'b1;
                end
                time_count <= time_count + 1'b1;
            end
        end
    end
endmodule
