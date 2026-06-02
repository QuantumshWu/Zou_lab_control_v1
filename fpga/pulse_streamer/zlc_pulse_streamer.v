`timescale 1ns / 1ps
// Runtime-programmable edge-table pulse streamer.
//
// Host upload contract:
//   1. Hold reset high.
//   2. Set prog_count to the number of active edges.
//   3. For each edge, set prog_addr/prog_tick/prog_mask and pulse prog_we.
//   4. Release reset.
//   5. Pulse start to run the uploaded table once.

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
    output wire [CHANNEL_COUNT-1:0] out,
    output reg running = 1'b0,
    output reg done = 1'b0
);

    localparam integer MAX_EDGES = (1 << EDGE_ADDR_WIDTH);

    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] tick_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg [CHANNEL_COUNT-1:0] mask_mem [0:MAX_EDGES-1];

    reg [CHANNEL_COUNT-1:0] state_mask = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] time_count = {TICK_WIDTH{1'b0}};
    reg [TICK_WIDTH-1:0] final_tick = {TICK_WIDTH{1'b0}};
    reg [EDGE_ADDR_WIDTH:0] edge_index = {(EDGE_ADDR_WIDTH + 1){1'b0}};
    reg [EDGE_ADDR_WIDTH:0] active_count = {(EDGE_ADDR_WIDTH + 1){1'b0}};

    reg start_sync = 1'b0;
    reg start_prev = 1'b0;
    reg prog_we_sync = 1'b0;
    reg prog_we_prev = 1'b0;

    wire start_edge = start_sync && !start_prev;
    wire prog_we_edge = prog_we_sync && !prog_we_prev;
    wire [EDGE_ADDR_WIDTH-1:0] edge_addr = edge_index[EDGE_ADDR_WIDTH-1:0];

    assign out = state_mask;

    always @(posedge clk) begin
        start_sync <= start;
        start_prev <= start_sync;
        prog_we_sync <= prog_we;
        prog_we_prev <= prog_we_sync;

        if (prog_we_edge) begin
            tick_mem[prog_addr] <= prog_tick;
            mask_mem[prog_addr] <= prog_mask;
        end

        if (reset) begin
            running <= 1'b0;
            done <= 1'b0;
            state_mask <= {CHANNEL_COUNT{1'b0}};
            time_count <= {TICK_WIDTH{1'b0}};
            final_tick <= {TICK_WIDTH{1'b0}};
            edge_index <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
            active_count <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
        end else if (start_edge && !running) begin
            running <= (prog_count != 0);
            done <= (prog_count == 0);
            state_mask <= {CHANNEL_COUNT{1'b0}};
            time_count <= {TICK_WIDTH{1'b0}};
            final_tick <= (prog_count == 0) ? {TICK_WIDTH{1'b0}} : tick_mem[prog_count[EDGE_ADDR_WIDTH-1:0] - 1'b1];
            edge_index <= {(EDGE_ADDR_WIDTH + 1){1'b0}};
            active_count <= prog_count;
        end else if (running) begin
            if (edge_index < active_count && time_count == tick_mem[edge_addr]) begin
                state_mask <= mask_mem[edge_addr];
                edge_index <= edge_index + 1'b1;
            end

            if (time_count >= final_tick) begin
                running <= 1'b0;
                done <= 1'b1;
                state_mask <= {CHANNEL_COUNT{1'b0}};
            end else begin
                time_count <= time_count + 1'b1;
            end
        end
    end
endmodule
