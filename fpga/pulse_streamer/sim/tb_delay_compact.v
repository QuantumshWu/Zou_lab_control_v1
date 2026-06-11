`timescale 1ns/1ps
// COMPACTED event-FIFO map: only a SUBSET of channels gets an event FIFO, and a
// delayed channel is served by a slot whose index != channel (DELAY_CH_MAP).  This
// proves the slot->channel remap (evt_ch_of) drives the right output bit and that
// non-eligible channels pass through undelayed.
//   eligible (slot -> channel):  slot0->ch1 (d=3), slot1->ch5 (d=7), slot2->ch6 (d=2)
//   non-eligible: ch0/2/3/4/7  (d=0, passthrough)
// All 8 channels carry the SAME undelayed waveform (bit of hist), so each delayed
// channel can be checked against hist[t-d][0].
module tb_delay_compact;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, DTW=12, BUSC=4, BW=10;
  localparam integer NE=8;
  localparam integer NT=6000;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  reg [TW-1:0] etick [0:NE-1]; reg [CH-1:0] emask [0:NE-1];
  initial begin
    etick[0]=0;    emask[0]=8'hFF;   // all on
    etick[1]=5;    emask[1]=8'h00;
    etick[2]=6;    emask[2]=8'hFF;   // 1-tick later back on (stress)
    etick[3]=7;    emask[3]=8'h00;
    etick[4]=100;  emask[4]=8'hFF;
    etick[5]=160;  emask[5]=8'h00;
    etick[6]=2000; emask[6]=8'hFF;
    etick[7]=2400; emask[7]=8'h00;   // frame ends at 2400
  end

  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr[2:0]]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr[2:0]]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end
  wire [TW-1:0] edge_tick_rdata = tp[1];
  wire [CH-1:0] edge_mask_rdata = mp[1];

  // per-channel delays: only ch1=3, ch5=7, ch6=2 are nonzero (must match the map)
  localparam integer TDW = 32;
  wire [CH*TDW-1:0] delay_ticks_w;
  assign delay_ticks_w[0*TDW +: TDW] = 32'd0;
  assign delay_ticks_w[1*TDW +: TDW] = 32'd3;
  assign delay_ticks_w[2*TDW +: TDW] = 32'd0;
  assign delay_ticks_w[3*TDW +: TDW] = 32'd0;
  assign delay_ticks_w[4*TDW +: TDW] = 32'd0;
  assign delay_ticks_w[5*TDW +: TDW] = 32'd7;
  assign delay_ticks_w[6*TDW +: TDW] = 32'd2;
  assign delay_ticks_w[7*TDW +: TDW] = 32'd0;

  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  // DELAY_CH_MAP packed (6 bits/slot): slot0=ch1, slot1=ch5, slot2=ch6
  zlc_edge_streamer #(
    .CHANNEL_COUNT(CH), .NUM_SLOTS(NS),
    .DELAY_COMPACT(1), .NUM_DELAY_CH(3), .DELAY_CH_IDX_W(6),
    .DELAY_CH_MAP({6'd6, 6'd5, 6'd1})
  ) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(13'd8),.repeat_forever(1'b1),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd2400),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1),.repeat_from_loop_start(1'b0),.scan_enable(1'b0),.scan_count(32'd0),
    .edge_raddr(edge_raddr),.edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_prog_we(1'b0),.bus_prog_bus(2'd0),.bus_prog_addr(6'd0),.bus_prog_start_tick(32'd0),
    .bus_prog_stop_tick(32'd0),.bus_prog_start_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),.bus_prog_start_value(10'd0),
    .bus_prog_stop_value(10'd0),.bus_prog_mode(2'd0),.bus_prog_value_select(3'd0),
    .bus_prog_stop_value_select(3'd0),.bus_counts({BUSC*7{1'b0}}),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks(delay_ticks_w),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  initial begin
    reset=1; start=0;
    repeat (50) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  reg [CH-1:0] hist [0:NT];
  integer t = -1; integer errs = 0; integer started = 0;
  integer dch; integer dexp; reg expected;
  always @(posedge clk) begin : oracle
    if (running && !started) begin started = 1; t = -1; end
    if (started) begin
      t = t + 1;
      if (t <= NT) hist[t] = dut.state_mask;
      if (t > 0) begin
        // delayed channels: out[ch] == hist[t-d][ch] (all channels share bit 0's waveform)
        for (dch = 0; dch < CH; dch = dch + 1) begin
          dexp = (dch==1) ? 3 : (dch==5) ? 7 : (dch==6) ? 2 : 0;
          if (dexp == 0) begin
            // passthrough (no slot): out == undelayed state_mask this tick
            if (out[dch] !== dut.state_mask[dch]) begin
              errs = errs + 1;
              if (errs <= 8) $display("  PASSTHRU MISMATCH t=%0d ch=%0d", t, dch);
            end
          end else begin
            expected = (t >= dexp) ? hist[t-dexp][dch] : 1'b0;
            if (out[dch] !== expected) begin
              errs = errs + 1;
              if (errs <= 8) $display("  DELAY MISMATCH t=%0d ch=%0d d=%0d out=%b exp=%b",
                                       t, dch, dexp, out[dch], expected);
            end
          end
        end
      end
    end
  end

  initial begin
    wait(reset==1); wait(reset==0);
    repeat (NT) @(posedge clk);
    $display("==== compact map: %0d cycles, %0d mismatches ====", t, errs);
    $display("%s", (errs==0 && t > 4000) ? "COMPACT-MAP-OK" : "**FAIL**");
    $finish;
  end
endmodule
