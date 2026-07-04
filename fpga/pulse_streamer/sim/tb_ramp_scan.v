`timescale 1ns/1ps
// REAL-ENGINE proof of the edge+RAMP DAC scan: a ramp whose STOP endpoint reads scan
// slot 0 at runtime (stop_value_select = 1).  Two scan points (codes 420 and 900, both
// STEEP Bresenham: delta > span exercises the deferred divmod + multi-code stepping).
// For every tick of both points the engine's bus_out must equal the integer staircase
//   v(t) = vstart + floor((t - t0) * delta / span)   (landing exactly on the code),
// which is byte-identical to the Python model (_rtl_bus_held_value) and the preview.
module tb_ramp_scan;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, BUSC=4, BW=10, TDW=32, SAW=12;
  localparam integer T0=10, T1=40, TF=60;          // ramp window [10,40), frame 60 ticks
  localparam [BW-1:0] VSTART=10'd320;
  localparam [TW-1:0] P0VAL=32'd420, P1VAL=32'd900;

  reg clk=0; always #10 clk=~clk;
  reg reset, start;

  // edges: e0@0 (ch5 on), e1@50 (off), e2@60 anchor; 2 scan points -> plays twice
  reg [TW-1:0] etick [0:2]; reg [CH-1:0] emask [0:2];
  initial begin
    etick[0]=0;  emask[0]=8'h20;
    etick[1]=50; emask[1]=8'h00;
    etick[2]=TF; emask[2]=8'h00;
  end
  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr%3]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr%3]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end

  // scan BRAM stub (registered, like the real IP): point p -> slot value
  reg [NS*TW-1:0] scanmem [0:3];
  reg [NS*TW-1:0] spipe [0:2];
  wire [SAW-1:0] scan_raddr;
  always @(posedge clk) begin spipe[0]<=scanmem[scan_raddr[1:0]]; spipe[1]<=spipe[0]; spipe[2]<=spipe[1]; end
  wire [NS*TW-1:0] scan_rdata = spipe[1];
  initial begin scanmem[0]=P0VAL; scanmem[1]=P1VAL; scanmem[2]=0; scanmem[3]=0; end

  reg [CH*TDW-1:0]   delay_ticks_w     = {CH*TDW{1'b0}};
  reg [BUSC*TDW-1:0] bus_delay_ticks_w = {BUSC*TDW{1'b0}};
  reg                bus_prog_we = 1'b0;
  reg [1:0]          bus_prog_bus = 2'd0;
  reg [5:0]          bus_prog_addr = 6'd0;
  reg [TW-1:0]       bus_prog_start_tick=0, bus_prog_stop_tick=0;
  reg [BW-1:0]       bus_prog_start_value=0, bus_prog_stop_value=0;
  reg [1:0]          bus_prog_mode=2'd0;
  reg [2:0]          bus_prog_value_select=3'd0, bus_prog_stop_value_select=3'd0;
  reg [BUSC*7-1:0]   bus_counts = {BUSC*7{1'b0}};

  wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_edge_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS), .BUS_COUNT(BUSC), .BUS_WIDTH(BW),
                      .SCAN_ADDR_WIDTH(SAW)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(13'd3),.repeat_forever(1'b0),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(TF[TW-1:0]),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1),.repeat_from_loop_start(1'b0),.scan_enable(1'b1),.scan_count(32'd2),
    .edge_raddr(edge_raddr),.edge_tick_rdata(tp[1]),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(mp[1]),
    .scan_raddr(scan_raddr),.scan_rdata(scan_rdata),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_prog_we(bus_prog_we),.bus_prog_bus(bus_prog_bus),.bus_prog_addr(bus_prog_addr),
    .bus_prog_start_tick(bus_prog_start_tick),.bus_prog_stop_tick(bus_prog_stop_tick),
    .bus_prog_start_tick_coeffs({NS*CW{1'b0}}),.bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_start_value(bus_prog_start_value),.bus_prog_stop_value(bus_prog_stop_value),
    .bus_prog_mode(bus_prog_mode),.bus_prog_value_select(bus_prog_value_select),
    .bus_prog_stop_value_select(bus_prog_stop_value_select),
    .bus_counts(bus_counts),
    .bus_delay_ticks(bus_delay_ticks_w),.delay_ticks(delay_ticks_w),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  task prog_seg(input [5:0] a, input [TW-1:0] stk, input [TW-1:0] etk,
                input [BW-1:0] v0, input [BW-1:0] v1, input [1:0] m,
                input [2:0] s0, input [2:0] s1);
    begin
      @(posedge clk);
      bus_prog_bus=2'd0; bus_prog_addr=a;
      bus_prog_start_tick=stk; bus_prog_stop_tick=etk;
      bus_prog_start_value=v0; bus_prog_stop_value=v1; bus_prog_mode=m;
      bus_prog_value_select=s0; bus_prog_stop_value_select=s1;
      bus_prog_we = ~bus_prog_we; @(posedge clk); @(posedge clk);
    end
  endtask

  // expected staircase for point value PV at frame tick t (matches the Python model)
  function [BW-1:0] expect_v;
    input integer t; input [TW-1:0] pv;
    integer delta, k, moves;
    begin
      if (t < T0) expect_v = VSTART;
      else if (t >= T1) expect_v = pv[BW-1:0];
      else begin
        delta = pv - VSTART;       // both points ramp upward here
        k = t - T0;
        moves = (k * delta) / (T1 - T0);
        if (moves > delta) moves = delta;
        expect_v = VSTART + moves;
      end
    end
  endfunction

  integer t, started, errs, p, ft;
  reg [BW-1:0] hist [0:2*60+10];
  initial begin
    reset=1; start=0;
    repeat (8) @(posedge clk);
    prog_seg(6'd0, 32'd0, 32'd0, VSTART, VSTART, 2'd1, 3'd0, 3'd0);   // edge 320 @0
    prog_seg(6'd1, T0[TW-1:0], T1[TW-1:0], VSTART, 10'd0, 2'd2, 3'd0, 3'd1); // RAMP stop<-s0
    bus_counts[0*7 +: 7] = 7'd2;
    repeat (140) @(posedge clk);                                       // arm with reset held
    reset=0; @(posedge clk); start=1; repeat(4) @(posedge clk); start=0;
    t=-1; started=0;
    while (t < 2*TF) begin
      @(posedge clk);
      if (running && !started) begin started=1; t=-1; end
      if (started) begin t=t+1; if (t>=0 && t<=2*TF) hist[t]=bus_out[0*BW +: BW]; end
      if (done) t = 2*TF;
    end
    // The TB's shift-register edge memory shifts the WHOLE playback 2 ticks early vs the
    // real-BRAM latency (a uniform harness offset, TTL and bus together -- same as
    // tb_da_ttl_align/tb_biasy; absolute phase is covered by the full-top tb_t_ff).  So
    // sample t corresponds to frame tick t+2; compare inside [0, TF-2).
    errs = 0;
    for (p = 0; p < 2; p = p + 1) begin
      $write("P%0d bus:", p);
      for (ft = 0; ft < TF; ft = ft + 6) $write(" %0d@%0d", hist[p*TF+ft], ft);
      $write("\n");
      for (ft = 0; ft < TF-2; ft = ft + 1)
        if (hist[p*TF+ft] !== expect_v(ft+2, (p==0)?P0VAL:P1VAL)) begin
          if (errs < 8) $display("  MISMATCH p%0d t%0d got %0d expect %0d", p, ft, hist[p*TF+ft], expect_v(ft+2, (p==0)?P0VAL:P1VAL));
          errs = errs + 1;
        end
    end
    $display("mismatches=%0d  %s", errs, (errs==0) ? "RAMP-SCAN-OK" : "**RAMP-SCAN-BAD**");
    $finish;
  end
endmodule
