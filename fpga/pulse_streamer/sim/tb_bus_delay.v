`timescale 1ns/1ps
// PER-DA-BIT event-scheduler DAC delay on the REAL engine (the unified design: each DA bit is
// its own 1-bit event FIFO; the bus's BUS_WIDTH bits share one per-bus delay).  Program one
// HOLD segment (value 1023) on buses 0..2 so each undelayed bus value steps SAFE(512)->1023 at
// FIRE, with per-bus delays d0=5, d1=0 (passthrough), d2=1 (the register path).  ORACLE: record
// the engine's UNDELAYED dut.bus_value_active[b] every cycle and assert the delayed pin output
// bus_out[b] == value[t-d_b], holding the SAFE mid-code (512) before t == d_b.
module tb_bus_delay;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, BUSC=4, BW=10, TDW=32;
  localparam integer NT=3000;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  reg [TW-1:0] etick [0:1]; reg [CH-1:0] emask [0:1];
  initial begin etick[0]=0; emask[0]=8'h00; etick[1]=2400; emask[1]=8'h00; end
  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr[0]]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr[0]]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end
  wire [TW-1:0] edge_tick_rdata = tp[1];
  wire [CH-1:0] edge_mask_rdata = mp[1];

  // per-bus delays (32b): bus0=5, bus1=0, bus2=1, bus3=0
  wire [BUSC*TDW-1:0] bus_delay_ticks_w;
  assign bus_delay_ticks_w[0*TDW +: TDW] = 32'd5;
  assign bus_delay_ticks_w[1*TDW +: TDW] = 32'd0;
  assign bus_delay_ticks_w[2*TDW +: TDW] = 32'd1;
  assign bus_delay_ticks_w[3*TDW +: TDW] = 32'd0;

  // bus segment program inputs (driven during the reset hold, the load phase)
  reg                bus_prog_we = 1'b0;
  reg [1:0]          bus_prog_bus = 2'd0;
  reg [5:0]          bus_prog_addr = 6'd0;
  reg [TW-1:0]       bus_prog_start_tick = 32'd0, bus_prog_stop_tick = 32'd0;
  reg [BW-1:0]       bus_prog_start_value = 10'd0, bus_prog_stop_value = 10'd0;
  reg [1:0]          bus_prog_mode = 2'd0;
  reg [BUSC*7-1:0]   bus_counts = {BUSC*7{1'b0}};

  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_edge_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS), .BUS_COUNT(BUSC), .BUS_WIDTH(BW)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(13'd2),.repeat_forever(1'b1),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd2400),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1),.repeat_from_loop_start(1'b0),.scan_enable(1'b0),.scan_count(32'd0),
    .edge_raddr(edge_raddr),.edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_prog_we(bus_prog_we),.bus_prog_bus(bus_prog_bus),.bus_prog_addr(bus_prog_addr),
    .bus_prog_start_tick(bus_prog_start_tick),.bus_prog_stop_tick(bus_prog_stop_tick),
    .bus_prog_start_tick_coeffs({NS*CW{1'b0}}),.bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_start_value(bus_prog_start_value),.bus_prog_stop_value(bus_prog_stop_value),
    .bus_prog_mode(bus_prog_mode),.bus_prog_value_select(3'd0),.bus_prog_stop_value_select(3'd0),
    .bus_counts(bus_counts),
    .bus_delay_ticks(bus_delay_ticks_w),.delay_ticks({CH*TDW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  task prog_seg(input [1:0] b);   // one HOLD segment, value 1023, whole frame
    begin
      @(posedge clk);
      bus_prog_bus=b; bus_prog_addr=6'd0; bus_prog_start_tick=32'd0; bus_prog_stop_tick=32'd2400;
      bus_prog_start_value=10'd1023; bus_prog_stop_value=10'd1023; bus_prog_mode=2'd0;
      bus_prog_we = ~bus_prog_we;   // edge -> latch
      @(posedge clk); @(posedge clk);
    end
  endtask

  integer bi;
  initial begin
    reset=1; start=0;
    repeat (20) @(posedge clk);          // let reset_sync settle high
    prog_seg(2'd0); prog_seg(2'd1); prog_seg(2'd2);
    for (bi=0; bi<3; bi=bi+1) bus_counts[bi*7 +: 7] = 7'd1;   // 1 segment each
    repeat (20) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  reg [BW-1:0] hist0 [0:NT]; reg [BW-1:0] hist1 [0:NT]; reg [BW-1:0] hist2 [0:NT];
  integer t=-1, errs=0, started=0; reg [BW-1:0] e0,e1,e2;
  always @(posedge clk) begin : oracle
    if (running && !started) begin started=1; t=-1; end
    if (started) begin
      t = t + 1;
      if (t <= NT) begin
        hist0[t]=dut.bus_value_active[0]; hist1[t]=dut.bus_value_active[1]; hist2[t]=dut.bus_value_active[2];
      end
      if (t > 0) begin
        e0 = (t >= 5) ? hist0[t-5] : 10'd512;
        e1 = hist1[t];                       // d=0 passthrough
        e2 = (t >= 1) ? hist2[t-1] : 10'd512;
        if (bus_out[0*BW +: BW] !== e0) begin errs=errs+1;
          if (errs<=6) $display("  BUS0 t=%0d out=%0d exp=%0d", t, bus_out[0*BW +: BW], e0); end
        if (bus_out[1*BW +: BW] !== e1) begin errs=errs+1;
          if (errs<=6) $display("  BUS1 t=%0d out=%0d exp=%0d", t, bus_out[1*BW +: BW], e1); end
        if (bus_out[2*BW +: BW] !== e2) begin errs=errs+1;
          if (errs<=6) $display("  BUS2 t=%0d out=%0d exp=%0d", t, bus_out[2*BW +: BW], e2); end
      end
    end
  end

  initial begin
    wait(reset==1); wait(reset==0);
    repeat (NT) @(posedge clk);
    $display("==== bus delay: %0d cycles checked, %0d mismatches ====", t, errs);
    $display("%s", (errs==0 && t > 1000) ? "BUS-DELAY-OK" : "**FAIL**");
    $finish;
  end
endmodule
