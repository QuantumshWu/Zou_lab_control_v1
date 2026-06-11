`timescale 1ns/1ps
// Does the DA-bus delay behave IDENTICALLY to a TTL-channel delay?  (User: "DA delay 和
// TTL delay 是一个机制".)  Drive the REAL engine with a TTL edge on ch5 and a DA-bus value
// change on bus0, BOTH at the same tick T0, and the SAME per-output delay D.  Record out[5]
// and bus_out[bus0] every tick and assert: (a) at D=0 they change on the SAME tick (= T0),
// (b) at D=N both change on the SAME tick (= T0+N) -- i.e. the DA delay shifts exactly like
// the TTL delay, no extra offset, no "delay does nothing".
module tb_da_ttl_align;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, BUSC=4, BW=10, TDW=32;
  localparam integer NT=400, T0=40, TTLBIT=5;
  localparam [BW-1:0] VA=10'd200, VB=10'd800;   // bus0 steps VA->VB at T0
  reg clk=0; always #10 clk=~clk;
  reg reset, start;
  integer DUT_D;                                 // delay under test (set per run)

  // ---- edge table: e0@0 mask0, e1@T0 mask=(1<<TTLBIT), e2@T0+40 mask0 (ch5 pulse [T0,T0+40)) ----
  reg [TW-1:0] etick [0:2]; reg [CH-1:0] emask [0:2];
  initial begin
    etick[0]=0;     emask[0]=8'h00;
    etick[1]=T0;    emask[1]=(8'h1<<TTLBIT);
    etick[2]=T0+40; emask[2]=8'h00;
  end
  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr%3]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr%3]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end
  wire [TW-1:0] edge_tick_rdata = tp[1];
  wire [CH-1:0] edge_mask_rdata = mp[1];

  // ---- per-output delays: ch5 (TTL) and bus0 (DA) BOTH = DUT_D, everything else 0 ----
  reg [CH*TDW-1:0]   delay_ticks_w   = {CH*TDW{1'b0}};
  reg [BUSC*TDW-1:0] bus_delay_ticks_w = {BUSC*TDW{1'b0}};

  reg                bus_prog_we = 1'b0;
  reg [1:0]          bus_prog_bus = 2'd0;
  reg [5:0]          bus_prog_addr = 6'd0;
  reg [TW-1:0]       bus_prog_start_tick=32'd0, bus_prog_stop_tick=32'd0;
  reg [BW-1:0]       bus_prog_start_value=10'd0, bus_prog_stop_value=10'd0;
  reg [1:0]          bus_prog_mode=2'd0;
  reg [BUSC*7-1:0]   bus_counts = {BUSC*7{1'b0}};

  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_edge_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS), .BUS_COUNT(BUSC), .BUS_WIDTH(BW)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(13'd3),.repeat_forever(1'b1),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd300),.loop_end_coeffs({NS*CW{1'b0}}),
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
    .bus_delay_ticks(bus_delay_ticks_w),.delay_ticks(delay_ticks_w),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  task prog_bus_seg(input [5:0] a, input [TW-1:0] stk, input [BW-1:0] val);
    begin
      @(posedge clk);
      bus_prog_bus=2'd0; bus_prog_addr=a; bus_prog_start_tick=stk; bus_prog_stop_tick=32'd300;
      bus_prog_start_value=val; bus_prog_stop_value=val; bus_prog_mode=2'd0;
      bus_prog_we = ~bus_prog_we;
      @(posedge clk); @(posedge clk);
    end
  endtask

  integer tt0_out, tt0_bus, errs;
  reg [BW-1:0] bhist [0:NT]; reg ohist [0:NT];
  integer t, started, i;

  task run_with_delay(input integer D);
    begin
      DUT_D = D;
      reset=1; start=0;
      delay_ticks_w = {CH*TDW{1'b0}};      delay_ticks_w[TTLBIT*TDW +: TDW] = D[TDW-1:0];
      bus_delay_ticks_w = {BUSC*TDW{1'b0}}; bus_delay_ticks_w[0*TDW +: TDW] = D[TDW-1:0];
      repeat (8) @(posedge clk);
      prog_bus_seg(6'd0, 32'd0, VA);       // bus0 seg0 = VA from tick 0
      prog_bus_seg(6'd1, T0[TW-1:0], VB);  // bus0 seg1 = VB from tick T0
      bus_counts[0*7 +: 7] = 7'd2;
      repeat (120) @(posedge clk);         // hold reset: ARM loop inits scan_first_values=0
      reset=0; @(posedge clk); start=1; repeat(4) @(posedge clk); start=0;
      // record
      t=-1; started=0;
      while (t < NT) begin
        @(posedge clk);
        if (running && !started) begin started=1; t=-1; end
        if (started) begin t=t+1; if (t<=NT) begin ohist[t]=out[TTLBIT]; bhist[t]=bus_out[0*BW +: BW]; end end
      end
      // find first rising edge of out[5] and first VA->VB change of bus_out
      tt0_out=-1; tt0_bus=-1;
      for (i=1;i<=NT;i=i+1) begin
        if (tt0_out<0 && ohist[i]==1'b1 && ohist[i-1]==1'b0) tt0_out=i;
        if (tt0_bus<0 && bhist[i]==VB && bhist[i-1]==VA)     tt0_bus=i;
      end
      $display("  D=%0d : TTL out[5] rises @t=%0d , DA bus_out VA->VB @t=%0d  %s",
               D, tt0_out, tt0_bus, (tt0_out==tt0_bus && tt0_out>0) ? "ALIGNED" : "*** MISALIGNED ***");
      if (!(tt0_out==tt0_bus && tt0_out>0)) errs=errs+1;
    end
  endtask

  initial begin
    errs=0;
    run_with_delay(0);     // d=0: TTL and DA must change on the SAME tick
    run_with_delay(5);     // d=5: both shifted by 5, still SAME tick
    run_with_delay(25);    // d=25: both shifted by 25, still SAME tick
    $display("==== da/ttl align: %0d misalignments ====", errs);
    $display("%s", (errs==0) ? "DA-TTL-ALIGN-OK" : "**FAIL**");
    $finish;
  end
endmodule
