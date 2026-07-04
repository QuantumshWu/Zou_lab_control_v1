`timescale 1ns/1ps
// ============================================================================
// da_clk PHASE vs DAC parallel-data race  --  the "third DA value between two
// edge periods" bug (pulses/T.json: da_bias_y steps -192->388 = code 320->900).
//
// PART A (REAL ENGINE, logic proof): program TWO edge/HOLD bus segments on bus 0
//   (value 320 from tick 0, value 900 from tick T) with delay 0 and FIRE.  Record
//   dut.bus_value_active[0] and the pin word bus_out[0] every tick and assert the
//   transition is a SINGLE-CYCLE step 320->900 with NO intermediate value and
//   bus_out==bus_value_active every tick.  => the engine LOGIC is glitch-free; the
//   "third value" is NOT produced inside the FPGA fabric.
//
// PART B (I/O TIMING MODEL, mechanism + fix): the 10 da_bias_y data bits leave the
//   FPGA on `posedge clk` (registered bus_value_active -> combinational mux -> pin)
//   and the DAC latch strobe da_clk1 == clk (a clk-button channel, out_final[39]).
//   So the DAC latches its parallel word on the SAME posedge the word changes, with
//   only uncontrolled per-bit routing skew as margin (the interface is unconstrained
//   in board.xdc -- no create_generated_clock / set_output_delay).  Model realistic
//   per-bit data-arrival times and sweep the latch-edge arrival: latching coincident
//   with the data change (da_clk = clk) captures a half-old/half-new MIX = a THIRD
//   code, while latching in the middle of the data eye (the FIX: da_clk on the clk
//   FALLING edge) always captures the clean new code.  This is exactly why a ~200 ms
//   HOLD gap "fixed" it on the bench (it moved the DAC step off the noisy period
//   boundary) and why the real fix removes the gap requirement entirely.
// ============================================================================
module tb_da_clk_phase;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, BUSC=4, BW=10, TDW=32;
  localparam integer NT=400;
  localparam integer TSTEP=40;                 // bus 0 steps 320->900 at this tick
  localparam [BW-1:0] VOLD=10'd320;            // -192 signed  (offset-binary code)
  localparam [BW-1:0] VNEW=10'd900;            //  388 signed
  reg clk=0, reset=0, start=0; always #10 clk=~clk;   // 20 ns period (50 MHz)

  // minimal dummy edge table (2 edges, no TTL activity) so the engine just runs
  reg [TW-1:0] etick [0:1]; reg [CH-1:0] emask [0:1];
  initial begin etick[0]=0; emask[0]=8'h00; etick[1]=300; emask[1]=8'h00; end
  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr[0]]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr[0]]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end
  wire [TW-1:0] edge_tick_rdata = tp[1];
  wire [CH-1:0] edge_mask_rdata = mp[1];

  wire [BUSC*TDW-1:0] bus_delay_ticks_w = {BUSC*TDW{1'b0}};   // NO delay (T.json delays={})

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
    .bus_delay_ticks(bus_delay_ticks_w),.delay_ticks({CH*TDW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  task prog_seg(input [5:0] a, input [TW-1:0] stk, input [BW-1:0] val);  // one HOLD/edge segment
    begin
      @(posedge clk);
      bus_prog_bus=2'd0; bus_prog_addr=a; bus_prog_start_tick=stk; bus_prog_stop_tick=32'd300;
      bus_prog_start_value=val; bus_prog_stop_value=val; bus_prog_mode=2'd0;   // mode 0 = constant
      bus_prog_we = ~bus_prog_we;
      @(posedge clk); @(posedge clk);
    end
  endtask

  initial begin
    reset=1; start=0;
    repeat (20) @(posedge clk);
    prog_seg(6'd0, 32'd0,     VOLD);     // bus0 seg0: 320 from tick 0
    prog_seg(6'd1, TSTEP[TW-1:0], VNEW); // bus0 seg1: 900 from tick TSTEP
    bus_counts[0*7 +: 7] = 7'd2;         // bus 0 has 2 segments
    repeat (120) @(posedge clk);         // hold reset long enough for the ARM loop to
                                         // initialise scan_first_values (=0, scan off)
    reset=0; @(posedge clk); start=1; repeat(4) @(posedge clk); start=0;
  end

  // ---- PART A: record the real engine and prove a clean single-cycle step ----
  reg [BW-1:0] vhist [0:NT]; reg [BW-1:0] ohist [0:NT];
  integer t=-1, started=0, i;
  always @(posedge clk) begin : rec
    if (running && !started) begin started=1; t=-1; end
    if (started && t < NT) begin
      t = t + 1;
      vhist[t] = dut.bus_value_active[0];
      ohist[t] = bus_out[0*BW +: BW];
    end
  end

  // ---- PART B: I/O-timing capture model (uses the proven endpoints VOLD/VNEW) ----
  // per-bit data arrival after the launching posedge (Tcko + routing skew), ns:
  real TD [0:BW-1];
  integer k, s; real clka; integer third_cur, third_fix, exa_cur;
  reg [BW-1:0] cap_cur, cap_fix; reg curbit, fixbit;
  initial begin
    for (k=0;k<BW;k=k+1) TD[k] = 0.5 + 0.3*k;   // 0.5 .. 3.2 ns spread across the 10 bits
  end

  function [BW-1:0] capture;        // capture[k] = data already arrived (TD<=latch) ? new : old
    input real latch_ns; integer kk; reg [BW-1:0] v;
    begin
      v = {BW{1'b0}};
      for (kk=0; kk<BW; kk=kk+1)
        v[kk] = (TD[kk] <= latch_ns) ? VNEW[kk] : VOLD[kk];
      capture = v;
    end
  endfunction

  integer aerr=0; integer tstep_found=-1;
  initial begin
    wait(reset==1); wait(reset==0);
    repeat (NT) @(posedge clk);

    // ---- PART A checks ----
    // find the single transition tick; assert exactly VOLD before, VNEW after, and
    // that NO recorded sample is ever a value other than VOLD or VNEW once running,
    // and that bus_out tracks bus_value_active every tick (delay 0).
    for (i=1;i<=t;i=i+1) begin
      if (ohist[i] !== vhist[i]) begin aerr=aerr+1;
        if (aerr<=4) $display("  [A] bus_out!=active at t=%0d out=%0d active=%0d", i, ohist[i], vhist[i]); end
      if (vhist[i]!==VOLD && vhist[i]!==VNEW) begin aerr=aerr+1;
        if (aerr<=4) $display("  [A] THIRD value in fabric at t=%0d value=%0d", i, vhist[i]); end
      if (vhist[i]==VNEW && vhist[i-1]==VOLD) tstep_found = i;   // the step tick
    end
    if (tstep_found < 0) begin aerr=aerr+1; $display("  [A] never stepped VOLD->VNEW"); end
    else $display("  [A] clean single-cycle step %0d->%0d at recorded tick %0d (no intermediate)",
                  VOLD, VNEW, tstep_found);

    // ---- PART B sweep ----
    third_cur=0; third_fix=0; exa_cur=-1;
    for (s=0; s<=24; s=s+1) begin                 // latch arrival 0.00 .. 6.00 ns
      clka = 0.25*s;
      cap_cur = capture(clka);                    // CURRENT: da_clk = clk -> latch ~coincident
      cap_fix = capture(clka + 10.0);             // FIX: da_clk on clk FALLING edge -> +10 ns (eye centre)
      if (cap_cur!==VOLD && cap_cur!==VNEW) begin
        third_cur = third_cur + 1;
        if (exa_cur<0) begin exa_cur = cap_cur;
          $display("  [B] CURRENT (da_clk=clk): latch@%.2fns -> code %0d  (THIRD value, not %0d/%0d)",
                   clka, cap_cur, VOLD, VNEW); end
      end
      if (cap_fix!==VNEW) third_fix = third_fix + 1;
    end
    $display("  [B] CURRENT design: %0d of 25 swept skews capture a THIRD value", third_cur);
    $display("  [B] FIXED  design: %0d of 25 swept skews mis-capture (latch at data-eye centre)", third_fix);

    $display("==== da_clk phase: A_errs=%0d  third_current=%0d  third_fixed=%0d ====",
             aerr, third_cur, third_fix);
    // PASS == engine logic clean (aerr 0, stepped once) AND current phase is provably
    // hazardous (>=1 third value) AND the eye-centre fix is provably immune (0).
    $display("%s", (aerr==0 && tstep_found>0 && third_cur>0 && third_fix==0)
                   ? "DA-CLK-PHASE-OK" : "**FAIL**");
    $finish;
  end
endmodule
