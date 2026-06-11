`timescale 1ns/1ps
// TTL EVENT-SCHEDULER delay verification on the REAL engine: channels with delays
// {0, 1, 2, 7, 1000} ticks, a dense-toggle program (incl. 1-tick edges) repeating
// forever, behavioral aligned-latency edge BRAMs.  ORACLE: record the engine's
// UNDELAYED stream (dut.state_mask) every cycle and assert out[t] == in[t-d] for
// every delayed channel on every cycle (0 before t=d), across frames and seams.
module tb_delay_sched;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, DTW=12, BUSC=4, BW=10;
  localparam integer NE=8;
  localparam integer NT=12000;          // > 3 frames + the 1000-tick delayed tail
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  // edges: dense toggles on bits 0..4 (all channels share the same waveform so each
  // delayed channel can be checked against the same undelayed reference bit).
  reg [TW-1:0] etick [0:NE-1]; reg [CH-1:0] emask [0:NE-1];
  initial begin
    etick[0]=0;    emask[0]=8'h1F;   // all five test channels ON at t=0
    etick[1]=5;    emask[1]=8'h00;   // off
    etick[2]=6;    emask[2]=8'h1F;   // 1-tick later back ON (stress)
    etick[3]=7;    emask[3]=8'h00;   // 1-tick pulse off
    etick[4]=100;  emask[4]=8'h1F;
    etick[5]=160;  emask[5]=8'h00;
    etick[6]=2000; emask[6]=8'h1F;
    etick[7]=2400; emask[7]=8'h00;   // frame ends at 2400 (loop_end)
  end

  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr[2:0]]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr[2:0]]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end
  wire [TW-1:0] edge_tick_rdata = tp[1];
  wire [CH-1:0] edge_mask_rdata = mp[1];

  // per-channel delays: ch0 d=0, ch1 d=1, ch2 d=2, ch3 d=7, ch4 d=1000 (32b fields)
  localparam integer TDW = 32;
  wire [CH*TDW-1:0] delay_ticks_w;
  assign delay_ticks_w[0*TDW +: TDW] = 32'd0;
  assign delay_ticks_w[1*TDW +: TDW] = 32'd1;
  assign delay_ticks_w[2*TDW +: TDW] = 32'd2;
  assign delay_ticks_w[3*TDW +: TDW] = 32'd7;
  assign delay_ticks_w[4*TDW +: TDW] = 32'd1000;
  assign delay_ticks_w[CH*TDW-1: 5*TDW] = {(CH-5)*TDW{1'b0}};

  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_edge_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS)) dut (
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
    repeat (200) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  // oracle: undelayed history (sampled value DURING each cycle) + per-cycle asserts
  reg [CH-1:0] hist [0:NT];           // hist[t] = undelayed state_mask during cycle t
  integer t = -1; integer errs = 0; integer started = 0;
  integer dch; integer dexp;
  always @(posedge clk) begin : oracle
    integer di; reg expected;
    if (running && !started) begin started = 1; t = -1; end
    if (started) begin
      t = t + 1;
      if (t <= NT) hist[t] = dut.state_mask;
      // check the four delayed channels (the undelayed waveform is bit 0 of hist)
      for (di = 1; di <= 4; di = di + 1) begin
        dexp = (di==1) ? 1 : (di==2) ? 2 : (di==3) ? 7 : 1000;
        expected = (t >= dexp && (t-dexp) <= NT) ? hist[t-dexp][0] : 1'b0;
        if (t > 0 && out[di] !== expected) begin
          errs = errs + 1;
          if (errs <= 8)
            $display("  MISMATCH t=%0d ch=%0d d=%0d out=%b expect=%b", t, di, dexp, out[di], expected);
        end
      end
      // d=0 channel: passthrough equality
      if (t > 0 && out[0] !== dut.state_mask[0]) begin
        errs = errs + 1;
        if (errs <= 8) $display("  MISMATCH d0 t=%0d", t);
      end
    end
  end

  integer dbg=0;
  always @(posedge clk) if (started && dbg<25) begin
    $display("  dbg t=%0d sm=%h prev=%h dct2=%0d cnt2=%0d out2=%b evtout2=%b g=%0d",
      t, dut.state_mask, dut.prev_undelayed, dut.del_ch_ticks[2], dut.g_evtfifo[2].cnt,
      out[2], dut.evt_out[2], dut.g_time);
    dbg=dbg+1;
  end
  initial begin
    wait(reset==1); wait(reset==0);
    repeat (NT) @(posedge clk);
    $display("==== delay scheduler: %0d cycles checked, %0d mismatches ====", t, errs);
    $display("%s", (errs==0 && t > 7000) ? "DELAY-SCHED-OK" : "**FAIL**");
    $finish;
  end
endmodule
