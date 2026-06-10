`timescale 1ns/1ps
`ifndef EDGE_LAT
 `define EDGE_LAT 2
`endif
`ifndef TICK_EXTRA
 `define TICK_EXTRA 2
`endif
`ifndef MASK_EXTRA
 `define MASK_EXTRA 2
`endif
// Standalone xsim testbench for the REAL zlc_edge_streamer engine, driven by the user's
// EXACT uploaded edge table (scaled /500 for speed) with parameterized-latency behavioral
// edge BRAMs.  Goal: find the true streaming-prefetch behaviour without guessing.
//   EDGE_LAT  = BRAM read latency for ALL three edge BRAMs (the forced value is 2).
//   TICK_EXTRA= extra register stages on the TICK read only (e92a78a added 1).
// emCCD is bit 11.  Correct: on@2500 off@4500 (pulse1), on@12001 off@14001 (pulse2, 2000t).
// 40ms bug = on@10001 off@14001 (pulse2 = 4000t, e7 dropped).
module tb_edge_streamer;
  localparam integer CH = 62, EAW = 12, TW = 32, NS = 4, CW = 16, DTW = 12, BUSC = 4, BW = 10;
  localparam integer EDGE_LAT  = `EDGE_LAT;
  localparam integer TICK_EXTRA = `TICK_EXTRA;

  reg clk = 0, reset = 0, start = 0;
  always #10 clk = ~clk;   // 50 MHz (20 ns)

  // --- program (10 edges, scaled /500), emCCD = bit 11 ---
  localparam integer NE = 10;
  reg [TW-1:0]  tickmem [0:4095];
  reg [NS*CW-1:0] coeffmem [0:4095];
  reg [CH-1:0]  maskmem [0:4095];
  integer i;
  initial begin
    for (i = 0; i < 4096; i = i + 1) begin tickmem[i]=0; coeffmem[i]=0; maskmem[i]=0; end
    tickmem[0]=0;     maskmem[0]=62'h685;
    tickmem[1]=1000;  maskmem[1]=62'h200;
    tickmem[2]=2500;  maskmem[2]=62'ha08;
    tickmem[3]=4500;  maskmem[3]=62'h200;
    tickmem[4]=5000;  maskmem[4]=62'h200;
    tickmem[5]=5001;  maskmem[5]=62'h200;
    tickmem[6]=10001; maskmem[6]=62'ha00;   // trap+emCCD
    tickmem[7]=12001; maskmem[7]=62'h208;   // probe+trap (emCCD off)
    tickmem[8]=14001; maskmem[8]=62'h0;
    tickmem[9]=14101; maskmem[9]=62'h0;
  end

  // --- engine wiring ---
  wire [EAW-1:0] edge_raddr;
  wire [TW-1:0]  edge_tick_rdata;
  wire [NS*CW-1:0] edge_coeff_rdata;
  wire [CH-1:0]  edge_mask_rdata;
  wire [11:0]    scan_raddr;
  wire [CH-1:0]  out;
  wire [BUSC*BW-1:0] bus_out;
  wire running, done;
  wire [31:0] scan_cursor; wire underflow;

  // --- behavioral edge BRAMs: latency EDGE_LAT (registered addr+mem+output stages) ---
  // generic N-stage pipeline so doutb(C) = mem[edge_raddr(C-EDGE_LAT)].
  reg [TW-1:0]    tpipe [0:7];
  reg [NS*CW-1:0] cpipe [0:7];
  reg [CH-1:0]    mpipe [0:7];
  integer k;
  always @(posedge clk) begin
    tpipe[0] <= tickmem[edge_raddr];
    cpipe[0] <= coeffmem[edge_raddr];
    mpipe[0] <= maskmem[edge_raddr];
    for (k = 1; k < 8; k = k + 1) begin
      tpipe[k] <= tpipe[k-1]; cpipe[k] <= cpipe[k-1]; mpipe[k] <= mpipe[k-1];
    end
  end
  // TICK_EXTRA delays the tick read; MASK_EXTRA delays coeff+mask.  Real asymmetric
  // (wide-read) coeff/mask are slower than the symmetric tick -> MASK_EXTRA>0 models the hw.
  assign edge_tick_rdata  = tpipe[EDGE_LAT-1+TICK_EXTRA];
  assign edge_coeff_rdata = cpipe[EDGE_LAT-1+`MASK_EXTRA];
  assign edge_mask_rdata  = mpipe[EDGE_LAT-1+`MASK_EXTRA];

  zlc_edge_streamer #(.CHANNEL_COUNT(CH)) dut (
    .clk(clk), .reset(reset), .start(start),
    .prog_count(NE[EAW:0]), .repeat_forever(1'b1),
    .loop_start_addr({EAW{1'b0}}), .loop_end_tick(32'd14101), .loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1), .repeat_from_loop_start(1'b0),
    .scan_enable(1'b0), .scan_count(32'd0),
    .edge_raddr(edge_raddr), .edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata(edge_coeff_rdata), .edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr), .scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11), .bank_chunk0(32'd0), .bank_chunk1(32'd0),
    .scan_cursor(scan_cursor), .underflow(underflow),
    .bus_prog_we(1'b0), .bus_prog_bus(2'd0), .bus_prog_addr(6'd0),
    .bus_prog_start_tick(32'd0), .bus_prog_stop_tick(32'd0),
    .bus_prog_start_tick_coeffs({NS*CW{1'b0}}), .bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_start_value(10'd0), .bus_prog_stop_value(10'd0),
    .bus_prog_mode(2'd0), .bus_prog_value_select(3'd0), .bus_prog_stop_value_select(3'd0),
    .bus_counts({BUSC*(6+1){1'b0}}),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),
    .delay_ticks({CH*DTW{1'b0}}),
    .out(out), .bus_out(bus_out), .running(running), .done(done)
  );

  // --- stimulus: reset, hold a while (let the ARM FSM seed shadows), then FIRE ---
  initial begin
    reset = 1; start = 0;
    repeat (200) @(posedge clk);   // arm_step seeds sh_e0..e3 while reset held
    reset = 0;
    @(posedge clk); start = 1; @(posedge clk); start = 0;
  end

  // --- trace emCCD (bit 11) edges in tick units (out is the delayed output; delay=0) ---
  reg prev = 0; integer tcount = 0; reg emccd;
  integer fired_on1=-1, fired_off1=-1, fired_on2=-1, fired_off2=-1, nedges=0;
  always @(posedge clk) begin
    if (running || done) tcount = tcount + 1;
    emccd = out[11];
    if (running && emccd !== prev) begin
      if (emccd) begin
        if (fired_on1<0) fired_on1=tcount; else if (fired_on2<0) fired_on2=tcount;
      end else begin
        if (fired_off1<0) fired_off1=tcount; else if (fired_off2<0) fired_off2=tcount;
      end
      nedges = nedges + 1;
      $display("[t=%0d] emCCD %s (out=0x%h)", tcount, emccd ? "ON " : "OFF", out[15:0]);
      prev = emccd;
    end
  end

  initial begin
    @(negedge reset);
    repeat (15000) @(posedge clk);
    $display("==== RESULT EDGE_LAT=%0d TICK_EXTRA=%0d ====", EDGE_LAT, TICK_EXTRA);
    $display("pulse1 on=%0d off=%0d  pulse2 on=%0d off=%0d", fired_on1, fired_off1, fired_on2, fired_off2);
    if (fired_on2>=0 && fired_off2>=0)
      $display("pulse2 width = %0d ticks (correct=2000 ; 40ms-bug=4000)", fired_off2-fired_on2);
    $finish;
  end
endmodule
