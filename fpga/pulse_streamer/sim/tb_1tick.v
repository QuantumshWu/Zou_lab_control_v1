`timescale 1ns/1ps
// 1-tick back-to-back stress: edges at consecutive ticks with ALTERNATING masks, so every
// 20 ns edge must be visible on its own cycle.  Verifies FIFO_DEPTH/PIPE still sustains the
// design's headline 1-tick capability AFTER the pend-depth fix.
module tb_1tick;
  localparam integer CH=62, EAW=12, TW=32, NS=4, CW=16, DTW=12, BUSC=4, BW=10, NE=20;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;
  reg [12:0] wa=0; reg [31:0] wd=0; reg [3:0] we=0; reg wt=0, wm=0;
  wire [EAW-1:0] edge_raddr; wire [TW-1:0] edge_tick_rdata; wire [63:0] mrd;
  wire [CH-1:0] edge_mask_rdata = mrd[CH-1:0];
  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  blk_mem_gen_edge_tick u_t(.clka(clk),.ena(wt),.wea(we),.addra(wa[11:0]),.dina(wd),.douta(),
    .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(edge_raddr),.dinb(32'b0),.doutb(edge_tick_rdata));
  blk_mem_gen_edge_mask u_m(.clka(clk),.ena(wm),.wea(we),.addra(wa),.dina(wd),.douta(),
    .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(edge_raddr),.dinb(32'b0),.doutb(mrd));
  reg [31:0] ticks[0:NE-1]; reg [31:0] masks[0:NE-1];
  zlc_edge_streamer #(.CHANNEL_COUNT(CH)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(NE[12:0]),.repeat_forever(1'b0),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd0),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1),.repeat_from_loop_start(1'b0),.scan_enable(1'b0),.scan_count(32'd0),
    .edge_raddr(edge_raddr),.edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),.bank_ready(2'b11),
    .bank_chunk0(32'd0),.bank_chunk1(32'd0),.scan_cursor(scan_cursor),.underflow(underflow),
    .bus_prog_we(1'b0),.bus_prog_bus(2'd0),.bus_prog_addr(6'd0),.bus_prog_start_tick(32'd0),
    .bus_prog_stop_tick(32'd0),.bus_prog_start_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),.bus_prog_start_value(10'd0),
    .bus_prog_stop_value(10'd0),.bus_prog_mode(2'd0),.bus_prog_value_select(3'd0),
    .bus_prog_stop_value_select(3'd0),.bus_counts({BUSC*7{1'b0}}),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));
  integer i;
  task pa; input t; input [12:0] a; input [31:0] d; begin
    @(posedge clk); wt<=t; wm<=~t; we<=4'hF; wa<=a; wd<=d;
    @(posedge clk); @(posedge clk); wt<=0; wm<=0; we<=0; @(posedge clk); end
  endtask
  initial begin
    // e0..e9 at ticks 0..9 (1-tick back-to-back), then a gap, then e10..e19 at 100..109
    for (i=0;i<10;i=i+1)  begin ticks[i]=i;       masks[i]=(i[0])?62'h001:62'h002; end
    for (i=10;i<20;i=i+1) begin ticks[i]=90+i;    masks[i]=(i[0])?62'h001:62'h002; end
    reset=1; start=0;
    for (i=0;i<NE;i=i+1) pa(1'b1, i, ticks[i]);
    for (i=0;i<NE;i=i+1) begin pa(1'b0, 2*i, masks[i]); pa(1'b0, 2*i+1, 32'd0); end
    repeat (200) @(posedge clk); reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end
  // record the tick at which each edge's mask appears; compare to expected (ticks[i]+1)
  integer tcount=0; reg [CH-1:0] op=62'hx; integer seen=0; integer bad=0; integer exp_t;
  integer dbg=0;
  always @(posedge clk) if (running && dbg<16) begin
    $display("  T%0d tc=%0d nv=%0d a0t=%0d pend=%b land=%b fire=%b infl=%0d fi=%0d raddr=%0d trd=%0d",
      dbg, dut.time_count, dut.arm_nv, dut.arm_t[0], dut.pend, dut.landed, dut.do_fire,
      dut.inflight, dut.fetch_idx, dut.edge_raddr, dut.edge_tick_rdata);
    dbg=dbg+1;
  end
  always @(posedge clk) begin
    if (running||done) tcount=tcount+1;
    if ((running||done) && out!==op) begin
      // each visible change = the next edge firing; expected fire tick = ticks[seen]+1
      exp_t = ticks[seen]+1;
      $display("  edge#%0d out=0x%h at t=%0d (expect %0d) %s", seen, out[7:0], tcount, exp_t, (tcount==exp_t)?"OK":"**LATE**");
      if (tcount != exp_t) bad=bad+1;
      seen=seen+1; op<=out;
    end
  end
  initial begin
    wait(reset==1); wait(reset==0);
    repeat (400) @(posedge clk);
    $display("==== 1-tick: %0d edges fired, %0d off-schedule ====", seen, bad);
    $finish;
  end
endmodule
