`timescale 1ns/1ps
// DEFINITIVE: real zlc_edge_streamer + REAL tick & mask IP BRAMs (true synthesized latency),
// preloaded with the user's EXACT uploaded edge table (scaled /500), then FIRE and watch emCCD.
module tb_real_engine;
  localparam integer CH=62, EAW=12, TW=32, NS=4, CW=16, DTW=12, BUSC=4, BW=10;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;   // 50MHz

  reg [12:0] wa=0; reg [31:0] wd=0; reg [3:0] we=0; reg wen_t=0, wen_m=0;
  wire [EAW-1:0] edge_raddr;
  wire [TW-1:0]  edge_tick_rdata;
  wire [63:0]    edge_mask_rdata64;
  wire [CH-1:0]  edge_mask_rdata = edge_mask_rdata64[CH-1:0];
  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;

  blk_mem_gen_edge_tick u_tick(
    .clka(clk),.ena(wen_t),.wea(we),.addra(wa[11:0]),.dina(wd),.douta(),
    .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(edge_raddr),.dinb(32'b0),.doutb(edge_tick_rdata));
  blk_mem_gen_edge_mask u_mask(
    .clka(clk),.ena(wen_m),.wea(we),.addra(wa),.dina(wd),.douta(),
    .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(edge_raddr),.dinb(32'b0),.doutb(edge_mask_rdata64));

  zlc_edge_streamer #(.CHANNEL_COUNT(CH)) dut (
    .clk(clk),.reset(reset),.start(start),
    .prog_count(13'd10),.repeat_forever(1'b1),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd14101),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1),.repeat_from_loop_start(1'b0),
    .scan_enable(1'b0),.scan_count(32'd0),
    .edge_raddr(edge_raddr),.edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_prog_we(1'b0),.bus_prog_bus(2'd0),.bus_prog_addr(6'd0),
    .bus_prog_start_tick(32'd0),.bus_prog_stop_tick(32'd0),
    .bus_prog_start_tick_coeffs({NS*CW{1'b0}}),.bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_start_value(10'd0),.bus_prog_stop_value(10'd0),
    .bus_prog_mode(2'd0),.bus_prog_value_select(3'd0),.bus_prog_stop_value_select(3'd0),
    .bus_counts({BUSC*7{1'b0}}),.bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  integer i;
  reg [31:0] ticks [0:9]; reg [31:0] masks [0:9];

  task pa_write;
    input tgt; input [12:0] a; input [31:0] d;
    begin
      @(posedge clk); wen_t<=tgt; wen_m<=~tgt; we<=4'hF; wa<=a; wd<=d;
      @(posedge clk); @(posedge clk); wen_t<=0; wen_m<=0; we<=4'h0; @(posedge clk);
    end
  endtask

  initial begin
    ticks[0]=0; ticks[1]=1000; ticks[2]=2500; ticks[3]=4500; ticks[4]=5000;
    ticks[5]=5001; ticks[6]=10001; ticks[7]=12001; ticks[8]=14001; ticks[9]=14101;
    masks[0]='h685; masks[1]='h200; masks[2]='ha08; masks[3]='h200; masks[4]='h200;
    masks[5]='h200; masks[6]='ha00; masks[7]='h208; masks[8]=0; masks[9]=0;
    reset=1; start=0; wen_t=0; wen_m=0; we=0;
    for (i=0;i<10;i=i+1) pa_write(1'b1, i, ticks[i]);
    for (i=0;i<10;i=i+1) begin pa_write(1'b0, 2*i, masks[i]); pa_write(1'b0, 2*i+1, 32'd0); end
    repeat (200) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  reg rprev=0; reg [CH-1:0] o_prev=0; reg [EAW-1:0] r_prev=0; integer tcount=0;
  always @(posedge clk) begin
    if (running && !rprev) $display("[RUNNING] start");
    rprev<=running;
    if (running && (out!==o_prev || edge_raddr!==r_prev)) begin
      $display("  t=%0d raddr=%0d tick=%0d mask=0x%h out=0x%h", tcount, edge_raddr, edge_tick_rdata, edge_mask_rdata, out[15:0]);
      o_prev<=out; r_prev<=edge_raddr;
    end
  end
  reg prev=0; reg em;
  integer on1=-1,off1=-1,on2=-1,off2=-1;
  always @(posedge clk) begin
    if (running||done) tcount=tcount+1;
    em=out[11];
    if (running && em!==prev) begin
      if (em) begin if(on1<0)on1=tcount; else if(on2<0)on2=tcount; end
      else    begin if(off1<0)off1=tcount; else if(off2<0)off2=tcount; end
      $display("[t=%0d] emCCD %s out=0x%h", tcount, em?"ON ":"OFF", out[15:0]);
      prev=em;
    end
  end
  integer dc=0;
  always @(posedge clk) if (running) begin
    if (dc<20 || (dc%500==0)) $display("  [dut dc=%0d] tc=%0d aidx=%0d acnt=%0d armnv=%0d armt0=%0d step=%0d kick=%0d sm=0x%h",
      dc, dut.time_count, dut.edge_index, dut.active_count, dut.arm_nv, dut.arm_t[0], dut.arm_step, dut.arm_kicked, dut.state_mask[15:0]);
    dc=dc+1;
  end
  initial begin
    wait (reset==1); wait (reset==0);
    repeat (16000) @(posedge clk);
    $display("==== pulse1 w=%0d  pulse2 on=%0d off=%0d w=%0d ====", off1-on1, on2,off2,off2-on2);
    $display("EXPECT correct pulse2 width=2000 (20ms) ; 40ms-bug=4000");
    $finish;
  end
endmodule
