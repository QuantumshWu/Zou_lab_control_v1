`timescale 1ns/1ps
// Engine-faithful skew: a REGISTERED raddr (like edge_raddr) stepped 0->5 at a marked cycle,
// free-running counter, watch tdout and mdout.  The cycle each first reflects row5 = its
// latency; the DIFFERENCE is the tick/mask skew the prefetch sees.
module tb_bram_lat;
  reg clk=0; always #5 clk=~clk;
  reg [12:0] araddr=0; reg [31:0] dina=0; reg [3:0] wea=0; reg ena=0;
  reg [11:0] raddr=0; reg step=0;
  wire [31:0] tdout; wire [63:0] mdout;
  blk_mem_gen_edge_tick tk(.clka(clk),.ena(ena),.wea(wea),.addra(araddr[11:0]),.dina(dina),.douta(),
                           .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(raddr),.dinb(32'b0),.doutb(tdout));
  blk_mem_gen_edge_mask mk(.clka(clk),.ena(ena),.wea(wea),.addra(araddr),.dina(dina),.douta(),
                           .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(raddr),.dinb(32'b0),.doutb(mdout));
  // registered raddr exactly like the engine's edge_raddr
  always @(posedge clk) if (step) raddr <= 12'd5;
  integer c, tlat, mlat;
  task wr(input [12:0] a, input [31:0] d); begin
    @(posedge clk); ena<=1; wea<=4'hF; araddr<=a; dina<=d; @(posedge clk); ena<=0; wea<=0; end
  endtask
  initial begin
    wr(13'd5, 32'h55555555);   // tick row5
    wr(13'd6, 32'h66666666);   // mask row5 low word
    wr(13'd7, 32'h77777777);   // mask row5 high word  (row5 = {word7,word6})
    repeat (8) @(posedge clk);
    @(posedge clk); step<=1;   // raddr<-5 registered at the NEXT posedge
    @(posedge clk); step<=0;   // (raddr is now 5)
    c=0; tlat=-1; mlat=-1;
    repeat (7) begin #1;
      $display("c=%0d raddr=%0d tdout=%h mdout=%h", c, raddr, tdout, mdout);
      if (tlat<0 && tdout===32'h55555555) tlat=c;
      if (mlat<0 && mdout===64'h7777777766666666) mlat=c;
      @(posedge clk); c=c+1; end
    $display("==> tick latency=%0d  mask latency=%0d  SKEW(mask-tick)=%0d", tlat, mlat, mlat-tlat);
    $finish;
  end
endmodule
