`timescale 1ns/1ps
// FULL-CHAIN xsim: the REAL zlc_pulse_streamer_top + REAL zlc_edge_streamer + the FIVE
// REAL blk_mem_gen IPs, driven by the EXACT AXI write sequence decoded from the user's
// actual 40 ms hardware run (fpga/build/state/vivado_axi_session.log, last session),
// ticks scaled /500.  Covers everything the engine-only sim could not: the top's address
// decoder + ctrl regfile, the bus-segment mini-loader, the SAFE->upload->LOAD->FIRE
// command sequencing (arm FSM runs DURING upload, exactly like hardware), the clk mux,
// and the named pin map.  Watches the emCCD pin (out_final[11]).
//
// jtag_axi_0 is stubbed (tied off); axi_bram_ctrl_0 is replaced by a scripted writer
// that drives the bram_* port with the decoded word sequence -- byte-identical to what
// the real controller's burst beats present to the top's decoder.

// ---- fake JTAG master: tied off (the scripted bram writer below replaces the path) ----
module jtag_axi_0(
  input aclk, input aresetn,
  output [0:0] m_axi_awid, output [31:0] m_axi_awaddr, output [7:0] m_axi_awlen,
  output [2:0] m_axi_awsize, output [1:0] m_axi_awburst, output [0:0] m_axi_awlock,
  output [3:0] m_axi_awcache, output [2:0] m_axi_awprot, output [3:0] m_axi_awqos,
  output m_axi_awvalid, input m_axi_awready,
  output [31:0] m_axi_wdata, output [3:0] m_axi_wstrb, output m_axi_wlast,
  output m_axi_wvalid, input m_axi_wready,
  input [0:0] m_axi_bid, input [1:0] m_axi_bresp, input m_axi_bvalid, output m_axi_bready,
  output [0:0] m_axi_arid, output [31:0] m_axi_araddr, output [7:0] m_axi_arlen,
  output [2:0] m_axi_arsize, output [1:0] m_axi_arburst, output [0:0] m_axi_arlock,
  output [3:0] m_axi_arcache, output [2:0] m_axi_arprot, output [3:0] m_axi_arqos,
  output m_axi_arvalid, input m_axi_arready,
  input [0:0] m_axi_rid, input [31:0] m_axi_rdata, input [1:0] m_axi_rresp,
  input m_axi_rlast, input m_axi_rvalid, output m_axi_rready
);
  assign m_axi_awid=0; assign m_axi_awaddr=0; assign m_axi_awlen=0; assign m_axi_awsize=0;
  assign m_axi_awburst=0; assign m_axi_awlock=0; assign m_axi_awcache=0; assign m_axi_awprot=0;
  assign m_axi_awqos=0; assign m_axi_awvalid=0; assign m_axi_wdata=0; assign m_axi_wstrb=0;
  assign m_axi_wlast=0; assign m_axi_wvalid=0; assign m_axi_bready=0;
  assign m_axi_arid=0; assign m_axi_araddr=0; assign m_axi_arlen=0; assign m_axi_arsize=0;
  assign m_axi_arburst=0; assign m_axi_arlock=0; assign m_axi_arcache=0; assign m_axi_arprot=0;
  assign m_axi_arqos=0; assign m_axi_arvalid=0; assign m_axi_rready=0;
endmodule

// ---- scripted bram writer in axi_bram_ctrl_0's place ---------------------------------
module axi_bram_ctrl_0(
  input s_axi_aclk, input s_axi_aresetn,
  input [0:0] s_axi_awid, input [31:0] s_axi_awaddr, input [7:0] s_axi_awlen,
  input [2:0] s_axi_awsize, input [1:0] s_axi_awburst, input [0:0] s_axi_awlock,
  input [3:0] s_axi_awcache, input [2:0] s_axi_awprot, input s_axi_awvalid, output s_axi_awready,
  input [31:0] s_axi_wdata, input [3:0] s_axi_wstrb, input s_axi_wlast,
  input s_axi_wvalid, output s_axi_wready,
  output [0:0] s_axi_bid, output [1:0] s_axi_bresp, output s_axi_bvalid, input s_axi_bready,
  input [0:0] s_axi_arid, input [31:0] s_axi_araddr, input [7:0] s_axi_arlen,
  input [2:0] s_axi_arsize, input [1:0] s_axi_arburst, input [0:0] s_axi_arlock,
  input [3:0] s_axi_arcache, input [2:0] s_axi_arprot, input s_axi_arvalid, output s_axi_arready,
  output [0:0] s_axi_rid, output [31:0] s_axi_rdata, output [1:0] s_axi_rresp,
  output s_axi_rlast, output s_axi_rvalid, input s_axi_rready,
  output bram_rst_a, output bram_clk_a, output reg bram_en_a,
  output reg [3:0] bram_we_a, output reg [31:0] bram_addr_a,
  output reg [31:0] bram_wrdata_a, input [31:0] bram_rddata_a
);
  assign s_axi_awready=0; assign s_axi_wready=0; assign s_axi_bid=0; assign s_axi_bresp=0;
  assign s_axi_bvalid=0; assign s_axi_arready=0; assign s_axi_rid=0; assign s_axi_rdata=0;
  assign s_axi_rresp=0; assign s_axi_rlast=0; assign s_axi_rvalid=0;
  assign bram_rst_a = 1'b0;
  assign bram_clk_a = s_axi_aclk;

  task wr;                          // one word write, exactly one bram beat
    input [29:0] word; input [31:0] data;
    begin
      @(negedge s_axi_aclk);
      bram_en_a <= 1'b1; bram_we_a <= 4'hF;
      bram_addr_a <= {word, 2'b00}; bram_wrdata_a <= data;
      @(negedge s_axi_aclk);
      bram_en_a <= 1'b0; bram_we_a <= 4'h0;
    end
  endtask
  task cmd;                         // host _command(): COMMAND<=0 then COMMAND<=x
    input [31:0] x;
    begin wr(30'd1, 32'd0); wr(30'd1, x); end
  endtask
  task upload;                      // the decoded image (ticks scaled /500)
    begin
`include "replay_image.vh"
    end
  endtask

  integer k;
  initial begin
    bram_en_a=0; bram_we_a=0; bram_addr_a=0; bram_wrdata_a=0;
    repeat (50) @(negedge s_axi_aclk);
    // ---- replay the real session: prepare #1 ----
    cmd(32'd8);                       // CMD_SAFE (engine reset; arm FSM starts NOW)
    repeat (300) @(negedge s_axi_aclk);
    upload;                           // image words (arm may be mid-read, like hardware)
    cmd(32'd1);                       // CMD_LOAD
    repeat (600) @(negedge s_axi_aclk);
    // ---- prepare #2 (the host uploaded twice, byte-identical) ----
    cmd(32'd8);
    repeat (300) @(negedge s_axi_aclk);
    upload;
    cmd(32'd1);
    repeat (600) @(negedge s_axi_aclk);
    // ---- fire() ----
    wr(30'd16, 32'd3);                // BANK_READY = 0b11
    cmd(32'd2);                       // CMD_FIRE
    $display("[TB] FIRE issued");
  end
endmodule

// ---- the testbench ---------------------------------------------------------------
module tb_full_top;
  reg clk = 0; always #10 clk = ~clk;   // 50 MHz board clock
  wire [1:0] led;
  wire cooling, cooling_pgc, repump, probe, pushout, state_pre, trig, coil;
  wire grey_cooling, trap, UV, emCCD, microwave, address_w;
  wire GND1,GND4,GND5,GND6,GND7,GND8,GND9,GND10,GND11,GND12,GND13,GND14,GND15;
  wire cooling_shutter, repump_shutter, probe_shutter, bias;
  wire [9:0] da_dipole, da_bias_y, da_bias_x, da_bias_z;
  wire da_clk0, da_clk1, da_clk2, da_clk3;

  zlc_pulse_streamer_top dut (
    .clk(clk), .led(led),
    .cooling(cooling), .cooling_pgc(cooling_pgc), .repump(repump), .probe(probe),
    .pushout(pushout), .state_pre(state_pre), .trig(trig), .coil(coil),
    .grey_cooling(grey_cooling), .trap(trap), .UV(UV), .emCCD(emCCD),
    .microwave(microwave), .address(address_w),
    .GND1(GND1),.GND4(GND4),.GND5(GND5),.GND6(GND6),.GND7(GND7),.GND8(GND8),
    .GND9(GND9),.GND10(GND10),.GND11(GND11),
    .cooling_shutter(cooling_shutter), .GND12(GND12), .repump_shutter(repump_shutter),
    .GND13(GND13), .probe_shutter(probe_shutter), .GND14(GND14), .bias(bias), .GND15(GND15),
    .da_dipole(da_dipole), .da_clk0(da_clk0),
    .da_bias_y(da_bias_y), .da_clk1(da_clk1),
    .da_bias_x(da_bias_x), .da_clk2(da_clk2),
    .da_bias_z(da_bias_z), .da_clk3(da_clk3)
  );

  // tick counter starts when the engine actually runs (led[0] = zlc_running)
  integer tcount = 0; reg run_prev = 0;
  reg em_prev = 0; integer lastOn = -1;
  always @(posedge clk) begin
    if (led[0]) tcount = tcount + 1;
    if (led[0] && !run_prev) $display("[TB] running at sim time %0t", $time);
    run_prev = led[0];
    if (led[0] && emCCD !== em_prev) begin
      if (emCCD) lastOn = tcount;
      else $display("[PULSE] on=%0d off=%0d width=%0d (expect 1000@2500 and 2000@10001 each frame of 14101)", lastOn, tcount, tcount-lastOn);
      em_prev = emCCD;
    end
  end

  initial begin
    repeat (60000) @(posedge clk);   // upload (~3k) + 3.5 frames (49k)
    $display("==== DONE ====");
    $finish;
  end
endmodule
