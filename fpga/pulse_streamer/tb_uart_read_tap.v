`timescale 1ns/1ps
// End-to-end UART bring-up proof: drive real frames in on uart_rx, DECODE the serial replies on
// uart_tx.  Covers every UART bug hardware surfaced (all invisible to the functional Python model):
//   1. read-tap latency   -- top's u_rd_data tap must be COMBINATIONAL (registered -> stale reads).
//   2. reply byte mux      -- `pj[1:0]<<3` truncates to 2 bits -> every payload byte = byte 0
//                             (0x5A87FD36 -> 0x36363636).  Must be {pj[1:0],3'b000}.
//   3. write last-word drop -- D_COMMIT dropped u_active the SAME cycle the last u_we takes effect,
//                             so the top's uart_sel=u_active mux routed it to idle JTAG (every write
//                             lost its last word; a 1-word write lost everything).  Fixed by D_WEND.
//
// The tb models the top's UART write into ctrl_reg WITH the u_active gate, so bug 3 is reproduced.
// Compile: xvlog zlc_uart_bridge.v tb_uart_read_tap.v ; xelab tb_uart_read_tap -s t ; xsim t -R
module tb_uart_read_tap;
    localparam [31:0] LAYOUT = 32'h5A87FD36;   // current shipped image.build_fingerprint (word-63 readback)
    real BITT = 333.333;                            // 3 Mbaud bit period (ns)

    reg clk = 1'b0;  always #10 clk = ~clk;         // 50 MHz
    reg  rst = 1'b1;
    reg  uart_rx = 1'b1;
    wire uart_tx;
    wire [29:0] u_word_addr; wire [31:0] u_wdata; wire u_we, u_active;
    wire [5:0]  u_rd_word;   wire u_rd_req;
    reg  [31:0] u_rd_data;
    reg  [31:0] ctrl_reg [0:63];

    // top model: UART write into CTRL (uart_sel=u_active priority mux + registered ctrl write) ...
    always @(posedge clk) if (u_active && u_we) ctrl_reg[u_word_addr[5:0]] <= u_wdata;
    // ... and the CTRL read tap (word 63 -> hardwired LAYOUT id, else regfile).  MUST be combinational.
`ifdef REGISTERED_BUG
    always @(posedge clk) u_rd_data <= (u_rd_word == 6'd63) ? LAYOUT : ctrl_reg[u_rd_word];
`else
    always @(*)           u_rd_data  = (u_rd_word == 6'd63) ? LAYOUT : ctrl_reg[u_rd_word];
`endif

    zlc_uart_bridge #(.CLK_HZ(50_000_000), .BAUD(3_000_000)) dut (
        .clk(clk), .rst(rst), .uart_rx(uart_rx), .uart_tx(uart_tx),
        .u_word_addr(u_word_addr), .u_wdata(u_wdata), .u_we(u_we), .u_active(u_active),
        .u_rd_word(u_rd_word), .u_rd_req(u_rd_req), .u_rd_data(u_rd_data)
    );

`ifdef DBG
    always @(posedge clk) if (u_active && u_we) $display("  [%0t] CTRL WRITE reg[%0d] <= %h", $time, u_word_addr[5:0], u_wdata);
    always @(u_rd_word) $display("  [%0t] RD u_rd_word=%0d u_rd_data=%h", $time, u_rd_word, u_rd_data);
    always @(dut.wbuf[0]) $display("  [%0t] wbuf[0] <= %h", $time, dut.wbuf[0]);
    always @(posedge clk) if (dut.sst==2'd1 && dut.tx_busy)
        $display("  [%0t] EMIT t_i=%0d pj=%0d cur=%h wbuf0=%h", $time, dut.t_i, dut.pj, dut.cur_byte, dut.wbuf[0]);
`endif

    task send_byte(input [7:0] b);
        integer i; begin
            uart_rx = 1'b0; #(BITT);
            for (i = 0; i < 8; i = i + 1) begin uart_rx = b[i]; #(BITT); end
            uart_rx = 1'b1; #(BITT);
        end
    endtask
    task recv_byte(output [7:0] b);
        integer i; begin
            @(negedge uart_tx); #(BITT * 1.5);
            for (i = 0; i < 8; i = i + 1) begin b[i] = uart_tx; #(BITT); end
        end
    endtask

    reg [7:0] fw  [0:19];   // WRITE(40,[AABBCCDD,11223344],seq=2)
    reg [7:0] fr0 [0:11];   // READ(40,seq=3)
    reg [7:0] fr1 [0:11];   // READ(41,seq=4)
    reg [7:0] frl [0:11];   // READ(63,seq=1)
    reg [7:0] rb  [0:63];   // ACK(9)+3 reads(13 each)=48 bytes; MUST hold all (an [0:31] here read X)
    integer k, fails;
    reg [31:0] pw;

    task send_arr(input integer which, input integer n);
        integer i; reg [7:0] bb; begin
            for (i = 0; i < n; i = i + 1) begin
                case (which) 0: bb=fw[i]; 1: bb=fr0[i]; 2: bb=fr1[i]; 3: bb=frl[i]; default: bb=8'h00; endcase
                send_byte(bb);
            end
        end
    endtask

    // ALWAYS-ON receiver: the bridge replies within a few cycles of the last request byte, so a
    // send-then-recv model races (misses the reply's SYNC0).  Collect every uart_tx byte concurrently.
    integer nrx;
    initial begin : collector
        nrx = 0;
        forever begin recv_byte(rb[nrx]); nrx = nrx + 1; end
    end
    // a count=1 reply is 13 bytes; payload word 0 (LE) is at base+7..base+10
    task check_reply(input integer base, input [31:0] exp, input [255:0] label);
        begin
            pw = {rb[base+10], rb[base+9], rb[base+8], rb[base+7]};
            $display("TB: %0s -> hdr %02x %02x %02x st=%02x payload=0x%08X (expect 0x%08X)",
                     label, rb[base], rb[base+1], rb[base+2], rb[base+4], pw, exp);
            if (!(rb[base]==8'h5a && rb[base+1]==8'ha5 && rb[base+2]==8'h81 && rb[base+4]==8'h00 && pw===exp))
                fails = fails + 1;
        end
    endtask

    initial begin
        fw[0]=8'h5a;fw[1]=8'ha5;fw[2]=8'h01;fw[3]=8'h02;fw[4]=8'h28;fw[5]=8'h00;fw[6]=8'h00;fw[7]=8'h00;
        fw[8]=8'h02;fw[9]=8'h00;fw[10]=8'hdd;fw[11]=8'hcc;fw[12]=8'hbb;fw[13]=8'haa;fw[14]=8'h44;fw[15]=8'h33;
        fw[16]=8'h22;fw[17]=8'h11;fw[18]=8'hee;fw[19]=8'h6d;
        fr0[0]=8'h5a;fr0[1]=8'ha5;fr0[2]=8'h02;fr0[3]=8'h03;fr0[4]=8'h28;fr0[5]=8'h00;fr0[6]=8'h00;fr0[7]=8'h00;fr0[8]=8'h01;fr0[9]=8'h00;fr0[10]=8'h61;fr0[11]=8'h6d;
        fr1[0]=8'h5a;fr1[1]=8'ha5;fr1[2]=8'h02;fr1[3]=8'h04;fr1[4]=8'h29;fr1[5]=8'h00;fr1[6]=8'h00;fr1[7]=8'h00;fr1[8]=8'h01;fr1[9]=8'h00;fr1[10]=8'h85;fr1[11]=8'h31;
        frl[0]=8'h5a;frl[1]=8'ha5;frl[2]=8'h02;frl[3]=8'h01;frl[4]=8'h3f;frl[5]=8'h00;frl[6]=8'h00;frl[7]=8'h00;frl[8]=8'h01;frl[9]=8'h00;frl[10]=8'h47;frl[11]=8'hdf;
        for (k = 0; k < 64; k = k + 1) ctrl_reg[k] = 32'hDEAD0000 + k;   // init != written values
        fails = 0;

        #200 rst = 1'b0;
        #4000;
        send_arr(0, 20); wait (nrx >= 9);            // WRITE 2 words -> 9-byte ACK (count 0)
        $display("TB: WRITE ack hdr %02x %02x %02x st=%02x cnt=%02x%02x", rb[0],rb[1],rb[2],rb[4],rb[6],rb[5]);
        if (!(rb[0]==8'h5a && rb[1]==8'ha5 && rb[2]==8'h81 && rb[4]==8'h00)) fails = fails + 1;

        send_arr(1, 12); wait (nrx >= 22);           // READ w40 -> +13 bytes (reply base 9)
        send_arr(2, 12); wait (nrx >= 35);           // READ w41 -> +13 bytes (reply base 22)
        send_arr(3, 12); wait (nrx >= 48);           // READ w63 -> +13 bytes (reply base 35)

        check_reply(9,  32'hAABBCCDD, "READ w40 (first)");
        check_reply(22, 32'h11223344, "READ w41 (LAST word -> the drop candidate)");
        check_reply(35, LAYOUT,       "READ w63 (LAYOUT_ID)");

        if (fails == 0) $display("TB RESULT: PASS -- write (incl. last word) + reads + LAYOUT all correct on the wire");
        else            $display("TB RESULT: FAIL -- %0d check(s) failed (nrx=%0d)", fails, nrx);
        $finish;
    end
    initial begin #600000 $display("TB RESULT: FAIL -- timeout"); $finish; end
endmodule
