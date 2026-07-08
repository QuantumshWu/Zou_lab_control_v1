`timescale 1ns/1ps
// Cycle-accurate proof of the UART read-tap timing fix (zlc_pulse_streamer_top read tap).
//
// The bridge sets u_rd_word with a NON-BLOCKING assign in D_READ (valid only in the next state,
// D_RLAT) and latches u_rd_data into wbuf THAT SAME D_RLAT cycle.  The top's read tap therefore must
// be COMBINATIONAL (u_rd_data = f(u_rd_word)); a REGISTERED tap adds a second cycle so the bridge
// captures the PREVIOUS word -> stale reads (observed on hardware: LAYOUT_ID read back 0).
//
// Compile FIXED : xvlog zlc_uart_bridge.v tb_uart_read_tap.v          (combinational tap -> PASS)
// Compile BUG   : xvlog -d REGISTERED_BUG zlc_uart_bridge.v tb_uart_read_tap.v  (registered -> FAIL/stale)
module tb_uart_read_tap;
    localparam [31:0] LAYOUT = 32'h5A4C4C02;

    reg clk = 1'b0;  always #10 clk = ~clk;         // 50 MHz (20 ns)
    reg  rst = 1'b1;
    reg  uart_rx = 1'b1;                            // idle high
    wire uart_tx;
    wire [29:0] u_word_addr; wire [31:0] u_wdata; wire u_we, u_active;
    wire [5:0]  u_rd_word;   wire u_rd_req;
    reg  [31:0] u_rd_data;
    reg  [31:0] ctrl_reg [0:63];

    // --- read tap under test: word 63 -> hardwired LAYOUT id, else the CTRL regfile ---
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

    // 8N1 LSB-first at 3 Mbaud (333.33 ns/bit)
    real BITT = 333.333;
    task send_byte(input [7:0] b);
        integer i;
        begin
            uart_rx = 1'b0; #(BITT);                 // start bit
            for (i = 0; i < 8; i = i + 1) begin uart_rx = b[i]; #(BITT); end
            uart_rx = 1'b1; #(BITT);                 // stop bit
        end
    endtask

    // READ(word=63, count=1, seq=1) frame == Python uf.encode_read(63,1,seq=1) == 5a a5 02 01 3f 00 00 00 01 00 47 df
    reg [7:0] frame [0:11];
    integer k;
    initial begin
        frame[0]=8'h5a; frame[1]=8'ha5; frame[2]=8'h02; frame[3]=8'h01;
        frame[4]=8'h3f; frame[5]=8'h00; frame[6]=8'h00; frame[7]=8'h00;
        frame[8]=8'h01; frame[9]=8'h00; frame[10]=8'h47; frame[11]=8'hdf;
        // distinctive CTRL contents so a stale/wrong read is obvious (word 63 must still read LAYOUT)
        for (k = 0; k < 64; k = k + 1) ctrl_reg[k] = 32'hDEAD0000 + k;

        #200 rst = 1'b0;                             // release bridge reset
        #4000;                                       // settle
        for (k = 0; k < 12; k = k + 1) send_byte(frame[k]);
        #4000;                                       // let the FSM stage wbuf[0]

        $display("TB: dut.wbuf[0] = 0x%08X  (expect 0x%08X)", dut.wbuf[0], LAYOUT);
        if (dut.wbuf[0] === LAYOUT)
            $display("TB RESULT: PASS -- UART read staged the correct LAYOUT_ID");
        else
            $display("TB RESULT: FAIL -- UART read staged 0x%08X (STALE), not the LAYOUT_ID", dut.wbuf[0]);
        $finish;
    end

    initial begin #200000 $display("TB RESULT: FAIL -- timeout"); $finish; end
endmodule
