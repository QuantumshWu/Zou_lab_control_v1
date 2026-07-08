`timescale 1ns/1ps
// End-to-end proof of the UART READ path: drive a real READ(LAYOUT_ID) frame in on uart_rx and DECODE
// the serial reply on uart_tx (NOT just peek an internal reg).  This catches BOTH read-path bugs:
//   1. read-tap latency  -- the top's u_rd_data tap must be COMBINATIONAL (a registered tap latches the
//      PREVIOUS word into wbuf -> stale reads).
//   2. reply byte mux    -- `pj[1:0]<<3` is self-determined to 2 bits, so the shift truncates to 0 and
//      every payload byte comes out as byte 0 (0x5A4C4C02 -> 0x02020202 on the wire).  Must be a wide
//      offset ({pj[1:0],3'b000}).  An earlier tb that only checked wbuf[0] MISSED this -- decode the wire.
//
// Compile FIXED : xvlog zlc_uart_bridge.v tb_uart_read_tap.v                 (combinational tap -> PASS)
// Compile BUG   : xvlog -d REGISTERED_BUG zlc_uart_bridge.v tb_uart_read_tap.v  (registered tap -> stale)
module tb_uart_read_tap;
    localparam [31:0] LAYOUT = 32'h5A4C4C02;
    real BITT = 333.333;                            // 3 Mbaud bit period (ns)

    reg clk = 1'b0;  always #10 clk = ~clk;         // 50 MHz (20 ns)
    reg  rst = 1'b1;
    reg  uart_rx = 1'b1;                            // idle high
    wire uart_tx;
    wire [29:0] u_word_addr; wire [31:0] u_wdata; wire u_we, u_active;
    wire [5:0]  u_rd_word;   wire u_rd_req;
    reg  [31:0] u_rd_data;
    reg  [31:0] ctrl_reg [0:63];

    // read tap under test (mirrors zlc_pulse_streamer_top): word 63 -> hardwired LAYOUT id, else CTRL regfile
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

    // ---- send one 8N1 byte (LSB first) on uart_rx ----
    task send_byte(input [7:0] b);
        integer i;
        begin
            uart_rx = 1'b0; #(BITT);
            for (i = 0; i < 8; i = i + 1) begin uart_rx = b[i]; #(BITT); end
            uart_rx = 1'b1; #(BITT);
        end
    endtask

    // ---- receive one 8N1 byte (LSB first) from uart_tx ----
    task recv_byte(output [7:0] b);
        integer i;
        begin
            @(negedge uart_tx);          // start bit
            #(BITT * 1.5);               // to the middle of data bit 0
            for (i = 0; i < 8; i = i + 1) begin b[i] = uart_tx; #(BITT); end
        end
    endtask

    // reply collector (runs concurrently)
    reg [7:0] rx [0:31];
    integer   nrx = 0;
    initial begin : collector
        forever begin
            recv_byte(rx[nrx]);
            nrx = nrx + 1;
        end
    end

    // READ(word=63,count=1,seq=1) == uf.encode_read(63,1,seq=1) == 5a a5 02 01 3f 00 00 00 01 00 47 df
    reg [7:0] frame [0:11];
    integer k;
    reg [31:0] payload;
    initial begin
        frame[0]=8'h5a; frame[1]=8'ha5; frame[2]=8'h02; frame[3]=8'h01;
        frame[4]=8'h3f; frame[5]=8'h00; frame[6]=8'h00; frame[7]=8'h00;
        frame[8]=8'h01; frame[9]=8'h00; frame[10]=8'h47; frame[11]=8'hdf;
        for (k = 0; k < 64; k = k + 1) ctrl_reg[k] = 32'hDEAD0000 + k;   // distinctive; word 63 must read LAYOUT

        #200 rst = 1'b0;
        #4000;
        for (k = 0; k < 12; k = k + 1) send_byte(frame[k]);

        wait (nrx >= 13);            // full 13-byte reply received (hdr7 + payload4 + crc2)

        // reply = SYNC0 SYNC1 RESP SEQ STATUS CNT[2] PAYLOAD[4] CRC[2]; payload word is LE
        payload = {rx[10], rx[9], rx[8], rx[7]};
        $display("TB: reply hdr = %02x %02x %02x seq=%02x st=%02x cnt=%02x%02x", rx[0],rx[1],rx[2],rx[3],rx[4],rx[6],rx[5]);
        $display("TB: reply payload bytes = %02x %02x %02x %02x  -> word 0x%08X (expect 0x%08X)",
                 rx[7], rx[8], rx[9], rx[10], payload, LAYOUT);
        $display("TB: (internal wbuf[0] = 0x%08X)", dut.wbuf[0]);
        if (payload === LAYOUT && rx[0]==8'h5a && rx[1]==8'ha5 && rx[2]==8'h81)
            $display("TB RESULT: PASS -- UART reply on the wire carries LAYOUT_ID");
        else
            $display("TB RESULT: FAIL -- UART reply payload 0x%08X != 0x%08X", payload, LAYOUT);
        $finish;
    end

    initial begin #400000 $display("TB RESULT: FAIL -- timeout (got %0d reply bytes)", nrx); $finish; end
endmodule
