`timescale 1ns/1ps
// Proof that the bridge handles PIPELINED writes: 4 WRITE frames sent BACK-TO-BACK on uart_rx with NO
// inter-frame gap and WITHOUT reading each ACK first (what host-side batching does -- concatenate all
// frames into one serial write).  The host optimisation relies on: (a) the decoder committing each
// frame while the next streams in, and (b) the reply serializer not LOSING an ACK when a later frame's
// rpl_go fires (the 9-byte ACK finishes well before the next >=16-byte frame completes, so the
// serializer is idle in time).  Verifies all 4 ACKs come back AND all 4 words committed (read back).
// Compile: xvlog zlc_uart_bridge.v tb_uart_pipeline.v ; xelab tb_uart_pipeline -s t ; xsim t -R
module tb_uart_pipeline;
    real BITT = 333.333;
    reg clk = 1'b0; always #10 clk = ~clk;
    reg rst = 1'b1, uart_rx = 1'b1; wire uart_tx;
    wire [29:0] u_word_addr; wire [31:0] u_wdata; wire u_we, u_active;
    wire [5:0] u_rd_word; wire u_rd_req; reg [31:0] u_rd_data; reg [31:0] ctrl_reg [0:63];

    always @(posedge clk) if (u_active && u_we) ctrl_reg[u_word_addr[5:0]] <= u_wdata;
    always @(*) u_rd_data = ctrl_reg[u_rd_word];

    zlc_uart_bridge #(.CLK_HZ(50_000_000), .BAUD(3_000_000)) dut (
        .clk(clk), .rst(rst), .uart_rx(uart_rx), .uart_tx(uart_tx),
        .u_word_addr(u_word_addr), .u_wdata(u_wdata), .u_we(u_we), .u_active(u_active),
        .u_rd_word(u_rd_word), .u_rd_req(u_rd_req), .u_rd_data(u_rd_data));

    task send_byte(input [7:0] b); integer i; begin
        uart_rx=1'b0; #(BITT);
        for (i=0;i<8;i=i+1) begin uart_rx=b[i]; #(BITT); end
        uart_rx=1'b1; #(BITT); end
    endtask
    task recv_byte(output [7:0] b); integer i; begin
        @(negedge uart_tx); #(BITT*1.5);
        for (i=0;i<8;i=i+1) begin b[i]=uart_tx; #(BITT); end end
    endtask

    reg [7:0] wr [0:63];    // 4 WRITE frames (16 B each) = 64 B, sent back-to-back
    reg [7:0] rd [0:47];    // 4 READ  frames (12 B each) = 48 B
    reg [7:0] rb [0:127];
    integer k, j, fails; integer nrx;

    initial begin : collector
        nrx = 0; forever begin recv_byte(rb[nrx]); nrx = nrx + 1; end
    end

    // W40..W43 = encode_write(40..43,[0x11111111,0x22222222,0x33333333,0x44444444],seq=1..4)
    initial begin
        {wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7]}   = 64'h5aa50101_28000000;
        {wr[8],wr[9],wr[10],wr[11],wr[12],wr[13],wr[14],wr[15]} = 64'h01001111_11113142;
        {wr[16],wr[17],wr[18],wr[19],wr[20],wr[21],wr[22],wr[23]} = 64'h5aa50102_29000000;
        {wr[24],wr[25],wr[26],wr[27],wr[28],wr[29],wr[30],wr[31]} = 64'h01002222_2222b828;
        {wr[32],wr[33],wr[34],wr[35],wr[36],wr[37],wr[38],wr[39]} = 64'h5aa50103_2a000000;
        {wr[40],wr[41],wr[42],wr[43],wr[44],wr[45],wr[46],wr[47]} = 64'h01003333_33332c6a;
        {wr[48],wr[49],wr[50],wr[51],wr[52],wr[53],wr[54],wr[55]} = 64'h5aa50104_2b000000;
        {wr[56],wr[57],wr[58],wr[59],wr[60],wr[61],wr[62],wr[63]} = 64'h01004444_4444aafd;
        // R40..R43 = encode_read(40..43,1,seq=5..8)
        {rd[0],rd[1],rd[2],rd[3],rd[4],rd[5]}   = 48'h5aa50205_2800; {rd[6],rd[7],rd[8],rd[9],rd[10],rd[11]}  = 48'h00000100_44cc;
        {rd[12],rd[13],rd[14],rd[15],rd[16],rd[17]} = 48'h5aa50206_2900; {rd[18],rd[19],rd[20],rd[21],rd[22],rd[23]} = 48'h00000100_6651;
        {rd[24],rd[25],rd[26],rd[27],rd[28],rd[29]} = 48'h5aa50207_2a00; {rd[30],rd[31],rd[32],rd[33],rd[34],rd[35]} = 48'h00000100_e727;
        {rd[36],rd[37],rd[38],rd[39],rd[40],rd[41]} = 48'h5aa50208_2b00; {rd[42],rd[43],rd[44],rd[45],rd[46],rd[47]} = 48'h00000100_aee8;
        for (k=0;k<64;k=k+1) ctrl_reg[k] = 32'hDEAD0000 + k;
        fails = 0;
        #200 rst = 1'b0; #4000;

        for (k=0;k<64;k=k+1) send_byte(wr[k]);        // 4 WRITE frames BACK-TO-BACK, no ACK read between
        wait (nrx >= 36);                              // collect 4 ACKs (9 B each)
        for (k=0;k<4;k=k+1) begin
            if (!(rb[k*9]==8'h5a && rb[k*9+1]==8'ha5 && rb[k*9+2]==8'h81 && rb[k*9+4]==8'h00)) begin
                fails=fails+1; $display("TB: ACK %0d malformed/lost: %02x %02x %02x st=%02x", k, rb[k*9],rb[k*9+1],rb[k*9+2],rb[k*9+4]);
            end
        end
        $display("TB: 4 back-to-back WRITE frames -> %0d/4 well-formed ACKs collected", 4-fails);

        // READ back ONE AT A TIME (reads are synchronous, never pipelined: a 13-byte read reply is
        // LONGER than the 12-byte read request, so back-to-back reads CAN drop a reply -- unlike the
        // 9-byte WRITE ack that always fits before the next >=16-byte write frame).
        for (k=0;k<4;k=k+1) begin : chk
            reg [31:0] pw; integer b;
            for (j=0;j<12;j=j+1) send_byte(rd[k*12+j]);
            wait (nrx >= 36 + (k+1)*13);
            b = 36 + k*13; pw = {rb[b+10],rb[b+9],rb[b+8],rb[b+7]};
            $display("TB: read w%0d = 0x%08X (expect 0x%08X)", 40+k, pw, 32'h11111111*(k+1) & 32'hFFFFFFFF);
            if (pw !== (32'h11111111*(k+1) & 32'hFFFFFFFF)) fails=fails+1;
        end
        if (fails==0) $display("TB RESULT: PASS -- 4 pipelined writes all ACKed AND all 4 words committed");
        else          $display("TB RESULT: FAIL -- %0d error(s) (nrx=%0d)", fails, nrx);
        $finish;
    end
    initial begin #2000000 $display("TB RESULT: FAIL -- timeout"); $finish; end
endmodule
