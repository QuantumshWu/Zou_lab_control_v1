`timescale 1ns / 1ps
// =============================================================================
// zlc_uart_bridge -- UART fast-control side-channel decoder.
//
// A dedicated serial control link that writes the SAME flat, word-addressed
// register/BRAM map as the JTAG-to-AXI path (host.image.region_bases), so a
// program is BYTE-IDENTICAL on either transport -- a pure transport swap that
// removes the ~1 s Vivado-Tcl/JTAG per-transaction overhead (a whole ~24 KB
// program ~82 ms at 3 Mbaud, a scan-point step ~us).
//
// It presents the top a write interface (u_word_addr/u_wdata/u_we) that the top
// MUXes against the axi_bram_ctrl bram_* side before the region decode, and a
// CTRL-region read tap (u_rd_word/u_rd_req -> u_rd_data) for STATUS/CURSOR/
// LAYOUT_ID.  Only TWO wire opcodes: WRITE (a run of words at a base word addr)
// and READ (a run back).  COMMAND / scan-step / PING are COMPOSED by the host
// from WRITE/READ, so this FSM stays small and image.py is the single source.
//
// Frame (LSB-first, LE fields):
//   request = SYNC0(5A) SYNC1(A5) OP SEQ ADDR[4] COUNT[2] PAYLOAD[4*COUNT] CRC16[2]
//   reply   = SYNC0     SYNC1     RESP(81) SEQ STATUS COUNT[2] PAYLOAD CRC16[2]
// CRC-16/CCITT-FALSE (poly 1021, init FFFF) over OP/RESP..last payload byte.
//
// SAFETY: a whole WRITE frame is BUFFERED (<= FRAME_WORDS words) and its CRC
// verified BEFORE any u_we fires, so a corrupt frame commits NOTHING -- this
// exactly mirrors the Python oracle host/uart_bridge_model.py (equivalence-tested).
// On any fault (bad SYNC/opcode/CRC/framing) it resynchronises by re-hunting the
// SYNC pair (idle line 0xFF never matches 5A,A5).
//
// Verified: the protocol/decoder logic is proven in Python (uart_bridge_model +
// tests/test_uart_bridge_equivalence.py, always-run).  tb_uart_bridge.v drives THIS
// module under Vivado xsim as the on-demand cross-check (needs the Xilinx toolchain;
// not in CI) -- same pattern as the engine tbs.  Real 3 Mbaud baud-lock + FT2232
// ch-B wiring are the rig steps.
//
// Single sys-clk domain (no CDC).  Every output has exactly ONE driving always
// block (synthesisable, no multi-driver).
// =============================================================================

module zlc_uart_bridge #(
    parameter integer CLK_HZ    = 50_000_000,
    parameter integer BAUD      = 3_000_000,      // fractional NCO divider -> any FT/CP2102 rate
    parameter integer OVERSAMPLE = 16,            // samples per bit (RX + TX)
    parameter integer ADDR_WORD_WIDTH = 30,       // == top word_addr width
    parameter integer FRAME_WORDS = 256           // max data words per frame (== host MAX_FRAME_WORDS)
)(
    input  wire clk,
    input  wire rst,                              // power-on reset only, NOT eng_reset
    input  wire uart_rx,
    output wire uart_tx,
    // write interface presented to the top (muxed against the AXI bram write side)
    output reg  [ADDR_WORD_WIDTH-1:0] u_word_addr,
    output reg  [31:0]                u_wdata,
    output reg                        u_we,       // 1-clk commit pulse
    output reg                        u_active,   // level: high from a frame's OP byte to frame done
    // CTRL-region read tap (STATUS/CURSOR/LAYOUT_ID)
    output reg  [5:0]  u_rd_word,
    output reg         u_rd_req,                  // 1-clk pulse; top returns u_rd_data next cycle
    input  wire [31:0] u_rd_data
);
    localparam integer FW_AW = $clog2(FRAME_WORDS);

    localparam [7:0] SYNC0 = 8'h5A, SYNC1 = 8'hA5;
    localparam [7:0] OP_WRITE = 8'h01, OP_READ = 8'h02, RESP = 8'h81;
    localparam [7:0] ST_OK = 8'h00;

    // ---------------------------------------------------------------- CRC-16/CCITT-FALSE
    function [15:0] crc_byte;
        input [15:0] crc; input [7:0] b; integer i; reg [15:0] c;
        begin
            c = crc ^ {b, 8'h00};
            for (i = 0; i < 8; i = i + 1) c = c[15] ? ((c << 1) ^ 16'h1021) : (c << 1);
            crc_byte = c;
        end
    endfunction

    // ---------------------------------------------------------------- baud NCO (OVERSAMPLE x baud tick)
    localparam integer NCO_INC = BAUD * OVERSAMPLE;
    reg [31:0] nco; reg os_tick;
    always @(posedge clk) begin
        if (rst) begin nco <= 0; os_tick <= 1'b0; end
        else if (nco + NCO_INC >= CLK_HZ) begin nco <= nco + NCO_INC - CLK_HZ; os_tick <= 1'b1; end
        else begin nco <= nco + NCO_INC; os_tick <= 1'b0; end
    end

    // ---------------------------------------------------------------- UART RX (8N1, 16x oversample)
    reg [1:0] rx_sync; always @(posedge clk) rx_sync <= {rx_sync[0], uart_rx};
    wire rx_in = rx_sync[1];
    localparam RX_IDLE=2'd0, RX_START=2'd1, RX_DATA=2'd2, RX_STOP=2'd3;
    reg [1:0] rx_st; reg [4:0] os_cnt; reg [2:0] bit_cnt; reg [7:0] rx_sh;
    reg [7:0] rx_byte; reg rx_valid; reg rx_ferr;
    always @(posedge clk) begin
        if (rst) begin rx_st<=RX_IDLE; rx_valid<=1'b0; rx_ferr<=1'b0; os_cnt<=0; bit_cnt<=0; end
        else begin
            rx_valid<=1'b0; rx_ferr<=1'b0;
            if (os_tick) case (rx_st)
                RX_IDLE:  if (~rx_in) begin rx_st<=RX_START; os_cnt<=0; end
                RX_START: if (os_cnt==(OVERSAMPLE/2-1)) begin
                              if (~rx_in) begin rx_st<=RX_DATA; os_cnt<=0; bit_cnt<=0; end
                              else rx_st<=RX_IDLE;
                          end else os_cnt<=os_cnt+1'b1;
                RX_DATA:  if (os_cnt==OVERSAMPLE-1) begin
                              os_cnt<=0; rx_sh<={rx_in, rx_sh[7:1]};
                              if (bit_cnt==3'd7) rx_st<=RX_STOP; else bit_cnt<=bit_cnt+1'b1;
                          end else os_cnt<=os_cnt+1'b1;
                RX_STOP:  if (os_cnt==OVERSAMPLE-1) begin
                              os_cnt<=0; rx_st<=RX_IDLE;
                              if (rx_in) begin rx_byte<=rx_sh; rx_valid<=1'b1; end
                              else rx_ferr<=1'b1;
                          end else os_cnt<=os_cnt+1'b1;
            endcase
        end
    end

    // ---------------------------------------------------------------- decoder FSM
    localparam D_HUNT=4'd0, D_SYNC1=4'd1, D_OP=4'd2, D_SEQ=4'd3, D_ADDR=4'd4, D_CNT=4'd5,
               D_DATA=4'd6, D_CRC=4'd7, D_COMMIT=4'd8, D_READ=4'd9, D_RLAT=4'd10, D_WEND=4'd11;
    reg [3:0]  dst;
    reg [7:0]  f_op, f_seq;
    reg [31:0] f_addr, word_acc;
    reg [15:0] f_count, w_idx, rd_i;
    reg [1:0]  b_idx, byte_in_word, crc_bytes;
    reg [15:0] crc_run, crc_rx;
    (* ram_style="distributed" *) reg [31:0] wbuf [0:FRAME_WORDS-1];   // WRITE payload + READ reply staging

    // handshake to the reply serializer (single-writer here, single-reader there)
    reg        rpl_go; reg [7:0] rpl_seq; reg [15:0] rpl_count;

    always @(posedge clk) begin
        if (rst) begin dst<=D_HUNT; u_we<=1'b0; u_active<=1'b0; u_rd_req<=1'b0; rpl_go<=1'b0; end
        else begin
            u_we<=1'b0; u_rd_req<=1'b0; rpl_go<=1'b0;
            if (rx_ferr) begin dst<=D_HUNT; u_active<=1'b0; end   // framing error -> abort + resync
            else begin
                case (dst)
                    D_HUNT:  begin u_active<=1'b0; if (rx_valid && rx_byte==SYNC0) dst<=D_SYNC1; end
                    D_SYNC1: if (rx_valid) begin
                                 if (rx_byte==SYNC1) dst<=D_OP;
                                 else if (rx_byte==SYNC0) dst<=D_SYNC1;
                                 else dst<=D_HUNT; end
                    D_OP:    if (rx_valid) begin
                                 f_op<=rx_byte; crc_run<=crc_byte(16'hFFFF, rx_byte); u_active<=1'b1;
                                 dst <= (rx_byte==OP_WRITE || rx_byte==OP_READ) ? D_SEQ : D_HUNT; end
                    D_SEQ:   if (rx_valid) begin f_seq<=rx_byte; crc_run<=crc_byte(crc_run,rx_byte); b_idx<=0; dst<=D_ADDR; end
                    D_ADDR:  if (rx_valid) begin
                                 f_addr<={rx_byte, f_addr[31:8]}; crc_run<=crc_byte(crc_run,rx_byte);
                                 if (b_idx==2'd3) begin b_idx<=0; dst<=D_CNT; end else b_idx<=b_idx+1'b1; end
                    D_CNT:   if (rx_valid) begin
                                 crc_run<=crc_byte(crc_run,rx_byte);
                                 if (b_idx==2'd0) begin f_count[7:0]<=rx_byte; b_idx<=1; end
                                 else begin
                                     f_count[15:8]<=rx_byte; b_idx<=0; w_idx<=0; byte_in_word<=0; crc_bytes<=0;
                                     if (f_op==OP_WRITE) dst <= ({rx_byte,f_count[7:0]}==16'd0) ? D_CRC : D_DATA;
                                     else dst<=D_CRC;
                                 end end
                    D_DATA:  if (rx_valid) begin
                                 crc_run<=crc_byte(crc_run,rx_byte); word_acc<={rx_byte, word_acc[31:8]};
                                 if (byte_in_word==2'd3) begin
                                     byte_in_word<=0; wbuf[w_idx[FW_AW-1:0]]<={rx_byte, word_acc[31:8]};
                                     if (w_idx==f_count-1) dst<=D_CRC; else w_idx<=w_idx+1'b1;
                                 end else byte_in_word<=byte_in_word+1'b1; end
                    D_CRC:   if (rx_valid) begin
                                 if (crc_bytes==2'd0) begin crc_rx[7:0]<=rx_byte; crc_bytes<=1; end
                                 else begin
                                     if ({rx_byte, crc_rx[7:0]}==crc_run) begin
                                         if (f_count==16'd0) begin
                                             // A protocol-legal count==0 frame commits / reads NOTHING and
                                             // ACKs with count 0 (matches the Python oracle UartBridgeModel).
                                             // No u_we is asserted, so u_active drops now.  This guards the
                                             // `w_idx==f_count-1` / `rd_i==f_count-1` UNDERFLOW (0-1==0xFFFF)
                                             // that would else loop D_COMMIT/D_RLAT 65536 times, scribbling
                                             // the whole register/BRAM map.
                                             rpl_seq<=f_seq; rpl_count<=16'd0; rpl_go<=1'b1;
                                             dst<=D_HUNT; u_active<=1'b0;
                                         end else if (f_op==OP_WRITE) begin w_idx<=0; dst<=D_COMMIT; end
                                         else begin rd_i<=0; dst<=D_READ; end
                                     end else begin dst<=D_HUNT; u_active<=1'b0; end   // bad CRC -> commit nothing
                                 end end
                    // --- COMMIT: one u_we per buffered word (CRC already verified), then ACK ---
                    D_COMMIT: begin
                                 u_word_addr <= f_addr[ADDR_WORD_WIDTH-1:0] + w_idx;
                                 u_wdata     <= wbuf[w_idx[FW_AW-1:0]];
                                 u_we        <= 1'b1;
                                 if (w_idx==f_count-1) begin
                                     rpl_seq<=f_seq; rpl_count<=16'd0; rpl_go<=1'b1;   // ACK (count 0) -> host retries on timeout
                                     dst<=D_WEND;                                      // hold u_active 1 more cycle (see D_WEND)
                                 end else w_idx<=w_idx+1'b1;
                              end
                    // The LAST word's u_we (asserted in D_COMMIT via non-blocking) actually drives the
                    // write THIS cycle -- so u_active MUST still be high now, or the top's uart_sel=u_active
                    // priority mux routes the write to the idle JTAG side and DROPS it (every write frame
                    // would lose its last word; a 1-word write loses everything).  Retire only after it lands.
                    D_WEND:  begin dst<=D_HUNT; u_active<=1'b0; end
                    // --- READ: request word rd_i, latch it next cycle into wbuf, then reply ---
                    D_READ:  begin u_rd_word <= f_addr[5:0] + rd_i[5:0]; u_rd_req<=1'b1; dst<=D_RLAT; end
                    D_RLAT:  begin
                                 wbuf[rd_i[FW_AW-1:0]] <= u_rd_data;    // 1-cycle read latency
                                 if (rd_i==f_count-1) begin
                                     rpl_seq<=f_seq; rpl_count<=f_count; rpl_go<=1'b1; dst<=D_HUNT; u_active<=1'b0;
                                 end else begin rd_i<=rd_i+1'b1; dst<=D_READ; end
                              end
                    default: dst<=D_HUNT;
                endcase
            end
        end
    end

    // ---------------------------------------------------------------- reply serializer (SINGLE driver of TX)
    // Emits SYNC0 SYNC1 RESP SEQ STATUS COUNT[2] PAYLOAD[4*COUNT] CRC16[2] one byte per idle TX window.
    // reply data words are in wbuf[0..rpl_count-1] (staged by D_RLAT).
    localparam S_IDLE=2'd0, S_LOAD=2'd1, S_WAIT=2'd2;
    reg [1:0]  sst;
    reg [15:0] t_i, t_total, t_pcut;      // t_pcut = flat index of the first CRC byte (7 + 4*count)
    reg [15:0] t_crc_run;                 // reply CRC accumulated as bytes are emitted (no runtime loop)
    reg [7:0]  hdr [0:6];
    reg        tx_send; reg [7:0] tx_data; wire tx_busy;

    // combinational current-byte mux over the flat reply index t_i (header / payload / CRC)
    reg [7:0]  cur_byte; reg [15:0] pj;
    always @(*) begin
        cur_byte = 8'h00; pj = 16'd0;
        if (t_i < 16'd7) cur_byte = hdr[t_i[2:0]];                     // SYNC0 SYNC1 RESP SEQ ST CNT[2]
        else if (t_i < t_pcut) begin
            pj = t_i - 16'd7;
            // LE byte pj%4 of reply word pj/4.  Bit offset MUST be a wide expression: `pj[1:0]<<3`
            // is self-determined to the 2-bit width of pj[1:0], so the <<3 truncates to 0 and every
            // payload byte would come out as byte 0 (0x5A4C4C02 -> 0x02020202 on the wire).  Concatenate
            // 3 zero bits instead ( == pj[1:0]*8, a proper 5-bit 0/8/16/24 offset).
            cur_byte = wbuf[pj[FW_AW+1:2]][ {pj[1:0], 3'b000} +: 8 ];
        end else cur_byte = (t_i == t_pcut) ? t_crc_run[7:0] : t_crc_run[15:8];
    end

    // Race-free TX handshake: S_LOAD asserts `send` and HOLDS it until the TX has accepted it
    // (busy RISES), only THEN goes to S_WAIT for busy to FALL.  This avoids the 1-cycle window where
    // `send` is still propagating and busy has not yet risen (which a bare ~busy check would misread
    // as "byte done").  `tx_send` is driven ONLY in this block (single driver).
    always @(posedge clk) begin
        if (rst) begin sst<=S_IDLE; tx_send<=1'b0; end
        else case (sst)
            S_IDLE: begin
                tx_send<=1'b0;
                if (rpl_go) begin
                    hdr[0]<=SYNC0; hdr[1]<=SYNC1; hdr[2]<=RESP; hdr[3]<=rpl_seq; hdr[4]<=ST_OK;
                    hdr[5]<=rpl_count[7:0]; hdr[6]<=rpl_count[15:8];
                    t_pcut  <= 16'd7 + (rpl_count<<2);
                    t_total <= 16'd7 + (rpl_count<<2) + 16'd2;
                    t_crc_run <= 16'hFFFF;                    // seeded; folded per emitted byte in S_LOAD
                    t_i<=0; sst<=S_LOAD;
                end
            end
            S_LOAD: begin
                tx_data <= cur_byte;
                if (tx_busy) begin                            // TX accepted this byte (busy rose)
                    tx_send<=1'b0;
                    if (t_i >= 16'd2 && t_i < t_pcut)         // fold RESP..last-payload into the reply CRC
                        t_crc_run <= crc_byte(t_crc_run, cur_byte);
                    sst<=S_WAIT;
                end else tx_send<=1'b1;                        // request; hold until accepted
            end
            S_WAIT: begin
                tx_send<=1'b0;
                if (~tx_busy) begin                                   // byte fully shifted out
                    if (t_i==t_total-1) sst<=S_IDLE;
                    else begin t_i<=t_i+1'b1; sst<=S_LOAD; end
                end
            end
        endcase
    end

    zlc_uart_tx #(.OVERSAMPLE(OVERSAMPLE)) zlc_txser (
        .clk(clk), .rst(rst), .os_tick(os_tick), .send(tx_send), .data(tx_data),
        .busy(tx_busy), .tx(uart_tx)
    );
endmodule

// ------------------------------------------------------------------- 8N1 TX byte serializer
module zlc_uart_tx #(parameter integer OVERSAMPLE=16)(
    input  wire clk, input wire rst, input wire os_tick,
    input  wire send, input wire [7:0] data,
    output reg  busy, output reg tx
);
    reg [4:0] os; reg [3:0] bitn; reg [9:0] sh;    // {stop=1, data[8], start=0}, shifted LSB-first
    always @(posedge clk) begin
        if (rst) begin busy<=1'b0; tx<=1'b1; os<=0; bitn<=0; end
        else if (~busy) begin
            tx<=1'b1;
            if (send) begin sh<={1'b1, data, 1'b0}; busy<=1'b1; os<=0; bitn<=0; tx<=1'b0; end   // start bit now
        end else if (os_tick) begin
            if (os==OVERSAMPLE-1) begin
                os<=0;
                if (bitn==4'd9) begin busy<=1'b0; tx<=1'b1; end                 // last bit shifted -> idle
                else begin tx<=sh[1]; sh<={1'b1, sh[9:1]}; bitn<=bitn+1'b1; end  // next bit
            end else os<=os+1'b1;
        end
    end
endmodule
