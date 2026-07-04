`timescale 1ns/1ps
// Real zlc_edge_streamer driven through a MULTI-SWEEP STREAMED scan with a small BANK_SIZE so
// K = ceil(N/BANK_SIZE) > 2 (here BANK_SIZE=4, N=10 -> K=3, ODD: the case that used to gap).
// A behavioral CYCLIC host-refill model feeds chunks 0,1,..,K-1,0,1,.. one-ahead into the
// alternating ping-pong bank (bank = monotonic_chunk % 2), exactly matching the engine's
// scan_bank_base parity.  Asserts: the engine re-sweeps points 0..N-1 forever with the
// CORRECT slot each point and NEVER stalls (underflow) at the wrap -- truly seamless.
module tb_scan_wrap;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, DTW=12, BUSC=4, BW=10;
  localparam integer BANK_SIZE=4, SAW=3;          // SAW = clog2(BANK_SIZE)+1 = 3 (2 banks x 4)
  localparam integer NPTS=10;                      // K = ceil(10/4) = 3 (odd)
  localparam integer KCH=(NPTS+BANK_SIZE-1)/BANK_SIZE;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  // --- edge program: 2 edges/point, e0@tick0 (bit0=1), e1@tick10 (off); frame=10 ticks ---
  reg [TW-1:0] etick [0:3]; reg [CH-1:0] emask [0:3];
  initial begin etick[0]=0; emask[0]=8'h1; etick[1]=10; emask[1]=8'h0; etick[2]=0; etick[3]=0; emask[2]=0; emask[3]=0; end

  wire [EAW-1:0] edge_raddr; wire [SAW-1:0] scan_raddr;
  wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out; wire running, done;
  wire [TW-1:0] scan_cursor_w; wire underflow;
  // edge reads (behavioral, aligned latency 2 like the real IPs)
  reg [TW-1:0] etpipe[0:2]; reg [CH-1:0] empipe[0:2];
  always @(posedge clk) begin
    etpipe[0]<=etick[edge_raddr[1:0]]; etpipe[1]<=etpipe[0]; etpipe[2]<=etpipe[1];
    empipe[0]<=emask[edge_raddr[1:0]]; empipe[1]<=empipe[0]; empipe[2]<=empipe[1];
  end
  wire [TW-1:0] edge_tick_rdata = etpipe[1];
  wire [CH-1:0] edge_mask_rdata = empipe[1];

  // --- behavioral 2-bank scan memory (NS*TW per entry); host writes it; engine reads w/ lat 2 ---
  reg [NS*TW-1:0] scanmem [0:2*BANK_SIZE-1];
  reg [NS*TW-1:0] spipe[0:2];
  always @(posedge clk) begin spipe[0]<=scanmem[scan_raddr]; spipe[1]<=spipe[0]; spipe[2]<=spipe[1]; end
  wire [NS*TW-1:0] scan_rdata = spipe[1];

  reg [1:0] bank_ready; reg [TW-1:0] bank_chunk0, bank_chunk1;

  zlc_edge_streamer #(.CHANNEL_COUNT(CH),.SCAN_ADDR_WIDTH(SAW),.BANK_SIZE(BANK_SIZE),
                      .NUM_SLOTS(NS)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(13'd2),.repeat_forever(1'b1),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd10),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1),.repeat_from_loop_start(1'b0),
    .scan_enable(1'b1),.scan_count(NPTS[31:0]),
    .edge_raddr(edge_raddr),.edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr),.scan_rdata(scan_rdata),
    .bank_ready(bank_ready),.bank_chunk0(bank_chunk0),.bank_chunk1(bank_chunk1),
    .scan_cursor(scan_cursor_w),.underflow(underflow),
    .bus_prog_we(1'b0),.bus_prog_bus(2'd0),.bus_prog_addr(6'd0),.bus_prog_start_tick(32'd0),
    .bus_prog_stop_tick(32'd0),.bus_prog_start_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),.bus_prog_start_value(10'd0),
    .bus_prog_stop_value(10'd0),.bus_prog_mode(2'd0),.bus_prog_value_select(3'd0),
    .bus_prog_stop_value_select(3'd0),.bus_counts({BUSC*7{1'b0}}),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  // --- behavioral CYCLIC host refill model ---------------------------------------------------
  // scanmem entry for (bank,offset) holds the slot vector (= the point index it represents).
  // chunk c (data) = points [c*BANK_SIZE .. ); host loads chunk (mono mod K) into bank (mono%2)
  // one-ahead, with REFILL_LAT cycles of write latency.  base = (#wraps * K) parity.
  localparam integer REFILL_LAT=6;
  integer mono;            // monotonic chunk count the host has STARTED loading (0,1,2,..)
  integer load_at;         // cycle when the in-flight load completes
  integer load_bank, load_chunk;
  reg     load_busy;
  integer cyc;
  task load_chunk_into; input integer datachunk; input integer bank; integer j; integer gpt; begin
    for (j=0;j<BANK_SIZE;j=j+1) begin
      gpt = datachunk*BANK_SIZE + j;                 // global point index (may be >= NPTS in last chunk)
      scanmem[bank*BANK_SIZE + j] = (gpt<NPTS) ? gpt[NS*TW-1:0] : {NS*TW{1'b0}};
    end
  end endtask

  // engine monotonic position from observed cursor + wrap detection
  integer eng_mono; reg [TW-1:0] cprev; integer sweeps;
  reg [NS*TW-1:0] slot_seq [0:255]; integer nseen; integer stalls_after_warmup;

  initial begin
    reset=1; start=0; bank_ready=2'b00; bank_chunk0=0; bank_chunk1=0;
    // preload monotonic chunk 0 -> bank0, chunk1 -> bank1 (base=0)
    load_chunk_into(0,0); bank_chunk0=0;
    load_chunk_into(1%KCH,1); bank_chunk1=(1%KCH);
    bank_ready=2'b11;
    mono=2; load_busy=0; eng_mono=0; cprev=0; sweeps=0; nseen=0; stalls_after_warmup=0;
    cyc=0; load_at=0; load_bank=0; load_chunk=0;
    repeat (300) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  // host cyclic refill: track the engine's MONOTONIC chunk position directly (increment on
  // every chunk-boundary crossing INCLUDING the wrap), and keep the bank for the NEXT
  // monotonic chunk loaded with chunk ((eng_mono+1) mod K).  This mirrors the real host's
  // own monotonic next_chunk counter -- robust at the wrap (no cursor/sweeps NBA lag).
  always @(posedge clk) begin
    cyc<=cyc+1;
    if (running) begin
      // detect a chunk-boundary crossing: cursor wrapped (cursor<cprev) OR entered a new chunk
      if (scan_cursor_w !== cprev) begin
        if (scan_cursor_w < cprev || (scan_cursor_w / BANK_SIZE) != (cprev / BANK_SIZE))
          eng_mono <= eng_mono + 1;
        cprev<=scan_cursor_w;
      end
      // complete an in-flight load
      if (load_busy && cyc>=load_at) begin
        if (load_bank==0) bank_chunk0<=load_chunk[TW-1:0]; else bank_chunk1<=load_chunk[TW-1:0];
        bank_ready[load_bank]<=1'b1; load_busy<=0;
      end
      // start a one-ahead load if the next bank doesn't already hold the next chunk
      if (!load_busy) begin : refill
        integer nb; integer nc;
        nb = (eng_mono+1) % 2;
        nc = (eng_mono+1) % KCH;
        if (((nb==0)?bank_chunk0:bank_chunk1) != nc[TW-1:0]) begin
          bank_ready[nb]<=1'b0;          // de-arm during rewrite
          load_chunk_into(nc, nb);       // (write data immediately; arm after latency)
          load_bank<=nb; load_chunk<=nc; load_at<=cyc+REFILL_LAT; load_busy<=1;
        end
      end
    end
  end

  // record the slot the engine latches each new point + watch for wrap stalls
  reg [NS*TW-1:0] slot_prev; integer started=0;
  always @(posedge clk) begin
    if (running) begin
      if (!started) begin started<=1; slot_prev<=dut.slot_active; slot_seq[0]<=dut.slot_active; nseen<=1; end
      else if (dut.slot_active !== slot_prev) begin
        slot_seq[nseen]<=dut.slot_active; nseen<=nseen+1; slot_prev<=dut.slot_active;
      end
      // a stall at the wrap shows as underflow asserting; count it (after warmup)
      if (underflow && cyc>100) stalls_after_warmup<=stalls_after_warmup+1;
    end
  end

  integer i; integer bad;
  initial begin
    wait(reset==1); wait(reset==0);
    repeat (900) @(posedge clk);      // ~ several sweeps (each point ~11 ticks, 10 pts = 110)
    // verify the slot sequence is 0,1,2,...,9,0,1,2,... (correct points, seamless re-sweep)
    bad=0;
    for (i=0;i<nseen;i=i+1) if (slot_seq[i] !== (i % NPTS)) begin
      bad=bad+1;
      if (bad<=6) $display("  MISMATCH seq[%0d]=%0d expected %0d", i, slot_seq[i], i%NPTS);
    end
    $display("==== scan wrap test: K=%0d (odd) N=%0d : points_seen=%0d  seq_mismatches=%0d  wrap_stalls=%0d ====",
             KCH, NPTS, nseen, bad, stalls_after_warmup);
    $display("%s", (bad==0 && stalls_after_warmup==0 && nseen>=NPTS*2) ? "SEAMLESS-OK" : "**FAIL**");
    $finish;
  end
endmodule
