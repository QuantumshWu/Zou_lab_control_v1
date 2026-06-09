# ZLC FPGA Pulse Streamer

Vivado Tcl and HDL sources for the neutral-atom runtime pulse streamer. The
user-facing Windows entry points live one directory up (`fpga\build_and_program.bat`,
`fpga\run_server.bat`).

This is a short subsystem pointer. For the full design tutorial (the edge-table
RTL walk, the 1-tick FIFO prefetch pipeline, the 2-bank streaming scan window,
the affine N-slot scan engine, the analog-bus DAC engine, the JTAG-to-AXI host
upload flow, and resource budgets) see the **FPGA manual** in
`docs/fpga_manual/`.

## Files

- `zlc_edge_streamer.v`: the engine. A global edge table held in three parallel
  block RAMs (tick 32b / coeff 64b / mask 62b, forced `READ_LATENCY_B=2`), a
  depth-`FIFO_DEPTH` (=`RD_LAT`+1=3) continuous edge prefetch that hides the BRAM
  latency so back-to-back 1-tick (20 ns) edges fire one per clock, a 2-bank
  ping-pong scan window (`BANK_SIZE`=2048, 4096 resident points) for unbounded
  streamed scans, and the affine effective-tick MAC + analog-bus DAC engine.  A
  scanned digital DELAY whose edges reorder past other channels is pulled out of
  the global table onto its own DISJOINT output bit and played by a 1-bit affine
  sub-player ("delay lane") -- LUTRAM tables, the shared MAC, +0 RAMB36/+0 DSP.
- `zlc_pulse_streamer_top.v`: top wrapper. Region-decoded BRAMs behind an
  `axi_bram_ctrl` (edge tables + scan window + bus image + lane image) plus a CTRL
  register file (the COMMAND/STATUS mailbox + streaming `CURSOR`/`BANK_READY`/
  `BANK*_CHUNK` handshake), driving the engine and the board output pins / four
  10-bit DAC buses.  A mini-loader copies the bus + lane images into the engine
  LUTRAM at LOAD.
- `create_project.tcl`: create project (jtag_axi + axi_bram_ctrl + 6 BRAMs),
  `zlc_force_latency2` forces the edge BRAMs to `READ_LATENCY_B=2`, synth,
  implement, write bitstream + probes.
- `program_fpga.tcl`: program the device with the generated `.bit`/`.ltx`.
- `diagnose_hw_target.tcl`: non-destructive hardware-target diagnostic.
- `host/`: the host-side Python that the runtime uses --- `image.py` packs the
  compiled program into the BRAM image and reports `solve_capacity`;
  `engine_model.py` is the cycle-accurate behavioral model used by the contract
  tests (no Verilog simulator is in the repo).

## Contract Summary

Target FPGA is the Artix-7 35T `xc7a35tfgg484-2`. The default board XDC is
`fpga\board_config\board.xdc` (see `fpga/board_config/README.md`; override with
`ZLC_PS_XDC`; 62 outputs, `ch00..ch61`; `emCCD=ch11/M13`, `trig=ch06/R17`). The
bitstream is fixed; every
`On Pulse` packs a fresh program image and uploads it over JTAG-to-AXI through
`axi_bram_ctrl`, then drives the CTRL mailbox. One edge row means "at this
absolute FPGA tick, set all outputs to this mask".

Scans use named slots: each edge row stores a base tick plus `NUM_SLOTS`
fixed-point coefficients, and the FPGA computes
`effective_tick = base + (sum_j coeff_j * slot_j) >>> COEFF_FRAC_BITS` while
iterating the scan-point table. The scan window is a 2-bank ping-pong (the engine
plays the resident points and exposes `CURSOR`; the host refills the freed bank
with the next chunk under the `BANK_READY`/`BANK*_CHUNK` handshake), so the
scan-point file is finite but unbounded. Analog buses upload through a separate
LUTRAM segment table (`bus_id, start_tick, stop_tick, start_value, stop_value,
mode`, plus dual `value_select` for scanned endpoints) so a ramp costs one
segment, not hundreds of TTL edge rows.

Default profile (from `host.image.StreamerParams` / `solve_capacity` on the 35T):
`CHANNEL_COUNT=62`, `NUM_SLOTS=4`, `MAX_EDGES=4096`, `BANK_SIZE=2048` (4096
resident points), `TICK_WIDTH=32`, `COEFF_WIDTH=16`, `COEFF_FRAC_BITS=8`,
`RD_LAT=2`, `FIFO_DEPTH=3`, `CLOCK_HZ=50 MHz` (20 ns tick). Vivado
`report_utilization` is the final resource authority; the budgeted estimate is
RAMB36 78% (LUT 26%, FF 12%, DSP 9%).

## CTRL Register-File Mailbox

The host never bit-bangs probes; it reads and writes a small CTRL register file
over `axi_bram_ctrl`. The mailbox words (see `host.image.CtrlWords`):

```text
COMMAND     host -> top   rising-edge LOAD(1) / FIRE(2) / RESET(4) / SAFE(8)
STATUS      top -> host   LOADED(1) / RUNNING(2) / DONE(4) / ERROR(8) / UNDERFLOW(16)
PROG_COUNT                number of edge rows
SCAN_COUNT                TOTAL scan points N (may exceed the resident window)
SCAN_ENABLE / REPEAT_FOREVER
LOOP_START / LOOP_COUNT / LOOP_END_TICK / LOOP_END_LO / LOOP_END_HI
BUS_COUNTS                packed per-bus segment counts
BANK_SIZE / SLOT_COUNT
CURSOR      top -> host   scan points consumed so far (drives streaming refill)
BANK_READY  host -> top   bit b = bank b is loaded and ready
BANK0_CHUNK / BANK1_CHUNK host -> top   sweep-chunk index resident in each bank
```

Lifecycle: `prepare` (SAFE, upload image, arm banks, LOAD) / `fire` (FIRE) /
`wait_done` (poll STATUS; stream-refill the freed scan bank behind the cursor) /
`safe_state`. The engine only advances into a bank when `BANK_READY` AND that
bank holds the right chunk, so a late refill STALLs (hold, `STATUS_UNDERFLOW`),
never a wrong point. The cycle-accurate behavior is locked by
`host.engine_model` against the reference player + 200 fuzz programs.
