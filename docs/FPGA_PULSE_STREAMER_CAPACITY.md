# FPGA Pulse-Streamer Capacity Notes

This note records the resource assumptions for the runtime-programmable
`zlc_pulse_streamer` backend. It is intended for maintainers and future
engineering agents, not as a user tutorial.

## Identified FPGA

The board label `ATK91131A` is not sufficient to estimate FPGA resources. Use
the Vivado part name from the active hardware project instead.

The historical `address_switch` Vivado project in `references/` targets:

```text
xc7a35tfgg484-2
```

That is an Artix-7 35T in an FGG484 package. The Artix-7 35T resource class is
approximately:

```text
logic cells: 33,280
CLB LUTs:    20,800
CLB FFs:     41,600
BRAM:        1,800 Kb = 50 x 36 Kb blocks
DSP slices:  90
max user I/O in this class: 250
```

The existing XDC in the reference project already constrains 44 physical pins
with `LVCMOS33`, including the old laser/control outputs and several 10-bit
DAC-style buses.

## Runtime Design

The pulse streamer does not store one row per named pulse. It compiles the
entire `PulseSequence` into an edge table:

```text
tick[i]: integer FPGA clock tick
mask[i]: complete output state at that tick
```

For `CHANNEL_COUNT = 40`, each mask is a 40-bit word. Bit 0 corresponds to
`channels[0]`, bit 1 to `channels[1]`, and so on. When multiple channels change
at the same physical time, they become one edge with a complete 40-bit state.

The generated core contains:

```text
tick_mem[MAX_EDGES]   // TICK_WIDTH bits per entry
mask_mem[MAX_EDGES]   // CHANNEL_COUNT bits per entry
state_mask            // current output state
time_count            // free-running counter during a shot
edge_index            // next edge to consume
active_count          // number of uploaded edges
```

During `prepare`, the host writes the table into `tick_mem` and `mask_mem`.
During `fire`, the host toggles `start`. The FPGA then runs by comparing
`time_count` with `tick_mem[edge_index]`; when they match, it copies
`mask_mem[edge_index]` into the output register and advances to the next edge.

The shot-level timing is therefore clocked by the FPGA. Network, RPyC, Python,
Vivado, and JTAG are only used before the shot starts, or to send the start
command.

## 40-Channel Feasibility

A 40-channel pulse-streamer design is realistic for this FPGA class. The output
state itself is only a 40-bit register plus 40 output pins. The edge table RAM is
the meaningful resource.

For the current generated design, each edge stores:

```text
tick: 32 bits
mask: CHANNEL_COUNT bits
```

For 40 channels, each edge is approximately 72 bits.

| MAX_EDGES | Edge table bits | Approx bytes | Approx 64x1 LUTRAM cells |
| --- | ---: | ---: | ---: |
| 128 | 9,216 | 1.125 KiB | 144 |
| 1024 | 73,728 | 9 KiB | 1,152 |
| 4096 | 294,912 | 36 KiB | 4,608 |
| 16384 | 1,179,648 | 144 KiB | 18,432 |

The first-light HDL explicitly marks `tick_mem` and `mask_mem` as distributed
RAM because the core uses simple asynchronous table reads. This keeps the
control logic straightforward, but the approximate LUTRAM column is only a
lower-bound bit-packing estimate; Vivado utilization also includes mapping
overhead, VIO/debug logic, routing, and normal control registers.

The checked-in 4-channel first-light profile keeps 1024 edges. The checked-in
40-channel full-channel profile uses 128 edges by default, because local Vivado
2019.1 synthesis showed that the 40-channel/1024-edge distributed-memory version
over-uses the xc7a35t logic budget. This is still enough for the camera-imaging
preset and many compact pulse tables because API/GUI repeats are uploaded as
loop metadata instead of expanded rows. If the experiment needs hundreds or
thousands of unique non-repeating edges on all 40 channels, move the edge table
to a BRAM-friendly synchronous-read pipeline and replace Vivado/VIO upload with
AXI, JTAG-to-AXI, UART/SPI, Ethernet, or a FIFO/BRAM write port.

At 100 MHz, a 32-bit tick counter covers about 42.9 seconds. One tick is 10 ns.

## Latency Estimates

There are three different latencies to keep separate.

### Sequence Quantization

`PulseSequence.edges()` converts seconds to ticks with:

```text
tick = round(time_seconds * clock_hz)
```

At 100 MHz, the tick period is 10 ns, so the quantization error is at most about
5 ns per edge. The generated program validates that ticks are strictly
increasing and fit in `TICK_WIDTH`.

### FPGA Execution Latency

The core samples `start` through a small registered transition detector. During
`reset=1`, `prog_we` is a level write-enable for the upload RAM: each VIO commit
that changes `prog_addr/prog_tick/prog_mask` while `prog_we=1` rewrites the
selected table entry. Once a start transition is visible at the module input,
the core needs roughly:

```text
1 clock: capture start into start_sync
1 clock: detect start_event and enter running state
1 clock: apply the first mask if tick_mem[0] == 0
```

At 100 MHz, this is about 30 ns from the start input becoming visible to the
first zero-tick output update. This is a fixed pipeline offset. Relative timing
between pulse edges remains on the FPGA tick grid.

If the first edge is at tick `T`, the output update appears after the same fixed
start pipeline plus `T` FPGA clocks. The fixed offset can be calibrated or simply
absorbed into the definition of the qCMOS trigger channel.

### Host/Vivado Upload Latency

The current backend uploads through Vivado VIO. This is intentionally simple for
first deployment, but it is not a high-throughput transport. The default server
uses a persistent Vivado Tcl session, so Vivado startup, hardware-target open,
`.ltx` loading, and probe lookup are paid once when
`fpga/run_server.bat` starts.

For each edge, the generated Tcl currently stages:

```text
set addr
set tick
set mask
prog_we = 1
```

The generated Tcl batches `addr/tick/mask/prog_we=1` into one VIO commit per
edge, then commits `prog_we=0` and releases reset at the end. The prepare path
therefore costs roughly `edge_count + constant` VIO commits. This is much better
than one commit per probe or a high/low write-enable pair for every edge, and it
is faster than launching a fresh Vivado batch process for every action. The
remaining latency is still dominated by VIO/JTAG commits. It happens before the
qCMOS is armed, so it does not add shot-internal jitter, but it can make large
scans slow.

The `fire` path toggles `zlc_start` in one VIO commit after a program has been
prepared. The absolute time from the Python call to the FPGA start transition is
host/Vivado/JTAG dependent, but after that transition reaches the FPGA, the
pulse timing is deterministic.

The `wait_done` path polls `zlc_done`; the default generated Tcl poll interval is
20 ms. This affects when the host learns that a shot has completed, not the
timing of the pulse sequence itself.

## Main Constraints

The limiting factors are expected to be board-level and workflow-level rather
than LUT count:

- 40 channels require 40 routed, usable output pins with compatible voltage
  banks and connectors.
- Lab-facing TTL/BNC outputs should normally use output buffers, level shifters,
  or line drivers. Do not treat FPGA pins as rugged instrument outputs.
- Vivado/VIO is convenient for first deployment because it avoids designing a
  new transport, but it is not a high-throughput upload path. Large edge tables
  or high-rate scans should eventually move to AXI, UART, SPI, Ethernet, or a
  dedicated FIFO/BRAM write interface.
- The qCMOS trigger should be one named channel in the same edge table as the
  laser/control channels, so camera timing and pulse timing share one clocked
  source of truth.

## Practical Starting Point

Start synthesis with:

```text
CHANNEL_COUNT = 40
MAX_EDGES = 128
TICK_WIDTH = 32
CLOCK_HZ = 100_000_000
```

Then inspect Vivado utilization and timing. If the experiment sequences exceed
128 unique uploaded edges, first check whether the sequence can use a repeat
bracket or `repeat_forever`. If it truly needs a larger unique table, redesign
the memory path around BRAM before attempting larger 40-channel bitstreams on an
xc7a35t.

The 40-channel bitstream needs a new top-level wrapper and XDC pin map. The
Python backend already supports `ZLC_PS_CHANNEL_COUNT` and generated HDL with
arbitrary channel names, but the Vivado VIO probe widths and physical output
constraints must match the generated bundle exactly.

The repository now uses the 40-channel FPGA path as the operational entry
point. Use `fpga/build_and_program.bat` to check, build, and program the 40ch
bitstream, then use `fpga/run_server.bat` to start a server whose hardware
channels are always `ch00 ... ch39`. A GUI or API pulse may configure only a
subset of those channels; the compiler fills every missing channel with an off
bit before uploading the 40-bit masks. The real build uses
`zlc_pulse_streamer_top_40ch.v` after completing
`zlc_pulse_streamer_40ch.xdc` from the template, or after setting
`ZLC_PS_40CH_XDC` to a verified package-pin map.

## External References

- AMD Artix 7 product family page:
  <https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/artix-7.html>
- AMD/Xilinx 7 Series FPGA overview datasheet DS180:
  <https://docs.amd.com/v/u/en-US/ds180_7Series_Overview>
