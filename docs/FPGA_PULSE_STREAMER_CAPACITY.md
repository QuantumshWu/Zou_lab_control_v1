# FPGA Pulse-Streamer Capacity Notes

This note records resource assumptions for the runtime-programmable
`zlc_pulse_streamer` backend. It is maintainer-facing; user tutorials should use
`docs/neutral_atom_hardware_manual/REAL_HARDWARE_RUNBOOK.md`.

## Identified FPGA

The historical address-switch Vivado project in `references/` targets:

```text
xc7a35tfgg484-2
```

That is an Artix-7 35T in an FGG484 package. The resource class is
approximately:

```text
logic cells: 33,280
CLB LUTs:    20,800
CLB FFs:     41,600
BRAM:        1,800 Kb = 50 x 36 Kb blocks
DSP slices:  90
```

The checked-in address-switch XDC constrains the original named outputs,
including `cooling`, `probe`, `trig`, `trap`, several shutter/control outputs,
and four 10-bit TTL-style buses:

```text
da_dipole, da_bias_y, da_bias_x, da_bias_z
```

The runtime wrapper maps those XDC ports onto a contiguous `ch00..ch61`
hardware channel order.  GUI visibility is independent from this hardware
width; upload always pads to the full channel order.

## Runtime Design

The pulse streamer stores an edge table:

```text
tick[i]: integer FPGA clock tick
mask[i]: complete output state at that tick
```

Bit 0 corresponds to `channels[0]`, bit 1 to `channels[1]`, and so on.  When
multiple channels change at the same physical time, they become one edge with a
complete output mask.

The scan-capable core contains:

```text
tick_mem[MAX_EDGES]       // base tick for each edge-template row
axis0_coeff_mem[MAX_EDGES] // fixed-point coefficient for scan_axis_names[0]
axis1_coeff_mem[MAX_EDGES] // fixed-point coefficient for scan_axis_names[1]
mask_mem[MAX_EDGES]       // CHANNEL_COUNT bits per row
scan_axis0_mem[MAX_SCAN]  // tick value for scan_axis_names[0]
scan_axis1_mem[MAX_SCAN]  // tick value for scan_axis_names[1]
scan_bus_value_mem[MAX_SCAN] // packed 4x10-bit static DA values
```

For ordinary pulses, axis coefficients are zero. For hardware scans, the GUI/API
uses named parameters and a linked table file. The host currently maps at most
two active timing parameters onto the two hardware axes, preserving the names in
`RuntimeSequenceProgram.scan_axis_names`:

```text
# vars: camera_exposure_ns(ns), trig_delay(ns)
2000 0
4000 20

scan_axis_names = ["camera_exposure_ns", "trig_delay"]
hardware rows   = [(2000 ticks, 0 ticks), (4000 ticks, 20 ticks)]
```

The FPGA computes each edge time as:

```text
effective_tick = base_tick + ((axis0_coeff*axis0_tick + axis1_coeff*axis1_tick) >> frac_bits)
```

The FPGA never receives host-side names or separate per-parameter objects. Names
are host metadata; the current bitstream receives two timing tick columns plus
one packed static DA/bus value word per scan row.  Ramp-mode DA scans still use
the non-scan bus-segment path and are not combined with scan rows.

## Current Capacity Shape

Each edge-template row stores:

```text
tick:        32 bits
mask:        CHANNEL_COUNT bits
axis0_coeff: 16 bits signed fixed point
axis1_coeff: 16 bits signed fixed point
```

So the approximate row width is:

```text
edge_row_bits = 64 + CHANNEL_COUNT
```

For the current address-switch profile, `CHANNEL_COUNT=62`, so each edge row is
about 126 bits.  Each scan row stores two timing ticks plus one packed 4x10-bit
DA word, or 104 bits.

Approximate memory scale:

| MAX_EDGES | Edge-template bits | Approx bytes | Approx LUTRAM 64x1 cells |
| --- | ---: | ---: | ---: |
| 128 | 16,128 | 1.97 KiB | 252 |
| 1024 | 129,024 | 15.75 KiB | 2,016 |
| 2048 | 258,048 | 31.5 KiB | 4,032 |
| 4096 | 516,096 | 63 KiB | 8,064 |

The HDL currently marks `tick_mem`, `mask_mem`, and scan RAM as distributed RAM
because the run path uses simple asynchronous reads.  The current core avoids
multi-read-port RAM pressure by latching first-edge, loop-start, and final-tick
metadata during upload.  During a shot it reads only the current `edge_index`
row.

If future experiments need many thousands of unique non-repeating edges, the
next architecture should move the table to a BRAM-friendly synchronous-read
pipeline and a faster upload transport such as AXI, JTAG-to-AXI, UART/SPI,
Ethernet, or a FIFO/BRAM write port.

## Configurable Resource Target

The build path should treat the resource target as a planning target, not a hard
Vivado guarantee.  The default is 70% of the LUT/FF class budget:

```powershell
$env:ZLC_PS_RESOURCE_TARGET_PCT = "70"
$env:ZLC_PS_MAX_EDGES = "512"
$env:ZLC_PS_MAX_SCAN_POINTS = "256"
```

`fpga\build_and_program.bat --check` and the Tcl scripts print the selected
profile and the estimated edge/scan capacity.  The estimate is a budget guide.
The authoritative evidence is the actual Vivado `report_utilization` and
`report_timing_summary` for the generated project.

For first-light address-switch hardware, start with:

```text
CHANNEL_COUNT = inferred from XDC, normally 62
MAX_EDGES = 512
MAX_SCAN_POINTS = 256
TICK_WIDTH = 32
CLOCK_HZ = 50_000_000
RESOURCE_TARGET = 70%
```

The default Artix-7 35T profile keeps five active named scan-parameter slots
per FPGA chunk. At most two of those slots may affect timing; static DA value
scan columns are packed into per-row bus values. Larger linked scan files can
be run as consecutive host/API chunks while each FPGA program stays within
`MAX_SCAN_POINTS`. If a sequence exceeds `MAX_EDGES`, first check whether it
can use a repeat bracket, `repeat_forever`, a named scan table with at most two
active timing parameters, or a smaller scan chunk. If it truly needs a larger
unique edge table, increase
`ZLC_PS_MAX_EDGES` and re-run `--check` before building a board bitstream.

## Timing Constants

The checked-in real-hardware default is 50 MHz because measured pulses were
twice as long when the software/server assumed 100 MHz.  At 50 MHz:

```text
tick period: 20 ns
32-bit tick counter range: about 85.9 seconds
edge quantization error: at most half a 20 ns tick
```

If the board clock is changed to a verified 100 MHz source, update both the XDC
clock constraint and `ZLC_PS_CLOCK_HZ=100000000`.

## Execution And Upload Latency

Once the FPGA sees a start transition, the relative pulse timing is on the FPGA
tick grid.  The VIO-facing `reset`, `start`, and `prog_we` controls pass through
two synchronizer stages before the runtime state machine uses them.  A zero-tick
first edge therefore appears after a small fixed pipeline offset; relative edge
spacing remains deterministic.

Vivado/VIO upload latency is outside the shot.  The persistent server starts one
Vivado Tcl process, opens the hardware target, loads the `.ltx`, and reuses that
session for `prepare/fire/wait_done/safe_state`.  After one successful prepare,
the Python backend can perform differential upload for the next compatible
program: changed edge rows, changed scan points, and shadow-critical rows are
rewritten; unchanged rows are skipped.

The camera external-trigger output should remain one output bit in the same edge
table as laser/control channels.  In the checked-in camera-imaging preset, that
hardware output is `ch11`, physical label `emCCD`, package pin `M13`.  The same
XDC also defines `ch06/trig/R17`; it remains available as a separate output but
is not the preset qCMOS trigger.

## Scan And Bus Design Notes

The scan-template design is intentionally separate from table expansion.  A
one-dimensional scan or a two-parameter grid should stay one ordered named scan
table plus one edge template whenever edge ordering is invariant:

```text
edge row: base_tick, axis0_coeff, axis1_coeff, mask
scan RAM: (axis0_point0, axis1_point0), (axis0_point1, axis1_point1), ...
scan bus RAM: packed static da_dipole/da_bias_y/da_bias_x/da_bias_z values per row
scan_axis_names: ["camera_exposure_ns", "trig_delay"]
```

The compiler rejects a scan if the same edge rows would reorder for different
points.  That rejection is a real safety boundary; split the scan into multiple
templates or prepare one pulse per point instead of uploading an ambiguous
template.

Analog bus ramps are the main case that should not consume the digital edge
table.  The current implementation folds `da_dipole`, `da_bias_x/y/z`, and
similar buses into GUI/API rows, then uploads them through a separate bus
segment memory instead of expanding every stair-step into `prog_mask` rows:

```text
bus_id
start_tick
stop_tick
start_value
stop_value
mode        // edge, ramp, hold
```

The FPGA runs one small bus engine per logical bus.  `edge` updates the bus at
`start_tick`; `hold` is represented by the absence of a new segment; `ramp` uses
a DDA accumulator and snaps to `stop_value` at `stop_tick`, so a long ramp
consumes one segment rather than hundreds of digital edge rows.  Digital output
RAM then only needs laser, shutter, camera, and other TTL transitions.
Partial-delay-heavy designs can use the same principle: move channel-local
timing metadata out of the global edge table only when it reduces repeated
full-mask edge rows.

Current limitation: hardware scan arrays can combine timing scan with static
edge-mode DA value scan, but not with analog bus ramp segments in the same
upload.  Run one prepared ramp pulse per scan point, or add a scan-aware bus
segment table if that experiment becomes common.

## Current Operational Entry

Use:

```powershell
.\fpga\build_and_program.bat
.\fpga\run_server.bat
.\pulse_gui.bat --state .\pulses\camera_imaging_address_switch.json
```

The real top-level wrapper is:

```text
fpga\pulse_streamer\zlc_pulse_streamer_top_address_switch.v
```

The default board XDC is:

```text
references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc
```

Set `ZLC_PS_XDC` only when using a verified alternative package-pin map.
