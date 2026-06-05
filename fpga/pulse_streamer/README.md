# ZLC FPGA Pulse Streamer

This directory contains the Vivado Tcl and HDL sources for the neutral-atom
runtime pulse streamer.  The user-facing Windows entry points live one
directory up:

```powershell
.\fpga\build_and_program.bat
.\fpga\run_server.bat
```

The current hardware contract is the historical `address_switch` pin map.  The
default XDC is:

```text
references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc
```

The server and GUI infer the full hardware channel list from that XDC.  In the
checked-in map there are 62 controllable output channels, saved and uploaded as
`ch00..ch61` in FPGA bit order.  GUI display labels such as `trap`, `probe`, or
`emCCD` are labels; the hardware upload still uses `chNN` bit positions.

The original XDC has both a physical `emCCD` output and a physical `trig`
output.  In the inferred channel list, `emCCD` is `ch11` on package pin `M13`;
`trig` is `ch06` on package pin `R17`.  The checked-in camera-imaging preset
uses `ch11/emCCD/M13` as the camera trigger.  Do not change package pins for
this; select the intended channel in the pulse JSON or trigger-channel config.

## Files

- `zlc_pulse_streamer.v`: runtime edge-table pulse-streamer core.
- `zlc_pulse_streamer_top_address_switch.v`: address-switch top wrapper and
  VIO contract.  It exposes named ports from the original XDC and maps them to
  the runtime core's `out[0..61]` mask bits.
- `create_project_address_switch.tcl`: create the Vivado project, create VIO
  IP, synthesize, implement, and write the bitstream.
- `program_fpga_address_switch.tcl`: program the FPGA with the generated `.bit`
  and `.ltx` probes.
- `check_address_switch_synth.tcl`: no-output-pin synthesis self-check.  It
  verifies the VIO/top-level HDL path and resource target without producing a
  board-ready bitstream.
- `diagnose_hw_target.tcl`: non-destructive Vivado hardware-target diagnostic.

## Batch Entry Points

From the repository root:

```powershell
.\fpga\build_and_program.bat --help
.\fpga\run_server.bat --help
```

`fpga\build_and_program.bat` supports:

```powershell
.\fpga\build_and_program.bat                # build and program
.\fpga\build_and_program.bat --build-only   # build only
.\fpga\build_and_program.bat --program-only # program existing bit/LTX
.\fpga\build_and_program.bat --check        # no-XDC synth self-check
.\fpga\build_and_program.bat --diagnose     # list Vivado hw targets/devices
.\fpga\run_server.bat --check-config        # print resolved server artifact paths
```

For a different carrier or cable map, provide a board XDC with the same named
ports and set:

```powershell
$env:ZLC_PS_XDC = "D:\fpga_pin_maps\my_address_switch_board.xdc"
.\fpga\build_and_program.bat
```

The scripts refuse to build if the selected XDC is missing, still contains
placeholder pin names, or does not define the address-switch outputs expected by
the wrapper.

## Vivado Discovery And Paths

The batch files search for Vivado in this order:

1. `ZLC_PS_VIVADO_BIN`, then `ZLC_VIVADO_BIN`.
2. `C:\Xilinx\Vivado\*\bin\vivado.bat` and
   `D:\Xilinx\Vivado\*\bin\vivado.bat`.
3. `vivado.bat` or `vivado` on `PATH`.

Vivado 2019 debug cores are path-length sensitive.  The build script defaults
to the project directory `fpga\build\address_switch` and checks the expected
debug-core temporary path before starting the slow Vivado run.  If an old
environment variable points back into `fpga\pulse_streamer\build`, the batch
script ignores it and falls back to `fpga\build`.

The printed `ZLC build root: ...` and `ZLC project dir: ...` lines are the
source of truth.  The default generated artifacts are:

```text
<ZLC project dir>\address_switch.xpr
<ZLC project dir>\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.bit
<ZLC project dir>\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.ltx
```

## Server Defaults

After the FPGA has been programmed:

```powershell
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

`run_server.bat` uses `ZLC_PY_CMD`, then `ZLC_FPGA_SERVER_PYTHON`, then the
repository-local `.zlc_python_path` written by `install_requirements.bat`,
before falling back to `python` or `py -3` on PATH.

The default server profile is:

```text
backend:     vivado-session
host:        0.0.0.0
port:        18861
clock:       50 MHz, so one FPGA tick is 20 ns
channels:    inferred from ZLC_PS_XDC, normally ch00 ... ch61
trigger:     inferred from XDC label emCCD, normally ch11
edge rows:   512
scan rows:   256 ordered named-parameter rows
scan DA RAM: 4 packed 10-bit bus values per scan row
```

The default edge table therefore contains 512 rows before you raise
`ZLC_PS_MAX_EDGES` and rebuild the address-switch bitstream.

The server does not separately print a trigger line in the batch summary; the
authoritative trigger channel is part of the service snapshot and is inferred
from the same XDC as the channel list.

## Runtime Contract

The pulse streamer is a fixed bitstream plus a runtime table.  Building the
bitstream creates the clocked state machine, address-switch output pins, and
Vivado VIO control probes.  It does not bake a particular experiment pulse into
Verilog.

Every GUI/API `On Pulse` compiles the current Python `PulseTableState` or
`PulseSequence` into a compact table and uploads that table into FPGA RAM
through the already-built VIO probes.

One edge row means:

```text
at this absolute FPGA tick, change all outputs to this mask
```

It is not a list of per-channel commands.  A GUI state that only displays four
channels is still uploaded as a full-width mask.  Channels that are hidden or
absent from the GUI state are zero-padded at compile time.

During a shot the state machine reads only the current `edge_index` row plus
latched first-edge, loop-start, and final-tick metadata.  This is why the
checked-in RAM strategy can stay simple while still supporting repeat brackets
and differential uploads.

## Timing

The real-hardware default is 50 MHz.  One tick is 20 ns.  Python validates that
period durations, channel delays, timing scan columns, and scan points align to
this tick grid before upload.  A saved GUI JSON may contain a smaller historical
`time_step_ns`, but hardware compile uses `1e9 / clock_hz`; with the default
clock that is 20 ns.

The synchronized VIO start path adds a fixed small latency before the first
runtime row reaches the pins.  That latency is a constant start offset, not a
scale factor.  Pulse widths and delays are determined by FPGA tick counts.

## Scan Arrays

For scan programs, the uploaded edge table is a template.  Each row stores:

```text
base_tick
axis0 coefficient
axis1 coefficient
output mask
```

The scan payload is one ordered named table file.  The compact Artix-7 35T
profile allows at most five active named scan parameters per FPGA chunk.
Duration and delay parameters use at most two hardware timing axes:

```text
# vars: camera_exposure_ns(ns), trig_delay(ns), dipole_code
2000 0 320
4000 20 360
```

`scan_axis_names` preserves the host-side timing names, for example
`["camera_exposure_ns", "trig_delay"]`.  Analog/DA columns such as
`dipole_code` are packed into a per-row bus-value RAM and are not treated as
edge-time axes.

At runtime the FPGA computes:

```text
effective_tick = base_tick + ((axis0_coeff*axis0_tick + axis1_coeff*axis1_tick) >> frac_bits)
```

This keeps a large one-dimensional or two-dimensional timing scan from
multiplying the edge table length.  Static DA value scan also stays compact:
the FPGA loads one packed 4x10-bit DA word for each scan row and holds it for
that row.

The bitstream still has a finite scan-row RAM.  The default 35T build stores
256 scan rows in one prepared program.  Larger linked scan files can be run by
the host/API as consecutive chunks, each uploaded as a normal runtime program,
instead of increasing the FPGA LUTRAM footprint.

The compiler checks that every scan point keeps the same edge ordering.  If two
edges swap order for different timing rows, one template cannot describe the
whole scan safely.  Split the scan into multiple templates or run one prepared
pulse per point.

## Analog Bus Rows

The original address-switch XDC includes 10-bit TTL buses:

```text
da_dipole[0..9]
da_bias_x[0..9]
da_bias_y[0..9]
da_bias_z[0..9]
```

The GUI/API fold each bus into one logical channel row.  The FPGA uploads these
rows through a separate bus-segment table, not through ordinary `prog_mask`
stair-step rows.  For a 10-bit bus, valid codes are `0..1023`.

Each period supports three bus modes:

- `Edge`: jump to the edited integer code at this period start.
- `Ramp`: linearly staircase from the previous numeric anchor to this period's
  code over the period.
- `Hold`: keep the previous value, or keep following the active ramp.

The preview draws bus rows as hollow/stair-step analog traces rather than
filled digital blocks.

Current limitation: hardware scan arrays can scan static edge-mode DA/bus
values, but cannot combine with analog bus ramp segments in the same upload.
For a scanned ramp, run one prepared ramp pulse per scan row or extend the bus
segment table with scan coefficients.

The hardware contract keeps the existing digital edge-template table for TTL
events and adds a separate bus-segment table:

```text
bus_id, start_tick, stop_tick, start_value, stop_value, mode
```

Each bus engine generates edge/ramp/hold values locally with a DDA stepper.  A
ramp costs one bus segment instead of one digital edge per stair step, leaving
the digital edge limit for real TTL transitions.

## Prepare And Fire

During `prepare`, the backend holds reset high.  While reset is high the output
register is forced to zero, runtime counters are idle, and the FPGA write side
is enabled.  The host writes metadata and edge rows through VIO.  For each row
it stages `prog_addr`, `prog_tick`, x/y tick coefficients, and `prog_mask`, then
toggles `prog_we`; the FPGA writes exactly once on that synchronized toggle.

After upload, the backend releases reset.  `prepare` does not start the pulse.
It only leaves the FPGA holding a validated table and metadata.

During `fire`, the backend sends an explicit `zlc_start` low-high-low pulse.
The FPGA only treats the synchronized rising edge as a start event.  After that
event reaches the FPGA, microsecond timing is owned by the FPGA clock.  Python,
RPyC, Windows scheduling, Vivado, and JTAG latency no longer determine pulse
edges inside the shot.

## Repeat Semantics

`repeat_forever=True` means the whole uploaded table restarts forever.  A finite
period bracket inside that table is finite; after it finishes, post-loop rows
run once, then the whole table restarts if `repeat_forever` is true.  This can
look like an extra periodic pulse if the table head contains load/cooling/probe
states.

For camera acquisition or finite debugging, call:

```python
pulse.on_pulse(wait=True, repeat_forever=False)
```

For a steady oscilloscope train, make the whole table be the steady train and
call:

```python
pulse.on_pulse(wait=False, repeat_forever=True)
```

## Capacity

The default profile uses:

```text
CHANNEL_COUNT      = 62
MAX_EDGES          = 512
MAX_SCAN_POINTS    = 256
EDGE_ADDR_WIDTH    = 9
SCAN_ADDR_WIDTH    = 8
TICK_WIDTH         = 32
CLOCK_HZ           = 50 MHz
RESOURCE_TARGET    = 70%
```

`fpga\build_and_program.bat` prints a Python-side capacity estimate before
starting Vivado.  The target is configurable:

```powershell
$env:ZLC_PS_RESOURCE_TARGET_PCT = "70"
$env:ZLC_PS_MAX_SCAN_POINTS = "256"
```

The final authority is Vivado `report_utilization` from the real build or from
`fpga\build_and_program.bat --check`.

## VIO Probe Contract

```text
probe_out0  zlc_reset              width 1
probe_out1  zlc_start              width 1
probe_out2  zlc_prog_we            width 1
probe_out3  zlc_prog_addr          width 9
probe_out4  zlc_prog_tick          width 32
probe_out5  zlc_prog_mask          width 62
probe_out6  zlc_prog_count         width 10
probe_out7  zlc_repeat_forever     width 1
probe_out8  zlc_loop_start_addr    width 9
probe_out9  zlc_loop_end_tick      width 32
probe_out10 zlc_loop_count         width 32
probe_out11 zlc_prog_tick_x_coeff  width 16
probe_out12 zlc_prog_tick_y_coeff  width 16
probe_out13 zlc_scan_enable        width 1
probe_out14 zlc_scan_prog_we       width 1
probe_out15 zlc_scan_prog_addr     width 8
probe_out16 zlc_scan_prog_x        width 32
probe_out17 zlc_scan_prog_y        width 32
probe_out18 zlc_scan_count         width 9
probe_out19 zlc_loop_end_x_coeff   width 16
probe_out20 zlc_loop_end_y_coeff   width 16
probe_out21 zlc_bus_prog_we        width 1
probe_out22 zlc_bus_prog_bus       width 2
probe_out23 zlc_bus_prog_addr      width 6
probe_out24 zlc_bus_prog_start_tick width 32
probe_out25 zlc_bus_prog_stop_tick width 32
probe_out26 zlc_bus_prog_start_value width 10
probe_out27 zlc_bus_prog_stop_value width 10
probe_out28 zlc_bus_prog_mode      width 2
probe_out29 zlc_bus_counts         width 28
probe_out30 zlc_scan_prog_bus_values width 40
probe_in0   zlc_running            width 1
probe_in1   zlc_done               width 1
```
