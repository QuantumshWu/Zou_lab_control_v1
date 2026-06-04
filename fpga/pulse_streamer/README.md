# ZLC FPGA Pulse Streamer

This directory contains the Vivado Tcl and HDL sources for the neutral-atom
runtime pulse streamer. The user-facing Windows entry points live one directory
up:

```powershell
.\fpga\build_and_program.bat
.\fpga\run_server.bat
```

The hardware profile is now 40-channel by default and by contract. The pulse
GUI may show only `ch00..ch03`, or any other subset, but the backend always
compiles against the server's full `ch00..ch39` hardware list. Hidden or missing
channels are uploaded as zero bits in the 40-bit `prog_mask`.

## Files

- `zlc_pulse_streamer.v`: runtime edge-table pulse-streamer core.
- `zlc_pulse_streamer_top_40ch.v`: 40-output top wrapper and VIO contract.
- `zlc_pulse_streamer_40ch.xdc`: checked-in 40-output pin map derived from the
  historical `address_switch` XDC.  The first four bits are `ch00=trap`,
  `ch01=cooling`, `ch02=probe`, and `ch03=old trig/qcm_trigger`.
- `zlc_pulse_streamer_40ch.xdc.template`: blank template for a different board.
- `create_project_40ch.tcl`: create the Vivado project, create VIO IP,
  synthesize, implement, and write the 40ch bitstream.
- `program_fpga_40ch.tcl`: program the FPGA with the generated `.bit` and
  `.ltx` probes.
- `check_40ch_synth.tcl`: no-XDC 40ch synthesis self-check. It verifies the
  40-bit VIO/top-level HDL path without producing a board-ready bitstream.
- `diagnose_hw_target.tcl`: non-destructive Vivado hardware-target diagnostic.

## Batch Entry Points

From the repository root:

```powershell
.\fpga\build_and_program.bat --help
.\fpga\run_server.bat --help
```

`fpga\build_and_program.bat` supports:

```powershell
.\fpga\build_and_program.bat              # build and program 40ch
.\fpga\build_and_program.bat --build-only # build 40ch only
.\fpga\build_and_program.bat --program-only
.\fpga\build_and_program.bat --check      # no-XDC 40ch synth self-check
.\fpga\build_and_program.bat --diagnose   # list Vivado hw targets/devices
.\fpga\run_server.bat --check-config      # print resolved server artifact paths
```

The real build uses the checked-in 40ch XDC by default:

```powershell
.\fpga\build_and_program.bat
```

For a different carrier or pin map, keep the board XDC elsewhere:

```powershell
$env:ZLC_PS_40CH_XDC = "D:\fpga_pin_maps\zlc_pulse_streamer_40ch_my_board.xdc"
.\fpga\build_and_program.bat
```

The script refuses to build if the selected XDC is missing or still contains
`<PIN_CHxx>` placeholders.  The server and standalone GUI infer the full
hardware width by reading `get_ports {ch[n]}` from the selected XDC and fall
back to 40 channels when no XDC is available.

## Vivado Discovery

The two batch files search for Vivado in this order:

1. `ZLC_PS_VIVADO_BIN`, then `ZLC_VIVADO_BIN`.
2. `C:\Xilinx\Vivado\*\bin\vivado.bat` and
   `D:\Xilinx\Vivado\*\bin\vivado.bat`.
3. `vivado.bat` or `vivado` on `PATH`.

Vivado 2019 debug cores are path-length sensitive.  The build script uses the
short project name `p40` under `fpga\build` and checks the expected debug-core
temporary path before starting the slow Vivado run. If an old environment
variable points back into `fpga\pulse_streamer\build`, the batch script ignores
it and falls back to `fpga\build`. The printed `ZLC build root: ...` and
`ZLC project dir: ...` lines are the source of truth. The default is
`fpga\build\p40`; if the repository path itself is too long, move the repo to a
shorter project folder such as `D:\ZLC`, or set `ZLC_PS_PROJECT_DIR` to a short
project-local path. If your Vivado install is elsewhere:

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
```

## Server

After the FPGA has been programmed:

```powershell
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

`run_server.bat` uses `ZLC_PY_CMD`, then `ZLC_FPGA_SERVER_PYTHON`, then the
repository-local `.zlc_python_path` written by `install_requirements.bat`,
before falling back to `python` or `py -3` on PATH.  Path-style entries are run
through `call "...\python.exe"`, so Python installs under directories such as
`C:\Program Files` work.  This keeps the FPGA server in the same Python
environment as the installed editable package.

`--check-config` does not open the persistent server.  It verifies that the
batch file can resolve the same short project directory, bitstream, `.ltx`, full
`ch00..ch39` channel list, and `ch03` trigger that the long-running server will
use.

The server defaults are:

```text
backend:  vivado-session
host:     0.0.0.0
port:     18861
clock:    100 MHz
channels: ch00 ... ch39
trigger:  ch03
max rows: 1024
```

The default persistent `vivado-session` backend opens the hardware target and
loads the `.ltx` probes once during server startup, then reuses the same Vivado
Tcl process for `prepare/fire/wait_done/safe_state`.  This moves the slow first
Vivado connection out of the first GUI `On Pulse`.

## Runtime Contract

The pulse streamer is a fixed bitstream plus a runtime table.  Building the
bitstream creates the clocked state machine, 40 output pins, and Vivado VIO
control probes.  It does not bake a particular experiment pulse into Verilog.
Every GUI/API `On Pulse` compiles the current Python `PulseTableState` or
`PulseSequence` into a small table of edges and uploads that table into FPGA
RAM through the already-built VIO probes.

One row in that table means: "at this absolute FPGA tick, change all outputs to
this mask".  It is not a list of per-channel commands.  A four-channel camera
shot on a 40-channel FPGA is still uploaded as 40-bit masks; bits `ch04..ch39`
are simply zero.  This is why the frontend can hide channels without changing
the FPGA design or the server channel count.

Python uploads an edge table:

```text
ticks: absolute FPGA clock ticks
masks: 40-bit output mask after each tick
```

During `prepare`, the backend holds reset high.  While reset is high the output
register is forced to zero, the runtime counters are idle, and the write side
of the FPGA core is enabled.  The host writes metadata (`prog_count`,
`repeat_forever`, `loop_start_addr`, `loop_end_tick`, `loop_count`) and uploads
edge rows through VIO.  For each uploaded row it stages `prog_addr`,
`prog_tick`, and `prog_mask`, then toggles `prog_we`; the FPGA writes exactly
once on that synchronized toggle.  The Python uploader now puts the first edge
row in the same VIO commit as `reset=1` and the metadata, so a prepare with
rows saves one JTAG/VIO commit compared with a separate empty metadata commit.

The VIO-facing `reset`, `start`, and `prog_we` controls pass through two
FPGA-clock synchronizer stages before the state machine uses them.  This is
intentional.  Vivado/VIO updates several probes from the PC side; the FPGA must
not see a level-style write enable while `addr/tick/mask` are halfway through
changing.  The synchronized `prog_we` transition is the only write event.  A
single row upload therefore has this shape:

```text
PC/Vivado: set prog_addr, prog_tick, prog_mask, flip prog_we, commit VIO
FPGA:      after synchronization, sees prog_we changed once
FPGA:      writes tick_mem[prog_addr] and mask_mem[prog_addr]
```

After all needed rows have been written, the backend releases reset.  `prepare`
does not start the pulse.  It only leaves the FPGA holding a validated table
and metadata in RAM/registers.  During `fire`, the backend sends an explicit
`zlc_start` low-high-low pulse; the FPGA only treats the synchronized rising
edge as a start event.  After that rising edge reaches the FPGA, timing is
owned entirely by the FPGA clock.  Python, RPyC, Windows scheduling, Vivado,
and JTAG latency no longer determine microsecond pulse edges inside that shot.

So the user-visible actions are:

```text
Pulse GUI On Pulse:
  read widgets -> PulseTableState
  compile to ticks/masks/loop metadata
  RemoteSequencer.prepare(...)
    server compiles against full ch00..ch39
    Vivado session uploads changed rows through VIO
  RemoteSequencer.fire()
    one VIO start pulse
  FPGA runs the table on its own clock

Pulse GUI Stop Pulse:
  set_safe_state / abort
    reset high, start low, repeat disabled, output forced low
```

`wait_done` only makes sense for finite programs and is an API/camera workflow,
not a Pulse GUI button.  A repeating table is supposed to keep running, so
Python refuses to wait forever unless the caller provides a timeout or asks the
API for a finite shot.

With the current 40ch top, the uploaded table has at most 1024 rows:

```text
EDGE_ADDR_WIDTH = 10
prog_count      = 11 bits
prog_mask       = 40 bits
tick width      = 32 bits
clock           = 100 MHz by default
```

The core caches the first edge, loop-start edge, and final tick while the host
uploads the table.  During a shot it reads only the current `edge_index` row
from `tick_mem/mask_mem`; this avoids the large multi-read-port memory mapping
that made earlier 40ch builds use excessive LUT/FF resources.

On Vivado 2019.1, `fpga\build_and_program.bat --check` synthesized this 1024-row
40ch profile with 0 errors and reported about 2406 LUTs (11.57%) and 1127 FFs
(2.71%) on `xc7a35tfgg484-2`.

The hardware can repeat a full table or run a finite bracketed loop without
expanding every cycle into VIO rows.  The Pulse GUI exposes the period bracket
and the preview shows the whole table as `∞` when no bracket overrides it, but
it does not expose a separate whole-table repeat switch.  For single-shot camera
tests, ask the API/controller for a finite shot; otherwise a qCMOS trigger pulse
can intentionally reappear once per table period.

If a long finite bracket is nested inside `repeat_forever`, the table head also
reappears whenever the bracketed block has completed and the full table restarts.
That can look like a periodic "extra" pulse on a scope if the first periods turn
on load/cooling/probe channels.  For a perfectly steady scope train, make the
repeating table contain only the steady camera cycle, or disable outer
`repeat_forever` and fire finite acquisitions from the camera/readout workflow.

`loop_end_tick` is the boundary where the bracketed loop restarts while
`loops_remaining > 1`; it is not the end of the whole table.  The FPGA keeps the
uploaded final row as `final_tick`, so any post-loop idle or cleanup periods run
once before `repeat_forever` starts the table again.

Latency has three separate pieces that should not be mixed together.

1. Server startup opens Vivado, connects to hw_server, opens the target, loads
   the `.ltx`, and finds the VIO probes.  This is seconds-scale and should
   happen before the experiment loop by running `fpga\run_server.bat`.
2. Runtime `prepare` writes VIO rows.  This is the slow part when `pulse.x`
   changes, because a different exposure width changes at least one tick row.
   It scales with the number of committed rows, not with the physical duration
   of the pulse.
3. Runtime `fire` is one small VIO commit for the start pulse.  Once the FPGA
   has started, internal pulse timing is clocked in hardware.

The persistent Tcl session caches matched VIO probe handles after the first
lookup, which avoids repeated `get_hw_probes` scans during large prepares.
After a successful prepare, the persistent session also performs differential upload:
if the next program has the same channel order and clock, it writes only
changed edge rows plus the shadow-critical rows `0`, `loop_start_index`, and
final.  If only loop metadata changes, for example a finite bracket moves to a
different period while the edge table rows are otherwise identical, the new
`loop_start_index` row is still rewritten after the new `loop_start_addr` probe
has been staged, so the HDL's loop-start shadow mask cannot retain an old row.

The sequencer service also caches the last uploaded `sequence_id`.  Pressing
`On Pulse` again with an unchanged pulse table reuses FPGA RAM and only sends
`fire`.  `Stop Pulse`, `safe_state`, and `abort` invalidate the service-level
prepare cache because the hardware reset/repeat probes are deliberately driven
safe, though the persistent Vivado session can still reuse unchanged RAM rows
on the next prepare.

The present VIO path is reliable for bring-up and for human-paced GUI tuning,
but it is not a true high-speed control bus.  A 100 ms-class update is plausible
only when the Vivado session is already warm, the edge table is small, and
differential prepare rewrites only a few rows.  If the experiment eventually
needs high-rate per-shot updates with many changed rows, the correct next
architecture is to keep this FPGA timing core and replace only the upload transport
with a faster path such as UART, PCIe, Ethernet, or a memory-mapped
bridge.  The frontend/API boundary would stay the same: Python still emits
`ticks/masks/metadata`; only the device adapter changes how those words reach
the FPGA.

The VIO probe contract is:

```text
probe_out0  zlc_reset           width 1
probe_out1  zlc_start           width 1
probe_out2  zlc_prog_we         width 1
probe_out3  zlc_prog_addr       width 10
probe_out4  zlc_prog_tick       width 32
probe_out5  zlc_prog_mask       width 40
probe_out6  zlc_prog_count      width 11
probe_out7  zlc_repeat_forever  width 1
probe_out8  zlc_loop_start_addr width 10
probe_out9  zlc_loop_end_tick   width 32
probe_out10 zlc_loop_count      width 32
probe_in0   zlc_running         width 1
probe_in1   zlc_done            width 1
```

Display labels such as `trap`, `cooling`, `probe`, and `qcm_trigger` are GUI
labels unless they are also actual server channel names. For the 40ch server,
the hardware names and mask bit order are `ch00..ch39`.
