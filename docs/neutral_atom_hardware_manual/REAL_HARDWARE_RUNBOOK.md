# Zou_lab_control Neutral Atom Hardware Runbook

This is the short operational hardware runbook. The default FPGA path is a
40-channel runtime pulse-streamer bitstream. The control computer sends a
`PulseSequence` or GUI `PulseTableState`; the FPGA/Vivado computer compiles it
against the full `ch00..ch39` channel list, uploads `ticks/masks` through Vivado
VIO, and the FPGA clock executes the pulses.

The GUI is only a frontend. It may display four channels, ten channels, or all
forty. At upload time the sequencer server still produces a full-width 40-bit
program; hidden or unconfigured channels are off.

## 0. Install

On both computers:

```powershell
cd C:\path\to\Zou_lab_control_v1
install_requirements.bat
```

`install_requirements.bat` records the selected interpreter in the ignored
local file `.zlc_python_path`.  Root `pulse_gui.bat`,
`start_tutorials_jupyter_lab.bat`, and `fpga\run_server.bat` use that
interpreter before falling back to PATH, so the GUI, tutorials, and FPGA server
open in the same Python environment that received PyQt, RPyC, Jupyter, and the
editable package install.  To override one terminal, set
`ZLC_PULSE_GUI_PYTHON` for the GUI, `ZLC_TUTORIALS_PYTHON` for the tutorial
launcher, or `ZLC_FPGA_SERVER_PYTHON` for the FPGA server.  To verify the
tutorial launcher without opening a browser, run:

```powershell
start_tutorials_jupyter_lab.bat --check
```

or:

```powershell
python -m pip install -e .
```

Vivado is separate from Python requirements and only needs to be installed on
the Verilog/FPGA computer.

## 1. Build And Program On The FPGA Computer

Use a short checkout path when possible, for example `D:\ZLC`.  The build
script writes generated Vivado projects under `fpga\build` and uses the short
project names `p40` and `c40`.  The printed `ZLC build root: ...` and
`ZLC project dir: ...` lines are the source of truth.  The default real
bitstream project is `fpga\build\p40`.

If an old terminal still has `ZLC_PS_PROJECT_DIR`,
`ZLC_PS_CHECK_PROJECT_DIR`, `ZLC_PS_VIVADO_BIT`, or `ZLC_PS_VIVADO_LTX`
pointing inside the repository, for example `...\fpga\pulse_streamer\build\...`,
the batch script ignores that value and falls back to `fpga\build`. This avoids
using the old long-path project layout and prevents the server from loading a
stale probes file.

```powershell
cd D:\ZLC
.\fpga\build_and_program.bat --help
.\fpga\build_and_program.bat --check
```

`--check` performs a no-XDC 40ch synthesis self-check. It is useful before the
board pin map is ready. It does not create a board-ready bitstream.

For a real build, the repo uses the checked-in
`fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc`.  It was generated from the
historical `address_switch` XDC.  The first four physical outputs are:

```text
ch00 -> trap
ch01 -> cooling
ch02 -> probe
ch03 -> old trig / qcm_trigger
```

Then build and program:

```powershell
.\fpga\build_and_program.bat
```

If your XDC lives elsewhere:

```powershell
$env:ZLC_PS_40CH_XDC = "D:\fpga_pin_maps\zlc_pulse_streamer_40ch_my_board.xdc"
.\fpga\build_and_program.bat
```

The script refuses to build if the selected XDC is missing or still contains
`<PIN_CHxx>` placeholders. It also returns nonzero if synthesis,
implementation, `.bit/.ltx` generation, or programming fails. On success it
prints a completion message and waits for the user to close the window; set
`ZLC_NO_PAUSE=1` for automation.

Generated files live under the `ZLC project dir` printed by the build script,
for example:

```text
<ZLC project dir>\p40.xpr
<ZLC project dir>\p40.runs\impl_1\zlc_pulse_streamer_top_40ch.bit
<ZLC project dir>\p40.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx
```

Vivado discovery order:

1. `ZLC_PS_VIVADO_BIN`, then `ZLC_VIVADO_BIN`;
2. `C:\Xilinx\Vivado\*\bin\vivado.bat` and
   `D:\Xilinx\Vivado\*\bin\vivado.bat`;
3. `vivado.bat` or `vivado` on `PATH`.

Manual override:

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
```

Vivado GUI route:

1. Open Vivado 2019.x.
2. If the project does not exist yet, run `.\fpga\build_and_program.bat --build-only`.
3. Open `p40.xpr` under the printed `ZLC project dir`.
4. Run Synthesis, Run Implementation, then Generate Bitstream.
5. Open Hardware Manager, Open Target, Auto Connect.
6. Select the FPGA device and Program Device.
7. Choose `zlc_pulse_streamer_top_40ch.bit` and
   `zlc_pulse_streamer_top_40ch.ltx`.

If Vivado sees a Digilent target but no FPGA device, run:

```powershell
.\fpga\build_and_program.bat --diagnose
```

Then check board power, JTAG/mode jumpers, power-source jumper, cable seating,
and Hardware Manager Auto Connect.

## 2. Start The 40ch Sequencer Server

After programming:

```powershell
cd D:\ZLC
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

`--check-config` prints the resolved project, bitstream, probes file, 40-channel
list, and trigger channel, then exits.  Use it after `--build-only` or a full
build to confirm the server will point at the same `.bit/.ltx` before starting
the long-running process.  The batch wrapper prints a completion line and keeps
the window open after this check; set `ZLC_NO_PAUSE=1` when running it from an
automation script.

Defaults:

```text
host:     0.0.0.0
port:     18861
backend:  vivado-session
clock:    100 MHz
channels: ch00 ... ch39
trigger:  ch03
```

The persistent `vivado-session` backend opens Vivado Tcl once, loads the `.ltx`
probes once, and reuses that process for `prepare/fire/wait_done/safe_state`.
Matched VIO probe handles are cached after their first lookup in that Tcl
session. This warm start happens before clients connect, so the first GUI
`On Pulse` does not pay the Vivado startup cost.

After the first successful prepare, the persistent session also keeps a Python
copy of the uploaded edge table.  A later prepare with the same channel order
and FPGA clock uses differential upload: it rewrites only edge rows whose
`tick` or `mask` changed, plus the shadow-critical rows `0`,
`loop_start_index`, and the final row.  The prepare log reports this as, for
example, `wrote 3/6 edge rows`.  Exact repeat `On Pulse` calls with the same
`sequence_id` are skipped by the sequencer service and only fire again; changed
`pulse.x` values still prepare a new program, but the Vivado session can upload
only the changed rows.
Set `ZLC_PS_SERVER_BACKEND=command` only for debugging the older
subprocess-per-action path.

## 3. Pulse GUI

Root GUI entry remains separate from FPGA scripts:

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861 --state .\pulses\camera_imaging_40ch.json
```

From a control computer, replace `127.0.0.1` with the FPGA computer IP printed
by the server.

`pulses\camera_imaging_40ch.json` stores forty hardware channels but only shows
the first four by default.  Its `camera_exposure` period uses `duration="x"`
with default `x_ns=19980000`, so `pulse.x = ...` in the control notebook scans
probe/readout exposure without rebuilding the pulse table by hand:

```text
ch00 label trap
ch01 label cooling
ch02 label probe
ch03 label qcm_trigger
```

This is still a 40ch pulse for the FPGA. The GUI visible-channel list only
controls editor clutter. When `On Pulse` is pressed, the current state is sent
to the sequencer, compiled against `ch00..ch39`, uploaded, and fired. Channels
not present in the state or not visible in the GUI are zero in the uploaded
40-bit masks.

The camera preset compiles at 100 MHz to:

```text
ticks: 0, 200000, 210000, 212000, 2210000, 2212000
masks: 3, 1, 13, 5, 1, 0
trigger_count: 1
repeat_forever: true
```

All high bits above `ch03` are zero.  The Pulse GUI does not expose a separate
whole-table repeat switch; the visible repeat control is the period bracket.
For single-shot oscilloscope or camera debugging, call
`pulse.on_pulse(wait=True, repeat_forever=False)` from a controller created with
`exp.timing.bind_pulse("pulses/camera_imaging_40ch.json")`, or let the camera
readout helper generate the finite trigger sequence. Otherwise a qCMOS trigger
pulse can intentionally reappear once per table period.

If the scope shows a periodic extra pulse every many frames or every few
seconds, first check whether the table contains a long finite repeat bracket
inside `repeat_forever=True`. In that mode the FPGA runs the bracketed span a
finite number of times, then finishes the rest of the table, then the outer
`repeat_forever` restarts from period 0. If period 0 turns on load/cooling/probe
channels, that table-head pulse is expected. For a steady scope train, build a
pulse table whose whole table is only the steady camera cycle, or run finite
shots through the camera/readout workflow.

## 4. Control/qCMOS Computer

Open:

```powershell
jupyter lab tutorials\neutral_atom_hardware_quickstart.ipynb
```

Connect to the FPGA computer:

```python
exp = na.connect(
    "remote_template",
    sequencer={"host": "192.168.0.20", "port": 18861},
    open_devices=True,
)
```

Then configure imaging, run preflight, capture raw camera frames, calibrate
sitemap/threshold, detect, and scan detection time from the notebook.

## Runtime Principle

The fixed FPGA bitstream stores an edge table:

```text
ticks: absolute FPGA clock ticks
masks: full 40-bit output masks
```

During `prepare`, Python writes the table and repeat metadata through VIO while
reset is asserted.  A full prepare writes each row by staging `prog_addr`,
`prog_tick`, and `prog_mask`, then toggling `prog_we`; the FPGA writes once on
that synchronized toggle. Differential prepares use the same row-write protocol
but only for changed rows plus shadow-critical rows. The
VIO-facing `reset`, `start`, and `prog_we` controls pass through two FPGA-clock
synchronizer stages before the state machine uses them. This avoids corrupt
table rows while Vivado updates multiple VIO probes. During `fire`, Python sends
a low-high-low `zlc_start` pulse and the FPGA reacts only to the rising edge.
After that, `time_count` advances on
the FPGA clock and updates outputs when `time_count == tick_mem[edge_index]`.

Repeat is metadata, not expanded rows. A GUI table with `repeat infinity` is
uploaded once with `zlc_repeat_forever=1`. A finite repeat bracket uses
`loop_start_addr`, `loop_end_tick`, and `loop_count`. This is why a simple GUI
pulse using only four visible channels can still run cleanly on the 40ch
bitstream without changing Verilog.
