# Zou_lab_control Neutral Atom Hardware Runbook

This is the short operational runbook for the real two-computer hardware path.
The default FPGA path is the address-switch pulse-streamer bitstream.  The
control computer sends a `PulseSequence` or GUI `PulseTableState`; the
FPGA/Vivado computer compiles it against the full XDC-inferred hardware channel
order, uploads `ticks/masks` through Vivado VIO, and the FPGA clock executes the
pulse timing.

The GUI is only a frontend.  It may display four channels, ten channels, or every
XDC channel.  At upload time the sequencer server still produces a full-width
program; hidden or unconfigured channels are off.

## 0. Install

On both computers:

```powershell
cd C:\path\to\Zou_lab_control_v1
install_requirements.bat
```

`install_requirements.bat` installs the Python package in editable mode and
records the selected interpreter in `.zlc_python_path`.  Root `pulse_gui.bat`,
`start_tutorials_jupyter_lab.bat`, and `fpga\run_server.bat` use that
interpreter before falling back to `PATH`.

Vivado is separate from Python requirements and only needs to be installed on
the FPGA/Vivado computer.

## 1. Build And Program On The FPGA Computer

Use a short checkout path when possible, for example `D:\ZLC`.

```powershell
cd D:\ZLC
.\fpga\build_and_program.bat --help
.\fpga\build_and_program.bat --check
.\fpga\build_and_program.bat
```

The script writes Vivado projects under `fpga\build`.  The default real project
is:

```text
fpga\build\address_switch\address_switch.xpr
fpga\build\address_switch\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.bit
fpga\build\address_switch\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.ltx
```

The default XDC source is the historical address-switch pin map:

```text
references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc
```

To use a different board-level XDC, set `ZLC_PS_XDC`:

```powershell
$env:ZLC_PS_XDC = "D:\fpga_pin_maps\my_address_switch_board.xdc"
.\fpga\build_and_program.bat
```

The address-switch wrapper uses the original port names and maps them to a
runtime bit vector.  For the camera-imaging preset, the important physical
outputs are:

```text
ch09 -> trap    -> package pin M17
ch00 -> cooling -> package pin F15
ch03 -> probe   -> package pin N15
ch11 -> emCCD   -> package pin M13
```

The same XDC also has `ch06 -> trig -> package pin R17`.  That output remains
available, but the checked-in camera/qCMOS preset uses `emCCD` (`ch11`) as the
camera trigger.  Do not change package-pin mapping to make this work; the pulse
JSON and trigger-channel inference select the existing `emCCD` channel.

The hardware clock is 50 MHz, so the timing quantum is 20 ns.  If a GUI field
shows `Step: 20 ns`, the displayed duration should match the scope duration.

The original XDC also contains 10-bit TTL-style buses that the GUI/API may treat
as logical analog-value channels:

```text
da_dipole: ch18..ch27        bit order da_dipole[0]..da_dipole[9]
da_bias_y: ch38..ch29        bit order da_bias_y[0]..da_bias_y[9]
da_bias_x: ch40..ch49        bit order da_bias_x[0]..da_bias_x[9]
da_bias_z: ch60..ch51        bit order da_bias_z[0]..da_bias_z[9]
```

`--check` performs a no-board self-check for the HDL/VIO width and capacity
contract.  A full build and program returns nonzero if synthesis,
implementation, bitstream/LTX generation, or programming fails.

The capacity/resource profile is configurable before build:

```powershell
$env:ZLC_PS_RESOURCE_TARGET_PCT = "70"
$env:ZLC_PS_MAX_EDGES = "512"
$env:ZLC_PS_MAX_SCAN_POINTS = "256"
.\fpga\build_and_program.bat --check
```

The default resource target is 70%.  The script prints the selected edge-table
and scan-array capacity; Vivado `report_utilization` remains the final resource
evidence for the generated project.

Vivado discovery order:

1. `ZLC_PS_VIVADO_BIN`, then `ZLC_VIVADO_BIN`;
2. `C:\Xilinx\Vivado\*\bin\vivado.bat` and
   `D:\Xilinx\Vivado\*\bin\vivado.bat`;
3. `vivado.bat` or `vivado` on `PATH`.

Manual Vivado route:

1. Open Vivado 2019.x.
2. If the project does not exist yet, run `.\fpga\build_and_program.bat --build-only`.
3. Open `fpga\build\address_switch\address_switch.xpr`.
4. Run Synthesis, Run Implementation, then Generate Bitstream.
5. Open Hardware Manager, Open Target, Auto Connect.
6. Select the FPGA device and Program Device.
7. Choose `zlc_pulse_streamer_top_address_switch.bit` and
   `zlc_pulse_streamer_top_address_switch.ltx` from the same implementation
   directory.

If Vivado sees a Digilent target but no FPGA device, run:

```powershell
.\fpga\build_and_program.bat --diagnose
```

Then check board power, JTAG/mode jumpers, power-source jumper, cable seating,
and Hardware Manager Auto Connect.

## 2. Start The Sequencer Server

After programming:

```powershell
cd D:\ZLC
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

`--check-config` prints the resolved project, bitstream, probes file, full
channel list, trigger channel, clock, edge capacity, and scan capacity, then
exits.

Default runtime settings:

```text
host:            0.0.0.0
port:            18861
backend:         vivado-session
clock:           50 MHz
time step:       20 ns
channels:        inferred from the selected XDC, fallback ch00 ... ch61
trigger channel: inferred from the XDC label "emCCD", normally ch11
edge rows:       configured by build/runtime capacity settings
scan RAM:        ordered named scan rows: up to two timing-axis ticks plus packed DA bus values
```

The persistent `vivado-session` backend opens Vivado Tcl once, loads the `.ltx`
probes once, and reuses that process for `prepare/fire/wait_done/safe_state`.
After the first successful prepare, later prepares with the same channel order
and FPGA clock can use differential upload: only changed edge rows, changed scan
points, and shadow-critical rows are rewritten.

## 3. Pulse GUI

Open the standalone GUI from the repository root:

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861 --state .\pulses\camera_imaging_address_switch.json
```

From a control computer, replace `127.0.0.1` with the FPGA computer IP.  For
offline editing:

```powershell
.\pulse_gui.bat --no-sequencer --state .\pulses\camera_imaging_address_switch.json
```

`pulses\camera_imaging_address_switch.json` stores the full hardware channel
order and only shows the camera-imaging subset by default:

```text
visible: ch09 trap, ch00 cooling, ch03 probe, ch11 emCCD
```

The `camera_exposure` period uses `duration="camera_exposure_ns"` with default
`camera_exposure_ns=19_980_000`, so
`pulse.set_variable("camera_exposure_ns", value_ns)` in a control notebook
scans probe/readout exposure without rebuilding the pulse table.

When `On Pulse` is pressed, the GUI sends the current state to the sequencer.
The sequencer compiles it against the full XDC-inferred channel order, uploads
the edge table, and fires the FPGA.  Channels not present in the state or hidden
in the GUI are zero in the uploaded masks.

The camera preset compiles at 50 MHz to:

```text
ticks: 0, 100000, 105000, 106000, 1105000, 1106000
masks: 513, 512, 2568, 520, 512, 0
trigger_count: 1
repeat_forever: true
```

These masks mean:

```text
513 = ch00 + ch09
512 = ch09
2568 = ch03 + ch09 + ch11
520 = ch03 + ch09
```

In the standalone Pulse GUI, the left raw hardware column displays XDC package
pins when available.  For this preset, expect to see `M17`, `F15`, `N15`, and
`M13` in the raw column, with the hardware bit name such as `ch11` kept in the
tooltip and JSON/API state.

For single-shot scope or camera debugging, run a finite shot from the API, for
example:

```python
pulse = exp.timing.bind_pulse("pulses/camera_imaging_address_switch.json")
pulse.on_pulse(wait=True, repeat_forever=False)
```

For normal camera acquisition, let the camera/readout helper generate the finite
trigger sequence so the camera is armed before the FPGA fires.

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
masks: full-width output masks
```

During `prepare`, Python writes the table and repeat metadata through VIO while
reset is asserted. A full prepare writes each row by staging `prog_addr`,
`prog_tick`, optional two-axis timing coefficients, and `prog_mask`, then
toggling `prog_we`; the FPGA writes once on that synchronized toggle. Scan
programs upload one ordered named-parameter table after the host packs at most
two active timing parameters into the two hardware scan columns. The current
bitstream probe names for those columns are still `scan_prog_x` and
`scan_prog_y`; treat those as internal axis0/axis1 wires, not user-facing x/y
variables.

Repeat is metadata, not expanded rows.  A GUI table with repeat infinity is
uploaded once with `zlc_repeat_forever=1`.  A finite repeat bracket uses
`loop_start_addr`, `loop_end_tick`, and `loop_count`.

Scan is also metadata plus a compact template, not an expanded table. In the
Pulse GUI, press the dot beside a duration or delay field to bind it to a named
scan parameter, then link a table file:

```text
# vars: camera_exposure_ns(ns), trig_delay(ns)
1000 0
2000 20
4000 40
```

All values are ns and the GUI snaps numeric entries to the active hardware step
using the same Fluent line-edit resolution logic as duration and delay fields.
Rows whose delay or duration contains named parameters or expressions such as
`100000 - camera_exposure_ns` keep those symbolic expressions in the editor and
preview. The GUI does not expand the scan table into hundreds of period
columns; the FPGA iterates the packed scan rows internally.

Analog bus rows are folded views of 10-bit TTL groups.  Current hardware
uploads them through a separate bus-segment table with rows such as `bus_id,
start_tick, stop_tick, start_value, stop_value, mode`.  A small FPGA bus engine
generates the stair-step ramp locally, while the digital edge table remains
reserved for laser, shutter, camera, and other TTL transitions.  Hardware scan
arrays do not currently combine with analog bus segments in the same upload;
prepare one bus-ramp pulse per scan point when that case is needed.
