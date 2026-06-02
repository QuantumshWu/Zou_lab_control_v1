# Zou_lab_control Neutral Atom Hardware Runbook

This is the short operational version of the hardware manual. The preferred
FPGA path is now a fixed runtime pulse-streamer bitstream: the control computer
sends `PulseSequence`, the FPGA computer uploads the compiled `ticks/masks`
edge table, and the FPGA clock executes the pulses.

## 0. Install on both computers

```powershell
cd C:\path\to\Zou_lab_control_v1
python -m pip install -e .
```

## 1. Build and program the FPGA pulse-streamer

Run this on the Verilog/FPGA computer:

```powershell
cd C:\path\to\Zou_lab_control_v1
.\fpga\pulse_streamer\build_4ch_bitstream.bat
.\fpga\pulse_streamer\program_4ch_fpga.bat
```

This creates and programs the first-light `trap/cooling/probe/qcm_trigger`
bitstream from `fpga\pulse_streamer\zlc_pulse_streamer.v`,
`zlc_pulse_streamer_top_4ch.v`, and `zlc_pulse_streamer_4ch.xdc`. The generated
Vivado project and bit/probe files are normally:

```text
fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.xpr
fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.bit
fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.ltx
```

The build/program batch files return nonzero if Vivado fails, if synth/impl did
not complete, or if the `.bit/.ltx` outputs are missing.

Vivado GUI route for the same operation:

1. Open **Xilinx Vivado 2019.2** from the Windows Start Menu, or run
   `C:\Xilinx\Vivado\2019.2\bin\vivado.bat`.
2. If the project does not exist yet, run `build_4ch_bitstream.bat` first.
   Otherwise choose **File -> Project -> Open** and open
   `fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.xpr`.
3. In **Flow Navigator**, run **Run Synthesis**, then **Run Implementation**,
   then **Generate Bitstream**. Check the Tcl Console/Messages panel for errors.
4. To program the board, choose **Flow Navigator -> Program and Debug -> Open
   Hardware Manager**.
5. Click **Open Target -> Auto Connect**.
6. Select the FPGA device in the Hardware window and click **Program Device**.
7. Set **Bitstream file** to
   `...\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.bit`.
8. Set **Probes file** to
   `...\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.ltx`.
9. Click **Program**.

Expected time: first 4-channel build is usually 8-20 minutes, sometimes up to
about 30 minutes on a slow machine or first IP-cache generation. Incremental
builds are usually 3-10 minutes. Programming through JTAG is usually
30 seconds to 2 minutes; reserve 3-5 minutes if Hardware Manager starts slowly.

The Vivado VIO probe contract is:

```text
probe_out0 zlc_reset      width 1
probe_out1 zlc_start      width 1
probe_out2 zlc_prog_we    width 1
probe_out3 zlc_prog_addr  width 10
probe_out4 zlc_prog_tick  width 32
probe_out5 zlc_prog_mask  width 4
probe_out6 zlc_prog_count width 11
probe_in0  zlc_running    width 1
probe_in1  zlc_done       width 1
```

The backend can find either the semantic names (`zlc_reset`, `zlc_prog_tick`,
...) or Vivado's native port names (`probe_out0`, `probe_out4`, ...).

Before connecting the qCMOS, run the FPGA-only smoke test and check the four
outputs on an oscilloscope:

```powershell
.\fpga\pulse_streamer\smoke_test_4ch_upload.bat
```

The batch file defaults the Vivado project, bitstream, and `.ltx` probe paths to
the `build\zlc_pulse_streamer_4ch` outputs above.

Expected timing at 100 MHz: `trap` high 0-10 us, `cooling` high 0-3 us,
`probe` high 2-6 us, and `qcm_trigger` high 2-3 us.

For 40-channel hardware, copy
`fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc.template` to
`fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc`, fill every `ch00` through
`ch39` package pin, confirm bank voltage/electrical level, then run
`build_40ch_bitstream.bat` and `program_40ch_fpga.bat`. The 40-channel build
will stop if `<PIN_CHxx>` placeholders remain. Expect roughly 15-35 minutes for
the first 40-channel build and 1-3 minutes for programming. Do not connect the
qCMOS or lasers until the pin map has been checked with a scope.

## 2. Start the FPGA server

Preferred Jupyter route:

```powershell
cd C:\path\to\Zou_lab_control_v1
jupyter lab tutorials\neutral_atom_fpga_server.ipynb
```

Edit the Vivado paths in the notebook, then run the final server cell. The
cell blocks while the server is running. Keep the kernel alive.

Equivalent PowerShell route:

```powershell
cd C:\path\to\Zou_lab_control_v1
$env:PYTHONPATH = (Get-Location).Path

$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
$env:ZLC_PS_VIVADO_PROJECT = "$PWD\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.xpr"
$env:ZLC_PS_VIVADO_BIT = "$PWD\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.bit"
$env:ZLC_PS_VIVADO_LTX = "$PWD\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.ltx"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "0"
$env:ZLC_PS_VIO_FILTER = 'CELL_NAME=~"*vio*"'
$env:ZLC_PS_MAX_EDGES = "1024"
$env:ZLC_PS_TICK_WIDTH = "32"
$env:ZLC_PS_CHANNEL_COUNT = "4"

Test-Path $env:ZLC_PS_VIVADO_BIN
Test-Path $env:ZLC_PS_VIVADO_PROJECT
Test-Path $env:ZLC_PS_VIVADO_BIT
Test-Path $env:ZLC_PS_VIVADO_LTX

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --host 0.0.0.0 `
  --port 18861 `
  --channels trap cooling probe qcm_trigger `
  --trigger-channels qcm_trigger `
  --clock-hz 100000000 `
  --state-dir D:\zlc_sequencer_state `
  --prepare-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer prepare" `
  --fire-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer fire" `
  --wait-done-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer wait_done" `
  --safe-state-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer safe_state"
```

Use `ZLC_PS_VIVADO_PROGRAM_ON_RUN="1"` for the first run so Vivado programs the
bitstream and probes. After the board is programmed, set it to `"0"` for
faster table uploads.

## 3. Control/qCMOS computer

Open the real hardware notebook:

```powershell
cd C:\path\to\Zou_lab_control_v1
jupyter lab tutorials\neutral_atom_hardware_quickstart.ipynb
```

Use the IP printed by the FPGA server:

```python
exp = na.connect(
    "remote_template",
    sequencer={"host": "192.168.1.123", "port": 18861},
    open_devices=True,
)
```

Then run the cells in order:

```python
exp.timing.configure_imaging(...)
preflight = exp.timing.preflight()
preflight.raise_if_failed()

capture = exp.camera.capture(frames=1, display=True)
grid_shape = (5, 7)
sitemap = exp.readout.sitemap(frames=20, grid_shape=grid_shape, roi_radius=1, display=True)
threshold = exp.readout.thresholds(frames=120, site=0, display=True)
shot = exp.readout.detect(display=True)
clock_hz = exp.devices.sequencer.clock_hz
time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(8e-3 * clock_hz)), 40, dtype=int)
times = time_ticks / clock_hz
scan = exp.readout.detection_time(times, shots=30, live=True, display=True)
```

## 4. Optional pulse GUI

On a desktop Python/Qt environment, the pulse GUI is a frontend for the same
sequencer:

```python
pulse_gui = zf.show_pulse_gui(experiment=exp)
```

The standalone root launcher does not require an experiment config:

```powershell
.\pulse_gui.bat
```

To connect that launcher directly to an already running FPGA sequencer server:

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861
```

On the control computer, replace `127.0.0.1` with the FPGA computer IP printed
by the server. The launcher defaults to 40 hardware channels named
`ch00...ch39`, a 100 MHz clock, `ch03` as trigger channel, and only the first
four channels visible. To open the normal camera imaging preset:

```powershell
.\pulse_gui.bat --state .\pulses\camera_imaging_40ch.json
```

The preset keeps hardware names as `ch00...ch39`; display labels are explicitly
stored only for `ch00=trap`, `ch01=cooling`, `ch02=probe`, and
`ch03=qcm_trigger`. All delays are 0 ns. At 100 MHz it compiles to 6 edges:

```text
ticks: 0, 200000, 210000, 212000, 2210000, 2212000
masks: 3, 1, 13, 5, 1, 0
trigger_count: 1
```

This is well within the first runtime design limits of 1024 edges, 32-bit
ticks, and 40 mask bits.

When a sequencer is attached, the GUI `step (ns)` defaults to
`1e9 / sequencer.clock_hz`; at 100 MHz this is 10 ns. Durations, delays, and
`x` scan values must be integer multiples of that step. `Prepare` validates the
same FPGA clock grid before uploading the runtime edge table.

For 40-channel hardware, keep the server `--channels` order, the control config,
and the GUI state in the same order. The channel list is the hardware channel
name list and FPGA bit order, for example `ch00`, `ch01`, and so on. The GUI
does not infer device meanings for these names. The Name panel shows the
hardware channel on the left and an optional display label on the right; when a
display label is edited, the Delay row and period checkbox text follow that
label, and the Preview y-axis uses the label as well, while saved/compiled
pulses still use the hardware channel name. The GUI
can hide unused channels and add them back without changing FPGA mask bit
positions. When all channels are shown, the channel-name column, delay column,
and period checkbox columns share one vertical scroll position so each channel
row remains aligned across the editor. The period timeline has its own
horizontal scrollbar. `X` clears all period states for a channel without hiding
it. `Hide Off` hides channels that are off in every period and keeps at least
four channels visible. It ignores delay values when deciding whether a channel
can be hidden, and stored delay and display-name values are kept in the saved
state. Adding a hidden channel back restores its original hardware-order
position.
The Preview tab calls `zf.plot(..., kind="pulse")`; it hides always-off channels
by default, can show them with the toggle, uses display labels for the y-axis,
and marks repeat mode as
`repeat ∞`, `repeat Pm-Pn xN`, or `repeat ∞ + Pm-Pn xN`. A bracket covering all
periods is a finite outer repeat. An internal bracket is drawn as a colored
nested bracket while the full sequence remains `∞`. The preview plots the
unexpanded period table; hardware prepare still uses the expanded repeat
sequence. `Save Pulse` stores JSON in the project `pulses/` directory by
default, or in `ZLC_PULSE_DIR` when that environment variable is set. New pulse
names default to `pulse_YYYYMMDD_HHMMSS`. `Save Figure` is the one-line button
at the right side of the Preview top bar and stores the preview PNG. If the
window is too large for the current display, pass
`scale=0.82` or `window_ratio=0.90`; these change only frontend geometry, not
pulse timing.

On the FPGA computer, open the GUI from a second Python process after the server
is running:

```python
sequencer = na.RemoteSequencer(
    host="127.0.0.1",
    port=18861,
    channels=["trap", "cooling", "probe", "qcm_trigger"],
    clock_hz=100_000_000,
    trigger_channels=["qcm_trigger"],
)
pulse_gui = zf.show_pulse_gui(channels=sequencer.channels, sequencer=sequencer, scale=0.82, window_ratio=0.90)
```

## 5. Loader and extension rule

Keep `load_devices` configs simple: each device has `type` and `params`; use
`"$device:name"` for dependencies. New hardware classes should inherit the
right base class and either use a full import path in JSON or be registered:

```python
na.register_device_class("MySequencer", MySequencer)
na.device_class_registry()
```

Do not put FPGA transport details into the control notebook. The only control
PC contract is `RemoteSequencer.prepare/fire/wait_done`.

## 6. First-light fallback

For camera-only trigger checks:

```python
exp = na.connect("manual_template", open_devices=True)
capture = exp.camera.capture(frames=1, display=True, timeout_ms=60000)
```

Then manually provide a TTL positive edge to the qCMOS external trigger input.

The old `legacy_address_switch` backend remains available for comparison, but
it is not the recommended path. It can only write `pulse_lasting/cycle_counts`
and cannot express a full edge table.

## 7. Quick troubleshooting

- Remote connection fails: check FPGA computer IP, port `18861`, firewall, and
  that the server cell or PowerShell process is still running.
- Vivado fails: check `ZLC_PS_VIVADO_BIN`, `.xpr`, `.bit`, `.ltx`, and whether
  the VIO core probes use the required names.
- qCMOS timeout: check that the `qcm_trigger` FPGA output is physically wired
  to the qCMOS external trigger input and is a TTL positive edge.
- Sequence rejected: check `MAX_EDGES`, `TICK_WIDTH`, channel names, and
  overlapping pulses in `exp.timing.preflight()`. If the message says
  "clock grid", set GUI `step (ns)` to the FPGA tick and use integer multiples.
- Sitemap requires `grid_shape`: real hardware configs do not have a virtual
  trap array, so pass `grid_shape=(rows, cols)` explicitly.
