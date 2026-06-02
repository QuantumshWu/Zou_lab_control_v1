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

## 1. Generate the FPGA pulse-streamer HDL

Run this on the Verilog/FPGA computer:

```powershell
cd C:\path\to\Zou_lab_control_v1
python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer generate_hdl `
  --output-dir D:\zlc_pulse_streamer_hdl `
  --channels trap cooling probe qcm_trigger `
  --max-edges 1024 `
  --tick-width 32
```

Add `D:\zlc_pulse_streamer_hdl\zlc_pulse_streamer.v` to a Vivado project.
Use `zlc_pulse_streamer_top_example.v` as the wiring reference. Create a VIO IP
with these probes:

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

Map `out[0:3]` to `trap`, `cooling`, `probe`, and `qcm_trigger`. The qCMOS
external trigger input must receive the `qcm_trigger` output.

Generate bitstream and probes. You should end up with paths like:

```text
D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.xpr
D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.bit
D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.ltx
```

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
$env:ZLC_PS_VIVADO_PROJECT = "D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.xpr"
$env:ZLC_PS_VIVADO_BIT = "D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.bit"
$env:ZLC_PS_VIVADO_LTX = "D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.ltx"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "1"
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
times = np.linspace(0.2e-3, 8e-3, 40)
scan = exp.readout.detection_time(times, shots=30, live=True, display=True)
```

## 4. Loader and extension rule

Keep `load_devices` configs simple: each device has `type` and `params`; use
`"$device:name"` for dependencies. New hardware classes should inherit the
right base class and either use a full import path in JSON or be registered:

```python
na.register_device_class("MySequencer", MySequencer)
na.device_class_registry()
```

Do not put FPGA transport details into the control notebook. The only control
PC contract is `RemoteSequencer.prepare/fire/wait_done`.

## 5. First-light fallback

For camera-only trigger checks:

```python
exp = na.connect("manual_template", open_devices=True)
capture = exp.camera.capture(frames=1, display=True, timeout_ms=60000)
```

Then manually provide a TTL positive edge to the qCMOS external trigger input.

The old `legacy_address_switch` backend remains available for comparison, but
it is not the recommended path. It can only write `pulse_lasting/cycle_counts`
and cannot express a full edge table.

## 6. Quick troubleshooting

- Remote connection fails: check FPGA computer IP, port `18861`, firewall, and
  that the server cell or PowerShell process is still running.
- Vivado fails: check `ZLC_PS_VIVADO_BIN`, `.xpr`, `.bit`, `.ltx`, and whether
  the VIO core probes use the required names.
- qCMOS timeout: check that the `qcm_trigger` FPGA output is physically wired
  to the qCMOS external trigger input and is a TTL positive edge.
- Sequence rejected: check `MAX_EDGES`, `TICK_WIDTH`, channel names, and
  overlapping pulses in `exp.timing.preflight()`.
- Sitemap requires `grid_shape`: real hardware configs do not have a virtual
  trap array, so pass `grid_shape=(rows, cols)` explicitly.
