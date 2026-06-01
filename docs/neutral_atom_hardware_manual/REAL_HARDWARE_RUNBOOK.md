# Zou_lab_control Neutral Atom Hardware Runbook

This runbook is the short operational version of the PDF. It uses the new
`Zou_lab_control.neutral_atom` architecture only. Do not call old
`PythonCamDemo` acquisition scripts or old Python address-switch interfaces
from the experiment notebooks.

## 0. Install on both computers

Run this once on both the control/qCMOS computer and the Verilog/FPGA computer:

```powershell
cd C:\path\to\Zou_lab_control_v1
python -m pip install -e .
```

## 1. Verilog/FPGA Computer

Preferred Jupyter route:

```powershell
cd C:\path\to\Zou_lab_control_v1
jupyter lab tutorials\neutral_atom_fpga_server.ipynb
```

In the notebook, edit:

```python
os.environ["ZLC_VIVADO_PROJECT"] = r"D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.xpr"
os.environ["ZLC_VIVADO_BIT"] = r"D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.runs\impl_1\main.bit"
os.environ["ZLC_VIVADO_LTX"] = r"D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.runs\impl_1\main.ltx"
os.environ["ZLC_LEGACY_PULSE_PARAM"] = "pulse_lasting"
os.environ["ZLC_LEGACY_CYCLE_PARAM"] = "cycle_counts"
```

Then run the final cell:

```python
na.run_sequencer_server(...)
```

That cell blocks while the server is running. Keep the kernel alive.

Equivalent PowerShell route:

```powershell
cd C:\path\to\Zou_lab_control_v1

$env:ZLC_VIVADO_PROJECT = "D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.xpr"
$env:ZLC_VIVADO_BIT = "D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.runs\impl_1\main.bit"
$env:ZLC_VIVADO_LTX = "D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.runs\impl_1\main.ltx"
$env:ZLC_VIVADO_PROGRAM_ON_RUN = "0"
$env:ZLC_LEGACY_START_PARAM = "config_ready"
$env:ZLC_LEGACY_DEBUG_PARAM = "debug"
$env:ZLC_LEGACY_PULSE_PARAM = "pulse_lasting"
$env:ZLC_LEGACY_CYCLE_PARAM = "cycle_counts"
$env:ZLC_LEGACY_PROBE_CHANNEL = "probe"
$env:ZLC_LEGACY_WAIT_FOR_DURATION = "1"
$env:ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED = "0"
$env:ZLC_LEGACY_VIO_DEFAULTS = "{}"

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --host 0.0.0.0 `
  --port 18861 `
  --channels trap cooling probe qcm_trigger `
  --trigger-channels qcm_trigger `
  --clock-hz 100000000 `
  --state-dir D:\zlc_sequencer_state `
  --prepare-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch prepare" `
  --fire-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch fire" `
  --wait-done-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch wait_done" `
  --safe-state-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch safe_state"
```

For the current address-switch bitstream, `prepare` writes `pulse_lasting` and
`cycle_counts` through VIO for every prepared sequence, and `fire` raises
`config_ready`; `wait_done` waits for the finite sequence duration if needed
and then lowers `config_ready` again. Replace the environment paths and legacy
probe names if your bitstream uses different VIO names. If your FPGA clock is
not 100 MHz, change `--clock-hz` on both server and device config.

Leave `ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED=0` until an oscilloscope
confirms that the qCMOS trigger input sees exactly one positive edge per
address-switch cycle in run mode. The original `address_switch` Verilog has
two `emCCD` pulses per cycle and does not explicitly drive `trig` in run mode;
the backend intentionally refuses `prepare` until this is confirmed or the
Verilog is patched.

## 2. Control/qCMOS Computer

Open the real hardware notebook:

```powershell
cd C:\path\to\Zou_lab_control_v1
jupyter lab tutorials\neutral_atom_hardware_quickstart.ipynb
```

The main connection cell is:

```python
exp = na.connect(
    "remote_template",
    sequencer={"host": "192.168.0.20", "port": 18861},
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
scan = exp.readout.detection_time(times, shots=30, live=False, display=True)
```

For first-light without the remote server:

```python
exp = na.connect("manual_template", open_devices=True)
```

Then `exp.camera.capture()` will arm the qCMOS and print the manual trigger
message. Start the FPGA/manual trigger while the camera is waiting.

## 3. What Must Not Happen

- `neutral_atom_hardware_quickstart.ipynb` must not use `na.connect("virtual")`.
- `capture` must not display sitemap circles.
- The control notebook must not import or call old `PythonCamDemo` modules.
- The FPGA notebook must expose the new `sequencer_server`; any Vivado/Tcl
  backend lives behind `--prepare-command`, `--fire-command`, or
  `--wait-done-command`.

## 4. Quick Troubleshooting

- Remote connection fails: check FPGA computer IP, port `18861`, firewall, and
  that the server cell is still running.
- qCMOS open fails: check DCAM install, camera index, camera power/USB/Camera
  Link, and `QCMOSCamera` config.
- qCMOS timeout: check trigger cable, `config_ready`, `pulse_lasting`,
  `cycle_counts`, trigger polarity, `trigger_channels`, and whether the
  sequence has one trigger per requested frame.
- Sitemap requires `grid_shape`: real hardware configs do not have a virtual
  `trap_array`, so pass `grid_shape=(rows, cols)` explicitly.
