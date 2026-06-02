# Zou_lab_control Project Overview

`Zou_lab_control` is a neutral-atom experiment-control codebase with a notebook-first workflow. It separates hardware control, timing, image analysis, plotting, and result objects so the same experiment logic can run with virtual devices, real qCMOS hardware, and a remote FPGA sequencer.

## Main User Workflows

- Offline tutorial: `tutorials/neutral_atom_tutorial.ipynb`
- Real hardware tutorial: `tutorials/neutral_atom_hardware_quickstart.ipynb`
- FPGA/Vivado server tutorial: `tutorials/neutral_atom_fpga_server.ipynb`
- One-click launcher on Windows: `start_tutorials_jupyter_lab.bat`

Typical real-hardware flow:

```python
import Zou_lab_control.neutral_atom as na

exp = na.connect(
    "remote_template",
    sequencer={"host": "192.168.1.123", "port": 18861},
    open_devices=True,
)

exp.timing.configure_imaging(exposure=2e-3, load=True)
preflight = exp.timing.preflight()
preflight.raise_if_failed()

capture = exp.camera.capture(frames=1, display=True)
sitemap = exp.readout.sitemap(frames=20, grid_shape=(5, 7), display=True)
threshold = exp.readout.thresholds(frames=120, site=0, display=True)
shot = exp.readout.detect(display=True)
```

## Package Map

```text
Zou_lab_control/
  frontend/                 plotting, live sessions, notebook setup
  neutral_atom/
    session.py              NeutralAtomSession and connect()
    configs/                virtual/manual/remote device configs
    core/                   data records, analysis helpers, result objects
    devices/                camera, sequencer, FPGA, virtual devices
    operations/             standalone calibration/detection functions
    subsystems/             exp.readout and exp.timing workflows
    timing/                 PulseSequence, trigger counting, Verilog helpers
    views/                  neutral-atom plotting adapters
```

## Layer Responsibilities

- Camera devices acquire images and expose configuration. They do not know site maps, thresholds, or simulator truth.
- Sequencer devices prepare/fire/wait pulse sequences. Real qCMOS acquisition prepares the sequence before arming the camera, then fires after `cap_start`.
- `PulseSequence` is the timing source of truth. It compiles to edge-table ticks and masks for Verilog or runtime upload.
- Readout calibration owns camera-space site centers, ROI radius, thresholds, and occupancy decisions.
- Frontend code owns figures, artists, widgets, and live refresh. Worker code should not mutate Matplotlib artists directly.

## Real FPGA Path

The recommended backend is a fixed pulse-streamer bitstream:

```text
Control PC notebook
  -> RemoteSequencer over RPyC
  -> SequencerService on FPGA/Vivado PC
  -> fpga_pulse_streamer command backend
  -> Vivado/VIO upload of ticks/masks
  -> zlc_pulse_streamer FPGA edge-table RAM
```

This design makes each acquisition upload a new sequence without rebuilding Verilog. Network and Vivado commands handle setup/start; microsecond timing is executed by FPGA clocked logic.

For FPGA resource sizing, pin-count assumptions, and 40-channel pulse-streamer notes, see `docs/FPGA_PULSE_STREAMER_CAPACITY.md`.

## Documentation Map

- `docs/neutral_atom_hardware_manual/neutral_atom_hardware_quickstart_zh.pdf`: full hardware tutorial.
- `docs/neutral_atom_hardware_manual/REAL_HARDWARE_RUNBOOK.md`: short operational checklist.
- `docs/FRONTEND_FLUENT_STYLE_GUIDE.md`: PyQt/Fluent visual rules and pulse GUI QA checklist.
- `docs/neutral_atom_manual/neutral_atom_manual_zh.tex`: broader neutral-atom software manual.
- `docs/frontend_manual/frontend_manual_zh.tex`: frontend/live plotting manual.
- `docs/control_migration_manual/` and `docs/2d_rearrangement_manual/`: historical-code migration notes.

Historical reference code lives in `references/` and is intentionally ignored by git.

## Verification Checklist

Before handing off changes:

```powershell
pytest -q
python -m py_compile (rg --files -g "*.py" Zou_lab_control tests)
git diff --check
```

For notebook/template changes, also validate notebook JSON:

```powershell
python -m json.tool tutorials\neutral_atom_fpga_server.ipynb > $null
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
```
