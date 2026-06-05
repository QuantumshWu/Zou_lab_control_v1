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

## Standalone Entry Points

The repo root keeps only user-facing launchers that are not tied to one
hardware board:

- `install_requirements.bat`: install the editable package and record the
  Python path in `.zlc_python_path`.
- `pulse_gui.bat`: open the pulse GUI as a frontend, either local or remote.
- `start_tutorials_jupyter_lab.bat`: open the checked-in tutorials.

The FPGA side is intentionally grouped under `fpga/`:

- `fpga/build_and_program.bat`: build/check/diagnose/program the
  address-switch pulse-streamer bitstream.
- `fpga/run_server.bat`: start the XDC-inferred sequencer server.
- `fpga/pulse_streamer/`: HDL, XDC, Vivado Tcl, and the design note.

Generated Vivado projects, `.runs`, `.cache`, `.hw`, `.sim`, `.ltx`, journals,
and server state live under `fpga\build` by default.  The batch files print
`ZLC build root` and `ZLC project dir`; the default real project is
`fpga\build\address_switch`, and that printed `ZLC project dir` is the source
of truth for `.xpr/.bit/.ltx`.

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
  -> fpga_pulse_streamer persistent Vivado-session backend
  -> Vivado/VIO upload of ticks/masks
  -> zlc_pulse_streamer FPGA edge-table RAM
```

This design makes each acquisition upload a new sequence without rebuilding Verilog. The default server keeps one Vivado Tcl process alive while the server runs; microsecond timing is executed by FPGA clocked logic.

For FPGA resource sizing, pin-count assumptions, and pulse-streamer capacity
notes, see `docs/FPGA_PULSE_STREAMER_CAPACITY.md`.

## Documentation Map

- `docs/neutral_atom_hardware_manual/neutral_atom_hardware_quickstart_zh.pdf`: full hardware tutorial.
- `docs/neutral_atom_hardware_manual/REAL_HARDWARE_RUNBOOK.md`: short operational checklist.
- `docs/FRONTEND_FLUENT_STYLE_GUIDE.md`: PyQt/Fluent visual rules and pulse GUI QA checklist.
- `docs/neutral_atom_manual/neutral_atom_manual_zh.tex`: broader neutral-atom software manual.
- `docs/frontend_manual/frontend_manual_zh.tex`: frontend/live plotting manual.
- `docs/control_migration_manual/` and `docs/2d_rearrangement_manual/`: historical-code migration notes.

Historical reference code lives in `references/` and is intentionally ignored by git.

## Verification Checklist

Before handing off changes, prefer the scoped matrix in `tests/README.md`.  Run
only the checks that cover the files and behavior you touched, then broaden the
sweep when the change crosses subsystem boundaries.

Common targeted checks:

```powershell
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract or dry_run_uses_short_project_artifacts"
pytest -q tests\test_frontend_smoke.py -k "render_tex_pdf or pulse_gui"
python -m py_compile (rg --files -g "*.py" Zou_lab_control tests fpga)
git diff --check
```

For notebook/template changes, also validate notebook JSON:

```powershell
python -m json.tool tutorials\neutral_atom_fpga_server.ipynb > $null
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
```

Use full `pytest -q` for broad handoff, release-like sweeps, or changes that
touch frontend, neutral-atom runtime, FPGA scripts, and docs at the same time.
