# Zou_lab_control

Notebook-first neutral-atom experiment control with typed data, pulse, device,
analysis, storage, and Qt workbench boundaries.

The current public products are:

- a complete virtual readout path for capture, calibration, occupancy, fit, and GUI work;
- an offline/virtual/remote Pulse GUI using the current `PulseDocument` model;
- a sequencer-only `remote_pulse` installation;
- a full `hardware` installation package composing the remote FPGA, qCMOS DCAM
  readout camera, and Pylon MOT camera;
- typed direct-file experiment artifacts that can be reopened without exposing raw devices.

The full hardware package is software-ready for real-device E0/bring-up, but a
particular apparatus is not qualified merely because the package exists or the
pulse server connects. DeviceManager initialization must pass the active camera
and trigger-path qualifications on that apparatus before it publishes a runtime.

## Start

```powershell
install_requirements.bat
start_tutorials_jupyter_lab.bat
device_manager.bat
pulse_gui.bat
task_console.bat
```

The installer records the selected interpreter in the ignored
`.zlc_python_path`; the root launchers use that interpreter before falling back
to `PATH`.

There is exactly one user tutorial:
[`tutorials/neutral_atom_tutorial.ipynb`](tutorials/neutral_atom_tutorial.ipynb).
It executes the current virtual installation from request inspection through
capture, provenance, site-map/PSF calibration, per-model fidelity, occupancy,
a named-axis autonomous scan, release-recapture survival, and the formal GUI
entry points. The survival scan is not presented as a µK fit because that
public analysis/artifact owner has not been delivered.

For real hardware, follow
[`docs/REAL_HARDWARE_BRINGUP_zh.md`](docs/REAL_HARDWARE_BRINGUP_zh.md), start the
frozen-bitstream server with `fpga\run_server.bat`, and initialize either the
sequencer-only or complete hardware installation through DeviceManager.

## Package ownership

```text
zlc_data/          multidimensional data, selections, reductions, and fit
zlc_storage/       canonical encoding and durable atomic filesystem operations
zlc_pulse/         PulseDocument, compiler, target contract, and wire protocol
zlc_neutral_atom/  experiment domains, devices, SignalDataPlane, hosted runtime
zlc_frontend/      figure/selector/Fit semantics, renderers, and shared Qt surfaces
zlc_workbench/     Qt product composition and layout only
Zou_lab_control/   stable public API and desktop composition adapter
fpga/              frozen pulse-streamer server/transport/deployment assets
tutorials/         the single executable user tutorial
```

The same public API is used by scripts, notebooks, and desktop products:

```python
from pathlib import Path

from Zou_lab_control.api import InstallationConfigDocument, connect
from Zou_lab_control.api import WorkspacePaths

installation = InstallationConfigDocument.from_parameters(
    "virtual",
    {"seed": 7},
)
exp = connect(
    installation,
    workspace=WorkspacePaths.for_workspace(Path.cwd()),
)
```

For a real apparatus, author and initialize the `hardware` backend in
DeviceManager. Programmatic composition uses the same
`InstallationConfigDocument.from_parameters("hardware", values)` API; there are
no backend-specific convenience classmethods.

Ordinary application and GUI code receives typed facades and immutable artifacts,
not raw camera, sequencer, registry, or SDK objects.

## Outputs and saved files

The composition root creates one explicit `WorkspacePaths` value from the
selected project root. User-authored pulses and tasks live in that project;
artifacts and operator-facing exports live under `project/_output`. Leaf
packages never infer another root from their package path, current directory,
home directory, or environment.

| Product or action | Default location |
|---|---|
| Calibration record and optional report | `workspace.output_root / "calibrations"` |
| MOT-field acquisition | `workspace.output_root / "captures"` |
| TaskConsole figure export | `workspace.output_root / "figures/task-console"` |
| DataFigure / FigureViewer export | `workspace.output_root / "figures/data-figure"` |
| Pulse preview export | `workspace.output_root / "figures/pulses"` |

Each Calibration run folder commits the reloadable `calibration.json` record
first.  A post-FINAL best-effort export may then add the shared-`zlc_plot`
`report/*.png` pages and, only when `save_frames` was explicitly selected,
original-dtype `source_frames.npy` plus `source_frame_validity.npy`.
`CalibrationArtifactRef` names the record directly; there is no hidden
repository, pointer file, result-bundle manifest, or CAS. MOT reuses the generic
Capture artifact and derives its typed FINAL outputs statelessly from that
capture.

Measurements and processors publish typed live/FINAL signals to the Logic
tree; they do not silently create arbitrary files.  Save/Export is an explicit
Figure action.  Editable pulse inputs belong in `pulses/`, while generated
preview images belong under `WorkspacePaths.output_root` as listed above.

## Design and operations

- [System architecture — sole normative design](docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md)
- [Real-hardware bring-up](docs/REAL_HARDWARE_BRINGUP_zh.md)
- [FPGA server notes](fpga/README.md)
- [Verification guide](tests/README.md)

## Targeted verification

Prefer the smallest current product test that exercises the changed boundary.
Do not repair historical tests by restoring removed architecture.

```powershell
python -m pytest -q tests\test_tutorial_notebook_spine.py
python -m json.tool tutorials\neutral_atom_tutorial.ipynb > $null
git diff --check
```
