# Zou_lab_control

Notebook-first neutral-atom experiment control with typed data, pulse, device,
analysis, storage, and Qt workbench boundaries.

The current public products are:

- a complete virtual readout path for capture, calibration, occupancy, fit, and GUI work;
- an offline/virtual/remote Pulse GUI using the current `PulseDocument` model;
- a pulse-only remote FPGA installation;
- content-addressed experiment artifacts that can be reopened without exposing raw devices.

Complete real qCMOS + remote-sequencer readout composition is not yet published.
Pulse-server connectivity must not be interpreted as camera/readout readiness.

## Start

```powershell
install_requirements.bat
start_tutorials_jupyter_lab.bat
pulse_gui.bat
task_console.bat
```

The installer records the selected interpreter in the ignored
`.zlc_python_path`; the root launchers use that interpreter before falling back
to `PATH`.

There is exactly one user tutorial:
[`tutorials/neutral_atom_tutorial.ipynb`](tutorials/neutral_atom_tutorial.ipynb).
It executes the current virtual installation from request inspection through
capture, provenance, site-map/PSF calibration, occupancy, and the formal GUI
entry points.

For the currently supported real pulse-only path, follow
[`docs/REAL_HARDWARE_BRINGUP_zh.md`](docs/REAL_HARDWARE_BRINGUP_zh.md) and start
the server with `fpga\run_server.bat`.

## Package ownership

```text
zlc_data/          multidimensional data, selections, reductions, and fit
zlc_storage/       canonical encoding and content-addressed persistence
zlc_pulse/         PulseDocument, compiler, target contract, and wire protocol
zlc_neutral_atom/  experiment domains, repositories, runtime, and device ports
zlc_frontend/      headless figure/selector semantics and rendering
zlc_workbench/     Qt composition and product windows
Zou_lab_control/   notebook composition facade and public launch glue
fpga/              frozen pulse-streamer server/transport/deployment assets
tutorials/         the single executable user tutorial
```

The public notebook entry is:

```python
from pathlib import Path
import Zou_lab_control.notebook as zlc

exp = zlc.connect("virtual", repository=Path("results") / "experiment")
```

Ordinary notebook and GUI code receives typed facades and immutable artifacts,
not raw camera, sequencer, registry, or SDK objects.

## Design and operations

- [System architecture](docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md)
- [Design charter](docs/DESIGN_CHARTER_zh.md)
- [Migration ledger](docs/MIGRATION_LEDGER_zh.md)
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
