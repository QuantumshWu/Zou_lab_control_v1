# Zou_lab_control

Notebook-first neutral-atom experiment control with a standalone frontend layer
and a standalone FPGA pulse-streamer hardware side.

## Start Here

- Install Python/Jupyter/PyQt/RPyC requirements:

  ```powershell
  install_requirements.bat
  ```

- Open tutorials:

  ```powershell
  start_tutorials_jupyter_lab.bat
  ```

- Open the pulse editor without an experiment config:

  ```powershell
  pulse_gui.bat
  ```

The installer records the selected interpreter in the ignored local file
`.zlc_python_path`.  The root launchers use that interpreter before falling
back to PATH, so GUI, notebooks, and the FPGA server share the same editable
checkout.

If VS Code/Jupyter already has the right kernel selected but PowerShell cannot
find that Python, run this from a notebook in `tutorials/`:

```python
%run ../install_current_kernel.py
```

It installs the same editable checkout into the active kernel and writes
`.zlc_python_path` for the launchers.

## Repository Layout

```text
Zou_lab_control/
  frontend/       standalone plotting, PyQt/Fluent GUI, PDF, notebook helpers
  neutral_atom/   device contracts, qCMOS/readout/session/timing logic
fpga/             standalone 40ch pulse-streamer build/server side
pulses/           checked-in PulseTableState presets
tutorials/        generated Jupyter notebooks
docs/             manuals, runbooks, design notes, generated PDFs
tests/            targeted verification matrix and tests
references/       historical source archives for comparison only
```

## Real Hardware Path

```text
control/qCMOS computer
  -> RemoteSequencer
  -> FPGA/Vivado computer running fpga\run_server.bat
  -> persistent Vivado/VIO session
  -> fixed zlc_pulse_streamer_top_40ch bitstream
```

The FPGA side is always a 40-channel hardware contract by default.  The GUI may
show only `ch00..ch03`, but upload compiles against the full hardware order
`ch00..ch39`; hidden or unconfigured channels are off.

Use these hardware-side commands on the FPGA/Vivado computer:

```powershell
fpga\build_and_program.bat --check
fpga\build_and_program.bat
fpga\run_server.bat --check-config
fpga\run_server.bat
```

Vivado projects, `.runs`, `.cache`, `.hw`, `.sim`, `.ltx`, journals, and server
state are generated under `fpga\build` by default.  The FPGA batch files print
`ZLC build root` and `ZLC project dir`; the default project is
`fpga\build\p40`, and that printed path is the source of truth for
`.xpr/.bit/.ltx`.

## Key Docs

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [FPGA submodule](fpga/README.md)
- [Pulse-streamer design](fpga/pulse_streamer/README.md)
- [Frontend submodule](Zou_lab_control/frontend/README.md)
- [Pulse presets](pulses/README.md)
- [Hardware runbook](docs/neutral_atom_hardware_manual/REAL_HARDWARE_RUNBOOK.md)
- [Test strategy](tests/README.md)

## Targeted Verification

Prefer scoped checks from [tests/README.md](tests/README.md) instead of running
the full suite for every small change.  Typical handoff checks are:

```powershell
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract"
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
git diff --check
```

Use full `pytest -q` for broad handoff or cross-subsystem changes.
