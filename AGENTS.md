# Zou_lab_control Agent Overview

This file is for future coding agents and maintainers. User-facing manuals must stay tutorial-like and should not include internal review language, blame, or conversational notes.

## First Read

- `docs/PROJECT_OVERVIEW.md`: concise system map for a new reader.
- `docs/DOCUMENTATION_GUIDE.md`: how manuals, notebook templates, and generated docs should be written.
- `docs/FPGA_PULSE_STREAMER_CAPACITY.md`: FPGA capacity notes for the runtime pulse-streamer design.
- `docs/FRONTEND_FLUENT_STYLE_GUIDE.md`: PyQt/Fluent visual rules, pulse GUI layout contract, and screenshot QA checklist.
- `docs/neutral_atom_hardware_manual/REAL_HARDWARE_RUNBOOK.md`: short operational hardware runbook.
- `tests/README.md`: targeted verification matrix; prefer scoped tests over full-suite runs unless the change is broad.
- `tutorials/neutral_atom_fpga_server.ipynb`: runs on the FPGA/Vivado computer.
- `tutorials/neutral_atom_hardware_quickstart.ipynb`: runs on the control/qCMOS computer.

## Core Architecture

`Zou_lab_control.neutral_atom` is organized around explicit boundaries:

- `devices/`: hardware adapters and device contracts. Devices own hardware actions only.
- `timing/`: `PulseSequence`, trigger counting, edge tables, and Verilog generation.
- `operations/`: pure image/calibration/detection algorithms that can run offline.
- `subsystems/`: experiment-level workflows, such as `exp.readout` and `exp.timing`.
- `views/`: plotting adapters to the frontend.
- `frontend/`: plotting, live updates, widgets, and notebook/PyQt-facing UI utilities.

Important rule: camera capture shows raw images. Calibration overlays belong to readout results, not to the camera device.

## Frontend Fluent Rule

PyQt frontends should reuse `Zou_lab_control.frontend.qt_fluent` and follow the Confocal GUI visual grammar documented in `docs/FRONTEND_FLUENT_STYLE_GUIDE.md`. For the pulse GUI, keep channel names, delays, and period checkbox rows aligned under a shared vertical scroll; period cards may scroll horizontally as a timeline, but not with independent per-card vertical scrollbars.

For pulse GUI visual changes, verify both the inner editor and the `FluentWindow` wrapper. Screenshots must wait for Qt rendering: after `show()` or any major state change, run the event loop and delay before `grab()` (for example `app.processEvents(); QtTest.QTest.qWait(1000); app.processEvents()`). Do not grab immediately after `show()`. Native Windows Qt screenshots are preferred for visual QA; offscreen screenshots can miss text, so also use object-level checks for button text, geometry, and state.

## Hardware Path

The default real-hardware path is the FPGA pulse-streamer backend:

1. Control computer calls `RemoteSequencer.prepare/fire/wait_done`.
2. FPGA/Vivado computer runs `Zou_lab_control.neutral_atom.devices.sequencer_server`.
3. The server command backend calls `Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer`.
4. The pulse-streamer backend uploads a runtime `ticks/masks` edge table through Vivado/VIO.
5. The fixed `zlc_pulse_streamer` bitstream executes pulse timing on the FPGA clock.

The old `legacy_address_switch` module remains available for comparison only. Do not make it the default path in hardware tutorials.

## Device Loader

`load_devices` loads a simple JSON/dict graph:

- each device entry has `type` and `params`;
- dependencies use `"$device:name"`;
- built-in classes are resolved lazily to avoid unnecessary hardware imports;
- external device classes can use a full import path or `register_device_class()`.

Keep this simple. Do not introduce a heavyweight dependency-injection framework unless the codebase actually needs it.

## Common Commands

Prefer the scoped matrix in `tests/README.md` when a change touches only one
boundary. Start with the smallest command that proves the edited boundary, for
example:

```powershell
pytest -q tests\test_neutral_atom_lightweight.py -k "pulse or sequencer or qcmos or readout"
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract or xdc or differential_edge_upload"
pytest -q tests\test_frontend_smoke.py -k "render_tex_pdf or pulse_gui"
python -m py_compile (rg --files -g "*.py" Zou_lab_control tests fpga)
python -m json.tool tutorials\neutral_atom_fpga_server.ipynb > $null
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
python -m json.tool tutorials\neutral_atom_tutorial.ipynb > $null
```

Use full `pytest -q` only for broad handoff, release-like sweeps, or changes
that genuinely cross many subsystems.

Generate pulse-streamer HDL:

```powershell
python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer generate_hdl `
  --output-dir D:\zlc_pulse_streamer_hdl `
  --channels trap cooling probe trig `
  --max-edges 1024 `
  --tick-width 32
```

For address-switch FPGA sizing, resource targets, and constraints, read
`docs/FPGA_PULSE_STREAMER_CAPACITY.md`.

## Documentation Rule

Separate audiences:

- User manuals teach concepts, workflow, API calls, expected results, and troubleshooting.
- Agent/maintainer notes record architecture constraints, review findings, anti-patterns, and implementation warnings.

Do not put sentences like "this is a serious architecture error" into user manuals. Rephrase as neutral behavior: what the component does, where the state lives, and what result the reader should expect.
