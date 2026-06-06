# Test Strategy

Run the smallest check that proves the changed boundary still works.  Full `pytest -q` is reserved for broad handoff, release-like sweeps, or changes that touch multiple subsystems at once.

## Targeted Matrix

Use these as starting points, then narrow `-k` to the behavior you touched.

```powershell
# Neutral-atom API, pulse compilation, remote sequencer, qCMOS workflow
pytest -q tests\test_neutral_atom_lightweight.py -k "pulse or sequencer or qcmos or readout"

# Control-computer readout scan through the RPyC RemoteSequencer JSON protocol
pytest -q tests\test_neutral_atom_lightweight.py -k "remote_detection_time_scan_uses_bound_pulse_controller_over_json_protocol"

# FPGA launcher/HDL/Tcl contracts without opening Vivado
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract or dry_run_uses_short_project_artifacts or xdc or differential_edge_upload"

# Frontend plotting, PDF rendering, notebook-template generation
pytest -q tests\test_frontend_smoke.py -k "frontend or render_tex_pdf or notebook"

# Pulse GUI behavior; requires PyQt5 and may skip when Qt canvas is unavailable
pytest -q tests\test_frontend_smoke.py -k "pulse_gui"
```

For notebook edits, validate only the notebooks that changed:

```powershell
python -m json.tool tutorials\neutral_atom_fpga_server.ipynb > $null
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
python -m json.tool tutorials\neutral_atom_tutorial.ipynb > $null
```

For Python syntax after focused edits:

```powershell
python -m py_compile (rg --files -g "*.py" Zou_lab_control tests)
```

## FPGA/Vivado Checks

The normal unit tests do not build or program hardware.  Use Vivado commands
only when HDL/Tcl/XDC/batch behavior changed and a Vivado machine is available.

```powershell
cmd /c fpga\build_and_program.bat --help
cmd /c fpga\run_server.bat --help

# HDL/VIO width self-check; uses Vivado but not board pin constraints
fpga\build_and_program.bat --check

# Real hardware path; run only on the FPGA/Vivado computer
fpga\build_and_program.bat --build-only
fpga\build_and_program.bat --program-only
fpga\run_server.bat --check-config
fpga\run_server.bat
```

Do not use stale Vivado products from `fpga\pulse_streamer\build`.  Current
scripts write to the printed project directory, normally `fpga\build\address_switch`, and
ignore old `fpga\pulse_streamer\build` project, bitstream, and probes paths.

## GUI Screenshot Checks

When pulse GUI layout changes, verify the inner `PulseSequenceEditor` and the
outer `FluentWindow`.  Let Qt render before grabbing screenshots:

```python
app.processEvents()
QtTest.QTest.qWait(1000)
app.processEvents()
editor.grab_screenshot(path)
```

Object-level checks for button text, visible channels, labels, and geometry are
still useful because offscreen screenshots can miss native Windows text.

## Cleanup

Test runs may create `.pytest_cache` or `__pycache__`; remove them before
handoff if they were created only by the current verification pass.  PDF
generation should use `Zou_lab_control.frontend.render_tex_pdf(...)`, which
compiles in a temporary directory and leaves only the final PDF or a
`.build.log` on failure.
