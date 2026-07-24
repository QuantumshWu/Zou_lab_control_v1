# Test Strategy

Run the smallest check that proves the changed boundary still works.  Full `pytest -q` is reserved for broad handoff, release-like sweeps, or changes that touch multiple subsystems at once.  A small change you are confident in does not need a full-suite run.

## Principles (see the repo-root `AGENTS.md` for the full rules)

- **Contract tests, Python side only.** There is no iverilog/cocotb in the repo: RTL behaviour is checked by a faithful Python mirror plus `xsim` (the real IP netlist — the strongest hardware evidence), and verilog port widths are locked by a Python contract test (`test_..._vio_widths_match_python_generator`).
- **Every GUI uses the same two visual-debug paths.**  The fast path selects
  `QT_QPA_PLATFORM=offscreen` before `ensure_qt_app()`, then enters the formal
  composition/open function, drives real Qt input, and captures the untouched
  outer `FluentWindow.grab()`.  It is the normal visual-and-behaviour iteration
  path and must not force DPI, size, style, tabs, or controller state.  The slow
  final/dispute path starts the formal `.py/.bat` launcher and uses desktop
  mouse input plus a screen capture.  Both paths share the same application
  owner, composition, sizing/style, and operator sequence; only the driver is
  different.  Ad-hoc widgets or a separately constructed QApplication are not
  evidence.  This rule applies to PulseGUI, TaskConsole, DeviceManager,
  FigureViewer, and every future GUI.
- **Performance optimizations must be logic/appearance-neutral.**  Only make the same output faster (analytic Jacobian, skip-if-unchanged guards, cached invariants); never change cadence/appearance.  Prove equivalence (e.g. fit `popt` agrees numerically).
- **After delete/refactor:** `git grep` of the dead identifier == 0; `python -m compileall` clean; no stray TODO/FIXME.

## Targeted Matrix

Use these as starting points, then narrow `-k` to the behavior you touched.

```powershell
# Neutral-atom notebook API, qCMOS adapter and exact capture workflow
pytest -q tests\test_notebook_experiment_facade.py tests\test_zlc_dcam_camera_adapter.py tests\test_triggered_capture_pipeline.py

# Control-computer readout scan through the RPyC RemoteSequencer JSON protocol
pytest -q tests\test_remote_sequencer_execution_endpoint.py tests\test_w3_api_segmented_scan.py

# FPGA launcher/HDL/Tcl contracts + host image packer + AXI session, without opening Vivado
pytest -q tests\test_public_hardware_boundary.py tests\test_zlc_pulse_fpga_image.py tests\test_zlc_pulse_hardware_backend.py tests\test_zlc_pulse_transports.py

# Frontend Figure contracts and the formal Qt host
pytest -q tests\test_zlc_frontend_figure.py tests\test_zlc_single_panel_host.py tests\test_qt_app_single_entry.py

# Pulse GUI behavior through current composition paths
pytest -q tests\test_pulse_gui_remote_connection.py tests\test_pulse_gui_scan_runtime_qt.py tests\test_pulse_schedule_view_contract.py
```

For notebook edits, validate only the notebooks that changed:

```powershell
python -m json.tool tutorials\neutral_atom_tutorial.ipynb > $null
```

For Python syntax after focused edits (`rg`/ripgrep on PATH; or substitute
`Get-ChildItem -Recurse -Filter *.py`):

```powershell
python -m py_compile (rg --files -g "*.py" Zou_lab_control tests)
```

## FPGA/Vivado Checks

The normal unit tests do not build or program hardware.  Use Vivado commands
only when HDL/Tcl/XDC/batch behavior changed and a Vivado machine is available.
These are run by the user on the FPGA/Vivado computer — an agent never launches
a Vivado build/synth/program itself (see the repo-root `AGENTS.md` §1); `xsim` /
`xvlog` / `xelatex` are fine.

```powershell
cmd /c fpga\build_and_program.bat --help
cmd /c fpga\run_server.bat --help

# HDL synth + capacity self-check; uses Vivado but not board pin constraints
fpga\build_and_program.bat --check

# Real hardware path; run only on the FPGA/Vivado computer
fpga\build_and_program.bat --build-only
fpga\build_and_program.bat --program-only
fpga\run_server.bat --check-config
fpga\run_server.bat
```

The scripts write to the printed project directory, normally
`fpga\build\ps`; that printed path is the source of truth for the
generated `impl_1\zlc_pulse_streamer_top.{bit,ltx}`.

## GUI Screenshot Checks

Application-specific fast flows reuse `tests/gui_user_flow.py`; they define only
their visible input sequence.  The shared owner selects `QT_QPA_PLATFORM=offscreen`
before the sole `ensure_qt_app()` call, then captures the untouched formal outer
window.  On Windows that same owner registers the product's declared system font
when the offscreen plugin provides no fonts; `test_qt_app_single_entry.py`
requires actual glyph pixels, so an empty-text screenshot cannot pass.  For
PulseGUI, run from the repo root:

```powershell
python tests\pulse_gui_user_flow.py --out .gui-evidence\pulse
```

This fast path must remain offscreen and therefore must not open a desktop
window.  It is valid daily visual evidence because it constructs the same formal
window through the same sole QApplication owner.  Final or disputed appearance
uses the slow path: launch the corresponding root `.py/.bat` entry, drive the
visible GUI with desktop mouse/keyboard, and take a screen screenshot.  Both paths
use the same application owner, composition, sizing/style, and product input
sequence.  Object-level checks for button text, visible channels, labels, and
geometry remain complementary behavior oracles.

## Cleanup

Test runs may create `.pytest_cache` or `__pycache__`; remove them before
handoff if they were created only by the current verification pass.
