# Test Strategy

Run the smallest check that proves the changed boundary still works.  Full `pytest -q` is reserved for broad handoff, release-like sweeps, or changes that touch multiple subsystems at once.  A small change you are confident in does not need a full-suite run.

## Principles (see the repo-root `AGENTS.md` for the full rules)

- **Contract tests, Python side only.** There is no iverilog/cocotb in the repo: RTL behaviour is checked by a faithful Python mirror plus `xsim` (the real IP netlist — the strongest hardware evidence), and verilog port widths are locked by a Python contract test (`test_..._vio_widths_match_python_generator`).
- **Every GUI uses the same two visual-debug paths.**  The fast path is
  `ensure_qt_app()` → the formal composition/open function → real Qt input →
  outer `FluentWindow.grab()` at this machine's actual display scale.  It must
  not force DPI, size, style, tabs, or controller state.  The slow final/dispute
  path starts the formal `.py/.bat` launcher and uses desktop mouse input plus a
  screen capture.  Both paths share the same application owner, composition,
  sizing/style, and operator sequence.  `offscreen`/`minimal` is behavior-only
  evidence and can never be reported as visual acceptance.  This rule applies
  to PulseGUI, TaskConsole, DeviceManager, FigureViewer, and every future GUI.
- **Performance optimizations must be logic/appearance-neutral.**  Only make the same output faster (analytic Jacobian, skip-if-unchanged guards, cached invariants); never change cadence/appearance.  Prove equivalence (e.g. fit `popt` agrees numerically).
- **After delete/refactor:** `git grep` of the dead identifier == 0; `python -m compileall` clean; no stray TODO/FIXME.

## Targeted Matrix

Use these as starting points, then narrow `-k` to the behavior you touched.

```powershell
# Neutral-atom API, pulse compilation, remote sequencer, qCMOS workflow
pytest -q tests\test_neutral_atom_lightweight.py -k "pulse or sequencer or qcmos or readout"

# Control-computer readout scan through the RPyC RemoteSequencer JSON protocol
pytest -q tests\test_neutral_atom_lightweight.py -k "remote_detection_time_scan_uses_bound_pulse_controller_over_json_protocol"

# FPGA launcher/HDL/Tcl contracts + host image packer + AXI session, without opening Vivado
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract or xdc or image_solver or top_regions or vivado_axi_session"

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
window.  Final or disputed appearance uses the slow path: launch the corresponding
root `.py/.bat` entry, drive the visible GUI with desktop mouse/keyboard, and take
a screen screenshot.  Both paths use the same application owner, composition,
sizing/style, and product input sequence.  Object-level checks for button text,
visible channels, labels, and geometry remain complementary behavior oracles.

## Cleanup

Test runs may create `.pytest_cache` or `__pycache__`; remove them before
handoff if they were created only by the current verification pass.  PDF
generation should use `Zou_lab_control.frontend.render_tex_pdf(...)`, which
compiles in a temporary directory and leaves only the final PDF or a
`.build.log` on failure.
