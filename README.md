# Zou_lab_control

Notebook-first neutral-atom experiment control: a standalone frontend layer
(plotting + PyQt/Fluent pulse GUI) and a standalone FPGA pulse-streamer hardware
side, joined over RPyC. The same experiment logic runs with virtual devices,
real qCMOS hardware, and a remote FPGA sequencer.

## Start Here

```powershell
install_requirements.bat            # install the editable package + record .zlc_python_path
start_tutorials_jupyter_lab.bat     # open the checked-in tutorials
pulse_gui.bat                        # open the pulse editor (offline if no server)
task_console.bat                     # open the live experiment dashboard (virtual camera)
```

The installer records the selected interpreter in the ignored local file
`.zlc_python_path`. The root launchers use that interpreter before falling back
to PATH, so the GUI, notebooks, and the FPGA server share the same editable
checkout. If VS Code/Jupyter already has the right kernel but PowerShell cannot
find that Python, run `%run ../install_current_kernel.py` from a notebook in
`tutorials/`.

### First day (new lab member)

Follow this order; everything before step 4 runs on a **virtual** backend (no
hardware needed), so you can practice the whole readout flow at your desk.

1. `install_requirements.bat`, then `start_tutorials_jupyter_lab.bat`.
2. **Learn the model + scripted readout:** `tutorials/neutral_atom_tutorial.ipynb`
   (connect virtual, calibrate, detect, scan) — start here. The plotting/pulse
   primitives are in `tutorials/frontend_tutorial.ipynb`.
3. **Drive the live GUI:** `tutorials/task_console_tutorial.ipynb` + `task_console.bat`
   (Add Panel → measurement/plot, run a calibration, watch a scan fill in).
4. **Go to real hardware:** read `docs/REAL_HARDWARE_BRINGUP_zh.md` (the
   first-power-on checklist), start the FPGA side with
   `tutorials/neutral_atom_fpga_server.ipynb`, then bring up the control PC with
   `tutorials/neutral_atom_hardware_quickstart.ipynb`.

A per-notebook one-liner + the recommended order also lives in
[tutorials/README.md](tutorials/README.md).

## Repository Layout

```text
Zou_lab_control/
  frontend/       standalone plotting, PyQt/Fluent GUI, PDF, notebook helpers
  neutral_atom/   device contracts, qCMOS/readout/session/timing logic
fpga/             standalone JTAG-to-AXI pulse-streamer build/server side
pulses/           checked-in PulseTableState presets
tutorials/        generated Jupyter notebooks
docs/             reference manuals, maintainer notes, bring-up checklist, generated PDFs
tests/            targeted verification matrix and tests
references/       historical source archives (ignored by git)
```

## Documentation

Four reference PDF manuals (built from `.texbody` sources), a real-hardware
first-day checklist, and one maintainer note:

- **First-power-on checklist:** [docs/REAL_HARDWARE_BRINGUP_zh.md](docs/REAL_HARDWARE_BRINGUP_zh.md)
  — environment prerequisites, the DCAM/`dcamapi.dll` setup, the firewall/port-18861
  notes, the ordered first-power-on verification, and a symptom → root-cause → fix
  table for the errors a first-timer hits at the machine. Read this before step 4 above.

| Manual | Source dir | Covers |
| --- | --- | --- |
| Main | `docs/main_manual/` | System architecture, neutral-atom session/devices/timing, `PulseSequence` vs `PulseTableState`, the sequencer lifecycle (prepare/fire/wait_done/safe_state), the real-hardware runbook, and the N-slot scan model end to end |
| Frontend | `docs/frontend_manual/` | The `qt_fluent` widget library and layout primitives, the pulse GUI (Edit/Preview/Scan tabs), the per-field scan-dot workflow, the plotting API, and PDF rendering |
| FPGA | `docs/fpga_manual/` | The Artix-7 35T edge-table pulse-streamer RTL, the 1-tick FIFO prefetch pipeline, the 2-bank streaming scan window, the affine N-slot scan engine, the analog-bus DAC engine, the event-scheduler output delays, the JTAG-to-AXI host upload flow, and the resource budget |
| Device | `docs/device_manual/` | Device configuration/contracts, `load_devices`, camera acquisition, the readout pipeline (sitemap/thresholds/detect), trap calibration, and the virtual backends |

- Maintainer/agent notes (architecture invariants, anti-patterns, QA): see
  [docs/MAINTAINER_NOTES.md](docs/MAINTAINER_NOTES.md).
- Subsystem pointers: [fpga/README.md](fpga/README.md),
  [fpga/pulse_streamer/README.md](fpga/pulse_streamer/README.md),
  [pulses/README.md](pulses/README.md),
  [Zou_lab_control/frontend/README.md](Zou_lab_control/frontend/README.md).
- Test strategy: [tests/README.md](tests/README.md).

### Building the manuals

Each manual is generated from a `.texbody` template into a `.tex` wrapper and
compiled with XeLaTeX (2-pass, in a temporary build dir). XeLaTeX must be on
PATH.

```powershell
python -c "from Zou_lab_control.neutral_atom.notes import build_main_manual, build_fpga_manual, build_device_manual; build_main_manual(); build_fpga_manual(); build_device_manual()"
python -c "from Zou_lab_control.frontend.notes import build_frontend_manual; build_frontend_manual()"
```

## Real Hardware Path

```text
control/qCMOS computer
  -> RemoteSequencer (RPyC)
  -> FPGA/Vivado computer running fpga\run_server.bat
  -> persistent Vivado hw_axi session (JTAG-to-AXI)
  -> zlc_pulse_streamer_top bitstream (edge-table engine)
```

The host packs the compiled program into a BRAM image and uploads it over
JTAG-to-AXI (`VivadoAxiStreamerSession` in
`Zou_lab_control/neutral_atom/devices/axi_session.py`); a CTRL register-file
mailbox carries COMMAND/STATUS and the streaming-scan handshake. The FPGA side
infers the full hardware contract from the board XDC. The
GUI may show only a subset such as `ch09/ch00/ch03/ch11`, but upload compiles
against the full hardware order; hidden or unconfigured channels are off. The
clock is 50 MHz (20 ns tick). Hardware-side commands on the FPGA/Vivado
computer:

```powershell
fpga\build_and_program.bat --check
fpga\build_and_program.bat
fpga\run_server.bat --check-config
fpga\run_server.bat
```

Generated Vivado projects and server state live under `fpga\build\ps`
by default (short name `ps` keeps Vivado's deep run/.Xil temp path under the
Windows MAX_PATH limit while staying in-repo); the printed `ZLC project dir` is
the source of truth for the
generated `impl_1\zlc_pulse_streamer_top.{bit,ltx}`. The full runbook is in the
**main manual**.

## Frontend

The pulse GUI edits a `PulseTableState` and drives a supplied sequencer; it is a
frontend only, not a separate hardware-control layer. Scanning uses named slots
`s0, s1, ...`: bind any duration/delay/DAC field (a scan dot in the GUI, or
`state.bind_field(kind, target)`), then provide an `N_points x N_slots`
`scan_table`. The **frontend manual** covers the widget library, the
Edit/Preview/Scan tabs, the scan-dot workflow, and the plotting/PDF API. Open
the editor remotely or offline:

```powershell
pulse_gui.bat --remote-host 192.168.0.20 --state .\pulses\camera_imaging_address_switch.json   # 192.168.0.20 = a PLACEHOLDER; use your FPGA PC's IP
pulse_gui.bat --no-sequencer --state .\pulses\camera_imaging_address_switch.json
```

The **task console** (`task_console.bat`) is the experiment-side dashboard: a
configurable grid of live panels (2D image with side distribution and draggable
clim, rolling trace, bimodal-fit histogram, 1D vector), each wired to named
experiment signals through a small Python expression (`value = occupied -
b_occupied`).  It opens empty: from **Add Panel** you assemble the logic nodes
(a camera Measurement publishing `frame`, a Judge-occupancy Processor turning it
into occupancy / loading-rate signals) and plot panels reading those signals.
A real experiment is the same graph publishing into a `SignalHub` from a real
camera (see the frontend manual's Task-console chapter).  Layouts save/load as
one JSON in `tasks/`.

## Targeted Verification

Prefer scoped checks from [tests/README.md](tests/README.md) over the full suite
for small changes. Typical handoff checks:

```powershell
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract"
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
git diff --check
```

Use full `pytest -q` for broad or cross-subsystem changes.
