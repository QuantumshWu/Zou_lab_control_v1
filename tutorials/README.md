# Tutorials — read in this order

Every notebook except the real-hardware ones runs on the **virtual** backend (no
hardware), so you can practice the entire calibrate → readout → scan flow at your
desk. Switching to the real machine changes only the `na.connect(...)` line.

| # | Notebook | What it teaches | Hardware? |
| - | --- | --- | --- |
| 1 | `neutral_atom_tutorial.ipynb` | **Start here.** The mental model (device → measurement → processor → task → plot, all coupled through one `SignalHub`) and the scripted API: `na.connect("virtual")`, calibrate a sitemap + thresholds, detect atoms, run a detection-time / temperature scan, save a run. | virtual |
| 2 | `frontend_tutorial.ipynb` | The plotting + pulse primitives the GUI is built on: 1D / 2D / histogram / monitor / site-map plots (`zf.plot`, `zf.run`), and the `PulseTableState` scan-slot model. | virtual |
| 3 | `task_console_tutorial.ipynb` | The live **GUI** that ties it together: open the console, **Add Panel** to assemble logic nodes (a camera measurement, a Judge-occupancy processor) + plot panels, run a calibrate task, watch a scan fill in. Includes a pure-API path that runs headless. | virtual |
| 4 | `neutral_atom_fpga_server.ipynb` | Bring up the **FPGA side**: start the RPyC sequencer server on the Vivado PC. | **real** |
| 5 | `neutral_atom_hardware_quickstart.ipynb` | Bring up the **control PC**: connect the real qCMOS + remote sequencer, preflight, sitemap, thresholds, detect, first real readout. | **real** |
| 6 | `qcmos_live_2d.ipynb` | Side path: connect ONLY the qCMOS camera (no sequencer) and watch a live 2D image in the task console. Useful for camera/alignment work. | camera only |

Before step 4, read [../docs/REAL_HARDWARE_BRINGUP_zh.md](../docs/REAL_HARDWARE_BRINGUP_zh.md)
(the first-power-on checklist + the "most common on-site errors" table).

To understand the architecture or extend the system (add a measurement / processor
/ plot type), read [../docs/task_console_design/architecture_v2_zh.md](../docs/task_console_design/architecture_v2_zh.md)
and the frontend manual; `Zou_lab_control/neutral_atom/operations/measurements/temperature.py`
is the worked template to copy for a new measurement.
