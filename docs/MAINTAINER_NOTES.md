# Maintainer Notes

This file is a current operations index. It is not a migration diary or an
architecture authority.

Use these sources in order:

1. `AGENTS.md` for repository-wide rules and the lab-network trust boundary.
2. `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md` for current architecture and product
   acceptance contracts.
3. `docs/DESIGN_CHARTER_zh.md` for durable design constraints.
4. `docs/REAL_HARDWARE_BRINGUP_zh.md` and `fpga/README.md` for hardware
   qualification and troubleshooting.
5. `docs/MIGRATION_LEDGER_zh.md` only when historical migration evidence is
   explicitly needed.

## Current ownership

- `zlc_data/`: named multidimensional data, validity, selections, transforms,
  and fit execution.
- `zlc_storage/`: canonical encoding and content-addressed persistence.
- `zlc_pulse/`: `PulseDocument`, compilation, target contracts, and wire
  protocol.
- `zlc_neutral_atom/`: experiment domains, runtime, repositories, and device
  ports.
- `zlc_frontend/`: headless display values, figure projections, renderers,
  selector semantics, and the shared plot vocabulary in
  `zlc_frontend.plot_kind`.
- `zlc_frontend.qt_widgets/`: the sole Qt widget and Fluent component owner.
- `zlc_workbench/`: Qt composition and product windows. TaskConsole's layout,
  panel, logic-row, and stable producer-instance records live only in
  `zlc_workbench.task_console.console_records/console_state`.
- `Zou_lab_control/notebook/`: public typed facade and application composition.

Ordinary notebook and GUI code must use typed facades and immutable artifacts;
it must not reach into raw camera, sequencer, registry, runtime-service, or SDK
objects. Removed private modules and facade forwarding names are not supported
compatibility surfaces.

## UI and render ownership

Qt owns widgets and transient overlays. Worker lanes own Matplotlib figures,
artists, rendering, and expensive projections. Cross-thread publication uses
immutable fronts and revision checks.

There are two intentionally different raster surfaces:

- Interactive typed plots use `QtRasterBoard` with the same-revision
  `ViewportTransform`; zoom and pan update typed display state and trigger a
  worker re-render.
- Static encoded or multi-page reports use `FrozenRasterView` at native pixels
  inside `FluentScrollArea`. They do not resample a bitmap to imitate data-space
  zoom or pan, and they do not produce Selection, Fit, or ROI authority.

Do not create a second Fluent scale, widget facade, plot-kind vocabulary,
selector owner, renderer, or GUI-side data transform.

## Data and runtime invariants

- A logical scalar uses a canonical trailing length-one value carrier; rank-zero
  payload arrays are not accepted.
- Data preserves named axes, physical shape, dtype, validity, lineage, and exact
  artifact identity across runtime, storage, analysis, and display boundaries.
- Autonomous pulse/scan execution is device-timed. Host-stepped execution must
  not be described as timing-equivalent or substituted silently.
- The sequencer is a player. Camera, threshold, calibration, and feedback
  decisions stay in their owning readout/runtime domains.
- Closing a product retires its owned work, waits for required terminal/safe
  acknowledgements, and prevents stale publication.

## Hardware troubleshooting boundary

The currently supported real product is pulse-only. A successful remote pulse
connection does not qualify qCMOS capture, calibration, occupancy, or complete
real readout composition.

For pulse hardware, record the current target manifest, server snapshot, run
diagnostics, bitstream/build fingerprint, and oscilloscope evidence. Follow
`docs/REAL_HARDWARE_BRINGUP_zh.md`; do not infer readiness from a GUI connection
alone and do not rely on untracked session state.

The FPGA pin/clock source is `fpga/board_config/board.xdc` (override only through
the documented deployment mechanism). The default clock is 50 MHz, or 20 ns per
tick. Server and deployment details live in `fpga/README.md` and
`fpga/pulse_streamer/README.md`.

## Verification

Use the smallest current test closure that exercises the changed boundary. Do
not restore deleted architecture merely to make a historical test collect.

Useful repository checks:

```powershell
python -m pytest -q tests\test_tutorial_notebook_spine.py
python -m json.tool tutorials\neutral_atom_tutorial.ipynb > $null
git diff --check
```

GUI evidence must launch the formal product composition and allow the event loop
to settle before capture. Generated screenshots, caches, tutorial result trees,
and machine-local interpreter records are not source artifacts.
