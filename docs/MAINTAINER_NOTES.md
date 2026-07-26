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

Use `main` only as an on-demand oracle for a specifically identified legacy
behavior or physical algorithm. It is not the default UI or architecture
authority.

## Current ownership

- `zlc_data/`: named multidimensional data, validity, selections, transforms,
  and fit execution.
- `zlc_storage/`: canonical encoding and content-addressed persistence.
- `zlc_pulse/`: `PulseDocument`, compilation, target contracts, and wire
  protocol.
- `zlc_neutral_atom/`: experiment domains, repositories, devices, and generic
  runtime. `SignalDataPlane`, `HostedRun`, `HostedProcessor`, and
  `LiveDatasetHost` are the sole generic signal/data/hosted-lifecycle owners.
- `zlc_frontend/`: display values, figure projections, Fit/selector semantics,
  renderers, and the shared plot vocabulary in `zlc_frontend.plot_kind`.
- `zlc_frontend.qt_widgets/`: the sole Qt widget and Fluent component owner.
  `FigureSurfaceHost`, `FigureOutputAuthority`, and `FigureSurfaceLane` own the
  reusable interactive figure surface, derived outputs, and render lane.
- `zlc_workbench/`: Qt product composition and layout. It consumes the neutral
  runtime and frontend surfaces; it does not own their signal, run, processor,
  dataset, Figure, selector, Fit, or renderer contracts.
- `Zou_lab_control/api/`: the stable public Experiment API and application
  composition used by scripts, notebooks, and desktop products.
- `Zou_lab_control/workbench/`: the narrow desktop composition adapter.

Ordinary notebook and GUI code must use typed facades and immutable artifacts;
it must not reach into raw camera, sequencer, registry, runtime-service, or SDK
objects. Removed private modules and facade forwarding names are not supported
compatibility surfaces.

## UI and render ownership

Qt owns widgets and transient overlays. Worker lanes own Matplotlib figures,
artists, rendering, and expensive projections. Cross-thread publication uses
immutable fronts and revision checks.

There are two intentionally different raster roles:

- Interactive typed plots enter through `FigureSurfaceHost`. Its internal
  `QtRasterBoard` consumes the same-revision `ViewportTransform`; zoom and pan
  update typed display state and trigger `FigureSurfaceLane` to re-render.
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
- A producer publishes through `SignalDataPlane`; generic Task/Measurement
  lifecycle uses `HostedRun`, Processor lifecycle uses `HostedProcessor`, and
  live dataset publication uses `LiveDatasetHost`. Workbench only projects
  these typed states into Qt.
- Autonomous pulse/scan execution is device-timed. Host-stepped execution must
  not be described as timing-equivalent or substituted silently.
- The sequencer is a player. Camera, threshold, calibration, and feedback
  decisions stay in their owning readout/runtime domains.
- Closing a product retires its owned work, waits for required terminal/safe
  acknowledgements, and prevents stale publication.

## Hardware troubleshooting boundary

There are two real installation products: sequencer-only `remote_pulse`, and
the complete `hardware` package (remote FPGA + qCMOS DCAM + Pylon MOT camera).
The latter is software-ready for real-device E0/bring-up, but no apparatus is
qualified until DeviceManager initialization completes the active trigger-path
checks on those exact devices. A successful pulse connection alone proves only
the sequencer path.

For pulse hardware, record the current target manifest, server snapshot, run
diagnostics, bitstream/build fingerprint, and oscilloscope evidence. Follow
`docs/REAL_HARDWARE_BRINGUP_zh.md`; do not infer readiness from a GUI connection
alone and do not rely on untracked session state.

The FPGA pin/clock source is `fpga/board_config/board.xdc` (override only through
the documented deployment mechanism). The default clock is 50 MHz, or 20 ns per
tick. Server and deployment details live in `fpga/README.md` and
`fpga/pulse_streamer/README.md`. Launcher overrides are
`ZLC_FPGA_PYTHON` for Python and `ZLC_PS_VIVADO_BIN` for Vivado.

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

Keep those products in the ignored `_output/`, `.gui-evidence/`/`.gui_evidence/`,
or tool-cache trees; `.claude/` is session state and is never repository source.
`git ls-files -ci --exclude-standard` must remain empty, so an ignore rule can
never be used to conceal an already tracked generated artifact.
