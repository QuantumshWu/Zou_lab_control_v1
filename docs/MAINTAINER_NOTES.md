# Maintainer Notes

This is the single agent/maintainer note for `Zou_lab_control`. It consolidates
the former `DOCUMENTATION_GUIDE.md`, `PROJECT_OVERVIEW.md`,
`FPGA_PULSE_STREAMER_CAPACITY.md`, `FRONTEND_FLUENT_STYLE_GUIDE.md`,
`AGENTS.md`, and the implementation half of the old hardware runbook.

It records architecture constraints, invariants, anti-patterns, and review
findings. User-facing tutorials live in the four PDF manuals
(`docs/main_manual`, `docs/frontend_manual`, `docs/fpga_manual`,
`docs/device_manual`) and must stay
tutorial-like: explain behaviour and state ownership, not blame.

## 1. Documentation Layout And Rules

There are two audiences. Keep them separate.

- **User manuals** (the four PDFs) teach concepts, workflow, API calls,
  expected output, and troubleshooting in a neutral instructional voice.
- **Maintainer notes** (this file, plus review findings) may name
  anti-patterns, failure modes, and invariants directly.

Style rules for manuals:

- Prefer "this component does X" over "do not do Y" unless it is safety-critical.
- Prefer "recommended path" / "fallback path" over "right/wrong".
- Always state who owns the state: camera, sequencer, session, readout
  calibration, frontend plot.
- Do not put sentences like "this is a serious architecture error" into a
  manual. Rephrase as neutral behaviour. That invariant belongs here.
- Keep historical-code discussion in `references/`, not in quickstarts.

Source of truth for generated docs (edit the template, then rebuild the PDF):

- main manual body: `Zou_lab_control/neutral_atom/content/manual_templates/main_manual_zh.texbody`
- frontend manual body: `Zou_lab_control/frontend/content/manual_templates/frontend_manual_zh.texbody`
- FPGA manual body: `Zou_lab_control/neutral_atom/content/manual_templates/fpga_manual_zh.texbody`
- device & experiment manual body: `Zou_lab_control/neutral_atom/content/manual_templates/device_manual_zh.texbody`
- shared preamble: `Zou_lab_control/frontend/templates/zlc_frontend_notes.sty`

The build entry points are in `Zou_lab_control/neutral_atom/content/manuals.py`
and `Zou_lab_control/frontend/content/manuals.py`; both call
`Zou_lab_control.frontend.notes.render_tex_pdf` / `render_notes_pdf`. See
section 10 for the build commands.

Notebook markdown should be short and operational: say what the next cell does,
show the concrete call, and link to a manual for background.

## 2. Core Architecture

`Zou_lab_control.neutral_atom` is organized around explicit boundaries:

- `devices/`: hardware adapters and device contracts. Devices own hardware
  actions only.
- `timing/`: `PulseSequence` (sequence.py), `PulseTableState` (pulse_table.py),
  trigger counting, edge tables, and Verilog generation (verilog.py).
- `operations/`: pure image/calibration/detection algorithms that run offline.
- `subsystems/`: experiment-level workflows such as `exp.readout`, `exp.timing`.
- `views/`: plotting adapters to the frontend.
- `frontend/`: plotting, live updates, Fluent widgets, notebook/PyQt utilities.

Invariants:

- Camera capture shows **raw** images. Calibration overlays belong to readout
  results, not the camera device.
- `PulseSequence` / `PulseTableState` is the timing source of truth. The GUI is
  a frontend; it must not create a separate hardware-control layer.
- Frontend code owns figures, artists, widgets, and live refresh. Worker code
  must not mutate Matplotlib artists directly.
- `load_devices` loads a simple JSON/dict graph: each entry has `type`/`params`,
  dependencies use `"$device:name"`, built-in classes resolve lazily, external
  classes use a full import path or `register_device_class()`. Do not grow this
  into a heavyweight dependency-injection framework.
- **Sequencer / streamer is purely a player.** The sequencer and the FPGA edge
  streamer contain NO camera/trigger judgment. The streamer only plays digital
  edges, analog-bus segments, and event-scheduler output delays; the engine HDL
  (`zlc_edge_streamer.v`) has no camera/acquire/readout/detect logic at all. A
  trigger channel is just one more digital output the player drives — the decision
  about *when* to count or threshold lives in the acquisition/feedback subsystem
  (`subsystems/`, readout), not in timing. Do not push exposure/threshold/feedback
  decisions down into the sequencer; keep playback and acquisition decoupled.

### Readout-fidelity calibration: the reference-bracket flow

`TrapCalibration.signals(image, method=)` extracts one scalar per site by one of three
readout models — `box` (square ROI, no background subtraction), `psf` (per-site
matched-filter, annulus background), `uniform_psf` (one shared kernel, annulus). The
background model is PART of the readout and is carried per method in `by_method`, so
`signals(method=m)` reads on the SAME scale `m`'s thresholds were trained on.

`CalibrateReadoutTask` (`operations/logic.py`) follows the Rb87 single-atom flow:

1. **Reference brackets** (`_collect_bracket_groups`): each shot is ONE correlated
   long-short-long camera sequence imaging the SAME atoms. **The template FILE itself IS the
   bracket** — `default_imaging_template()` / `pulses/imaging_template.json` is six periods
   (`load`, `image_0`, `gap_0`, `image_1`, `gap_1`, `image_2`): one cooling/load cycle, then
   THREE camera-trigger frames separated by trap-held gaps (the gap holds ONLY the persistent
   trap so the emCCD line drops and re-rises into three DISTINCT triggers; cooling/probe/emCCD
   are off in the gap, so no re-cooling rearranges the atoms mid-bracket). **`file == fired`**:
   the cali does NOT unroll/derive a bracket (the deleted `with_imaging_bracket`) — it loads the
   template and sets ONLY the two exposure durations BY NAME via **API slots**: `set_api("a1",
   reference_exposure)` (the long reference frames, image_0 + image_2 share the `a1` handle) and
   `set_api("a2", readout_exposure)` (the short readout, image_1). `_imaging_layout` reads the
   frame count + readout index from the template (emCCD-trigger periods; the `a2`-tagged one is
   the readout), so a user's own template with N triggers works too. `reference_exposure` (LONG)
   and `readout_exposure` (SHORT) are both explicit cali params — open the template in the pulse
   GUI and you SEE exactly the long-short-long the cali fires. Comparing the two long frames
   tells whether the atom survived; when they
   AGREE (strict consensus) they vote the ground-truth occupancy for the short readout, and a
   shot where they disagree (atom loss, modelled by the virtual `detection_lifetime`) is dropped
   as ambiguous. (`timing.reference_bracket_sequence` builds the equivalent bracket FROM SCRATCH
   — explicit channels, no template — for tests/notebooks without a template file.)
2. **Site map + PSF** from averaging the bracket's LONG reference frames (each images a real
   ~50% loading; the average reveals every trap). This is the SAME path live and from saved
   frames — there is no separate site-map acquisition pass.
3. **Held-out per-method fidelity** (`operations.fidelity.characterize_readout`, run per
   method in `calibration_report._held_out_by_method`): each method's per-site threshold is
   trained on a split of the labelled short readout and scored on a HELD-OUT split. Because
   box / per-site PSF / uniform PSF weight the photons differently, their held-out fidelity
   at a fidelity-limited readout DIFFERS — this is why the report computes all three. The
   self-consistent otsu-split fidelity estimate is affine-invariant and CANNOT tell the
   methods apart, so it is only the fallback for folder runs that kept no reference brackets.
4. The reference-trained per-site thresholds are written back into the calibration
   (`with_method_thresholds`, `threshold_method="per_site_reference"`) so `detect` reads on
   the trained boundary, not the otsu quick split.

**Per-frame exposure.** A real externally-triggered camera integrates each frame for the
window ITS trigger gates, so a heterogeneous bracket images successive frames for different
durations. The virtual camera honours this via `devices.virtual.exposures_per_frame`
(parallel to the cooling/trap-off per-frame helpers); a uniform repeated sequence yields one
exposure per frame, identical to the legacy behaviour. The real qCMOS adapter, given a
non-uniform bracket, integrates for the longest probe (the trigger gates each frame; atoms
scatter only during their own probe), keeping virtual == real.

### Fluent tab bar (frontend): water-fill, no scroll arrows

`FluentTabWidget` (`frontend/qt_fluent._FluentTabBar`) is a pivot-underline tab bar. When the
tabs overflow the bar it WATER-FILLS: only the widest tabs are capped to a shared width and
their labels elide with `...`, while short tabs (Monitor / Logic) keep their natural width;
every tab — including its right-side close `x` — stays inside the bar, so no native scroll
arrows ever appear and no close `x` is clipped. `sizeHint` reports the NATURAL total (so the
QTabWidget grants the bar its full width; `tabSizeHint` caps to the actual width); reporting
the capped sum would feed back and collapse the bar. The corner `...` overflow button lists
every full title. Do NOT re-introduce an equal `width // count` squeeze (it crams short tabs
to slivers) or native scroll chevrons (they crowd the close `x`).

## 3. Real Hardware Path

Default real-hardware path (the only one hardware tutorials should use):

```text
control/qCMOS computer
  -> RemoteSequencer.prepare/fire/wait_done (RPyC)
  -> SequencerService on FPGA/Vivado computer (fpga\run_server.bat)
  -> VivadoAxiStreamerSession (persistent Vivado hw_axi / JTAG-to-AXI)
  -> axi_bram_ctrl: pack + upload BRAM image (edges/scan/bus) + CTRL mailbox
  -> zlc_pulse_streamer_top.bit on the FPGA (edge-table engine)
```

The FPGA side infers the full hardware contract from the board XDC
(`fpga\board_config\board.xdc` — see that folder's README; override with `ZLC_PS_XDC`;
62 controllable outputs, fallback `ch00..ch61`). GUI visibility is a view
operation only; the server always pads to full hardware width and zeros
hidden/unconfigured channels. The RPyC server binds `0.0.0.0` with pickle
enabled by design for the isolated lab network — the declared trust posture is
AGENTS.md §2「信任边界」.

Key fixed facts:

- Default FPGA clock is **50 MHz**, so one tick is **20 ns**. If measured pulses
  are 2x the set value, something still assumes 100 MHz — fix the
  control/server/GUI clock, not the pin map.
- Camera-imaging preset trigger is `ch11/emCCD/M13`. The XDC also defines
  `ch06/trig/R17`; that is a separate output, not the preset trigger.
- Camera-imaging visible subset: `ch09 trap (M17)`, `ch00 cooling (F15)`,
  `ch03 probe (N15)`, `ch11 emCCD (M13)`.
- Four 10-bit analog buses: `da_dipole`, `da_bias_y`, `da_bias_x`, `da_bias_z`.

**Plot region → device, coordinate contract (who converts what; THREE layers).**
A plot's selector/zoom is a GENERIC interface: it yields a rectangle as four
endpoints `(x_min, x_max, y_min, y_max)` in the panel's axis coordinates, and
serves EVERY 2-D panel (a camera frame, a 2-D parameter scan, …).

1. **Frontend** (`PanelEditor._read_region`): hands the endpoints, unchanged, to
   the producing source's `region_to_acquisition_parameters(...)` and fills the
   named Edit fields. No device shape, ever.
2. **Acquisition layer** (the logic node): speaks PLOT coordinates. `CameraMeasurement`'s
   spatial acquisition parameter is `region = [x_min, x_max, y_min, y_max]`
   ENDPOINTS — NOT the device `[x,w,y,h]` — exposed by `acquisition_parameters()`
   and accepted by `set_acquisition_parameters(region=...)`. (A 2-D-scan source
   would expose its axis ranges the same way.) The endpoint→device-ROI conversion
   is hidden INSIDE `set_acquisition_parameters` (`region → [x, w, y, h] →
   camera.configure(roi=...)`). So the Edit box shows endpoints matching the
   selector; everything in the acquisition/measurement layer stays plot-shaped.
3. **Device layer** (`QCMOSCamera`): adapts to hardware. SNAPS the ROI to the
   camera's sub-array grid (`SUBARRAY*` must be **multiples of 4** — query
   `prop_getattr` step/max), writes it in the safe order (positions→0, then sizes,
   then positions, `SUBARRAYMODE` ON last), and READS BACK the applied window via
   `prop_setgetvalue`; `camera.roi` reports that read-back. So the node's `region`,
   the 2-D panel axes and the Edit `now:` all reflect what the camera truly images.
   An unchecked write would be silently clamped → wrong region (the bug this fixed).

`VirtualCamera` mirrors the device layer: `configure(roi=...)` snaps (shared
`devices.base.snap_subarray`) and `acquire()` CROPS to it, so a virtual test runs
the SAME ROI path real hardware does (default `roi=None` = full frame). The console
`_coord_frames()` reads the node's `region` endpoints (index 0 = x_min, index 2 =
y_min give the 2-D axis origin). **Refresh** (`PanelEditor.rebuild`) first ticks
the console (`refresh_once`) so the snapshot mirrors the MOST-RECENT hub frame, not
the last timer-tick render.

`prepare` drives SAFE, packs + uploads the BRAM image over JTAG-to-AXI, arms the
scan banks, then drives LOAD (rising-edge COMMAND, waits `STATUS_LOADED`); it does
not start. `fire` drives FIRE; only the synchronized rising edge is a start event.
After that, the FPGA clock owns edge timing. `wait_done` polls `STATUS` (and
stream-refills the freed scan bank behind `CURSOR` for streamed scans);
`safe_state` drives SAFE. The COMMAND word is cleared to 0 before each command for
a clean rising edge.

Streaming refill: for scans larger than the resident 2-bank window the host
refills the freed bank behind the cursor with the next chunk; the engine only
advances into a bank when `BANK_READY` AND that bank holds the right chunk, so a
late refill STALLs (`STATUS_UNDERFLOW`), never a wrong point. `repeat_forever`
re-sweeps a streamed scan via a host background refill thread that supplies chunks
CONTINUOUSLY and CYCLICALLY (chunk `(mono%K)` into bank `mono%2`, one-ahead) -- the
sweep wrap is just another chunk boundary, so the re-sweep is SEAMLESS for any N
(`scan_bank_base` toggles by `K&1` so chunk 0 lands in the alternating bank).

## 4. N-Slot Scan Model

Scans bind named slots `s0, s1, ...` in bind order. **There is no `x`/`y` axis concept** — any
per-field value is bound to a slot, never to a fixed `x`/`y` channel.

- Any scannable per-field value (period duration, analog-bus DAC value) can
  be bound to a named scan **slot** `s0, s1, ...` in bind order, via the GUI
  scan dot or `state.bind_field(kind, target)` where `kind in
  {"duration","dac"}`. A channel delay is a FIXED per-channel output delay
  line: it takes an API slot (`aN`) but never a scan slot (`FIELD_KINDS` in
  `timing/pulse_table.py` is the single source; `bind_field("delay", ...)`
  raises).
- `scan_table` is an `N_points x N_slots` array, loadable from `.npy/.csv/.txt`
  (`load_scan_table`) or built in the GUI Scan tab. Row = one scan point, column
  `j` = slot `s{j}` in that slot's display unit (ns for time slots, integer DAC
  code for `dac` slots).
- Host compiles to an affine edge template plus a streamed scan-point table:
  `compile_pulse_table_scan_runtime_program` (sequencer.py). The FPGA evaluates
  `effective_tick = base + (sum_j coeff_j * slot_j) >>> COEFF_FRAC_BITS` and
  iterates scan points seamlessly. `RuntimeSequenceProgram` schema carries
  `slot_count/slot_kinds/tick_slot_coeffs/loop_end_slot_coeffs/scan_points/
  scan_coeff_frac_bits` plus ticks/masks/bus_segments.
- Time expressions in durations may only be affine in slots (numbers,
  `s0..`, `+ - * /`, parentheses); the compiler rejects non-affine scan timing.
- Duration and DAC value can scan in any combination and seamlessly: the
  analog-bus segments carry affine start/stop ticks (same `effective_tick`) and a
  dual `value_select` so a ramp can scan BOTH endpoints (scanned-A -> scanned-B)
  and an edge/hold segment can track a scanned DAC code.
- The host validates global effective-tick monotonicity before upload
  (`validate_pulse_streamer_program`); a scan that reorders the merged edges is
  rejected, not silently dropped.

Anti-patterns: do not expand a scan grid into many GUI columns or many prepared
pulse tables when one ordered `scan_table` describes it; do not re-introduce
separate `x_array`/`y_array` objects.

Snap-to-tick (single source). The clock can only land on whole ticks (>= 20 ns),
so literal time values are snapped to the grid, and there is exactly **one** snap
source: `PulseTableState.snapped()` (`timing/pulse_table.py`). Rule: period
durations floor to `>= 1` tick (a duration never collapses to zero — e.g. 5 ns ->
20 ns); channel delays and scan-point values round to the nearest tick (ties away
from zero, sign preserved); DAC scan points round to the nearest integer code; and
slot **expressions** (`"s0"`, `"20+s0"`, anything non-numeric) are preserved
literally — the compiler snaps their affine base instead, so bindings are never
corrupted. It never raises (it auto-snaps), mirroring the confocal
`align_to_resolution`. The same grid rule is applied on both ends so what the user
sees and what the hardware runs always agree: the pulse-transfer API snaps the
whole state once via `snapped()` in `sequencer.timing_payload_to_dict`, and the GUI
applies the identical rule field-by-field through the `align_to_resolution`-backed
resolution widgets (`pulse_gui.py` `set_resolution`) plus `snap_scan_table`, all of
which share the `quantized_time_steps` floor/round-to-nearest logic in
`timing/pulse_table.py`.

## 5. Frontend Fluent Rules

Source of truth is the historical Confocal GUI Fluent layer under
`references/source_archives/Confocal_GUIv2_refactored_v6/...`. Reuse
`Zou_lab_control/frontend/qt_fluent.py`; do not create one-off Qt styles per GUI.

Layout primitives that structurally prevent cutoff/overlap (use these instead of
hand-tuned fixed geometry):

- `Metrics` — scaled spacing/size tokens (`margin/gap_row/gap_item/gap_tight/
  row_h/dot`). Read them at use time; they track the active DPI scale.
- `measure_text_width(texts, ...)` — content-driven label-column width.
- `ElidedLabel` — elides with `...` and exposes full text as tooltip.
- `FluentScanDot` — round per-field toggle: hollow grey when unbound, filled
  orange with its 1-based slot number when bound.
- `mark_scan_field(widget, bound=...)` — applies the orange + disabled look to a
  field bound to a scan slot.
- `FluentLabeledField` / `FluentFormGrid` — `label : widget` rows with a shared,
  aligned label column and one row height, so stacked forms line up without
  per-row fixed geometry.

Use `set_fluent_scale()` / `scaled_px()` for fixed geometry; do not hard-code a
second scaling system. Use `FluentComboBox/FluentSpinBox/FluentScrollArea` so
popups, counters, and thin scrollbars share styling. Closed combo boxes ignore
wheel events.

Visual rules: Segoe UI 12pt, background `#F3F3F3`, text `#323130`, accent
`#77AADD`, radius 4px, flat 1px `DIVIDER` card borders. `FluentGroupBox` is the
white card with grey title pill — delineated by a painted 1px border, NOT a drop
shadow (the soft shadow re-rasterised + blurred the whole card on every paint —
~250ms of a 35-site grid card's cold load at 3× dpr — so it was removed). Keep the
flat card border a sealed construction detail; do not replace the editor with plain tables.

### Pulse GUI Layout Contract

`PulseSequenceEditor` has **Edit / Preview / Scan** tabs.

- Edit: `Channel Names and Duration`, `Delay & Scan`, horizontal period-card
  timeline (not a grid), bottom `Control Buttons`, `Channel View`. The
  channel-name column, delay column, and period checkbox columns share **one**
  vertical scroll. Period cards may scroll horizontally; they must not have
  independent vertical scrollbars. For the same channel, raw-label, delay edit,
  and first period checkbox must share `mapTo(editor, QPoint(0,0)).y()`.
- A dot next to a per-field value cycles its binding. A **duration** or **DAC**
  field cycles none -> scan (`sN`, orange, value HIDDEN/disabled) -> api (`aN`,
  violet, value KEPT and still editable) -> none. A **channel delay** field is not
  scannable, so its dot cycles none -> api (`aN`, violet) -> none only. A scan slot
  is swept from the `scan_table`; an API slot is a named handle several fields can
  share that `set_api(name, v)` / `state.aN = v` set BY NAME without rewriting the
  field (e.g. both long frames of the imaging bracket share `a1`). The Scan tab
  lets the user write Python that assigns an `N_points x N_slots` array to a
  `scan_table` variable (namespace has `np`, `math`, `n_slots`).
- Preview reuses `frontend.plot(..., kind="pulse")`, redraws on tab open / `Show
  off rows` toggle (no manual refresh button), plots the **unexpanded** period
  table, y-axis labels are channel display labels (no `Pulse` title), repeats
  use `×∞` / `×N` (never the literal `inf`), slot-bound regions are drawn as
  spanning translucent markers, analog-bus rows are hollow stair-steps.
- Saving writes the bundle together: pulse `.json` + preview `.png` +
  `<stem>_program.json` (the compiled runtime program, wire domain) +
  `<stem>_scan.npy` (when a scan table exists).  Loading accepts any bundle
  member and redirects to the pulse `.json` (picking `<stem>_program.json` /
  `<stem>_scan.npy` by mistake is handled).
- Raw left column shows package pins (`M17`, `M13`) when the XDC map is
  available; `chNN` stays in tooltip and saved/API state. Hiding is a view op;
  `Hide Off` hides channels with no period on; clearing a channel turns its
  period states off but preserves label/delay.

### Screenshot QA

After `show()` or any state change, run the event loop and wait before `grab()`:

```python
editor.show()
editor.grab_screenshot(path, settle_ms=1000)   # preferred helper
# or: app.processEvents(); QtTest.QTest.qWait(1000); app.processEvents(); editor.grab().save(path)
```

Prefer native Windows Qt screenshots; offscreen captures can miss text, so also
run object-level checks (button text fits, geometry, state, `show_all_channels`
keeps the full list, `On Pulse` prepares-then-fires, `Stop Pulse` calls safe
state). Capture default visible channels, all XDC channels at scroll top, and
all XDC channels mid-scroll. Verify both the inner editor and the `FluentWindow`
wrapper.

## 6. FPGA: Hardware, Capacity, RTL

Target FPGA is **Xilinx Artix-7 35T `xc7a35tfgg484-2`** (FGG484). Approx class:

```text
logic cells: 33,280   CLB LUTs: 20,800   CLB FFs: 41,600
BRAM: 1,800 Kb (50 x 36 Kb)   DSP: 90
```

RTL: `fpga/pulse_streamer/zlc_edge_streamer.v` (the engine, `parameter NUM_SLOTS`,
`COEFF_FRAC_BITS=8`, `RD_LAT=2`, `FIFO_DEPTH=4`) and
`zlc_pulse_streamer_top.v` (top, `NUM_SLOTS=4`, 62 channels + 4x10-bit DAC buses,
`axi_bram_ctrl` + CTRL regfile + region-decoded BRAMs, no VIO). The host-side
image packer + cycle-accurate engine model are in `fpga/pulse_streamer/host/`
(`image.py`, `engine_model.py`); `infer_xdc_*` + `validate_pulse_streamer_program`
stay in `Zou_lab_control/neutral_atom/devices/fpga_pulse_streamer.py`.

Default profile (from `host.image.StreamerParams` / `solve_capacity` on the 35T):
`CHANNEL_COUNT=62`, `MAX_EDGES=4096`, `BANK_SIZE=2048` (4096 resident scan points,
UNBOUNDED via streaming), `NUM_SLOTS=4`, `TICK_WIDTH=32`, `COEFF_WIDTH=16`,
`COEFF_FRAC_BITS=8`, `RD_LAT=2`, `FIFO_DEPTH=4`, `CLOCK_HZ=50e6`. The edge tables
live in three parallel block RAMs (tick 32b / coeff 64b / mask 62b, forced
`READ_LATENCY_B=2`); the scan window is one BRAM; the bus segment tables stay in
LUTRAM (distributed) because the bus/ramp engine reads them combinationally each
tick. Vivado `report_utilization`/`report_timing_summary` are the final authority;
the Python estimate is a budget guide (RAMB36 78%, LUT 26%, FF 12%, DSP 9%).

The minimal pulse width AND resolution is **1 tick (20 ns)**: a depth-`FIFO_DEPTH`
(=`RD_LAT`+2=4) continuous edge prefetch hides the read pipeline (issue->data-valid =
`RD_LAT`+1, including the registered `edge_raddr`), so
back-to-back 1-tick edges fire one per clock. Four gapless reload sites
(start / loop-rewind / scan-advance / repeat) reseed the FIFO with `FIFO_DEPTH`
shadows at every boundary, so the last edge of point k and the first edge of point
k+1 are adjacent with no gap. Cycle behavior is proven by
`host.engine_model.rtl_mirror_play == reference_play` at read latency 1/2/3 + 200
fuzz programs, plus the xsim testbenches in `fpga/pulse_streamer/sim/` running the
real RTL.

Build/program workflow on the Vivado computer:

```powershell
fpga\build_and_program.bat --check     # no-board HDL synth + capacity self-check
fpga\build_and_program.bat             # build + program (create_project.tcl -> program_fpga.tcl)
fpga\run_server.bat --check-config     # print resolved project/bit/ltx/xdc/clock/capacity
fpga\run_server.bat                     # start persistent server (jtag-axi backend)
```

Configurable before build: `ZLC_PS_XDC`, `ZLC_PS_VIVADO_BIN`, `ZLC_PS_CLOCK_HZ`.
Capacity is fixed by `host.image.solve_capacity` (no per-build override). Vivado
2019 debug cores are path-length sensitive — keep the checkout short (`D:\ZLC`).
The printed `ZLC build root` / `ZLC project dir` are the source of truth for the
generated `impl_1\zlc_pulse_streamer_top.{bit,ltx}`; the default project is
`fpga\build\ps` (short name `ps` -> `ps.runs`, chosen so Vivado's deep
run/.Xil temp path stays under MAX_PATH while the build remains in-repo).

### Edge-table engine + JTAG-to-AXI streaming (the one design)

There is ONE design, no variants, no backward compat. The board target is the
global affine **edge-table engine** (`fpga/pulse_streamer/zlc_edge_streamer.v`),
fed over **JTAG-to-AXI**: the host packs the program into a BRAM image and writes
it through an `axi_bram_ctrl`; a CTRL register-file mailbox carries
COMMAND/STATUS + the streaming handshake. The engine has ONE edge pointer, so a
repeat / loop-rewind / scan-advance is a single-cycle pointer + shadow reload
(gapless), and it is cheap in LUTs (one comparator + one affine MAC, not per-
channel players). This matches how real pulse streamers are built (Swabian = one
global RLE stream + hardware loop; SpinCore PulseBlaster = one global instruction
stream + LOOP opcodes).

**1-tick prefetch.** The edge table is three parallel block RAMs (tick 32b /
coeff 64b / mask 62b, forced `READ_LATENCY_B=2` so `RD_LAT=2` is deterministic).
A depth-`FIFO_DEPTH` (=`RD_LAT`+2=4) continuous prefetch issues one BRAM read per
cycle and an "arm" FIFO holds the next edges, reseeded with `FIFO_DEPTH` shadows
at every boundary; that hides the 2-cycle latency so back-to-back 1-tick (20 ns)
edges fire one per clock. Four gapless reload sites: start / loop-rewind /
scan-advance / repeat.

**Unbounded streaming scan.** The scan window is a 2-bank ping-pong
(`BANK_SIZE`=2048 pow2 -> 4096 resident points) in one BRAM. The engine plays
point 0..N-1, addressing bank `(idx/BANK_SIZE)%2`, exposes `CURSOR`, and the host
refills the freed bank behind the cursor with the next chunk. The `BANK_READY` +
`BANK*_CHUNK` handshake means the engine only advances when the bank is ready AND
holds the right chunk, so a late refill STALLs (hold, `STATUS_UNDERFLOW`), never a
wrong point. `repeat_forever` re-sweeps a streamed scan via a host background
refill thread that streams chunks CONTINUOUSLY and CYCLICALLY (monotonic chunk
`mono` -> data `mono%K` into bank `mono%2`, one-ahead) -- the wrap is just another
chunk boundary, so the WHOLE re-sweep is gapless (no inter-sweep hold).

**Pieces (all in-repo, Python-verified + xsim-verified pre-hardware):**

- **Image layout / packer + capacity (`fpga/pulse_streamer/host/image.py`).**
  Single source of truth for the host<->FPGA AXI write contract AND the geometry
  the RTL localparams + create-project tcl derive from. `pack_program` lays out a
  CTRL regfile (magic, COMMAND/STATUS mailbox, scalars, `CURSOR`, `BANK_READY`,
  `BANK*_CHUNK`) + TICK/COEFF/MASK edge BRAMs (read in parallel) + the 2-bank SCAN
  window + the BUS segment image; `unpack_program` is the decoder; `pack->unpack ==
  program` is the round-trip contract. `solve_capacity(part, channel_count)`
  re-derives `max_edges`/`bank_size`/addr widths from the part's RAMB36/LUT budget
  at `<=target_pct` (default 90); the 35T resolves to **4096 edges + bank_size 2048
  (4096 resident) + UNBOUNDED streaming, RAMB36 78%**.
- **Engine model (`fpga/pulse_streamer/host/engine_model.py`).** Cycle-accurate
  Python mirror. `rtl_mirror_play == reference_play` at read latency 1/2/3 + 200
  fuzz programs proves the 1-tick prefetch; `streaming_scan_play` proves the 2-bank
  ping-pong + late-refill STALL; `bus_play` proves the bus/ramp engine incl.
  scanned ramp endpoints. The second pre-hardware layer is the xsim testbenches
  in `fpga/pulse_streamer/sim/` running the real RTL (real BRAM IP netlists where
  it matters).
- **Top (`zlc_pulse_streamer_top.v`).** `jtag_axi` -> `axi_bram_ctrl` -> region-
  decoded BRAMs (3 parallel edge + scan + bus image) + CTRL regfile, driving the
  engine; engine `out` (62) + `bus_out` (4x10b DACs) go to the board pins.
  `create_project.tcl` builds it (`zlc_force_latency2` forces the edge BRAMs to
  `READ_LATENCY_B=2`); `program_fpga.tcl` leaves the `jtag_axi` core discoverable
  as a `hw_axi`. Structure is contract-tested (`test_final_top_regions_match_image_*`).
- **Host session (`devices/axi_session.py :: VivadoAxiStreamerSession`).** One
  persistent `vivado -mode tcl` hw_axi session. `prepare` SAFE + pack + upload +
  arm banks + LOAD (waits `STATUS_LOADED`); `fire` FIRE (and starts the background
  refill thread for streamed/repeat scans); `wait_done` polls STATUS and refills
  the freed bank behind `CURSOR`; `safe_state` SAFE. COMMAND is cleared to 0
  before each command for a clean rising edge. The Tcl executor is injectable, so
  the full pack -> upload -> LOAD -> fire -> stream flow is unit-tested without
  Vivado (`test_vivado_axi_session_*`). `run_server.bat` default backend is
  `jtag-axi`; `build_and_program.bat` builds + programs the bitstream (project
  `fpga/build/ps` -- the short name "ps" keeps Vivado's deep run/.Xil temp path
  under the Windows MAX_PATH limit while the build stays in-repo).

**Capacity (35T `xc7a35tfgg484-2`, target `<=90%`):** 62 digital + 4x10-bit DAC;
20 ns tick (1-tick min width AND resolution); `NUM_SLOTS=4` affine slots
(duration/DAC value, any combination, seamless; channel delays are fixed
API-set values, never scanned); **4096 edges + 4096
resident scan points + UNBOUNDED streaming**; RAMB36 78% (LUT 26%, FF 12%, DSP
9%) from `solve_capacity`. The bus segment tables are LUTRAM (distributed), not
RAMB36, because the bus/ramp engine reads them combinationally each tick.

## 7. AXI4 Burst Upload (transport architecture)

The host<->FPGA transport is JTAG-to-AXI: `jtag_axi_0` (a Vivado AXI master driven
from Tcl over the JTAG cable) -> `axi_bram_ctrl_0` -> the region-decoded BRAM image
behind `zlc_pulse_streamer_top.v`. The architecturally important fact is that both IP
are configured as **full AXI4, not AXI4-Lite**.

Why this matters. Over JTAG-to-AXI a single-beat write costs roughly 10 ms (the cost is
the JTAG round-trip, not the transfer). AXI4-Lite has no burst, so every 32-bit word is
one transaction; a 4096-edge program is several thousand words, i.e. a multi-second
upload on every `On Pulse`. Full AXI4 lets the master issue an INCR burst of up to 256
beats (AWLEN max) per transaction, so an address-contiguous run of words moves in one
round-trip. A complete 4096-edge image then uploads in ~100 ms.

Configuration source of truth (`fpga/pulse_streamer/create_project.tcl`):
`CONFIG.PROTOCOL {AXI4}` on both `jtag_axi_0` and `axi_bram_ctrl_0`, with
`CONFIG.M_AXI_ID_WIDTH {1}` / `CONFIG.ID_WIDTH {1}` matched, 32-bit data/address, and
`SUPPORTS_NARROW_BURST {0}`. The top (`zlc_pulse_streamer_top.v`) wires the master and
slave 1:1 including the burst sidebands (`awid/awlen/awsize/awburst/awlock/awcache/wlast`
and the read mirror). A drift back to `AXI4LITE` (or dropping the burst sidebands) is
silent: synthesis still succeeds, but `-len N` is ignored and uploads return to seconds.
The contract test asserts `CONFIG.PROTOCOL {AXI4}` is present and `{AXI4LITE}` is absent,
and that `m_axi_awlen`/`m_axi_awburst`/`m_axi_wlast` (master) and `.s_axi_awlen(`/
`.s_axi_awburst(`/`.s_axi_wlast(` (slave port map) are wired.

Host side (`Zou_lab_control/neutral_atom/devices/axi_session.py`,
`VivadoAxiStreamerSession`). Word writes are queued as `(byte_addr, value)` and coalesced
at flush:

- `_burst_runs` walks the **pending list in insertion order** (never globally sorted) and
  merges only strictly address-contiguous entries (stride 4), capped at `burst_max`
  (default 256, clamped to [1, 256]). Order is load-bearing: a COMMAND rising edge is two
  writes to the *same* address (0 then cmd), and a `BANK_READY` de-arm/re-arm pair is two
  writes to the same address; both must stay ordered single-beat writes, so they are
  never merged or reordered.
- **4 KB boundary split (AXI4 IHI0022 A3.4.1).** A single AXI burst must NOT cross a
  4 KB (4096-byte) address boundary, and `create_hw_axi_txn -burst INCR` does NOT
  auto-split, so `_burst_runs` ALSO stops a run at every 4 KB boundary
  (`_AXI_BURST_BOUNDARY_BYTES`) — a run's beats all stay in one 4 KB page. This was a
  **real-hardware scan glitch**: the dense scan region (4 words/point, points adjacent)
  base sits 256 B into a 4 KB page, so an un-split 2000-point bank upload emitted 7
  boundary-crossing bursts; the swept value jumped at scan point **240** then every
  **+256** (`4096 / (4 * num_slots)` points — the only 256-scan-point period in the
  system; bank_size=2048 pts and burst_max=64 pts do not match). Because a page is a
  multiple of `burst_max` beats the split self-realigns, so it costs ~0 extra
  transactions. The split is byte- and order-preserving (same words, same addresses,
  only transaction grouping changes); `test_axi_burst_4kb_boundary.py` guards it (real
  scan geometry: 7 crossings before, 0 after). Topology note: the on-chip path is a
  *single* `axi_bram_ctrl` with no interconnect, whose flat BRAM address counter would
  carry linearly across a 4 KB boundary — so if the glitch ever survives this fix on
  real hardware, suspect the `jtag_axi` master / `create_hw_axi_txn` burst framing, not
  the fabric.
- `_write_burst_tcl` emits one `create_hw_axi_txn ... -len N -type write -burst INCR`. The
  `-burst INCR` is explicit because the Vivado default burst type is not guaranteed INCR
  across versions and FIXED would write every beat to the base address (silent
  corruption). For a multi-word burst the `-data` argument is **one concatenated hex value
  whose least-significant (rightmost) word lands at the base address** — so the per-beat
  words are emitted high-address-first, i.e. the contiguous values are concatenated in
  REVERSE. This byte order is the one easy silent failure mode.
- `_flush` sends several bursts per Vivado round-trip (`write_batch` bounds bursts, not
  words) to amortise the host<->Tcl latency.
- `axi_self_test` is a warm-start bring-up check: it burst-writes a known ramp into the
  scan-BRAM region, reads it back single-beat, and raises if it does not match. This
  catches exactly the two silent faults — wrong burst `-data` byte order, or a still-Lite
  bitstream that ignores `-len` — before any real pulse upload.

Streaming bound. A scan with more than `2*bank_size` points does **not** make the upload
grow without bound. The engine plays a 2-bank ping-pong window; the host streams — it
refills the bank behind the consumed `CURSOR` with the next chunk and re-arms
`BANK_READY` (see section 3 and 6). Total scan points are limited only by host memory, and
a late refill STALLs (`STATUS_UNDERFLOW`, hold), never a wrong point.

## 7b. UART Fast-Control Side-Channel (root fix for the ~1 s apply latency)

Vivado hw_axi over JTAG is a **debug** path: each register word / command / STATUS-poll is a
separate synchronous Python↔Vivado-Tcl↔JTAG round-trip (~10–20 ms of interpreter + JTAG-scan
overhead each), so one pulse apply is ~7–10 round-trips ≈ ~1 s — the cost is the **number of
transactions**, not the tiny JTAG payload. The UART side-channel removes Vivado + per-transaction
JTAG entirely: a whole ~24 KB program uploads in **~82 ms** at 3 Mbaud, a scan-point step in
**~sub-ms**. It is a **byte-identical transport swap** — it writes the SAME `region_bases` word map
from `image.pack_program`, so the engine + readout are untouched (safe to ship after sim, unlike a
calibration change).

Pieces (all share the ONE register map + `image.pack_program`):
- **Wire protocol** `fpga/pulse_streamer/host/uart_frame.py` — single source: SYNC `5A A5`, two
  opcodes only (WRITE run-of-words / READ), LE word address, CRC-16/CCITT-FALSE. COMMAND / scan-step
  / PING are **composed by the host** from WRITE/READ, so `image.py` stays the register-map source.
  `MAX_FRAME_WORDS=256` = the RTL frame-buffer depth.
- **RTL** `fpga/pulse_streamer/zlc_uart_bridge.v` — baud NCO, 8N1 RX, a decoder that BUFFERS a whole
  WRITE frame and verifies CRC BEFORE any write fires (a corrupt frame commits NOTHING), auto-increment
  word writes, a CTRL READ tap, and an 8N1 reply serializer (WRITE→ACK, READ→data). `zlc_pulse_streamer_top`
  MUXes its `(u_word_addr/u_wdata/u_we)` against the axi_bram_ctrl side BEFORE the region decode
  (`uart_sel = u_active`, priority mux — UART and JTAG never used together), so a UART write is
  byte-for-byte a JTAG write. The AXI/JTAG stack stays wired (bring-up/ILA/fallback).
- **Host** `devices/uart_session.py::UartStreamerSession` **subclasses** `VivadoAxiStreamerSession` and
  inherits the ENTIRE protocol (prepare/fire/wait_done/_command/_load_chunk/streaming/scan_progress);
  only `_queue_word`/`_flush`/`_read_word`/`start`/`close` change (frame `uart_frame` packets over an
  injectable serial transport — real `PySerialTransport` or `FakeUartTransport` backed by the RTL model).
- **Behavioural model** `host/uart_bridge_model.py` — the Python decoder oracle the RTL mirrors (the
  `engine_model.py` role for the bridge). **Primary, always-run verification** is
  `tests/test_uart_bridge_equivalence.py` + `test_uart_session.py`: `pack_program` → host UART encode →
  model decode == `pack_program` (byte-identical), and the real prepare/fire runs over `FakeUartTransport`.
- **Server** `sequencer_server.py --backend uart --uart-port COM3 [--uart-baud 3000000]` wires the same
  5 callbacks to a `UartStreamerSession` (virtual==real; transport swap is server-side only).

RTL is verified on the rig, not here (no Verilog sim in this repo — same as the AXI integration).
**Rig checklist (USER runs; we never run Vivado build):** (1) find the UART carrier — is the board's
FT2232 **channel-B** TXD/RXD routed to 2 FPGA pins? yes → reuse the existing USB cable; no → external
USB-UART on 2 spare LVCMOS33 pins. (2) fill `uart_rx`/`uart_tx` in `board.xdc` (commented placeholders).
(3) build + program (first bring-up at `BAUD=115200` to isolate wiring from baud-lock, then 3 Mbaud).
(4) `link_self_test` (scratch write+read-back over UART) + `READ(LAYOUT_ID)==0x5A4C4C02`. (5) time a
full apply (~<100 ms) and a step (~sub-ms); if apply >100 ms set the FT2232 latency-timer to 1 ms —
look at USB buffering, not the RTL (sim already proved the byte count is small).

## 8. Per-Channel OUTPUT Delay (TTL + DAC event schedulers)

A channel delay is a **physical OUTPUT delay**, not baked into the edge ticks:
`output_delayed[t] = output_undelayed[t - d]`, **zero before fire**, never disturbing
another channel; negative delays fold via the global shift `G = max(0, -min(delays))`.
The edge table is emitted UNDELAYED; the delays ride `channel_delays` (TTL) and
`bus_delays` (DAC) -- one 32-bit word per channel and per bus in the AXI DELAY
register region (`R_DELAY_BASE`, see `image.region_bases`).  `pack_program` writes
ALL delay words every upload (zero when undelayed) so no value from a previous
program can linger (the real-machine "DA inherits the last program's delay" bug).

**TTL = EVENT SCHEDULER** (`zlc_edge_streamer.v`): a TTL waveform is toggle-sparse, so
the engine queues TOGGLES instead of buffering one bit per tick.  When the undelayed bit
flips at tick `t`, it pushes `{t + d - 1, level}` into that channel's `EVT_DEPTH`-deep
(default 64) 49-bit LUTRAM FIFO; a free-running 48-bit `g_time` pops it by equality into
the output register (`d == 1` is one register; `d == 0` bypasses).  This is a **TRUE
physical delay**: `out[t] = in[t-d]` for every `t`, **silent for the first `d` ticks**
(first frame already correct), with **NO modulo / cyclic reduction** -- the old
`d % sweep_period` reduction (which played the first `floor(d/S)` sweeps early) is GONE.
Storage scales with toggles IN FLIGHT, not delay length: the bound is the 32-bit field
(~42.9 s at 20 ns/tick).

**Per-SLOT distributed-RAM FIFO (g_evtfifo generate loop) -- do NOT use a 3D reg array.**
Each delay slot owns its own 2D `(* ram_style="distributed" *) reg [48:0] fifo[0:EVT_DEPTH-1]`
with its own wr/rd/cnt/obit, instantiated in a `generate` loop; `evt_out` is the OR of the
per-slot contributions.  A single flat 3D array `evt_mem[slot][depth]` does **not** infer
as distributed RAM (each slot has an INDEPENDENT wr/rd pointer; a single shared write pointer
would let it infer), so Vivado falls back to flip-flops -- at the then-default depth 256
that was 18*256*49 = 226k FF + 256:1 read muxes, which the 35T cannot place (a real build
failed exactly this way).
The FIFOs are **COMPACTED** to the delay-eligible channels only (the 18 real TTL outputs,
`DELAY_COMPACT`/`NUM_DELAY_CH`/`DELAY_CH_MAP`); the 40 DAC-bus bits (pin driven by `bus_out`)
and the 4 `da_clk` pins do NOT get a FIFO, so the deep LUTRAM is paid only where it can be
used.  GUI greys out + the API rejects a delay on a non-eligible channel.

**Capacity contract (EXACT, full schedule, no modulo).**  An event pushed at tick `t`
occupies the FIFO until `t + d - 1`, so occupancy == the channel's toggle count inside its
own d-window.  `_check_delay_event_capacity` (fpga_pulse_streamer.py) reconstructs the
channel's UNDELAYED toggle stream over the WHOLE program -- every scan point at its
affine-shifted edge times, bracket loops, the repeat-forever wrap -- and takes the exact
maximum window count (a periodic-stream formula handles `d >= sweep period`).  A delay whose
in-flight count exceeds `EVT_DEPTH` is REJECTED with the longest physical delay reported;
nothing is silently dropped.  **DAC buses delay at the SEGMENT level** (`g_busseg`, replacing the
old per-DA-bit value FIFO): the engine captures each RESOLVED bus segment it applies -- an
edge/ramp descriptor `{emit = g_time + d, vstart, target, span, step, mode, frame_end}` -- into a
per-BUS FIFO (`sfifo`, `BUS_EVT_DEPTH` deep; `bus_count` of them), and a delayed re-player re-runs
that ramp `d` ticks later on the free-running `g_time` base.  So `out[t] = bus_value[t-d]` by
construction, and the buffer holds ONE descriptor per segment -- a dense `0->1012 over 500 ticks`
ramp is a SINGLE entry, not ~500 value events (the density asymmetry the old per-bit FIFO had is
gone).  The bus's `BUS_WIDTH` bits share one per-bus 32-bit delay (`del_bus_ticks`), so the DAC
delay range MATCHES TTL and a negative TTL delay's global shift G reaches the buses with no
mismatch.  Capacity = SEGMENTS in flight per bus <= `BUS_EVT_DEPTH` (edge/ramp = 1 each); a single
frame can never overflow (bounded by `max_bus_segments`), only a repeat-forever `d` spanning many
frames.  `bus_evt_fifo_depth` is reconfigurable from streamer_config.json; the per-bus segment
TABLE depth is `1 << bus_seg_addr_width` (`max_bus_segments`) which -- unlike `evt_fifo_depth` --
is NOT a passed generic, so the `.v` default must equal the config (a contract test pins it).

**streamer_config.json is the ONE geometry source (2026-07-09 refactor).**  Editing the JSON
propagates to the host, the bitstream, and the build with NO hand-edit, and geometry that would
overflow a fixed word/BRAM is rejected mechanically.  `fpga/pulse_streamer/host/image.py` is the ONE
computational authority (`StreamerParams` + `region_bases` + `build_ip_sizes` + `build_fingerprint`);
it PROJECTS the config into two derived artifacts, one per target:
- **`zlc_geometry.vh`** (`image.emit_geometry_vh`): every RTL geometry parameter default (incl. the
  `LAYOUT_FINGERPRINT`) as a `` `define ``.  `zlc_pulse_streamer_top.v`, `zlc_edge_streamer.v`, and the
  testbench `` `include `` it and default every geometry parameter to a macro (`= \`ZLC_...`) -- so no
  `.v` carries a hand-typed geometry literal or a hand-computed fingerprint.  There are **no `-generic`
  overrides any more** (the header replaced them).  Derived widths (`SCAN_ADDR_WIDTH`/`BUS_INDEX_WIDTH`
  come from `image` via the header; `COEFF_PORTB_BITS`/`MASK_PORTB_BITS` are `$clog2`-style localparams).
  The committed header is pinned to `emit_geometry_vh(default_params())` by
  `test_all_geometry_params_config_matches_rtl_defaults`; `build_and_program.bat` regenerates it in
  place from the active config (and hashes it) before synth.
- **`geom.tcl`** (`image.emit_geom_tcl`, via `build_ip_sizes`): the BRAM-IP sizes `create_project.tcl`
  sources -- busimg `Write_Depth_A`, axi_bram `MEM_DEPTH`, and the port-B widths are now DERIVED
  (power-of-two that covers the used words), never the old literals `2048`/`65536`/`64`/`128`, so a
  `bus_seg_addr_width`/`max_edges` bump can't silently overflow a fixed BRAM.

`check_rtl_assumptions(params)` is the ONE overflow gate (called at load / pack / both emitters):
`num_slots*coeff_width==64`, bus-flags word `<=32`, **`bus_count*(bus_seg_addr_width+1)<=32`** (the
single 32-bit `BUS_COUNTS` word -- a fingerprint-invisible overflow), `bank_size`/`max_edges` pow2,
**pow2 `evt_fifo_depth`/`bus_evt_fifo_depth`** (event-FIFO ring pointers), 32-bit `ttl_delay_max_ticks`,
`channel_count+bus_count<=delay_region_words`.  `coeff_frac_bits`/`slot_mul_width` reach the timing/
sequencer compilers + `engine_model` via the dependency-free `Zou_lab_control._streamer_geometry` seam
(mirrors `_clock`), so a config edit changes the emitted coefficients, not just the fingerprint.  The
single-source REFACTOR itself was byte-identical (it changed no synthesized value).  A later config
edit unified the delay depths to `evt_fifo_depth = bus_evt_fifo_depth = 64` (DAC in-flight segments
32->64, matching TTL); that IS a geometry change -> the shipped fingerprint is now `0x5AFC7CFB` (was
`0x5A87FD36`) and the bitstream must be rebuilt.  Each `params` field's meaning is documented in the
config's own `_field_docs` block.

Proven: `delay_line_reference` (out[t]=in[t-d]) is the unchanged ground truth;
`engine_model.rtl_delay_line_mirror` mirrors the scheduler cycle-exactly; the REAL RTL is
verified in xsim -- `tb_delay_sched.v` (delays {0,1,2,7,1000}, 1-tick toggles, repeat seams:
11,996 cycles, 0 mismatches, DELAY-SCHED-OK), `tb_delay_compact.v` (non-identity slot->channel
map, COMPACT-MAP-OK), `tb_evt_depth.v` (FIFO-depth boundary behavior pinned at a small
EVT_DEPTH=16 so the TB can actually hit the boundary, EVT-DEPTH-OK).  LUT is the binding axis on
the 35T: the g_busseg segment-delay build at EVT=128 / BUS_EVT=64 / bus_seg_addr_width=6 placed
OVER the device (~21958 of 20800 Slice LUTs), so the shipped config drops to EVT=64 (halves the
TTL event LUTRAM, ~-1.2k LUT) and bus_seg_addr_width=5 (halves the per-bus segment tables, ~-0.7k
LUT), bringing it to ~20.0k -- under the part with ~0.8k margin.  (The `estimate_resources` model
is calibrated to the OLDER 2026-06-29 routed build and does NOT yet include the g_busseg cost, so
its LUT% reads low; the real fit is confirmed by the on-bench rebuild.)

## 9. Pulse API, sync-to-device and GUI state semantics

`PulseController` (sequencer.py) is the notebook-facing API and shares the exact
sequencer path with the GUI: `on_pulse()/off_pulse()`, `set_channel_delay()/
get_channel_delay()` (delay calibration), `load_pulse()/save_pulse()`,
`set_scan_table()`, `synced_state()`.  **Both `prepare` AND `fire` record the SOURCE timing
as a syncable `PulseTableState` (always carries `periods`) in `last_payload_json`**, via the
single `_record_source_payload` seam shared by `SequencerService` and `VirtualSequencer`: a
GUI-authored table is recorded verbatim (names + scan/API slots preserved); a BARE
`PulseSequence` (a Task firing `to_sequence()`) is reconstructed into a period table via
`PulseTableState.from_sequence`.  So the GUI's **Sync** never sees a "raw payload it cannot
sync" -- whatever ANYONE fires (GUI, notebook API, Task) reloads into the editor.
`snapshot()` publishes it; `RemoteSequencer` flattens it across RPyC.  The GUI's **Sync**
button and `pulse.synced_state()` both read this single source of truth.

**API slots** (`pulse_table.ApiSlot`, parallel to scan slots): a NAMED handle (`a1`, `a2`...)
on a period duration / channel delay / DAC value that the API/Task sets BY NAME --
`state.set_api("a1", v)` or `state.a1 = v` -- WITHOUT rewriting the field (the number stays;
several fields can share one name).  In the pulse GUI a duration/DAC cell's dot CYCLES
none -> scan (`sN`, orange, value hidden) -> API (`aN`, violet, value kept) -> none; a delay
cell cycles none -> API -> none (delay is not scannable).  `read_state` carries API slots by
index (`_carry_api_slots`).  This is what lets the Calibrate task set the imaging template's
exposures by name without parsing "which period's which signal".  GUI state semantics (confocal style -- stars + status dot, never button base
colours): any edit while RUNNING/PREPARED adds the `*` suffix to On Pulse ("On Pulse*")
and turns the STATUS DOT orange (UNSYNCED); the button itself stays green so it cannot
be confused with the permanently-orange Remove/Load/Sync.  The star is present in every
run state except RUNNING-in-sync ("pressing would apply something new").  The debounced
summary pass compares the state key against the applied key and restores the run state
if the edit was reverted.  Save shows "Save*" + yellow while dirty, "Save" + accent when
clean; Add Bracket is accent (yellow is reserved for Save-dirty).

## 10. Building The Manuals

```powershell
python -c "from Zou_lab_control.neutral_atom.notes import build_main_manual, build_fpga_manual, build_device_manual; build_main_manual(); build_fpga_manual(); build_device_manual()"
python -c "from Zou_lab_control.frontend.notes import build_frontend_manual; build_frontend_manual()"
```

Each builder generates example figures into `assets/`, fills the `.texbody`
template, assembles the `.tex` wrapper **in memory**, and runs `render_tex_pdf`
(XeLaTeX, 2-pass, in a temp dir) with `assets=<dir>` so figures resolve. XeLaTeX
must be on PATH (or pass `xelatex=`). **Only the `.pdf` lands in `docs/<manual>/`**
(plus the committed `assets/`); no `.tex`/`.sty`/`.aux`/`.log`/`.toc`/`.out` is
ever written there (`docs/**/*.tex` and `*.sty` are gitignored). A failed build
leaves only a `.build.log` next to the target PDF. `render_tex_pdf(tex, out_pdf)`
accepts a tex **string** or a `.tex` **path**; `write_notes_tex` /
`compile_notes_pdf` are legacy in-place helpers kept for debug, NOT on the build
path. See §18.

## §18. frontend public-API seal (the visual-design contract)

The frontend exposes a **small, sealed** surface: callers pass DATA, the frontend
owns ART/GEOMETRY/dpi/typography. The **single authoritative, numbered statement
of the rules — plus the failure history that motivates each — is
`Zou_lab_control/frontend/AGENTS.md`** (do not re-list the rule text here; it
drifts). `frontend/__init__.py`'s module docstring is the in-code pointer to it.
Mechanically enforced by `tests/test_frontend_plot_contract.py` (every plot is a
`BaseLivePlot`), `tests/test_frontend_layout_uniformity.py` (one form system) and
`tests/test_frontend_smoke.py` (sealed-kwargs rejection, `DEFAULT_STYLE`
read-only, scale parity).

## 11. Verification

Tests are owned by another agent; see `tests/README.md` for the scoped matrix.
Prefer the smallest scoped check that covers the edited boundary; use full
`pytest -q` only for broad handoff. Typical doc-adjacent checks:

```powershell
pytest -q tests\test_frontend_smoke.py -k "render_tex_pdf or pulse_gui"
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract or scan"
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
git diff --check
```

## 12. Framework Review (architecture assessment)

A whole-framework review was done after the scan-redesign work. **Verdict: no
large rewrite is warranted.** The layering (`devices` contract / `timing` truth
/ `core` algorithms / `subsystems` capability bundles / `views`+`frontend`
plotting) is clean, and the virtual / command / jtag-axi sequencer paths share
one prepare/fire/wait_done/safe_state surface. The remaining items are targeted
robustness/clarity improvements, not redesigns.

What is sound and should NOT be churned:
- The `BaseDevice`/`CameraDevice`/`SequencerDevice`/`TrapArrayDevice` contract,
  the JSON+`$device:` registry, and the `DeviceSet` container.
- `PulseSequence` (time truth) vs `PulseTableState` (GUI compile model) split.
- The N-slot affine scan engine (now also drives affine analog-bus ticks, so
  DAC value + duration scan together — see §4).

Targeted improvement backlog (priority order; none blocking):
1. **Sequencer class roles.** Five classes (`Virtual/Runtime/Manual/Remote/
   Verilog` Sequencer) serve distinct roles but the naming/role split is easy to
   confuse. Document the decision table (now partly in the device manual);
   consider a short `docs` table or clearer names. No behaviour change needed.
2. **Calibration ↔ device binding.** `TrapCalibration` lives on the session, not
   the camera; swapping cameras can leave a stale calibration. Low-risk guard:
   record `grid_shape`/`reducer`/`ordering` in `metadata` and validate against
   the trap array on `detect`. (`detect` already uses the stored reducer/radius,
   so the train/infer reducer mismatch is not reachable: `TrapCalibration.detect`
   is the single readout path.)
3. **Calibration schema version.** Add a `schema_version` to the
   `TrapCalibration` payload so old `.npz`/`.json` can be migrated safely.
4. **Exposure source of truth.** The sequence's probe width is the truth; assert
   the camera exposure matches it after `acquire` to catch silent drift.

These are recorded so future work is guided; the user explicitly accepts that
many concrete neutral-atom devices/experiments are not implemented yet — the
skeleton, contracts, and docs are the deliverable.

## 13. Config Single-Source, Robustness, Audit Fixes (2026-06-09)

### Single user-editable config: `fpga/board_config/streamer_config.json`
The reconfigurable, **compile-affecting** specifics (part, clock, edge/scan/delay/bus
geometry) now live in ONE JSON. `fpga/pulse_streamer/host/image.py` owns the loader
(`load_streamer_config` / `params_from_config` / `default_params` / `default_part` /
`default_clock_hz`) with a robust fallback to built-in defaults if the file is missing.
Re-sourced from it (no more scattered literals):
- `axi_session.DEFAULT_PARAMS` + `DEFAULT_RUNTIME_CLOCK_HZ`,
- `fpga_pulse_streamer.DEFAULT_*` validator constants (this fixed a real drift: the old
  `DEFAULT_MAX_EDGES=1024` was HALF the synthesized 4096),
- the capacity estimate.

`params` must match `zlc_pulse_streamer_top.v` localparams — editing the JSON does NOT
re-synthesize; it re-aligns host validation/estimation. `test_streamer_config_is_single_
source_for_host_geometry` guards that the config == the host constants == the shipped RTL.

### `estimate_resources.bat` (repo root, double-click)
Runs `python -m fpga.pulse_streamer.host.image --config ...` →
`check_config_capacity` → `format_capacity_report`: a LUT/FF/DSP/RAMB36 pass-fail table for
the configured part, exit 0 (fits) / 1 (over budget). `solve_capacity` and the config check
share ONE accounting model: `estimate_resources(params, part, target_pct)`
(`test_estimate_resources_matches_solve_capacity*`). `build_and_program.bat` calls the same
CLI for its pre-build estimate, with the configured `fpga_part`.

### Robustness to board / XDC / Vivado / part changes
- **Synthesis part** now honors `streamer_config.json`'s `fpga_part` (build bat exports it
  to `ZLC_PS_FPGA_PART`; `create_project.tcl` reads it raw — NOT via `env_or`, which
  path-normalizes). Moving to another Artix-7 retargets the build without editing `.tcl`.
- **Vivado discovery** adds a `for /d` glob of `C:\Xilinx\Vivado\*` / `D:\` (newest wins)
  after the fixed version list, so a future release in the default location is auto-found;
  `ZLC_PS_VIVADO_BIN` / PATH still override.
- **DAC/analog ports are auto-detected** from XDC label patterns `base[bit]` (≥2 contiguous
  bits) — verified the shipped 62-port board infers correctly (`da_clk0..3` are legitimately
  4 of the 62 channels, NOT spurious). Order-dependence of name-only XDCs is documented in
  `board_config/README.md`. All env knobs are tabulated there.

### Correctness bugs fixed (host-side; from the adversarial audit)
- `aligned_to_channels` dropped `clk_channels` → a clk-wired channel silently reverted to
  engine-driven on align. Now filtered+carried.
- `validate()` allowed a clk channel that is also a DAC-bus member (inferred OR explicit) →
  double-drive. Now rejected at the contract gate (`__init__`/`from_dict`).
- `snap_scan_table` silently truncated too-wide rows via `zip()` → now normalizes width
  first (raises too-wide, pads too-short).
- `compile_pulse_table_scan_runtime_program` didn't snap on a DIRECT call → a 0 ns scanned
  duration became a 0-tick period. Snap now happens inside the compiler, regardless of entry
  point. (Each guarded by a regression test.)
- `compile_runtime_program_for_payload`: bound slots + EMPTY table intentionally degrades to
  a static program (a run is never blocked); documented inline (a direct `compile_scan` still
  errors — the strict explicit-scan path).

### Signed DAC semantics + period names (2026-06-09; **REBUILD REQUIRED** — BUS_SAFE_VALUE)
The DAC driver is bipolar OFFSET-BINARY: wire code 0 = −FS, code 2^(B−1) (=512) = true 0 V.
- **User layer is SIGNED LSB** (−512..+511, 0 = 0 V): GUI value fields, `analog_bus_modes`
  entries, `set_bus_value`, ScanSlot dac nominals, scan-table dac columns, JSON.  Helpers:
  `pulse_table.bus_zero_code` / `bus_signed_range`; per-slot `scan_slot_dac_ranges()`
  (replaces `scan_slot_dac_maxes`; `snap_scan_table(dac_ranges=...)`).
- **Wire layer stays raw code** (`RuntimeBusSegment` values, program `scan_points` dac
  columns, validator, packers, RTL).  The signed→code (+2^(B−1)) conversion happens in
  exactly TWO places: `_pulse_table_bus_segments` (segment emit) and `point_slot_value`
  (scan column) — nothing else converts.
- **RTL idles at mid-scale** (`BUS_SAFE_VALUE = 1 << (BUS_WIDTH-1)` engine parameter):
  power-up initial, reset/CMD_SAFE clear, FIRE re-init, and the delayed-read gate all use
  it, so an undriven DAC outputs 0 V — never −FS.  Mirrors updated (`bus_play`,
  `bus_value_at`, `bus_delay_line_reference`/ring mirror default safe 512).  An untouched
  bus = all-hold plan → NO segments emitted and member bits stay 0 (the "unused" marker).
- **Even code count**: 2^B codes have no exact middle; convention 0 V = 2^(B−1), so the
  signed range is asymmetric by 1 LSB (−512..+511).  Preview places the 0 V dashed
  reference mid-row and draws negatives below it; the trace dict carries min/max.
- **Period names**: each PeriodCard has an editable name field (below the unit combo so the
  cross-panel Duration alignment is unchanged; PANEL_TOP_HEIGHT 152→178); the card title
  keeps "Period i/N".  `to_period` round-trips the name (it used to be dropped);
  `unrolled_bracket` copies carry it per-copy.

### RTL findings — RESOLVED 2026-06-09 (user-authorized fixes; **REBUILD REQUIRED**)
The three bring-up items from the adversarial RTL hunt are now fixed/guarded. The
`zlc_edge_streamer.v` change means the next hardware session MUST re-synthesize
(`fpga\build_and_program.bat`) — the deployed bitstream still has the old behavior.
- **U4 — delayed-output tail at `done`: FIXED IN RTL.** The state chain now has a
  `done`-but-emitting branch that keeps `bnd_delay_advance` high after the final tick, so
  the event schedulers' `g_time` keeps advancing: a channel/bus with delay `d` drains the
  events still QUEUED in its FIFO (its last `d` ticks of toggles) and settles LOW (those
  tail values are the rest state — `state_mask`/`bus_value_active` are cleared at `done`).
  Before the fix `g_time` FROZE at `done` and a delayed channel could hold a stale HIGH
  value for the ms-scale window until the host reacted. This realizes exactly the
  contract `rtl_mirror_play`/`delay_line_reference` always promised (out[t]=in[t-d] for the
  whole stream); `repeat_forever` was never affected (never reaches `done`). Locked by
  `test_pulse_streamer_rtl_advances_delay_rings_after_done` +
  `test_delay_tail_emits_after_done_contract`. NOTE: an agent-suggested gate on a
  fill counter was rejected — such a counter saturates on long runs, which would disable
  the fix exactly when it matters; the unconditional advance is safe (a new FIRE clears
  every scheduler's wr/rd/cnt).
- **B1/B2 — `da_clk0..3` = `out_final[28/39/50/61]`: the clk button wires these strobe pins
  to the FPGA clk.** (The former compile-time `_warn_idle_dac_clock_pins` warning -- fired when
  a `da_clkN` pin was driven-but-idle -- was REMOVED at the user's request: it was noisy and
  read as an inexplicable error.  Enabling the da_clk pins is still the user's responsibility
  via the GUI clk button.)
- **B1/B2 — REVISED 2026-06-11 (⚠️ needs bitstream rebuild): the strobe is now `~clk`, NOT
  `clk` — the "third DA value between two edge periods" race.** The earlier note said "BY
  DESIGN, no RTL change"; that was WRONG and missed a real source-synchronous output hazard.
  The 40 DAC data bits (`zlc_bus_out` → `da_bias_*`/`da_dipole`) are launched on `posedge clk`,
  so a DAC value CHANGES on the rising edge. With the strobe = plain `clk` the DAC latched on
  that SAME rising edge — coincident with the data transition AND (at a period boundary) with
  ~30 TTL outputs all switching — so a value change was captured half-old/half-new = a
  sporadic THIRD code. User-visible on `pulses/T.json` (da_bias_y steps −192→388 = code
  320→900): a third level appeared sporadically between the two edge periods, and a ~200 ms
  HOLD gap "fixed" it only by moving the DAC step off the busy boundary (a band-aid). FIX:
  the clk mux now drives `out_final[n] = clk_en[n] ? ~clk : out[n]`, so the DAC latches on the
  clk FALLING edge = the CENTRE of the data eye (~10 ns settled each side at 50 MHz) and the
  quiet half-cycle (nothing else switches there) → always captures the clean settled word, no
  gap needed, for every DAC and every transition. The latch interface is otherwise
  unconstrained in `board.xdc` (no `create_generated_clock`/`set_output_delay`); the
  half-period margin is what makes it robust at 50 MHz (add ODDR clock-forwarding + output
  constraints if the rate ever rises). Proven in `sim/tb_da_clk_phase.v` (the engine step is
  glitch-free tick-by-tick; a coincident latch captures a third code for realistic per-bit
  skew, the eye-centre latch never does). Locked by `test_top_has_per_channel_clk_mux`
  (asserts `~clk` and the "DAC LATCH PHASE" rationale comment, rejects plain `clk`).
- **U1 (superseded 2026-06-09) — ramp engine is now a multi-LSB Bresenham stepper.** The
  original engine moved at most 1 LSB/tick then snapped at `stop_tick`; the user ruled
  that out ("按照计算出来的 step 来尽量靠近 ramp"). The RTL now computes `step = Δ//span`
  and `rem = Δ%span` (a combinational BUS_WIDTH+1-bit restoring divider,
  `zlc_bus_ramp_divmod`, engaged only when span < Δ ≤ 2^BUS_WIDTH−1) and per tick moves
  `step` (+1 on remainder-accumulator carry), saturating AT the target. TIMING/AREA: the
  divmod is DEFERRED from segment apply to the FIRST stepping tick (`bus_ramp_steep`
  flag; `rem` parks Δ in between) — the divider reads registered operands (short path,
  off the LUTRAM-read/endpoint-mux cone) and is instantiated once per bus, not once per
  apply call site; the first tick provably cannot carry (accum = rem < span), so the
  output is bit-identical to dividing at apply —
  i.e. `value(k) = vstart ± floor(k·Δ/span)` for ANY slope, landing exactly on the
  target at `stop_tick`. Gentle ramps (Δ ≤ span) keep the historic carry-only path,
  bit-identical to before. Mirrors updated in lockstep: `engine_model.bus_play`
  (step/rem state), `engine_model.bus_value_at` (unified closed form `floor(k·Δ/span)`,
  drives the bus delay line), and the preview `pulse_table._analog_bus_value_at_tick`
  (same staircase in the signed user domain). Steep ramps remain ALLOWED for any
  duration (validator does not reject). **Bitstream REBUILD REQUIRED.**
- **T3 — edge-BRAM latency-2 force: BUILD-TIME HARD CHECK.** `zlc_force_latency2`
  (create_project.tcl) now READS BACK both register properties and `error`s out if either
  did not take (e.g. a future Vivado renames it) — a silent latency-1 BRAM would shift
  every edge a cycle early on hardware with no error anywhere.
- **B3/B4/U7 — parameterization traps: GUARDED.** Comment guards at the RTL call sites
  (`COEFF_BITS==64` cap assembly, 32b flags word, `scan_addr_of` bank concat) + a host hard
  gate `image.check_rtl_assumptions` called by `pack_program` (and surfaced as
  `streamer_config.json` load warnings): geometries the shipped RTL would silently corrupt
  (num_slots*coeff_width != 64, flags > 32b, non-pow2 bank/edges, tick_width != 32) cannot
  reach the FPGA.

### DRY done + remaining backlog
Done (safe, test-guarded): single `streamer_config.json` source; one `estimate_resources`
accounting model; one `sN` slot-ref parser (`pulse_table.is_slot_ref`/`slot_ref_index`, reused
by sequencer + GUI); `UNIT_TO_NS` imports the timing `UNITS_TO_NS`; `_channel_delays_list`
helper; deleted dead `BUS_SEGMENT_MODES`.
Backlog (deferred — larger/riskier, none blocking): unify `effective_tick` vs
`_apply_affine_ticks` (one narrowing-aware helper); thread `tick_ns` into `_check_delay_cap`
(drop the hardcoded 20 ns in the seconds hint); move `PulseTableState.bus_value`-style packing out of the
GUI; route NamesPanel/ChannelPanel rows through `FluentLabeledField` + a `set_field_locked`
helper; split the pure `_pulse_table_*`/`_affine_*` compiler block out of the 2.2k-line
`sequencer.py`. The cross-layer delay-depth/`coeff_frac_bits` constants remain test-guarded
mirrors (cross-package import direction); kept as-is.

## 14. 64-bit Tick Architecture Study (feasibility, design, verdict)

What "going 64-bit" would actually mean, what it costs on the xc7a35t, and whether it is
needed.  Baseline (current 32-bit ticks, estimate_resources on xc7a35tfgg484-2):
RAMB36 27/50 (54%), LUT 12,246/20,800 (58.9%), FF ~9,000 (21.6%), DSP 52/90 (57.8%).

**What 32 bits caps today.**  A 32-bit tick at 20 ns is ~85.9 s.  That bounds ONE frame
(a single scan point's duration), NOT total runtime: repeat_forever runs indefinitely,
sweeps can hold millions of points, and the experiment loop is unbounded.  The TTL delay
register field is likewise 32-bit; the host enforces a conservative default cap of
`(1<<31)-1` ticks (~42.9 s, `streamer_config.json` `ttl_delay_max_ticks`), and with
repeat_forever a longer delay is reduced modulo the sweep period anyway (§8).  Separately, the scheduler's free-running `g_time`
is 48-bit: it wraps after ~65 days of continuous uptime -- the one true long-run hazard.

**Design if needed (the key trick: split base from offsets).**
- Widen the BASE timeline only: edge base ticks, `tc` frame counter, `loop_end_tick`,
  comparators -> 64-bit.  Keep SCAN-POINT slot values and coefficients at 32/16-bit: a
  per-point affine OFFSET stays bounded (+-42.9 s per point is plenty), so the
  `coeff x slot` products and ALL 52 DSP mappings are untouched; only the final
  accumulate/add widens (LUT carry chains, not DSPs).
- Edge tick BRAM 4096x32 -> 4096x64: +4 RAMB36 (27 -> 31, 62% of budget).  Mask/coeff
  BRAMs unchanged.
- CTRL layout: `LOOP_END_TICK` becomes LO/HI (loop-end coeffs already are); the packed
  image's tick words double (pack/unpack/verify_upload/host mirrors updated in lockstep
  -- no compat needed, one rebuild).
- Prefetch shadows (sh_e0..e4) and comparators widen: ~+1k LUT, ~+1.5k FF.  A 64-bit
  carry chain is ~17 CARRY4 (~4 ns), comfortably inside the 20 ns tick -- no extra
  pipeline stage, the 1-tick playback contract is unaffected.
- While in there: widen `g_time` 48 -> 64 (+16 FF + comparator slice per channel,
  negligible) to remove the 65-day wrap.

**Projected totals**: RAMB36 31/50 (69% of the 45-block budget), LUT ~13.3k (64%),
FF ~10.5k (25%), DSP 52 (unchanged).  Verdict: FEASIBLE on the 35T with margin.

**Verdict / recommendation.**  Not needed now -- no experiment requires a single frame
longer than 85.9 s, and delays are covered by the mod-period reduction.  If a >85.9 s
frame ever appears, the split-width design above is the path (do NOT widen slot values:
that would triple DSP usage for nothing).  The cheap `g_time` 48->64 widening is worth
folding into whatever rebuild happens next if multi-month uninterrupted uptime becomes
a real operating mode.

## 15. Register-layout handshake (2026-06-11; root cause of the garbled-first-frame DA)

**Incident.** Commit 6d60e53 (08:44) moved `CtrlWords.CLK_ENABLE` 46->20 (delay_depth removal)
and required a bitstream rebuild.  The user's board still ran the 03:57 bitstream (layout v1,
clk mask at 46/47) when the 11:57 server session started with the NEW host -- the session log
(`fpga/build/state/vivado_axi_session.log`) proves it: the self-test ramp went to words 22..37
(the v2 scratch).  Consequences of the silent mismatch: the host wrote the clk mask into v1's
DEAD reserved words 20/21, the REAL clk mask (46/47) was never written or cleared (stale /
power-up value), so the `da_clkN` DAC strobes ran uncontrolled -- garbled, first-frame-vs-
steady-state-inconsistent DAC output on the scope while every simulated layer (engine xsim,
full-top xsim with the real IPs and the real command flow, host pack/flush) is provably
frame-identical when layouts agree.  A second simulated demonstration: packing with
`bank_size=512` against a top elaborated with the default 2048 sends the bus image into the
scan region -- the mini-loader copies zeros and ALL DA output is silently wrong.

**Fix (the handshake).** The top hardwires CTRL word 63 readback to `ZLC_LAYOUT_ID`
(32'h5A4C4C02 = layout v2; writes land in `ctrl_reg[63]` but are never read back).  The host
(`image.REGISTER_LAYOUT_ID`, `CtrlWords.LAYOUT_ID`) verifies it in `axi_self_test` (server
bring-up) and at every `prepare()` (after the pure-software validation, before the first
hardware write).  A mismatch -- e.g. an old bitstream returning power-up 0 -- raises
immediately with a "rebuild the bitstream + restart the server together" instruction instead
of silently mis-driving registers.  BUMP BOTH IDs on ANY CtrlWords/region change.  Locked by
`test_register_layout_handshake_rtl_matches_host` (RTL<->host id equality + pack never
occupies words 1/2/63) and `test_axi_session_refuses_mismatched_register_layout`.

## 16. Pre-delivery audit notes (2026-06-12)

**Duplicate-test purge.** `tests/test_neutral_atom_lightweight.py` carried a ~1300-line
duplicated region: 50 top-level functions (48 tests + 2 helpers) were defined TWICE with
byte-identical bodies -- the second definition shadowed the first, so the first copies never
ran (pure dead weight, no lost coverage).  Removed via an AST-exact pass; the collected test
count is unchanged.  If you ever see pytest collect fewer tests than `def test_` greps,
check for shadowed duplicates first.

**StreamerParams defaults are LOCKED to streamer_config.json.**  `bank_size` had drifted
(dataclass default 512 vs config 2048).  The runtime path was safe -- `axi_session` uses
`default_params()` (the config read) and the build uses `emit_geom_tcl` (same source), so
host and bitstream agreed at 2048 -- but every direct `StreamerParams()` user (tests, the
tb_t_ff image generator) silently packed a 512-bank register geometry.  Defaults now equal
the config and `test_streamer_params_defaults_match_config` pins every field.

**Known limitation: dense 1-tick edge bursts slip 1 tick at the prefetch handover.**
`tb_1tick.v` (real BRAM IPs): in a run of BACK-TO-BACK 1-tick-spaced edges, edges 7..10 fire
one tick late (the streaming prefetch takes over from the 5 ARM-preloaded shadows; the
self-healing `>=` fire makes up the slack inside the burst, nothing is dropped, and the next
burst after any idle gap is exactly on time).  Present since the dd5d72d prefetch fix (the
guarantee there is NO DROPPED EDGES; the TB prints the slip count rather than failing).
Real experiment pulses (us/ms periods) never produce 7+ consecutive 20 ns edges, so this is
a microbenchmark-only startup-window effect.  Eliminating it would need a deeper shadow
preload (FIFO_DEPTH+shadow growth + rebuild) -- deliberately NOT done before delivery.

## 17. Persistent-Vivado self-heal: the "debug core, must restart server" wedge (2026-06-12)

**Symptom (real machine, reported many times).** After any error (typically a rejected
delay) followed by a retry, On Pulse failed with a debug-core-flavoured error and NO pulse
could ever be applied again -- only a full server restart helped.

**Root cause: the restart machinery itself had never once worked.** Three independent
defects, all in `axi_session.py`:

1. **Stale-sentinel queue poisoning (the permanence).** `self._queue` was created once in
   `__init__` and SHARED across process generations.  When a session closed (one transient
   action timeout / broken pipe is enough), the dead generation's reader thread EOF'd and
   pushed its `None` sentinel into that shared queue.  The next transaction (on the freshly
   restarted Vivado) read the stale `None` and raised "process exited unexpectedly"; the
   retry killed ITS fresh process, whose reader enqueued the NEXT sentinel -- every retry
   consumed one stale sentinel and produced a new one, forever.  The session could never
   self-heal.  Fix: every `start()` creates a NEW queue, and the reader thread is bound to
   `(process, queue)` ARGUMENTS, never to `self` -- a dead generation can only poison its
   own queue.  Regression: `test_axi_session_self_heals_after_close_restart` (fails on the
   old code with exactly the observed error).
2. **`open_hw_target -jtag_mode on` fallback (the "debug core" message).** Raw JTAG mode
   does not enumerate debug cores, so after a reconnect that took this fallback,
   `get_hw_axis` came back empty and init failed with the "No JTAG-to-AXI core" error on
   every attempt.  Fix: retry a plain `open_hw_target` after `close_hw_target` + a 2 s
   settle; never raw mode.  Pinned by `test_axi_session_init_tcl_never_uses_raw_jtag_mode`.
3. **Teardown races.** A stream-refill transaction whose stop event is set now aborts
   BEFORE `_ensure_process` (a dying stream thread could otherwise spawn a competing Vivado
   fighting over the JTAG target); `close()` joins the stream thread for 2 s only (it may be
   blocked on the `_io_lock` WE hold on the failure path); `_stop_stream_thread` never
   self-joins (close() reached from inside the stream thread's own timeout).  `start()`
   clears `_pending` so a dead transaction's queued words never leak into the new session.

Also: `action_timeout` default 30 s -> 120 s (one transient slow JTAG transaction must not
tear the session down), and `RemoteSequencer.open()` drops a CLOSED connection and
reconnects (a dead conn object used to be returned forever, so even restarting the server
did not revive an open GUI/notebook).

**Earlier diagnosis was incomplete:** commit 87ea492 fixed a real `_io_lock` re-entrancy
deadlock in the same path, but the restart it unblocked then ALWAYS failed on defect 1 --
which is why the "fixed" bug kept coming back on the real machine.

## 19. Generic scanned-measurement abstraction + release-recapture thermometry (2026-06-14)

Design doc: `docs/task_console_design/` (Measurement + one-key-temperature chapters).
This section is the maintainer-side condensate.

### The one engine behind every live scan

A "scanned measurement" = sweep one bound pulse parameter, acquire a few camera
frames per point, reduce those frames to a number (or one number per site) that
becomes the live curve's y. Detection-time/fidelity and release-recapture
temperature are the SAME shape; only three roles differ, so each is a role, not a
new scan loop (`operations/measurement.py`):

- **`ScanAxis(slot, values, label, unit, kind)`** — WHAT is swept: a named slot
  (`duration` set via `pulse.set_time`, taking ns at the wire; or `dac` set via
  `pulse.set_slot`, signed user value). `values`/`unit` stay in the user unit;
  `scale_to_ns` converts.
- **`ShotPlan` (protocol)** — HOW MANY frames + how the per-point sequence is
  built (`frame_sequence` only). `NFramePlan` = N independent frames, one trigger
  each (detection-time). `ReleaseRecapturePlan` = one acquire returns `[f0, f1]`.
- **`PointReducer` (protocol)** — HOW frames + calibration become y.
  `OtsuFidelityReducer` (pool `calibration.signals`, otsu split, gaussian-split
  fidelity). `SurvivalReducer` (occupancy compare on two frames).

Engine: **`ScannedMeasurement(pulse, camera, sequencer, calibration, axis, plan,
reducer, shots_per_point)`**. `measure(value, index)` = `axis.apply` →
`plan.sequence_for` → `camera.acquire(plan.n_frames, ...)` → `reducer.reduce`
(averaged over `shots_per_point`) — this is the per-point callback. `run(live=...)`
hands it to `active_plotter().run` when a viewer is registered (background worker,
UI-thread refresh, cooperative `stop()`), else runs the same callback synchronously
and still returns a complete `ScanResult` (x + `data_y` `(n_points, n_series)`).

### Scan taxonomy: temperature, readout-duration AND pulse-scan are ONE concept (two tiers)

A recurring question ("isn't Temperature just a kind of pulse-scan?") — yes. EVERY scan in the
system is the same concept: **sweep one or more NAMED pulse slots, reduce each point to a value**.
They split into two tiers by WHERE the per-point reduce lives, NOT by being different machines:

- **Coupled tier — `ScannedMeasurement` (inline reducer).** `Temperature` (trap-off → `SurvivalReducer`)
  and `Readout-duration`/`Fidelity vs duration` (readout window → `OtsuFidelityReducer`) are the SAME
  builder differing ONLY in `(slot, plan, reducer)`, so they BOTH call the one spine
  **`ReadoutSubsystem._build_slot_scan(controller, calibration, *, slot, values, label, plan,
  reducer, shots_per_point)`** (the `ScanAxis` + `ScannedMeasurement` assembly lives there ONCE —
  `build_temperature_scan` / `build_detection_scan` are 3-line callers). The reducer is inline here
  *on purpose*: survival needs a TWO-FRAME pair from one loading and fidelity needs the per-frame
  Otsu split — both depend on the multi-frame `ShotPlan` STRUCTURE a generic single-frame processor
  cannot see. **Both pulses are SELECTABLE templates** (#H3u-4, #H3v-1): each exposes a `template`
  ParamDecl editable in the pulse GUI, whose ONE swept DURATION slot is the scan axis — the same
  "pick a pulse" entry the generic `pulse_scan` has, so both are *visibly* pulse-scans; only the reduce
  stays coupled. `Temperature` defaults to `pulses/release_recapture.json`, sweeping the trap-off slot
  `s0` (`temperature.py::_resolve_release_recapture_template`). `Fidelity vs duration` defaults to the
  single-image `pulses/probe_template.json`, sweeping the imaging template's **image-period readout
  window** (`readout_duration.py::_resolve_imaging_template` binds a `duration` scan slot on the `image`
  period; `build_detection_scan`'s `_ExposureConfiguringPlan` matches the camera gate per point, so the
  imaging light-on time AND the camera exposure follow the sweep together — virtual==real). Both
  resolvers map role channels onto the session sequencer (virtual roles / real `ch00..`) via
  `imaging_channel_kwargs`. Each spec carries `metadata["scan_tier"]="coupled"` (the single source the
  boundary test + docs read) and NO `"node":"pulse_scan"` key, so the console builds it as a plain
  `ScannedMeasurementNode`. **The ONE honest physical difference**: temperature's swept duration is a
  STREAMER trap-pulse channel; fidelity's is the readout window (the imaging light-on time, whose camera
  gate the plan matches) — both are "a duration of the loaded readout template", realised on different
  hardware.

  **Why these are MEASUREMENTS, not Tasks** (#H3v-1): a recurring question is "isn't Temperature a
  Task?". No. A `Task` (e.g. `CalibrateReadoutTask`) publishes NOTHING to the SignalHub, yields an
  off-hub artifact on `self.result`, and renders a SINGLE mid-run frame in a fixed Monitor panel — the
  shape for a WORKFLOW that produces an artifact (centres/thresholds → a folder). Temperature/fidelity
  instead produce a live, hub-published, **fittable x-y curve** (`survival` vs `t_off`,
  `fidelity` vs `detection_time`) that the operator plots in ANY panel and fits (`na.fit_temperature`,
  `operations.fidelity.characterize_readout`). The three properties they share with a task —
  template + UI-params→auto-generated scan table + internal (camera/occupancy-derived) signal — are NOT
  task-defining; they are exactly how a COUPLED pulse-scan MEASUREMENT works. So they stay measurements.
- **Decoupled tier — `PulseScanNode` (y from a separate processor's signal, §20).** The GUI
  `pulse_scan` sweeps named pulse slots exactly the same way, but instead of an inline reducer it
  PUBLISHES the raw frames and reads y from another running node's signal via a `signal_expr`
  (e.g. `rate` off a Judge-occupancy processor). This is the decouplable case — y is a single-frame
  signal expression, so the reduce can live in its own node.

So "Temperature is a pulse-scan" is literally true at the model level (`ScanAxis` over a bound pulse
slot); it sits in the coupled tier because its reduce is frame-structural. Adding a new
single-slot coupled scan = a new `(plan, reducer)` pair + a `_build_slot_scan` call, never a new
scan loop.

### The tier boundary is a HARD contract (do not try to merge the engines)

The recurring temptation is to "finish" the unification by routing temperature / detection-time through
the decoupled `PulseScanNode`. That is IMPOSSIBLE without breaking physics, and the boundary is enforced
mechanically — pinned by `tests/test_scan_tier_boundary.py`:

- **Decoupled per-point acquisition = `acquire(1)`: ONE camera trigger, ONE frame** (`logic.py`
  `PulseScanNode.shot` → `camera.acquire(1, sequence=...)`). Its y is then a SINGLE-FRAME
  `signal_expr` over the published `frame` (read off another node, e.g. `rate`).
- **The coupled reducers need a multi-frame structure a single published frame cannot carry:**
  - `SurvivalReducer` needs EXACTLY 2 frames from ONE atom loading (`temperature.py`: raises unless
    `len(frames)==2`), produced by `ReleaseRecapturePlan` (`n_frames` fixed at 2 — the two camera
    triggers of ONE release-recapture sequence, trap-off between them). "Survival" is per-atom only
    because both reads share the loading; two independent loadings make it meaningless.
  - `OtsuFidelityReducer` needs the per-point frame SET to pool counts and Otsu-split — a statistic
    over the point's frames, not a single value.
- **The pulse carries its own triggers (no 1→N expansion):** each pulse defines its OWN camera
  capture edges — a single-image readout pulse has one, a release-recapture bracket has two (trap-off
  between them). The camera reads N frames off whatever sequence it is handed; a 2-trigger release-
  recapture pulse is ONE loading read twice, while a 1-trigger imaging pulse read N times reloads
  between frames (independent shots, decided per-frame by the atom device). So the coupled tier's
  release-recapture pulse INTRINSICALLY produces 2 frames of one loading — the decoupled node's
  per-point `acquire(1)` would only image once, never the two-read survival. The trigger-counting util
  `count_trigger_pulses` lives in the camera seam (`devices/camera_trigger.py`), not the sequencer:
  the streamer is a pure pulse streamer and never inspects which channel gates a camera.

Therefore: **a measurement whose y needs ≥2 frames from one loading (survival) or a per-point frame-set
statistic (Otsu fidelity) is IRREDUCIBLY coupled.** The promotion path (ship the pulse as `pulses/*.json`
+ move the reduce into a `@processor`) applies ONLY to single-frame reduces, NOT to survival/fidelity.
Do NOT ship a `pulses/release_recapture.json`: the pulse is built programmatically by
`build_release_recapture_pulse` from the live sequencer's channel roles, so a static JSON would hard-code
`trap`/`probe`/`emCCD` names and drift on real `chNN` configs — the programmatic builder is the single source.

Naming note (two different "fidelity" things): **"Fidelity vs duration"** = `readout_duration.py`
(the detection-time scan, key `readout`) IS a coupled pulse-scan (covered above). The module
`operations/fidelity.py` (`characterize_readout`, held-out train/test threshold fitting over labelled
image GROUPS) is NOT a scan at all — no swept axis, no per-point acquire loop — so it is not a pulse-scan
special case; it is the rigorous offline counterpart to the live `OtsuFidelityReducer` quick-look.

### Two invariants every scan/pulse flow depends on (api slot AND scan slot must fire; manual config == one-click)

Both are pinned by `tests/test_scan_slot_and_manual_parity.py`.

- **A manually-configured logic node behaves like the one-click task.** A node's `y` (or a plot/processor
  source) binds a signal BY NAME, resolved at RUN time — the name may reference a signal that is not live yet
  (a pulse-scan reading its OWN `frame_0`, or a not-yet-started producer's output). The console signal picker
  (`fill_grouped_signal_combo`) must therefore ROUND-TRIP any configured name, live or not: it adds a
  configured `current` to the name pool so BOTH the tree and flat picker render it as a "waiting" leaf and
  read it back. The tree branch used to drop a not-listed name → `read_editable_combo` returned `''` →
  `collect_values()` lost the input → `_start_logic_node` built a node with an EMPTY y-expression → every
  scan point NaN → an empty grid on Start (looked like an async/daemon race; it was the picker dropping the
  input). One rule for both branches — the flat branch's old special-case is gone.
- **A fireable pulse template authors on the connected board's clock grid (resolution comes FROM the
  device).** The ONE loader for a template that is about to fire is the timing layer's
  `resolve_fireable_template(template, default_name=..., default_factory=..., sequencer=...)`
  (`timing/pulse_table.py`, beside `resolve_pulse_template`): it resolves the file and then forces
  `time_step_ns = hardware_tick_ns(seqr)` — the tick is read off the CONNECTED sequencer's `clock_hz` (a
  `SequencerDevice` property the real `RemoteSequencer` reads back from the FPGA on connect, and the
  `VirtualSequencer` carries), exactly as confocal reads a device's resolution off the device, NOT a constant
  baked into a caller and NOT whatever a saved file under the (gitignored) `pulses/` folder carried. Only when
  NO device is in scope (a GUI template preview) does it fall back to the streamer-config default
  `DEFAULT_CLOCK_HZ`. A finer authoring grid (an old save with `time_step_ns = 1`) lets an api/scan DURATION
  sweep produce sub-tick durations (`np.linspace` → 5.95 ns …) that `set_api` snaps to the template grid but
  that still fail the clock-grid validation → the sweep cannot fire ("api slot does not work"). Snapping to
  the board's tick at load makes **author == snap == fire** for every template on whatever clock the connected
  board reports, so both the software api sweep (`SCAN_MODE_API`) and the hardware scan table
  (`SCAN_MODE_SCAN`) of a duration slot always fire. Callers with a device in scope pass it:
  `pulse_scan.build` → `s.devices.sequencer` (via `_resolve_probe_template`, the probe-flavoured binding of
  the same helper, which the GUI slot preview also imports), `mot_field.run` → `self.sequencer`, and the
  Calibrate task's `_resolve_template` (logic.py) → `self.sequencer` (its GUI slot preview calls the same
  classmethod without a device and falls back to the config-default grid, like every other preview). The
  tick is pinned by `tests/test_scan_slot_and_manual_parity.py`. The CHANNEL CATALOG follows the same
  device-owned rule: a saved template is a SUBSET of the board's channels, so the loader (and the pulse
  GUI's Load) expands a subset template onto the connected device's full channel list via
  `aligned_to_channels` (catalog order, missing channels as off rows — the compiled program is identical,
  and "Show All" then really lists every device channel); a NON-subset template is left untouched (the
  prepare layer rejects unknown channels, `resolve_coupled_template` owns role→device remapping). The
  virtual catalog (`devices/virtual.DEFAULT_CHANNELS`, 25 channels) and `VirtualMotCamera.coil_buses`
  both derive from the one `MOT_COIL_BUSES` source; the shipped `pulses/*.json` templates are regenerated
  onto that full catalog. Pinned by `tests/test_pulse_template_channel_catalog.py`.

### A task's mid-run panel is DECLARED, sized and coloured from single sources (no console special-case)

The four fixes below all trace to ONE lesson: a task's mid-run visualization must be built by the SAME path,
with the SAME single sources, a user's manually-composed panel uses — never a bespoke branch with magic numbers.

- **Declared, not hand-assembled.** `_set_task_running` no longer hand-builds a grid PanelConfig inline. A task
  declares only DATA — its spec's `default_kind`/`mid_run_key` and (for a scanning task) `node.grid_shape` (a
  plain tuple; `neutral_atom/**` never imports `frontend`) — and `_task_mid_run_config(spec, node)` maps that to
  an ordinary `PanelConfig` fed to the SAME `_new_panel_card` that manual Add-Panel and save/load use. A scanning
  task (`default_kind="grid"` + a ≥2-D grid_shape) faceting its LAST scan axis is the only special shape; its
  `sub_plot_kind` auto-derives (`_resolved_sub_kind` → `default_sub_plot_kind`), not hand-set.
- **Size from the ONE recommendation rule.** The panel's `size` is `recommended_grid_size(n_cells)`
  (`live.py`: `optimal_grid_size(*grid_shape_for(n_cells, max_cols=_SITE_MAX_COLS))` — the SAME rule `GridPlot`
  uses), never a magic constant. A 3-cell MOT grid opens `2x2`, not the old hardcoded `4x4`.
- **cmap: render == Setting, one source.** A panel's default colormap has ONE home: `PANEL_PARAMS[kind].default`
  now REFERENCES `PALETTE["cmap_scan"]`/`["cmap_camera"]` (not a `"inferno"`/`"gray"` literal), so `PALETTE` is
  the single colour source. The live facet grid (`_build_facet_plotter`) resolves its cmap through the ONE
  `_resolved_cmap(sub, params)` (operator's pick ELSE the kind default) — the SAME resolver the Setting popup,
  the standalone 2-D panel and the save state use — so a grid whose params carry no explicit cmap draws with the
  SAME default the Setting shows. (Before: the grid fell through to `ImageCell`'s own grey default while the
  Setting said inferno — the render-vs-Setting divergence.) The na-side calibration report (`build_grid_figure`)
  keeps its camera-crop grey default; that is a different, device-frame path, correctly separate.

### Repeat is a measurement param; repeat_mode is a plot param (the data model)

**Uniform output contract (#H3n), enforced by the base class + `tests/test_measurement_output_contract.py`:**
EVERY acquiring measurement publishes its primary block with shape **`(repeat, *points_shape, *data_shape)`**
— `repeat` = the repeat-axis depth, `points_shape` = the swept parameter space, `data_shape` = the
per-point data. `LogicNode` declares `points_shape`/`data_shape` (default `()`); the nodes set them:

| node | `points_shape` | `data_shape` | block |
|------|------|------|------|
| camera | `(1,)` (a frame sweeps no input param) | `(H, W)` (the image IS the data) | `(repeat, 1, H, W)` |
| 1-D scan | `(n_points,)` | `(dim,)` | `(repeat, n_points, dim)` |
| 2-D scan | `(n0*n1,)` (param1×param2) | `(1,)` | `(repeat, n0*n1, 1)` (+ `_grid` reshape) |

`repeat` is the **ONE measurement** param, **0 = ∞** (#H3u-2): the Acquisition `Repeat (0 = ∞)` int,
declared ONCE in `_acquisition_param_decls(repeat_default)` and **auto-injected** through the SAME
`_rebuild_form` as every param (never a hand-placed widget). There is **NO separate Free-run toggle** —
0 IS infinite, the same semantics everywhere (and the same as the scan-repeat count). `_repeat_value(values)`
→ **`int`**: `repeat=K>0` keeps a K-deep block (K passes/photos **averaged**) then **STOPS**; `repeat=0`
rolls a **1-deep ring forever** (a live monitor showing the latest). The kept block depth is
`_ring = max(1, repeat)` (K for finite, 1 for ∞). A scan re-runs the whole sweep `repeat` times; a
**camera takes exactly `repeat` photos then FINISHES** (or rolls forever at 0). **Per-type default:** a
camera defaults to `0` (∞, a live monitor); a scan defaults to `1` (one finite sweep) — the node ctor
defaults mirror the GUI form defaults so a headless `run_to_completion()` terminates. `frame` = the
`(repeat, 1, H, W)` block itself (`repeat`=`_ring`); `frame_i` stay the single per-trigger images
(`OccupancyProcessor` reduces a >2-D `frame` to its newest filled (H,W) slice for the per-shot judge).
**Two-cameras fix:** the built node's `instance_label` is set to its row TITLE, so its provider label
matches the declared row (the empty-prefix camera no longer shows `frame` under both "camera" and
"Camera (live frames)").

HOW the repeats become a picture is a **plot** parameter, `repeat_mode` (each plot panel's Setting
combo, persisted in `config.params`): `average | add | replace | roll | create`. The pipeline
(`PanelCard._signal_then_repeat` → `_eval_signal_per_slice`) runs the `value = ...` expression **once
per repeat**, presenting every repeat-carrying signal it reads (the bound `signal` AND any raw hub
signal it names directly, e.g. `value = frame[0]`) as that repeat's whole core — so the repeat axis
stays OUTSIDE the expression and `signal` only ever sees one frame / curve. The re-stacked block (any
3-D value, regardless of how it was named) is then collapsed by the ONE owned reducer
`frontend.live.reduce_repeat(raw, mode)`: `average` = `nanmean` over the repeats that **have data**
(the true running mean = a long exposure for a camera, magnitude-stable; this is what "average" means,
not a sum), `add` = `nansum`, `replace`/`roll` = the latest, `create` = every repeat as its own column
(one line per repeat, **1-D only**). A 1-D panel keeps the reduced 2-D shape so the dimension axis
**and** `create` draw as **multiple lines** via Live1D's native multi-line + `×N` ylabel
(`repeats_with_data` drives `update(repeat_cur=)`). Line style is **confocal-exact**: every line
solid, `alpha=1`, the global `lines.linewidth=1`, colours CYCLE `LINE_CYCLE` (grey `#808080`,
skyblue, …) by column index — a lone line is grey (identical to confocal's `repeat=1`), no per-repeat
fade. There is **no** measurement-side averaging anywhere (that was the camera live-stutter); the
plot owns every reduction.

**Auto-reshape (#H3o) — STRUCTURE-DRIVEN, decided by the DATA dimensionality, not a size threshold.**
The three axes (repeat / points / data) are ORTHOGONAL. A node declares `points_shape` / `data_shape`
/ `grid_shape` (base-class fields); the console threads them to `PanelCard` via `structure_provider`
(`_signal_structure` → `_node_for_signal`). `PanelCard._bound_structure()` returns them ONLY for the
default `value = signal` source — a custom expression rewrites the core, so structure is then advisory
and the code degrades to shape inference. `_coerce` reshapes by `len(data_shape)`:
- **`data_shape` 1-D** (a scan's `(dim,)` series): each data series is its OWN line; a 1-D `(points,
  dim)` value stays 2-D (multi-line); it never reshapes to an image.
- **`data_shape` 2-D** (a camera frame `(H,W)`): the data is an image → a **2-D** panel imshows it; a
  **1-D** panel UNROLLS it to ONE trace.
- **`grid_shape`** un-flattens a 2-D scan's `(n0*n1,)` points to an `(n0,n1)` map on a 2-D panel (a
  scan's `data_shape` stays `(1,)`, so the 2-D-ness is in the POINTS, recovered via `grid_shape`).
- a 1-D-data / 1-D-points value on a **2-D** panel RAISES (never a silently-wrong image); **dist**
  flattens to a histogram (structure not consulted).

`repeat_mode` stays orthogonal: `reduce_repeat`/`repeats_with_data` accept any `(repeat, *rest)`
(camera 4-D block included), reducing axis 0 only; **`create` is orthogonal to the data axes** —
a 3-D scan block → `(points, R*dim)` (confocal columns), a ≥4-D image block → `(prod(core), R)` so a
camera frame + create draws **R repeat-traces, NOT H image-rows** (the bug #H3o fixed). `_DIM_MULTILINE_MAX`
is deleted; `tests/test_panel_reshape_orthogonal.py` encodes the full corner-case matrix as the guard.

### Single source of truth: build_*_scan + declarative spec

`ReadoutSubsystem.build_temperature_scan(...)` / `build_detection_scan(...)` are the
ONLY assemblers. Both `exp.readout.temperature(...)` (runs it) and the GUI's
`measurement_specs()` closures call them, so the notebook one-liner and the GUI
Start button drive identical acquisition — they cannot drift.

`MeasurementSpec` / `ParamDecl` (declared once; API default AND GUI control derive
from the one declaration — explicit declaration, NOT signature reflection/AST, to
avoid the confocal AST-guess pitfall). `ParamDecl.kind` ∈
`float/int/axis_range/bool/choice/text/path/signal/signal_expr/pulse_param/pulse_slots`
(the whitelist is enforced in `measurement.py`); **no value is ever `eval`'d** — consumers
validate/coerce by kind (the confocal free-text-eval lesson). `MeasurementSpec.build`
closure captures `exp`, so the console never holds the session (decoupling).

**Device-role injection (declare `devices=[...]`, get a dropdown + a resolved device).**
A spec that uses a device does NOT hand-roll a `choices=camera_names()` ParamDecl or index
`s.devices[name]`; it DECLARES its device roles on the decorator —
`@measurement(devices=["camera"])` / `@task(devices=[("camera", {"default": "monitor_camera"})])` —
and the base wires it in ONE place. The mechanism: `OpenRegistry.register`/`decorator` carry the
`devices=` list; the single funnel `OpenRegistry.discovered_specs` (the only place
`readout.session.devices` is in scope) calls `CatalogSpec.with_devices_bound(device_set, roles)`,
which (a) appends one `choice` ParamDecl per role (`device_param`/`device_params_for`, choices =
`DeviceSet.device_names(base_type)`, i.e. every device of that role's type — a camera role lists
BOTH `camera` and `monitor_camera` because they are the same `CameraDevice` domain, just different
physical devices) and (b) wraps the build-callable (`_bind_device_args`) so the operator's chosen
NAME is resolved to the device INSTANCE and injected as that keyword before `build` runs. So a
factory's `build(*, camera, ...)` receives a `CameraDevice`, never a string. **Pitfall:
`_DEVICE_INJECTABLE = ("build", "make_node")` is applied only to attributes that are real dataclass
FIELDS (field-guarded) — `MeasurementSpec.build` is a field (its `make_node` method calls the wrapped
build), `ProcessorSpec.make_node` is a field; `dataclasses.replace` cannot swap a method.** Adding a
new device DOMAIN (RF, DAQ, …) is one `register_device_domain(key, base_type)` call — no spec edits.
`camera_spec` (an imperative builder, not a `@measurement` factory) calls `with_devices_bound(...)`
by hand. Readout-locked measurements (temperature / readout-fidelity) intentionally do NOT declare a
role (they image single atoms and must run on the science camera, not a MOT monitor). Guard:
`tests/test_device_role_injection_contract.py` (Pin A: no hand-rolled camera dropdown via AST; Pin B:
declared roles → real choice params; Pin C: injection passes a device, not a string). The GUI face of
the device-DOMAIN registry is `exp.device_manager()` / the task-console "Devices" button.

`ScannedMeasurementNode` (`operations/logic.py`) wraps a measurement as a console
logic node: each `shot()` advances ONE scan point and publishes the CUMULATIVE
`{x_key: x[:k], y_key: y[:k], scan_done, shot}`; finite-scan **self-stop** (sets its
own stop event after the last point, so a background `start()` thread exits).
`run_to_completion()` for headless/tests.

**Virtual == real guard.** Engine + node touch only `camera.acquire` /
sequencer (via pulse) / `calibration.signals|detect` / `active_plotter().run` — zero
import of virtual/qcmos, zero simulation ground truth. Living in `operations/` they
are caught by `tests/test_virtual_equals_real_contract.py`.

### Data-processing actions: ProcessorSpec (the one-shot sibling of a measurement)

A measurement SWEEPS a parameter into a live curve; a **processor** runs ONCE over
freshly-acquired or saved frames and produces a structured result (per-site arrays +
scalars). They are the SAME shape — declared params (`ParamDecl`) in, named data out
— differing only in execution (swept vs one-shot) and whether a default plot is
declared. To add one: drop a module into `operations/processors/` with a
`@processor` factory `build(readout) -> ProcessorSpec` (auto-discovered by
`processor_registry`, exactly like `@measurement`); it appears in
`exp.readout.processor_specs()` and the console's Add-Panel **"Data processing"**
category with no catalog edit.

- `ProcessorSpec` (`operations/processor.py`): `params` (`ParamDecl`, reused — incl.
  the `text` kind for a folder path), `run(ctx) -> {signal: value}`, `result_keys`,
  `summary_keys` (scalars), and the OPTIONAL default-view binding
  `default_kind`/`default_value_key` (e.g. `sites` + `fidelity_site` — empty
  `default_kind` = a pure data action whose outputs the user wires manually).
- `run(ctx)` DRIVES existing analysis (the built-in `readout_fidelity` calls
  `ReadoutSubsystem.characterize_from_dir`) — it re-implements no readout/fidelity
  math; `readout` is captured in the factory closure so the console stays decoupled
  (it drives the action through the plain spec list, never holding the subsystem).
- `ProcessorRun` (`operations/logic.py`) runs the spec ONCE on its owner thread,
  publishes the result (+ a `processor_done` scalar) to the hub, and self-stops; the
  cooperative-stop event is shared so a long grab cancels cleanly. The console's
  result panel reuses the EXISTING `sites` atom kind (camera underlay + per-site
  circles); the scalars are visible in every panel's signal legend.
- `discovered_processor_specs` FAILS LOUD on duplicate `result_keys` (two processors
  would clobber a shared-hub signal). Virtual==real: a processor's only data source
  is `camera.acquire` or a saved folder, so it is caught by the same
  `test_virtual_equals_real_contract` guard.

### Release-recapture thermometry physics (`operations/temperature.py`)

Trap OFF for `t_off` → atoms fly ballistically → trap ON → survival = recaptured /
initially-occupied. Model (`release_recapture_survival`, 3-D isotropic point source,
initial position spread neglected = short-time tight-trap approx): per-axis recapture
is the Gaussian velocity fraction inside `|v| < r_c/t`, isotropic survival =
`baseline · P_axis(t)**3`, with 1-D spread `σ_v = sqrt(k_B T / m)`. Uses the shared
`normal_cdf` from `_readout_math.py` (does NOT re-implement erf). Ref: Tuchendler et
al., PRA 78, 033425 (2008).

- **Capture-radius degeneracy (key).** The curve fixes only `r_c / sqrt(T)`, so
  `capture_radius` (metres, from trap geometry ≈ tweezer waist) MUST be supplied;
  `fit_temperature` takes it as a known input and fits T (+ optional baseline) only.
- **The 6-period pulse** (`build_release_recapture_pulse`): image1_expose,
  image1_settle, **trap_off (duration bound to scan slot s0 = t_off)**,
  trap_recapture, image2_expose, image2_settle — two emCCD triggers, one trap-off
  between. Bound via `bind_field("duration","2")` — the SAME slot mechanism a scanned
  readout duration uses, so hardware can stream a whole t_off table. Must be a SINGLE
  two-trigger sequence the camera reads as TWO frames of ONE atom loading, NOT a repeated
  single-trigger sequence: a single-trigger pulse the camera reads twice is a fresh shot per
  frame (the virtual atom device reloads when the base sequence carries fewer triggers than
  the frame count), so the two images would be different loadings with no trap-off between —
  survival needs the SAME atoms imaged before/after one trap-off.
- **`fit_temperature` is pure post-processing** on a finished `ScanResult` — it never
  runs inside the acquire loop, keeping the live engine free of physics models.

### Virtual trap-off loss model (data-source side ONLY)

The virtual camera models release-recapture loss **at the data source**: it parses
the trap-off duration out of the FIRED `PulseSequence`, computes a recapture survival
probability with stdlib `erf`, and drops atoms accordingly. `reload()` is once-per-shot
to avoid compounding loss across shots. This does NOT pollute the analysis layer (the
analysis only ever sees camera frames and runs `detect`). End to end on virtual,
survival decays ~0.97 → ~0.06 over 0..300 µs and `fit_temperature` recovers ~44–52 µK
(injected 50 µK). This is "fake only the lowest data source" in action — the same
discipline as the loading model.

### task_console live-refresh: BLIT (the full-draw floor was NOT intrinsic)

Source: memory note `task-console-live-perf-floor`; blit landed 2026-07 (`BaseLivePlot._compose_blit`,
split into a two-phase `compose()`/`present()` so the board composes every panel's buffer then presents
them together in one coherent frame).
The old cost: `BaseLivePlot.draw()` re-rasterised the WHOLE 300-dpi figure every tick
(`canvas.draw_idle()+flush_events()`, cProfile top = `draw_text` glyph rasterisation across all
axes), ~12 ms per 2x2 panel — 5-6 panels at 100 ms saturated the budget. Confocal-GUIv2 (the
reference) always blitted; this reimplementation had never ported it.

**Now the live tick BLITS** (single-sourced in `BaseLivePlot`): restore a cached chrome-only
background + `draw_artist()` only the data artists (a generic `ax.lines/images/collections/
patches/texts` enumerator, z-sorted) + `canvas.blit()` — ~0.6 ms. The chrome signature
(dpi/size/title/per-axes pos+lims+ticks/image clim+cmap) gates the cache; a tick whose chrome
moved does a plain full draw and DEFERS the capture, so a jittery panel never pays a recapture
penalty (non-regressive) and only a stable panel blits. Deviation from confocal (which marks
artists `animated`): the bg is captured with the data artists momentarily HIDDEN, so save/inline/
snapshot full renders need zero special-casing. Notebook (ipympl) / headless keep the full draw.

The blocker the OLD attempt hit ("live autoscale changes limits every tick so the fast-path never
fires") is real but was solved by **dead-banding EVERY autoscale limit**, not by giving up: the 1D
y-axis relim already had hysteresis; the side-distribution count axis (`_count_axis_limit`), the 2D
colour limit (routed through `relim`), and the histogram count-y axis (`_apply_count_yscale(force=)`)
did NOT — those were the axes whose ticks moved every frame. With them dead-banded the chrome holds
and blit engages. Colorbar ticks now sit at the committed clim ends (guides still show the raw
min/max). **Iron law: blit engages only if ALL of a panel's autoscale axes have a dead-band; a new
plot with an autoscaling secondary axis must wire it in or it silently falls back to full draw.**
Measured (2x2, offscreen, update() end-to-end): 1d 13→0.9 ms, 2d 20→2.3 ms, monitor 18→6.9 ms
(monitor residual = the per-tick σ curve_fit + mathtext in `update_core`, not the draw). Guarded by
`tests/test_frontend_blit_render.py` (engages + equals full draw + no ghosting + recapture-on-relim).

A separately-rejected speed-up (still off the table): **freezing title/xlabel positions** — draw
−19% but sf1.5 flips 4845 px (`show()`'s first draw freezes the position at the wrong dpr), violating
DPR neutrality.

An earlier compliant speed-up that also landed = **tightening figure margins**. `PANEL_MARGINS_PX`
is `(110, 86, 80, 70)` (L, R, B, T): R/B/T are pulled in from confocal's stock
`(110,110,100,40)` so the data area fills ~50% of the figure (a smaller agg buffer makes
the ~30–40% of draw cost that scales with area faster; `draw_text` unchanged). The LEFT
stays at the confocal `110` — it is `STOCK_MARGINS_PX[0]`, the minimum that holds a 4–5
digit y-tick label (e.g. a qCMOS ROI pixel like `1180`) PLUS the rotated y-title; tighter
clips the y-title past the figure's left edge (true for every panel kind). T `70` is the
always-reserved title slot (`TITLE_SLOT_PX`). (Hysteretic autoscale is now done — it is what
makes blit engage, above; staggered redraw and low-dpi panel mode remain unused and would change
visible behavior.)

**Frontend geometry tokens have ONE source each (no inline magic, no re-typed test
literals).** The visual design is owned by the frontend, never a per-call/host knob; but a
value is written ONCE and everything (including tests) reads/derives from it, so a tweak
propagates and nothing drifts. Where they live: `style.py` owns the stock figure geometry
(`DESIGN_DPI = 300`, `STOCK_DATA_PX = (480,360)`, `STOCK_MARGINS_PX = (110,110,100,40)`;
`figure.figsize` and `canvas.FigureSpec` defaults derive from these). `live.py` owns the
panel/pulse/site geometry plus `TITLE_SLOT_PX = 70` (the title-slot floor used by BOTH
`PANEL_MARGINS_PX[3]` and `_with_title_margin`, so they can never disagree) and the named
axes splits `_DIST_SPLIT` / `_IMAGE_SPLIT`. `qt_fluent.py` owns `FLUENT_SCALE_MIN/MAX`
(the scale clamp band) and `screen_fit_window_size(window_ratio)` — the ONE screen-fit
window rule the task console AND the pulse editor both call (it was duplicated verbatim in
both, the same drift class as the shared scale rule). The contract tests in
`test_frontend_smoke.py` (`test_panel_plot_spec_is_the_confocal_modular_region`,
`test_embedded_canvas_invariants_across_screen_scales`, `test_task_console_cards_are_modular`,
`test_fluent_auto_scale_is_shared_between_guis`) IMPORT these constants and assert
RELATIONS (e.g. `panel_display_size == round((data+margins)*scale)`), never re-typing the
literal — the fix for the bug where a test asserted a stale `(110,110,100,70)` long after
the code had moved on.

## 20. Decoupled pulse-scan node + the one `signal_expr` evaluator (2026-06-23)

### `operations/signal_expr.py` — the single multi-slot signal + value-expression evaluator
A "source" anywhere in the system — a plot panel's data source, a processor's input, a
pulse-scan's y — is the SAME object: a list of picked hub-signal names (the *slots*, read as
`signal` / `signal[i]`) plus a one-line `value = ...` expression. `SignalExpr` owns the
slot-packing rule (scalar for one input, list for many), the `value` contract, `co_names()`
(names referenced + picked slots, for version-gating), an optional `resolve` hook (the
frame-coherence rewrite, injected by the frontend, never baked into the analysis layer), and
the ONE namespace: `NAMESPACE_HELPERS` names the helper set
(history/latest/names/shot/np/numpy/math), `namespace_helpers(hub)` binds it, and
`hub_namespace(hub, snapshot=None)` layers those helpers on a signal snapshot (latest by
default; the console passes its shot-coherent `snapshot_at` view + its reserved view keys,
a reactive processor its coherent inputs) — so a panel, a pulse-scan y and a processor
source have identical expression capabilities, and the GUI's reference-exclusion set
derives from the same constant. Dependency-free, so the analysis layer and the GUI share
ONE definition. `PanelCard._with_signal_slots` / `_evaluate` / `_co_names` delegate to it;
`task_console.SOURCE_EXPR_HELP()` is a lazy getter of `signal_expr.SIGNAL_EXPR_HELP` (one
help text). `tests/test_signal_expr.py` guards it.

### Pulse-scan is a device-driving `PulseScanNode`, with a DECOUPLED y
`PulseScanNode` (`operations/logic.py`) replaces the old frame-reducing measurement. **x** = the
scan points: api slots are FIXED once on the base state, scan slots are resolved per point via
`PulseTableState.with_slots_resolved({s0: row, ...})` — the SAME named-slot resolver the
hardware scan + pulse GUI use (api and scan slots are different mechanisms and stay separate; no
clearing, no per-period editing). **y** is decoupled from the readout: per point the node fires
+ acquires the camera `frame`, PUBLISHES it (bare `frame`, the same signal a `CameraMeasurement`
publishes), lets the subscribed consumer (e.g. a Judge-occupancy processor) recompute, then
evaluates a `signal_expr` over the hub for y. So the readout pipeline lives in its own node and
pulse-scan just sweeps + records its output.

Settling y to THIS point's frame is **race-free without any cross-thread `step()`**:
- GUI / live: the consumer runs on its own daemon thread, so the node WAITS for the picked y
  signals' per-signal version (`SignalHub.signal_versions()`) to advance past the pre-publish
  snapshot — it only READS the hub.
- headless / notebook / tests: an optional inline `settle` callback steps the consumer once,
  single-threaded (its thread is not running), so the value is fresh immediately.
Both read y through the same `SignalExpr.evaluate` once the consumer is fresh. A 5 s timeout
means a mis-wired y never wedges the scan. The device lockout (`DEVICE_DRIVING_KINDS`) makes
pulse-scan the sole device driver; processors are not device-driving, so the workflow is: start
the producer (occupancy) FIRST, then pulse-scan.

`pulse_scan.py` `build()` returns a `PulseScanPlan` (base state + scan/api arrays + camera/sequencer
+ y `SignalExpr` + `extra_delay_s`); the spec carries `metadata={"node": "pulse_scan"}` to mark its
scan TIER. `MeasurementSpec.make_node(hub, prefix=, repeat=)` (the `ProcessorSpec.make_node`
counterpart) owns the spec→live-node assembly: it reads that tier tag and returns a `PulseScanNode`
for the decoupled `"pulse_scan"` tier, else a `ScannedMeasurementNode` (the coupled
temperature/fidelity tier reduces inline over a loading's frames). The console's `_build_logic_node`
just calls `spec.make_node(...)` — it never imports a concrete na node class to pick one by the
metadata string. Pulse-scan
images ONCE per point (`triggered_frames(camera, sequencer, sequence, 1)`, the single arm-fire-read
helper — no hand-rolled `camera.acquire`) — it is decoupled from the camera's exposure/averaging,
so there is no frame-count knob; the params are `template` + `pulse_slots` (api fixed/sweep + scan
program + extra settle) + `y` (signal_expr) + `y_name` (the output signal name).

### Scan POINTS are ONE `scan_table` program (the pulse-GUI model), not per-slot
The scan points are a single `(N_points x n_slots)` table — one ROW per scan point, one COLUMN
per bound scan slot, the slots advanced in LOCKSTEP — built by a small Python program assigned to
`scan_table`, EXACTLY like the pulse GUI Scan tab. There is no separate points box per slot. The
templates (`column_stack` default, `grid`) + the evaluator live ONCE in
`timing/pulse_table.py::scan_table_template` / `evaluate_scan_table_code` (analysis layer, so both
GUIs + `build()` share them; `pulse_gui._template_*` delegate to them). `build()` evaluates the
program, then `snap_scan_table(..., time_step_ns=state.time_step_ns, dac_ranges=...)` makes each
column hardware-legal IN ITS NATIVE UNIT (a duration column → whole clock ticks in ns, a DAC
column → integer LSB code — the axis unit is derived per slot KIND, never assumed to be time).
The column count MUST equal the bound slot count (lockstep contract), else `build()` raises. The
GUI `_PulseSlotsWidget` renders TWO peer sections (hierarchy = weight+colour, never a wrapper
header): **API slots** (always shown) + **Scan table** (always shown). Its value is
`{"api": {...}, "scan_code": "<python>", "api_scan": "<python>", "extra_delay": <s>}`.

### API slots also sweep — in SOFTWARE — the analogue of the hardware scan table
Scan slots (`sN`) stream on the FPGA; api slots (`aN`) are fixed handles set per shot in software.
So an api slot can be **fixed** (a numeric value, `set_api` once at build) OR **swept**: the API
section's `api_scan` program (a `(N x n_api)` table, one column per api slot in order, the same
`evaluate_scan_table_code` evaluator as the scan table — no hardware snap, `set_api` interprets each
value in the slot's native unit) drives `PulseTableState.with_api_resolved({aN: row})` per point —
the software twin of `with_slots_resolved`. A pulse with ONLY api slots is therefore fully
sweepable (x = the swept api slot); scan + api slots may both sweep in lockstep. The per-point loop
is **load → on_pulse → wait the pulse done (`camera.acquire`) → settle → next**, and the settle is
owned by the DEVICE so callers don't hand-roll timing: `SequencerDevice.settle(seconds, *, stop=)`
idles the (adjustable) `extra_delay` between points (`VirtualSequencer` scales it
by `sleep_scale` like `wait_done`; the wait is cooperatively cancellable on Stop). Guarded by
`test_pulse_param_scan.py` (api-only sweep drives each point; settle is called once per point).

### Repeat: TWO systems, three orthogonal axes, processors are typed transforms (#H3o)
There are EXACTLY TWO repeat knobs in the whole pipeline — never three. The design panel
(workflow `processor-update-mode-design`) verified this against the code and rejected every attempt
to add a processor-side mode.

1. **ACQUIRE-FILL (measurement layer).** Every acquiring `Measurement` OWNS the repeat axis and
   FILLS a `(ring, *points_shape, *data_shape)` BLOCK in `shot()` (the camera ring; a scan's raw
   block) where `ring = max(1, repeat)`. Driven by ONE user field, auto-injected as an acquisition
   ParamDecl: `repeat:int`, **0 = ∞** (#H3u-2) — `K>0` keeps K shots & averages then stops; `0` rolls
   a 1-deep ring forever. No separate free-run toggle. The measurement NEVER collapses the repeat axis.
2. **DISPLAY-COLLAPSE (plot layer).** The plot collapses the repeat axis FOR DISPLAY ONLY via
   `repeat_mode` (`live.reduce_repeat`, modes `average/add/replace/roll/create`). Stored data is
   never mutated; this is the SINGLE place a repeat axis is collapsed.

The three axes (repeat = block axis 0 / points = middle / data = trailing) are orthogonal (#H3n/#H3o):
`reduce_repeat` collapses axis 0 (only on a `ndim>=3` block), `_coerce` reshapes points/data.

**A processor has NO user-facing mode.** A `Processor` is a pure typed transform; its relationship to
the repeat axis is a STATIC class attribute `repeat_contract`, NEVER a runtime knob or form field:
* `"preserve"` — maps each repeat slice 1:1 and emits a block whose LEADING axis 0 IS the repeat, so
  the SAME plot `reduce_repeat` machinery collapses it. **`OccupancyProcessor` is `preserve`** (#H3q,
  #H3s-F3): the repeat axis flows THROUGH it — fed the camera's `(repeat,1,H,W)` block it judges EVERY
  slice and publishes `occupied`/`counts` as CLEAN `(repeat, n_sites)` blocks (a leading repeat axis,
  NO vestigial middle 1) and `frame_judged` as `(repeat, H, W)`. Repeat-collapse is **structure-driven,
  not an ndim guess**: each signal's `SignalSpec` declares its OWN `points_shape`/`data_shape`, so
  `core_ndim = len(points)+len(data)` (occupancy: points=(), data=(n_sites,) → core_ndim 1) tells
  `reduce_repeat(block, mode, core_ndim=core_ndim)` that axis 0 is the repeat exactly when
  `block.ndim == 1 + core_ndim`. `reduce_repeat`'s legacy ndim≥3 fallback is kept verbatim for callers
  that pass no `core_ndim` (camera `(repeat,1,H,W)` / scan `(repeat,pts,dim)` blocks), so those paths
  are byte-identical. Static geometry — `centers` (N,2) / `thresholds` (N,) — declares NO contract
  slot (its SignalSpec leaves points/data `None`), so it prints its raw shape and carries no repeat
  axis a consumer could mistake for one.
* `"reduce"` (the base default) — emits a result with NO repeat axis (a statistic over a shot set, e.g.
  a `FidelityProcessor`). Nothing is left for the plot to collapse, so it can't collide with `repeat_mode`.

So the everyday quantities all come out without a third knob, via ONE mechanism (plot `repeat_mode` over
the block): **S1** averaged site-map image = a `frame` panel `repeat_mode='average'`; **S2** per-site
loading PROBABILITY = a `sites`/`2d` panel bound to `occupied` with `repeat_mode='average'` (averaging
the camera's `repeat` shots recovers every site — the user sets how many via `repeat`); **S3**
loading-rate-vs-time = the processor's scalar `rate` (this block's loading fraction) on a rolling panel;
**S4** readout fidelity = a `reduce` operation (`readout_fidelity` spec) over a shot set. `OccupancyProcessor`
keeps ONLY the scalar `rate` (no good substitute as a pulse-scan's default y); the cumulative
`rate_sites`/`rate_grid`/`_occ_sum` accumulators are **deleted** — they duplicated `repeat_mode=average`.
The `counts` histogram flattens the WHOLE block (every repeat × site sample), never averaged
(`_signal_then_repeat` skips the reduce for `kind=='hist'`); a `create`-mode sites panel collapses back
to one value/site (`_coerce`). The site-map underlay reduces `frame_judged` (`'replace'`) to one coherent
`(H,W)` shot (`_sites_aux`). Also deleted: the dead third system `Measurement.UPDATE_MODES` /
`_postprocess` / `update_mode` / `repeats` / `_accum` (the base accumulate-then-emit that violated the
block contract; every concrete node already bypassed it). Guarded by `test_processor_repeat_contract.py`
(every processor declares a valid contract, never as a ctor arg; OccupancyProcessor is the canonical
`preserve` and `repeat_mode=average` over `occupied` recovers the per-site probability), `test_scan_repeat.py`,
`test_measurement_output_contract.py`, `test_panel_reshape_orthogonal.py`.

**DECOUPLING (#H3o).** A panel reads EXACTLY the signal it is bound to — there is NO frame-coherence
rewrite. A `frame` panel shows the camera's own block (averaged per `repeat_mode`), INDEPENDENT of any
running Judge; a Judge publishes its OWN keys (`occupied`, `frame_judged`) and never touches `frame`.
The site-map underlay still tracks the rings' shot because `frame_judged` is published ATOMICALLY with
`occupied` (one transform dict) and the map resolves both from the SAME producing node via
`_sites_aux` — a separate path, not a binding rewrite. (Per-shot semantics: the site-map underlay is
the single judged shot, not the 30-shot mean — that is correct for occupancy.) Guarded by
`test_camera_measurement_multitrigger.py::test_2d_frame_panel_shows_camera_average_decoupled_from_judge`.

### Panel board: TOP-LEFT GRAVITY packing over pixel AABBs (#H3s-F8)
There is **no column grid**: the board is a pure pixel plane and `PanelConfig.col`/`.row` ARE the
card's pixel top-left (the `.cols` column-span + `.rows` cell-span are deleted).  A card's WIDTH still
scales with the size (`cols // 2` base widths joined by one `GAP`, so 1x4 is wider than 1x2); its
HEIGHT HUGS the plot — `_card_size` = chrome + the size's own figure height, **zero blank padding**
below at every size.  `_compact(configs, active=None, board_w=None)` is a TOP-LEFT GRAVITY packer:
every card floats UP then LEFT (`_gravity_slot` = top-most then left-most feasible candidate point)
until blocked by another card or the board edge, with a **uniform `GAP` on all four sides** (and as
the margin from the (0,0) origin).  `GAP` = `GRID_UNIT` (the single spacing constant) = the horizontal
inter-card gap the user liked; reuse it, never add a new art/geom knob.  Cards pack in READING ORDER
of their current `(row, col)`, the `active` (just-dropped) card winning a tie — so a card dropped
low-right snaps up-left, dropping it back where it was reproduces the layout (stable + deterministic:
a settled board is a fixed point), and cards sit side by side until the board (`board_w` = the live
scroll-viewport width; a two-wide fallback headless) is full, then wrap.  `_arrange` passes the
viewport width; the drop-release records the raw drop pixel as the seed and lets `_compact` re-pack.
Saved layouts round-trip the reading ORDER (exact pixels are recomputed on load).  Guarded by
`test_board_gravity.py`, `test_panel_grid_spacing.py`, `test_panel_hug_layout.py` (+ the smoke layout
test).

### Setting frame: show-all + grow-not-shrink (height hysteresis)
`_size_settings_popup` sizes the Setting popup to show ALL its content by default (scroll only past
a screen-derived bound), with a per-popup height high-water mark (`_settings_h_hwm`): it GROWS
immediately (incl. when `_on_size` enlarges the panel) but NEVER shrinks back within a session; the
mark resets in `_build_settings` when the popup is rebuilt with different content.  Guarded by
`test_setting_frame_grow.py`.

### FluentTreeComboBox: click the name to expand + popup grows to fit
Both behaviours live inside the reusable widget: `_ExpandableTreeView.mousePressEvent` toggles a
parent (header) row on a click ANYWHERE on the row (name or triangle), exactly once; `_resize_popup_to_contents`
(on `expanded`/`collapsed` + `showPopup`) re-grows the open popup to fit every visible row
(`n_visible × sizeHintForRow`, screen-clamped, forcing view + container `setFixedHeight`).  Guarded
by `test_tree_combo_expand.py`.

### One source MECHANISM everywhere; per-kind only the slot-count + the value SIZE
The signal-picker + `value = ...` expression box (the `SignalExpr` mechanism) is on EVERY source
field — every plot kind (the site map included), the occupancy source, the pulse-scan y. What
differs per plot kind is only (a) whether it can GROW slots (`+signal` / `−signal`) and (b) the
SHAPE its `value` must be. Both are data-driven, never an inline `kind == "sites"` check:
`panel_allows_multi_slot` / `PANEL_SINGLE_SLOT_KINDS` declare the site map single-slot (its ring
centres + frame underlay resolve from `signal[0]`'s producing node, so a 2nd slot is meaningless)
— it keeps the expression box but has NO `+/-`; every other kind is multi-slot. `PANEL_INPUT_FORMAT`
declares the accepted `value` size (sites = a per-site `(N,)` vector, 2D = an `(H×W)` frame,
monitor = a scalar), enforced in `PanelCard._coerce`. Guard: `test_task_cali_modes_and_plot_split`
asserts sites is single-slot (no `add_slot_button`, keeps `source_edit`) while 2D is multi.

### Every `kind="signal"` source is now `signal_expr`
The new `ParamDecl` kind `"signal_expr"` (whitelisted in `measurement.py::ParamDecl.__post_init__`
— `processor.py` only re-exports it) renders the reusable `_SignalExprWidget` (multi-slot
grouped signal picker + `+/- signal` + a `value = ...` editor with the floating `Edit…`). Its
value is `{"inputs": [...], "source": "value = ..."}`. `OccupancyProcessor.source` is one such
field: the node builds a `SignalExpr`, sets `consumes = tuple(expr.inputs)` (so it reacts to the
picked signals), and `transform` judges `expr.evaluate(inputs)` — `value = signal` on `frame` is
the single-frame default, `value = (signal[0] + signal[1]) / 2` averages two. The console's
frame-coherence resolver matches on `name in node.consumes` (the canonical reactive input set).

## 21. The catalog / node / UI / layout base-class framework (#H3r, 2026-06-26)

The five layers (device / measurement / processor / task / plot) are pinned down by base classes that
OWN and ENFORCE the shared contract, so adding a new one is "declare your specifics" and a wrong
addition fails LOUD (at import or at the publish boundary), not silently. Each rule is a pytest
contract test — the single mechanical guard. Change the framework = change the base.

- **Catalog specs** — `operations/_spec.py::CatalogSpec` (frozen, kw_only, dependency-free) is the
  base of `MeasurementSpec` / `ProcessorSpec` / `TaskSpec`. It owns `name`/`params`/`metadata` +
  `param()`/`defaults()` and ENFORCES (`__post_init__`) a non-empty name + a tuple of ParamDecl
  params + UNIQUE param keys; `collision_key()` is ABSTRACT and `__init_subclass__` raises at import
  if a new spec forgets it (the OpenRegistry de-dup rule lives ON the spec, not per-registry).
  Subclass adds only its fields + `collision_key`. Guard: `tests/test_spec_base.py`.
- **Open registries** — `operations/_open_registry.py::OpenRegistry` is the ONE auto-discovery +
  ordered-registration + dedup-by-name + collision machinery; `measurement_registry` /
  `processor_registry` / `task_registry` are thin shells binding the public names
  (`register_*`/`<noun>`/`unregister_*`/`registered_*`) and re-exported symmetrically from
  `operations` AND `neutral_atom`. Guard: `tests/test_registry_public_api_symmetry.py`.
- **Reactive processor node** — `operations/logic.py::Processor`: `provides` (a class fact) is the
  SINGLE output-key source; `output_keys()`/`published_signals()` and the spec's `result_keys`
  derive from it. `shot()` ENFORCES publish-time conformance — a processor may only emit keys it
  declared (an undeclared signal raises, never leaks). `repeat_contract` (reduce|preserve) is a
  static class attr, never a user knob. Guards: `tests/test_processor_output_contract.py`,
  `test_processor_repeat_contract.py`.
- **Acquiring measurement node** — `operations/logic.py::Measurement`: declares `primary_signal`
  (the key carrying the `(repeat,*points_shape,*data_shape)` contract block) and
  `_assert_primary_shape(out)`. Camera / Scanned / PulseScan all `return self._assert_primary_shape(out)`
  from `shot()`, so ANY measurement (incl. a future one) that mis-shapes its block (forgets the
  repeat axis, sets only one of points/data shape) fails LOUD at publish, not as a wrong plot.
  Guard: `tests/test_measurement_output_contract.py`. (The full ring/pass de-dup across the three
  acquiring nodes is intentionally NOT done — the per-pass NaN-clear + the `repeat=0`=∞ roll are
  data-correctness logic that must be reproduced on real frames, not refactored blind.)
- **UI param injection** — `frontend/param_widgets.py::ParamWidgetHandler` (ABC, 5 abstractmethods:
  build/read/write/is_empty/refresh) + `PARAM_WIDGETS[kind]`. The measurement form, plot Setting,
  Edit tab, and `_make_param_widget` ALL dispatch through it; `ParamSpec` is deleted (plot params are
  `ParamDecl` with a `display` data flag), so a ParamDecl kind is rendered/read/seeded/validated in
  ONE place. Adding a kind = one whitelist entry + one handler. Guard:
  `tests/test_param_widget_registry.py`.
  - **Device param declared ONCE — `DeviceProperty` (confocal `ManagedProperty`).** A concrete device
    declares each tunable knob as a `devices/base.py::DeviceProperty` class attribute: the descriptor IS
    the Python property (validated get/set — `float`/`int` clamp to bounds, `bool` coerces, `choice`
    rejects a value outside `choices`, a getter-only prop is read-only) AND auto-registers into the
    device's runtime-control catalog. `BaseDevice.runtime_controls()` DERIVES the whole catalog from the
    MRO-merged `DeviceProperty` descriptors (`collect_device_properties`), so `runtime_controls` is never
    hand-typed and the property + its GUI can't drift. Three kinds -> three widgets:
    `float`->spin box, `bool`->switch, `choice`->combo (see `VirtualRF`: δ/freq/power floats, `drive_on`
    bool, `waveform` choice; `VirtualLaser`: wavelength/saturation floats, `beam_on` bool, `on_d1`
    read-back). A device whose control routes through a backend-specific path (a camera's `exposure`
    setter -> `configure`, its abstract per-backend getter) overrides `runtime_controls` instead — the
    documented escape hatch. Guards: `test_device_runtime_controls.py::test_runtime_controls_are_derived_from_device_properties`
    / `test_device_property_covers_the_three_param_kinds` / `test_device_property_is_the_property_and_validates`.
  - **Numeric widget = spin box, decided on the declaration.** `FloatHandler`/`IntHandler` build a blank
    line edit vs a scrollable `FluentDoubleSpinBox` from the ONE predicate `ParamDecl.blank_allowed`
    (explicit `optional` wins; else infer "no default and not required" = an optional API arg). A device
    knob is never blank, so `RuntimeControl.__post_init__` pins its numeric decl `optional=False` (and
    clears `required`) — every device float/int control is a scrollable spin box BY CONSTRUCTION, an
    author can't declare one that degrades to a line edit. Guard:
    `test_device_runtime_controls.py::test_every_numeric_device_control_renders_a_scrollable_spinbox`.
  - **Row label + unit are one source.** `ParamDecl.row_label()` = `"<label> (<unit>) *"` — the ONE
    thing the config editor, device viewer, measurement Edit, Setting popup, and signal-expr title read
    (no re-typed idiom). A live READ-BACK is engineering-scaled with its unit SI-prefixed via the ONE
    `param_widgets.format_reading` (`6.8e9 Hz -> "6.8 GHz"`, using `qt_fluent.eng_mantissa_prefix`); the
    editable spin box stays plain decimal (confocal never SI-scales its editor).
  - **Wiring rule is one helper.** Every handler routes its change signal through `param_widgets._wire`
    (re-validate + optional instant-apply); the composite `signal_expr`/`pulse_slots` handlers were the
    two that bypassed it. Guard: `test_param_widget_registry.py::test_editable_handlers_route_edits_through_instant_apply`.
- **Plot selection -> scan range (confocal `_read_range`), enforced at the base.** A 1-D plot of a
  scanning measurement's signal wires its area selector to `PanelEditor._read_x_range`, gated on the
  producing node DECLARING a scan axis (`TaskConsole._node_scan_range_key` = the node's first
  `axis_range` param) — so EVERY measurement with a scan range gets the linkage, never wired per node
  (the drift that left `gm_detuning` without it). The selection is staged onto the producing
  measurement's OWN Logic-tab Edit form (`_form_for_node` -> `MeasurementPanel.set_axis_range`); a 2-D
  image plot still routes to the camera node's ROI (`_read_region` -> `region_to_acquisition_parameters`).
  Guard: `test_frontend_smoke.py::test_1d_scan_plot_selection_stages_producing_measurement_range`.
- **Plot kinds** — `frontend/live.py::PLOT_KINDS` (tuple of `PlotKind`: key/cls/label/render_family/
  panel/input_format/input_slots/single_slot) is the single source; `live.plot()` looks the class up
  and task_console's `PANEL_KINDS`/`PANEL_INPUT_FORMAT`/`PANEL_INPUT_SLOTS`/`PANEL_SINGLE_SLOT_KINDS`
  are DERIVED from it (byte-identical to the old literals); `data_figure` reads the declared
  `render_family` (`"auto"` sentinel keeps the site map's conditional 1D/2D). Guard:
  `tests/test_plot_kind_table.py`.
- **Panel layout** — `frontend/task_console.py::GAP` (= `GRID_UNIT`) is the ONE spacing setting;
  `_compact` is a top-left gravity packer (see the board section above) that leaves exactly `GAP` on
  all four sides of every card and as the origin margin, so spacing is uniform, never a leftover-pixel
  gap. Guard: `tests/test_panel_grid_spacing.py`, `tests/test_board_gravity.py`.


## 22. Systematic single-source review — where each "one fact" now lives (2026-06-29)

A 7-wave DRY/decoupling pass turned "the same fact copied in N places, kept in sync by hand"
into ONE structurally-enforced source (a helper / constant / `__post_init__` / contract test).
Behaviour-neutral throughout. The single sources added/relocated, by layer:

**timing**
- `timing.pulse_table.resolve_pulse_template(template, *, default_name, default_factory)` — the ONE
  pulse-template path resolver (the Calibrate task + the Pulse-scan measurement delegate). Anchors
  the `pulses/` lookup to the project root via `_paths.project_path`, NOT a per-caller `parents[N]`
  count.
- `PulseTableState._set_bus_target(bus, period_index, value)` — the ONE "keep a ramp, else force an
  edge" rule + `analog_bus_modes` plan writeback (scan-slot bind / api-field set / slot resolve).
- `timing.pulse_table.scan_target_label` — the ONE STATE-FUL name label for a `(kind, target)`;
  `enumerate_pulse_params` routes every label through it (the GUI dropdown + the scan axis read
  identically). Its complement is the frontend's STATE-FREE index label `pulse_gui.slot_label`.

**devices**
- `RuntimeSequenceProgram.__post_init__` — the ONE enforcement of "a finite K-sweep
  (`scan_repeats>0`) needs >= 2 scan points": on the COMPILED program (the single chokepoint the
  real + virtual backends both build), never the eager `PulseTableState.validate` (which would
  raise on a legitimate GUI mid-edit). Guard: `tests/test_audit_hardening.py`.

**analysis (core / operations / subsystems)**
- `core.analysis.fit_gaussian_spot_2d(data, yy, xx, *, x0, y0, offset0, amp)` — the ONE 2D-gaussian
  spot fit (`_gauss2d` model + bounds + centroid fallback); `psf.fit_site_psfs` + the sub-pixel
  `analysis._refine_center_subpixel` both call it.
- `operations.measurement.otsu_fidelity_from_frames(frames, calibration, site)` — the ONE
  signals -> `(counts, threshold, FidelityEstimate)` pipeline; the live `OtsuFidelityReducer` and the
  held-out reference path in `subsystems/readout.py::_scan_detection_time` both delegate.
- `core.calibration.TrapCalibration.with_thresholds` / `with_method_thresholds` use
  `dataclasses.replace` (re-runs `__post_init__`) instead of re-spelling every field; `to_dict`/
  `from_dict` stay explicit per-field. Guard: the `set(to_dict()) == {fields}` parity assertion in
  `test_neutral_atom_lightweight::test_psf_calibration_serialization_round_trip`.
- `operations.calibration.ALL_READOUT_METHODS` — the ONE readout-method allowlist (defined above the
  first function so `calibrate_sitemap_from_images` validates `method` against it).

**frontend (art-internal; na never imports frontend)**
- `BaseLivePlot._build_distribution_band(image_artist, values, *, n_bins, guide_minmax)` — the ONE
  side clim-distribution band (`Live2DDis` + `LiveSiteMap`): histogram poly + draggable clim lines +
  DragHLine, guide lines drawn (same artist order) only when `guide_minmax` is given. Strictly
  appearance-neutral (rendered-PNG pixel-diff identical).
- `frontend.site_map(centers, occupied, *, image, roi_radius, labels)` — the sealed site-map view
  (a `LiveSiteMap`), registered on the viewer namespace. `views/plots.py::plot_detection_image`
  routes through it, so the occupancy ring art comes only from `style.SITE_OCCUPANCY_STYLE` and na
  hard-codes no ring colours.
- `frontend.data_figure.resolve_save_base(path, stem)` — the ONE figure+npz save-path resolver
  (`DataFigure.save` + the grid `_GridData.save`).
- `frontend.qt_fluent._bound_field_style(*, selector, text, border, fill)` — the ONE scan/api
  bound-field stylesheet recipe (`mark_scan_field` / `set_scan_bound` orange + `set_api_bound`
  violet).
- `task_console`: `_new_panel_card` (the ONE PanelCard provider block), `_repeat_mode_value` (the ONE
  clamped repeat-mode reader), `_kind_repeat_modes`, `_relim` (the ONE `relim` default reader).

### More single sources (the same "one fact, one home" pass, continued)

**timing**
- `PulseTableState._duration_ns_unquantized(*, slots)` (`timing/pulse_table.py`) — the ONE
  evaluate-and-validate prefix every period-duration read shares: `eval_time_expr` + the unit-set
  check + the `"unsupported pulse duration unit"` error literal. `duration_steps` and `duration_ns`
  each call it and then keep their OWN boundary policy (negative-raise / zero-raise / quantize-or-not);
  the shared part is the prefix only — the boundary strategies are a deliberate per-method difference,
  NOT something to merge.
- `PulseTableState._remapped_target(kind, target)` (`timing/pulse_table.py`) — the ONE bracket
  index-remap rule (`duration` numeric target and `dac` `bus@period` target run through
  `_expand_bracket_index`; a `delay` channel-name target passes through). `unrolled_bracket`'s
  scan-slot and api-slot loops both call it, so a finite-bracket compile remaps both identically.
  Serialization stays each slot type's own (`ScanSlot`/`ApiSlot` carry different fields) — only the
  target rule is shared.
- `unrolled_bracket` carries `api_slots` through the rebuilt state, symmetric with `scan_slots`, so
  every API binding (a duration/delay/DAC handle) survives a finite-bracket compile — the bound names
  reach `api_names` / `validate` / server-sync, not only the value baked into the period.

**devices**
- `devices.sequencer._fold_global_delay_shift(channel_raw, bus_raw) -> (ch, bus, G)` — the ONE
  negative-delay global-shift fold: `G = max(0, -min(all raw))`, add `G` to every channel + bus raw
  delay, drop the now-zero entries. The scan and non-scan compilers BOTH call it, so a negative TTL/bus
  delay produces the SAME phase on real hardware whichever path compiled it (the two paths used to
  inline the same algorithm and could drift; virtual mirrors both, so a drift is only catchable by the
  cross-path assert this single source enables).
- `CameraDevice._reject_unknown_configure_keys(allowed, got)` (`devices/base.py`) — the ONE
  unknown-`configure`-key contract: an unrecognised keyword raises a clear `ValueError` listing the
  backend's configurable keys, on BOTH backends (`VirtualCamera` allows `{exposure, roi}`,
  `QCMOSCamera` allows `{exposure, readout_speed, roi}`). A mistyped key fails the same loud way on
  virtual and real instead of one silently swallowing it and the other raising `TypeError`. Guard:
  `tests/test_camera_configure_contract.py`.

**analysis (core / operations / subsystems)**
- `core.calibration.TrapCalibration.readout_exposure(fallback)` — the ONE reader of the
  exposure-self-match invariant (`metadata["threshold_exposure"]`, the gate time the thresholds were
  learnt at). Every readout that must image at the calibration's exposure — `detect`, the live
  calibrate-task adoption, the temperature survival frames — goes through HERE with its own `fallback`,
  instead of each reaching into `metadata` with its own defensive spelling. A threshold is
  exposure-specific, so a missed/mistyped lookup re-floors occupancy / sticks survival at the
  false-positive rate (#issue-2 / #H3v-2); one accessor removes that whole class of drift.
- `core.calibration.READOUT_KINDS` (+ `readout_kind(method)`) — the ONE method->kind table
  (`box`->`box`, `psf`/`uniform_psf`->`kernel`) that `signals()` dispatches on. The readout KIND is
  declared, NOT inferred from a `"psf" in m` substring on the method NAME (a future `matched_filter`
  would miss the substring and be read as box, every count silently wrong). core owns it and never
  imports operations; `operations.calibration.ALL_READOUT_METHODS` validates every offered method is
  registered here.
- `TrapCalibration.roi_radius` / `reducer` are the BOX readout's extraction geometry only — a PSF
  (kernel) calibration sets both to `None` in `__post_init__` (it reads through `psf_weights`/
  `psf_boxes`), so no dead box-only state rides on a PSF calibration. `core.results` rings fall back to
  the box default when a calibration carries no `roi_radius`.
- `operations.measurements._coupled_template.resolve_coupled_template(...)` — the ONE coupled-tier
  template resolver (load-or-default -> compare channels -> rebuild via `imaging_channel_kwargs` ->
  optionally bind a duration scan slot). `temperature` and `readout_duration` both delegate, differing
  only in `default_name` / `default_factory` / `role_keys` / bind target and — made EXPLICIT, since it
  had silently drifted — the `missing_policy`: temperature passes `"raise"` (an operator-named but
  absent file fails LOUD), readout_duration passes `"fabricate"` (a missing/unnamed template falls back
  to its default).
- `operations.fidelity.FidelityReport.SUMMARY_KEYS` — the ONE list of scalar summary keys; `summary()`
  self-asserts its returned dict equals `SUMMARY_KEYS`, and the `readout_fidelity` processor declares
  `summary_keys = FidelityReport.SUMMARY_KEYS` instead of hand-copying the tuple, so a processor's
  declared scalars can never drift from what the report publishes (a drift would trip the
  publish-boundary `_assert_declared`).
- `ProcessorSpec` no longer carries a `consumes` field — the canonical reactive input set is
  `node.consumes` (derived from the node's `source_expr`, see §20), the ONE source the frame-coherence
  resolver matches on; a spec-level copy could only drift from it.

**frontend (art-internal; na never imports frontend)**
- `frontend/_validate.py` — the dependency-free seam (same pattern as `_paths` / `_readout_math` /
  `_viewer_registry`) holding `_strict_bool` / `_positive_float` / `_positive_int` / `_non_negative_int`,
  the ONE place the notebook-facing scalars (a refresh interval, a flag, a count) are rejected the same
  way across `plot()` / `ArrayWatcher` / the acquisition `Session`. The analysis layer keeps its OWN
  validators in `core/analysis.py`; this seam never crosses the sealed boundary.
- `param_widgets.coerce_short_labels(provider)` — the ONE `{full -> short}` label-map normaliser
  (callable-guard, `str()` both ends, drop empties, swallow provider errors to `{}`) every grouped
  signal picker feeds `fill_grouped_signal_combo`, so the signal_expr / plot-Setting-slot / form
  pickers render identically.  The whole grouped-picker cluster (`signal_state` /
  `grouped_signal_items` / `signal_tree_groups` / `fill_grouped_signal_combo` /
  `read_editable_combo` / `coerce_short_labels`) lives in `param_widgets` — the LEAF the console
  imports — so the leaf never lazy back-imports `task_console` (the old cycle);
  `task_console` forward-imports the three it uses.
- `task_console._make_unit_cycle_row(on_click, label_w, *, with_label)` — the ONE x-axis unit-cycle
  row, so the Setting popup and the Edit tab get the identical button width / flush-left idiom / single
  tooltip (`with_label=True` also builds the live current-unit label).
- `live._dist_count_xlim(n)` — the ONE side distribution-band count-axis upper limit (small headroom:
  `peak*1.5`, floored to `peak+5`, never below 10) shared by every band (`Live2DDis` / `LiveSiteMap` /
  `LiveLiveDis`, init and update alike), so the bar column never touches the right edge and the five
  call sites cannot drift into two headroom rules. Appearance-neutral.

## 23. Grey-molasses cooling: virtual `laser` + `rf` + a detuning scan (2026-07-07)

The virtual config has two more devices so an operator can *simulate grey molasses*: a `VirtualLaser`
(a beam LOCKED blue on the Rb87 D1 line) and a `VirtualRF` (the microwave/EOM sideband). Ordinary
registry devices (domains `laser` / `rf`, contracts `LaserDevice` / `RFSourceDevice` in
`devices/base.py`), wired into the trap array by `virtual_config()`:
`"trap_array": {"params": {"laser": "$device:laser", "rf": "$device:rf"}}`.

**The detuning is the RF's, not the laser's.** A laser here has NO tunable detuning — the D1 lock is
fixed. The detuning that matters for grey molasses is the TWO-PHOTON (Raman) detuning δ between the
cooling and repump beams, which the RF sideband PRODUCES. So:
- `LaserDevice` controls = `wavelength_nm` (on-D1?) + `saturation` (power-broadens the dark resonance).
  NO `detuning_gamma`.
- `RFSourceDevice`'s **`two_photon_detuning_gamma` is the primary writable control** (the grey-molasses
  knob): a two-way property whose getter derives δ from `frequency_hz` and whose setter moves
  `frequency_hz` to realise δ. The device owns the δ↔Hz conversion (its Rb87 reference), so a scan just
  writes δ in Γ. `frequency_hz` / `power_dbm` are also writable (raw set-points).

**One physics source** — `grey_molasses_cooling_factor(two_photon_detuning_gamma, saturation, on_d1)` is
a *multiplier on the cooling floor* (`VirtualTrapArray._cooling_floor_K()` multiplies
`cooled_temperature_K` by it; `reload`/`set_occupancy`/`cool` seed temperature from that one method):
- **Relative, not absolute** — factor is exactly `1.0` at the optimum (δ = 0 on D1), so the *default*
  rig (RF on the 6.834 GHz hyperfine line, laser on D1) preserves the calibrated 50 µK floor;
  `test_virtual_atom_physics` / `test_calibration_report_and_noise` do **not** move. Mis-tuning only
  warms. A rig with no laser/RF keeps its plain floor (`_cooling_floor_K` early-returns).
- **Failure modes** — off D1 (`on_d1` false) → no cooling (`GM_HOT_FACTOR`, ×6). δ ≠ 0 → a **Fano**
  feature about δ = 0: steep heating for δ > 0, gentle for δ < 0, power-broadened half-width
  `0.05 + 0.02·s` Γ, so a *fine* δ scan resolves the narrow dark-resonance dip.
- Rb87 anchors are single-source constants in `virtual.py` (`RB87_D1_WAVELENGTH_NM = 794.98`,
  `RB87_D1_LINEWIDTH_HZ = 5.746e6`, `RB87_HYPERFINE_HZ = 6.834682610e9`).

**Scanning a DEVICE control (the real architectural gap this filled).** The scan engine could only
sweep a PULSE slot (`ScanAxis` → `pulse.set_time` / `set_slot`). Grey-molasses detuning is a *device*
knob, so the framework gained a general **`DeviceControlAxis`** (`operations/measurement.py`): it
carries a `write(value)` callable and its `apply` writes the DEVICE (leaving the pulse fixed);
`kind="device"` / `is_time=False` make `ScannedMeasurement` skip the pulse-slot validation + clock snap.
It pairs with **`FixedReleaseRecapturePlan`** (`operations/temperature.py`) — release-recapture at a
FIXED `t_off` (the swept value went to the device, not the pulse). The assembly spine is
**`ReadoutSubsystem.build_device_survival_scan(values, write, *, pulse, t_off_s, label, y_label, …)`** —
the device-knob sibling of `build_temperature_scan`, reusing the SAME `SurvivalReducer` +
`calibration.detect`-only survival path (identical virtual/real). This is now the way to scan ANY device
control against release-recapture, not just detuning.

**The measurement** — `operations/measurements/grey_molasses_detuning.py` (`@measurement(devices=["rf"])`,
auto-discovered; key `gm_detuning`, x=`detuning`, **y=`recapture`** so it does not collide with
Temperature's `survival`). It sweeps the RF's `two_photon_detuning_gamma` (routed through the device's
own validated setter) and release-recaptures at a fixed `t_off`; the recapture rate PEAKS at the optimum
δ (the coldest cloud). Reuses the Temperature measurement's release-recapture template prep
(`_resolve_release_recapture_template` / `_match_imaging_exposure`) so the two cannot drift.

Guarded by `tests/test_grey_molasses_cooling.py` (factor optimum=1.0 / off-D1 / Fano asymmetry /
saturation-broadening / laser-has-no-detuning + RF-owns-it / trap-floor-follows-δ / connect-wires-them /
the detuning-scan recapture-rate peaks at δ=0 end-to-end). Adding two devices bumps the virtual roster,
so `test_device_config_io.py`'s expected name list includes `laser` / `rf`.

## 24. Live-path performance root fixes: native dtype, binned dis, tick budget, drag protocol (2026-07-09)

Root cause of the real-pylon "camera measurement + 2d is very laggy / unable to allocate 18.xxM":
a 1920x1200 float64 frame is exactly 17.58 MiB, and the live path allocated several per tick while
the dis panel additionally ran `np.sort(2.3M)` (Otsu seed) + a raw-sample fit every tick on the GUI
thread. Five orthogonal fixes, each at its own layer:

1. **Data plane = native dtype end-to-end.** `CameraMeasurement`'s finite ring is native-dtype and
   publishes the `(filled, 1, H, W)` slice of repeats that HOLD data — no NaN prefill, no float64
   forcing, no publish-side copy (the hub's `_stored_array` makes the one defensive copy). The
   output contract (`_assert_primary_shape`) now reads: leading axis = repeats holding data
   (1..ring). Consumers (`reduce_repeat` / `facet_cells` / `coerce_panel_value` / `_as_data_y` /
   the console identity path) pass integer blocks through NATIVE — an integer block cannot carry
   NaN sentinels, so the isfinite machinery is skipped and pool/replace are zero-copy views; float
   blocks (a scan's NaN-prefilled array) keep gap semantics unchanged. A `facet=repeat` grid now
   grows cells as repeats fill (the old up-front all-NaN R-deep block was itself the blow-up).
   `_configure_signal_storage` fails LOUD if a node's `output_specs()` raises (the silent fallback
   used to downgrade image streams to the 2048-deep default ring).

2. **dis panel is O(bins), never O(samples).** `fit_histogram_curves(edges, counts, mode)` takes
   ONLY the binned histogram: Otsu in its classic binned form, weighted-bin seeds, bounds from the
   edges. `histogram_binned` adds an exact bincount fast path for small-domain integer samples
   (bin-for-bin identical to `np.histogram`, ~5x at 2.3 MP). Threshold L/R fractions interpolate
   inside the binned counts. Measured: 2.3 MP uint8 dis update with a NEW shot per tick ≈ 29 ms
   (was 170+ ms).

3. **Console tick: per-panel beat honoured + compose time budget.** The "coherence beats the
   throttle" override is gone (a live camera advanced `disp` every shot, so update_ms was dead for
   camera panels); the beat gates WHEN, the coherent clock decides WHAT. The compose phase carries
   `_TICK_BUDGET_FRACTION` of the base interval; overrun panels stay stale and retry next tick from
   a rotating start index (fair degradation) — the Qt event loop always breathes.

4. **Selector-on-live protocol** (`selectors.begin/end_figure_interaction`): every drag freezes its
   own panel's recompose (catch-up on release via the frame key); a blitted panel
   (`fig._zlc_blit_dirty`, maintained by `BaseLivePlot.compose`) forces ONE full draw at drag start
   so widget useblit backgrounds are fresh (no ghost frame under the rectangle). `ZoomPan` batches
   both axis limits per gesture (`fig._zlc_lim_batch`) and calls the figure-registered
   `_zlc_lim_refresh` hook ONCE — half the re-decimation per zoom/pan. The image decimation layer
   itself (`_decimate_image_view`) is load-bearing and unchanged.

5. **VirtualMotCamera is a faithful pylon twin** (1920x1200 @ 0.05 s, Mono8): elliptical 40x20 px
   FWHM spot at peak≈93 counts over offset 7 + read noise, and free-run mode runs a REAL producer
   thread at the exposure pace with a latest-frame slot (Basler LatestImageOnly) — a slow consumer
   genuinely drops frames, so the console's amber "display behind" advisory behaves identically on
   virtual and real rigs. Render ≈ 35 ms/frame < the 50 ms pace.

Plus two architecture items from the same round: `RoiProcessor`
(`operations/processors/roi.py`: crop + mean/sum/max, region == the 2-D panel selector's pixel
endpoints via `region_to_acquisition_parameters`, publishes native `roi_frame` + scalar
`roi_value` — the stock chain for "select a region, watch its distribution / total counts"), and
the sequencer family converged to **VirtualSequencer / RemoteSequencer (+ ManualSequencer
first-light)** — RuntimeSequencer (a strict subset of Virtual) and VerilogSequencer (production
dead code) are deleted, the service-level wall-clock scan-progress simulation went with them, and
`VirtualSequencer.wait_done` enforces the same deadlock guard + protocol bookkeeping as the bare
service (`WAIT_FOREVER_MESSAGE` single source).

## 25. The render thread: compose off the GUI thread + the selector→ROI gesture (2026-07-09, W round)

The V3 tick budget/rotor only *rescheduled* the GUI-thread render burst; the W round removed it.
`frontend/render_loop.py` is the console's ONE background worker: every steady-tick compose (the
numpy display prep, matplotlib artist updates and the Agg rasterisation) runs there, strictly one
batch at a time.  The GUI thread keeps scheduling (frame-key/beat gates), presents the finished
FRONT buffers together (one coherent shot per flip) and serves interaction.  Measured: GUI `_tick`
207 ms → 1.5 ms with three 2.3 MP live panels; event-loop latency p95 0.5 ms.

**Ownership protocol (the whole thread-safety story; hardened by the audit round `6d6fc53`).**  A
batch owns its figures from `submit` until the GUI **consumes** it (`RenderLoop.deliver_pending`)
— NOT merely until the worker finished computing.  The first cut marked idle before emitting
`job_done`; the adversarial review confirmed the race (a tick could re-submit the same figures
while the finished batch's present/structural pass was still queued — two threads on one Agg
buffer).  Now the worker publishes the result and emits a payload-free `job_done`;
`deliver_pending` (the queued slot AND `barrier()` both land there — whichever runs first wins,
the other no-ops) pops it, marks idle, then runs the consumer.  `barrier()` therefore waits out
the compose AND delivers the batch itself: its caller returns to a settled board with nothing in
flight or pending.  Every GUI path that mutates a live figure holds the barrier first — the canvas
mouse/wheel entries (`EmbeddedFigureCanvas._zlc_wait_render`, hook `_zlc_render_barrier` hung at
the ONE selector apply point), the canvas lifecycle funnel (`draw()`, `_zlc_resync`, dpi-ratio
change, the paintEvent fallback), Setting edits incl. the structure handlers (facet / sub-kind /
size) and the in-place mutators (title / unit cycle / fixed lims), source apply, the coalesced
rebuild, Edit-tab open, panel remove/teardown, `refresh_once`, `_save_board_image`'s
`board.grab()`.  Structural (re)builds (Qt widget creation) are probed by the single dirtiness
rule `_needs_structural_build` and handed back to the GUI pass in `_on_render_batch`
(membership-gated FIRST: a card removed mid-flight is never composed into).  `set_status` defers
itself off-thread (flushed at present) so `compose` never forks per thread.  Compose is Agg-ONLY
end to end: the stock QtAgg `draw()` hides a `self.update()` Qt call, bypassed via
`_zlc_draw_agg_only`; `present()` snapshots the buffer into a FRONT QImage the paintEvent blits,
so an async paint never races the worker; drag interactions drop the front and own the figure for
the drag (`_zlc_present` skips a mid-drag figure — it catches up on release).  **Fairness**: a
beat falling on a busy (or mid-drag) tick is OWED (`card._beat_owed`) and served on the next idle
tick regardless of the beat modulo — a slow-beat panel can never phase-lock onto busy ticks and
starve behind a heavy fast-beat sibling (the deleted rotor's guarantee, without the rotor).
Tests drive ONE deterministic frame with `conftest.tick(con)` (= `_tick()` + `barrier()`): a bare
`_tick()` only SUBMITS, and without a Qt event loop the queued delivery never runs.  Contract:
`tests/test_render_loop_contract.py`.

**W-round data-plane regression fixed first** (`dfb5dde`): `SignalExpr.co_names()` folds the bound
inputs in for version-gating, and the raw-signal detector used it — the identity zero-copy path
NEVER fired and every bound panel float64-stacked the 2.3 MP frame per tick (~52 ms/panel).  The
detector now uses `direct_names()` (the source TEXT's own identifiers); `reduce_repeat` passes a
single integer repeat slice through as the native view.

**Signal-loop guards** (user-reported): a processor could pick its own output as its source —
fresh hub = silent starvation forever, primed hub = a full-CPU republish spiral with a frozen shot
clock.  Three guards, one per scope: `Processor.__init__` rejects `consumes ∩ published_signals()`
(base single source); `TaskConsole._reactive_ring` walks REACTIVE edges over the RUNNING nodes
only (GUI rows and notebook-injected `running_nodes=` processors alike) at each start — the start
that would CLOSE a ring is the one refused, so start order cannot smuggle one in, and a stopped
row's stale stored values can never false-reject a legal start (a pulse-scan's y reading its own
relayed frame stays legal — it is a bounded PULL, never self-sustaining); the processor Edit
picker no longer offers the node's own declared outputs.

**Selector→hub chain completed** (the V6 gap): drawing an area rectangle on a LIVE image panel
(selectors ON) now retargets every RUNNING region-capable processor consuming that signal through
the thread-safe apply path, or — when none exists — CREATES the stock `ROI crop` row seeded with
the signal + rectangle and STARTS it (`_on_panel_area_select`; `roi.region_values` is the one
endpoint→params mapping).  **Coordinate frames** (audit fix): the selector yields AXIS
coordinates, and a panel's axes carry the PRODUCING node's declared `region` origin — a consuming
`RoiProcessor`'s region params are the frame's OWN pixels, so `_forward_area_select` subtracts the
panel's `_roi_built` origin before forwarding (an ROI-of-ROI / hardware-sub-array drag crops the
drawn box, not a clamped corner sliver).  A retarget also marks the layout dirty and reseeds an
already-open Logic Edit tab.  Fit results are hub signals: `FitProcessor` ("Fit center") publishes
`fit_x0/fit_y0/fit_amplitude/fit_size/fit_offset` scalars per shot — in the CONSUMED frame's own
pixel coordinates — from the ONE `_readout_math.gaussian2d_center` model `DataFigure.center` also
uses (an overlay fit on a region-declaring source reports axis pixels; they differ by the origin).

**Persistent status strip**: `qt_fluent.FluentStatusStrip` (severity dot + eliding message +
optional action) replaces both the header summary label an error used to overwrite and the
transient orange task banner whose show/hide shifted the layout.  One priority ladder in
`_update_summary`: node error > running task (+Stop action) > display-behind advisory > idle
summary.  figure_viewer mounts the same strip.
