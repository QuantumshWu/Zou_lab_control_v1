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
section 7 for the build commands.

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
(`references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc`,
62 controllable outputs, fallback `ch00..ch61`). GUI visibility is a view
operation only; the server always pads to full hardware width and zeros
hidden/unconfigured channels.

Key fixed facts:

- Default FPGA clock is **50 MHz**, so one tick is **20 ns**. If measured pulses
  are 2x the set value, something still assumes 100 MHz — fix the
  control/server/GUI clock, not the pin map.
- Camera-imaging preset trigger is `ch11/emCCD/M13`. The XDC also defines
  `ch06/trig/R17`; that is a separate output, not the preset trigger.
- Camera-imaging visible subset: `ch09 trap (M17)`, `ch00 cooling (F15)`,
  `ch03 probe (N15)`, `ch11 emCCD (M13)`.
- Four 10-bit analog buses: `da_dipole`, `da_bias_y`, `da_bias_x`, `da_bias_z`.

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
re-sweeps a streamed scan via a host background refill thread; the inter-sweep
seam is a brief safe hold.

## 4. N-Slot Scan Model (current design)

This replaced the old `x`/`y` affine scan. **There is no more `x`/`y` notion.**

- Any per-field value (period duration, channel delay, analog-bus DAC value) can
  be bound to a named scan **slot** `s0, s1, ...` in bind order, via the GUI
  scan dot or `state.bind_field(kind, target)` where `kind in
  {"duration","delay","dac"}`.
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
- Time expressions in durations/delays may only be affine in slots (numbers,
  `s0..`, `+ - * /`, parentheses); the compiler rejects non-affine scan timing.
- Duration, delay, and DAC value can scan in any combination and seamlessly: the
  analog-bus segments carry affine start/stop ticks (same `effective_tick`) and a
  dual `value_select` so a ramp can scan BOTH endpoints (scanned-A -> scanned-B)
  and an edge/hold segment can track a scanned DAC code.
- The host validates global effective-tick monotonicity before upload
  (`validate_pulse_streamer_program`); a scan that reorders the merged edges is
  rejected, not silently dropped.

Anti-patterns: do not expand a scan grid into many GUI columns or many prepared
pulse tables when one ordered `scan_table` describes it; do not re-introduce
separate `x_array`/`y_array` objects.

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
`#77AADD`, radius 4px, soft card shadows. `FluentGroupBox` is the white card with
grey title pill — keep the no-border soft-shadow frame; do not add hard outlines
or replace the editor with plain tables.

### Pulse GUI Layout Contract

`PulseSequenceEditor` has **Edit / Preview / Scan** tabs.

- Edit: `Channel Names and Duration`, `Delay & Scan`, horizontal period-card
  timeline (not a grid), bottom `Control Buttons`, `Channel View`. The
  channel-name column, delay column, and period checkbox columns share **one**
  vertical scroll. Period cards may scroll horizontally; they must not have
  independent vertical scrollbars. For the same channel, raw-label, delay edit,
  and first period checkbox must share `mapTo(editor, QPoint(0,0)).y()`.
- A scan dot next to a duration/delay/DAC field binds it: the field turns orange
  and disabled and shows its slot number. The Scan tab lets the user write
  Python that assigns an `N_points x N_slots` array to a `scan_table` variable
  (namespace has `np`, `math`, `n_slots`).
- Preview reuses `frontend.plot(..., kind="pulse")`, redraws on tab open / `Show
  off rows` toggle (no manual refresh button), plots the **unexpanded** period
  table, y-axis labels are channel display labels (no `Pulse` title), repeats
  use `×∞` / `×N` (never the literal `inf`), slot-bound regions are drawn as
  spanning translucent markers, analog-bus rows are hollow stair-steps.
- Saving writes the bundle together: pulse `.json` + preview `.png` +
  `<stem>_scan.npy` (when a scan table exists).
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
`COEFF_FRAC_BITS=8`, `RD_LAT=2`, `FIFO_DEPTH=3`) and
`zlc_pulse_streamer_top.v` (top, `NUM_SLOTS=4`, 62 channels + 4x10-bit DAC buses,
`axi_bram_ctrl` + CTRL regfile + region-decoded BRAMs, no VIO). The host-side
image packer + cycle-accurate engine model are in `fpga/pulse_streamer/host/`
(`image.py`, `engine_model.py`); `infer_xdc_*` + `validate_pulse_streamer_program`
stay in `Zou_lab_control/neutral_atom/devices/fpga_pulse_streamer.py`.

Default profile (from `host.image.StreamerParams` / `solve_capacity` on the 35T):
`CHANNEL_COUNT=62`, `MAX_EDGES=4096`, `BANK_SIZE=2048` (4096 resident scan points,
UNBOUNDED via streaming), `NUM_SLOTS=4`, `TICK_WIDTH=32`, `COEFF_WIDTH=16`,
`COEFF_FRAC_BITS=8`, `RD_LAT=2`, `FIFO_DEPTH=3`, `CLOCK_HZ=50e6`. The edge tables
live in three parallel block RAMs (tick 32b / coeff 64b / mask 62b, forced
`READ_LATENCY_B=2`); the scan window is one BRAM; the bus segment tables stay in
LUTRAM (distributed) because the bus/ramp engine reads them combinationally each
tick. Vivado `report_utilization`/`report_timing_summary` are the final authority;
the Python estimate is a budget guide (RAMB36 78%, LUT 26%, FF 12%, DSP 9%).

The minimal pulse width AND resolution is **1 tick (20 ns)**: a depth-`FIFO_DEPTH`
(=`RD_LAT`+1=3) continuous edge prefetch hides the 2-cycle BRAM read latency, so
back-to-back 1-tick edges fire one per clock. Four gapless reload sites
(start / loop-rewind / scan-advance / repeat) reseed the FIFO with `FIFO_DEPTH`
shadows at every boundary, so the last edge of point k and the first edge of point
k+1 are adjacent with no gap. Cycle behavior is proven by
`host.engine_model.rtl_mirror_play == reference_play` at read latency 1/2/3 + 200
fuzz programs (no Verilog simulator in repo).

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
`fpga\build\pulse_streamer`.

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
A depth-`FIFO_DEPTH` (=`RD_LAT`+1=3) continuous prefetch issues one BRAM read per
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
refill thread; within a sweep it is gapless, the inter-sweep seam is a brief safe
hold.

**Pieces (all in-repo, all Python-verified pre-hardware, no Verilog sim):**

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
  scanned ramp endpoints. This is the pre-hardware proof (no Verilog simulator is
  in the repo).
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
  `fpga/build/pulse_streamer`).

**Capacity (35T `xc7a35tfgg484-2`, target `<=90%`):** 62 digital + 4x10-bit DAC;
20 ns tick (1-tick min width AND resolution); `NUM_SLOTS=4` affine slots
(delay/duration/DAC value, any combination, seamless); **4096 edges + 4096
resident scan points + UNBOUNDED streaming**; RAMB36 78% (LUT 26%, FF 12%, DSP
9%) from `solve_capacity`. The bus segment tables are LUTRAM (distributed), not
RAMB36, because the bus/ramp engine reads them combinationally each tick.

## 7. Building The Manuals

```powershell
python -c "from Zou_lab_control.neutral_atom.notes import build_main_manual, build_fpga_manual, build_device_manual; build_main_manual(); build_fpga_manual(); build_device_manual()"
python -c "from Zou_lab_control.frontend.notes import build_frontend_manual; build_frontend_manual()"
```

Each builder generates example figures into `assets/`, fills the `.texbody`
template, writes the `.tex` wrapper, and runs `render_tex_pdf` (XeLaTeX, 2-pass,
in a temp dir). XeLaTeX must be on PATH (or pass `xelatex=`). A failed build
leaves only a `.build.log` next to the target PDF.

## 8. Verification

Tests are owned by another agent; see `tests/README.md` for the scoped matrix.
Prefer the smallest scoped check that covers the edited boundary; use full
`pytest -q` only for broad handoff. Typical doc-adjacent checks:

```powershell
pytest -q tests\test_frontend_smoke.py -k "render_tex_pdf or pulse_gui"
pytest -q tests\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract or scan"
python -m json.tool tutorials\neutral_atom_hardware_quickstart.ipynb > $null
git diff --check
```

## 9. Framework Review (architecture assessment)

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
  DAC value + duration + delay scan together — see §4).

Targeted improvement backlog (priority order; none blocking):
1. **Sequencer class roles.** Five classes (`Virtual/Runtime/Manual/Remote/
   Verilog` Sequencer) serve distinct roles but the naming/role split is easy to
   confuse. Document the decision table (now partly in the device manual);
   consider a short `docs` table or clearer names. No behaviour change needed.
2. **Calibration ↔ device binding.** `TrapCalibration` lives on the session, not
   the camera; swapping cameras can leave a stale calibration. Low-risk guard:
   record `grid_shape`/`reducer`/`ordering` in `metadata` and validate against
   the trap array on `detect`. (`detect` already uses the stored reducer/radius,
   so the train/infer reducer mismatch is not actually reachable via
   `TrapCalibration.detect` — only via calling `detect_atoms` directly.)
3. **Virtual sitemap hidden behaviour.** `VirtualCamera` keys "all sites loaded"
   off the sequence *name* `"sitemap"`. Replace with an explicit
   `force_all_sites` parameter so virtual ≠ real behaviour is not silent.
4. **Calibration schema version.** Add a `schema_version` to the
   `TrapCalibration` payload so old `.npz`/`.json` can be migrated safely.
5. **Exposure source of truth.** The sequence's probe width is the truth; assert
   the camera exposure matches it after `acquire` to catch silent drift.

These are recorded so future work is guided; the user explicitly accepts that
many concrete neutral-atom devices/experiments are not implemented yet — the
skeleton, contracts, and docs are the deliverable.
