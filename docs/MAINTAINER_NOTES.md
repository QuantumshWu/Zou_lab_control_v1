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
  -> fpga_pulse_streamer persistent Vivado-session backend
  -> Vivado/VIO upload of ticks/masks/coeffs/scan_points/bus_segments
  -> zlc_pulse_streamer_top_address_switch.bit on the FPGA
```

The FPGA side infers the full hardware contract from the address-switch XDC
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

`prepare` holds reset high, writes rows via `prog_we` toggle, releases reset; it
does not start. `fire` sends an explicit `start` low-high-low; only the
synchronized rising edge is a start event. After that, the FPGA clock owns edge
timing. `wait_done` polls `done`; `safe_state` resets outputs.

Differential upload: after one successful prepare with the same channel order
and clock, later prepares rewrite only changed edge rows, changed scan points,
and the shadow-critical rows `0`, `loop_start_index`, and the final row.

The historical `legacy_address_switch` module remains for comparison only. Do
not make it the default path in tutorials.

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
  `effective_tick = base + (sum_j coeff_j * slot_j) >> COEFF_FRAC_BITS` and
  iterates scan points seamlessly. `RuntimeSequenceProgram` schema carries
  `slot_count/slot_kinds/tick_slot_coeffs/loop_end_slot_coeffs/scan_points/
  scan_coeff_frac_bits` plus ticks/masks/bus_segments.
- Time expressions in durations/delays may only be affine in slots (numbers,
  `s0..`, `+ - * /`, parentheses); the compiler rejects non-affine scan timing.
- The compiler rejects a scan if edge ordering would change across scan points;
  split into chunks or prepare one pulse per point. `dac` slots cannot currently
  share an upload with the analog-bus segment path.

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

RTL: `fpga/pulse_streamer/zlc_pulse_streamer.v` (core, `parameter NUM_SLOTS`,
`COEFF_FRAC_BITS=8`) and `zlc_pulse_streamer_top_address_switch.v` (top,
`NUM_SLOTS=4`, 62 channels + 4x10-bit DAC buses + VIO). Python generator,
validator, and Tcl are in
`Zou_lab_control/neutral_atom/devices/fpga_pulse_streamer.py`.

Default profile: `CHANNEL_COUNT=62`, `MAX_EDGES=1024`, `MAX_SCAN_POINTS=1024`,
`NUM_SLOTS=4`, `TICK_WIDTH=32`, `COEFF_WIDTH=16`, `COEFF_FRAC_BITS=8`,
`CLOCK_HZ=50e6`, `RESOURCE_TARGET=70%`. Big tables (`tick_mem`, `mask_mem`,
`coeff_mem`, `scan_value_mem`) are `ram_style="distributed"` (LUTRAM) because the
run path uses async single-row reads. To shrink on the 35T: lower `MAX_EDGES` /
`MAX_SCAN_POINTS`, or migrate the big tables to a BRAM-friendly synchronous-read
pipeline. Vivado `report_utilization`/`report_timing_summary` are the final
authority; the Python/Tcl estimate is only a budget guide.

Packed VIO probe contract for `NUM_SLOTS=4` (27 `probe_out`, 2 `probe_in`):
`prog_tick_coeffs = 64b`, `scan_prog_values = 128b`, `loop_end_coeffs = 64b`.
The full table is in the FPGA manual; keep it in sync with the top wrapper's
`vio_0` instantiation.

Build/program workflow on the Vivado computer:

```powershell
fpga\build_and_program.bat --check     # no-board HDL/VIO width + capacity self-check
fpga\build_and_program.bat             # build + program
fpga\run_server.bat --check-config     # print resolved project/bit/ltx/xdc/clock/capacity
fpga\run_server.bat                     # start persistent server
```

Configurable before build: `ZLC_PS_RESOURCE_TARGET_PCT`, `ZLC_PS_MAX_EDGES`,
`ZLC_PS_MAX_SCAN_POINTS`, `ZLC_PS_XDC`, `ZLC_PS_VIVADO_BIN`. Vivado 2019 debug
cores are path-length sensitive — keep the checkout short (`D:\ZLC`). The
printed `ZLC build root` / `ZLC project dir` are the source of truth for
`.xpr/.bit/.ltx`; the default project is `fpga\build\address_switch`.

### Edge-table loader over JTAG-to-AXI (current architecture)

The board target is the **validated affine edge-table engine**
(`fpga/pulse_streamer/zlc_pulse_streamer.v`) driven over **JTAG-to-AXI** by an
on-chip **program loader** (`zlc_axi_program_loader.v`).  This replaces the
short-lived per-channel run-length detour, which was **discarded**: on a single
BRAM read port, re-arming 62 channel players at every repeat/scan boundary costs
~23.6 us and cannot be seamless for short passes.  The global edge-table engine
has ONE edge pointer, so repeat = single-cycle pointer/shadow reload and
scan-point advance = single-cycle slot-vector reload -> both gapless, and it is
*cheaper* in LUTs (one comparator + one affine MAC, not 62 players).  This
matches how real pulse streamers are built (Swabian = one global RLE instruction
stream + hardware loop; SpinCore PulseBlaster = one global instruction stream +
LOOP opcodes).

Why a loader instead of changing the engine: the engine is byte-for-byte the
already-validated, host<->RTL tick-verified `zlc_pulse_streamer.v` (seamless loop
at lines ~470-475, single-cycle scan-point advance at ~477-493, N-slot affine
`zlc_effective_tick`, DAC-value scan via bus `value_select`).  Instead of VIO
probes, the loader copies the program out of the AXI BRAM into the engine's
`prog_*`/`scan_prog_*`/`bus_prog_*` ports (the engine's own write contract:
toggle `prog_we` while reset is held; shadow regs latch from `prog_count`/
`loop_start_addr` during writes), releases reset, and pulses start.  The gapless
behaviour and tick-exactness therefore carry over unchanged; only the *delivery*
of the tables changed.

**Pieces (all in-repo, all Python-verified pre-hardware):**

- **Image layout / packer (`devices/edgetable_image.py`).** Packs a
  `RuntimeSequenceProgram` into 32-bit BRAM words: a CTRL block (magic, COMMAND/
  STATUS mailbox, prog_count, scan_count, loop_*, bus_counts, slot_count) at
  offset 0, then the EDGE table (tick + N-slot coeffs + 62-bit mask, 5 words/edge),
  SCAN-point table (N_slots words/point), and per-bus BUS-segment table (affine
  start/stop ticks + coeffs + value/mode/value_select, 7 words/seg).  Only used
  rows are emitted (sparse), so a typical program is a few hundred words.  Region
  bases are fixed at the maxima (1024 edges / 1024 points / 4x64 bus segs); total
  span ~11040 words << the 32768-word (32/50 RAMB36) program BRAM.
  `unpack_program` is the loader-walk decoder; `pack->unpack == program` is the
  contract (test `test_edgetable_image_*`).
- **Loader FSM (`zlc_axi_program_loader.v`).** Sequential, NOT timing-critical
  (runs at prepare).  Polls COMMAND (rising-edge detected: LOAD/FIRE/SAFE/RESET),
  on LOAD walks the image exactly like `unpack_program` and drives the engine's
  toggle-write ports (holding each row stable, settling each BRAM read for
  latency 1 OR 2), then on FIRE releases reset + pulses start, writing STATUS
  (LOADED/RUNNING/DONE/ERROR) back to BRAM for the host to poll.  A
  `repeat_forever` program never asserts DONE, so the FSM keeps polling commands
  (so SAFE/RESET/LOAD still work while it runs forever).  RTL constants are
  contract-tested against `edgetable_image` (`test_edgetable_loader_rtl_constants_*`).
- **Loader co-sim (`devices/edgetable_loader_model.py`).** No Verilog simulator in
  repo, so the loader FSM is mirrored cycle-accurately in Python and its captured
  engine writes are checked to reconstruct the decoded image
  (`test_edgetable_loader_fsm_cosim_reconstructs_program`, incl. a 200-edge stress
  + no-scan/no-bus cases).  This is the pre-hardware proof the new sequencer walks
  edges/scan/bus in the right order with correct addresses + field slices.
- **Top (`zlc_pulse_streamer_loader_top.v`).** `jtag_axi_0` -> `axi_bram_ctrl_0`
  (AXI4-Lite, single-port-BRAM, BMG EXTERNAL) -> `blk_mem_gen_0` (TDP): port A =
  AXI image upload + mailbox, port B = loader.  Loader -> engine; engine `out`
  (62) + `bus_out` (40, DACs now real) -> the exact 62-pin address_switch board
  map.  Build = `create_project_loader.tcl` (strict 32768 BRAM depth + readback
  guard, short project name `l` for the Vivado path-length fix);
  `program_fpga_loader.tcl` leaves the jtag_axi core discoverable as a `hw_axi`.
- **Host session (`devices/axi_session.py :: VivadoAxiStreamerSession`).** One
  persistent `vivado -mode tcl`; `prepare` packs + uploads the image then drives
  LOAD (waits STATUS_LOADED); `fire` drives FIRE; `wait_done` polls STATUS_DONE
  (treats RUNNING as success for repeat_forever); `safe_state` drives SAFE.  The
  COMMAND mailbox always writes 0 before a command for a clean rising edge.  The
  Tcl executor is injectable, so the full pack->upload->loader-model->LOAD->fire
  flow is unit-tested without Vivado (`test_vivado_axi_session_*`).  `run_server.bat`
  default backend is `jtag-axi`; `build_and_program.bat` builds + programs the
  loader bitstream (project `fpga/build/l`).

**Capacity (35T, current build):** 62 digital + 4x10-bit DAC; 20 ns tick;
NUM_SLOTS=4 affine slots (delay/duration/DAC any combination); up to 1024 edges +
1024 scan points in the LUTRAM engine tables (the image BRAM holds far more).
on_pulse -> first output = 1 clock (20 ns); the <100 ms budget is upload-only, and
re-firing an already-loaded program is a few control writes.  Bigger edge/point
counts (BRAM-resident engine tables + a 1-edge prefetch pipeline, and burst /
AXI4-full upload to keep a cold load < 100 ms) are the documented v-next; the
current single-beat upload is correct but ~sub-second for a few-hundred-word cold
load, then unlimited fast re-fires.

**Removed (no legacy residue):** `zlc_runlength_engine.v`,
`zlc_pulse_streamer_runlength{,_top}.v`, `create_project_runlength.tcl`,
`program_fpga_runlength.tcl`, and the per-channel host modules `devices/runlength.py`,
`devices/axi_map.py`, `devices/axi_transport.py`.  The VIO edge-table path
(`zlc_pulse_streamer_top_address_switch.v` + `VivadoPulseStreamerSession`,
`--backend vivado-session`) is kept as an alternate way to drive the *same* engine.

### Architecture-D: BRAM tables + depth-1 prefetch (2048 edges + 4096 points)

The loader path is LUTRAM-bound (~1024 edges/points = the 35T's ~400 Kib
distributed RAM).  To reach **2048 edges + 4096 scan points at <=75%** (the
user's "②+④" goal), the **edge + scan tables move to BLOCK RAM** while the bus
segment tables **stay in LUTRAM** (the bus/ramp engine reads them combinationally
every tick — moving them to BRAM would break that).  This "D" choice was
adversarially reviewed as optimal over moving the bus tables too.

* **Engine `zlc_pulse_streamer_d.v`**: edge 0 comes from the `first` shadow
  (instant), every later edge is a **depth-1 prefetch** read into `pre_*` (valid
  `RD_SETTLE` cycles after edge_index advances).  Same-cycle fire (`cur` is a
  register).  Cost: **min edge spacing = RD_SETTLE+1 = 3 ticks (60 ns)** — not a
  core need (repeat/scan are still gapless), host-enforced.  *Proven* by the
  cycle model `devices/edgetable_engine_model.py:prefetch_d1_play` == the
  validated combinatorial `reference_play` for spacing >= 3, STALL otherwise
  (`test_edgetable_d1_prefetch_*`, cross-checked vs an independent brute force).
  A depth-(latency+1) FIFO variant (`prefetch_play`) is also proven for 1-tick
  spacing if a future build wants it.
* **Top `zlc_pulse_streamer_d_top.v`**: reuses the proven `axi_bram_ctrl` for the
  AXI handshakes + a simple combinational write-address decoder to the asymmetric
  edge BRAM (32b write / 256b read), scan BRAM (32b / 128b), bus-image BRAM, and
  a CTRL regfile + bus mini-loader (copies the bus image into the engine LUTRAM).
  Build `create_project_d.tcl` (defensive IP config), program `program_fpga_d.tcl`.
* **Parameterisation `edgetable_image.solve_capacity(part, channel_count)`** is the
  single source of truth: derives max_edges/max_scan_points/addr widths/BRAM
  depths from the part's RAMB36/LUT budget at <=75%; 35T -> 2048 edges + 4096
  points + 512 streaming window = 62% RAMB36.  Host: `pack_program_d` +
  `VivadoAxiStreamerSession(variant="d")` (enforces min spacing); server
  `ZLC_PS_VARIANT=d`; bats `set ZLC_PS_VARIANT=d` builds `fpga/build/d`.
* **Status:** engine algorithm + capacity + host image round-trip + structural
  contracts are Python-tested; the multi-BRAM AXI integration is BLIND (no
  Verilog sim) and needs **on-board bring-up** (synth via create_project_d.tcl,
  grep `ZLC IPDUMP`/`ZLC TRY-FAIL` to converge version-specific IP props, then
  scope on the bench).  This is the standard no-sim RTL reality, not a shortcut.

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
plotting) is clean and the three backends share one session surface. The
remaining items are targeted robustness/clarity improvements, not redesigns.

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
