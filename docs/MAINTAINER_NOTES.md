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
  edges, analog-bus segments, and delay lanes; the engine HDL
  (`zlc_edge_streamer.v`) has no camera/acquire/readout/detect logic at all. A
  trigger channel is just one more digital output the player drives — the decision
  about *when* to count or threshold lives in the acquisition/feedback subsystem
  (`subsystems/`, readout), not in timing. Do not push exposure/threshold/feedback
  decisions down into the sequencer; keep playback and acquisition decoupled.

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
  `fpga/build/ps` -- the short name "ps" keeps Vivado's deep run/.Xil temp path
  under the Windows MAX_PATH limit while the build stays in-repo).

**Capacity (35T `xc7a35tfgg484-2`, target `<=90%`):** 62 digital + 4x10-bit DAC;
20 ns tick (1-tick min width AND resolution); `NUM_SLOTS=4` affine slots
(delay/duration/DAC value, any combination, seamless); **4096 edges + 4096
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

## 8. Unbounded Per-Channel OUTPUT Delay (the membership delay line)

A channel delay is a **physical OUTPUT delay**, not baked into the edge ticks:
`output_delayed[t] = output_undelayed[t - d]`, **zero before fire**, with the loop period
preserved at the frame length `T`. The edge table is emitted UNDELAYED (every channel at
its nominal position); the delay rides `channel_delays` / `delay_channels`. This is the
literal physical delay -- **ANY** length (positive, negative, zero, arbitrarily large or
small), it never disturbs another channel, and the first frame is REAL (silent until
`t = d`, no cyclic wrapped-in tail). The compilers are
`compile_pulse_table_runtime_program` -> `_pulse_table_edge_table` (constant frame) and
`compile_pulse_table_scan_runtime_program` -> `_pulse_table_affine_delay_channels` (scan),
both in `sequencer.py`.

THE KEY DESIGN -- no buffer, no cap. The RTL (`zlc_edge_streamer.v`) does **NOT** buffer the
delayed signal. A buffer of depth `N` would cap the frame period at `N` ticks; the old
design used a per-channel SRL ring of depth `DELAY_DEPTH=2048` and so capped `T <= 2048`.
Instead, each delayed channel stores its **own** undelayed ON intervals `[a_i, b_i)` over
`[0, T)` in a tiny per-channel LUTRAM (`del_iv_start_mem` / `del_iv_stop_mem` +
`ram_style="distributed"`, with affine tick coeffs so a scanned DURATION moves the
interval), and produces the delayed bit COMBINATIONALLY each tick by **evaluating
membership** at the shifted phase:

```
shifted = (time_count - off) mod T        # off = d mod T, the FULL phase (TICK_WIDTH)
bit     = OR_i (a_i <= shifted < b_i)      # over the channel's resolved ON intervals
out_bit = started ? bit : 0
```

No buffer -> `T` (and the delay) are **UNBOUNDED**. `off` is now the full `TICK_WIDTH` phase
(not a ring index), the modulo is a single conditional `+T`, and `skip = floor(d/T)` whole
periods are handled by a startup gate -- so the delay LENGTH is unbounded too.

Startup gate (silent first frame, opens at EXACTLY `t = d`). Purely COMBINATIONAL from the
engine's own `time_count` and a per-player `del_frame_idx` (frames elapsed since fire, ++ at
each gapless seam) vs the targets `del_skip` / `del_off`:

```
started = (frame_idx > skip) || (frame_idx == skip && time_count >= off)   # == t >= d
```

No decrementing counter, no off-by-one. `off==0 && skip==0` (zero delay) => `started` from
`t=0` and `shifted == time_count` => passthrough.

Merge (unchanged shape): `out = (state_mask & ~delayed_mask) | delayed_out`. The delayed
channel is cleared from the undelayed mask and re-driven by its membership result.

Negative / zero / huge. A NEGATIVE delay re-translates the WHOLE frame: the host folds the
global shift `G = max(0, -min(delays))` into EVERY channel's delay (a causal delay line
cannot lead), so every `off`/`skip` handed to the player is `>= 0`. Zero is passthrough.
A huge delay is just a wider `skip` (a 32-bit-plus counter).

Host -> FPGA contract (`fpga/pulse_streamer/host/image.py`). Per delayed channel the CTRL
regfile carries `delay_count` / packed `delay_bits` / per-channel `delay_off` (one 32-bit
word -- the full phase) / packed `delay_iv_counts` / per-channel `delay_skip`; the ON
intervals (+ affine coeffs) go into the **DELAY image** region (one of the 6 BRAMs). The
top's delay mini-loader copies the image rows into the engine's per-channel interval LUTRAM
via `delay_prog_*` (a `prog_we`-toggle loader, exactly like the bus-segment loader). `off`
and `skip` are computed on the host from `T = loop_end_tick` (constant `T` => constant
`off`/`skip`; the affine ON intervals carry the per-scan-point period shift via the shared
MAC `zlc_effective_tick`). There is NO HW divider.

DAC buses. A DAC value reaching the output via the TTL-folded path (fixed bus values,
`fold_analog_buses`) inherits the membership player directly -- each bus bit is a delayed
TTL channel, so its delay is unbounded with identical capability. A DAC value reaching the
output via the value-engine / `value_select` path (a scanned DAC value, OR a fixed bus value
emitted as bus segments) is now ALSO delayed by an UNBOUNDED membership BUS-delay player --
the DAC-value counterpart of the TTL player. The bus SEGMENTS stay at their NOMINAL phase
(the delay is no longer baked into the segment ticks, which capped it at one frame and
rejected `d > T`); the per-bus delay is carried as a separate `RuntimeBusDelay`
(`bus_index`, `delay`), packed into the `BUS_DELAY_*` CTRL words as `off = d mod T` /
`skip = floor(d/T)`, and the engine produces the delayed value by EVALUATING the bus value
at the shifted phase `(time_count - off) mod T` (RTL `zlc_bus_value_at`: walk the segments,
hold the active one, resolve literal/`value_select` value, closed-form ramp staircase),
gated by the SAME `skip`/`off` startup as the TTL player -- NO buffer, so a scanned DAC value
is delayable by ANY amount, incl. `> one frame`, positive/negative-via-G/zero. The per-bus
delay shares the SAME global shift `G` as the TTL channels. Proven by
`engine_model.bus_value_at` (== `bus_play` undelayed) + `engine_model.rtl_bus_delay_play`
(== `membership_bus_delay_play` delayed) in
`test_bus_value_at_combinational_equals_bus_play_undelayed`,
`test_rtl_bus_delay_player_no_cap_extreme_battery`,
`test_scanned_dac_value_delayed_beyond_one_frame_compiles_and_streams`,
`test_image_bus_delay_ctrl_packing_roundtrip`, and the RTL structure lock
`test_edge_streamer_has_unbounded_bus_delay_path`.

Capacity. The per-channel interval tables are LUTRAM (+0 RAMB36); the only RAMB36 cost is
the small DELAY image staging BRAM (`blk_mem_gen_delayimg`, 8*8*6 = 384 words = 1 RAMB36,
the 6th BRAM). `NUM_DELAYS=8` delayed channels, `MAX_DELAY_INTERVALS=8` ON intervals each
(`DEFAULT_NUM_DELAYS` / `DEFAULT_MAX_DELAY_INTERVALS` in `fpga_pulse_streamer.py`, matching
the RTL params). `validate_pulse_streamer_program` enforces only `count <= num_delays` and
`intervals <= max_delay_intervals` -- there is **NO** frame-period / delay-length cap.
The per-bus DAC delay is likewise enforced only as `count <= bus_count` and a valid
`bus_index` -- **NO** frame-period / delay-length cap on a DAC value either (`off`/`skip`
in `BUS_DELAY_*` CTRL words, the bus value evaluated at the shifted phase, no buffer).

Tick-0 seed anchor (unchanged). The engine seeds its time counter from edge 0, so the
UNDELAYED edge table must begin at tick 0 at every scan point; every compiler path prepends
an all-off tick-0 edge and `validate_pulse_streamer_program` backstops it. (Because the edge
table is undelayed, no channel reorders -- the delay is entirely on the output -- so the old
"delay lane" / reordering machinery is gone.)

Proof (no Verilog simulator). `engine_model.delay_line_reference` is the exact buffered
ground truth; `phase_offset_play` and `membership_delay_play` / `membership_bus_delay_play`
are the no-buffer realisations, proven == the reference for the full battery (off far above
2048, `T` up to 100000, `d` up to 10^7, zero, negative-via-G, multi-channel) by
`test_membership_delay_no_cap_extreme_battery` / `test_membership_bus_delay_no_cap_extreme_battery`.
`test_rtl_delay_player_mirror_matches_physical_delay_any_length` walks the EXACT registers
of the RTL membership player (per-channel interval LUTRAM eval at the shifted phase +
`del_frame_idx` startup gate, NO buffer) and equals all three. `test_image_delay_ctrl_packing_matches_rtl_unpack`
round-trips the host->image->RTL contract (off/skip/iv_counts + intervals). The contract
test asserts the RTL has NO `ring` / `DELAY_DEPTH` / lane strings.

## 10. Building The Manuals

```powershell
python -c "from Zou_lab_control.neutral_atom.notes import build_main_manual, build_fpga_manual, build_device_manual; build_main_manual(); build_fpga_manual(); build_device_manual()"
python -c "from Zou_lab_control.frontend.notes import build_frontend_manual; build_frontend_manual()"
```

Each builder generates example figures into `assets/`, fills the `.texbody`
template, writes the `.tex` wrapper, and runs `render_tex_pdf` (XeLaTeX, 2-pass,
in a temp dir). XeLaTeX must be on PATH (or pass `xelatex=`). A failed build
leaves only a `.build.log` next to the target PDF.

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

### RTL bring-up checklist (NOT edited — verify on hardware; speculative RTL edits are
forbidden because a wrong edit costs a failed synth/route cycle)
The adversarial RTL hunt found three items to confirm on the bench rather than patch:
- **U4 — delayed-output tail at `done`.** `bnd_delay_advance` is set only while `running`
  (`zlc_edge_streamer.v` ~694); after a FINITE sequence finishes, the delay rings stop, so a
  channel with delay `d` does not flush its last `d` ticks. Harmless when the final frame
  returns to 0 (the usual case) and irrelevant for `repeat_forever` (never `done`); the
  Python mirror agrees, so it is not a host/RTL *surprise*. If a finite run must emit a
  non-zero delayed tail, pad the program's final tick by the max delay (HOST change) rather
  than touch the RTL.
- **B1/B2 — `da_clk0..3` = `out_final[28/39/50/61]`.** These DAC strobe pins are driven by the
  engine bits for those channels (or by `clk` if the channel is clk-enabled). Confirmed the
  board's DAC scan works as-is; if a future board needs them tied to the FPGA clk, mark those
  channels as clk channels. Don't let a normal program toggle 28/39/50/61.
- **B3/B4/U7 — parameterization traps** (`COEFF_BITS==64`, flags-word width, `scan_addr_of`):
  correct at the shipped `NUM_SLOTS=4` / `BUS_WIDTH=10` / `BANK_SIZE=2048`; only a concern if
  those change. Documented at the call sites.

### DRY done + remaining backlog
Done (safe, test-guarded): single `streamer_config.json` source; one `estimate_resources`
accounting model; one `sN` slot-ref parser (`pulse_table.is_slot_ref`/`slot_ref_index`, reused
by sequencer + GUI); `UNIT_TO_NS` imports the timing `UNITS_TO_NS`; `_channel_delays_list`
helper; deleted dead `BUS_SEGMENT_MODES`.
Backlog (deferred — larger/riskier, none blocking): unify `effective_tick` vs
`_apply_affine_ticks` (one narrowing-aware helper); one `validate_delay_depth(tick_ns=)` (drop
the hardcoded 20 ns in the µs hint); move `PulseTableState.bus_value`-style packing out of the
GUI; route NamesPanel/ChannelPanel rows through `FluentLabeledField` + a `set_field_locked`
helper; split the pure `_pulse_table_*`/`_affine_*` compiler block out of the 2.2k-line
`sequencer.py`. The cross-layer delay-depth/`coeff_frac_bits` constants remain test-guarded
mirrors (cross-package import direction); kept as-is.
