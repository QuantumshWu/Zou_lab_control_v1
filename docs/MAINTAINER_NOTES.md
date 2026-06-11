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
re-sweeps a streamed scan via a host background refill thread that feeds chunks
CONTINUOUSLY and CYCLICALLY (chunk `(mono%K)` into bank `mono%2`, one-ahead) -- the
sweep wrap is just another chunk boundary, so the re-sweep is SEAMLESS for any N
(`scan_bank_base` toggles by `K&1` so chunk 0 lands in the alternating bank).

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

## 8. Per-Channel OUTPUT Delay (TTL event scheduler + DAC rings)

A channel delay is a **physical OUTPUT delay**, not baked into the edge ticks:
`output_delayed[t] = output_undelayed[t - d]`, **zero before fire**, never disturbing
another channel; negative delays fold via the global shift `G = max(0, -min(delays))`.
The edge table is emitted UNDELAYED; the delays ride `channel_delays` (TTL, one 32-bit
word per channel in the AXI DELAY register region R_DELAY_BASE) and `bus_delays`
(DAC, dense CTRL words).

**TTL = EVENT SCHEDULER** (`zlc_edge_streamer.v`): a TTL waveform is toggle-sparse, so
the engine queues TOGGLES instead of buffering one bit per tick.  When the undelayed bit
flips at tick `t`, it pushes `{t + d - 1, level}` into that channel's `EVT_DEPTH`-deep
(default 256) 49-bit LUTRAM FIFO; a free-running 48-bit `g_time` pops it by equality into
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
would let it infer), so Vivado falls back to flip-flops -- at depth 256 that is 18*256*49 = 226k
FF + 256:1 read muxes, which the 35T cannot place (a real build failed exactly this way).
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
nothing is silently dropped.  **DAC buses are UNIFIED into the same event scheduler**: each DA
bit is its own 1-bit event FIFO (`g_busdly`, fed from `bus_value_active`, reset to that bit's
`BUS_SAFE_VALUE` level so an untouched bus idles at 0 V), and the bus's `BUS_WIDTH` bits share one
per-bus 32-bit delay (`del_bus_ticks`) -- so the DAC delay range MATCHES TTL and a negative TTL
delay's global shift G can reach the buses with no mismatch.  The
per-bit FIFO is shallower (`BUS_EVT_DEPTH`, default 64; there are bus_count*bus_width = 40 of them)
to fit LUT -- per the conservative model per-bit @256 = 102.9% (over the device), @64 = ~81%.
Capacity = value-change events in flight per bit <= `BUS_EVT_DEPTH`; the only stressor is a long
DELAYED ramp (one event/step).  `bus_evt_fifo_depth` is reconfigurable from streamer_config.json.

**Reconfigurable depth (single source).**  `evt_fifo_depth` lives only in
`streamer_config.json`.  The host (validator/estimate) reads it; the BUILD reads it too --
`image.emit_geom_tcl` turns the config into `$zlc_top_generics` (EVT_FIFO_DEPTH,
EDGE_ADDR_WIDTH, BANK_SIZE), `build_and_program.bat --emit-geom-tcl` generates `geom.tcl`,
and `create_project.tcl` sources it + `set_property generic ... [current_fileset]` so editing
the JSON changes the SYNTHESIZED bitstream (e.g. 256->128 if a build is LUT-tight).  A
contract test asserts every generic names a real top parameter (else Vivado ignores it).

Proven: `delay_line_reference` (out[t]=in[t-d]) is the unchanged ground truth;
`engine_model.rtl_delay_line_mirror` mirrors the scheduler cycle-exactly; the REAL RTL is
verified in xsim -- `tb_delay_sched.v` (delays {0,1,2,7,1000}, 1-tick toggles, repeat seams:
11,996 cycles, 0 mismatches, DELAY-SCHED-OK), `tb_delay_compact.v` (non-identity slot->channel
map at depth 256, COMPACT-MAP-OK), `tb_evt_depth.v` (FIFO-depth boundary pinned at 16,
EVT-DEPTH-OK).  Estimate at depth 256: 67.7% LUT / 21.6% FF on the 35T (the per-slot LUTRAM
is ~0.7-1.0 LUT per 64x1 cell).

## 9. Pulse API, sync-to-device and GUI state semantics

`PulseController` (sequencer.py) is the notebook-facing API and shares the exact
sequencer path with the GUI: `on_pulse()/off_pulse()`, `set_channel_delay()/
get_channel_delay()` (delay calibration), `load_pulse()/save_pulse()`,
`set_scan_table()`, `synced_state()`.  `SequencerService.prepare` records the SOURCE
payload (`last_payload_json`) of every successful prepare -- from the GUI or any raw-API
caller -- and publishes it in `snapshot()`; `RemoteSequencer` flattens it across RPyC.
The GUI's **Sync** button and `pulse.synced_state()` both read this single source of
truth.  GUI state semantics (confocal style -- stars + status dot, never button base
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
  to the FPGA clk.** New safety net: `_warn_idle_dac_clock_pins` (sequencer.py) warns at
  compile time when DAC buses are driven while a `da_clkN`-labeled channel is neither
  clk-enabled nor toggled (a frozen DAC would otherwise be silent). A warning, not an error.
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
  feeds the bus delay line), and the preview `pulse_table._analog_bus_value_at_tick`
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
field is likewise 32-bit (~42.9 s), and with repeat_forever a longer delay is reduced
modulo the sweep period anyway (§8).  Separately, the scheduler's free-running `g_time`
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
