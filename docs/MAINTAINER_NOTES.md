# Maintainer checkpoint

This file is only the current hand-off checkpoint. Normative architecture lives in
`docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`; execution and recovery rules live in
`AGENTS.md`. Do not preserve superseded slice narratives here.

## Repository state

- Branch: `codex/system-architecture-migration`.
- Closure base HEAD: `41f39da29b644f637dd2c9f413eaae3a388300cc`.
- Current checkpoint: the commit containing this file closes the reopened M1–M6
  semantic boundary. The next dependency cut is M7 only.
- External audit ledger: `../ARCHITECTURE_AUDIT_CURRENT.md`. It remains a temporary
  implementation checklist until every M7 gate is closed, then must be deleted.
- Expected worktree exception: untracked user file `pulses/scan_test.json`. Never
  read, modify, move, delete, stage, or commit it.
- RTL, Tcl, XDC, bitstream, wire protocol, and deployed hardware assets are frozen
  and unchanged.

## Resume protocol

1. Read the complete active `/goal`, then `AGENTS.md` and this file.
2. Derive state from branch, HEAD/tree, status, recent log, and diff. Do not replay
   M1–M6 or re-explain already closed work.
3. Read audit §1.0–§1.6, M7, §9, and only the System Architecture sections needed
   for the failing M7 gate.
4. Recheck production owner/deletion/public-concept/LOC deltas, frozen hardware,
   and user-file isolation before editing.
5. A broad-test failure may repair only a still-valid physical/public contract.
   Rewrite or delete stale tests; never restore a retired owner or compatibility
   surface.

## Reopened M1–M6 closure

The implementation now follows the System Architecture's smaller terminal model:

- Runtime identity uses typed stream/generation/sequence/event/parent facts. Ordinary
  live values, Camera payloads, signal joins, Figure derivations, spans, and ordinary
  artifacts no longer compute payload SHA or mirror the same lineage through digest
  lattices.
- Experiment application is the only Run/admission lifecycle owner. SignalPlane owns
  generations, routes, publications, exact parents, dependency closure, and atomic
  withdrawal; its duplicate Run lifecycle API/state is gone.
- Project `_output` is the only experimental output root. Capture, PulseScan,
  Calibration, Occupancy, Fit, and Figure persistence use typed relative path refs,
  original-dtype direct files, and domain-record-last atomic publication. Generic CAS,
  repository leases, prepared commit/pending inspection, and hidden repository roots
  are gone. Hardware interprocess leases remain unchanged.
- Calibration owns physical readout facts and raw per-site samples. The generic
  frontend Distribution/Histogram owner alone performs bins, bimodal analysis,
  threshold validation, overlay, and style. Synthetic population axes, centered-zero
  thresholds, pooled pseudo-pages, and leaf-local plotting truth are gone. Explicit
  frame saving and saved-frame recalibration write below
  `_output/calibrations/<run-name>/`.
- Camera finite/live products preserve source dtype and fixed `R×P×data_shape`.
  Occupancy, MOT, duration fidelity, release-recapture, selectors, and Fit retain exact
  parent/event semantics without duplicate repository, presentation, or result owners.
- PulseScan remains autonomous streamed and consumes an arbitrary associated `y`
  signal. `exp.nodes.pulse_scan` now provides public scan-slot/API-slot authoring,
  bind, prepare, start, run, load, and materialize operations. TaskConsole stores a
  portable relative pulse path; the PulseScan package resolves it through the same
  application-injected pulse loader used by the public API before program binding.
- Tutorial PulseScan code imports no neutral internals. Direct artifact/archive modules
  are named for what they do rather than pretending to be generic repositories.

## Deleted owner closure

Ten production modules are deleted from the base tree:

- `zlc_data/bimodal_distribution.py`
- `zlc_neutral_atom/logic_nodes/mot_field/ui/__init__.py`
- `zlc_neutral_atom/logic_nodes/mot_field/ui/view_projection.py`
- `zlc_neutral_atom/logic_nodes/pulse_scan/repository.py`
- `zlc_neutral_atom/logic_nodes/readout/calibration/task_output.py`
- `zlc_neutral_atom/logic_nodes/readout/occupancy/repository.py`
- `zlc_neutral_atom/runtime/commit.py`
- `zlc_storage/content_store.py`
- `zlc_storage/repository_lease.py`
- `zlc_workbench/data_figure/archive_repository.py`

The three replacement files are direct domain operations, not new layers:

- `zlc_neutral_atom/logic_nodes/pulse_scan/artifact.py`
- `zlc_neutral_atom/logic_nodes/readout/occupancy/artifact.py`
- `zlc_workbench/data_figure/archive_io.py`

Production imports of every retired module and of CAS/commit/presentation mirror
symbols are zero.

## Fixed-scope complexity

- Package Python (`Zou_lab_control`, `zlc_data`, `zlc_frontend`, `zlc_neutral_atom`,
  `zlc_pulse`, `zlc_storage`, `zlc_workbench`): 381 modules, 161,601 physical lines,
  918 top-level classes, 552 dataclass declarations, and 32 enum declarations.
- Root launchers: 4 modules / 396 lines.
- Formal FPGA Python outside verification/tests: 8 modules / 2,825 lines.
- Complete production diff from the closure base, including the three newly named
  direct files: `+5,823/-15,233`, net `-9,410`, added/deleted ratio `0.382`.
- The cut adds no manager, registry, public session/lane, compatibility model, memory
  budget, or hardware capability. Its three new filenames replace three deleted
  repository-named owners.

## Current evidence

- Direct artifact/Calibration/Camera/runtime group: 88 passed, followed by all three
  corrected stale-contract cases passing.
- Core hardware-installation, endpoint, operator, runtime, data/Fit/transform,
  Calibration-result, and main-oracle group: 197 passed.
- Architecture/product/GUI group: 87 passed before four stale test contracts were
  identified; all four now pass individually. The DeviceManager→Camera→2-D/Area flow,
  finite Camera progress→FINAL flow, Grid Setting/Edit/Fit flow, and running
  Camera→PulseScan flow all use formal offscreen composition and real Qt controls.
- PulseScan public API plus executable tutorial: 18 passed.
- Final production compileall plus import-DAG/first-party-import/discovery ratchets:
  39 passed.
- `git diff --check`, retired-symbol/module scans, zero frozen-hardware diff, and user
  file isolation pass.

## Next action: M7 only

Run the final current-contract evidence matrix and broad suite last. Close failures by
their owning physical/public boundary, delete stale tests/docs/tutorial residue in the
same cut, and do not restore retired private APIs. Recompute the fixed-scope metrics,
all System Architecture §9 gates, import/dead-symbol scans, formal product flows,
frozen-hardware diff, and exact staged-file list. Only after every gate passes may the
temporary audit ledger be deleted and the Goal be marked complete.
