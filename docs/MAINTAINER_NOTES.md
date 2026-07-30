# Maintainer checkpoint

This file is only the current hand-off checkpoint. Normative architecture lives in
`docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`; execution and recovery rules live in
`AGENTS.md`. Do not preserve superseded slice narratives here.

## Repository state

- Branch: `codex/system-architecture-migration`.
- Reopened M1–M6 closure HEAD: `a74b12bed9c1153a9c66956e6752a7bddd8d32c4`.
- Current checkpoint: the commit containing this file closes the M7 software
  evidence and cleanup cut. It is GO for the connected-hardware E0 run; it does
  not claim that the real qCMOS/FPGA path has already been exercised.
- External audit ledger: `../ARCHITECTURE_AUDIT_CURRENT.md`. It remains a temporary
  implementation checklist until the real-hardware E0 gate is closed, then must
  be deleted rather than retained as a second architecture document.
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
  `zlc_pulse`, `zlc_storage`, `zlc_workbench`): 381 modules, 161,566 physical lines,
  918 top-level classes, 552 dataclass declarations, and 32 enum declarations.
- Root launchers: 4 modules / 396 lines.
- Formal FPGA Python outside verification/tests: 8 modules / 2,825 lines.
- Complete package diff from `41f39da`, with renames expanded for a stable deletion
  count: `+5,846/-15,291`, net `-9,445`, added/deleted ratio `0.382`.
- The M7 production delta itself is `+43/-78`, net `-35`; its current-contract test
  delta is `+17/-124`, net `-107`. It adds no class, dataclass, enum, module, manager,
  registry, session/lane, compatibility model, memory budget, or hardware capability.

## M7 closure evidence

- Final broad current-contract suite: `1,139 passed, 1 skipped, 1 warning` in
  250.18 s. The skip is the platform where NumPy `longdouble` is `float64`; the
  warning is the documented Windows Jupyter/ZMQ Proactor-loop compatibility warning.
- The only real broad-suite production defect was an application-boundary regression:
  the public facade interpreted Capture storage fields. Capture now owns the direct
  artifact-to-`ArtifactDatasetSource` projection, loads metadata first, checks cancel,
  and materializes exactly once; the facade only delegates.
- Remaining broad-suite failures were stale contracts for removed payload digests,
  panel provenance, old `RunSnapshot` fields, and absolute Pulse paths. Their tests
  now exercise the current public/physical contracts. The zero-consumer
  `canonical_value_array` left by payload hashing is deleted rather than preserved for
  a test.
- The canonical style owner still uses bundled Helvetica Light for ordinary text and
  now supplies ordered Arial glyph fallback for the Pulse infinity symbol. Ordinary
  raster bytes were unchanged and the missing-glyph warning is gone; no leaf-local
  font workaround was added.
- Production compileall plus import-DAG, application-boundary, public-hardware,
  installation, device/Logic-node discovery, and first-party-import ratchets:
  `58 passed`.
- The full production retired-symbol/module scan is zero, including the point/repeat,
  Presented/presentation-sidecar, Fit lane/live-pin, TaskConsole lock, CAS/repository,
  ordinary digest, Calibration fork, old panel-provenance, and old `RunSnapshot`
  vocabularies. Forbidden reverse imports and concrete Logic-node imports from the
  application tree are also zero.
- Direct 2,304×2,304 `uint8` radial-Fit profile through the formal data entry uses the
  private regular representation: packing 3.20 ms with 79,937-byte `tracemalloc`
  peak; observations remain readonly, share source memory, and keep `uint8`; only two
  2,304-element float64 axis vectors and broadcast validity exist. A warmed complete
  solve took 446.25 ms, converged in 10 evaluations over all 5,308,416 pixels, and
  used about 4.4–4.6 MiB peak working set. No H×W coordinate/index grid or full-size
  float64 image is formed. The formal Qt product tests separately prove off-owner
  submission, continuing live base fronts, exact overlay/parameter publication,
  shared selector geometry, and no GUI-thread Agg during wheel/pan.
- `git diff --check`, exact user-file isolation, and both current and cumulative
  RTL/Tcl/XDC/bitstream diff checks pass with zero frozen-hardware changes.

## Remaining external acceptance

Only System Architecture §9.8 remains unclaimed: run the real E0 qualification and a
representative autonomous streamed scan against the connected qCMOS/FPGA installation,
using current working-point readback and per-run terminal/count/stamp/quiet-window
reconciliation. A virtual adapter or short synthetic test cannot substitute for that
physical evidence. If it passes, delete the external audit ledger and close the Goal;
if it exposes a real adapter/RTL defect, reopen only the proven physical owner and keep
the frozen-hardware rule until separate authorization exists.
