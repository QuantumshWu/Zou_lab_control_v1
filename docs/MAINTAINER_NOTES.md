# Maintainer checkpoint

This file is a current hand-off note, not an architecture authority. Normative design and acceptance gates live only in `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`; execution/recovery rules live only in `AGENTS.md`.

## Repository state

- Branch: `codex/system-architecture-migration`
- M0 normative design baseline: `aaaa063ca5a8535c4f316df95e62574edce46cd3`; canonical observation-address clarification: `0af9354`.
- Last code checkpoint before the normative freeze: `09fad53 Fix live Figure transactions and restore plot fits`
- Completed architecture cut in this checkpoint: **M1 — minimal PointTable/GridTopology/source-binding replacement**
- Active next cut: **M2 — Signal transaction**
- Active worktree state: M1 has one point truth and no production compatibility model. After this checkpoint the only expected worktree exception is the untracked user file below; resume from M2 and do not replay M1.
- Expected worktree exception: untracked user file `pulses/scan_test.json`; never read, modify, stage or commit it.
- External temporary audit ledger: `../ARCHITECTURE_AUDIT_CURRENT.md`; it is not a product/design authority and is deleted only after M0–M7 evidence is fully merged and all implementation cuts finish.

## Resume protocol

1. Read the complete active `/goal`.
2. Check `git status --short --branch`, `git log -5 --oneline`, current tree and this checkpoint.
3. Read `../ARCHITECTURE_AUDIT_CURRENT.md` §1.1–§1.6, the active M cut, its matrix rows and §9; read only the System Architecture sections for the active cut.
4. Derive completion from Git/current code; do not replay closed audits or fixed historical slice numbers.
5. Recheck active-diff owners, deletion contract, public concepts and fixed-scope LOC/module deltas before editing.
6. Keep `pulses/scan_test.json` untouched.

## M0 closure evidence

The refreshed M0 authority closes all of the following together:

- System Architecture is the sole normative design and contains the converged Point, View/Fit, Signal transaction, Run admission, hardware, storage, form and product-flow contracts.
- The adversarial audit corrections are merged into System Architecture §4, §5, §8, §9 and §10; this checkpoint does not restate those contracts.
- System Architecture §3.3–§3.5 no longer turns point semantics into a public type family: PointColumn/GridTopology reuse AxisId, one discriminated AxisSourceRef replaces five ref classes, `point_ordinals` is the row-filter payload, grouping is a plain source tuple, and the sole resolved value is data-owned ResolvedPointRows. A point ordinal itself is the observation address, so the result has no mirrored address field. Frontend validation delegates privately from the existing Figure contract and does not own a second resolved DTO/module.
- The stale per-revision Presented*, horizontal M1 and public session/lane/admission type implications are absent from the target design except where named as deletion evidence.
- `AGENTS.md` contains execution protocol only.
- this file contains checkpoint/operations only.
- the duplicate design charter and every reference to it are deleted.
- no contradictory old architecture remains in tracked docs.
- Markdown/link checks, dead-reference search and `git diff --check` pass.

## Operational entry points

- Desktop composition: DeviceManager creates the shared Experiment and opens TaskConsole/PulseGUI.
- Real-hardware qualification: `docs/REAL_HARDWARE_BRINGUP_zh.md`
- FPGA server operations: `fpga/README.md`
- Verification entry: `tests/README.md`

Use the narrowest current test or product flow for the active boundary. Broad verification belongs to M7.

## M1 closure evidence

- `zlc_data/schema.py` owns the sole `PointColumn`/`PointTable`/`GridTopology`/`ResolvedPointRows` model; `axis.py` owns canonical scalar/source normalization and `codec.py` owns its wire form. `SourceViewBinding` is the only frontend binding. No point manager, registry, second resolver result or new module was added.
- Production references to Dataset `PointLayout`, `point_axes`, `point_layout`, `cell_layout`, `TransformedSchema`, `FitCoordinateSource`, `RowComponentValidity`, `AxisViewBinding`, `RepeatViewMode`, `display_selections`, `AxisViewRole.SLIDER`, `fit_axis_ids` and FitGrid are zero. The six obsolete FitGrid owner files are deleted.
- A real non-grid PulseScan with correlated authored rows A/B/A preserves ordinals `0,1,2`, exact rows and `grid_topology=None` through Dataset, artifact, Figure and Fit without Cartesian expansion.
- A real MOT Field 2×2×2 producer uses the same point table plus explicit topology; its scalar physical shape is `(1,8,1)`, topology is `(2,2,2)`, and the frontend renders non-empty faceted data without inferring topology.
- Saved Curve and Histogram Fit results reopen through the canonical DataFigure/Figure renderer for overview, focus and export. Focus remains display-only and never rewrites the committed Fit authority.
- Fixed-scope production Python changed from 174,687 to 172,061 physical lines: 421 to 415 modules, 1,025 to 1,016 classes, 602 to 598 dataclasses and 35 to 33 enums. The complete cut is `+8,464/-11,090`, net `-2,626`, with zero new production modules and six deletions.
- Production compileall and `git diff --check` pass. No RTL/Tcl/XDC, tests, tutorials, fixtures or temporary evidence changed. Forty-nine historical tests still name the replaced model; per user direction and System Architecture M7 they must be rewritten or deleted there, never used to restore compatibility during M1.
- Next action is M2 only. Recover its owner/deletion/public-concept contract from the Goal, audit M2/C rows and System Architecture §5/§8 M2 before touching product code.
