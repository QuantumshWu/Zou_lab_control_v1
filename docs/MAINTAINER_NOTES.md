# Maintainer checkpoint

This file is a current hand-off note, not an architecture authority. Normative design and acceptance gates live only in `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`; execution/recovery rules live only in `AGENTS.md`.

## Repository state

- Branch: `codex/system-architecture-migration`
- M0 normative design baseline: `aaaa063ca5a8535c4f316df95e62574edce46cd3`; canonical observation-address clarification: `0af9354`.
- Last code checkpoint before the normative freeze: `09fad53 Fix live Figure transactions and restore plot fits`
- Completed architecture cut in this checkpoint: **M0 second convergence — the public point/source/resolver vocabulary is now minimal as well as the transaction/Fit/milestone wording**
- Active next cut: **M1 — minimal PointTable/GridTopology/source-binding replacement**
- Active worktree state: M1 is deliberately uncommitted and not a checkpoint. The core/data and early Camera/runtime readers are mid-replacement; M0 is closed by the normative design commit above, so work resumes only inside the recorded M1 deletion contract.
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

## Active M1 cut contract

- Relevant authority: System Architecture §3, §8 M1 and §9; audit rows A01–A17, M1 and §9. A stale audit exception that mixed `PointRows SAMPLE` with `GridDimension BATCH/FACET` was removed; System Architecture's strict raw/topology separation governs.
- Existing owner to replace in place: `zlc_data/schema.py`, with scalar normalization in `axis.py` and the sole wire representation in `codec.py`. Do not add point/resolver modules.
- Fixed baseline before product edits: 52 production files contain the old Dataset point model, covering 32,791 physical lines and about 406 old `PointLayout`/`point_axes`/`point_layout` references. The cut must end net smaller; numeric deltas remain review triggers rather than automatic rollback.
- Allowed new public values: `PointColumn`, `PointTable`, `GridTopology`, one discriminated `AxisSourceRef`, and one `ResolvedPointRows`. `SourceViewBinding` replaces `AxisViewBinding`; it is not an additional model. Point selection/group requests remain normalized tuples/fields, not public class families. No `GridDimension`, `PointCoordinateId`, five source-ref subclasses, point manager/registry, frontend resolved DTO or new module is authorized.
- Required deletion closure: Dataset `PointLayout`; `DatasetSchema.point_axes`, `point_layout` and `_cell_layout/cell_layout`; point-layout codec/exports; leaf `ScanPointTable`; `TransformedSchema`; `FitCoordinateSource`; Cartesian point selection helpers; `AxisViewRole.SLIDER`; `RepeatViewMode`; point `display_selections`; AxisId-only View/Fit point bindings; `MotFieldProgram` axes/layout mirrors; and capture join's copied scan layout. Generic `AxisLayout` remains only for genuine FitResultBatch batch grids.
- Non-grid witness: real PulseScan over three correlated authored columns with a repeated coordinate tuple. It must preserve ordinals `0,1,2`, P=3, exact rows and `grid_topology=None` through Dataset, artifact, Figure and Fit without Cartesian expansion.
- Grid witness: real MOT Field 2×2×2 flow. The producer must freeze the same row-table model plus explicit topology; physical scalar shape is `(1,8,1)`, topology is `(2,2,2)`, and the formal Qt Grid figure must render non-empty data without frontend inference.
- Next action: implement only the core `axis.py`/`schema.py`/`codec.py`/`layout.py` replacement, then carry those two witnesses through existing runtime/transform/frontend owners before mechanically migrating the remaining readers. No Git checkpoint or product acceptance may expose both models.
