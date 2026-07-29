# Maintainer checkpoint

This file is a current hand-off note, not an architecture authority. Normative design and acceptance gates live only in `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`; execution/recovery rules live only in `AGENTS.md`.

## Repository state

- Branch: `codex/system-architecture-migration`
- M0 normative design baseline: `aaaa063ca5a8535c4f316df95e62574edce46cd3`; canonical observation-address clarification: `0af9354`.
- Last code checkpoint before the normative freeze: `09fad53 Fix live Figure transactions and restore plot fits`
- Completed architecture cuts in this checkpoint: **M1 — minimal PointTable/GridTopology/source-binding replacement**, **M2 — exact Signal transaction**, and **M3 — Figure/View/Fit convergence**.
- Active next cut: **M4 — Run, pulse and storage boundary**.
- Active worktree state after the M3 checkpoint: M1–M3 each have one mutable truth owner and no production compatibility model. The only expected worktree exception is the untracked user file below; resume from M4 and do not replay M1–M3.
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
Test-source migration does not belong to M7: every cut must rewrite its directly affected tests to the current public/physical contract or delete them with the retired behavior. M3 includes the one-time catch-up for M1/M2 test debt before it may close.

## M1 closure evidence

- `zlc_data/schema.py` owns the sole `PointColumn`/`PointTable`/`GridTopology`/`ResolvedPointRows` model; `axis.py` owns canonical scalar/source normalization and `codec.py` owns its wire form. `SourceViewBinding` is the only frontend binding. No point manager, registry, second resolver result or new module was added.
- Production references to Dataset `PointLayout`, `point_axes`, `point_layout`, `cell_layout`, `TransformedSchema`, `FitCoordinateSource`, `RowComponentValidity`, `AxisViewBinding`, `RepeatViewMode`, `display_selections`, `AxisViewRole.SLIDER`, `fit_axis_ids` and FitGrid are zero. The six obsolete FitGrid owner files are deleted.
- A real non-grid PulseScan with correlated authored rows A/B/A preserves ordinals `0,1,2`, exact rows and `grid_topology=None` through Dataset, artifact, Figure and Fit without Cartesian expansion.
- A real MOT Field 2×2×2 producer uses the same point table plus explicit topology; its scalar physical shape is `(1,8,1)`, topology is `(2,2,2)`, and the frontend renders non-empty faceted data without inferring topology.
- Saved Curve and Histogram Fit results reopen through the canonical DataFigure/Figure renderer for overview, focus and export. Focus remains display-only and never rewrites the committed Fit authority.
- Fixed-scope production Python changed from 174,687 to 172,061 physical lines: 421 to 415 modules, 1,025 to 1,016 classes, 602 to 598 dataclasses and 35 to 33 enums. The complete cut is `+8,464/-11,090`, net `-2,626`, with zero new production modules and six deletions.
- Production compileall and `git diff --check` passed at the M1 code checkpoint. Forty-nine tests still named the replaced model; the later test-source ruling makes that deferral invalid. They must be rewritten to current contracts or deleted during the M3 catch-up and the owning downstream cuts, never used to restore M1 compatibility.
- Next action is M2 only. Recover its owner/deletion/public-concept contract from the Goal, audit M2/C rows and System Architecture §5/§8 M2 before touching product code.

## M2 frozen preflight

- Fixed-scope baseline before M2 product edits is 415 production Python modules and 172,061 physical lines. The worktree has no product diff; `pulses/scan_test.json` remains the sole untracked user file and is outside the cut.
- The only mutable Signal generation/frontier authority will be `SignalDataPlane`. The existing acquisition `EventRef`/reservation owner, `MonitorDataset` materialization owner, Camera association adapter, PulseScan exact collector and frozen RTL remain unchanged.
- The sole allowed new public concept is one immutable `SignalPublication`, earned by atomic siblings, exact parent references and monotonic transaction sequence and consumed by Processor routing, Figure-derived publication and Workbench operations. M2 adds zero public presentation types and no public Prepared/Presented/Route/Manager/borrow/lease framework.
- Delete, do not wrap: `_CausalEdge`/`_CausalComponent` retrospective reconstruction, source-component capture/plumbing, per-revision `source_transform` association guesses, `ConsolePresentationIndex`, Window presentation reconciliation/topology repair and `_card_output_presentations`, dynamic mixed `FigureOutputRequest`/`FigureOutputFront`/`FigureOutputSession`, raster-lane output sessions, and TaskConsole recursive association heuristics.
- Keep and reassign: `SignalValue` and `SignalFront`; pure `FigureAreaCommit`/`FigureCrossCommit` and materializers; generation-static `FigureOutputPresentation`; the existing Fit solver lane until M3; frontend SiteMap rendering rebuilt from exact signal/artifact facts; exact hardware association and reservation contracts.
- Area and Cross are separate continuous routes driven from exact source publications independently of paint cadence. Fit parameters are a separate event-result publication and never withdraw or block selector routes. Workbench constructs and presents connected panel fronts from one exact `SignalFront`; a failed attachment/render keeps the prior complete group and is nonfatal.
- Formal association is a frozen neutral route fact backed directly by the existing `SignalEventAssociationSource`; TaskConsole never infers it from panel names or display transforms. No EmissionSlot/PointEmissionMap framework is introduced because the current FormalPulseScan product has no fixed-K publication consumer.
- Required M2 evidence is: raw→ROI dual panels; raw→ROI→ROI→Fit under at least three interleaved retained revisions; an independent producer advancing concurrently; atomic Camera frame siblings; and FormalPulseScan over Camera/Area/Occupancy with Fit/Histogram rejected before FIRE. Retirement with active/pending work must not revive a generation.

## M2 closure evidence

- `SignalDataPlane` is the sole mutable generation/publication/frontier owner. Its one new public value, immutable `SignalPublication`, freezes the sibling bundle, owner generation, monotonic sequence and exact parent publications; only the plane constructs `SignalPublication` and `SignalFront`.
- Retrospective `_Causal*` reconstruction, source-component plumbing, per-revision presentation data, `ConsolePresentationIndex`, mixed `FigureOutputRequest`/`FigureOutputFront`/`FigureOutputSession`, raster output sessions and Window topology repair are deleted. Area and Cross now have independent continuous routes; Fit parameters use an exact-parent event route; SiteMap presentation is projected from exact signal/artifact facts.
- Product witnesses passed for raw→ROI, three-revision raw→ROI→ROI→Fit with an independently advancing producer, atomic Camera frame siblings, exact Fit completion tokens, complete render groups and restarted-owner generation isolation. Retirement and terminal counterexamples proved that stale, fake, duplicate or late publications cannot revive a generation.
- FormalPulseScan preflight accepted Camera, Area and Occupancy event sources and rejected Fit and Histogram before cursor/FIRE. Formal association is validated against the frozen route and value schema; TaskConsole performs no name-, panel- or display-based reconstruction.
- Fixed-scope production Python changed from 415 modules / 172,061 lines / 1,016 classes / 598 dataclasses / 33 enums to 414 / 171,987 / 1,004 / 589 / 33. Production diff is `+3,062/-3,136`, net `-74`, with zero new modules and one deleted owner module.
- Production compileall, changed-module import smoke, `git diff --check`, forbidden-symbol search and constructor-owner search passed at the M2 code checkpoint. Its directly affected stale tests are part of the same M3 catch-up; `pulses/scan_test.json` remains untouched and untracked.
- Next action is M3 only. Recover its Fit-surface, regular-raster, canonical Figure/Form and deletion contracts before changing product code.

## M3 frozen preflight

- Fixed-scope baseline is 414 production Python modules / 171,987 physical lines / 1,004 classes / 589 dataclasses / 33 enums. M3 must add no production module and should be materially net-negative; a net increase or another Manager/Session/Lane owner stops the cut for compression review.
- Existing `PlotPanelContract/PlotPanelSession`, `PanelFrame/BoardFrame`, `EvaluatedProjectionIdentity`, `ViewSpec/SourceViewBinding`, serial Matplotlib render owner, `FitResultBatch` and the six data-owned Fit responsibility files are retained and extended in place. B14–B16, B21 and B25 are already closed and will not be replayed.
- The only allowed new public concept is frontend-owned `FigureIntent`, consumed by TaskConsole, Calibration, DataFigure/FigureViewer and API projection. Typed `PlotKind` replaces the current string vocabulary in place and may not coexist with a second registry. No public Front/Session/Lane/Manager/problem type is allowed.
- Delete rather than wrap: repeat-mode mappings/UI and stale alternatives; parallel Figure/Fit translators and DataFigure front/runtime; `FigureFitLane`; card-global Fit source/result/pin state; DataFigure's second Fit executor/draft state; Workbench display-limit/form authority; Histogram scale enum; numeric text sentinels; Calibration's duplicate live/report presentation defaults; leaf-local font overrides/workarounds; leaf UI reverse Workbench imports; unstable root-barrel implementation exports. Main's canonical Helvetica Light asset remains packaged and is registered only by the frontend style owner.
- `zlc_data.fit_problem/fit_solver` remains the sole representation/solve owner. A private regular 1D/2D path retains readonly original-dtype views, axis vectors, ROI slices and compact validity; irregular inputs retain the packed fallback. Workbench composition owns one private compute submit/cancel/completion seam; frontend owns only authoring, projection and vector overlays.
- Product evidence must prove: ambiguous defaults need input; repeat and facet remain orthogonal with exact FixedIndex; one exact Figure front/geometry drives selector and Fit; Monitor Fit never pins base; Edit Fit is isolated; Fit completion performs no base compose; 2304² uint8 packing loses the current full-grid/full-float64 cost; Calibration live/report and all Figure hosts share canonical pixels/forms; no reverse leaf→Workbench edge or stale public owner remains.

## M3 closure evidence

- `PlotKind`, `FigureIntent`, display forms, view evaluation and Fit authoring now have one frontend owner. TaskConsole, DataFigure/FigureViewer, API projection and Calibration consume that owner instead of maintaining parallel vocabularies, forms or Fit runtimes. Calibration core remains headless; its optional UI adapter delegates report rendering to the same frontend report owner.
- Monitor Fit consumes successive live fronts without pinning the base raster; Edit Fit owns an isolated snapshot. Fit parameters publish through their exact source publication, while overlays remain presentation state. The existing data Fit problem/solver owns regular-raster handling with readonly original-dtype data and axis vectors; no public Fit lane/session/problem framework was added.
- Area, Cross and Fit use the same `SinglePanelHost` interaction/publication path and exact frontend geometry. Product witnesses cover Camera→Area, retained ROI→ROI→Fit revisions with an independent producer, visible signal-picker reconciliation, live orange Fit overlay plus parameter publication and Clear, PulseScan schema gating, MOT current presentation, and curve/pulse/image/histogram interaction through real Qt input.
- M1/M2 test debt was closed in the same cut: tests naming retired Point/Figure/private-board contracts were migrated to current public product surfaces or deleted with the retired behavior. No compatibility production model was restored. Test sources changed by `+5,193/-19,809`, net `-14,616`; cross-test imports and references to retired M1/M2/M3 symbols are zero.
- Fixed-scope production Python changed from 414 modules / 171,987 physical lines / 1,004 classes / 589 dataclasses / 33 enums to 408 / 171,900 / 993 / 583 / 33. Production diff is `+7,909/-7,996`, net `-87`, with zero new production modules and six deleted owner modules: `panel_policy`, `axis_navigator`, `figure_fit_lane`, `fit_draft`, `form_projection` and `frozen_raster`.
- Narrow current-contract evidence passed: 197 data/runtime/operator tests, 88 frontend/readout tests, 55 product/plane tests, 46 architecture/calibration tests, 35 unified interaction tests, 9 curve tests, 7 pulse tests, 6 PulseScan consumer tests, 2 exact modified Experiment facade tests, and the MOT current witness. Production compileall, `git diff --check`, forbidden/deleted-symbol scans, debug-code scan and the frozen-hardware diff gate pass at closure.
- The full Experiment-facade run still exposes two valid lifecycle failures (`close-retry` validation ordering and `close-race-capture` lookup timing): 11 tests pass and 2 fail. Those two case bodies and their close/admission production paths were not changed by M3; keep the tests and repair the single Experiment lifecycle/admission owner in M4/M5. Do not delete them and do not add M3 compatibility.
- Next action is M4 only. Its production edits must migrate or delete directly affected tests in the same cut; only broad cross-cut execution remains deferred to M7.
