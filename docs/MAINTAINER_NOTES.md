# Maintainer checkpoint

This file is a current hand-off note, not an architecture authority. Normative design and acceptance gates live only in `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`; execution/recovery rules live only in `AGENTS.md`.

## Repository state

- Branch: `codex/system-architecture-migration`
- Last code checkpoint before the normative freeze: `09fad53 Fix live Figure transactions and restore plot fits`
- Completed architecture cut in this checkpoint: **M0 refresh — normative architecture convergence after the M1/M2 adversarial rollback audit**
- Active next cut: **M1 — minimal PointTable/GridTopology/source-binding replacement**
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

The refreshed M0 commit closes all of the following together:

- System Architecture is the sole normative design and contains the converged Point, View/Fit, Signal transaction, Run admission, hardware, storage, form and product-flow contracts.
- The adversarial audit corrections are merged into System Architecture §4, §5, §8, §9 and §10; this checkpoint does not restate those contracts.
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
