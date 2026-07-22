"""Final-only repository authority and crash-reconciliation contracts."""

from __future__ import annotations

import pytest

from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.commit import (
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishVisibilityUnknown,
    PublishedManifest,
    RepositoryCommitCoordinator,
)
from zlc_neutral_atom.runtime.resources import ResourceArbiter
from zlc_neutral_atom.runtime.run import RunController, RunFailed, RunPlan
from zlc_storage import RepositoryRootLease


_DIGEST = "a" * 64


class _CommitHarness:
    def __init__(self, tmp_path, *, publish=None, recover=None) -> None:
        self.root = tmp_path / "repository"
        self.root_lease = RepositoryRootLease(self.root)
        self.journal = PersistentCommitJournal(
            self.root / "final-commit.zlcj",
            "capture-repository",
        )
        self.target = CommitTarget(
            "capture-repository",
            "capture",
            "zlc_neutral_atom.CaptureArtifact",
            f"capture/{_DIGEST}",
            _DIGEST,
        )
        self.publish = (
            (lambda: PublishedManifest(self.target.target_ref, _DIGEST, "published"))
            if publish is None
            else publish
        )
        self.recover = (lambda _intent: None) if recover is None else recover
        self.coordinator = RepositoryCommitCoordinator(
            self.journal,
            self.recover,
            root_lease=self.root_lease,
        )
        self.resources = ResourceArbiter()
        self.controller = RunController(self.resources)

    def operation(self, context, *, run_id=None):
        selected_run = context.authorize_commit_preparation() if run_id is None else run_id
        return self.coordinator.prepare(
            f"capture-final-{selected_run}-{_DIGEST}",
            selected_run,
            self.target,
            self.publish,
        )

    def plan(self, finalize) -> RunPlan:
        return RunPlan(
            name="final commit authority fixture",
            resource_claims=(),
            bound_devices=(),
            preflight=lambda _context: "prepared",
            execute=lambda _context, prepared: prepared,
            cleanup=lambda _context, _prepared, _primary: CleanupReport.complete(),
            finalize=finalize,
            requires_final_commit=True,
        )

    def close(self) -> None:
        assert self.controller.shutdown(2.0)
        self.resources.shutdown()
        self.coordinator.close()


def test_final_commit_is_unforgeable_and_single_use(tmp_path) -> None:
    harness = _CommitHarness(tmp_path)

    def finalize(context, _result):
        operation = harness.operation(context)
        assert isinstance(operation, FinalCommit)
        first = context.commit_final(operation)
        with pytest.raises(RuntimeError, match="already consumed"):
            context.commit_final(operation)
        return first

    try:
        handle = harness.controller.start(harness.plan(finalize))
        assert handle.result(3.0) == "published"
        assert handle.snapshot().final_committed
        assert harness.journal.pending() == ()
        committed = harness.journal.committed_for(harness.target)
        assert len(committed) == 1
        assert committed[0].commit_id.startswith("capture-final-")
    finally:
        harness.close()


def test_finalize_failure_abandons_unconsumed_authority(tmp_path) -> None:
    harness = _CommitHarness(tmp_path)

    def finalize(context, _result):
        harness.operation(context)
        raise RuntimeError("synthetic finalize failure")

    try:
        with pytest.raises(RunFailed, match="synthetic finalize failure"):
            harness.controller.start(harness.plan(finalize)).result(3.0)
        assert harness.coordinator._authorities == {}
        assert harness.journal.pending() == ()
    finally:
        harness.close()


def test_commit_subject_mismatch_fails_without_publication(tmp_path) -> None:
    published = []
    harness = _CommitHarness(
        tmp_path,
        publish=lambda: published.append(True),
    )

    def finalize(context, _result):
        operation = harness.operation(context, run_id="another-run")
        return context.commit_final(operation)

    try:
        with pytest.raises(RunFailed, match="another Run"):
            harness.controller.start(harness.plan(finalize)).result(3.0)
        assert published == []
        assert harness.journal.pending() == ()
    finally:
        harness.close()


def test_unknown_publish_visibility_is_reconciled_by_the_repository_owner(
    tmp_path,
) -> None:
    visible = {"value": False}
    harness = None

    def publish():
        visible["value"] = True
        raise PublishVisibilityUnknown("acknowledgement lost")

    def recover(_intent):
        if not visible["value"]:
            return None
        assert harness is not None
        return PublishedManifest(
            harness.target.target_ref,
            _DIGEST,
            "recovered",
        )

    harness = _CommitHarness(tmp_path, publish=publish, recover=recover)

    def finalize(context, _result):
        return context.commit_final(harness.operation(context))

    try:
        handle = harness.controller.start(harness.plan(finalize))
        assert handle.result(3.0) == "recovered"
        snapshot = handle.snapshot()
        assert snapshot.final_committed
        assert "acknowledgement lost" in snapshot.commit_recovery_warning
        assert harness.journal.pending() == ()
    finally:
        harness.close()


def test_absent_manifest_recovery_aborts_instead_of_forging_success(tmp_path) -> None:
    def publish():
        raise PublishVisibilityUnknown("visibility unknown")

    harness = _CommitHarness(
        tmp_path,
        publish=publish,
        recover=lambda _intent: None,
    )

    def finalize(context, _result):
        return context.commit_final(harness.operation(context))

    try:
        with pytest.raises(RunFailed, match="visibility unknown"):
            harness.controller.start(harness.plan(finalize)).result(3.0)
        assert harness.journal.pending() == ()
        assert harness.journal.committed_for(harness.target) == ()
    finally:
        harness.close()


def test_coordinator_and_journal_have_no_checkpoint_or_kind_surface(tmp_path) -> None:
    harness = _CommitHarness(tmp_path)
    try:
        assert not hasattr(harness.coordinator, "prepare_checkpoint")
        assert not hasattr(harness.journal, "checkpoint")
        assert not hasattr(harness.target, "kind")
        assert not hasattr(harness.journal, "startup_reconciliations")
    finally:
        harness.close()


def test_repository_root_cannot_close_while_authority_is_live(tmp_path) -> None:
    harness = _CommitHarness(tmp_path)
    operation = harness.coordinator.prepare(
        "capture-final-detached-" + _DIGEST,
        "detached",
        harness.target,
        harness.publish,
    )
    try:
        with pytest.raises(RuntimeError, match="outstanding operations"):
            harness.coordinator.close()
        operation.abandon()
        harness.coordinator.close()
    finally:
        harness.controller.shutdown(2.0)
        harness.resources.shutdown()
