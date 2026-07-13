"""Crash-consistent final-artifact intent contracts."""

from __future__ import annotations

import threading

import pytest

import zlc_neutral_atom.runtime as runtime_api
import zlc_neutral_atom.runtime.commit as commit_api

from zlc_neutral_atom.runtime import (
    CheckpointCommit,
    CommitIntent,
    CommitKind,
    CommitRecovery,
    CommitSubject,
    CommitTarget,
    MemoryCommitJournal,
    PublishedManifest,
    FinalCommit,
)
from zlc_neutral_atom.runtime.commit import (
    PersistentCommitJournal,
    RepositoryCommitCoordinator,
    _consume_commit_authority,
    _journal_mutation_authority,
    reconcile_pending_commits,
)
from zlc_storage import RepositoryRootLease


REPOSITORY_ID = "test-repository"


def mutation(journal: PersistentCommitJournal):
    return _journal_mutation_authority(journal)


def intent(commit_id: str) -> CommitIntent:
    return CommitIntent(
        kind=CommitKind.FINAL,
        commit_id=commit_id,
        run_id="test-run",
        safety_bundle_id="test-safety-bundle",
        target=CommitTarget(
            repository_id=REPOSITORY_ID,
            artifact_kind="test-artifact",
            schema_version="1",
            target_ref=f"artifacts/{commit_id}",
            expected_manifest_digest="0" * 64,
        ),
        created_at=1.0,
    )


def test_memory_commit_resolution_is_mutually_exclusive():
    journal = MemoryCommitJournal(REPOSITORY_ID)
    journal.begin(intent("committed"))
    journal.mark_committed("committed")
    journal.mark_committed("committed")
    with pytest.raises(ValueError, match="cannot become aborted"):
        journal.mark_aborted("committed")

    journal.begin(intent("aborted"))
    journal.mark_aborted("aborted")
    journal.mark_aborted("aborted")
    with pytest.raises(ValueError, match="cannot become committed"):
        journal.mark_committed("aborted")
    assert journal.pending() == ()


def test_ephemeral_commit_journal_cannot_mint_production_authority():
    journal = MemoryCommitJournal(REPOSITORY_ID)
    with pytest.raises(ValueError, match="ephemeral"):
        RepositoryCommitCoordinator(
            journal,
            lambda _intent: CommitRecovery(committed=False),
        )


def test_durable_commit_writer_types_are_not_public_runtime_api(tmp_path):
    for name in (
        "PersistentCommitJournal",
        "RepositoryCommitCoordinator",
        "reconcile_pending_commits",
    ):
        assert not hasattr(runtime_api, name)
        assert name not in commit_api.__all__

    journal = PersistentCommitJournal(tmp_path / "commit-intents.zlcj", REPOSITORY_ID)
    for name in ("begin", "mark_committed", "mark_aborted"):
        assert not hasattr(journal, name)
    with pytest.raises(PermissionError, match="coordinator authority"):
        journal._begin(object(), intent("forged"))


def test_durable_coordinator_rejects_a_lease_for_another_root(tmp_path):
    journal_root = tmp_path / "journal-root"
    journal = PersistentCommitJournal(
        journal_root / "commit-intents.zlcj",
        REPOSITORY_ID,
    )
    wrong_lease = RepositoryRootLease(tmp_path / "other-root", owner="wrong-root")
    try:
        with pytest.raises(ValueError, match="different roots"):
            RepositoryCommitCoordinator(
                journal,
                lambda _intent: CommitRecovery(False),
                root_lease=wrong_lease,
            )
    finally:
        wrong_lease.close()


@pytest.mark.parametrize(
    ("record_id", "value", "message"),
    (
        (
            "committed:not-the-payload",
            {"kind": "COMMITTED", "commit_id": "record-binding"},
            "record_id differs",
        ),
        (
            "repository-copy",
            {"kind": "REPOSITORY", "repository_id": REPOSITORY_ID},
            "unique and first",
        ),
    ),
)
def test_persistent_journal_binds_record_ids_and_repository_marker(
    tmp_path,
    record_id,
    value,
    message,
):
    path = tmp_path / "commit-intents.zlcj"
    journal = PersistentCommitJournal(path, REPOSITORY_ID)
    if value["kind"] == "COMMITTED":
        mutation(journal).begin(intent("record-binding"))
    journal._journal.append(record_id, value)
    with pytest.raises(ValueError, match=message):
        PersistentCommitJournal(path, REPOSITORY_ID)


def test_persistent_commit_journal_recovers_pending_and_both_resolutions(tmp_path):
    path = tmp_path / "commit-intents.zlcj"
    journal = PersistentCommitJournal(path, REPOSITORY_ID)
    writer = mutation(journal)
    writer.begin(intent("pending"))
    writer.begin(intent("committed"))
    writer.mark_committed("committed")
    writer.begin(intent("aborted"))
    writer.mark_aborted("aborted")

    reopened = PersistentCommitJournal(path, REPOSITORY_ID)
    assert reopened.pending() == (intent("pending"),)
    reopened_writer = mutation(reopened)
    reopened_writer.mark_committed("pending")
    reopened_writer.mark_committed("pending")
    assert PersistentCommitJournal(path, REPOSITORY_ID).pending() == ()


def test_commit_journal_exposes_only_immutable_committed_intent_snapshots(tmp_path):
    path = tmp_path / "commit-intents.zlcj"
    journal = PersistentCommitJournal(path, REPOSITORY_ID)
    committed = intent("z-committed")
    first_committed = intent("a-committed")
    pending = intent("pending")
    writer = mutation(journal)
    writer.begin(committed)
    writer.mark_committed(committed.commit_id)
    writer.begin(first_committed)
    writer.mark_committed(first_committed.commit_id)
    writer.begin(pending)

    snapshot = journal.committed()
    assert snapshot == (first_committed, committed)
    assert isinstance(snapshot, tuple)
    with pytest.raises(AttributeError):
        snapshot[0].commit_id = "changed"

    reopened = PersistentCommitJournal(path, REPOSITORY_ID)
    assert reopened.committed() == (first_committed, committed)
    assert reopened.pending() == (pending,)

    memory = MemoryCommitJournal(REPOSITORY_ID)
    for commit_id in ("z-last", "a-first"):
        memory.begin(intent(commit_id))
        memory.mark_committed(commit_id)
    assert tuple(item.commit_id for item in memory.committed()) == (
        "a-first",
        "z-last",
    )


def test_persistent_journal_and_commit_coordinator_wiring_are_final_and_immutable(
    tmp_path,
):
    journal = PersistentCommitJournal(tmp_path / "commit-intents.zlcj", REPOSITORY_ID)
    lease = RepositoryRootLease(tmp_path, owner="journal-wiring-test")
    try:
        coordinator = RepositoryCommitCoordinator(
            journal,
            lambda _intent: CommitRecovery(False),
            root_lease=lease,
        )
    except BaseException:
        lease.close()
        raise
    with pytest.raises(TypeError, match="final"):
        class _JournalSubclass(PersistentCommitJournal):
            pass

    with pytest.raises(TypeError, match="final"):
        class _CoordinatorSubclass(RepositoryCommitCoordinator):
            pass

    with pytest.raises(AttributeError, match="immutable"):
        journal.repository_id = "redirected"
    with pytest.raises(AttributeError, match="immutable"):
        coordinator.repository_id = "redirected"
    lease.close()


def test_persistent_commit_journal_rejects_conflicting_intent(tmp_path):
    journal = PersistentCommitJournal(
        tmp_path / "commit-intents.zlcj", REPOSITORY_ID
    )
    writer = mutation(journal)
    writer.begin(intent("same-id"))
    with pytest.raises(ValueError, match="conflicting content"):
        writer.begin(
            CommitIntent(
                kind=CommitKind.FINAL,
                commit_id="same-id",
                run_id="another-run",
                safety_bundle_id="test-safety-bundle",
                target=intent("same-id").target,
                created_at=1.0,
            )
        )


def test_two_open_journals_cannot_resolve_one_intent_both_ways(tmp_path):
    path = tmp_path / "commit-intents.zlcj"
    first = PersistentCommitJournal(path, REPOSITORY_ID)
    second = PersistentCommitJournal(path, REPOSITORY_ID)
    first_writer = mutation(first)
    first_writer.begin(intent("shared"))
    first_writer.mark_committed("shared")
    with pytest.raises(ValueError, match="both ways"):
        mutation(second).mark_aborted("shared")
    assert first.pending() == second.pending() == ()


def test_persistent_commit_journal_path_is_bound_to_one_repository(tmp_path):
    path = tmp_path / "commit-intents.zlcj"
    PersistentCommitJournal(path, REPOSITORY_ID)
    with pytest.raises(ValueError, match="conflicting content"):
        PersistentCommitJournal(path, "another-repository")


def test_startup_reconciliation_receives_persistent_repository_target(tmp_path):
    path = tmp_path / "commit-intents.zlcj"
    journal = PersistentCommitJournal(path, REPOSITORY_ID)
    pending = intent("restart-pending")
    mutation(journal).begin(pending)
    observed = []

    def recover(value):
        observed.append(value.target)
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=value.target.target_ref,
                manifest_digest=value.target.expected_manifest_digest,
                result=value.target.target_ref,
            ),
        )

    reconciled = reconcile_pending_commits(mutation(journal), REPOSITORY_ID, recover)
    assert observed == [pending.target]
    assert reconciled[0].recovery.result.result == "artifacts/restart-pending"
    assert PersistentCommitJournal(path, REPOSITORY_ID).pending() == ()


def test_startup_recovery_cannot_launder_wrong_manifest_digest(tmp_path):
    journal = PersistentCommitJournal(
        tmp_path / "commit-intents.zlcj",
        REPOSITORY_ID,
    )
    pending = intent("wrong-recovery-digest")
    mutation(journal).begin(pending)

    with pytest.raises(ValueError, match="digest differs"):
        reconcile_pending_commits(
            mutation(journal),
            REPOSITORY_ID,
            lambda value: CommitRecovery(
                committed=True,
                result=PublishedManifest(
                    target_ref=value.target.target_ref,
                    manifest_digest="1" * 64,
                    result="laundered-result",
                ),
            ),
        )

    assert journal.pending() == (pending,)


def test_repository_coordinator_fails_closed_until_startup_pending_is_resolved(tmp_path):
    journal = PersistentCommitJournal(
        tmp_path / "commit-intents.zlcj",
        REPOSITORY_ID,
    )
    pending = intent("startup-gate")
    mutation(journal).begin(pending)

    def unavailable(_intent):
        raise OSError("repository unavailable")

    first_lease = RepositoryRootLease(tmp_path, owner="startup-gate-first")
    with pytest.raises(OSError, match="unavailable"):
        RepositoryCommitCoordinator(
            journal,
            unavailable,
            root_lease=first_lease,
        )
    first_lease.close()
    assert journal.pending() == (pending,)

    second_lease = RepositoryRootLease(tmp_path, owner="startup-gate-second")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda value: CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=value.target.target_ref,
                manifest_digest=value.target.expected_manifest_digest,
                result=value.target.target_ref,
            ),
        ),
        root_lease=second_lease,
    )
    assert coordinator.startup_reconciliations[0].intent == pending
    assert journal.pending() == ()
    second_lease.close()


def test_commit_authority_payload_is_immutable():
    journal = MemoryCommitJournal(REPOSITORY_ID)
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    target = intent("immutable-authority").target
    authority = coordinator.prepare(
        CommitKind.FINAL,
        "immutable-authority",
        CommitSubject("test-run", "test-safety-bundle"),
        target,
        lambda: PublishedManifest(
            target_ref=target.target_ref,
            manifest_digest=target.expected_manifest_digest,
            result="published",
        ),
    )

    replacements = {
        "kind": CommitKind.CHECKPOINT,
        "commit_id": "rogue-id",
        "run_id": "rogue-run",
        "safety_bundle_id": "rogue-safety",
        "target": intent("rogue-target").target,
        "journal": MemoryCommitJournal(REPOSITORY_ID),
        "recover": lambda value: CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=value.target.target_ref,
                manifest_digest=value.target.expected_manifest_digest,
                result="rogue",
            ),
        ),
        "_publish": lambda: PublishedManifest(
            target_ref=target.target_ref,
            manifest_digest=target.expected_manifest_digest,
            result="rogue",
        ),
    }
    for name, replacement in replacements.items():
        with pytest.raises(AttributeError, match="immutable"):
            setattr(authority, name, replacement)

    assert authority.target == target
    assert authority.kind is CommitKind.FINAL
    assert authority.commit_id == "immutable-authority"
    assert authority.run_id == "test-run"
    assert authority.safety_bundle_id == "test-safety-bundle"
    assert not hasattr(authority, "publish")
    assert not hasattr(authority, "journal")
    assert not hasattr(authority, "recover")
    assert not hasattr(authority, "_publish")


def test_checkpoint_commit_is_a_distinct_typed_operation_not_a_final_flag():
    journal = MemoryCommitJournal(REPOSITORY_ID)
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    target = intent("typed-checkpoint").target
    checkpoint = CheckpointCommit(
        coordinator.prepare(
            CommitKind.CHECKPOINT,
            "typed-checkpoint",
            CommitSubject("test-run", "test-safety-bundle"),
            target,
            lambda: PublishedManifest(
                target.target_ref,
                target.expected_manifest_digest,
                "raw-ref",
            ),
        ),
    )
    assert not isinstance(checkpoint, FinalCommit)
    assert checkpoint.target == target
    assert checkpoint.commit_id == "typed-checkpoint"
    assert checkpoint.run_id == "test-run"
    assert checkpoint.safety_bundle_id == "test-safety-bundle"
    with pytest.raises(TypeError, match="FINAL commit authority"):
        FinalCommit(checkpoint.authority)
    with pytest.raises(AttributeError, match="cannot assign"):
        checkpoint.commit_id = "mutated"


def test_abandon_and_consume_race_transfers_one_lease_borrow_exactly_once(
    tmp_path,
):
    journal = PersistentCommitJournal(
        tmp_path / "commit-intents.zlcj",
        REPOSITORY_ID,
    )
    lease = RepositoryRootLease(tmp_path, owner="abandon-consume-race")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(False),
        root_lease=lease,
    )
    target = intent("abandon-consume-race").target
    authority = coordinator.prepare(
        CommitKind.FINAL,
        "abandon-consume-race",
        CommitSubject("test-run", "test-safety-bundle"),
        target,
        lambda: PublishedManifest(
            target.target_ref,
            target.expected_manifest_digest,
            "result",
        ),
    )
    barrier = threading.Barrier(2)
    snapshots = []
    consume_errors = []

    def consume():
        barrier.wait()
        try:
            snapshots.append(_consume_commit_authority(authority))
        except RuntimeError as error:
            consume_errors.append(error)

    def abandon():
        barrier.wait()
        authority.abandon()

    threads = (threading.Thread(target=consume), threading.Thread(target=abandon))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2.0)
        assert not thread.is_alive()

    assert len(snapshots) + len(consume_errors) == 1
    assert coordinator._authorities == {}
    if snapshots:
        with pytest.raises(RuntimeError, match="outstanding commit authorities"):
            lease.close()
        snapshots[0].release_lifetime()
    lease.close()
