"""Crash-consistent final-artifact intent contracts."""

from __future__ import annotations

import pytest

from zlc_neutral_atom.runtime import (
    CheckpointCommit,
    CommitIntent,
    CommitKind,
    CommitRecovery,
    CommitSubject,
    CommitTarget,
    MemoryCommitJournal,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
    FinalCommit,
    reconcile_pending_commits,
)


REPOSITORY_ID = "test-repository"


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


def test_persistent_commit_journal_recovers_pending_and_both_resolutions(tmp_path):
    path = tmp_path / "commit-intents.zlcj"
    journal = PersistentCommitJournal(path, REPOSITORY_ID)
    journal.begin(intent("pending"))
    journal.begin(intent("committed"))
    journal.mark_committed("committed")
    journal.begin(intent("aborted"))
    journal.mark_aborted("aborted")

    reopened = PersistentCommitJournal(path, REPOSITORY_ID)
    assert reopened.pending() == (intent("pending"),)
    reopened.mark_committed("pending")
    reopened.mark_committed("pending")
    assert PersistentCommitJournal(path, REPOSITORY_ID).pending() == ()


def test_persistent_commit_journal_rejects_conflicting_intent(tmp_path):
    journal = PersistentCommitJournal(
        tmp_path / "commit-intents.zlcj", REPOSITORY_ID
    )
    journal.begin(intent("same-id"))
    with pytest.raises(ValueError, match="conflicting content"):
        journal.begin(
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
    first.begin(intent("shared"))
    first.mark_committed("shared")
    with pytest.raises(ValueError, match="both ways"):
        second.mark_aborted("shared")
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
    journal.begin(pending)
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

    reconciled = reconcile_pending_commits(journal, REPOSITORY_ID, recover)
    assert observed == [pending.target]
    assert reconciled[0].recovery.result.result == "artifacts/restart-pending"
    assert PersistentCommitJournal(path, REPOSITORY_ID).pending() == ()


def test_startup_recovery_cannot_launder_wrong_manifest_digest(tmp_path):
    journal = PersistentCommitJournal(
        tmp_path / "commit-intents.zlcj",
        REPOSITORY_ID,
    )
    pending = intent("wrong-recovery-digest")
    journal.begin(pending)

    with pytest.raises(ValueError, match="digest differs"):
        reconcile_pending_commits(
            journal,
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
    journal.begin(pending)

    def unavailable(_intent):
        raise OSError("repository unavailable")

    with pytest.raises(OSError, match="unavailable"):
        RepositoryCommitCoordinator(journal, unavailable)
    assert journal.pending() == (pending,)

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
    )
    assert coordinator.startup_reconciliations[0].intent == pending
    assert journal.pending() == ()


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
