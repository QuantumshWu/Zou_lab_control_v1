"""Current FINAL-only, crash-consistent commit contracts."""

from __future__ import annotations

import threading

import pytest

import zlc_neutral_atom.runtime.commit as commit_api
from zlc_neutral_atom.runtime.commit import (
    CommitIntent,
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishVisibilityUnknown,
    PublishedManifest,
    RepositoryCommitCoordinator,
    _consume_commit_authority,
    publish_manifest_with_visibility_reconciliation,
)
from zlc_storage import ContentRef, RepositoryRootLease, StoredManifest, sha256_digest


REPOSITORY_ID = "test-repository"
MANIFEST_NAMESPACE = "test-manifest"
MANIFEST_PAYLOAD = b"expected manifest payload"
MANIFEST_DIGEST = sha256_digest(MANIFEST_PAYLOAD)


class _ManifestFaultAuthority:
    def __init__(
        self,
        *,
        publish_result=None,
        publish_error: BaseException | None = None,
        visible: bytes | None = None,
        read_error: BaseException | None = None,
    ) -> None:
        self.publish_result = publish_result
        self.publish_error = publish_error
        self.visible = visible
        self.read_error = read_error
        self.publish_calls = []
        self.read_calls = []

    def publish_manifest(self, namespace, payload, *, expected_digest=None):
        self.publish_calls.append((namespace, payload, expected_digest))
        if self.publish_error is not None:
            raise self.publish_error
        return self.publish_result

    def read_manifest(self, namespace, digest, *, max_bytes=None):
        self.read_calls.append((namespace, digest, max_bytes))
        if self.read_error is not None:
            raise self.read_error
        return self.visible


def _publish_with_reconciliation(authority):
    return publish_manifest_with_visibility_reconciliation(
        authority,
        MANIFEST_NAMESPACE,
        MANIFEST_PAYLOAD,
        expected_digest=MANIFEST_DIGEST,
        max_bytes=4096,
    )


def _target(commit_id: str) -> CommitTarget:
    return CommitTarget(
        repository_id=REPOSITORY_ID,
        artifact_kind="test-artifact",
        artifact_format="tests.Artifact",
        target_ref=f"artifacts/{commit_id}",
        expected_manifest_digest="0" * 64,
    )


def _intent(commit_id: str) -> CommitIntent:
    return CommitIntent(
        commit_id=commit_id,
        run_id="test-run",
        safety_bundle_id="test-safety-bundle",
        target=_target(commit_id),
        created_at=1.0,
    )


def _open_coordinator(root, recover=lambda _intent: None):
    journal = PersistentCommitJournal(root / "commit-intents.zlcj", REPOSITORY_ID)
    lease = RepositoryRootLease(root)
    try:
        coordinator = RepositoryCommitCoordinator(
            journal,
            recover,
            root_lease=lease,
        )
    except BaseException:
        lease.close()
        journal.close()
        raise
    return journal, lease, coordinator


def _prepare(coordinator, commit_id: str) -> FinalCommit[str]:
    target = _target(commit_id)
    return coordinator.prepare(
        commit_id,
        "test-run",
        "test-safety-bundle",
        target,
        lambda: PublishedManifest(
            target.target_ref,
            target.expected_manifest_digest,
            "published-result",
        ),
    )


def test_manifest_visibility_success_returns_without_readback():
    stored = StoredManifest(
        MANIFEST_NAMESPACE,
        ContentRef(MANIFEST_DIGEST, len(MANIFEST_PAYLOAD)),
    )
    authority = _ManifestFaultAuthority(publish_result=stored)

    assert _publish_with_reconciliation(authority) is stored
    assert authority.publish_calls == [
        (MANIFEST_NAMESPACE, MANIFEST_PAYLOAD, MANIFEST_DIGEST)
    ]
    assert authority.read_calls == []


def test_explicit_unknown_visibility_is_propagated_without_readback():
    unknown = PublishVisibilityUnknown("storage owner classified visibility")
    authority = _ManifestFaultAuthority(publish_error=unknown)

    with pytest.raises(PublishVisibilityUnknown) as failure:
        _publish_with_reconciliation(authority)

    assert failure.value is unknown
    assert authority.read_calls == []


def test_verified_absence_preserves_the_original_publish_failure():
    publish_error = OSError("publication failed before replace")
    authority = _ManifestFaultAuthority(
        publish_error=publish_error,
        read_error=FileNotFoundError("manifest absent"),
    )

    with pytest.raises(OSError) as failure:
        _publish_with_reconciliation(authority)

    assert failure.value is publish_error
    assert authority.read_calls == [(MANIFEST_NAMESPACE, MANIFEST_DIGEST, 4096)]


@pytest.mark.parametrize(
    ("visible", "read_error", "message"),
    (
        (MANIFEST_PAYLOAD, None, "acknowledgement"),
        (b"unexpected", None, "unexpected bytes"),
        (None, PermissionError("unreadable"), "could not be verified"),
    ),
)
def test_every_non_absent_post_publish_state_is_visibility_unknown(
    visible,
    read_error,
    message,
):
    authority = _ManifestFaultAuthority(
        publish_error=OSError("durability acknowledgement failed"),
        visible=visible,
        read_error=read_error,
    )

    with pytest.raises(PublishVisibilityUnknown, match=message):
        _publish_with_reconciliation(authority)


def test_public_commit_surface_is_final_only():
    assert commit_api.__all__ == [
        "CommitIntent",
        "CommitTarget",
        "FinalCommit",
        "PublishVisibilityUnknown",
        "PublishedManifest",
    ]
    for removed in (
        "CheckpointCommit",
        "CommitKind",
        "CommitRecovery",
        "CommitSubject",
        "MemoryCommitJournal",
    ):
        assert not hasattr(commit_api, removed)


def test_coordinator_mints_one_immutable_final_authority(tmp_path):
    journal, lease, coordinator = _open_coordinator(tmp_path)
    authority = _prepare(coordinator, "immutable")
    try:
        assert authority.commit_id == "immutable"
        assert authority.run_id == "test-run"
        assert authority.safety_bundle_id == "test-safety-bundle"
        assert authority.target == _target("immutable")
        with pytest.raises(AttributeError, match="immutable"):
            authority.commit_id = "changed"

        snapshot = _consume_commit_authority(authority)
        assert snapshot.publish_validated(_intent("immutable")) == "published-result"
        snapshot.release_lifetime()
        with pytest.raises(RuntimeError, match="already consumed"):
            _consume_commit_authority(authority)
    finally:
        coordinator.close()
        lease.close()


def test_unconsumed_final_authority_blocks_root_close_until_abandoned(tmp_path):
    _journal, lease, coordinator = _open_coordinator(tmp_path)
    authority = _prepare(coordinator, "abandoned")
    with pytest.raises(RuntimeError, match="outstanding operations"):
        lease.close()
    authority.abandon()
    coordinator.close()
    lease.close()


def test_startup_reconciles_pending_intent_by_inspection_only(tmp_path):
    journal, lease, coordinator = _open_coordinator(tmp_path)
    authority = _prepare(coordinator, "restart")
    snapshot = _consume_commit_authority(authority)
    pending = _intent("restart")
    snapshot.begin_intent(pending)
    snapshot.release_lifetime()
    coordinator.close()
    lease.close()

    observed = []

    def recover(intent):
        observed.append(intent)
        return PublishedManifest(
            intent.target.target_ref,
            intent.target.expected_manifest_digest,
            "recovered",
        )

    reopened, reopened_lease, reopened_coordinator = _open_coordinator(
        tmp_path,
        recover,
    )
    try:
        assert observed == [pending]
        assert reopened.pending() == ()
        assert reopened.committed_for(pending.target) == (pending,)
    finally:
        reopened_coordinator.close()
        reopened_lease.close()


def test_recovery_rejects_a_manifest_for_another_target(tmp_path):
    _journal, lease, coordinator = _open_coordinator(tmp_path)
    authority = _prepare(coordinator, "wrong-digest")
    snapshot = _consume_commit_authority(authority)
    pending = _intent("wrong-digest")
    snapshot.begin_intent(pending)
    snapshot.release_lifetime()
    coordinator.close()
    lease.close()

    journal = PersistentCommitJournal(
        tmp_path / "commit-intents.zlcj",
        REPOSITORY_ID,
    )
    wrong_lease = RepositoryRootLease(tmp_path)
    try:
        with pytest.raises(ValueError, match="digest differs"):
            RepositoryCommitCoordinator(
                journal,
                lambda intent: PublishedManifest(
                    intent.target.target_ref,
                    "1" * 64,
                    "wrong",
                ),
                root_lease=wrong_lease,
            )
        assert journal.pending() == (pending,)
    finally:
        journal.close()
        wrong_lease.close()


def test_abandon_and_consume_race_transfers_one_borrow(tmp_path):
    _journal, lease, coordinator = _open_coordinator(tmp_path)
    authority = _prepare(coordinator, "race")
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
    for snapshot in snapshots:
        snapshot.release_lifetime()
    coordinator.close()
    lease.close()
