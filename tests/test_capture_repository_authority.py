"""Adversarial authority, resource, and recovery gates for raw captures."""

from __future__ import annotations

from dataclasses import replace
import gc
import pickle
import shutil
import threading

import pytest

from tests.test_capture_artifact_repository import (
    _deliver_when_armed,
    _runtime_and_spec,
)
from zlc_neutral_atom.artifacts.capture import (
    AdmittedCapture,
    CAPTURE_ARTIFACT_SCHEMA,
    CaptureArtifactRef,
    CaptureRepository,
    CaptureRepositoryResourcePolicy,
    CaptureResourceExceeded,
    DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY,
)
from zlc_neutral_atom import artifacts as artifact_api
from zlc_neutral_atom.runtime import (
    CommitIntent,
    CommitKind,
    CommitRecovery,
    CommitTarget,
    PublishVisibilityUnknown,
    RunFailed,
    compile_pipeline,
)
from zlc_neutral_atom.runtime.commit import (
    PersistentCommitJournal,
    RepositoryCommitCoordinator,
    _journal_mutation_authority,
)
from zlc_storage import (
    ContentAddressedStore,
    ContentRef,
    DirectoryDurabilityError,
    RepositoryRootBusy,
    RepositoryRootLease,
    decode,
)


@pytest.fixture
def capture_runtime():
    camera, runtime, spec = _runtime_and_spec()
    try:
        yield camera, runtime, spec
    finally:
        assert runtime.shutdown(timeout=2.0)


def _images():
    import numpy as np

    return (
        np.full((6, 8), 11, dtype=np.uint16),
        np.full((6, 8), 23, dtype=np.uint16),
    )


def _commit_capture(
    repository: CaptureRepository,
    camera,
    runtime,
    spec,
    *,
    checkpoint: bool = False,
    both: bool = False,
):
    base = compile_pipeline(spec)

    def finalize(context, result):
        if both:
            checkpoint_ref = context.commit_checkpoint(
                repository.checkpoint_commit(context, result)
            )
            final_ref = context.commit_final(repository.final_commit(context, result))
            assert checkpoint_ref == final_ref
            return final_ref
        if checkpoint:
            return context.commit_checkpoint(
                repository.checkpoint_commit(context, result)
            )
        return context.commit_final(repository.final_commit(context, result))

    plan = replace(
        base,
        name="capture authority fixture",
        finalize=finalize,
        requires_final_commit=not checkpoint or both,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        reference = handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
        return handle, reference
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)


def _manifest(repository: CaptureRepository, reference: CaptureArtifactRef):
    payload = repository._store_authority.read_manifest(
        "capture",
        reference.manifest_digest,
        max_bytes=repository.resource_policy.max_manifest_bytes,
    )
    return decode(payload)


def _pending_intent(
    reference: CaptureArtifactRef,
    artifact,
    *,
    kind: CommitKind,
    commit_id: str | None = None,
) -> CommitIntent:
    operation = "final" if kind is CommitKind.FINAL else "checkpoint"
    return CommitIntent(
        kind=kind,
        commit_id=(
            commit_id
            or f"capture-{operation}-{artifact.run_id}-{reference.manifest_digest}"
        ),
        run_id=artifact.run_id,
        safety_bundle_id=artifact.safety_bundle_id,
        target=CommitTarget(
            reference.repository_id,
            "capture",
            CAPTURE_ARTIFACT_SCHEMA,
            reference.target_ref,
            reference.manifest_digest,
        ),
        created_at=1.0,
    )


def _journal_writer(repository: CaptureRepository):
    return _journal_mutation_authority(repository._journal)


def test_visible_manifest_without_committed_intent_is_inspectable_but_not_admitted(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    source = CaptureRepository(tmp_path / "source", repository_id="captures")
    _handle, reference = _commit_capture(source, camera, runtime, spec)

    copied_root = tmp_path / "copied"
    shutil.copytree(source.root / "content", copied_root / "content")
    copied = CaptureRepository(copied_root, repository_id="captures")
    assert copied.load(reference).ref == reference
    with pytest.raises(PermissionError, match="no committed journal authority"):
        copied.admit(reference)
    assert not hasattr(copied, "put")


def test_capture_authority_types_are_exposed_from_the_artifact_owner_package():
    assert artifact_api.AdmittedCapture is AdmittedCapture
    assert artifact_api.CaptureRepository is CaptureRepository
    assert artifact_api.CaptureRepositoryResourcePolicy is CaptureRepositoryResourcePolicy
    assert artifact_api.CaptureResourceExceeded is CaptureResourceExceeded
    assert (
        artifact_api.DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY
        is DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY
    )


def test_prepared_unconsumed_authority_blocks_close_until_explicit_discard(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    operations = []
    base = compile_pipeline(spec)

    def finalize(context, result):
        operations.extend(
            (
                repository.checkpoint_commit(context, result),
                repository.checkpoint_commit(context, result),
            )
        )
        with pytest.raises(RuntimeError, match="outstanding commit authorities"):
            repository.close()
        with pytest.raises(RepositoryRootBusy, match="live owner"):
            CaptureRepository(repository.root)

        operations[0].abandon()
        tampered = operations[1].authority
        object.__setattr__(
            tampered,
            "_preparation",
            replace(tampered._preparation, commit_id="tampered-commit-id"),
        )
        tampered.abandon()
        assert repository._coordinator._authorities == {}
        return "prepared-only"

    plan = replace(
        base,
        name="prepared capture authority lifetime",
        finalize=finalize,
        requires_final_commit=False,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        assert handle.result(3.0) == "prepared-only"
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    assert repository._coordinator._authorities == {}

    repository.close()
    reopened = CaptureRepository(repository.root)
    reopened.close()


def test_abandoned_authority_is_reclaimed_when_finalize_raises(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    base = compile_pipeline(spec)

    def finalize(context, result):
        operation = repository.checkpoint_commit(context, result)
        assert operation.commit_id.startswith("capture-checkpoint-")
        raise RuntimeError("analysis failed after preparing capture commit")

    plan = replace(
        base,
        name="abandoned capture authority",
        finalize=finalize,
        requires_final_commit=False,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        with pytest.raises(RunFailed, match="analysis failed"):
            handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    gc.collect()
    assert repository._coordinator._authorities == {}
    repository.close()
    reopened = CaptureRepository(repository.root)
    reopened.close()


def test_inflight_commit_blocks_close_and_second_writer_without_poisoning_run(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    publish_entered = threading.Event()
    publish_release = threading.Event()
    real_publish = ContentAddressedStore._publish_manifest

    def blocked_publish(store, *args, **kwargs):
        if store is repository._store:
            publish_entered.set()
            if not publish_release.wait(3.0):
                raise TimeoutError("test did not release capture publication")
        return real_publish(store, *args, **kwargs)

    monkeypatch.setattr(
        ContentAddressedStore,
        "_publish_manifest",
        blocked_publish,
    )
    journal_constructions = 0
    real_journal_init = PersistentCommitJournal.__init__

    def counted_journal_init(journal, *args, **kwargs):
        nonlocal journal_constructions
        journal_constructions += 1
        real_journal_init(journal, *args, **kwargs)

    monkeypatch.setattr(
        PersistentCommitJournal,
        "__init__",
        counted_journal_init,
    )
    base = compile_pipeline(spec)

    def finalize(context, result):
        return context.commit_final(repository.final_commit(context, result))

    plan = replace(
        base,
        name="capture publication lease race",
        finalize=finalize,
        requires_final_commit=True,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        assert publish_entered.wait(2.0)
        with pytest.raises(RuntimeError, match="outstanding commit authorities"):
            repository.close()
        assert repository._root_lease.active
        with pytest.raises(RepositoryRootBusy, match="live owner"):
            CaptureRepository(repository.root)
        assert journal_constructions == 0

        publish_release.set()
        reference = handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
        assert repository.admit(reference).reference == reference
    finally:
        publish_release.set()
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    repository.close()
    reopened = CaptureRepository(repository.root)
    assert journal_constructions == 1
    reopened.close()


def test_lost_manifest_directory_ack_remains_pending_until_durable_confirm(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    import zlc_storage.content_store as content_store

    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    manifest_parent = repository.root / "content" / "manifests" / "capture"
    content_store.durability.durable_mkdir(manifest_parent)
    real_flush = content_store.durability.flush_directory
    target_flushes = 0

    def fail_first_two_target_flushes(directory):
        nonlocal target_flushes
        if directory == manifest_parent:
            target_flushes += 1
            if target_flushes <= 2:
                raise DirectoryDurabilityError(
                    "capture manifest directory acknowledgement lost"
                )
        return real_flush(directory)

    monkeypatch.setattr(
        content_store.durability,
        "flush_directory",
        fail_first_two_target_flushes,
    )
    base = compile_pipeline(spec)

    def finalize(context, result):
        return context.commit_final(repository.final_commit(context, result))

    plan = replace(
        base,
        name="capture durable-confirm retry",
        finalize=finalize,
        requires_final_commit=True,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        handle.wait_for(
            lambda snapshot: snapshot.phase == "final-commit-reconciliation-failed",
            3.0,
        )
        assert target_flushes == 2
        assert len(repository._journal.pending()) == 1
        with pytest.raises(RuntimeError, match="outstanding commit authorities"):
            repository.close()
        with pytest.raises(RepositoryRootBusy, match="live owner"):
            CaptureRepository(repository.root)

        assert handle.retry_recovery()
        reference = handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
        assert target_flushes >= 3
        assert repository._journal.pending() == ()
        assert repository.admit(reference).reference == reference
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    repository.close()
    reopened = CaptureRepository(repository.root)
    reopened.close()


def test_recovery_durable_confirmation_never_recreates_a_disappeared_manifest(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    _handle, reference = _commit_capture(
        repository,
        camera,
        runtime,
        spec,
        checkpoint=True,
    )
    artifact = repository.load(reference)
    _journal_writer(repository).begin(
        _pending_intent(reference, artifact, kind=CommitKind.FINAL)
    )
    manifest_path = repository._store._manifest_path(
        "capture",
        reference.manifest_digest,
    )
    repository.close()
    real_confirm = ContentAddressedStore._confirm_manifest_durable

    def disappear_before_confirm(store, namespace, digest, **kwargs):
        if store.root == repository.root / "content":
            store._manifest_path(namespace, digest).unlink()
        return real_confirm(store, namespace, digest, **kwargs)

    monkeypatch.setattr(
        ContentAddressedStore,
        "_confirm_manifest_durable",
        disappear_before_confirm,
    )
    with pytest.raises(FileNotFoundError):
        CaptureRepository(repository.root)
    assert not manifest_path.exists()


def test_checkpoint_and_final_admission_are_process_local_and_final_wins(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    _handle, reference = _commit_capture(
        repository,
        camera,
        runtime,
        spec,
        both=True,
    )

    admitted = repository.admit(reference)
    assert type(admitted) is AdmittedCapture
    assert admitted.reference == reference
    assert admitted.artifact.ref == reference
    assert admitted.commit_kind is CommitKind.FINAL
    assert admitted.commit_id.startswith("capture-final-")
    assert len(admitted.evidence_digest) == 64
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(admitted)
    with pytest.raises(AttributeError, match="immutable"):
        admitted._commit_id = "forged"
    with pytest.raises(TypeError, match="final"):
        class _AdmittedSubclass(AdmittedCapture):
            pass

    repository.close()
    reopened = CaptureRepository(
        repository.root,
        repository_id=repository.repository_id,
    )
    reopened_admitted = reopened.admit(reference)
    assert reopened_admitted.commit_kind is CommitKind.FINAL
    assert reopened_admitted.evidence_digest == admitted.evidence_digest


def test_checkpoint_only_capture_is_admitted_as_checkpoint(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    _handle, reference = _commit_capture(
        repository,
        camera,
        runtime,
        spec,
        checkpoint=True,
    )
    admitted = repository.admit(reference)
    assert admitted.commit_kind is CommitKind.CHECKPOINT
    assert admitted.commit_id.startswith("capture-checkpoint-")


def test_repository_is_final_and_rejects_shadow_or_authority_drift(tmp_path):
    repository = CaptureRepository(tmp_path / "captures", repository_id="captures")
    with pytest.raises(TypeError, match="final"):
        class _RepositorySubclass(CaptureRepository):
            pass

    for name, value in (
        ("root", tmp_path / "redirected"),
        ("repository_id", "redirected"),
        ("load", lambda _reference: object()),
        ("_store", ContentAddressedStore(tmp_path / "other-content")),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(repository, name, value)

    original_root = repository.root
    object.__setattr__(repository, "root", (tmp_path / "redirected").resolve())
    try:
        with pytest.raises(RuntimeError, match="authority changed"):
            _ = repository.startup_reconciliations
    finally:
        object.__setattr__(repository, "root", original_root)

    original_id = repository.repository_id
    object.__setattr__(repository, "repository_id", "redirected")
    try:
        with pytest.raises(RuntimeError, match="authority changed"):
            _ = repository.startup_reconciliations
    finally:
        object.__setattr__(repository, "repository_id", original_id)

    fake_store = ContentAddressedStore(tmp_path / "fake-store")
    original_store = repository._store
    object.__setattr__(repository, "_store", fake_store)
    try:
        with pytest.raises(RuntimeError, match="authority changed"):
            _ = repository.startup_reconciliations
    finally:
        object.__setattr__(repository, "_store", original_store)

    fake_root = tmp_path / "fake-repository"
    fake_lease = RepositoryRootLease(fake_root, owner="fake-capture")
    fake_journal = PersistentCommitJournal(
        fake_root / "fake-commit.journal",
        repository.repository_id,
    )
    original_journal = repository._journal
    object.__setattr__(repository, "_journal", fake_journal)
    try:
        with pytest.raises(RuntimeError, match="authority changed"):
            _ = repository.startup_reconciliations
    finally:
        object.__setattr__(repository, "_journal", original_journal)

    fake_coordinator = RepositoryCommitCoordinator(
        fake_journal,
        lambda _intent: CommitRecovery(False),
        root_lease=fake_lease,
    )
    original_coordinator = repository._coordinator
    object.__setattr__(repository, "_coordinator", fake_coordinator)
    try:
        with pytest.raises(RuntimeError, match="authority changed"):
            _ = repository.startup_reconciliations
    finally:
        object.__setattr__(repository, "_coordinator", original_coordinator)
        fake_lease.close()


def test_repository_copies_caller_policy_and_detects_internal_policy_drift(tmp_path):
    caller_policy = CaptureRepositoryResourcePolicy(max_cells=17)
    repository = CaptureRepository(
        tmp_path / "captures",
        resource_policy=caller_policy,
    )
    assert repository.resource_policy == caller_policy
    assert repository.resource_policy is not caller_policy

    object.__setattr__(caller_policy, "max_cells", 1)
    assert repository.resource_policy.max_cells == 17
    assert repository.startup_reconciliations == ()

    object.__setattr__(repository.resource_policy, "max_cells", 1)
    with pytest.raises(RuntimeError, match="authority changed"):
        _ = repository.startup_reconciliations


@pytest.mark.parametrize(
    ("field", "limit", "message"),
    (
        ("max_cells", 1, "metadata count"),
        ("max_manifest_bytes", 1, "manifest"),
        ("max_data_block_blob_bytes", 1, "DataBlock blob"),
        ("max_data_array_bytes", 1, "DataBlock arrays"),
        ("max_metadata_blob_bytes", 1, "metadata blob"),
    ),
)
def test_load_rejects_whole_oversized_multidimensional_content_without_reduction(
    tmp_path,
    capture_runtime,
    field,
    limit,
    message,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    _handle, reference = _commit_capture(repository, camera, runtime, spec)
    constrained = replace(
        DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY,
        **{field: limit},
    )
    repository.close()
    reopened = CaptureRepository(
        repository.root,
        repository_id=repository.repository_id,
        resource_policy=constrained,
    )
    with pytest.raises(CaptureResourceExceeded, match=message):
        reopened.load(reference)


def test_cell_budget_rejects_manifest_before_canonical_value_materialization(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    import zlc_storage.canonical as canonical

    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    _handle, reference = _commit_capture(repository, camera, runtime, spec)
    repository.close()
    constrained = CaptureRepository(
        repository.root,
        resource_policy=replace(
            DEFAULT_CAPTURE_REPOSITORY_RESOURCE_POLICY,
            max_cells=1,
        ),
    )
    materialized = False
    real_decode_value = canonical._decode_value

    def observed_materialization(*args, **kwargs):
        nonlocal materialized
        materialized = True
        return real_decode_value(*args, **kwargs)

    monkeypatch.setattr(canonical, "_decode_value", observed_materialization)
    with pytest.raises(CaptureResourceExceeded, match="metadata count"):
        constrained.load(reference)
    assert not materialized
    constrained.close()


def test_visible_manifest_with_missing_blob_fails_recovery_closed(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    _handle, reference = _commit_capture(
        repository,
        camera,
        runtime,
        spec,
        checkpoint=True,
    )
    artifact = repository.load(reference)
    _journal_writer(repository).begin(
        _pending_intent(reference, artifact, kind=CommitKind.FINAL)
    )
    manifest = _manifest(repository, reference)
    block_ref = ContentRef(
        manifest["data_block_blob"]["digest"],
        manifest["data_block_blob"]["size"],
    )
    repository._store._blob_path(block_ref.digest).unlink()
    repository.close()

    with pytest.raises(FileNotFoundError):
        CaptureRepository(
            repository.root,
            repository_id=repository.repository_id,
        )


def test_plain_post_replace_failure_becomes_unknown_and_reconciles(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    real_publish = ContentAddressedStore._publish_manifest
    calls = 0

    def publish_then_fail(store, *args, **kwargs):
        nonlocal calls
        stored = real_publish(store, *args, **kwargs)
        if store is repository._store:
            calls += 1
            if calls == 1:
                raise OSError("directory durability acknowledgement lost")
        return stored

    monkeypatch.setattr(
        ContentAddressedStore,
        "_publish_manifest",
        publish_then_fail,
    )
    handle, reference = _commit_capture(repository, camera, runtime, spec)

    assert calls == 1
    assert repository.admit(reference).reference == reference
    assert "acknowledgement" in handle.snapshot().commit_recovery_warning
    assert repository._journal.pending() == ()


def test_recovery_rejects_kind_commit_id_mismatch_before_absence(tmp_path):
    repository = CaptureRepository(tmp_path / "captures")
    digest = "1" * 64
    reference = CaptureArtifactRef(repository.repository_id, digest)
    fake_artifact = type(
        "Evidence",
        (),
        {"run_id": "run", "safety_bundle_id": "safety"},
    )()
    _journal_writer(repository).begin(
        _pending_intent(
            reference,
            fake_artifact,
            kind=CommitKind.CHECKPOINT,
            commit_id=f"capture-final-run-{digest}",
        )
    )
    repository.close()
    with pytest.raises(ValueError, match="commit id differs"):
        CaptureRepository(
            repository.root,
            repository_id=repository.repository_id,
        )


@pytest.mark.parametrize(
    "target_ref",
    [
        f"calibration/{'1' * 64}",
        f"capture/{'2' * 64}",
    ],
)
def test_capture_recovery_rejects_namespace_or_target_digest_mismatch(
    tmp_path,
    target_ref,
):
    repository = CaptureRepository(tmp_path / "captures")
    reference = CaptureArtifactRef(repository.repository_id, "1" * 64)
    evidence = type(
        "Evidence",
        (),
        {"run_id": "run", "safety_bundle_id": "safety"},
    )()
    intent = _pending_intent(reference, evidence, kind=CommitKind.FINAL)
    mismatched = replace(
        intent,
        target=replace(intent.target, target_ref=target_ref),
    )

    with pytest.raises(ValueError, match="target ref and digest differ"):
        repository._recover(mismatched)


def test_absent_manifest_recovery_aborts_by_inspection_without_publication(tmp_path):
    repository = CaptureRepository(tmp_path / "captures")
    digest = "2" * 64
    reference = CaptureArtifactRef(repository.repository_id, digest)
    evidence = type(
        "Evidence",
        (),
        {"run_id": "run", "safety_bundle_id": "safety"},
    )()
    _journal_writer(repository).begin(
        _pending_intent(reference, evidence, kind=CommitKind.FINAL)
    )
    repository.close()

    reopened = CaptureRepository(
        repository.root,
        repository_id=repository.repository_id,
    )
    assert len(reopened.startup_reconciliations) == 1
    assert not reopened.startup_reconciliations[0].recovery.committed
    assert reopened._journal.pending() == ()
    manifest_path = reopened._store._manifest_path("capture", digest)
    assert not manifest_path.exists()
