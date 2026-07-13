"""Raw capture checkpoints reuse the CaptureArtifact final-commit protocol."""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pytest

from tests.test_capture_artifact_repository import (
    _deliver_when_armed,
    _runtime_and_spec,
)
from zlc_neutral_atom.artifacts import CaptureArtifactRef, CaptureRepository
from zlc_neutral_atom.artifacts.capture import CAPTURE_ARTIFACT_SCHEMA
from zlc_neutral_atom.runtime import (
    CancellationToken,
    CheckpointCommit,
    CheckpointDisposition,
    FinalCommit,
    PostSafetyContext,
    PublishVisibilityUnknown,
    RunFailed,
    RunId,
    compile_pipeline,
)
from zlc_neutral_atom.runtime.commit import PersistentCommitJournal
from zlc_storage import ContentStoreAuthority


@pytest.fixture
def capture_runtime():
    camera, runtime, spec = _runtime_and_spec()
    try:
        yield camera, runtime, spec
    finally:
        assert runtime.shutdown(timeout=2.0)


def _images():
    return (
        np.full((6, 8), 11, dtype=np.uint16),
        np.full((6, 8), 23, dtype=np.uint16),
    )


def _run_plan(camera, runtime, plan):
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        result = handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
        return handle, result
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)


def test_capture_checkpoint_publishes_typed_artifact_without_finalizing_run(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    base = compile_pipeline(spec)
    observed = {}

    def finalize(context, result):
        operation = repository.checkpoint_commit(context, result)
        observed["operation"] = operation
        assert isinstance(operation, CheckpointCommit)
        assert not isinstance(operation, FinalCommit)
        with pytest.raises(TypeError, match="FinalCommit"):
            context.commit_final(operation)
        return context.commit_checkpoint(operation)

    plan = replace(
        base,
        name="capture checkpoint",
        finalize=finalize,
        requires_final_commit=False,
    )
    handle, reference = _run_plan(camera, runtime, plan)

    assert isinstance(reference, CaptureArtifactRef)
    assert repository.load(reference).ref == reference
    snapshot = handle.snapshot()
    assert not snapshot.final_committed
    assert snapshot.committed_checkpoint is not None
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED

    operation = observed["operation"]
    target = operation.target
    assert snapshot.committed_checkpoint.commit_id == operation.commit_id
    assert snapshot.committed_checkpoint.target == target
    assert target.repository_id == repository.repository_id
    assert target.artifact_kind == "capture"
    assert target.schema_version == CAPTURE_ARTIFACT_SCHEMA
    assert target.target_ref == reference.target_ref
    assert target.expected_manifest_digest == reference.manifest_digest
    assert operation.commit_id == (
        f"capture-checkpoint-{snapshot.run_id.value}-{reference.manifest_digest}"
    )
    assert repository._journal.pending() == ()
    assert repository._journal._intents[operation.commit_id].target == target
    assert repository._coordinator._authorities == {}


def test_capture_checkpoint_recovers_visible_manifest_after_lost_ack(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    repository_store_authority = repository._store.authority()
    real_publish = ContentStoreAuthority.publish_manifest
    publish_calls = 0

    def publish_then_lose_ack(authority, *args, **kwargs):
        nonlocal publish_calls
        stored = real_publish(authority, *args, **kwargs)
        if authority is not repository_store_authority:
            return stored
        publish_calls += 1
        if publish_calls == 1:
            raise PublishVisibilityUnknown("capture manifest acknowledgement lost")
        return stored

    monkeypatch.setattr(
        ContentStoreAuthority,
        "publish_manifest",
        publish_then_lose_ack,
    )
    base = compile_pipeline(spec)

    def finalize(context, result):
        return context.commit_checkpoint(repository.checkpoint_commit(context, result))

    plan = replace(base, name="lost-ack capture checkpoint", finalize=finalize)
    handle, reference = _run_plan(camera, runtime, plan)

    assert publish_calls == 1
    assert repository.load(reference).ref == reference
    snapshot = handle.snapshot()
    assert not snapshot.final_committed
    assert snapshot.committed_checkpoint is not None
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED
    assert "acknowledgement lost" in snapshot.commit_recovery_warning
    assert repository._journal.pending() == ()
    assert repository._coordinator._authorities == {}


def test_capture_checkpoint_and_final_factories_share_artifact_identity_not_intent(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    base = compile_pipeline(spec)
    observed = {}

    def finalize(context, result):
        checkpoint = repository.checkpoint_commit(context, result)
        final = repository.final_commit(context, result)
        observed.update(checkpoint=checkpoint, final=final)

        assert checkpoint.target == final.target
        assert checkpoint.commit_id != final.commit_id
        with pytest.raises(TypeError, match="FINAL commit authority"):
            FinalCommit(checkpoint.authority)
        with pytest.raises(TypeError, match="CHECKPOINT commit authority"):
            CheckpointCommit(final.authority)
        with pytest.raises(TypeError, match="FinalCommit"):
            context.commit_final(checkpoint)
        with pytest.raises(TypeError, match="CheckpointCommit"):
            context.commit_checkpoint(final)

        checkpoint_ref = context.commit_checkpoint(checkpoint)
        final_ref = context.commit_final(final)
        assert checkpoint_ref == final_ref
        return final_ref

    plan = replace(
        base,
        name="capture checkpoint then same raw final",
        finalize=finalize,
        requires_final_commit=True,
    )
    handle, reference = _run_plan(camera, runtime, plan)

    checkpoint = observed["checkpoint"]
    final = observed["final"]
    assert isinstance(checkpoint, CheckpointCommit)
    assert isinstance(final, FinalCommit)
    assert checkpoint.target == final.target
    assert checkpoint.target.target_ref == reference.target_ref
    assert checkpoint.target.expected_manifest_digest == reference.manifest_digest
    assert checkpoint.commit_id == (
        f"capture-checkpoint-{handle.run_id.value}-{reference.manifest_digest}"
    )
    assert final.commit_id == (
        f"capture-final-{handle.run_id.value}-{reference.manifest_digest}"
    )
    snapshot = handle.snapshot()
    assert snapshot.final_committed
    assert snapshot.committed_checkpoint is not None
    assert set(repository._journal._intents) == {
        checkpoint.commit_id,
        final.commit_id,
    }
    assert {
        intent.target for intent in repository._journal._intents.values()
    } == {checkpoint.target}
    assert {
        intent.kind.value for intent in repository._journal._intents.values()
    } == {"CHECKPOINT", "FINAL"}
    assert repository._journal.pending() == ()
    assert repository._coordinator._authorities == {}


def test_capture_checkpoint_factory_rejects_non_context_and_context_forgery(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        result = runtime.controller.run(compile_pipeline(spec))
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    with pytest.raises(TypeError, match="PostSafetyContext"):
        repository.checkpoint_commit(object(), result)
    with pytest.raises(PermissionError, match="minted by RunController"):
        PostSafetyContext(
            run_id=RunId("different-run"),
            cancellation=CancellationToken(),
            deadline=None,
            safety_bundle_id="different-safety-bundle",
            handle=object(),
        )

    assert repository._coordinator._authorities == {}
    assert repository._journal.pending() == ()
    manifest_root = tmp_path / "captures" / "content" / "manifests" / "capture"
    assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()


def test_wrong_capture_result_is_rejected_before_staging_or_authority(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    base = compile_pipeline(spec)

    def finalize(context, result):
        with pytest.raises(TypeError, match="exact pipeline result"):
            repository.checkpoint_commit(context, object())
        assert repository._coordinator._authorities == {}
        return result

    plan = replace(base, name="reject invalid capture result", finalize=finalize)
    _handle, _result = _run_plan(camera, runtime, plan)

    assert repository._coordinator._authorities == {}
    assert repository._journal._intents == {}
    blob_root = tmp_path / "captures" / "content" / "blobs"
    assert not blob_root.exists() or not any(path.is_file() for path in blob_root.rglob("*"))


def test_capture_checkpoint_authority_cannot_be_transplanted_between_runs(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    base = compile_pipeline(spec)
    captured = {}

    def finalize_first(context, result):
        captured["operation"] = repository.checkpoint_commit(context, result)
        captured["run_id"] = context.run_id.value
        return result

    first_plan = replace(
        base,
        name="mint first-run capture authority",
        finalize=finalize_first,
    )
    first_handle, _first_result = _run_plan(camera, runtime, first_plan)
    assert first_handle.run_id.value == captured["run_id"]
    # Leaving finalize without consuming the prepared operation abandons its
    # coordinator snapshot immediately.  The caller-held wrapper remains an
    # inert, run-bound value and cannot retain the repository-root lease.
    assert repository._coordinator._authorities == {}

    def finalize_second(context, _result):
        # A public wrapper can be recreated, but it cannot rewrite the complete
        # preparation identity sealed inside the one-shot authority.
        transplanted = CheckpointCommit(captured["operation"].authority)
        return context.commit_checkpoint(transplanted)

    second_plan = replace(
        base,
        name="reject first-run capture authority in second Run",
        finalize=finalize_second,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        second_handle = runtime.controller.start(second_plan)
        with pytest.raises(RunFailed, match="another Run"):
            second_handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    assert repository._coordinator._authorities == {}
    assert repository._journal._intents == {}
    manifest_root = tmp_path / "captures" / "content" / "manifests" / "capture"
    assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()


def test_expired_capture_context_is_rejected_before_staging_or_authority(
    tmp_path,
    capture_runtime,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    base = compile_pipeline(spec)

    def finalize(context, result):
        assert context.deadline is not None
        while time.monotonic() <= context.deadline:
            time.sleep(0.001)
        return repository.checkpoint_commit(context, result)

    plan = replace(
        base,
        name="reject expired capture staging",
        finalize=finalize,
        timeout_seconds=0.15,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        with pytest.raises(RunFailed, match="monotonic deadline"):
            handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    assert repository._coordinator._authorities == {}
    assert repository._journal._intents == {}
    blob_root = tmp_path / "captures" / "content" / "blobs"
    assert not blob_root.exists() or not any(path.is_file() for path in blob_root.rglob("*"))


def test_capture_staging_crossing_deadline_cannot_mint_commit_authority(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    real_stage = CaptureRepository._stage_pipeline_result

    def stage_then_expire(owner, result, context):
        staged = real_stage(owner, result, context)
        if owner is not repository:
            return staged
        assert context.deadline is not None
        while time.monotonic() <= context.deadline:
            time.sleep(0.001)
        return staged

    monkeypatch.setattr(
        CaptureRepository,
        "_stage_pipeline_result",
        stage_then_expire,
    )
    base = compile_pipeline(spec)

    def finalize(context, result):
        return repository.checkpoint_commit(context, result)

    plan = replace(
        base,
        name="expire between capture staging and authority",
        finalize=finalize,
        timeout_seconds=0.15,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        with pytest.raises(RunFailed, match="monotonic deadline"):
            handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    assert repository._coordinator._authorities == {}
    assert repository._journal._intents == {}
    blob_root = tmp_path / "captures" / "content" / "blobs"
    assert any(path.is_file() for path in blob_root.rglob("*"))
    manifest_root = tmp_path / "captures" / "content" / "manifests" / "capture"
    assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()


def test_capture_journal_begin_crossing_deadline_aborts_before_manifest_publish(
    tmp_path,
    capture_runtime,
    monkeypatch,
):
    camera, runtime, spec = capture_runtime
    repository = CaptureRepository(tmp_path / "captures")
    real_begin = PersistentCommitJournal._begin
    deadline_box = {}

    def begin_then_expire(journal, token, intent):
        real_begin(journal, token, intent)
        if journal is not repository._journal:
            return
        deadline = deadline_box["deadline"]
        while time.monotonic() <= deadline:
            time.sleep(0.001)

    monkeypatch.setattr(PersistentCommitJournal, "_begin", begin_then_expire)
    base = compile_pipeline(spec)

    def finalize(context, result):
        assert context.deadline is not None
        deadline_box["deadline"] = context.deadline
        return context.commit_checkpoint(repository.checkpoint_commit(context, result))

    plan = replace(
        base,
        name="expire after capture intent before manifest publish",
        finalize=finalize,
        timeout_seconds=0.25,
    )
    thread, failures = _deliver_when_armed(camera, _images())
    try:
        handle = runtime.controller.start(plan)
        with pytest.raises(RunFailed, match="monotonic deadline"):
            handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)

    assert repository._coordinator._authorities == {}
    assert repository._journal.pending() == ()
    assert set(repository._journal._intents) == repository._journal._aborted
    assert len(repository._journal._intents) == 1
    blob_root = tmp_path / "captures" / "content" / "blobs"
    assert any(path.is_file() for path in blob_root.rglob("*"))
    manifest_root = tmp_path / "captures" / "content" / "manifests" / "capture"
    assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()
