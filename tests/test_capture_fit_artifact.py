"""Formal capture-fit persistence without a generic analysis framework."""

from __future__ import annotations

from dataclasses import replace
import pickle
import threading

import numpy as np
import pytest
import zlc_neutral_atom.artifacts.capture_fit as capture_fit_module

from tests.test_capture_artifact_repository import (
    _deliver_when_armed,
    _runtime_and_spec,
)
from zlc_data import (
    BlockId,
    SCAN_POINT,
    encode_fit_result_batch,
    fit_spec_for,
)
from zlc_neutral_atom.artifacts import (
    AdmittedCaptureFitResult,
    CaptureFitResultRepository,
    CaptureFitResultArtifactRef,
    CaptureFrameSource,
    CaptureRepository,
    FitExecution,
    compile_capture_artifact_pipeline,
)
from zlc_storage import (
    ContentStoreAuthority,
    ContentCorruptionError,
    RepositoryRootBusy,
    content_ref_from_tree,
    decode,
    encode,
)


_CURVE = (1200, 890, 670, 520, 415, 350)


def _commit_capture(
    repository: CaptureRepository,
    *,
    block_id: str,
    curve: tuple[int, ...] = _CURVE,
):
    camera, runtime, spec = _runtime_and_spec(point_size=len(curve))
    spec = replace(
        spec,
        materializer=replace(spec.materializer, block_id=BlockId(block_id)),
    )
    plan = compile_capture_artifact_pipeline(spec, repository)
    thread, failures = _deliver_when_armed(
        camera,
        tuple(np.full((6, 8), value, dtype=np.uint16) for value in curve),
    )
    try:
        reference = runtime.controller.start(plan).result(5.0)
        thread.join(2.0)
        assert not thread.is_alive()
        assert failures == []
        return reference
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        assert runtime.shutdown(timeout=2.0)


@pytest.fixture
def committed_capture(tmp_path):
    repository = CaptureRepository(tmp_path / "captures")
    try:
        reference = _commit_capture(repository, block_id="fit-source-a")
        yield repository, reference
    finally:
        repository.close()


def _execution(fit_repository, capture_repository, capture_reference):
    source = capture_repository.admit(capture_reference)
    scan_axes = tuple(
        axis.axis_id
        for axis in source.artifact.frame_source.schema.point_axes
        if axis.role == SCAN_POINT
    )
    assert len(scan_axes) == 1
    spec = fit_spec_for(
        source.artifact.frame_source.schema,
        "exponential_decay",
        fit_axis_ids=scan_axes,
    )
    return fit_repository.execute(source, spec)


def test_execution_save_load_is_idempotent_and_manifest_has_no_mirror_truths(
    tmp_path,
    committed_capture,
    monkeypatch,
):
    capture_repository, capture_reference = committed_capture
    fit_root = tmp_path / "fits"
    fit_repository = CaptureFitResultRepository(fit_root)
    execution = _execution(
        fit_repository,
        capture_repository,
        capture_reference,
    )

    first = execution.save()
    second = execution.save()
    assert first == second
    assert isinstance(first, CaptureFitResultArtifactRef)
    assert execution.source_capture_ref == capture_reference
    assert execution.result.batch_layout.storage_size == 6 * 8

    manifest_payload = fit_repository._store_authority.read_manifest(
        "fit-result",
        first.manifest_digest,
    )
    manifest = decode(manifest_payload)
    assert set(manifest) == {
        "schema",
        "repository_id",
        "source_capture_ref",
        "result_blob",
    }
    assert len(list((fit_root / "content" / "manifests" / "fit-result").glob("*"))) == 1
    assert not list(fit_root.rglob("*.journal"))

    source = capture_repository.admit(capture_reference).artifact.frame_source
    frame_chunk_digests = frozenset(item.digest for item in source._chunk_refs)
    owner_read_blob = ContentStoreAuthority.read_blob

    def reject_frame_chunk_read(self, reference, *args, **kwargs):
        if reference.digest in frame_chunk_digests:
            raise AssertionError("loading a fit result must not read frame chunks")
        return owner_read_blob(self, reference, *args, **kwargs)

    def fail_materialization(*_args, **_kwargs):
        raise AssertionError("loading a fit result must not materialize frames")

    monkeypatch.setattr(ContentStoreAuthority, "read_blob", reject_frame_chunk_read)
    monkeypatch.setattr(CaptureFrameSource, "materialize", fail_materialization)
    loaded = fit_repository.load(first, capture_repository)
    assert isinstance(loaded, AdmittedCaptureFitResult)
    assert loaded.reference == first
    assert loaded.source_capture_ref == capture_reference
    assert encode_fit_result_batch(loaded.result) == encode_fit_result_batch(
        execution.result
    )
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(execution)
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(loaded)

    fit_repository.close()
    assert encode_fit_result_batch(execution.result) == encode_fit_result_batch(
        loaded.result
    )


def test_save_rejects_an_unreloadable_result_before_publishing_manifest(
    tmp_path,
    committed_capture,
    monkeypatch,
):
    capture_repository, capture_reference = committed_capture
    fit_root = tmp_path / "fits"
    repository = CaptureFitResultRepository(fit_root)
    try:
        execution = _execution(repository, capture_repository, capture_reference)
        monkeypatch.setattr(capture_fit_module, "_MAX_RESULT_BLOB_BYTES", 1)
        with pytest.raises(ValueError, match="result blob exceeds"):
            execution.save()
        manifests = fit_root / "content" / "manifests" / "fit-result"
        assert not manifests.exists() or not tuple(manifests.iterdir())
    finally:
        repository.close()


def test_declared_oversized_result_is_rejected_before_blob_open(
    tmp_path,
    committed_capture,
):
    capture_repository, capture_reference = committed_capture
    repository = CaptureFitResultRepository(tmp_path / "fits")
    try:
        reference = _execution(
            repository,
            capture_repository,
            capture_reference,
        ).save()
        manifest = decode(
            repository._store_authority.read_manifest(
                "fit-result",
                reference.manifest_digest,
            )
        )
        oversized_blob = dict(manifest["result_blob"])
        oversized_blob["size"] = capture_fit_module._MAX_RESULT_BLOB_BYTES + 1
        stored = repository._store_authority.publish_manifest(
            "fit-result",
            encode({**manifest, "result_blob": oversized_blob}),
        )
        oversized_ref = CaptureFitResultArtifactRef(
            repository.repository_id,
            stored.content.digest,
        )
        with pytest.raises(ValueError, match="result blob exceeds"):
            repository.load(oversized_ref, capture_repository)
    finally:
        repository.close()


def test_raw_result_cannot_be_promoted_and_repository_root_has_one_owner(
    tmp_path,
    committed_capture,
):
    capture_repository, capture_reference = committed_capture
    root = tmp_path / "fits"
    repository = CaptureFitResultRepository(root)
    execution = _execution(repository, capture_repository, capture_reference)
    try:
        assert not hasattr(repository, "save")
        with pytest.raises(PermissionError, match="authority"):
            repository._save_execution(execution.result)  # type: ignore[arg-type]
        with pytest.raises(PermissionError, match="only be minted"):
            FitExecution(
                object(),
                repository=repository,
                source_admission=capture_repository.admit(capture_reference),
                result=execution.result,
            )
        forged_result = replace(
            execution.result,
            residual_sum_squares=np.full_like(
                execution.result.residual_sum_squares,
                12_345.0,
            ),
        )
        with pytest.raises(TypeError, match="dataclass"):
            replace(execution, _result=forged_result)
        with pytest.raises(AttributeError, match="immutable"):
            execution._result = forged_result
        reference = execution.save()
        with pytest.raises(PermissionError, match="only be minted"):
            AdmittedCaptureFitResult(
                object(),
                reference=reference,
                source_capture_ref=capture_reference,
                result=forged_result,
            )
        with pytest.raises(AttributeError, match="immutable"):
            repository.repository_id = "forged"
        foreign = CaptureFitResultRepository(tmp_path / "foreign-fits")
        try:
            with pytest.raises(PermissionError, match="authority"):
                foreign._save_execution(execution)
        finally:
            foreign.close()
        with pytest.raises(RepositoryRootBusy):
            CaptureFitResultRepository(root)
    finally:
        repository.close()


def test_capture_fit_repository_is_final_and_detects_root_authority_drift(
    tmp_path,
):
    with pytest.raises(TypeError, match="final"):
        class _DerivedCaptureFitRepository(CaptureFitResultRepository):
            pass

    repository = CaptureFitResultRepository(tmp_path / "fits")
    original_root = repository.root
    try:
        object.__setattr__(repository, "root", tmp_path / "other")
        with pytest.raises(RuntimeError, match="authority changed"):
            repository._require_integrity()
    finally:
        object.__setattr__(repository, "root", original_root)
        repository.close()


def test_save_and_close_share_one_repository_lifecycle_gate(
    monkeypatch,
    tmp_path,
    committed_capture,
):
    capture_repository, capture_reference = committed_capture
    root = tmp_path / "fits"
    repository = CaptureFitResultRepository(root)
    execution = _execution(repository, capture_repository, capture_reference)
    encode_entered = threading.Event()
    allow_encode = threading.Event()
    close_returned = threading.Event()
    references = []
    failures = []
    owner_encode = capture_fit_module.encode_fit_result_batch

    def blocking_encode(result):
        encode_entered.set()
        if not allow_encode.wait(2.0):
            raise TimeoutError("test did not release fit encoding")
        return owner_encode(result)

    def save():
        try:
            references.append(execution.save())
        except BaseException as error:
            failures.append(error)

    def close():
        try:
            repository.close()
        except BaseException as error:
            failures.append(error)
        finally:
            close_returned.set()

    monkeypatch.setattr(
        capture_fit_module,
        "encode_fit_result_batch",
        blocking_encode,
    )
    save_thread = threading.Thread(target=save)
    close_thread = threading.Thread(target=close)
    save_thread.start()
    assert encode_entered.wait(2.0)
    close_thread.start()
    assert not close_returned.wait(0.05)
    allow_encode.set()
    save_thread.join(2.0)
    close_thread.join(2.0)

    assert not save_thread.is_alive()
    assert not close_thread.is_alive()
    assert failures == []
    assert len(references) == 1
    CaptureFitResultRepository(root).close()


def test_load_rejects_foreign_fit_repository_and_wrong_capture_source(
    tmp_path,
    committed_capture,
):
    capture_repository, first_capture = committed_capture
    repository = CaptureFitResultRepository(tmp_path / "fits-a")
    foreign = CaptureFitResultRepository(
        tmp_path / "fits-b",
        repository_id="foreign-capture-fit",
    )
    try:
        reference = _execution(
            repository,
            capture_repository,
            first_capture,
        ).save()
        with pytest.raises(ValueError, match="another repository"):
            foreign.load(reference, capture_repository)

        second_capture = _commit_capture(
            capture_repository,
            block_id="fit-source-a",
            curve=tuple(reversed(_CURVE)),
        )
        first_admission = capture_repository.admit(first_capture)
        second_admission = capture_repository.admit(second_capture)
        first_source = first_admission.artifact.frame_source.ref(
            first_admission.artifact.provenance.generation
        )
        second_source = second_admission.artifact.frame_source.ref(
            second_admission.artifact.provenance.generation
        )
        assert first_source.block_id == second_source.block_id
        assert first_source.schema_fingerprint == second_source.schema_fingerprint
        assert first_source.revision == second_source.revision
        assert first_source.stream_generation != second_source.stream_generation
        manifest = decode(
            repository._store_authority.read_manifest(
                "fit-result",
                reference.manifest_digest,
            )
        )
        second_tree = dict(manifest["source_capture_ref"])
        second_tree["manifest_digest"] = second_capture.manifest_digest
        forged_payload = encode({**manifest, "source_capture_ref": second_tree})
        stored = repository._store_authority.publish_manifest(
            "fit-result",
            forged_payload,
        )
        wrong_source_ref = CaptureFitResultArtifactRef(
            repository.repository_id,
            stored.content.digest,
        )
        with pytest.raises(ValueError, match="source reference"):
            repository.load(wrong_source_ref, capture_repository)
    finally:
        foreign.close()
        repository.close()


@pytest.mark.parametrize("target", ("blob", "manifest"))
def test_load_fails_closed_on_content_corruption(
    target,
    tmp_path,
    committed_capture,
):
    capture_repository, capture_reference = committed_capture
    repository = CaptureFitResultRepository(tmp_path / "fits")
    try:
        reference = _execution(
            repository,
            capture_repository,
            capture_reference,
        ).save()
        if target == "manifest":
            path = repository._store._manifest_path(
                "fit-result",
                reference.manifest_digest,
            )
        else:
            manifest = decode(
                repository._store_authority.read_manifest(
                    "fit-result",
                    reference.manifest_digest,
                )
            )
            result_ref = content_ref_from_tree(manifest["result_blob"])
            path = repository._store._blob_path(result_ref.digest)
        path.write_bytes(b"corrupt")
        with pytest.raises(ContentCorruptionError):
            repository.load(reference, capture_repository)
    finally:
        repository.close()
