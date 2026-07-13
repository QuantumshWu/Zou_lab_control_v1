"""Trusted committed-capture to admitted-calibration artifact spine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import pickle
import shutil
import threading
import time

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    PointLayout,
)
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.artifacts import (
    AdmittedCapture,
    CaptureArtifactRef,
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.readout.analysis import (
    BoxAnalysisConfig,
    CalibrationAnalysisRequest,
    PsfAnalysisConfig,
)
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    ReadoutModelKind,
)
from zlc_neutral_atom.readout.contracts import CalibrationCaptureLayout
import zlc_neutral_atom.readout.calibration_repository as repository_impl
import zlc_storage.content_store as content_store_impl
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.calibration_repository import (
    AdmittedCalibration,
    CALIBRATION_MANIFEST_SCHEMA,
    CalibrationRepository,
    compile_calibration_artifact_plan,
)
from zlc_neutral_atom.runtime import (
    CancelOutcome,
    DatasetCellAddress,
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    RunCancelled,
    RunFailed,
)
from zlc_neutral_atom.runtime.commit import (
    CommitIntent,
    CommitKind,
    CommitTarget,
    PersistentCommitJournal,
)
from zlc_storage import (
    ContentAddressedStore,
    ContentRef,
    FramedJournal,
    RepositoryRootBusy,
    decode,
    encode,
    sha256_digest,
)
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


_CENTERS = ((7, 7), (24, 7), (7, 24), (24, 24))
_SPOT = np.array(
    ((0.42, 0.60, 0.42), (0.60, 1.00, 0.60), (0.42, 0.60, 0.42)),
    dtype=np.float64,
)


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _request(**changes) -> CalibrationAnalysisRequest:
    base = CalibrationAnalysisRequest(
        CalibrationCaptureLayout(AxisId("readout-event"), (0, 2), 1),
        (2, 2),
        box=BoxAnalysisConfig(1, BoxReducer.SUM),
        model_kinds=(
            ReadoutModelKind.UNIFORM_PSF,
            ReadoutModelKind.BOX,
            ReadoutModelKind.PER_SITE_PSF,
        ),
        default_model_kind=ReadoutModelKind.BOX,
        psf=PsfAnalysisConfig(1, BackgroundMode.NONE, 0),
        train_fraction=0.6,
        random_seed=3817,
        minimum_train_samples_per_class=1,
        minimum_test_samples_per_class=1,
        minimum_held_out_class_accuracy_lower_bound=0.0,
    )
    return replace(base, **changes)


def _frame(repeat: int, event: int, context: int) -> np.ndarray:
    image = np.zeros((32, 32), dtype=np.uint16)
    for site, (x, y) in enumerate(_CENTERS):
        occupied = (repeat + context + site) % 2 == 0
        if event in (0, 2):
            level = 2000.0 if occupied else 200.0
        else:
            level = 1000.0 if occupied else 100.0
        image[y - 1 : y + 2, x - 1 : x + 2] = np.rint(level * _SPOT).astype(
            np.uint16
        )
    return image


def _deliver_when_armed(camera: VirtualCamera, images: list[np.ndarray]):
    failures: list[BaseException] = []

    def source() -> None:
        try:
            deadline = time.monotonic() + 5.0
            state = camera._recent_state()
            with state["cond"]:
                while not state["armed"]:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("camera was not armed")
                    state["cond"].wait(remaining)
            camera._deliver(images)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=source, daemon=False)
    thread.start()
    return thread, failures


@dataclass
class _CommittedFixture:
    root: object
    runtime: LegacyNeutralAtomRuntime
    capture_repository: CaptureRepository
    capture_ref: CaptureArtifactRef
    request: CalibrationAnalysisRequest
    calibration_repository: CalibrationRepository
    calibration_ref: CalibrationArtifactRef


@pytest.fixture(scope="module")
def committed(tmp_path_factory):
    root = tmp_path_factory.mktemp("trusted-calibration")
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(32, 32), seed=11),
        exposure=1e-3,
    )
    # This fixture deliberately injects the entire finite hardware burst at
    # once.  Its concrete driver-retention proof must therefore cover that
    # burst; production adapters derive the same bound from hardware facts.
    camera.recent_capacity = 64
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"readout": camera},
            {"readout": {"type": "VirtualCamera", "params": {}}},
        )
    )
    description = runtime.describe_camera("readout")
    repeat_axis = _axis("repeat", REPEAT, 10)
    event_axis = _axis("readout-event", READOUT_EVENT, 3)
    context_axis = _axis("context", SCAN_POINT, 2)
    layout = PointLayout.rect_c((3, 2))
    cells = tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(repeat_axis.size)
        for point in range(layout.storage_size)
    )
    measurement = runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            repeat_axis,
            (event_axis, context_axis),
            layout,
            cells,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            len(cells),
            64 << 20,
            tuple(description.event_setting(index) for index in range(3)),
        )
    )
    spec = MinimalPipelineSpec(
        "trusted calibration raw capture",
        measurement,
        DatasetMaterializerSpec(
            BlockId("trusted-calibration-source"),
            PipelineMemoryProfile.for_current_runtime(128 << 20),
        ),
    )
    images = []
    for cell in cells:
        event, context = layout.multi_index(cell.point_storage_index)
        images.append(_frame(cell.repeat_index, event, context))
    capture_repository = CaptureRepository(root / "captures", repository_id="captures")
    source_thread, source_failures = _deliver_when_armed(camera, images)
    try:
        capture_handle = runtime.controller.start(
            compile_capture_artifact_pipeline(spec, capture_repository)
        )
        capture_ref = capture_handle.result(15.0)
        source_thread.join(5.0)
        assert not source_thread.is_alive()
        assert source_failures == []
        request = _request()
        calibration_repository = CalibrationRepository(
            root / "calibrations",
            repository_id="calibrations",
        )
        calibration_handle = runtime.controller.start(
            compile_calibration_artifact_plan(
                capture_ref,
                capture_repository,
                calibration_repository,
                request,
            )
        )
        calibration_ref = calibration_handle.result(20.0)
        assert calibration_handle.snapshot().final_committed
        yield _CommittedFixture(
            root,
            runtime,
            capture_repository,
            capture_ref,
            request,
            calibration_repository,
            calibration_ref,
        )
    finally:
        if source_thread.is_alive():
            camera.finish_record_capture()
            source_thread.join(2.0)
        shutdown_ok = runtime.shutdown(timeout=3.0)
        if "calibration_repository" in locals():
            calibration_repository.close()
        capture_repository.close()
        assert shutdown_ok


def _manifest_root(repository: CalibrationRepository):
    return repository.root / "content" / "manifests" / "calibration"


def _assert_no_manifest(repository: CalibrationRepository) -> None:
    root = _manifest_root(repository)
    assert not root.exists() or tuple(root.iterdir()) == ()


def _copy_calibration_repository(
    source: CalibrationRepository,
    destination,
    *,
    include_journal: bool,
) -> None:
    """Copy durable evidence without copying a live root's locked lease file."""

    shutil.copytree(source.root / "content", destination / "content")
    if include_journal:
        shutil.copy2(
            source.root / "calibration-commit.journal",
            destination / "calibration-commit.journal",
        )


def _append_pending_intent(root, intent: CommitIntent) -> None:
    """Inject the exact crash-state record below the sealed journal API."""

    FramedJournal(root / "calibration-commit.journal").append(
        f"intent:{intent.commit_id}",
        {
            "kind": "INTENT",
            "operation_kind": intent.kind.value,
            "commit_id": intent.commit_id,
            "run_id": intent.run_id,
            "safety_bundle_id": intent.safety_bundle_id,
            "target": {
                "repository_id": intent.target.repository_id,
                "artifact_kind": intent.target.artifact_kind,
                "schema_version": intent.target.schema_version,
                "target_ref": intent.target.target_ref,
                "expected_manifest_digest": (
                    intent.target.expected_manifest_digest
                ),
            },
            "created_at": intent.created_at,
        },
    )


def test_flat_run_commits_reopens_and_admits_only_after_exact_source_reload(
    committed,
    tmp_path,
):
    repository = committed.calibration_repository
    reference = committed.calibration_ref
    artifact = repository.load(reference)
    analysis_result = repository.load_analysis_result(reference)
    assert analysis_result.artifact.fingerprint == artifact.fingerprint
    assert analysis_result.diagnostics.bracket_count == (
        artifact.source_binding.bracket_count
    )
    assert artifact.source_binding.source_capture_ref == committed.capture_ref
    assert not isinstance(artifact, AdmittedCalibration)

    admitted = repository.admit(reference, committed.capture_repository)
    assert admitted.reference == reference
    assert admitted.artifact.fingerprint == artifact.fingerprint
    assert admitted.artifact_fingerprint == artifact.fingerprint
    assert admitted.source_capture_ref == committed.capture_ref
    assert admitted.commit_kind is CommitKind.FINAL
    assert admitted.commit_id.startswith("calibration-final-")
    assert len(admitted.evidence_digest) == 64
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(admitted)
    with pytest.raises(PermissionError, match="only be minted"):
        AdmittedCalibration(
            None,
            repository_token=None,
            reference=reference,
            artifact=artifact,
            commit_kind=CommitKind.FINAL,
            commit_id=admitted.commit_id,
            evidence_digest=admitted.evidence_digest,
        )
    with pytest.raises(AttributeError, match="immutable"):
        admitted._commit_id = "forged"
    with pytest.raises(TypeError, match="final"):
        class _ForgedAdmission(AdmittedCalibration):
            pass

    forged = object.__new__(AdmittedCalibration)
    for slot in AdmittedCalibration.__slots__:
        if slot == "__weakref__":
            continue
        object.__setattr__(forged, slot, object.__getattribute__(admitted, slot))
    object.__setattr__(
        forged,
        "_reference",
        CalibrationArtifactRef(reference.repository_id, "0" * 64),
    )
    with pytest.raises(PermissionError, match="authority is invalid"):
        forged.artifact

    copied_root = tmp_path / "reopened-calibrations"
    _copy_calibration_repository(repository, copied_root, include_journal=True)
    reopened = CalibrationRepository(
        copied_root,
        repository_id=repository.repository_id,
    )
    try:
        assert reopened.startup_reconciliations == ()
        assert reopened.load(reference).fingerprint == artifact.fingerprint
        reopened_admission = reopened.admit(
            reference,
            committed.capture_repository,
        )
        assert reopened_admission.reference == reference
        assert reopened_admission.commit_id == admitted.commit_id
        assert reopened_admission.evidence_digest == admitted.evidence_digest
    finally:
        reopened.close()


def test_visible_calibration_without_final_intent_is_inspectable_not_admitted(
    committed,
    tmp_path,
):
    copied_root = tmp_path / "visible-uncommitted-calibration"
    _copy_calibration_repository(
        committed.calibration_repository,
        copied_root,
        include_journal=False,
    )
    repository = CalibrationRepository(
        copied_root,
        repository_id=committed.calibration_repository.repository_id,
    )
    try:
        assert (
            repository.load(committed.calibration_ref).fingerprint
            == committed.calibration_repository.load(
                committed.calibration_ref
            ).fingerprint
        )
        with pytest.raises(PermissionError, match="no committed FINAL authority"):
            repository.admit(
                committed.calibration_ref,
                committed.capture_repository,
            )
        assert not hasattr(repository, "put")
    finally:
        repository.close()


def test_repository_root_has_one_live_owner_and_close_is_terminal(tmp_path):
    root = tmp_path / "exclusive-calibrations"
    repository = CalibrationRepository(root, repository_id="exclusive-calibrations")
    with pytest.raises(RepositoryRootBusy, match="live owner"):
        CalibrationRepository(root, repository_id="exclusive-calibrations")
    repository.close()
    with pytest.raises(RuntimeError, match="lease owner is closed"):
        repository.has(CalibrationArtifactRef(repository.repository_id, "0" * 64))

    reopened = CalibrationRepository(root, repository_id="exclusive-calibrations")
    reopened.close()


def test_preflight_loaded_capture_is_the_same_object_analyzed_and_request_is_snapshotted(
    committed,
    monkeypatch,
):
    capture_repository = committed.capture_repository
    calibration_repository = CalibrationRepository(
        committed.root / "same-object-calibrations",
        repository_id="same-object-calibrations",
    )
    request = replace(committed.request, random_seed=4812)
    frozen_fingerprint = request.fingerprint
    plan = compile_calibration_artifact_plan(
        committed.capture_ref,
        capture_repository,
        calibration_repository,
        request,
    )
    object.__setattr__(request, "random_seed", 9999)
    assert request.fingerprint != frozen_fingerprint

    admitted_artifact_ids: list[int] = []
    analyzed_ids: list[int] = []
    analyzed_request_fingerprints: list[str] = []
    real_admit = CaptureRepository.admit
    real_analyze = repository_impl.analyze_calibration

    def tracked_admit(self, reference):
        value = real_admit(self, reference)
        if self is capture_repository:
            admitted_artifact_ids.append(id(value.artifact))
        return value

    def tracked_analyze(capture, frozen_request):
        analyzed_ids.append(id(capture))
        analyzed_request_fingerprints.append(frozen_request.fingerprint)
        return real_analyze(capture, frozen_request)

    monkeypatch.setattr(CaptureRepository, "admit", tracked_admit)
    monkeypatch.setattr(repository_impl, "analyze_calibration", tracked_analyze)
    reference = committed.runtime.controller.start(plan).result(20.0)
    assert calibration_repository.has(reference)
    assert admitted_artifact_ids == analyzed_ids
    assert analyzed_request_fingerprints == [frozen_fingerprint]


def test_cancel_and_analysis_failure_publish_no_calibration_manifest(
    committed,
    monkeypatch,
):
    cancelled_repository = CalibrationRepository(
        committed.root / "cancelled-calibrations",
        repository_id="cancelled-calibrations",
    )
    entered = threading.Event()
    release = threading.Event()
    real_analyze = repository_impl.analyze_calibration

    def paused(capture, request):
        entered.set()
        assert release.wait(10.0)
        return real_analyze(capture, request)

    monkeypatch.setattr(repository_impl, "analyze_calibration", paused)
    handle = committed.runtime.controller.start(
        compile_calibration_artifact_plan(
            committed.capture_ref,
            committed.capture_repository,
            cancelled_repository,
            committed.request,
        )
    )
    assert entered.wait(5.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    release.set()
    with pytest.raises(RunCancelled):
        handle.result(20.0)
    _assert_no_manifest(cancelled_repository)

    monkeypatch.setattr(repository_impl, "analyze_calibration", real_analyze)
    failed_repository = CalibrationRepository(
        committed.root / "failed-calibrations",
        repository_id="failed-calibrations",
    )
    impossible = replace(committed.request, grid_shape_yx=(3, 3))
    with pytest.raises(RunFailed):
        committed.runtime.controller.start(
            compile_calibration_artifact_plan(
                committed.capture_ref,
                committed.capture_repository,
                failed_repository,
                impossible,
            )
    ).result(20.0)
    _assert_no_manifest(failed_repository)
    assert committed.capture_repository.load(committed.capture_ref).ref == (
        committed.capture_ref
    )


def test_manifest_and_derivation_tampering_are_rejected_current_only(committed):
    repository = committed.calibration_repository
    manifest = decode(
        repository._store.read_manifest(
            "calibration",
            committed.calibration_ref.manifest_digest,
        )
    )

    forged_manifest = decode(encode(manifest))
    forged_manifest["resource_summary"]["site_count"] += 1
    stored = repository._store.publish_manifest("calibration", encode(forged_manifest))
    with pytest.raises(ValueError, match="summary differs"):
        repository.load(
            CalibrationArtifactRef(repository.repository_id, stored.content.digest)
        )

    forged_manifest = decode(encode(manifest))
    forged_manifest["request_fingerprint"] = "0" * 64
    stored = repository._store.publish_manifest("calibration", encode(forged_manifest))
    with pytest.raises(ValueError, match="request_fingerprint differs"):
        repository.load(CalibrationArtifactRef(repository.repository_id, stored.content.digest))

    forged_manifest = decode(encode(manifest))
    forged_manifest["source_capture_evidence_digest"] = "0" * 64
    stored = repository._store.publish_manifest("calibration", encode(forged_manifest))
    with pytest.raises(ValueError, match="source_capture_evidence_digest differs"):
        repository.load(
            CalibrationArtifactRef(repository.repository_id, stored.content.digest)
        )

    derivation_ref = ContentRef(
        manifest["derivation_blob"]["digest"],
        manifest["derivation_blob"]["size"],
    )
    derivation = decode(repository._store.read_blob(derivation_ref))
    derivation["analysis_run_id"] = "forged-analysis-run"
    forged_derivation = repository._store.put_blob(encode(derivation))
    derivation_only = decode(encode(manifest))
    derivation_only["derivation_blob"] = {
        "digest": forged_derivation.digest,
        "size": forged_derivation.size,
    }
    derivation_only["evidence_digest"] = forged_derivation.digest
    stored = repository._store.publish_manifest("calibration", encode(derivation_only))
    with pytest.raises(ValueError, match="analysis_run_id differs"):
        repository.load(CalibrationArtifactRef(repository.repository_id, stored.content.digest))

    artifact_ref = ContentRef(
        manifest["artifact_blob"]["digest"],
        manifest["artifact_blob"]["size"],
    )
    legacy = {
        "schema": "zlc_neutral_atom.calibration-manifest.v1",
        "repository_id": repository.repository_id,
        "artifact_schema": manifest["artifact_schema"],
        "artifact_blob": {"digest": artifact_ref.digest, "size": artifact_ref.size},
        "artifact_fingerprint": manifest["artifact_fingerprint"],
        "resource_summary": manifest["resource_summary"],
    }
    stored = repository._store.publish_manifest("calibration", encode(legacy))
    with pytest.raises(ValueError, match="unknown field set|unsupported"):
        repository.load(CalibrationArtifactRef(repository.repository_id, stored.content.digest))


def test_admission_requires_the_concrete_exact_source_repository(
    committed,
    tmp_path,
    monkeypatch,
):
    with pytest.raises(TypeError, match="final"):
        class LyingCaptureRepository(CaptureRepository):
            pass

    wrong = CaptureRepository(tmp_path / "wrong-captures", repository_id="wrong-captures")
    with pytest.raises(ValueError, match="another repository"):
        committed.calibration_repository.admit(committed.calibration_ref, wrong)
    with pytest.raises(TypeError, match="CaptureRepository"):
        committed.calibration_repository.admit(
            committed.calibration_ref,
            lambda _reference: committed.capture_repository.load(committed.capture_ref),
        )
    real_build = repository_impl.build_calibration_work_plan
    capture = committed.capture_repository.load(committed.capture_ref)
    expected_work_plan = real_build(capture.block.schema, committed.request)
    drifted_work_plan = replace(
        expected_work_plan,
        source_cell_count=expected_work_plan.source_cell_count + 1,
    )
    monkeypatch.setattr(
        repository_impl,
        "build_calibration_work_plan",
        lambda *_args, **_kwargs: drifted_work_plan,
    )
    with pytest.raises(ValueError, match="work plan differs"):
        committed.calibration_repository.admit(
            committed.calibration_ref,
            committed.capture_repository,
        )

    monkeypatch.setattr(repository_impl, "build_calibration_work_plan", real_build)
    uncommitted_root = tmp_path / "visible-uncommitted-captures"
    shutil.copytree(
        committed.capture_repository.root / "content",
        uncommitted_root / "content",
    )
    uncommitted = CaptureRepository(
        uncommitted_root,
        repository_id=committed.capture_repository.repository_id,
    )
    assert uncommitted.load(committed.capture_ref).ref == committed.capture_ref
    with pytest.raises(PermissionError, match="no committed journal authority"):
        uncommitted.admit(committed.capture_ref)
    with pytest.raises(PermissionError, match="no committed journal authority"):
        committed.calibration_repository.admit(
            committed.calibration_ref,
            uncommitted,
        )

    target = CalibrationRepository(
        tmp_path / "laundering-target-calibrations",
        repository_id="laundering-target-calibrations",
    )
    with pytest.raises(RunFailed, match="no committed journal authority"):
        committed.runtime.controller.start(
            compile_calibration_artifact_plan(
                committed.capture_ref,
                uncommitted,
                target,
                committed.request,
            )
        ).result(10.0)
    _assert_no_manifest(target)


def test_admission_rechecks_capture_commit_evidence(committed, monkeypatch):
    real_admit = CaptureRepository.admit

    def drifted_admit(self, reference):
        admission = real_admit(self, reference)
        if self is not committed.capture_repository:
            return admission
        forged = object.__new__(AdmittedCapture)
        for slot in AdmittedCapture.__slots__:
            object.__setattr__(forged, slot, getattr(admission, slot))
        object.__setattr__(forged, "_evidence_digest", "0" * 64)
        return forged

    monkeypatch.setattr(CaptureRepository, "admit", drifted_admit)
    with pytest.raises(ValueError, match="admission evidence differs"):
        committed.calibration_repository.admit(
            committed.calibration_ref,
            committed.capture_repository,
        )


def test_restart_reconciles_visible_manifest_without_rerunning_analysis(
    committed,
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "restart-visible-calibrations"
    _copy_calibration_repository(
        committed.calibration_repository,
        root,
        include_journal=False,
    )
    repository = CalibrationRepository(
        root,
        repository_id=committed.calibration_repository.repository_id,
    )
    loaded = repository._load_record(committed.calibration_ref)
    target = CommitTarget(
        repository.repository_id,
        "calibration",
        CALIBRATION_MANIFEST_SCHEMA,
        committed.calibration_ref.target_ref,
        committed.calibration_ref.manifest_digest,
    )
    intent = CommitIntent(
        CommitKind.FINAL,
        (
            f"calibration-final-{loaded.derivation.analysis_run_id}-"
            f"{committed.calibration_ref.manifest_digest}"
        ),
        loaded.derivation.analysis_run_id,
        loaded.derivation.analysis_safety_bundle_id,
        target,
        time.time(),
    )
    with pytest.raises(ValueError, match="not a CalibrationArtifact target"):
        repository._recover(
            replace(
                intent,
                kind=CommitKind.CHECKPOINT,
                commit_id="checkpoint-is-not-calibration-final",
            )
        )
    repository.close()
    _append_pending_intent(root, intent)
    monkeypatch.setattr(
        repository_impl,
        "analyze_calibration",
        lambda *_args, **_kwargs: pytest.fail("recovery reran calibration analysis"),
    )
    reopened = CalibrationRepository(
        root,
        repository_id=repository.repository_id,
    )
    assert len(reopened.startup_reconciliations) == 1
    reconciliation = reopened.startup_reconciliations[0]
    assert reconciliation.intent == intent
    assert reconciliation.recovery.committed
    assert reconciliation.recovery.result.result == committed.calibration_ref


def test_online_commit_recovers_plain_error_after_manifest_became_visible(
    committed,
    monkeypatch,
):
    repository = CalibrationRepository(
        committed.root / "lost-ack-calibrations",
        repository_id="lost-ack-calibrations",
    )
    real_publish = ContentAddressedStore._publish_manifest
    publish_calls = 0

    def publish_then_lose_ack(self, *args, **kwargs):
        nonlocal publish_calls
        stored = real_publish(self, *args, **kwargs)
        if self is not repository._store:
            return stored
        publish_calls += 1
        raise OSError("directory fsync acknowledgement was lost")

    monkeypatch.setattr(
        ContentAddressedStore,
        "_publish_manifest",
        publish_then_lose_ack,
    )
    handle = committed.runtime.controller.start(
        compile_calibration_artifact_plan(
            committed.capture_ref,
            committed.capture_repository,
            repository,
            committed.request,
        )
    )
    reference = handle.result(20.0)
    assert publish_calls == 1
    assert handle.snapshot().final_committed
    assert "publication acknowledgement" in (
        handle.snapshot().commit_recovery_warning or ""
    )
    assert repository.load(reference).fingerprint
    assert repository._journal.pending() == ()


def test_close_refuses_inflight_commit_without_poisoning_owner(
    committed,
    monkeypatch,
):
    repository = CalibrationRepository(
        committed.root / "close-inflight-calibrations",
        repository_id="close-inflight-calibrations",
    )
    entered_publish = threading.Event()
    release_publish = threading.Event()
    real_publish = ContentAddressedStore._publish_manifest

    def paused_publish(store, *args, **kwargs):
        if store is repository._store:
            entered_publish.set()
            if not release_publish.wait(10.0):
                raise TimeoutError("test did not release calibration publication")
        return real_publish(store, *args, **kwargs)

    monkeypatch.setattr(
        ContentAddressedStore,
        "_publish_manifest",
        paused_publish,
    )
    handle = committed.runtime.controller.start(
        compile_calibration_artifact_plan(
            committed.capture_ref,
            committed.capture_repository,
            repository,
            committed.request,
        )
    )
    try:
        assert entered_publish.wait(5.0)
        with pytest.raises(RuntimeError, match="outstanding commit authorities"):
            repository.close()
        assert repository._root_lease.active
        assert repository.startup_reconciliations == ()
        release_publish.set()
        reference = handle.result(20.0)
        assert repository.load(reference).fingerprint
    finally:
        release_publish.set()
    repository.close()
    assert not repository._root_lease.active


def test_finalize_exception_abandons_prepared_calibration_authority(
    committed,
    monkeypatch,
):
    repository = CalibrationRepository(
        committed.root / "abandoned-calibration-authority",
        repository_id="abandoned-calibration-authority",
    )
    real_final_commit = CalibrationRepository.final_commit

    def mint_then_fail(self, context, executed):
        operation = real_final_commit(self, context, executed)
        if self is repository:
            assert operation.commit_id.startswith("calibration-final-")
            raise RuntimeError("analysis failed after preparing calibration commit")
        return operation

    monkeypatch.setattr(
        CalibrationRepository,
        "final_commit",
        mint_then_fail,
    )
    handle = committed.runtime.controller.start(
        compile_calibration_artifact_plan(
            committed.capture_ref,
            committed.capture_repository,
            repository,
            committed.request,
        )
    )
    with pytest.raises(RunFailed, match="failed after preparing calibration"):
        handle.result(20.0)
    assert repository._coordinator._authorities == {}
    repository.close()

    reopened = CalibrationRepository(
        repository.root,
        repository_id=repository.repository_id,
    )
    reopened.close()


def test_startup_recovery_bounds_visible_manifest_before_decoding(tmp_path):
    root = tmp_path / "oversized-recovery"
    repository = CalibrationRepository(root, repository_id="bounded-recovery")
    payload = b"x" * (repository.resource_policy.max_manifest_bytes + 1)
    digest = sha256_digest(payload)
    path = repository._store._manifest_path("calibration", digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    oversized_ref = CalibrationArtifactRef(repository.repository_id, digest)
    with pytest.raises(CalibrationResourceExceeded, match="manifest exceeds"):
        repository.has(oversized_ref)
    target = CommitTarget(
        repository.repository_id,
        "calibration",
        CALIBRATION_MANIFEST_SCHEMA,
        f"calibration/{digest}",
        digest,
    )
    intent = CommitIntent(
        CommitKind.FINAL,
        f"calibration-final-oversized-recovery-run-{digest}",
        "oversized-recovery-run",
        None,
        target,
        time.time(),
    )
    repository.close()
    _append_pending_intent(root, intent)
    with pytest.raises(CalibrationResourceExceeded, match="manifest exceeds"):
        CalibrationRepository(root, repository_id=repository.repository_id)


def test_recovery_requires_existing_target_directory_durability(
    committed,
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "recovery-directory-durability"
    _copy_calibration_repository(
        committed.calibration_repository,
        root,
        include_journal=False,
    )
    repository = CalibrationRepository(
        root,
        repository_id=committed.calibration_repository.repository_id,
    )
    loaded = repository._load_record(committed.calibration_ref)
    intent = CommitIntent(
        CommitKind.FINAL,
        (
            f"calibration-final-{loaded.derivation.analysis_run_id}-"
            f"{committed.calibration_ref.manifest_digest}"
        ),
        loaded.derivation.analysis_run_id,
        loaded.derivation.analysis_safety_bundle_id,
        CommitTarget(
            repository.repository_id,
            "calibration",
            CALIBRATION_MANIFEST_SCHEMA,
            committed.calibration_ref.target_ref,
            committed.calibration_ref.manifest_digest,
        ),
        time.time(),
    )
    repository.close()
    _append_pending_intent(root, intent)

    manifest_directory = (
        root / "content" / "manifests" / "calibration"
    ).resolve()
    real_flush = content_store_impl.durability.flush_directory
    failed_flushes = 0

    def lose_manifest_directory_flush(directory):
        nonlocal failed_flushes
        if Path(directory).resolve() == manifest_directory:
            failed_flushes += 1
            raise OSError("manifest directory flush acknowledgement lost")
        return real_flush(directory)

    monkeypatch.setattr(
        content_store_impl.durability,
        "flush_directory",
        lose_manifest_directory_flush,
    )
    with pytest.raises(OSError, match="flush acknowledgement lost"):
        CalibrationRepository(root, repository_id=repository.repository_id)
    assert failed_flushes == 1
    assert intent in PersistentCommitJournal(
        root / "calibration-commit.journal",
        repository.repository_id,
    ).pending()

    monkeypatch.setattr(
        content_store_impl.durability,
        "flush_directory",
        real_flush,
    )
    reopened = CalibrationRepository(
        root,
        repository_id=repository.repository_id,
    )
    try:
        assert len(reopened.startup_reconciliations) == 1
        assert reopened.startup_reconciliations[0].recovery.committed
        assert reopened._journal.pending() == ()
    finally:
        reopened.close()


def test_recovery_never_recreates_target_removed_after_validation(
    committed,
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "recovery-target-disappears"
    _copy_calibration_repository(
        committed.calibration_repository,
        root,
        include_journal=False,
    )
    repository = CalibrationRepository(
        root,
        repository_id=committed.calibration_repository.repository_id,
    )
    loaded = repository._load_record(committed.calibration_ref)
    intent = CommitIntent(
        CommitKind.FINAL,
        (
            f"calibration-final-{loaded.derivation.analysis_run_id}-"
            f"{committed.calibration_ref.manifest_digest}"
        ),
        loaded.derivation.analysis_run_id,
        loaded.derivation.analysis_safety_bundle_id,
        CommitTarget(
            repository.repository_id,
            "calibration",
            CALIBRATION_MANIFEST_SCHEMA,
            committed.calibration_ref.target_ref,
            committed.calibration_ref.manifest_digest,
        ),
        time.time(),
    )
    manifest_path = repository._store._manifest_path(
        "calibration",
        committed.calibration_ref.manifest_digest,
    )
    repository.close()
    _append_pending_intent(root, intent)

    real_confirm = ContentAddressedStore._confirm_manifest_durable

    def remove_before_confirmation(store, namespace, digest, **kwargs):
        if store.root == (root / "content").resolve():
            ContentAddressedStore._manifest_path(
                store,
                namespace,
                digest,
            ).unlink()
        return real_confirm(store, namespace, digest, **kwargs)

    monkeypatch.setattr(
        ContentAddressedStore,
        "_confirm_manifest_durable",
        remove_before_confirmation,
    )
    with pytest.raises(FileNotFoundError):
        CalibrationRepository(root, repository_id=repository.repository_id)
    assert not manifest_path.exists()
    assert intent in PersistentCommitJournal(
        root / "calibration-commit.journal",
        repository.repository_id,
    ).pending()


def test_visible_manifest_with_missing_blob_remains_pending(committed, tmp_path):
    root = tmp_path / "missing-blob-recovery"
    _copy_calibration_repository(
        committed.calibration_repository,
        root,
        include_journal=False,
    )
    repository = CalibrationRepository(
        root,
        repository_id=committed.calibration_repository.repository_id,
    )
    manifest = decode(
        repository._store.read_manifest(
            "calibration",
            committed.calibration_ref.manifest_digest,
        )
    )
    derivation_ref = ContentRef(
        manifest["derivation_blob"]["digest"],
        manifest["derivation_blob"]["size"],
    )
    repository._store._blob_path(derivation_ref.digest).unlink()
    target = CommitTarget(
        repository.repository_id,
        "calibration",
        CALIBRATION_MANIFEST_SCHEMA,
        committed.calibration_ref.target_ref,
        committed.calibration_ref.manifest_digest,
    )
    intent = CommitIntent(
        CommitKind.FINAL,
        (
            f"calibration-final-{manifest['analysis_run_id']}-"
            f"{committed.calibration_ref.manifest_digest}"
        ),
        manifest["analysis_run_id"],
        manifest["analysis_safety_bundle_id"],
        target,
        time.time(),
    )
    repository.close()
    _append_pending_intent(root, intent)
    with pytest.raises(FileNotFoundError):
        CalibrationRepository(root, repository_id=repository.repository_id)
    replayed = PersistentCommitJournal(
        root / "calibration-commit.journal",
        repository.repository_id,
    )
    assert intent in replayed.pending()


def test_repository_snapshots_policy_and_rejects_nested_authority_drift(tmp_path):
    caller_policy = CalibrationResourcePolicy(max_manifest_bytes=96 * 1024)
    repository = CalibrationRepository(
        tmp_path / "policy-authority",
        repository_id="policy-authority",
        resource_policy=caller_policy,
    )
    object.__setattr__(caller_policy, "max_manifest_bytes", 1)
    assert repository.resource_policy.max_manifest_bytes == 96 * 1024

    object.__setattr__(repository.resource_policy, "max_manifest_bytes", 1)
    with pytest.raises(RuntimeError, match="durability authority changed"):
        repository.has(
            CalibrationArtifactRef(repository.repository_id, "0" * 64)
        )

    clean = CalibrationRepository(
        tmp_path / "store-shadow-authority",
        repository_id="store-shadow-authority",
    )
    with pytest.raises(AttributeError, match="immutable"):
        clean._store.read_manifest = lambda *_args, **_kwargs: b"forged"


def test_repository_identity_drift_fails_before_analysis_or_publication(committed):
    repository = CalibrationRepository(
        committed.root / "identity-drift-calibrations",
        repository_id="identity-drift-calibrations",
    )
    plan = compile_calibration_artifact_plan(
        committed.capture_ref,
        committed.capture_repository,
        repository,
        committed.request,
    )
    original = committed.capture_repository.repository_id
    object.__setattr__(
        committed.capture_repository,
        "repository_id",
        "drifted-captures",
    )
    try:
        with pytest.raises(RunFailed, match="identity changed"):
            committed.runtime.controller.start(plan).result(10.0)
    finally:
        object.__setattr__(
            committed.capture_repository,
            "repository_id",
            original,
        )
    _assert_no_manifest(repository)

    plan = compile_calibration_artifact_plan(
        committed.capture_ref,
        committed.capture_repository,
        repository,
        committed.request,
    )
    object.__setattr__(repository, "repository_id", "drifted-calibrations")
    try:
        with pytest.raises(RunFailed, match="identity changed"):
            committed.runtime.controller.start(plan).result(10.0)
    finally:
        object.__setattr__(
            repository,
            "repository_id",
            "identity-drift-calibrations",
        )
    _assert_no_manifest(repository)

    plan = compile_calibration_artifact_plan(
        committed.capture_ref,
        committed.capture_repository,
        repository,
        committed.request,
    )
    original_store = repository._store
    other = CalibrationRepository(
        committed.root / "identity-drift-other",
        repository_id=repository.repository_id,
    )
    object.__setattr__(repository, "_store", other._store)
    try:
        with pytest.raises(RunFailed, match="durability authority changed"):
            committed.runtime.controller.start(plan).result(10.0)
    finally:
        object.__setattr__(repository, "_store", original_store)
    _assert_no_manifest(repository)


def test_executed_candidate_cannot_be_committed_by_same_id_other_root(
    committed,
    monkeypatch,
):
    repository_a = CalibrationRepository(
        committed.root / "authority-a",
        repository_id="shared-authority-id",
    )
    repository_b = CalibrationRepository(
        committed.root / "authority-b",
        repository_id="shared-authority-id",
    )
    plan = compile_calibration_artifact_plan(
        committed.capture_ref,
        committed.capture_repository,
        repository_a,
        committed.request,
    )
    real_final_commit = CalibrationRepository.final_commit

    def redirect(self, context, executed):
        if self is repository_a:
            return real_final_commit(repository_b, context, executed)
        return real_final_commit(self, context, executed)

    monkeypatch.setattr(CalibrationRepository, "final_commit", redirect)
    with pytest.raises(RunFailed, match="another durability authority"):
        committed.runtime.controller.start(plan).result(20.0)
    _assert_no_manifest(repository_a)
    _assert_no_manifest(repository_b)
