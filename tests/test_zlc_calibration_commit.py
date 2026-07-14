"""User-visible durability contract for readout calibration artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import (
    VirtualCamera,
    VirtualSequencer,
    VirtualTrapArray,
)
from Zou_lab_control.neutral_atom.ports import PortCatalog, PortSpec
from zlc_data import (
    READOUT_EVENT, REPEAT, SCAN_POINT, AxisId, AxisSpec, BlockId, PointLayout,
)
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.artifacts.capture import CaptureRepository, compile_capture_artifact_pipeline
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.readout.analysis import (
    CalibrationAnalysisRequest, CalibrationAnalysisResult, CalibrationComputation,
    analyze_calibration,
    estimate_calibration_analysis_peak_bytes,
)
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode, BoxReducer, ReadoutModelKind, ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_codec import (
    calibration_report_blob_refs,
    decode_calibration_artifact,
    decode_calibration_report,
    decode_calibration_report_arrays,
    encode_calibration_artifact,
    encode_calibration_reference_average,
    encode_calibration_reference_average_validity,
    encode_calibration_report_metadata,
)
from zlc_neutral_atom.readout.calibration_reference import (
    CALIBRATION_ARTIFACT_NAMESPACE, CalibrationArtifactRef,
)
from zlc_neutral_atom.readout.calibration_repository import (
    CALIBRATION_MANIFEST_FORMAT, CalibrationRepository,
    compile_calibration_artifact_plan,
)
import zlc_neutral_atom.readout.calibration_repository as repository_impl
import zlc_neutral_atom.readout.analysis as analysis_impl
from zlc_neutral_atom.readout.contracts import CalibrationCaptureLayout
from zlc_neutral_atom.runtime import (
    CancelOutcome, DatasetCellAddress, DatasetMaterializerSpec,
    MemoryQuarantineJournal, MinimalPipelineSpec, PipelineMemoryProfile,
    ResourceArbiter, RunCancelled, RunController, RunFailed,
)
from zlc_neutral_atom.timing.capture import TriggeredCaptureSpec
from zlc_neutral_atom.timing.capture_plan import compile_capture_cell_plan
from zlc_neutral_atom.timing.pulse import FinitePulseExecutionRequest
from zlc_pulse import (
    FIELD_DURATION,
    FrozenScanTable,
    PulseExecutionForm,
    PulseFieldRef,
    RepeatRegion,
    ScanParameter,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_storage import (
    ContentRef, ContentSizeLimitError, ContentStoreAuthority, content_ref_to_tree,
    decode, encode, sha256_digest,
)
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


_CENTERS = ((7, 7), (24, 7), (7, 24), (24, 24))
_SPOT = np.array(
    ((0.42, 0.60, 0.42), (0.60, 1.00, 0.60), (0.42, 0.60, 0.42)),
    dtype=np.float64,
)
_ROOT = Path(__file__).parents[1]


def _pulse_catalog(document) -> PortCatalog:
    return PortCatalog(
        document.target.raw_lanes,
        tuple(
            PortSpec(
                port.key,
                port.kind,
                port.lanes,
                port.label,
                port.bus_index,
                port.width,
                port.encoding,
                port.safe_value,
                port.latch_clock,
            )
            for port in document.target.ports
        ),
    )


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _request() -> CalibrationAnalysisRequest:
    return CalibrationAnalysisRequest(
        layout=CalibrationCaptureLayout(AxisId("readout-event"), (0, 2), 1),
        grid_shape_yx=(2, 2),
        box_radius=1,
        box_reducer=BoxReducer.SUM,
        psf_half_width=1,
        psf_background=BackgroundMode.NONE,
        psf_background_padding=2,
        model_kinds=(
            ReadoutModelKind.BOX,
            ReadoutModelKind.PER_SITE_PSF,
            ReadoutModelKind.UNIFORM_PSF,
        ),
        default_model_kind=ReadoutModelKind.BOX,
        train_fraction=0.5,
        split_seed=7,
        histogram_bins=32,
        max_drop=2,
        detector_min_distance=8,
        detector_threshold_rel=0.2,
        detector_refine_half=1,
        expected_centers_xy=np.asarray(_CENTERS, dtype="<f8"),
        maximum_site_residual_px=2.0,
    )


def _frame(repeat: int, event: int, context: int) -> np.ndarray:
    image = np.zeros((32, 32), dtype=np.uint16)
    for site, (x, y) in enumerate(_CENTERS):
        occupied = (repeat + context + site) % 2 == 0
        level = (2200.0 if occupied else 180.0) if event in (0, 2) else (
            1050.0 if occupied else 90.0
        )
        image[y - 1 : y + 2, x - 1 : x + 2] = np.rint(
            level * _SPOT
        ).astype(np.uint16)
    return image


def _deliver_when_armed(camera: VirtualCamera, images: list[np.ndarray]):
    failures: list[BaseException] = []

    def deliver() -> None:
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
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=deliver, daemon=False)
    thread.start()
    return thread, failures


@dataclass
class _CaptureFixture:
    root: Path
    repository_id: str
    repository: CaptureRepository
    reference: CaptureArtifactRef
    request: CalibrationAnalysisRequest
    result: CalibrationAnalysisResult


@pytest.fixture(scope="module")
def capture_fixture(tmp_path_factory) -> _CaptureFixture:
    root = tmp_path_factory.mktemp("compact-calibration-capture")
    document = load_pulse_document(_ROOT / "pulses" / "imaging_template.json")
    document = replace(
        document,
        repeat=RepeatRegion(
            document.periods[0].period_id,
            document.periods[-1].period_id,
            12,
        ),
        scan_parameters=(
            ScanParameter(
                "fixture_point_duration",
                PulseFieldRef(
                    FIELD_DURATION,
                    document.periods[0].period_id,
                    None,
                ),
                "fixture point duration",
                "s",
            ),
        ),
        scan_table=FrozenScanTable(
            ("fixture_point_duration",),
            ((document.periods[0].duration,),) * 2,
        ),
    )
    sequencer = VirtualSequencer(
        sleep_scale=0,
        port_catalog=_pulse_catalog(document),
    )
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(32, 32), seed=11),
        exposure=1e-3,
        capture_trigger_channels=("ch11",),
    )
    camera.recent_capacity = 128
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"readout": camera, "sequencer": sequencer},
            {
                "readout": {"type": "VirtualCamera", "params": {}},
                "sequencer": {"type": "VirtualSequencer", "params": {}},
            },
        )
    )
    description = runtime.describe_camera("readout")
    repeat_axis = _axis("repeat", REPEAT, 12)
    event_axis = _axis("readout-event", READOUT_EVENT, 3)
    context_axis = _axis("context", SCAN_POINT, 2)
    layout = PointLayout.rect_c((3, 2))
    cells = tuple(
        DatasetCellAddress(repeat, layout.storage_index((event, context)))
        for context in range(context_axis.size)
        for repeat in range(repeat_axis.size)
        for event in range(event_axis.size)
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
    capture = MinimalPipelineSpec(
        "compact calibration source",
        measurement,
        DatasetMaterializerSpec(
            BlockId("compact-calibration-source"),
            PipelineMemoryProfile(96 << 20),
        ),
    )
    pulse_artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    cell_plan = compile_capture_cell_plan(
        pulse_artifact,
        "ch11",
        measurement.capture_contract.dataset_schema,
        readout_event_axis_id=event_axis.axis_id,
        scan_point_layout=PointLayout.rect_c((context_axis.size,)),
        within_point_grouping=tuple(
            (repeat, event)
            for repeat in range(repeat_axis.size)
            for event in range(event_axis.size)
        ),
    )
    spec = TriggeredCaptureSpec(
        capture,
        runtime.bind_sequencer_port(),
        FinitePulseExecutionRequest(document, pulse_artifact),
        "ch11",
        cell_plan,
    )
    images = [
        _frame(cell.repeat_index, *layout.multi_index(cell.point_storage_index))
        for cell in cells
    ]
    repository_id = "calibration-test-captures"
    repository = CaptureRepository(root, repository_id=repository_id)
    thread, failures = _deliver_when_armed(camera, images)
    try:
        reference = runtime.controller.start(
            compile_capture_artifact_pipeline(spec, repository)
        ).result(20.0)
        thread.join(5.0)
        assert not thread.is_alive() and failures == []
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        assert runtime.shutdown(timeout=3.0)
    request = _request()
    result = analyze_calibration(repository.admit(reference), request)
    fixture = _CaptureFixture(
        root,
        repository_id,
        repository,
        reference,
        request,
        result,
    )
    try:
        yield fixture
    finally:
        fixture.repository.close()


def _controller() -> RunController:
    return RunController(ResourceArbiter(MemoryQuarantineJournal()))


def _repository(tmp_path: Path, name: str) -> CalibrationRepository:
    return CalibrationRepository(tmp_path / name, repository_id=f"{name}-calibrations")


def _plan(fixture: _CaptureFixture, repository: CalibrationRepository):
    return compile_calibration_artifact_plan(
        fixture.reference,
        fixture.repository,
        repository,
        fixture.request,
        memory_limit_bytes=512 << 20,
    )


def _encoded_record(
    repository_id: str,
    result: CalibrationAnalysisResult,
) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    artifact_payload = encode_calibration_artifact(result.artifact)
    average_payload = encode_calibration_reference_average(
        result.report.reference_average
    )
    validity_payload = encode_calibration_reference_average_validity(
        result.report.reference_average_validity
    )
    artifact_ref = ContentRef(sha256_digest(artifact_payload), len(artifact_payload))
    average_ref = ContentRef(sha256_digest(average_payload), len(average_payload))
    validity_ref = ContentRef(sha256_digest(validity_payload), len(validity_payload))
    report_payload = encode_calibration_report_metadata(
        result.report,
        reference_average_blob=average_ref,
        reference_average_validity_blob=validity_ref,
    )
    report_ref = ContentRef(sha256_digest(report_payload), len(report_payload))
    manifest = encode(
        {
            "format": CALIBRATION_MANIFEST_FORMAT,
            "repository_id": repository_id,
            "artifact_blob": content_ref_to_tree(artifact_ref),
            "report_blob": content_ref_to_tree(report_ref),
        }
    )
    return artifact_payload, average_payload, validity_payload, report_payload, manifest


def _expected_reference(
    repository_id: str,
    result: CalibrationAnalysisResult,
) -> CalibrationArtifactRef:
    return CalibrationArtifactRef(
        repository_id,
        sha256_digest(_encoded_record(repository_id, result)[-1]),
    )


def _commit(
    fixture: _CaptureFixture,
    repository: CalibrationRepository,
):
    handle = _controller().start(_plan(fixture, repository))
    return handle, handle.result(30.0)


def test_real_analysis_artifact_and_report_codec_roundtrip(capture_fixture):
    result = capture_fixture.result
    artifact_payload = encode_calibration_artifact(result.artifact)
    average_payload = encode_calibration_reference_average(
        result.report.reference_average
    )
    validity_payload = encode_calibration_reference_average_validity(
        result.report.reference_average_validity
    )
    average_ref = ContentRef(sha256_digest(average_payload), len(average_payload))
    validity_ref = ContentRef(sha256_digest(validity_payload), len(validity_payload))
    report_payload = encode_calibration_report_metadata(
        result.report,
        reference_average_blob=average_ref,
        reference_average_validity_blob=validity_ref,
    )
    artifact = decode_calibration_artifact(artifact_payload)
    average, validity = decode_calibration_report_arrays(
        average_payload,
        validity_payload,
        image_shape=result.artifact.frame_contract.frame_schema.data_shape,
    )
    report = decode_calibration_report(
        report_payload,
        reference_average=average,
        reference_average_validity=validity,
    )
    assert encode_calibration_artifact(artifact) == artifact_payload
    assert encode_calibration_report_metadata(
        report,
        reference_average_blob=average_ref,
        reference_average_validity_blob=validity_ref,
    ) == report_payload
    assert calibration_report_blob_refs(report_payload) == (average_ref, validity_ref)
    assert artifact.source_binding.source_capture_ref == capture_fixture.reference
    assert tuple(model.kind for model in artifact.models) == (
        ReadoutModelKind.BOX,
        ReadoutModelKind.PER_SITE_PSF,
        ReadoutModelKind.UNIFORM_PSF,
    )
    assert report.reference_box_signals.shape == (24, 2, 4)


def test_runcontroller_commit_load_report_and_admit(capture_fixture, tmp_path):
    repository = _repository(tmp_path, "committed")
    handle, reference = _commit(capture_fixture, repository)
    assert handle.snapshot().final_committed
    artifact = repository.load(reference)
    report = repository.load_report(reference)
    admitted = repository.admit(reference, capture_fixture.repository)
    assert isinstance(admitted, ResolvedCalibration)
    assert admitted.reference == reference and admitted.artifact is not None
    with pytest.raises(TypeError, match="cannot be assembled by callers"):
        ResolvedCalibration(
            CalibrationArtifactRef("another-calibration-repository", "f" * 64),
            admitted.artifact,
        )
    assert artifact.source_binding.source_capture_ref == capture_fixture.reference
    assert report.request == capture_fixture.request
    try:
        assert repository.load(reference).source_binding == artifact.source_binding
        assert repository.load_report(reference).request == report.request
    finally:
        repository.close()


def test_runtime_load_and_admit_never_read_report_blobs(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "artifact-only-load")
    _handle, reference = _commit(capture_fixture, repository)
    _repository_id, artifact_ref, report_ref = repository_impl._decode_manifest(
        repository._read_manifest(reference)
    )
    report_payload = repository._store_authority.read_blob(report_ref)
    average_ref, validity_ref = calibration_report_blob_refs(report_payload)
    forbidden = {report_ref, average_ref, validity_ref}
    real_read_blob = ContentStoreAuthority.read_blob
    real_read_manifest = ContentStoreAuthority.read_manifest
    blob_limits, manifest_limits = [], []

    def artifact_only(self, content, *, max_bytes=None):
        if content in forbidden:
            raise AssertionError("runtime artifact path read calibration diagnostics")
        blob_limits.append((content, max_bytes))
        return real_read_blob(self, content, max_bytes=max_bytes)

    def bounded_manifest(self, namespace, digest, *, max_bytes=None):
        manifest_limits.append((namespace, max_bytes))
        return real_read_manifest(self, namespace, digest, max_bytes=max_bytes)

    monkeypatch.setattr(ContentStoreAuthority, "read_blob", artifact_only)
    monkeypatch.setattr(ContentStoreAuthority, "read_manifest", bounded_manifest)
    try:
        assert repository.load(reference).source_binding.source_capture_ref == (
            capture_fixture.reference
        )
        assert repository.admit(reference, capture_fixture.repository).reference == reference
        with pytest.raises(AssertionError, match="diagnostics"):
            repository.load_report(reference)
        calibration_limits = [
            limit
            for namespace, limit in manifest_limits
            if namespace == CALIBRATION_ARTIFACT_NAMESPACE
        ]
        assert calibration_limits and set(calibration_limits) == {
            repository_impl._MAX_MANIFEST_BYTES
        }
        artifact_limits = [limit for content, limit in blob_limits if content == artifact_ref]
        assert artifact_limits and set(artifact_limits) == {repository_impl._MAX_ARTIFACT_BYTES}
    finally:
        repository.close()


def test_report_blob_io_does_not_block_runtime_admission(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "report-io-lock-scope")
    _handle, reference = _commit(capture_fixture, repository)
    _repository_id, _artifact_ref, report_ref = repository_impl._decode_manifest(
        repository._read_manifest(reference)
    )
    average_ref, _validity_ref = calibration_report_blob_refs(
        repository._store_authority.read_blob(report_ref)
    )
    report_reading, release_report, admit_done = (
        threading.Event(), threading.Event(), threading.Event()
    )
    report_errors, admit_errors = [], []
    real_read_blob = ContentStoreAuthority.read_blob

    def block_average(self, content, *, max_bytes=None):
        if content == average_ref:
            report_reading.set()
            if not release_report.wait(5.0):
                raise TimeoutError("test did not release report blob read")
        return real_read_blob(self, content, max_bytes=max_bytes)

    def load_report():
        try:
            repository.load_report(reference)
        except BaseException as exc:
            report_errors.append(exc)

    def admit_runtime():
        try:
            repository.admit(reference, capture_fixture.repository)
        except BaseException as exc:
            admit_errors.append(exc)
        finally:
            admit_done.set()

    monkeypatch.setattr(ContentStoreAuthority, "read_blob", block_average)
    report_thread = threading.Thread(target=load_report)
    admit_thread = threading.Thread(target=admit_runtime)
    admission_was_independent = False
    try:
        report_thread.start()
        assert report_reading.wait(2.0)
        admit_thread.start()
        admission_was_independent = admit_done.wait(2.0)
    finally:
        release_report.set()
        report_thread.join(5.0)
        admit_thread.join(5.0)
        repository.close()
    assert admission_was_independent
    assert not report_thread.is_alive() and not admit_thread.is_alive()
    assert report_errors == [] and admit_errors == []


def test_report_metadata_refs_raw_images_without_canonical_base64(
    capture_fixture,
    tmp_path,
):
    repository = _repository(tmp_path, "binary-diagnostics")
    _handle, reference = _commit(capture_fixture, repository)
    try:
        _repository_id, _artifact_ref, report_ref = repository_impl._decode_manifest(
            repository._read_manifest(reference)
        )
        report_payload = repository._store_authority.read_blob(report_ref)
        tree = decode(report_payload)
        average_ref, validity_ref = calibration_report_blob_refs(report_payload)
        assert "reference_average" not in tree
        assert "reference_average_validity" not in tree
        assert average_ref.size == capture_fixture.result.report.reference_average.nbytes
        assert validity_ref.size == (
            capture_fixture.result.report.reference_average_validity.nbytes
        )
        assert repository._store_authority.read_blob(average_ref) == (
            encode_calibration_reference_average(
                capture_fixture.result.report.reference_average
            )
        )
    finally:
        repository.close()


def test_report_limits_do_not_block_artifact_runtime_load(capture_fixture, tmp_path):
    root = tmp_path / "diagnostic-limits"
    repository_id = "diagnostic-limit-calibrations"
    repository = CalibrationRepository(root, repository_id=repository_id)
    _handle, reference = _commit(capture_fixture, repository)
    repository.close()

    memory_limited = CalibrationRepository(
        root,
        repository_id=repository_id,
        diagnostic_memory_limit_bytes=1,
    )
    try:
        assert memory_limited.load(reference).source_binding.source_capture_ref == (
            capture_fixture.reference
        )
        with pytest.raises(MemoryError, match="diagnostics require"):
            memory_limited.load_report(reference)
    finally:
        memory_limited.close()

    metadata_limited = CalibrationRepository(
        root,
        repository_id=repository_id,
        max_report_metadata_bytes=1,
    )
    try:
        assert metadata_limited.load(reference).source_binding.source_capture_ref == (
            capture_fixture.reference
        )
        with pytest.raises(ContentSizeLimitError):
            metadata_limited.load_report(reference)
    finally:
        metadata_limited.close()


def test_stage_rejects_diagnostic_memory_before_copying_or_writing_images(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = CalibrationRepository(
        tmp_path / "stage-diagnostic-preflight",
        repository_id="stage-diagnostic-preflight",
        diagnostic_memory_limit_bytes=1,
    )

    def forbidden_image_copy(_value):
        raise AssertionError("large diagnostic image was encoded before admission")

    monkeypatch.setattr(
        repository_impl,
        "encode_calibration_reference_average",
        forbidden_image_copy,
    )
    monkeypatch.setattr(
        repository_impl,
        "encode_calibration_reference_average_validity",
        forbidden_image_copy,
    )
    try:
        with pytest.raises(MemoryError, match="diagnostics require"):
            repository._stage_result(capture_fixture.result)
        blob_root = repository.root / "content" / "blobs"
        assert not blob_root.exists() or not any(
            path.is_file() for path in blob_root.rglob("*")
        )
    finally:
        repository.close()


def test_stage_rejects_payloads_its_own_loaders_cannot_read(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "self-readable-stage")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(repository_impl, "_MAX_ARTIFACT_BYTES", 1)
            with pytest.raises(MemoryError, match="calibration artifact requires"):
                repository._stage_result(capture_fixture.result)

        with monkeypatch.context() as patch:
            def reject_report(*_args, **_kwargs):
                raise ValueError("report decoder rejected staged bytes")

            patch.setattr(
                repository_impl,
                "decode_calibration_report",
                reject_report,
            )
            with pytest.raises(ValueError, match="decoder rejected"):
                repository._stage_result(capture_fixture.result)
    finally:
        repository.close()


def test_committed_artifact_remains_loadable_if_diagnostic_blob_is_lost(
    capture_fixture,
    tmp_path,
):
    repository = _repository(tmp_path, "lost-diagnostic")
    _handle, reference = _commit(capture_fixture, repository)
    _repository_id, _artifact_ref, report_ref = repository_impl._decode_manifest(
        repository._read_manifest(reference)
    )
    average_ref, _validity_ref = calibration_report_blob_refs(
        repository._store_authority.read_blob(report_ref)
    )
    repository._store._blob_path_for_digest(average_ref.digest).unlink()
    try:
        assert repository.load(reference).source_binding.source_capture_ref == (
            capture_fixture.reference
        )
        assert repository.admit(reference, capture_fixture.repository).reference == reference
        with pytest.raises(FileNotFoundError):
            repository.load_report(reference)
    finally:
        repository.close()


def test_cancel_and_execute_failure_never_publish(capture_fixture, tmp_path, monkeypatch):
    real_analyze = repository_impl.analyze_calibration

    cancel_repository = _repository(tmp_path, "cancelled")
    started = threading.Event()
    release = threading.Event()
    cancelled_result: list[CalibrationAnalysisResult] = []

    def paused(capture, request):
        result = real_analyze(capture, request)
        cancelled_result.append(result)
        started.set()
        assert release.wait(10.0)
        return result

    monkeypatch.setattr(repository_impl, "analyze_calibration", paused)
    handle = _controller().start(_plan(capture_fixture, cancel_repository))
    assert started.wait(20.0)
    assert handle.cancel("test cancellation") is CancelOutcome.REQUESTED
    release.set()
    with pytest.raises(RunCancelled):
        handle.result(20.0)
    cancelled_ref = _expected_reference(
        cancel_repository.repository_id,
        cancelled_result[0],
    )
    assert not handle.snapshot().final_committed
    assert not cancel_repository.has(cancelled_ref)
    cancel_repository.close()

    failed_repository = _repository(tmp_path, "failed")
    failed_result: list[CalibrationAnalysisResult] = []

    def failed(capture, request):
        result = real_analyze(capture, request)
        failed_result.append(result)
        raise RuntimeError("injected analysis failure")

    monkeypatch.setattr(repository_impl, "analyze_calibration", failed)
    failed_handle = _controller().start(_plan(capture_fixture, failed_repository))
    with pytest.raises(RunFailed, match="injected analysis failure"):
        failed_handle.result(20.0)
    failed_ref = _expected_reference(
        failed_repository.repository_id,
        failed_result[0],
    )
    assert not failed_handle.snapshot().final_committed
    assert not failed_repository.has(failed_ref)
    failed_repository.close()


def test_analysis_memory_budget_rejects_before_scientific_execute(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "analysis-memory-limit")
    source = capture_fixture.repository.admit(capture_fixture.reference).artifact.frame_source
    required = estimate_calibration_analysis_peak_bytes(
        source.schema,
        capture_fixture.request,
        source_read_scratch_bytes=source.max_read_scratch_bytes,
    )
    called = False

    def forbidden_analysis(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("analysis ran after memory admission failed")

    monkeypatch.setattr(repository_impl, "analyze_calibration", forbidden_analysis)
    plan = compile_calibration_artifact_plan(
        capture_fixture.reference,
        capture_fixture.repository,
        repository,
        capture_fixture.request,
        memory_limit_bytes=required - 1,
    )
    handle = _controller().start(plan)
    try:
        with pytest.raises(RunFailed) as failure:
            handle.result(20.0)
        assert failure.value.primary is not None
        assert failure.value.primary.original_type == "MemoryError"
        assert not called
        assert repository.startup_reconciliations == ()
    finally:
        repository.close()


def test_physical_manifest_without_final_commit_is_not_public(
    capture_fixture,
    tmp_path,
):
    repository = _repository(tmp_path, "ghost")
    result = capture_fixture.result
    (
        artifact_payload,
        average_payload,
        validity_payload,
        report_payload,
        _predicted_manifest,
    ) = _encoded_record(
        repository.repository_id,
        result,
    )
    artifact_ref = repository._store_authority.put_blob(artifact_payload)
    repository._store_authority.put_blob(average_payload)
    repository._store_authority.put_blob(validity_payload)
    report_ref = repository._store_authority.put_blob(report_payload)
    # Rebuild from the exact CAS-owned references; their values match the
    # predicted references above, but the storage owner remains authoritative.
    payload = repository_impl._manifest_payload(
        repository.repository_id,
        artifact_ref,
        report_ref,
    )
    digest = sha256_digest(payload)
    repository._store_authority.publish_manifest(
        CALIBRATION_ARTIFACT_NAMESPACE,
        payload,
        expected_digest=digest,
    )
    reference = CalibrationArtifactRef(repository.repository_id, digest)
    try:
        assert not repository.has(reference)
        with pytest.raises(PermissionError, match="FINAL commit"):
            repository.load(reference)
    finally:
        repository.close()


def test_unknown_manifest_format_is_rejected_after_final_commit(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "unknown-format")

    def unknown_manifest(repository_id, artifact_blob, report_blob):
        return encode(
            {
                "format": "unknown-calibration-manifest",
                "repository_id": repository_id,
                "artifact_blob": content_ref_to_tree(artifact_blob),
                "report_blob": content_ref_to_tree(report_blob),
            }
        )

    monkeypatch.setattr(repository_impl, "_manifest_payload", unknown_manifest)
    _handle, reference = _commit(capture_fixture, repository)
    try:
        assert repository.has(reference)
        with pytest.raises(ValueError, match="expected format"):
            repository.load(reference)
    finally:
        repository.close()


def test_source_capture_frame_contract_is_rechecked_before_publication(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "source-mismatch")
    original = capture_fixture.result
    contract = original.artifact.frame_contract
    mismatched_contract = replace(
        contract,
        exposure_seconds=contract.exposure_seconds * 2.0,
    )
    mismatched_artifact = replace(
        original.artifact,
        frame_contract=mismatched_contract,
        readout_physical_context=replace(
            original.artifact.readout_physical_context,
            integration_seconds=mismatched_contract.exposure_seconds,
        ),
    )
    mismatched = CalibrationComputation(
        mismatched_artifact,
        original.report,
    )
    monkeypatch.setattr(
        analysis_impl,
        "compute_calibration",
        lambda _capture, _request: mismatched,
    )
    handle = _controller().start(_plan(capture_fixture, repository))
    with pytest.raises(RunFailed) as failure:
        handle.result(20.0)
    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "ValueError"
    assert "FrameContract" in str(failure.value.primary)
    assert not repository.has(_expected_reference(repository.repository_id, mismatched))
    repository.close()


def test_source_capture_pulse_context_is_rederived_before_publication(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "pulse-context-mismatch")
    original = capture_fixture.result
    context = original.artifact.readout_physical_context
    mismatched_artifact = replace(
        original.artifact,
        readout_physical_context=replace(
            context,
            integration_start_offset_seconds=(
                context.integration_start_offset_seconds + 1e-9
            ),
        ),
    )
    mismatched = CalibrationComputation(mismatched_artifact, original.report)
    monkeypatch.setattr(
        analysis_impl,
        "compute_calibration",
        lambda _capture, _request: mismatched,
    )

    handle = _controller().start(_plan(capture_fixture, repository))
    with pytest.raises(RunFailed) as failure:
        handle.result(20.0)

    assert failure.value.primary is not None
    assert failure.value.primary.original_type == "ValueError"
    assert "readout physical context differs" in str(failure.value.primary)
    assert not repository.has(_expected_reference(repository.repository_id, mismatched))
    repository.close()


def test_pure_calibration_computation_cannot_enter_final_commit(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "pure-computation")
    pure = CalibrationComputation(
        capture_fixture.result.artifact,
        capture_fixture.result.report,
    )
    monkeypatch.setattr(
        repository_impl,
        "analyze_calibration",
        lambda _source, _request: pure,
    )

    handle = _controller().start(_plan(capture_fixture, repository))
    try:
        with pytest.raises(RunFailed) as failure:
            handle.result(20.0)
        assert failure.value.primary is not None
        assert failure.value.primary.original_type == "TypeError"
        assert "CalibrationAnalysisResult" in str(failure.value.primary)
        manifest_root = repository.root / "content" / "manifests" / "calibration"
        assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()
    finally:
        repository.close()


def test_group_contexts_are_rejoined_to_the_admitted_capture_before_publish(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    repository = _repository(tmp_path, "context-mismatch")
    original = capture_fixture.result
    mismatched_report = replace(
        original.report,
        group_contexts=tuple(reversed(original.report.group_contexts)),
    )
    mismatched = CalibrationComputation(
        original.artifact,
        mismatched_report,
    )
    monkeypatch.setattr(
        analysis_impl,
        "compute_calibration",
        lambda _capture, _request: mismatched,
    )
    handle = _controller().start(_plan(capture_fixture, repository))
    try:
        with pytest.raises(RunFailed) as failure:
            handle.result(20.0)
        assert failure.value.primary is not None
        assert failure.value.primary.original_type == "ValueError"
        assert "group contexts" in str(failure.value.primary)
        assert not handle.snapshot().final_committed
    finally:
        repository.close()


def _crash_after_calibration_manifest_publish(
    capture_root: str,
    capture_repository_id: str,
    capture_digest: str,
    calibration_root: str,
    calibration_repository_id: str,
    digest_sidecar: str,
) -> None:
    capture_repository = CaptureRepository(
        capture_root,
        repository_id=capture_repository_id,
    )
    calibration_repository = CalibrationRepository(
        calibration_root,
        repository_id=calibration_repository_id,
    )
    original_publish = ContentStoreAuthority.publish_manifest

    def crash_publish(self, namespace, payload, *, expected_digest=None):
        if namespace != CALIBRATION_ARTIFACT_NAMESPACE:
            return original_publish(
                self,
                namespace,
                payload,
                expected_digest=expected_digest,
            )
        Path(digest_sidecar).write_text(sha256_digest(payload), encoding="ascii")
        original_publish(
            self,
            namespace,
            payload,
            expected_digest=expected_digest,
        )
        os._exit(97)

    ContentStoreAuthority.publish_manifest = crash_publish
    reference = CaptureArtifactRef(capture_repository_id, capture_digest)
    _controller().start(
        compile_calibration_artifact_plan(
            reference,
            capture_repository,
            calibration_repository,
            _request(),
            memory_limit_bytes=512 << 20,
        )
    ).result(30.0)
    os._exit(98)


def test_restart_recovers_manifest_visible_before_commit_ack(
    capture_fixture,
    tmp_path,
    monkeypatch,
):
    calibration_root = tmp_path / "crash-recovery"
    calibration_repository_id = "recovered-calibrations"
    sidecar = tmp_path / "published-digest.txt"
    capture_fixture.repository.close()
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_after_calibration_manifest_publish,
        args=(
            str(capture_fixture.root),
            capture_fixture.repository_id,
            capture_fixture.reference.manifest_digest,
            str(calibration_root),
            calibration_repository_id,
            str(sidecar),
        ),
    )
    try:
        process.start()
        process.join(60.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            pytest.fail("crash-recovery child did not terminate")
        assert process.exitcode == 97
    finally:
        capture_fixture.repository = CaptureRepository(
            capture_fixture.root,
            repository_id=capture_fixture.repository_id,
        )
    reference = CalibrationArtifactRef(
        calibration_repository_id,
        sidecar.read_text(encoding="ascii"),
    )
    verified: list[ContentRef] = []
    real_verify_blob = ContentStoreAuthority.verify_blob

    def observe_verify(self, content, *, max_bytes=None):
        verified.append(content)
        return real_verify_blob(self, content, max_bytes=max_bytes)

    monkeypatch.setattr(ContentStoreAuthority, "verify_blob", observe_verify)
    recovered = CalibrationRepository(
        calibration_root,
        repository_id=calibration_repository_id,
    )
    try:
        reconciliations = recovered.startup_reconciliations
        assert len(reconciliations) == 1
        assert reconciliations[0].recovery.committed
        assert sorted(item.size for item in verified) == [32 * 32, 32 * 32 * 8]
        assert recovered.load(reference).source_binding.source_capture_ref == (
            capture_fixture.reference
        )
        assert recovered.admit(reference, capture_fixture.repository).reference == reference
    finally:
        recovered.close()
