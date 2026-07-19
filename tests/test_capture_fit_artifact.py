"""Fit-result persistence over an admitted current raw capture."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest

from fpga.pulse_streamer.host.image import DEFAULT_CONFIG_PATH, default_clock_hz
from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    REPEAT,
    encode_fit_result_batch,
    fit_result_decode_additional_peak_upper_bound_nbytes,
    fit_result_encode_additional_peak_upper_bound_nbytes,
    fit_result_retained_upper_bound_nbytes,
    fit_result_source_validation_additional_peak_upper_bound_nbytes,
    fit_spec_for,
)
import zlc_neutral_atom.artifacts.fit_result as fit_result_module
from zlc_neutral_atom.adapter_sdk import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.artifacts import (
    AdmittedFitResult,
    CaptureFrameSource,
    CaptureRepository,
    FitExecution,
    FitResultArtifactRef,
    FitResultRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.bootstrap._camera_endpoint import CameraCaptureEndpoint
from zlc_neutral_atom.bootstrap._sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
)
from zlc_neutral_atom.bootstrap._triggered_capture import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.bootstrap._virtual_hardware import VirtualSequencer
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.runtime.ports import DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceArbiter,
    ResourceKey,
)
from zlc_neutral_atom.runtime.run import RunController
from zlc_neutral_atom.runtime.safety_journal import PersistentSafetyJournal
from zlc_neutral_atom.timing.capture import TriggeredCaptureSpec
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_pulse import PulseExecutionForm, load_deployed_pulse_target, load_pulse_document
from zlc_storage import (
    ContentCorruptionError,
    ContentStoreAuthority,
    RepositoryRootBusy,
    canonical_digest,
    content_ref_from_tree,
    decode,
)


_ROOT = Path(__file__).parents[1]


class _Camera:
    max_pending_records = 2
    timeout = 1.0

    def __init__(self) -> None:
        self.expected = 0
        self.ordinal = 0
        self.armed = False

    def capture_working_point(self) -> CameraWorkingPoint:
        return CameraWorkingPoint(
            canonical_digest({"fixture": "capture-fit-camera"}),
            "EXTERNAL_TRIGGERED",
            (3, 4),
            (3, 4),
            (0, 0),
            (3, 4),
            (1, 1),
            np.dtype("<u2"),
            "count",
            ("ch11",),
            0.001,
            0.001,
            0.0,
            1.0,
            "fixture-readout",
        )

    def arm(self, frames: int, *, max_inflight_frames: int, timeout: float) -> None:
        assert max_inflight_frames == 2 and timeout > 0
        self.expected = frames
        self.ordinal = 0
        self.armed = True

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        assert n == 1 and exact and timeout > 0 and self.armed
        ordinal = self.ordinal
        self.ordinal += 1
        curve = np.array((1200, 800, 520, 340), dtype=np.uint16)
        image = np.tile(curve + ordinal * 10, (3, 1))
        return [
            CameraFrameRecord(
                image,
                ordinal,
                self.expected,
                100 + ordinal,
                200 + ordinal,
                1,
                1_000 + ordinal,
                10_000 + ordinal,
                ordinal % 2,
            )
        ]

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        self.armed = False
        return CameraCaptureTerminalRecord(self.expected, True, True, True)

    def capture_state(self) -> tuple[bool, int]:
        return self.armed, 0

    def close(self) -> None:
        self.armed = False


def _identity(name: str) -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        name,
        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        f"{name}-evidence",
        "fixture-assets-v1",
    )


def _bind_endpoint(broker, key, identity, endpoint, cleanup_operation):
    binding = None

    def current():
        assert binding is not None
        return binding

    binding = broker.bind(
        key=ResourceKey.parse(key),
        identity=broker.verify_identity(lambda: _identity(identity)),
        execute_command=lambda command: endpoint.execute_command(current(), command),
        cleanup_operations={cleanup_operation: endpoint.cleanup},
        verify_safe_state=endpoint.verify_safe_state,
        capability_probe=lambda: endpoint.capability_probe(current()),
        close_session=lambda command: endpoint.close_session(current(), command),
        interrupt_operations={cleanup_operation: endpoint.interrupt},
    )
    return broker.verify_capability(binding)


class _CaptureCase:
    def __init__(self, tmp_path) -> None:
        self.camera = _Camera()
        self.broker = DeviceBroker()
        camera_endpoint = CameraCaptureEndpoint(
            self.camera,
            "camera",
            exact_external_trigger_qualification_digest=canonical_digest(
                {"qualification": "deterministic fixture adapter"}
            ),
        )
        camera_port = BoundCapturePort(
            _bind_endpoint(
                self.broker,
                "device/camera",
                "fixture-camera",
                camera_endpoint,
                SafetyOperation.DISARM,
            ),
            (SafetyOperation.DISARM,),
        )
        target = load_deployed_pulse_target()
        self.sequencer = VirtualSequencer(
            target,
            clock_hz=default_clock_hz(DEFAULT_CONFIG_PATH),
            sleep_scale=0,
        )
        pulse_endpoint = VirtualSequencerExecutionEndpoint(self.sequencer)
        pulse_port = BoundPulsePort(
            _bind_endpoint(
                self.broker,
                "device/sequencer",
                "fixture-sequencer",
                pulse_endpoint,
                SafetyOperation.SAFE_STATE,
            ),
            (),
        )
        repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
        binding = bind_triggered_camera_acquisition(
            pulse_port,
            camera_port,
            pulse_document=load_pulse_document(
                _ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"
            ),
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channel="ch11",
            layout=TriggeredCameraLayout(
                repeat,
                AxisId("readout-event"),
                AxisId("scan-ordinal"),
                readout_events_per_repeat=3,
            ),
            transport_memory_limit_bytes=8 << 20,
        )
        pipeline = MinimalPipelineSpec(
            "capture fit source",
            binding.measurement,
            BlockId("capture-fit-source"),
            16 << 20,
            timeout_seconds=2.0,
        )
        triggered = TriggeredCaptureSpec(
            pipeline,
            binding.pulse_port,
            binding.pulse_request,
            binding.trigger_channel,
            binding.cell_plan,
        )
        self.capture_repository = CaptureRepository(tmp_path / "captures")
        self.safety = PersistentSafetyJournal(tmp_path / "safety.zlcj")
        self.resources = ResourceArbiter(self.safety)
        self.controller = RunController(self.resources)
        self.capture_reference = self.controller.start(
            compile_capture_artifact_pipeline(
                triggered,
                self.capture_repository,
            )
        ).result(5.0)

    def close(self) -> None:
        assert self.controller.shutdown(2.0)
        self.broker.shutdown()
        self.resources.shutdown()
        self.capture_repository.close()
        self.sequencer.close()
        self.camera.close()


@pytest.fixture
def capture_case(tmp_path):
    case = _CaptureCase(tmp_path)
    try:
        yield case
    finally:
        case.close()


def _execution(repository, case):
    spec = fit_spec_for(
        case.capture_repository.inspect_final(
            case.capture_reference,
        ).dataset_schema,
        "exponential_decay",
        fit_axis_ids=(AxisId("camera.x"),),
    )
    return repository.execute_capture(
        case.capture_repository,
        case.capture_reference,
        spec,
    )


def test_execution_save_load_is_idempotent_and_has_no_mirror_truths(
    tmp_path,
    capture_case,
    monkeypatch,
) -> None:
    repository = FitResultRepository(tmp_path / "fits")
    try:
        execution = _execution(repository, capture_case)
        first = execution.save(operation_memory_limit_bytes=512 << 20)
        second = execution.save(operation_memory_limit_bytes=512 << 20)
        assert first == second
        assert isinstance(first, FitResultArtifactRef)
        assert execution.source_artifact_ref == capture_case.capture_reference
        assert execution.result.batch_layout.storage_size == 9

        manifest = decode(
            repository._store_authority.read_manifest(
                "fit-result",
                first.manifest_digest,
            )
        )
        assert set(manifest) == {
            "schema",
            "repository_id",
            "source",
            "result_blob",
        }
        assert "checkpoint" not in manifest
        assert "raw_frames" not in manifest

        frame_digests = frozenset(
            item.digest
            for item in capture_case.capture_repository.load(
                capture_case.capture_reference
            ).frame_source._chunk_refs
        )
        owner_read_blob = ContentStoreAuthority.read_blob

        def reject_frame_read(self, reference, *args, **kwargs):
            if reference.digest in frame_digests:
                raise AssertionError("fit load must not read source frame chunks")
            return owner_read_blob(self, reference, *args, **kwargs)

        monkeypatch.setattr(ContentStoreAuthority, "read_blob", reject_frame_read)
        monkeypatch.setattr(
            CaptureFrameSource,
            "materialize",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("fit load must not materialize frames")
            ),
        )
        loaded = repository.load(
            first,
            capture_repository=capture_case.capture_repository,
        )
        assert isinstance(loaded, AdmittedFitResult)
        assert loaded.source_artifact_ref == capture_case.capture_reference
        assert encode_fit_result_batch(loaded.result) == encode_fit_result_batch(
            execution.result
        )
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(execution)
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(loaded)
    finally:
        repository.close()


def test_save_additional_workspace_is_admitted_before_encode_or_cas(
    tmp_path,
    capture_case,
    monkeypatch,
) -> None:
    repository = FitResultRepository(tmp_path / "fits")
    try:
        execution = _execution(repository, capture_case)
        required = (
            fit_result_encode_additional_peak_upper_bound_nbytes(
                execution.result
            )
            + fit_result_module._FIT_SAVE_REPOSITORY_FIXED_BYTES
        )
        encode_calls = 0
        put_calls = 0
        owner_encode = fit_result_module.encode_fit_result_batch
        owner_put = ContentStoreAuthority.put_blob

        def traced_encode(result):
            nonlocal encode_calls
            encode_calls += 1
            return owner_encode(result)

        def traced_put(authority, payload):
            nonlocal put_calls
            put_calls += 1
            return owner_put(authority, payload)

        monkeypatch.setattr(
            fit_result_module,
            "encode_fit_result_batch",
            traced_encode,
        )
        monkeypatch.setattr(ContentStoreAuthority, "put_blob", traced_put)

        with pytest.raises(MemoryError, match="additional workspace"):
            execution.save(operation_memory_limit_bytes=required - 1)
        assert encode_calls == 0
        assert put_calls == 0

        reference = execution.save(operation_memory_limit_bytes=required)
        assert isinstance(reference, FitResultArtifactRef)
        assert encode_calls == 1
        assert put_calls == 1
        default_reference = execution.save()
        assert default_reference == reference
        assert encode_calls == 2
        assert put_calls == 2
    finally:
        repository.close()


def test_load_codec_and_source_phases_share_one_aggregate_limit(
    tmp_path,
    capture_case,
    monkeypatch,
) -> None:
    repository = FitResultRepository(tmp_path / "fits")
    try:
        execution = _execution(repository, capture_case)
        save_required = (
            fit_result_encode_additional_peak_upper_bound_nbytes(
                execution.result
            )
            + fit_result_module._FIT_SAVE_REPOSITORY_FIXED_BYTES
        )
        reference = execution.save(
            operation_memory_limit_bytes=save_required
        )
        manifest_payload = repository._store_authority.read_manifest(
            "fit-result",
            reference.manifest_digest,
        )
        result_ref = content_ref_from_tree(
            decode(manifest_payload)["result_blob"]
        )
        inspection = capture_case.capture_repository.inspect_final(
            capture_case.capture_reference,
            memory_limit_bytes=512 << 20,
        )
        fixed = fit_result_module._FIT_LOAD_REPOSITORY_FIXED_BYTES
        inspection_peak = inspection.inspection_decode_peak_upper_bound_bytes
        inspection_retained = inspection.inspection_retained_upper_bound_bytes
        decode_peak = fit_result_decode_additional_peak_upper_bound_nbytes(
            result_ref.size
        )
        result_retained = fit_result_retained_upper_bound_nbytes(
            execution.result
        )
        validation_peak = (
            fit_result_source_validation_additional_peak_upper_bound_nbytes(
                execution.result,
                inspection.dataset_schema,
            )
        )
        required = fixed + max(
            inspection_peak,
            inspection_retained + decode_peak,
            inspection_retained + result_retained + validation_peak,
        )
        read_calls = 0
        owner_read = ContentStoreAuthority.read_blob

        def traced_read(authority, candidate, *args, **kwargs):
            nonlocal read_calls
            if candidate == result_ref:
                read_calls += 1
            return owner_read(authority, candidate, *args, **kwargs)

        monkeypatch.setattr(ContentStoreAuthority, "read_blob", traced_read)
        with pytest.raises(MemoryError, match="aggregate predecode peak"):
            repository.load(
                reference,
                capture_repository=capture_case.capture_repository,
                memory_limit_bytes=required - 1,
            )
        if fixed + inspection_retained + decode_peak == required:
            assert read_calls == 0

        admitted = repository.load(
            reference,
            capture_repository=capture_case.capture_repository,
            memory_limit_bytes=required,
        )
        assert admitted.reference == reference
        assert admitted.source_artifact_ref == capture_case.capture_reference
    finally:
        repository.close()


def test_raw_fit_result_cannot_be_promoted_without_execution_authority(
    tmp_path,
    capture_case,
) -> None:
    repository = FitResultRepository(tmp_path / "fits")
    try:
        execution = _execution(repository, capture_case)
        assert not hasattr(repository, "save")
        with pytest.raises(PermissionError, match="authority"):
            repository._save_execution(  # type: ignore[arg-type]
                execution.result,
                operation_memory_limit_bytes=512 << 20,
            )
        with pytest.raises(PermissionError, match="only be minted"):
            FitExecution(
                object(),
                repository=repository,
                source_artifact_ref=capture_case.capture_reference,
                result=execution.result,
            )
        reference = execution.save(operation_memory_limit_bytes=512 << 20)
        with pytest.raises(PermissionError, match="only be minted"):
            AdmittedFitResult(
                object(),
                reference=reference,
                source_artifact_ref=capture_case.capture_reference,
                result=execution.result,
            )
    finally:
        repository.close()


def test_fit_repository_root_has_one_immutable_owner(tmp_path, capture_case) -> None:
    root = tmp_path / "fits"
    repository = FitResultRepository(root)
    try:
        with pytest.raises(RepositoryRootBusy):
            FitResultRepository(root)
        with pytest.raises(AttributeError, match="immutable"):
            repository.repository_id = "forged"
        with pytest.raises(TypeError, match="final"):
            class _DerivedRepository(FitResultRepository):
                pass
    finally:
        repository.close()


@pytest.mark.parametrize("target", ("blob", "manifest"))
def test_load_fails_closed_on_content_corruption(
    target,
    tmp_path,
    capture_case,
) -> None:
    repository = FitResultRepository(tmp_path / "fits")
    try:
        reference = _execution(repository, capture_case).save(
            operation_memory_limit_bytes=512 << 20
        )
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
            repository.load(
                reference,
                capture_repository=capture_case.capture_repository,
            )
    finally:
        repository.close()


def test_foreign_fit_repository_rejects_reference(tmp_path, capture_case) -> None:
    first = FitResultRepository(tmp_path / "fits-a")
    second = FitResultRepository(
        tmp_path / "fits-b",
        repository_id="foreign-fit-repository",
    )
    try:
        reference = _execution(first, capture_case).save(
            operation_memory_limit_bytes=512 << 20
        )
        with pytest.raises(ValueError, match="another repository"):
            second.load(
                reference,
                capture_repository=capture_case.capture_repository,
            )
    finally:
        second.close()
        first.close()
