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
    fit_spec_for,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.artifacts import (
    AdmittedFitResult,
    FitExecution,
    FitResultArtifactRef,
    FitResultRepository,
)
from zlc_neutral_atom.capture.artifact import (
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.capture.frames import CaptureFrameSource
from zlc_neutral_atom.devices.camera.endpoint import CameraCaptureEndpoint
from zlc_neutral_atom.devices.simulation.sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
)
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.devices.simulation.apparatus import VirtualSequencer
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.runtime.ports import DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceArbiter,
    ResourceKey,
)
from zlc_neutral_atom.runtime.run import RunController
from zlc_neutral_atom.capture.triggered import TriggeredCaptureSpec
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_pulse import (
    PulseExecutionForm,
    load_deployed_pulse_target,
    load_pulse_document,
    pulse_target_manifest_from_lanes,
)
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

    def arm(
        self,
        frames: int,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        assert source_group_sizes is not None
        assert sum(source_group_sizes) == frames
        assert buffer_frame_count == frames and timeout > 0
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
            )
        )
        target = load_deployed_pulse_target()
        self.sequencer = VirtualSequencer(
            target,
            clock_hz=default_clock_hz(DEFAULT_CONFIG_PATH),
            sleep_scale=0,
        )
        pulse_endpoint = VirtualSequencerExecutionEndpoint(
            self.sequencer,
            pulse_target_manifest_from_lanes(target),
        )
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
                _ROOT / "pulses" / "imaging_template.json"
            ),
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channel="ch11",
            layout=TriggeredCameraLayout(
                repeat,
                AxisId("readout-event"),
                AxisId("scan-ordinal"),
                readout_events_per_repeat=3,
            ),
        )
        pipeline = MinimalPipelineSpec(
            "capture fit source",
            binding.capture,
            BlockId("capture-fit-source"),
        )
        triggered = TriggeredCaptureSpec(
            pipeline,
            binding.pulse_port,
            binding.pulse_request,
            binding.trigger_channel,
            binding.cell_plan,
        )
        self.capture_repository = CaptureRepository(tmp_path / "captures")
        self.resources = ResourceArbiter()
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
    artifact = case.capture_repository.load(case.capture_reference)
    spec = fit_spec_for(
        artifact.frame_source.schema,
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
        first = execution.save()
        second = execution.save()
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


def test_raw_fit_result_cannot_be_promoted_without_execution_authority(
    tmp_path,
    capture_case,
) -> None:
    repository = FitResultRepository(tmp_path / "fits")
    try:
        execution = _execution(repository, capture_case)
        assert not hasattr(repository, "save")
        with pytest.raises(PermissionError, match="authority"):
            repository._save_execution(execution.result)  # type: ignore[arg-type]
        with pytest.raises(PermissionError, match="only be minted"):
            FitExecution(
                object(),
                repository=repository,
                source_artifact_ref=capture_case.capture_reference,
                result=execution.result,
            )
        reference = execution.save()
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
        reference = _execution(repository, capture_case).save()
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
        reference = _execution(first, capture_case).save()
        with pytest.raises(ValueError, match="another repository"):
            second.load(
                reference,
                capture_repository=capture_case.capture_repository,
            )
    finally:
        second.close()
        first.close()
