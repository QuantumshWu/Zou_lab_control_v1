"""Capture authority, exact-retention, and physical terminal contracts."""

from __future__ import annotations

import hashlib
import pickle
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    PointLayout,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    VALID,
    ValidityContract,
    Value,
    ValueSchema,
)
from zlc_neutral_atom.runtime.capture import (
    BoundCapturePort,
    CaptureCapabilitySnapshot,
    CapturePreparedAck,
    CaptureProcessorInputBinding,
    CaptureRuntimeProfile,
    CaptureSessionState,
    CaptureStartedAck,
    CaptureStreamContract,
    CaptureTerminalAck,
    CapturedPayloadAck,
    CompleteCaptureCommand,
    PrepareCaptureCommand,
    ReadCaptureCommand,
    StartCaptureCommand,
    FrozenCaptureSpec,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetMode,
    FrozenDatasetEdge,
    SealedDatasetArtifact,
    dataset_cell_key_fingerprint,
)
from zlc_neutral_atom.runtime.ports import (
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
    SafeStateAck,
    SafetyOperation,
    SessionCloseCommand,
    SessionClosedAck,
    VerifiedDeviceCapability,
)
from zlc_neutral_atom.runtime.resources import (
    MemoryQuarantineJournal,
    ResourceArbiter,
    ResourceKey,
)
from zlc_neutral_atom.runtime.run import (
    RunContext,
    RunController,
    CleanupReport,
    RunFailed,
    RunCancelled,
    RunMode,
    RunPlan,
    RunStartRejected,
)
from zlc_neutral_atom.runtime.pipeline import (
    BoundMeasurement,
    compile_pipeline,
    DatasetMaterializerSpec,
    MeasurementDefinition,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    PipelineResult,
    resolve_measurement_definition,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ProducerFlowControl,
    SourceFailed,
    StreamId,
    TraceBinding,
    TraceContext,
    ReservationState,
)
from zlc_neutral_atom.catalog import DefinitionCatalog, DefinitionKey
from zlc_neutral_atom.processing.stream import (
    BoundStreamProcessor,
    ExactStreamProcessorWorker,
    StreamProcessorDefinition,
    StreamProcessorError,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def schema(points: int = 2) -> DatasetSchema:
    y = axis("camera.y", SPATIAL_Y, 2)
    x = axis("camera.x", SPATIAL_X, 3)
    return DatasetSchema(
        axis("repeat", REPEAT, 1),
        (axis("point", SCAN_POINT, points),),
        PointLayout.rect_c((points,)),
        ValueSchema((y, x), ValidityContract.value(), np.dtype("<u2"), "count"),
    )


def identity_camera_payload(payload: object, config: object) -> object:
    assert config is None
    return payload


CAPTURE_PROCESSOR_ENTERED = threading.Event()
CAPTURE_PROCESSOR_RELEASE = threading.Event()


def blocking_camera_payload(payload: object, config: object) -> object:
    assert config is None
    CAPTURE_PROCESSOR_ENTERED.set()
    if not CAPTURE_PROCESSOR_RELEASE.wait(1.0):
        raise TimeoutError("test did not release camera processor")
    return payload


@dataclass(frozen=True)
class CameraPayload:
    ordinal: int
    captured_at: float
    correlation_id: str
    stamp: int
    pixels: np.ndarray


@dataclass(frozen=True)
class CameraPayloadContract:
    value_schema: ValueSchema
    fingerprint: str = SHA_A
    max_retained_nbytes: int = 64

    def snapshot(self, payload: CameraPayload) -> CameraPayload:
        if not isinstance(payload, CameraPayload):
            raise TypeError("payload must be CameraPayload")
        pixels = np.array(payload.pixels, dtype=np.uint16, copy=True, order="C")
        pixels.setflags(write=False)
        return CameraPayload(
            int(payload.ordinal),
            float(payload.captured_at),
            str(payload.correlation_id),
            int(payload.stamp),
            pixels,
        )

    def validate(self, payload: CameraPayload) -> None:
        if not isinstance(payload, CameraPayload):
            raise TypeError("payload must be CameraPayload")
        if payload.ordinal < 0 or payload.stamp < 0:
            raise ValueError("payload counters must be non-negative")
        if payload.pixels.shape != self.value_schema.data_shape:
            raise ValueError("payload shape differs from ValueSchema")
        if payload.pixels.dtype != self.value_schema.dtype:
            raise ValueError("payload dtype differs from ValueSchema")
        if payload.pixels.flags.writeable:
            raise ValueError("payload snapshot must own a read-only array")

    @staticmethod
    def retained_nbytes(payload: CameraPayload) -> int:
        return int(payload.pixels.nbytes) + 32

    def digest(self, payload: CameraPayload) -> str:
        self.validate(payload)
        hasher = hashlib.sha256()
        for part in (
            str(payload.ordinal).encode("ascii"),
            payload.captured_at.hex().encode("ascii"),
            payload.correlation_id.encode("utf-8"),
            str(payload.stamp).encode("ascii"),
            memoryview(payload.pixels).cast("B"),
        ):
            hasher.update(len(part).to_bytes(8, "big"))
            hasher.update(part)
        return hasher.hexdigest()

    @staticmethod
    def source_ordinal(payload: CameraPayload) -> int:
        return payload.ordinal

    @staticmethod
    def captured_at(payload: CameraPayload) -> float:
        return payload.captured_at

    @staticmethod
    def correlation_id(payload: CameraPayload) -> str:
        return payload.correlation_id


@dataclass(frozen=True)
class FrameMetadata:
    ordinal: int
    stamp: int
    captured_at: float


@dataclass(frozen=True)
class FrameMetadataContract:
    fingerprint: str = SHA_B
    max_retained_nbytes: int = 24

    @staticmethod
    def snapshot(payload: CameraPayload) -> FrameMetadata:
        return FrameMetadata(payload.ordinal, payload.stamp, payload.captured_at)

    @staticmethod
    def validate(metadata: object) -> None:
        if not isinstance(metadata, FrameMetadata):
            raise TypeError("metadata must be FrameMetadata")

    @staticmethod
    def retained_nbytes(metadata: object) -> int:
        FrameMetadataContract.validate(metadata)
        return 24

    @staticmethod
    def digest(metadata: object) -> str:
        FrameMetadataContract.validate(metadata)
        assert isinstance(metadata, FrameMetadata)
        return hashlib.sha256(
            f"{metadata.ordinal}:{metadata.stamp}:{metadata.captured_at:.9f}".encode()
        ).hexdigest()


@dataclass(frozen=True)
class CameraEventAdapter:
    payload_contract: CameraPayloadContract
    metadata_contract: FrameMetadataContract
    operator_fingerprint: str = SHA_A

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.value_schema

    def value(self, payload: CameraPayload) -> Value:
        self.payload_contract.validate(payload)
        return Value(payload.pixels, VALID, self.value_schema)


@dataclass(frozen=True)
class CameraCaptureSpec:
    exposure_us: int
    expected_frames: int


def frozen_camera_spec(spec: CameraCaptureSpec) -> FrozenCaptureSpec:
    if not isinstance(spec, CameraCaptureSpec):
        raise TypeError("spec must be CameraCaptureSpec")
    if spec.exposure_us <= 0 or spec.expected_frames <= 0:
        raise ValueError("capture settings must be positive")
    return FrozenCaptureSpec(
        SHA_C,
        f"camera-spec-v1:{spec.exposure_us}:{spec.expected_frames}".encode(),
    )


def cells(dataset_schema: DatasetSchema) -> tuple[DatasetCellAddress, ...]:
    return tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(dataset_schema.repeat_axis.size)
        for point in range(dataset_schema.point_layout.storage_size)
    )


def metadata_digest(payloads, contract, count: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(contract.fingerprint.encode("ascii"))
    for payload in payloads[:count]:
        frozen = payload if not payload.pixels.flags.writeable else np_payload_copy(payload)
        metadata = contract.snapshot(frozen)
        hasher.update(contract.digest(metadata).encode("ascii"))
    return hasher.hexdigest()


def np_payload_copy(payload: CameraPayload) -> CameraPayload:
    pixels = np.array(payload.pixels, copy=True)
    pixels.setflags(write=False)
    return CameraPayload(
        payload.ordinal,
        payload.captured_at,
        payload.correlation_id,
        payload.stamp,
        pixels,
    )


class FakeCamera:
    def __init__(self, payloads: list[CameraPayload], metadata_contract):
        self.payloads = payloads
        self.metadata_contract = metadata_contract
        self.binding_id = ""
        self.generation = "camera-generation"
        self.session_id = ""
        self.read_count = 0
        self.started = False
        self.started_event = threading.Event()
        self.read_entered = threading.Event()
        self.closed = False
        self.close_failure = False
        self.close_attempted = False
        self.close_session_id_override: str | None = None
        self.close_joined = True
        self.fail_prepare = False
        self.fail_start = False
        self.terminal_produced_delta = 0
        self.terminal_drained_delta = 0
        self.terminal_source_stopped = True
        self.terminal_no_more = True
        self.terminal_joined = True
        self.terminal_digest_override: str | None = None
        self.block_read = threading.Event()
        self.release_read = threading.Event()
        self.block_capability_probe = threading.Event()
        self.capability_probe_entered = threading.Event()
        self.release_capability_probe = threading.Event()
        self.capability = None

    def execute(self, command):
        if isinstance(command, PrepareCaptureCommand):
            assert command.timeout_seconds == 0.5
            assert command.capture_spec_owner_fingerprint == SHA_C
            assert command.capture_spec_payload.startswith(b"camera-spec-v1:")
            self.session_id = command.session_id
            self.read_count = 0
            self.started = False
            self.closed = False
            if self.fail_prepare:
                raise RuntimeError("prepare failed after partial configuration")
            return CapturePreparedAck(
                command.session_id,
                self.binding_id,
                self.generation,
                SHA_D,
                SHA_C,
                command.capture_spec_fingerprint,
            )
        if isinstance(command, StartCaptureCommand):
            assert command.timeout_seconds == 0.5
            self.started = True
            self.started_event.set()
            if self.fail_start:
                raise RuntimeError("start failed after arm")
            return CaptureStartedAck(
                command.session_id,
                self.binding_id,
                self.generation,
            )
        if isinstance(command, ReadCaptureCommand):
            assert command.timeout_seconds == 0.5
            if self.block_read.is_set():
                self.read_entered.set()
                self.release_read.wait(2.0)
                raise RuntimeError("read interrupted")
            payload = self.payloads[self.read_count]
            self.read_count += 1
            return CapturedPayloadAck(
                command.session_id,
                self.binding_id,
                self.generation,
                payload,
            )
        if isinstance(command, CompleteCaptureCommand):
            assert command.timeout_seconds == 0.5
            digest = self.terminal_digest_override or metadata_digest(
                self.payloads,
                self.metadata_contract,
                command.expected_total_events,
            )
            return CaptureTerminalAck(
                command.session_id,
                self.binding_id,
                self.generation,
                command.expected_total_events + self.terminal_produced_delta,
                command.expected_total_events + self.terminal_drained_delta,
                self.terminal_source_stopped,
                self.terminal_no_more,
                self.terminal_joined,
                digest,
                SHA_D,
                SHA_C,
                frozen_camera_spec(
                    CameraCaptureSpec(10, command.expected_total_events)
                ).digest,
            )
        raise AssertionError(f"unexpected camera command {command!r}")

    def interrupt(self):
        self.release_read.set()
        return "camera-interrupt-ack"

    def probe_capability(self):
        if self.block_capability_probe.is_set():
            self.capability_probe_entered.set()
            self.release_capability_probe.wait(2.0)
        if self.capability is None:
            raise RuntimeError("camera capability is not initialized")
        return self.capability

    def close_session(self, command: SessionCloseCommand):
        assert command.timeout_seconds == 0.5
        self.close_attempted = True
        if self.close_failure:
            raise RuntimeError("session join acknowledgement lost")
        self.closed = True
        self.release_read.set()
        return SessionClosedAck(
            self.close_session_id_override or command.session_id,
            self.binding_id,
            self.generation,
            True,
            True,
            self.close_joined,
            "camera-session-joined",
        )

    def safe_state(self):
        if self.session_id and not self.closed:
            raise RuntimeError("camera session is not closed")
        return SafeStateAck("camera-safe-and-session-joined")


@dataclass
class CaptureHarness:
    camera: FakeCamera
    broker: DeviceBroker
    port: BoundCapturePort
    contract: CaptureStreamContract
    spec: FrozenCaptureSpec
    holder: dict

    def plan(
        self,
        *,
        consume: bool = True,
        build_materializer: bool = True,
        materializer_cells=None,
        materializer_adapter=None,
    ) -> RunPlan:
        def preflight(context: RunContext):
            trace = TraceBinding(context.run_id.value, self.contract.source_id)
            session = self.port.open_session(self.contract, trace, self.spec)
            self.holder["session"] = session
            reservation = session.reserve_exact()
            cursor = reservation.activate()
            self.holder.update(reservation=reservation, cursor=cursor)
            builder = None
            if build_materializer:
                materializer_edge = (
                    self.contract.dataset_edge
                    if materializer_cells is None and materializer_adapter is None
                    else FrozenDatasetEdge(
                        self.contract.dataset_schema,
                        materializer_adapter or self.contract.event_adapter,
                        materializer_cells or self.contract.expected_cells,
                    )
                )
                builder = DatasetBuilder(
                    BlockId("capture"),
                    reservation,
                    materializer_edge,
                    DatasetMode.FINITE_EXACT,
                )
                self.holder["builder"] = builder
                session.bind_exact_consumer(builder.exact_readiness())
            self.holder["builder"] = builder
            session.prepare(context)
            return session, cursor, builder

        def execute(context: RunContext, prepared):
            session, cursor, builder = prepared
            try:
                session.start(context)
                if not consume:
                    return session
                assert builder is not None
                for _ in self.contract.expected_cells:
                    session.capture_next(context)
                    builder.consume(cursor.next())
                completion = session.complete(context)
                artifact = builder.seal(completion.eos)
                assert (
                    artifact.provenance.ordered_metadata_digest
                    == completion.terminal.ordered_metadata_digest
                )
                return artifact, completion
            except BaseException as error:
                try:
                    session.fail(error)
                finally:
                    if builder is not None:
                        builder.abort()
                raise

        def cleanup(context: RunContext, _prepared, _primary):
            session = self.holder.get("session")
            builder = self.holder.get("builder")
            software_errors = []
            if builder is not None:
                try:
                    builder.close()
                except BaseException as error:
                    software_errors.append(error)
            if session is None:
                raise RuntimeError("capture cleanup lost its session owner")
            report = session.cleanup(context)
            if not software_errors:
                return report
            return CleanupReport(
                safety_proofs=report.safety_proofs,
                decisions=report.decisions,
                errors=(*report.errors, *software_errors),
            )

        return RunPlan(
            name="capture contract test",
            mode=RunMode.FINITE_EXACT,
            resource_claims=(self.port.resource_claim,),
            hazard_claims=(self.port.hazard_claim,),
            bound_devices=(self.port.device,),
            preflight=preflight,
            execute=execute,
            cleanup=cleanup,
            finalize=lambda _context, result: result,
            interrupt_operations=self.port.interrupt_operations,
        )


def harness(points: int = 2) -> CaptureHarness:
    dataset_schema = schema(points)
    payload_contract = CameraPayloadContract(dataset_schema.cell_schema)
    metadata_contract = FrameMetadataContract()
    event_adapter = CameraEventAdapter(payload_contract, metadata_contract)
    payloads = [
        CameraPayload(
            ordinal=index,
            captured_at=float(index + 1),
            correlation_id=f"frame-{index}",
            stamp=100 + index,
            pixels=np.full((2, 3), index + 1, dtype=np.uint16),
        )
        for index in range(points)
    ]
    camera = FakeCamera(payloads, metadata_contract)
    identity_evidence_digest = camera.generation
    key = ResourceKey.parse("device/camera/test")
    broker = DeviceBroker()
    identity = broker.verify_identity(
        lambda: DeviceIdentityAck(
            "camera-serial",
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
            identity_evidence_digest,
            "test-assets-v1",
        )
    )
    device = broker.bind(
        key=key,
        identity=identity,
        execute_command=camera.execute,
        capability_probe=camera.probe_capability,
        cleanup_operations={},
        close_session=camera.close_session,
        verify_safe_state=camera.safe_state,
        interrupt_operations={SafetyOperation.ABORT: camera.interrupt},
    )
    camera.binding_id = device.binding_id
    camera.generation = device.connection_generation
    capability = CaptureCapabilitySnapshot(
        binding_id=device.binding_id,
        stable_device_identity=device.stable_device_identity,
        connection_generation=device.connection_generation,
        capability_fingerprint=SHA_C,
        settings_fingerprint=SHA_D,
        payload_contract_fingerprint=payload_contract.fingerprint,
        capture_spec_owner_fingerprint=SHA_C,
        flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
        max_source_burst_events=points,
        driver_ring_bytes=128,
        adapter_record_retention_bytes=128,
        max_blocking_call_seconds=0.5,
        max_capture_spec_bytes=1024,
    )
    camera.capability = capability
    attestation = broker.verify_capability(device)
    port = BoundCapturePort(attestation, ())
    contract = CaptureStreamContract(
        stream_id=StreamId("camera.frames"),
        source_id="camera",
        dataset_schema=dataset_schema,
        payload_contract=payload_contract,
        event_adapter=event_adapter,
        expected_cells=cells(dataset_schema),
        capability=capability,
        runtime_profile=CaptureRuntimeProfile(
            required_consumer_lag_events=0,
            transport_memory_limit_bytes=2048,
        ),
        capture_spec_owner_fingerprint=SHA_C,
    )
    return CaptureHarness(
        camera,
        broker,
        port,
        contract,
        frozen_camera_spec(CameraCaptureSpec(10, points)),
        {},
    )


@dataclass
class ProcessorCaptureHarness:
    session: object
    reservation: object
    worker: ExactStreamProcessorWorker
    builder: DatasetBuilder


def processor_capture_harness(
    item: CaptureHarness,
    *,
    operator=identity_camera_payload,
    run_id: str = "processor-capture-run",
) -> ProcessorCaptureHarness:
    trace = TraceBinding(run_id, item.contract.source_id)
    session = item.port.open_session(item.contract, trace, item.spec)
    reservation = session.reserve_exact()
    input_cursor = reservation.activate()
    processor_input = session.processor_input_binding
    key_contract = processor_input.join_key_contract
    output_stream, output_producer = AcquisitionStream.create(
        StreamId("camera.processed"),
        item.contract.payload_contract,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=item.contract.total_events,
        retention_bytes=(
            item.contract.total_events
            * item.contract.payload_contract.max_retained_nbytes
        ),
        join_key_contract=key_contract,
    )
    output_reservation = output_stream.reserve(
        total_events=item.contract.total_events,
        max_inflight_events=item.contract.total_events,
        max_inflight_bytes=(
            item.contract.total_events
            * item.contract.payload_contract.max_retained_nbytes
        ),
        trace_binding=TraceBinding(run_id, "camera-processor"),
    )
    output_cursor = output_reservation.activate()
    builder = DatasetBuilder(
        BlockId("processed-camera"),
        output_reservation,
        item.contract.dataset_edge,
        DatasetMode.FINITE_EXACT,
    )
    definition = StreamProcessorDefinition(
        DefinitionKey("tests", "identity-camera", 1),
        "Identity camera",
        "tests.identity-camera.v1",
        item.contract.payload_contract.fingerprint,
        item.contract.payload_contract.fingerprint,
        dataset_cell_key_fingerprint(item.contract.dataset_schema),
    )
    bound = BoundStreamProcessor(
        definition,
        None,
        item.contract.payload_contract,
        item.contract.payload_contract,
        key_contract,
        output_stream.stream_id,
        "camera-processor",
        operator,
    )
    worker = ExactStreamProcessorWorker(
        bound,
        reservation,
        input_cursor,
        input_edge=item.contract.dataset_edge,
        output_producer=output_producer,
        output_cursor=output_cursor,
        output_builder=builder,
        deadline_monotonic=time.monotonic() + 2.0,
    )
    return ProcessorCaptureHarness(session, reservation, worker, builder)


def test_capture_session_exposes_one_atomic_processor_input_binding():
    item = harness(points=1)
    session = item.port.open_session(
        item.contract,
        TraceBinding("processor-input-binding", item.contract.source_id),
        item.spec,
    )

    binding = session.processor_input_binding

    assert session.processor_input_binding is binding
    assert binding.capture_contract is item.contract
    assert binding.stream is session.stream
    assert binding.payload_contract is item.contract.payload_contract
    assert binding.input_edge is item.contract.dataset_edge
    assert (
        binding.join_key_contract.fingerprint
        == item.contract.dataset_edge.key_contract_fingerprint
    )
    assert (
        binding.stream.payload_contract_fingerprint
        == binding.input_edge.payload_contract_fingerprint
        == binding.payload_contract.fingerprint
    )
    binding.input_edge.validate_stream(binding.stream)
    reservation = session.reserve_exact()
    reservation.activate()
    assert session.state is CaptureSessionState.NEW
    assert session.processor_input_binding is binding
    assert binding.require_reservation(reservation) is None
    reservation.abort(cancelled=True)
    reservation.release()
    with pytest.raises(RuntimeError, match="no longer registered"):
        binding.require_reservation(reservation)

    import zlc_neutral_atom.runtime as runtime_api

    assert runtime_api.CaptureProcessorInputBinding is CaptureProcessorInputBinding


def test_capture_session_stops_dispensing_processor_input_after_prepare():
    item = harness(points=1)
    plan = item.plan(consume=False)
    original_execute = plan.execute

    def execute(context: RunContext, prepared):
        session, _cursor, _builder = prepared
        assert session.state is CaptureSessionState.PREPARED
        with pytest.raises(RuntimeError, match="only available.*NEW"):
            session.processor_input_binding
        return original_execute(context, prepared)

    guarded_plan = RunPlan(**{**plan.__dict__, "execute": execute})
    RunController(ResourceArbiter(MemoryQuarantineJournal())).start(
        guarded_plan
    ).result(2.0)


def test_processor_input_binding_rejects_clone_stream_reservation():
    item = harness(points=1)
    session = item.port.open_session(
        item.contract,
        TraceBinding("processor-input-clone", item.contract.source_id),
        item.spec,
    )
    binding = session.processor_input_binding
    clone_stream, _clone_producer = AcquisitionStream.create(
        binding.stream.stream_id,
        binding.payload_contract,
        flow_control=binding.stream.flow_control,
        retention_events=binding.stream.retention_events,
        retention_bytes=binding.stream.retention_bytes,
        join_key_contract=binding.join_key_contract,
    )
    clone_reservation = clone_stream.reserve(
        total_events=item.contract.total_events,
        max_inflight_events=item.contract.max_inflight_events,
        max_inflight_bytes=item.contract.max_inflight_bytes,
        trace_binding=TraceBinding(
            "processor-input-clone",
            item.contract.source_id,
        ),
    )

    assert clone_stream is not binding.stream
    assert clone_stream.stream_id == binding.stream.stream_id
    assert (
        clone_stream.payload_contract_fingerprint
        == binding.stream.payload_contract_fingerprint
    )
    binding.input_edge.validate_stream(clone_stream)
    with pytest.raises(PermissionError, match="not minted by this CaptureSession"):
        binding.require_reservation(clone_reservation)

    clone_reservation.abort(cancelled=True)
    clone_reservation.release()


def test_processor_input_rejects_caller_reserved_interval_on_the_same_stream():
    item = harness(points=1)
    session = item.port.open_session(
        item.contract,
        TraceBinding("processor-input-rogue-reservation", item.contract.source_id),
        item.spec,
    )
    binding = session.processor_input_binding
    rogue = binding.stream.reserve(
        total_events=item.contract.total_events,
        max_inflight_events=item.contract.max_inflight_events,
        max_inflight_bytes=item.contract.max_inflight_bytes,
        trace_binding=TraceBinding(
            "processor-input-rogue-reservation",
            item.contract.source_id,
        ),
    )
    rogue.activate()

    with pytest.raises(PermissionError, match="not minted by this CaptureSession"):
        binding.require_reservation(rogue)
    with pytest.raises(AttributeError):
        object.__setattr__(binding, "_session_reservation", rogue)

    rogue.abort(cancelled=True)
    rogue.release()


def test_processor_input_binding_rejects_a_copied_clone_generation():
    item = harness(points=1)
    session = item.port.open_session(
        item.contract,
        TraceBinding("processor-input-forged-clone", item.contract.source_id),
        item.spec,
    )
    binding = session.processor_input_binding
    reservation = session.reserve_exact()
    reservation.activate()
    clone_stream, _clone_producer = AcquisitionStream.create(
        binding.stream.stream_id,
        binding.payload_contract,
        flow_control=binding.stream.flow_control,
        retention_events=binding.stream.retention_events,
        retention_bytes=binding.stream.retention_bytes,
        join_key_contract=binding.join_key_contract,
    )
    forged = object.__new__(CaptureProcessorInputBinding)
    for slot in CaptureProcessorInputBinding.__slots__:
        if slot == "__weakref__":
            continue
        object.__setattr__(forged, slot, object.__getattribute__(binding, slot))
    object.__setattr__(forged, "_stream", clone_stream)

    with pytest.raises(PermissionError, match="not registered"):
        forged.require_reservation(reservation)

    reservation.abort(cancelled=True)
    reservation.release()


def test_capture_processor_input_binding_is_sealed_immutable_and_process_local():
    item = harness(points=1)
    session = item.port.open_session(
        item.contract,
        TraceBinding("sealed-processor-input", item.contract.source_id),
        item.spec,
    )
    binding = session.processor_input_binding

    with pytest.raises(TypeError, match="exact ExactReservation"):
        binding.require_reservation(object())
    with pytest.raises(PermissionError, match="only be minted by CaptureSession"):
        CaptureProcessorInputBinding(
            object(),
            capture_contract=item.contract,
            stream=session.stream,
        )
    with pytest.raises(PermissionError, match="not minted by CaptureSession"):
        object.__new__(CaptureProcessorInputBinding).stream
    with pytest.raises(TypeError, match="sealed"):
        type(
            "ForgedCaptureProcessorInputBinding",
            (CaptureProcessorInputBinding,),
            {},
        )
    with pytest.raises(AttributeError, match="immutable"):
        binding.stream = session.stream
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(binding)


def emit_processor_capture_source(
    processor: ProcessorCaptureHarness,
    item: CaptureHarness,
) -> None:
    for payload, cell in zip(item.camera.payloads, item.contract.expected_cells):
        processor.session._producer.emit(
            payload,
            captured_at=payload.captured_at,
            trace=TraceContext(
                processor.session._trace_binding.run_id,
                item.contract.source_id,
                payload.correlation_id,
            ),
            join_key=cell,
        )


def run(harness_value: CaptureHarness, **plan_options):
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    return controller.start(harness_value.plan(**plan_options))


def minimal_pipeline(item: CaptureHarness, *, memory_limit: int = 4 << 20):
    definition = MeasurementDefinition(
        DefinitionKey("tests", "camera-capture", 1),
        "Camera capture",
        "tests.camera-request.v1",
        "tests.camera-binding.v1",
        SHA_C,
        item.contract.dataset_schema.fingerprint,
    )
    bound = BoundMeasurement(
        definition,
        item.port,
        item.contract,
        item.spec,
    )
    return MinimalPipelineSpec(
        "minimal camera pipeline",
        bound,
        DatasetMaterializerSpec(
            BlockId("pipeline-capture"),
            PipelineMemoryProfile.for_current_runtime(memory_limit),
        ),
    )


def test_capture_preserves_multidimensional_payload_and_co_seals_metadata():
    item = harness()
    artifact, completion = run(item).result(2.0)
    assert isinstance(artifact, SealedDatasetArtifact)
    assert artifact.block.values.shape == (1, 2, 2, 3)
    assert np.all(artifact.block.values[0, 0] == 1)
    assert np.all(artifact.block.values[0, 1] == 2)
    assert completion.terminal.joined
    assert item.camera.closed
    assert item.holder["reservation"].state is ReservationState.RELEASED
    assert not item.holder["session"].stream._reservations


@pytest.mark.parametrize(
    "field,value",
    [
        ("terminal_produced_delta", 1),
        ("terminal_drained_delta", -1),
        ("terminal_source_stopped", False),
        ("terminal_no_more", False),
        ("terminal_joined", False),
        ("terminal_digest_override", "f" * 64),
    ],
)
def test_terminal_mismatch_poison_prevents_sealed_artifact(field, value):
    item = harness()
    setattr(item.camera, field, value)
    handle = run(item)
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert item.holder["session"].state is CaptureSessionState.FAILED
    assert item.camera.closed
    assert item.holder["reservation"].state is ReservationState.RELEASED
    assert not item.holder["session"].stream._reservations


def test_prepare_requires_live_exact_consumer_before_any_hardware_command():
    item = harness()
    handle = run(item, build_materializer=False, consume=False)
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert not item.camera.started
    assert item.camera.session_id == ""
    assert not item.camera.close_attempted
    assert not item.camera.closed
    assert item.holder["reservation"].state is ReservationState.RELEASED
    assert not item.holder["session"].stream._reservations


@pytest.mark.parametrize("invalidation", ["closed-consumer", "failed-source"])
def test_prepare_revalidates_stale_readiness_before_hardware_command(invalidation):
    item = harness(points=1)
    base = item.plan()

    def preflight(context: RunContext):
        trace = TraceBinding(context.run_id.value, item.contract.source_id)
        session = item.port.open_session(item.contract, trace, item.spec)
        item.holder["session"] = session
        reservation = session.reserve_exact()
        cursor = reservation.activate()
        builder = DatasetBuilder(
            BlockId("stale-readiness"),
            reservation,
            item.contract.dataset_edge,
            DatasetMode.FINITE_EXACT,
        )
        item.holder["builder"] = builder
        session.bind_exact_consumer(builder.exact_readiness())
        if invalidation == "closed-consumer":
            builder.close()
        else:
            session._producer.fail(SourceFailed("failed before hardware prepare"))
        session.prepare(context)
        return session, cursor, builder

    plan = RunPlan(**{**base.__dict__, "preflight": preflight})
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    with pytest.raises(RunFailed):
        controller.start(plan).result(2.0)
    assert item.camera.session_id == ""
    assert not item.camera.close_attempted
    assert not item.camera.started
    item.holder["builder"].close()


def test_processor_readiness_requires_started_live_worker_before_capture_binding():
    item = harness(points=1)
    processor = processor_capture_harness(item)
    with pytest.raises(StreamProcessorError, match="has not started"):
        processor.worker.exact_readiness()
    assert item.camera.session_id == ""
    processor.worker.start()
    readiness = processor.worker.exact_readiness()
    processor.session.bind_exact_consumer(readiness)
    processor.session._validate_readiness(readiness, processor.reservation)
    assert item.camera.session_id == ""
    assert not item.camera.close_attempted
    processor.worker.close(2.0)


@pytest.mark.parametrize("terminal_state", ["failed", "done"])
def test_dead_processor_readiness_is_rejected_before_any_hardware_command(terminal_state):
    item = harness(points=1)
    processor = processor_capture_harness(item)
    processor.worker.start()
    readiness = processor.worker.exact_readiness()
    processor.session.bind_exact_consumer(readiness)
    if terminal_state == "failed":
        processor.worker.cancel("processor failed before capture prepare")
    else:
        emit_processor_capture_source(processor, item)
        processor.session._producer.finish()
    processor.worker.wait(2.0)
    with pytest.raises(Exception):
        processor.session._validate_readiness(readiness, processor.reservation)
    with pytest.raises(StreamProcessorError, match="not live"):
        processor.worker.exact_readiness()
    assert item.camera.session_id == ""
    assert not item.camera.close_attempted


def test_closing_processor_readiness_is_rejected_before_any_hardware_command():
    CAPTURE_PROCESSOR_ENTERED.clear()
    CAPTURE_PROCESSOR_RELEASE.clear()
    item = harness(points=1)
    processor = processor_capture_harness(item, operator=blocking_camera_payload)
    processor.worker.start()
    readiness = processor.worker.exact_readiness()
    processor.session.bind_exact_consumer(readiness)
    emit_processor_capture_source(processor, item)
    assert CAPTURE_PROCESSOR_ENTERED.wait(1.0)
    close_errors: list[BaseException] = []

    def close_worker():
        try:
            processor.worker.close(2.0)
        except BaseException as error:
            close_errors.append(error)

    closer = threading.Thread(target=close_worker)
    closer.start()
    deadline = time.monotonic() + 1.0
    while not processor.worker._closing and time.monotonic() < deadline:
        time.sleep(0.001)
    assert processor.worker._closing
    with pytest.raises(Exception):
        processor.session._validate_readiness(readiness, processor.reservation)
    with pytest.raises(StreamProcessorError, match="not live"):
        readiness._validate_terminal_sink()
    assert item.camera.session_id == ""
    assert not item.camera.close_attempted
    CAPTURE_PROCESSOR_RELEASE.set()
    closer.join(2.0)
    assert not closer.is_alive()
    assert close_errors == []


@pytest.mark.parametrize("mismatch", ["schedule", "adapter"])
def test_materializer_mismatch_is_rejected_before_camera_prepare_or_start(mismatch):
    item = harness()
    options = {}
    if mismatch == "schedule":
        options["materializer_cells"] = tuple(reversed(item.contract.expected_cells))
    else:
        options["materializer_adapter"] = CameraEventAdapter(
            item.contract.payload_contract,
            FrameMetadataContract(fingerprint=SHA_A),
        )
    with pytest.raises(RunFailed):
        run(item, **options).result(2.0)
    assert item.camera.session_id == ""
    assert not item.camera.started
    assert not item.camera.closed
    assert item.holder["reservation"].state is ReservationState.RELEASED
    assert not item.holder["session"].stream._reservations


@pytest.mark.parametrize("failure", ["fail_prepare", "fail_start"])
def test_partial_prepare_or_start_failure_is_monotonic_and_cleanup_joins(failure):
    item = harness()
    setattr(item.camera, failure, True)
    with pytest.raises(RunFailed):
        run(item).result(2.0)
    assert item.holder["session"].state is CaptureSessionState.FAILED
    assert item.camera.closed
    assert item.holder["reservation"].state is ReservationState.RELEASED
    assert not item.holder["session"].stream._reservations


def test_payload_snapshot_does_not_alias_mutated_driver_ring():
    item = harness(points=1)
    original = item.camera.payloads[0].pixels

    def execute(context: RunContext, prepared):
        session, cursor, builder = prepared
        session.start(context)
        session.capture_next(context)
        original[:] = 999
        builder.consume(cursor.next())
        completion = session.complete(context)
        artifact = builder.seal(completion.eos)
        return artifact, completion

    plan = item.plan()
    plan = RunPlan(
        **{
            **plan.__dict__,
            "execute": execute,
        }
    )
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    artifact, _completion = controller.start(plan).result(2.0)
    assert np.all(artifact.block.values == 1)


def test_capture_refuses_extra_read_before_touching_device():
    item = harness(points=1)

    def execute(context: RunContext, prepared):
        session, cursor, builder = prepared
        session.start(context)
        session.capture_next(context)
        builder.consume(cursor.next())
        with pytest.raises(RuntimeError, match="frozen event budget"):
            session.capture_next(context)
        assert item.camera.read_count == 1
        completion = session.complete(context)
        return builder.seal(completion.eos), completion

    plan = item.plan()
    plan = RunPlan(**{**plan.__dict__, "execute": execute})
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    controller.start(plan).result(2.0)


def test_cancel_uses_cleanup_capability_to_close_blocked_session():
    item = harness(points=1)
    item.camera.block_read.set()
    handle = run(item)
    assert item.camera.started_event.wait(1.0)
    assert item.camera.read_entered.wait(1.0)
    handle.cancel()
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert item.camera.closed
    assert item.holder["reservation"].state is ReservationState.RELEASED
    assert not item.holder["session"].stream._reservations


def test_cleanup_join_failure_prevents_success_and_quarantines_device():
    item = harness(points=1)
    item.camera.close_failure = True
    arbiter = ResourceArbiter(MemoryQuarantineJournal())
    handle = RunController(arbiter).start(item.plan())
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert item.port.device.key in {record.key for record in arbiter.quarantine_records()}


@pytest.mark.parametrize("wrong_session,joined", [(True, True), (False, False)])
def test_cleanup_receipt_must_bind_this_session_and_prove_join(wrong_session, joined):
    item = harness(points=1)
    if wrong_session:
        item.camera.close_session_id_override = "another-session"
    item.camera.close_joined = joined
    arbiter = ResourceArbiter(MemoryQuarantineJournal())
    with pytest.raises(RunFailed):
        RunController(arbiter).start(item.plan()).result(2.0)
    assert item.port.device.key in {record.key for record in arbiter.quarantine_records()}


def test_capture_device_operations_cannot_leave_owner_lane():
    item = harness(points=1)

    def execute(context: RunContext, prepared):
        session, cursor, builder = prepared
        failures = []

        def wrong_thread():
            try:
                session.start(context)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=wrong_thread)
        thread.start()
        thread.join()
        assert len(failures) == 1
        assert "owner I/O lane" in str(failures[0])
        assert not item.camera.started
        session.start(context)
        session.capture_next(context)
        builder.consume(cursor.next())
        completion = session.complete(context)
        return builder.seal(completion.eos), completion

    plan = item.plan()
    plan = RunPlan(**{**plan.__dict__, "execute": execute})
    RunController(ResourceArbiter(MemoryQuarantineJournal())).start(plan).result(2.0)


def test_capture_capability_requires_broker_minted_attestation():
    item = harness(points=1)
    with pytest.raises(PermissionError):
        VerifiedDeviceCapability(
            object(),
            broker=object(),
            device=item.port.device,
            snapshot=item.port.capability,
            nonce=object(),
        )


def test_superseded_capability_attestation_fails_before_prepare_command():
    item = harness(points=1)
    item.broker.verify_capability(item.port.device)
    with pytest.raises(RunFailed):
        run(item).result(2.0)
    assert item.camera.session_id == ""


def test_capability_probe_excludes_run_open_for_its_entire_epoch():
    item = harness(points=1)
    item.camera.block_capability_probe.set()
    failures = []

    def probe():
        try:
            item.broker.verify_capability(item.port.device)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=probe)
    thread.start()
    assert item.camera.capability_probe_entered.wait(1.0)
    with pytest.raises(RuntimeError, match="capability probe is in progress"):
        RunController(ResourceArbiter(MemoryQuarantineJournal())).start(item.plan())
    item.camera.release_capability_probe.set()
    thread.join(1.0)
    assert not failures


def test_terminal_ack_rejects_bool_counts_and_truthy_non_bool_flags():
    with pytest.raises(ValueError):
        CaptureTerminalAck(
            "s", "b", "g", True, 1, True, True, True, SHA_A, SHA_B, SHA_C, SHA_D
        )
    with pytest.raises(TypeError):
        CaptureTerminalAck(
            "s", "b", "g", 1, 1, 1, True, True, SHA_A, SHA_B, SHA_C, SHA_D
        )


def test_transport_budget_rejects_before_any_run_or_hardware_prepare():
    item = harness(points=1)
    assert item.contract.estimated_transport_bytes == 128 + 128 + 64 + 64 + 24 + 1024
    with pytest.raises(MemoryError):
        CaptureStreamContract(
            stream_id=item.contract.stream_id,
            source_id=item.contract.source_id,
            dataset_schema=item.contract.dataset_schema,
            payload_contract=item.contract.payload_contract,
            event_adapter=item.contract.event_adapter,
            expected_cells=item.contract.expected_cells,
            capability=item.contract.capability,
            runtime_profile=CaptureRuntimeProfile(0, 1),
            capture_spec_owner_fingerprint=item.contract.capture_spec_owner_fingerprint,
        )
    assert item.camera.session_id == ""


def test_capture_contract_rejects_mutable_state_hidden_in_frozen_owner():
    item = harness(points=1)

    @dataclass(frozen=True)
    class StatefulPayloadContract(CameraPayloadContract):
        cache: list = field(default_factory=list)

    payload = StatefulPayloadContract(item.contract.dataset_schema.cell_schema)
    adapter = CameraEventAdapter(payload, FrameMetadataContract())
    with pytest.raises(TypeError, match="intrinsically immutable"):
        CaptureStreamContract(
            stream_id=item.contract.stream_id,
            source_id=item.contract.source_id,
            dataset_schema=item.contract.dataset_schema,
            payload_contract=payload,
            event_adapter=adapter,
            expected_cells=item.contract.expected_cells,
            capability=item.contract.capability,
            runtime_profile=item.contract.runtime_profile,
            capture_spec_owner_fingerprint=(
                item.contract.capture_spec_owner_fingerprint
            ),
        )


def test_failed_stream_reports_source_failure_to_exact_cursor():
    item = harness(points=1)
    item.camera.payloads[0] = CameraPayload(
        9,
        1.0,
        "wrong-ordinal",
        100,
        np.ones((2, 3), dtype=np.uint16),
    )
    with pytest.raises(RunFailed):
        run(item).result(2.0)
    cursor = item.holder["cursor"]
    with pytest.raises(SourceFailed):
        cursor.next()
    assert item.holder["reservation"].state is ReservationState.RELEASED
    assert not item.holder["session"].stream._reservations


def test_minimal_pipeline_compiles_to_one_flat_run_and_returns_typed_result():
    item = harness(points=2)
    plan = compile_pipeline(minimal_pipeline(item))
    result = RunController(ResourceArbiter(MemoryQuarantineJournal())).start(plan).result(2.0)
    assert isinstance(result, PipelineResult)
    assert result.dataset.block.values.shape == (1, 2, 2, 3)
    assert np.all(result.dataset.block.values[0, 0] == 1)
    assert np.all(result.dataset.block.values[0, 1] == 2)
    assert result.capture_terminal.joined
    assert len(result.memory_profile_fingerprint) == 64
    assert item.camera.closed
    with pytest.raises(PermissionError):
        PipelineResult(
            object(),
            result.dataset,
            result.capture_terminal,
            result.aggregate_peak_bytes,
            result.memory_profile_fingerprint,
        )


def test_pipeline_result_cross_binds_processed_root_span_and_stage_chain():
    from zlc_neutral_atom.runtime.pipeline import _PIPELINE_RESULT_TOKEN

    item = harness(points=2)
    holder = {}
    base = item.plan()

    def preflight(context: RunContext):
        processor = processor_capture_harness(item, run_id=context.run_id.value)
        processor.worker.start()
        processor.session.bind_exact_consumer(processor.worker.exact_readiness())
        holder["processor"] = processor
        return processor

    def execute(context: RunContext, processor: ProcessorCaptureHarness):
        try:
            processor.session.prepare(context)
            processor.session.start(context)
            for _ in item.contract.expected_cells:
                processor.session.capture_next(context)
            completion = processor.session.complete(context)
            dataset = processor.worker.finish(completion.eos, 2.0)
            return PipelineResult(
                _PIPELINE_RESULT_TOKEN,
                dataset,
                completion,
                1,
                "e" * 64,
            )
        except BaseException as error:
            processor.session.fail(error)
            raise

    def cleanup(context: RunContext, processor, _primary):
        if processor is None:
            return item.port.verify_idle(context)
        processor.worker.close(2.0)
        return processor.session.cleanup(context)

    plan = RunPlan(
        **{
            **base.__dict__,
            "preflight": preflight,
            "execute": execute,
            "cleanup": cleanup,
            "finalize": lambda _context, result: result,
        }
    )
    result = RunController(ResourceArbiter(MemoryQuarantineJournal())).start(plan).result(2.0)
    assert not result.is_direct_raw_capture
    derivation = result.dataset.provenance.derivation
    assert derivation is not None
    assert derivation.chain_contract_digest == result.chain_contract_digest
    assert derivation.root_input_span == result._capture_completion.source_event_span
    assert derivation.root_input_span.stream_id == holder["processor"].session.stream.stream_id
    assert derivation.root_input_span.count == item.contract.total_events
    assert len(derivation.stages) == 1

    wrong_digest = (
        "0" * 64 if derivation.root_input_span.ordered_digest != "0" * 64 else "1" * 64
    )
    readiness = holder["processor"].session._exact_consumer_readiness
    assert readiness is not None
    forged_dataset = result.dataset._with_derivation(
        readiness,
        replace(derivation.root_input_span, ordered_digest=wrong_digest),
    )
    with pytest.raises(
        RuntimeError,
        match="derivation differs from capture readiness chain",
    ):
        PipelineResult(
            _PIPELINE_RESULT_TOKEN,
            forged_dataset,
            result._capture_completion,
            result.aggregate_peak_bytes,
            result.memory_profile_fingerprint,
        )


def test_pipeline_budget_rejects_before_run_or_hardware_prepare():
    item = harness(points=1)
    with pytest.raises(PermissionError):
        PipelineMemoryProfile(object(), 1024)
    with pytest.raises(MemoryError):
        compile_pipeline(minimal_pipeline(item, memory_limit=1))
    assert item.camera.session_id == ""
    assert not item.camera.started


def test_pipeline_terminal_mismatch_fails_and_releases_every_authority():
    item = harness(points=1)
    item.camera.terminal_joined = False
    plan = compile_pipeline(minimal_pipeline(item))
    with pytest.raises(RunFailed):
        RunController(ResourceArbiter(MemoryQuarantineJournal())).start(plan).result(2.0)
    assert item.camera.closed


def test_measurement_definition_resolution_is_pure_catalog_composition():
    item = harness(points=1)
    bound = minimal_pipeline(item).measurement
    catalog = DefinitionCatalog((bound.definition,))
    assert resolve_measurement_definition(catalog, bound.definition_key) is bound.definition


def test_pipeline_cancel_after_software_preflight_never_closes_unknown_session():
    item = harness(points=1)
    plan = compile_pipeline(minimal_pipeline(item))
    ready = threading.Event()
    release = threading.Event()
    original_preflight = plan.preflight

    def paused_preflight(context):
        prepared = original_preflight(context)
        ready.set()
        release.wait(1.0)
        return prepared

    plan = RunPlan(**{**plan.__dict__, "preflight": paused_preflight})
    arbiter = ResourceArbiter(MemoryQuarantineJournal())
    handle = RunController(arbiter).start(plan)
    assert ready.wait(1.0)
    handle.cancel()
    release.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert item.camera.session_id == ""
    assert not item.camera.closed
    assert not arbiter.quarantine_records()


def test_compiled_pipeline_plan_is_reusable_sequentially():
    item = harness(points=1)
    plan = compile_pipeline(minimal_pipeline(item))
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    first = controller.start(plan).result(2.0)
    second = controller.start(plan).result(2.0)
    assert isinstance(first, PipelineResult) and isinstance(second, PipelineResult)
    assert np.array_equal(first.dataset.block.values, second.dataset.block.values)
    assert first.dataset.provenance.generation != second.dataset.provenance.generation
    assert first.run_id != second.run_id
    assert len(first.chain_contract_digest) == 64
    assert first.chain_contract_digest == second.chain_contract_digest

    # Equal cardinality/content cannot authorize cross-run dataset substitution:
    # the result mint checks the opaque terminal reservation owned by readiness.
    from zlc_neutral_atom.runtime.pipeline import _PIPELINE_RESULT_TOKEN

    with pytest.raises(RuntimeError, match="another exact terminal consumer"):
        PipelineResult(
            _PIPELINE_RESULT_TOKEN,
            first.dataset,
            second._capture_completion,
            first.aggregate_peak_bytes,
            first.memory_profile_fingerprint,
        )


def test_compiled_pipeline_contends_on_one_physical_device():
    item = harness(points=1)
    item.camera.block_read.set()
    plan = compile_pipeline(minimal_pipeline(item))
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    first = controller.start(plan)
    assert item.camera.read_entered.wait(1.0)
    with pytest.raises(RunStartRejected):
        controller.start(plan)
    first.cancel()
    with pytest.raises(RunFailed):
        first.result(2.0)


def test_pipeline_attempts_physical_cleanup_when_builder_close_also_fails(monkeypatch):
    item = harness(points=1)
    item.camera.terminal_joined = False
    item.camera.close_failure = True
    close_calls = []

    def fail_builder_close(_builder):
        close_calls.append(True)
        raise RuntimeError("builder close failed")

    monkeypatch.setattr(DatasetBuilder, "close", fail_builder_close)
    arbiter = ResourceArbiter(MemoryQuarantineJournal())
    plan = compile_pipeline(minimal_pipeline(item))
    with pytest.raises(RunFailed):
        RunController(arbiter).start(plan).result(2.0)
    assert close_calls
    assert item.camera.close_attempted
    assert item.port.device.key in {record.key for record in arbiter.quarantine_records()}
