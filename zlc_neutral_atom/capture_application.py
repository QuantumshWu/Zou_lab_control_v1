"""Concrete finite-capture application seam shared by notebook and Workbench."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable
from uuid import uuid4

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.bootstrap._camera_endpoint import (
    CameraCaptureBindingRequest,
    bind_camera_measurement,
)
from zlc_neutral_atom.bootstrap._triggered_capture import (
    TriggeredCameraBinding,
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.camera_measurement import (
    CameraMeasurementDescriptor,
    CameraMeasurementRequest,
    camera_measurement_final_outputs,
)
from zlc_neutral_atom.dataset_output import (
    LiveDatasetOutput,
    single_live_dataset_output,
)
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
    DatasetPreviewSnapshot,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.pipeline import (
    CapturePreviewPort,
    CapturePreviewSpec,
    MinimalPipelineSpec,
    _notify_preview_failure,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.timing.capture import TriggeredCaptureSpec
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_pulse import PulseDocument, PulseExecutionForm
from zlc_storage import canonical_text, positive_integer


_CAPTURE_REPEAT_AXIS_ID = AxisId("capture.repeat")
_CAPTURE_SCAN_AXIS_ID = AxisId("capture.scan_row_ordinal")
CAPTURE_READOUT_EVENT_AXIS_ID = AxisId("capture.readout_event")
_CAMERA_REPEAT_AXIS_ID = AxisId("camera.repeat")
_CAMERA_READOUT_EVENT_AXIS_ID = AxisId("camera.readout_event")


@dataclass(frozen=True)
class CaptureRequest:
    pulse_document: PulseDocument
    execution_form: PulseExecutionForm
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    trigger_channel: str | None = None
    repeat_count: int = 1
    readout_events_per_repeat: int | None = None
    within_point_grouping: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if self.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
            raise ValueError("CaptureRequest requires a finite pulse execution form")
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        if self.trigger_channel is not None:
            canonical_text(self.trigger_channel, "trigger_channel")
        object.__setattr__(
            self,
            "repeat_count",
            positive_integer(self.repeat_count, "repeat_count"),
        )
        if self.readout_events_per_repeat is not None:
            object.__setattr__(
                self,
                "readout_events_per_repeat",
                positive_integer(
                    self.readout_events_per_repeat,
                    "readout_events_per_repeat",
                ),
            )
        if self.within_point_grouping is not None:
            try:
                grouping = tuple(tuple(pair) for pair in self.within_point_grouping)
            except TypeError as exc:
                raise TypeError(
                    "within_point_grouping must be an iterable of pairs"
                ) from exc
            object.__setattr__(self, "within_point_grouping", grouping)


@dataclass(frozen=True)
class PlanDescriptor:
    name: str
    camera_role: str
    sequencer_role: str
    execution_form: PulseExecutionForm
    trigger_channel: str
    expected_frames: int
    output_schema: DatasetSchema
    compiled_pulse_digest: str
    resource_claims: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapturePreviewImageContext:
    """Capture-owned physical image axes for one exact preview Dataset."""

    schema: DatasetSchema
    y_axis: AxisSpec
    x_axis: AxisSpec

    def __post_init__(self) -> None:
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if not isinstance(self.y_axis, AxisSpec) or self.y_axis.role != SPATIAL_Y:
            raise ValueError("y_axis must be the declared SPATIAL_Y axis")
        if not isinstance(self.x_axis, AxisSpec) or self.x_axis.role != SPATIAL_X:
            raise ValueError("x_axis must be the declared SPATIAL_X axis")
        data_axes = self.schema.cell_schema.data_axes
        if (
            len(data_axes) != 2
            or {axis.axis_id for axis in data_axes}
            != {self.y_axis.axis_id, self.x_axis.axis_id}
        ):
            raise ValueError(
                "preview image axes must exactly cover the Dataset data axes"
            )


class _PreparedExactCapture:
    """Shared one-shot UI command for the two exact camera ownership modes."""

    __slots__ = (
        "_capture",
        "_descriptor",
        "_lock",
        "_one_shot_name",
        "_pipeline",
        "_preview_block_id",
        "_preview_edge",
        "_preview_schema",
        "_repository",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        capture: MinimalPipelineSpec | TriggeredCaptureSpec,
        repository: CaptureRepository,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor,
        *,
        one_shot_name: str,
    ) -> None:
        if not isinstance(capture, (MinimalPipelineSpec, TriggeredCaptureSpec)):
            raise TypeError("capture must be an exact camera pipeline spec")
        if type(repository) is not CaptureRepository:
            raise TypeError("repository must be CaptureRepository")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._capture = capture
        self._pipeline = (
            capture.capture if isinstance(capture, TriggeredCaptureSpec) else capture
        )
        self._repository = repository
        self._start_run = start_run
        self._descriptor = descriptor
        self._one_shot_name = canonical_text(one_shot_name, "one_shot_name")
        self._preview_block_id = BlockId(
            f"capture-preview-{uuid4().hex}"
        )
        self._preview_edge = CapturePreviewSpec.dataset_edge_for_capture(
            self._pipeline
        )
        self._lock = threading.Lock()
        self._preview_schema: DatasetSchema | None = None
        self._started = False

    @property
    def descriptor(self) -> PlanDescriptor:
        return self._descriptor

    @property
    def preview_schema(self) -> DatasetSchema:
        with self._lock:
            if self._preview_schema is not None:
                return self._preview_schema
            schema = self._pipeline.measurement.capture_contract.dataset_schema
            readout_axes = tuple(
                axis for axis in schema.point_axes if axis.role == READOUT_EVENT
            )
            if (
                len(readout_axes) != 1
                or len(schema.point_axes) != 1
                or schema.point_layout.storage_size != readout_axes[0].size
            ):
                raise ValueError(
                    "finite Camera preview requires one explicit READOUT_EVENT "
                    "axis and no scan-point multiplexing"
                )
            self._preview_schema = self._preview_edge.schema
            return self._preview_schema

    @property
    def preview_image_context(self) -> CapturePreviewImageContext:
        """Return the closed physical image contract; never infer it in a GUI."""

        schema = self.preview_schema
        axes = schema.cell_schema.data_axes
        y_axes = tuple(axis for axis in axes if axis.role == SPATIAL_Y)
        x_axes = tuple(axis for axis in axes if axis.role == SPATIAL_X)
        if len(axes) != 2 or len(y_axes) != 1 or len(x_axes) != 1:
            raise ValueError(
                "finite capture preview requires declared SPATIAL_Y, SPATIAL_X "
                "axes covering the complete physical data cell"
            )
        return CapturePreviewImageContext(schema, y_axes[0], x_axes[0])

    def start(self) -> RunHandle:
        self._claim_start()
        plan = compile_capture_artifact_pipeline(
            self._capture,
            self._repository,
        )
        return self._start_run(plan)

    def start_with_preview(
        self,
        *,
        factory: Callable[[CapturePreviewSpec], CapturePreviewPort],
        source_ordinals: tuple[int, ...] | None = None,
    ) -> RunHandle:
        """Start once, optionally publishing only named physical frame ordinals."""

        if not callable(factory):
            raise TypeError("factory must be callable")
        self.preview_schema
        self._claim_start()
        preview_spec = CapturePreviewSpec(
            self._preview_block_id,
            self._preview_edge,
            source_ordinals,
        )
        preview = factory(preview_spec)
        plan = compile_capture_artifact_pipeline(
            self._capture,
            self._repository,
            preview=preview,
        )
        try:
            return self._start_run(plan)
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise

    def _claim_start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError(f"{self._one_shot_name} is one-shot")
            self._started = True


class PreparedFiniteCapture(_PreparedExactCapture):
    """Explicit pulse-owned finite Capture command."""

    def __init__(
        self,
        triggered: TriggeredCaptureSpec,
        repository: CaptureRepository,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: PlanDescriptor,
    ) -> None:
        if not isinstance(triggered, TriggeredCaptureSpec):
            raise TypeError("triggered must be TriggeredCaptureSpec")
        if not isinstance(descriptor, PlanDescriptor):
            raise TypeError("descriptor must be PlanDescriptor")
        super().__init__(
            triggered,
            repository,
            start_run,
            descriptor,
            one_shot_name="PreparedFiniteCapture",
        )


class PreparedFiniteCameraMeasurement(_PreparedExactCapture):
    """Passive finite form of the one public Camera Measurement."""

    def __init__(
        self,
        pipeline: MinimalPipelineSpec,
        repository: CaptureRepository,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: CameraMeasurementDescriptor,
        request: CameraMeasurementRequest,
    ) -> None:
        if not isinstance(pipeline, MinimalPipelineSpec):
            raise TypeError("pipeline must be MinimalPipelineSpec")
        if not isinstance(descriptor, CameraMeasurementDescriptor):
            raise TypeError("descriptor must be CameraMeasurementDescriptor")
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("request must be CameraMeasurementRequest")
        self._request = request
        super().__init__(
            pipeline,
            repository,
            start_run,
            descriptor,
            one_shot_name="PreparedFiniteCameraMeasurement",
        )

    @property
    def live_preview_output_name(self) -> str | None:
        """Return the only honest capacity-one Camera preview name, if any."""

        if self._request.frames_per_cycle != 1:
            return None
        return self._request.output_names[0]

    def live_dataset_outputs(
        self,
        frozen: DatasetPreviewSnapshot | MonitorDatasetSnapshot,
    ) -> dict[str, LiveDatasetOutput]:
        """Publish the finite preview under the request-owned Camera name."""

        output_name = self.live_preview_output_name
        if output_name is None:
            raise RuntimeError(
                "a capacity-one preview cannot identify a multi-frame Camera cycle"
            )
        output = single_live_dataset_output(output_name, frozen)
        return {output.name: output}

    def final_dataset_outputs(self, reference: CaptureArtifactRef):
        """Materialize the request-owned Camera outputs from its FINAL ref."""

        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("Camera FINAL result must be CaptureArtifactRef")
        source = self._repository.materialize_final(reference)
        return camera_measurement_final_outputs(
            reference,
            source,
            self._request,
        )


def bind_finite_capture_spec(
    *,
    binding: TriggeredCameraBinding,
    block_id: BlockId,
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    execution_form: PulseExecutionForm,
    name_prefix: str,
) -> tuple[TriggeredCaptureSpec, PlanDescriptor]:
    """Freeze the shared exact plan inputs after use-case intent is complete."""

    pipeline = MinimalPipelineSpec(
        f"{name_prefix} {binding.pulse_request.document.name}",
        binding.measurement,
        block_id,
    )
    triggered = TriggeredCaptureSpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    descriptor = PlanDescriptor(
        pipeline.name,
        camera_ref.role,
        sequencer_ref.role,
        execution_form,
        binding.trigger_channel,
        binding.expected_frames,
        binding.measurement.capture_contract.dataset_schema,
        binding.compiled_artifact.fingerprint,
        (
            str(binding.pulse_port.resource_claim.key),
            str(binding.measurement.capture_port.resource_claim.key),
        ),
    )
    return triggered, descriptor


def prepare_finite_capture(
    request: CaptureRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    repository: CaptureRepository,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedFiniteCapture:
    """Bind one ordinary finite request into a narrow one-shot command."""

    binding = bind_finite_capture_request(
        request,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )
    triggered, descriptor = bind_finite_capture_spec(
        binding=binding,
        block_id=BlockId(f"capture-{binding.compiled_artifact.fingerprint[:20]}"),
        camera_ref=request.camera_ref,
        sequencer_ref=request.sequencer_ref,
        execution_form=request.execution_form,
        name_prefix="Capture",
    )
    return PreparedFiniteCapture(triggered, repository, start_run, descriptor)


def bind_finite_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    camera_port: BoundCapturePort,
) -> tuple[MinimalPipelineSpec, CameraMeasurementDescriptor]:
    """Bind ``repeat=K`` Camera to K×E passive hardware-triggered frames."""

    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("request must be CameraMeasurementRequest")
    repeats = positive_integer(request.repeat, "repeat")
    events = positive_integer(request.frames_per_cycle, "frames_per_cycle")
    capability = camera_port.capability
    facts = capability.camera_capability_evidence.physical_facts
    repeat_axis = AxisSpec(
        _CAMERA_REPEAT_AXIS_ID,
        "repeat",
        REPEAT,
        repeats,
        tuple(range(repeats)),
    )
    event_axis = AxisSpec(
        _CAMERA_READOUT_EVENT_AXIS_ID,
        "readout event",
        READOUT_EVENT,
        events,
        tuple(range(events)),
    )
    point_layout = PointLayout.rect_c((events,))
    schema = DatasetSchema(
        repeat_axis,
        (event_axis,),
        point_layout,
        capability.payload_contract.value_schema,
    )
    schedule = DatasetCellSchedule.from_cells(
        schema,
        (
            DatasetCellAddress(repeat_index, event_index)
            for repeat_index in range(repeats)
            for event_index in range(events)
        ),
    )
    measurement = bind_camera_measurement(
        camera_port,
        CameraCaptureBindingRequest(
            request.camera_ref.role,
            repeat_axis,
            (event_axis,),
            point_layout,
            schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            tuple(facts.event_setting(index) for index in range(events)),
        ),
    )
    pipeline = MinimalPipelineSpec(
        f"Camera {request.camera_ref.role}",
        measurement,
        BlockId(f"camera-{schema.fingerprint[:20]}"),
    )
    descriptor = CameraMeasurementDescriptor(
        "Camera",
        request.camera_ref.role,
        schema,
        str(camera_port.resource_claim.key),
    )
    return pipeline, descriptor


def prepare_finite_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    camera_port: BoundCapturePort,
    repository: CaptureRepository,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedFiniteCameraMeasurement:
    """Prepare the finite branch of the one public Camera Measurement."""

    pipeline, descriptor = bind_finite_camera_measurement(
        request,
        camera_port=camera_port,
    )
    return PreparedFiniteCameraMeasurement(
        pipeline,
        repository,
        start_run,
        descriptor,
        request,
    )


def bind_finite_capture_request(
    request: CaptureRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> TriggeredCameraBinding:
    """Bind the physical camera/pulse source shared by capture and processors."""

    if not isinstance(request, CaptureRequest):
        raise TypeError("request must be CaptureRequest")
    return bind_triggered_camera_acquisition(
        pulse_port,
        camera_port,
        pulse_document=request.pulse_document,
        execution_form=request.execution_form,
        trigger_channel=request.trigger_channel,
        layout=TriggeredCameraLayout(
            repeat_axis=AxisSpec(
                _CAPTURE_REPEAT_AXIS_ID,
                "repeat",
                REPEAT,
                request.repeat_count,
                tuple(range(request.repeat_count)),
            ),
            readout_event_axis_id=CAPTURE_READOUT_EVENT_AXIS_ID,
            ordinal_scan_axis_id=_CAPTURE_SCAN_AXIS_ID,
            readout_events_per_repeat=request.readout_events_per_repeat,
            within_point_grouping=request.within_point_grouping,
        ),
    )


__all__ = [
    "CAPTURE_READOUT_EVENT_AXIS_ID",
    "CapturePreviewImageContext",
    "CaptureRequest",
    "PlanDescriptor",
    "PreparedFiniteCapture",
    "PreparedFiniteCameraMeasurement",
    "bind_finite_camera_measurement",
    "bind_finite_capture_request",
    "bind_finite_capture_spec",
    "prepare_finite_camera_measurement",
]
