"""Concrete finite-capture application seam shared by notebook and Workbench."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

from zlc_data import AxisId, AxisSpec, BlockId, DatasetSchema, READOUT_EVENT, REPEAT
from zlc_neutral_atom.artifacts import CaptureRepository, compile_capture_artifact_pipeline
from zlc_neutral_atom.bootstrap._triggered_capture import (
    TriggeredCameraBinding,
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime.capture import BoundCapturePort
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
from zlc_storage import canonical_text, positive_integer, positive_real


_CAPTURE_REPEAT_AXIS_ID = AxisId("capture.repeat")
_CAPTURE_SCAN_AXIS_ID = AxisId("capture.scan_row_ordinal")
CAPTURE_READOUT_EVENT_AXIS_ID = AxisId("capture.readout_event")


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
    timeout_seconds: float = 30.0

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
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class PlanDescriptor:
    name: str
    camera_role: str
    sequencer_role: str
    execution_form: PulseExecutionForm
    trigger_channel: str
    expected_frames: int
    output_shape: tuple[int, ...]
    output_schema_fingerprint: str
    compiled_pulse_digest: str
    resource_claims: tuple[str, ...]


class PreparedFiniteCapture:
    """One-shot app command that exposes no raw drive-capable Port."""

    __slots__ = (
        "_descriptor",
        "_lock",
        "_preview_edge",
        "_preview_schema",
        "_repository",
        "_start_run",
        "_started",
        "_triggered",
    )

    def __init__(
        self,
        triggered: TriggeredCaptureSpec,
        repository: CaptureRepository,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: PlanDescriptor,
    ) -> None:
        if not isinstance(triggered, TriggeredCaptureSpec):
            raise TypeError("triggered must be TriggeredCaptureSpec")
        if type(repository) is not CaptureRepository:
            raise TypeError("repository must be CaptureRepository")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        if not isinstance(descriptor, PlanDescriptor):
            raise TypeError("descriptor must be PlanDescriptor")
        self._triggered = triggered
        self._repository = repository
        self._start_run = start_run
        self._descriptor = descriptor
        self._preview_edge = CapturePreviewSpec.dataset_edge_for_capture(
            triggered.capture
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
            schema = self._triggered.capture.measurement.capture_contract.dataset_schema
            readout_axes = tuple(
                axis for axis in schema.point_axes if axis.role == READOUT_EVENT
            )
            if (
                len(readout_axes) != 1
                or readout_axes[0].size != 1
                or schema.point_layout.storage_size != 1
            ):
                raise ValueError(
                    "finite capture Workbench preview requires exactly one readout "
                    "event per repeat and no scan-point multiplexing"
                )
            self._preview_schema = self._preview_edge.schema
            return self._preview_schema

    def start(self) -> RunHandle:
        self._claim_start()
        plan = compile_capture_artifact_pipeline(
            self._triggered,
            self._repository,
        )
        return self._start_run(plan)

    def start_with_preview(
        self,
        *,
        block_id: BlockId,
        factory: Callable[[CapturePreviewSpec], CapturePreviewPort],
    ) -> RunHandle:
        if not isinstance(block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.preview_schema
        self._claim_start()
        preview_spec = CapturePreviewSpec(
            block_id,
            self._preview_edge,
        )
        preview = factory(preview_spec)
        plan = compile_capture_artifact_pipeline(
            self._triggered,
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
                raise RuntimeError("PreparedFiniteCapture is one-shot")
            self._started = True


def bind_finite_capture_spec(
    *,
    binding: TriggeredCameraBinding,
    block_id: BlockId,
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    execution_form: PulseExecutionForm,
    timeout_seconds: float,
    name_prefix: str,
) -> tuple[TriggeredCaptureSpec, PlanDescriptor]:
    """Freeze the shared exact plan inputs after use-case intent is complete."""

    pipeline = MinimalPipelineSpec(
        f"{name_prefix} {binding.pulse_request.document.name}",
        binding.measurement,
        block_id,
        timeout_seconds=timeout_seconds,
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
        binding.measurement.capture_contract.dataset_schema.physical_shape,
        binding.measurement.capture_contract.dataset_schema.fingerprint,
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

    if not isinstance(request, CaptureRequest):
        raise TypeError("request must be CaptureRequest")
    binding = bind_triggered_camera_acquisition(
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
    triggered, descriptor = bind_finite_capture_spec(
        binding=binding,
        block_id=BlockId(f"capture-{binding.compiled_artifact.fingerprint[:20]}"),
        camera_ref=request.camera_ref,
        sequencer_ref=request.sequencer_ref,
        execution_form=request.execution_form,
        timeout_seconds=request.timeout_seconds,
        name_prefix="Capture",
    )
    return PreparedFiniteCapture(triggered, repository, start_run, descriptor)


__all__ = [
    "CAPTURE_READOUT_EVENT_AXIS_ID",
    "CaptureRequest",
    "PlanDescriptor",
    "PreparedFiniteCapture",
    "bind_finite_capture_spec",
]
