"""One finite camera Measurement -> occupancy StreamProcessor application seam."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

from zlc_data import READOUT_EVENT, AxisId, BlockId, DatasetSchema
from zlc_neutral_atom.artifacts.capture import AdmittedCapture, CaptureRepository
from zlc_neutral_atom.bootstrap._triggered_capture import TriggeredCameraBinding
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.readout.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
from zlc_neutral_atom.readout.occupancy import (
    OccupancyStreamProcessorSpec,
    resolve_occupancy_stream_schema,
)
from zlc_neutral_atom.readout.occupancy_pipeline import (
    OccupancyPipelineSpec,
    compile_occupancy_pipeline,
)
from zlc_neutral_atom.runtime.pipeline import (
    BoundMeasurement,
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
    _notify_preview_failure,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_neutral_atom.timing.occupancy import (
    TriggeredOccupancySpec,
    compile_triggered_occupancy_pipeline,
)
from zlc_storage import canonical_digest


_DETECTION_RUN_DEADLINE_SECONDS = 300.0

__all__ = [
    "bind_occupancy_pipeline",
    "build_detection_request",
    "DetectionRequest",
    "PreparedFiniteOccupancy",
    "prepare_finite_camera_occupancy",
    "prepare_finite_occupancy",
    "prepare_detection_plan",
]


@dataclass(frozen=True)
class DetectionRequest:
    """Freeze two committed inputs and one concrete occupancy model."""

    source_capture_ref: CaptureArtifactRef
    calibration_ref: CalibrationArtifactRef
    readout_binding: ReadoutBindingKey
    readout_event_axis_id: AxisId
    model_kind: ReadoutModelKind

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be a concrete ReadoutModelKind")


def build_detection_request(
    source: AdmittedCapture,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> DetectionRequest:
    """Bind committed capture and calibration through their declared axes."""

    if type(source) is not AdmittedCapture:
        raise TypeError("source must be an admitted capture")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    source._require_authority()
    calibration._require_authority()
    artifact = source.artifact
    calibration_artifact = calibration.artifact
    binding = artifact.camera_provenance.binding
    if calibration_artifact.frame_contract.binding != binding:
        raise ValueError(
            "capture and calibration name different readout bindings"
        )
    event_axes = tuple(
        axis
        for axis in artifact.frame_source.schema.point_axes
        if axis.role == READOUT_EVENT
    )
    if len(event_axes) != 1 or event_axes[0].size != 1:
        raise ValueError(
            "detection requires exactly one singleton READOUT_EVENT axis"
        )
    selected_model = calibration_artifact.select_model(model_kind)
    return DetectionRequest(
        source.reference,
        calibration.reference,
        binding,
        event_axes[0].axis_id,
        selected_model.kind,
    )


def prepare_detection_plan(
    request: DetectionRequest,
    *,
    capture_repository: CaptureRepository,
    calibration_repository,
    occupancy_repository,
) -> RunPlan:
    """Compile committed-capture Occupancy from one complete request."""

    if not isinstance(request, DetectionRequest):
        raise TypeError("request must be DetectionRequest")
    from zlc_neutral_atom.readout.calibration_repository import (
        CalibrationRepository,
    )
    from zlc_neutral_atom.readout.occupancy_repository import (
        OccupancyRepository,
        compile_occupancy_artifact_plan,
    )

    if type(capture_repository) is not CaptureRepository:
        raise TypeError("capture_repository must be CaptureRepository")
    if type(calibration_repository) is not CalibrationRepository:
        raise TypeError("calibration_repository must be CalibrationRepository")
    if type(occupancy_repository) is not OccupancyRepository:
        raise TypeError("occupancy_repository must be OccupancyRepository")
    return compile_occupancy_artifact_plan(
        request.source_capture_ref,
        capture_repository,
        request.calibration_ref,
        calibration_repository,
        occupancy_repository,
        expected_readout_binding=request.readout_binding,
        readout_event_axis_id=request.readout_event_axis_id,
        model_kind=request.model_kind,
        timeout_seconds=_DETECTION_RUN_DEADLINE_SECONDS,
    )


class PreparedFiniteOccupancy:
    """One-shot flat exact run with an optional provisional counts view."""

    __slots__ = (
        "_counts_schema",
        "_lock",
        "_occupied_schema",
        "_start_run",
        "_started",
        "_spec",
    )

    def __init__(
        self,
        spec: OccupancyPipelineSpec | TriggeredOccupancySpec,
        *,
        counts_schema: DatasetSchema,
        occupied_schema: DatasetSchema,
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(spec, (OccupancyPipelineSpec, TriggeredOccupancySpec)):
            raise TypeError("spec must be an occupancy pipeline spec")
        if not isinstance(counts_schema, DatasetSchema):
            raise TypeError("counts_schema must be DatasetSchema")
        if not isinstance(occupied_schema, DatasetSchema):
            raise TypeError("occupied_schema must be DatasetSchema")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._spec = spec
        self._counts_schema = counts_schema
        self._occupied_schema = occupied_schema
        self._start_run = start_run
        self._lock = threading.Lock()
        self._started = False

    @property
    def counts_schema(self) -> DatasetSchema:
        return self._counts_schema

    @property
    def occupied_schema(self) -> DatasetSchema:
        return self._occupied_schema

    @property
    def preview_spec(self) -> ExactDatasetPreviewSpec:
        return ExactDatasetPreviewSpec(self._counts_schema.fingerprint)

    def start(self) -> RunHandle:
        self._claim_start()
        return self._start_run(self._compile())

    def start_with_preview(
        self,
        *,
        factory: Callable[[ExactDatasetPreviewSpec], ExactDatasetPreviewPort],
    ) -> RunHandle:
        if not callable(factory):
            raise TypeError("factory must be callable")
        self._claim_start()
        preview = factory(self.preview_spec)
        try:
            plan = self._compile(preview=preview)
            return self._start_run(plan)
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise

    def _claim_start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedFiniteOccupancy is one-shot")
            self._started = True

    def _compile(self, *, preview=None) -> RunPlan:
        if isinstance(self._spec, TriggeredOccupancySpec):
            return compile_triggered_occupancy_pipeline(
                self._spec,
                preview=preview,
            )
        return compile_occupancy_pipeline(self._spec, preview=preview)


def bind_occupancy_pipeline(
    measurement: BoundMeasurement,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None,
    timing_identity: object,
    name: str,
) -> tuple[OccupancyPipelineSpec, DatasetSchema, DatasetSchema]:
    if not isinstance(measurement, BoundMeasurement):
        raise TypeError("measurement must be BoundMeasurement")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an exact ResolvedCalibration")
    calibration._require_authority()
    selected_kind = calibration.artifact.select_model(model_kind).kind
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.finite-occupancy",
            "timing": timing_identity,
            "camera_schema": measurement.capture_contract.dataset_schema.fingerprint,
            "camera_arm": measurement.capture_spec.digest,
            "calibration": calibration_artifact_ref_to_tree(
                calibration.reference
            ),
            "model_kind": selected_kind.value,
        }
    )
    processor = OccupancyStreamProcessorSpec(
        calibration,
        StreamId(f"finite-occupancy-{identity}"),
        f"finite-occupancy-{identity}",
        selected_kind,
    )
    resolved = resolve_occupancy_stream_schema(
        processor,
        measurement.capture_contract.dataset_schema,
    )
    pipeline = OccupancyPipelineSpec(
        name,
        measurement,
        processor,
        BlockId(f"occupancy-counts-{identity[:20]}"),
        BlockId(f"occupancy-occupied-{identity[:20]}"),
    )
    return pipeline, resolved.counts_schema, resolved.occupied_schema


def prepare_finite_occupancy(
    binding: TriggeredCameraBinding,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedFiniteOccupancy:
    """Bind one exact source and one admitted calibration into one flat Run."""

    if not isinstance(binding, TriggeredCameraBinding):
        raise TypeError("binding must be TriggeredCameraBinding")
    pipeline, counts_schema, occupied_schema = bind_occupancy_pipeline(
        binding.measurement,
        calibration,
        model_kind=model_kind,
        timing_identity={"compiled_pulse": binding.compiled_artifact.fingerprint},
        name=f"Occupancy capture {binding.pulse_request.document.name}",
    )
    triggered = TriggeredOccupancySpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    return PreparedFiniteOccupancy(
        triggered,
        counts_schema=counts_schema,
        occupied_schema=occupied_schema,
        start_run=start_run,
    )


def prepare_finite_camera_occupancy(
    measurement: BoundMeasurement,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedFiniteOccupancy:
    """Bind occupancy to the passive finite branch of Camera Measurement."""

    pipeline, counts_schema, occupied_schema = bind_occupancy_pipeline(
        measurement,
        calibration,
        model_kind=model_kind,
        timing_identity={"owner": "independent-hardware-trigger"},
        name="Camera occupancy",
    )
    return PreparedFiniteOccupancy(
        pipeline,
        counts_schema=counts_schema,
        occupied_schema=occupied_schema,
        start_run=start_run,
    )
