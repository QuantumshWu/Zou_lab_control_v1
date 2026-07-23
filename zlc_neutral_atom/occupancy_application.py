"""One finite camera Measurement -> occupancy StreamProcessor application seam."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import BlockId, DatasetSchema
from zlc_neutral_atom.bootstrap._triggered_capture import TriggeredCameraBinding
from zlc_neutral_atom.readout.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_reference import (
    calibration_artifact_ref_to_tree,
)
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

__all__ = [
    "PreparedFiniteOccupancy",
    "prepare_finite_camera_occupancy",
    "prepare_finite_occupancy",
]


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


def _bind_occupancy_pipeline(
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
    pipeline, counts_schema, occupied_schema = _bind_occupancy_pipeline(
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

    pipeline, counts_schema, occupied_schema = _bind_occupancy_pipeline(
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
