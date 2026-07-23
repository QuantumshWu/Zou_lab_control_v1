"""One-shot application seam for the release-recapture Measurement."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import BlockId, DatasetSchema
from zlc_neutral_atom.readout.calibration import ResolvedCalibration
from zlc_neutral_atom.readout.calibration_reference import (
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout.coupled_measurements import (
    BoundTemperatureReleaseRecapture,
)
from zlc_neutral_atom.readout.release_recapture_pipeline import (
    ReleaseRecapturePipelineSpec,
    release_recapture_output_schema,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_neutral_atom.timing.release_recapture import (
    TriggeredReleaseRecaptureSpec,
    compile_triggered_release_recapture_pipeline,
)
from zlc_storage import canonical_digest


class PreparedTemperatureReleaseRecapture:
    """One immutable, one-shot flat survival pipeline."""

    __slots__ = ("_schema", "_spec", "_start_run", "_started", "_lock")

    def __init__(
        self,
        spec: TriggeredReleaseRecaptureSpec,
        *,
        output_schema: DatasetSchema,
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(spec, TriggeredReleaseRecaptureSpec):
            raise TypeError("spec must be TriggeredReleaseRecaptureSpec")
        if not isinstance(output_schema, DatasetSchema):
            raise TypeError("output_schema must be DatasetSchema")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._spec = spec
        self._schema = output_schema
        self._start_run = start_run
        self._started = False
        self._lock = threading.Lock()

    @property
    def output_schema(self) -> DatasetSchema:
        return self._schema

    def start(self) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError(
                    "PreparedTemperatureReleaseRecapture is one-shot"
                )
            self._started = True
        return self._start_run(
            compile_triggered_release_recapture_pipeline(self._spec)
        )


def prepare_temperature_release_recapture(
    bound: BoundTemperatureReleaseRecapture,
    calibration: ResolvedCalibration,
    *,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedTemperatureReleaseRecapture:
    if not isinstance(bound, BoundTemperatureReleaseRecapture):
        raise TypeError("bound must be BoundTemperatureReleaseRecapture")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    if calibration.reference != bound.request.calibration_ref:
        raise ValueError("calibration differs from the bound Measurement request")
    selected = calibration.artifact.select_model(bound.request.model_kind)
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.release-recapture",
            "program": bound.program.fingerprint,
            "camera_schema": (
                bound.camera_binding.measurement.capture_contract
                .dataset_schema.fingerprint
            ),
            "calibration": calibration_artifact_ref_to_tree(
                calibration.reference
            ),
            "model_kind": selected.kind.value,
            "per_site": bound.request.per_site,
        }
    )
    pipeline = ReleaseRecapturePipelineSpec(
        f"Temperature release-recapture {bound.program.document.name}",
        bound.camera_binding.measurement,
        calibration,
        selected.kind,
        bound.request.per_site,
        StreamId(f"release-recapture-{identity}"),
        f"release-recapture-{identity}",
        BlockId(f"release-recapture-{identity[:20]}"),
    )
    output_schema = release_recapture_output_schema(pipeline)
    triggered = TriggeredReleaseRecaptureSpec(
        pipeline,
        bound.camera_binding.pulse_port,
        bound.camera_binding.pulse_request,
        bound.camera_binding.trigger_channel,
        bound.camera_binding.cell_plan,
    )
    return PreparedTemperatureReleaseRecapture(
        triggered,
        output_schema=output_schema,
        start_run=start_run,
    )


__all__ = [
    "PreparedTemperatureReleaseRecapture",
    "prepare_temperature_release_recapture",
]
