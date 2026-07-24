"""Prepared monitor-only Camera -> Occupancy application boundary.

The Workbench resolves which row/output the operator selected.  This module
owns every physical consequence of that choice: the Camera output contract,
calibration admission identity, model selection, output vocabulary, and the
classification of one immutable monitor revision.  A desktop shell may host
the pure evaluation on a worker and namespace the returned outputs; it never
reconstructs an occupancy schema or carries a calibration resolver callback.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import OwnedSnapshot
from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from zlc_neutral_atom.readout.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
from zlc_neutral_atom.readout.occupancy import (
    OCCUPANCY_LIVE_OUTPUT_NAMES,
    ReactiveOccupancyMonitorEvaluation,
    evaluate_reactive_occupancy_monitor,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_storage import canonical_text, sha256_text


@dataclass(frozen=True, slots=True)
class ReactiveOccupancyMonitorRequest:
    """One admitted Camera output and calibration identity.

    Console row names are deliberately absent.  They are Workbench routing
    identities, not neutral-atom physics.  The request instead freezes the
    Camera application's own typed request and bare ``frame_i`` output name.
    """

    camera_request: CameraMeasurementRequest
    camera_output_name: str
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_request, CameraMeasurementRequest):
            raise TypeError("camera_request must be CameraMeasurementRequest")
        output_name = canonical_text(
            self.camera_output_name,
            "camera_output_name",
        )
        if output_name not in self.camera_request.output_names:
            raise ValueError(
                "camera_output_name is absent from the Camera Measurement request"
            )
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")
        object.__setattr__(self, "camera_output_name", output_name)


class PreparedReactiveOccupancyMonitor:
    """Closed, replayable evaluation over Camera monitor revisions.

    Repository admission happens once while preparing this value.  ``evaluate``
    is then a deterministic application operation over an immutable source
    snapshot and its exact monitor coverage/event identity.  It owns no QWidget,
    executor, subscription, mutable latest state, or device session.
    """

    __slots__ = ("_calibration", "_request", "_selected_model_kind")

    def __init__(
        self,
        request: ReactiveOccupancyMonitorRequest,
        calibration: ResolvedCalibration,
    ) -> None:
        if not isinstance(request, ReactiveOccupancyMonitorRequest):
            raise TypeError(
                "request must be ReactiveOccupancyMonitorRequest"
            )
        if type(calibration) is not ResolvedCalibration:
            raise TypeError("calibration must be an admitted ResolvedCalibration")
        calibration._require_authority()
        if calibration.reference != request.calibration_ref:
            raise ValueError(
                "admitted calibration differs from the frozen application request"
            )
        expected_binding = ReadoutBindingKey(
            request.camera_request.camera_ref.role
        )
        if calibration.artifact.frame_contract.binding != expected_binding:
            raise ValueError(
                "Camera Measurement role differs from the calibration readout binding"
            )
        selected = calibration.artifact.select_model(request.model_kind)
        self._request = request
        self._calibration = calibration
        self._selected_model_kind = selected.kind

    @property
    def request(self) -> ReactiveOccupancyMonitorRequest:
        return self._request

    @property
    def output_names(self) -> tuple[str, ...]:
        return OCCUPANCY_LIVE_OUTPUT_NAMES

    @property
    def selected_model_kind(self) -> ReadoutModelKind:
        return self._selected_model_kind

    def evaluate(
        self,
        source: OwnedSnapshot,
        coverage: MonitorCoverage,
        *,
        source_event_digest: str,
    ) -> ReactiveOccupancyMonitorEvaluation:
        """Classify one already-routed Camera revision atomically."""

        if not isinstance(source, OwnedSnapshot):
            raise TypeError("source must be OwnedSnapshot")
        if not isinstance(coverage, MonitorCoverage):
            raise TypeError(
                "reactive Occupancy requires a Camera monitor revision"
            )
        digest = sha256_text(source_event_digest, "source_event_digest")
        assert digest is not None
        evaluation = evaluate_reactive_occupancy_monitor(
            source,
            self._calibration,
            coverage,
            model_kind=self._selected_model_kind,
            source_event_digest=digest,
        )
        if tuple(evaluation.outputs) != self.output_names:
            raise RuntimeError(
                "reactive Occupancy application returned another output vocabulary"
            )
        if any(
            not isinstance(output, LiveDatasetOutput)
            for output in evaluation.outputs.values()
        ):
            raise RuntimeError(
                "reactive Occupancy application returned a non-live output"
            )
        return evaluation


def prepare_reactive_occupancy_monitor(
    request: ReactiveOccupancyMonitorRequest,
    calibration: ResolvedCalibration,
) -> PreparedReactiveOccupancyMonitor:
    """Freeze one admitted monitor-only Occupancy application."""

    return PreparedReactiveOccupancyMonitor(request, calibration)


__all__ = [
    "PreparedReactiveOccupancyMonitor",
    "ReactiveOccupancyMonitorRequest",
    "prepare_reactive_occupancy_monitor",
]
