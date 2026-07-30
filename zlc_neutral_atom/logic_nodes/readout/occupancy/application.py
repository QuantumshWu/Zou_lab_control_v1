"""One finite camera Measurement -> Occupancy Processor application seam."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zlc_data import READOUT_EVENT, AxisId
from zlc_neutral_atom.capture.artifact import (
    CaptureArtifact,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.runtime.run import RunPlan


_DETECTION_RUN_DEADLINE_SECONDS = 300.0

__all__ = [
    "build_detection_request",
    "DetectionRequest",
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
    source: CaptureArtifact,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> DetectionRequest:
    """Bind committed capture and calibration through their declared axes."""

    if not isinstance(source, CaptureArtifact):
        raise TypeError("source must be CaptureArtifact")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be ResolvedCalibration")
    artifact = source
    calibration_artifact = calibration.artifact
    binding = artifact.camera_provenance.binding
    if calibration_artifact.frame_contract.binding != binding:
        raise ValueError(
            "capture and calibration name different readout bindings"
        )
    event_columns = tuple(
        column
        for column in artifact.frame_source.schema.point_table.columns
        if column.role == READOUT_EVENT
    )
    if len(event_columns) != 1 or set(event_columns[0].values) != {0}:
        raise ValueError(
            "detection requires exactly one singleton READOUT_EVENT point column"
        )
    selected_model = calibration_artifact.select_model(model_kind)
    return DetectionRequest(
        source.ref,
        calibration.reference,
        binding,
        event_columns[0].coordinate_id,
        selected_model.kind,
    )


def prepare_detection_plan(
    request: DetectionRequest,
    *,
    captures_root: Path,
    calibrations_root: Path,
    occupancy_root: Path,
) -> RunPlan:
    """Compile committed-capture Occupancy from one complete request."""

    if not isinstance(request, DetectionRequest):
        raise TypeError("request must be DetectionRequest")
    from .artifact import compile_occupancy_artifact_plan

    for field, value in (
        ("captures_root", captures_root),
        ("calibrations_root", calibrations_root),
        ("occupancy_root", occupancy_root),
    ):
        if not isinstance(value, Path) or not value.is_absolute():
            raise TypeError(f"{field} must be an absolute Path")
    return compile_occupancy_artifact_plan(
        request.source_capture_ref,
        request.calibration_ref,
        captures_root=captures_root,
        calibrations_root=calibrations_root,
        occupancy_root=occupancy_root,
        expected_readout_binding=request.readout_binding,
        readout_event_axis_id=request.readout_event_axis_id,
        model_kind=request.model_kind,
        timeout_seconds=_DETECTION_RUN_DEADLINE_SECONDS,
    )
