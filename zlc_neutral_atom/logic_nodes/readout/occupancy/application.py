"""One finite camera Measurement -> Occupancy Processor application seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zlc_data import READOUT_EVENT, AxisId
from zlc_neutral_atom.capture.artifact import (
    AdmittedCapture,
    CaptureRepository,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ResolvedCalibration,
)
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
    "resolve_occupancy_calibration_input",
]


def resolve_occupancy_calibration_input(
    binding,
    *,
    resolve_final_or_saved: Callable[..., object],
    load_saved_calibration: Callable[[object], object],
) -> CalibrationArtifactRef:
    """Resolve one saved-or-FINAL Calibration reference exactly once."""

    if not callable(resolve_final_or_saved) or not callable(load_saved_calibration):
        raise TypeError("Occupancy artifact resolvers must be callable")

    def extract_reference(loaded: object) -> CalibrationArtifactRef:
        if type(loaded) is not ResolvedCalibration:
            raise TypeError("saved Calibration loader returned another value type")
        return loaded.reference

    reference = resolve_final_or_saved(
        binding,
        load_saved=load_saved_calibration,
        extract_reference=extract_reference,
    )
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("Occupancy Calibration input resolved another artifact type")
    return reference


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
        source.reference,
        calibration.reference,
        binding,
        event_columns[0].coordinate_id,
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
    from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
        CalibrationRepository,
    )
    from .repository import (
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
