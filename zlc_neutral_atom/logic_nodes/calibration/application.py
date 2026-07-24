"""Application request for one committed-capture calibration Run."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_neutral_atom.logic_nodes.camera_capture.artifact import (
    AdmittedCapture,
    CaptureRepository,
)
from zlc_neutral_atom.logic_nodes.camera_capture.reference import CaptureArtifactRef
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey

from .analysis import CalibrationComputation
from .calibration import CalibrationAnalysisRequest


_CALIBRATION_RUN_DEADLINE_SECONDS = 300.0


@dataclass(frozen=True)
class CalibrationArtifactRequest:
    """Freeze one committed capture, its binding, and calibration intent."""

    source_capture_ref: CaptureArtifactRef
    readout_binding: ReadoutBindingKey
    analysis: CalibrationAnalysisRequest

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")


def calibration_request_from_computation(
    computation: CalibrationComputation,
) -> CalibrationArtifactRequest:
    """Recover the exact editable request owned by one admitted computation.

    Artifact internals remain a readout-domain concern.  A GUI may edit the
    returned request, but it never reconstructs source binding or analysis
    authority from the artifact/report pair itself.
    """

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("computation must be CalibrationComputation")
    artifact = computation.artifact
    return CalibrationArtifactRequest(
        artifact.source_binding.source_capture_ref,
        artifact.frame_contract.binding,
        computation.report.request,
    )


def build_calibration_artifact_request(
    source: AdmittedCapture,
    analysis: CalibrationAnalysisRequest,
) -> CalibrationArtifactRequest:
    """Bind calibration intent to one admitted capture's physical identity."""

    if type(source) is not AdmittedCapture:
        raise TypeError("source must be an admitted capture")
    source._require_authority()
    if not isinstance(analysis, CalibrationAnalysisRequest):
        raise TypeError("analysis must be CalibrationAnalysisRequest")
    return CalibrationArtifactRequest(
        source.reference,
        source.artifact.camera_provenance.binding,
        analysis,
    )


def prepare_calibration_artifact_plan(
    request: CalibrationArtifactRequest,
    *,
    capture_repository: CaptureRepository,
    calibration_repository,
) -> RunPlan:
    """Compile one calibration request without exposing its physical join."""

    if not isinstance(request, CalibrationArtifactRequest):
        raise TypeError("request must be CalibrationArtifactRequest")
    from .repository import (
        CalibrationRepository,
        compile_calibration_artifact_plan,
    )

    if type(capture_repository) is not CaptureRepository:
        raise TypeError("capture_repository must be CaptureRepository")
    if type(calibration_repository) is not CalibrationRepository:
        raise TypeError("calibration_repository must be CalibrationRepository")
    return compile_calibration_artifact_plan(
        request.source_capture_ref,
        capture_repository,
        calibration_repository,
        request.analysis,
        expected_readout_binding=request.readout_binding,
        timeout_seconds=_CALIBRATION_RUN_DEADLINE_SECONDS,
    )


__all__ = [
    "CalibrationArtifactRequest",
    "build_calibration_artifact_request",
    "calibration_request_from_computation",
    "prepare_calibration_artifact_plan",
]
