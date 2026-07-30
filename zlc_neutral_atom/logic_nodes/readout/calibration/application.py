"""Application request for one direct CaptureArtifact calibration Run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from zlc_neutral_atom.capture.artifact import CaptureArtifact
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey

from .calibration import CalibrationAnalysisRequest

if TYPE_CHECKING:
    from .analysis import CalibrationComputation


_CALIBRATION_RUN_DEADLINE_SECONDS = 300.0


@dataclass(frozen=True)
class CalibrationArtifactRequest:
    """Freeze one persisted capture, its binding, and calibration intent."""

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
    """Recover the exact editable request owned by one loaded computation.

    Artifact internals remain a readout-domain concern.  A GUI may edit the
    returned request, but it never reconstructs source binding or analysis
    intent outside the artifact/report pair itself.
    """

    from .analysis import CalibrationComputation

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("computation must be CalibrationComputation")
    artifact = computation.artifact
    return CalibrationArtifactRequest(
        artifact.source_binding.source_capture_ref,
        artifact.frame_contract.binding,
        computation.report.request,
    )


def build_calibration_artifact_request(
    source: CaptureArtifact,
    analysis: CalibrationAnalysisRequest,
) -> CalibrationArtifactRequest:
    """Bind calibration intent to one loaded capture's physical identity."""

    if not isinstance(source, CaptureArtifact):
        raise TypeError("source must be CaptureArtifact")
    if not isinstance(analysis, CalibrationAnalysisRequest):
        raise TypeError("analysis must be CalibrationAnalysisRequest")
    return CalibrationArtifactRequest(
        source.ref,
        source.camera_provenance.binding,
        analysis,
    )


def prepare_calibration_artifact_plan(
    request: CalibrationArtifactRequest,
    *,
    captures_root: Path,
    calibrations_root: Path,
    on_committed: Callable[[CalibrationArtifactRef], None] | None = None,
) -> RunPlan:
    """Compile one calibration request without exposing its physical join."""

    if not isinstance(request, CalibrationArtifactRequest):
        raise TypeError("request must be CalibrationArtifactRequest")
    from .repository import compile_calibration_artifact_plan

    if not isinstance(captures_root, Path) or not captures_root.is_absolute():
        raise ValueError("captures_root must be an absolute Path")
    if not isinstance(calibrations_root, Path) or not calibrations_root.is_absolute():
        raise ValueError("calibrations_root must be an absolute Path")
    if on_committed is not None and not callable(on_committed):
        raise TypeError("on_committed must be callable or None")
    return compile_calibration_artifact_plan(
        request.source_capture_ref,
        captures_root,
        calibrations_root,
        request.analysis,
        expected_readout_binding=request.readout_binding,
        timeout_seconds=_CALIBRATION_RUN_DEADLINE_SECONDS,
        on_committed=on_committed,
    )


__all__ = [
    "CalibrationArtifactRequest",
    "build_calibration_artifact_request",
    "calibration_request_from_computation",
    "prepare_calibration_artifact_plan",
]
