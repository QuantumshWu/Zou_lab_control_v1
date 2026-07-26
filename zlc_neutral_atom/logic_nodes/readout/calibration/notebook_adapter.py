"""Notebook surface owned by readout Calibration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from zlc_neutral_atom.capture.application import CaptureRequest
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument

from .application import (
    CalibrationArtifactRequest,
    build_calibration_artifact_request,
)
from .calibration import (
    CalibrationAnalysisRequest,
    ReadoutModelKind,
    ResolvedCalibration,
    ThresholdMethod,
)
from .reference import CalibrationArtifactRef
from .sitemap import (
    SitemapAcquisitionProfile,
    SitemapCalibrationRequest,
    build_sitemap_analysis_request,
    build_sitemap_calibration_request,
)
from .task import CalibrationTaskIntent, PreparedCalibrationTask

if TYPE_CHECKING:
    from .analysis import CalibrationComputation, CalibrationReport


class SitemapCalibrationFailed(RuntimeError):
    __slots__ = ("source_capture_ref",)

    def __init__(self, source_capture_ref: CaptureArtifactRef) -> None:
        if not isinstance(source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        self.source_capture_ref = source_capture_ref
        super().__init__(
            "sitemap calibration failed; the committed raw capture remains "
            f"available as {source_capture_ref!r}"
        )


class SitemapCalibrationInterrupted(KeyboardInterrupt):
    __slots__ = ("source_capture_ref",)

    def __init__(self, source_capture_ref: CaptureArtifactRef) -> None:
        if not isinstance(source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        self.source_capture_ref = source_capture_ref
        super().__init__(
            "sitemap calibration interrupted; the committed raw capture remains "
            f"available as {source_capture_ref!r}"
        )


class CalibrationNotebookHost(Protocol):
    def resolve_sitemap_profile(
        self,
        camera_role: str | None,
    ) -> tuple[str, SitemapAcquisitionProfile]: ...

    def available_sitemap_camera_roles(self) -> tuple[str, ...]: ...

    def load_calibration_pulse(
        self,
        value: PulseDocument | str | Path,
    ) -> PulseDocument: ...

    def calibration_camera_ref(self, role: str) -> DeviceRef: ...

    def calibration_sequencer_ref(self, role: str) -> DeviceRef: ...

    def run_calibration_capture(self, request: CaptureRequest) -> CaptureArtifactRef: ...

    def bind_calibration_task(
        self,
        intent: CalibrationTaskIntent,
        application,
    ) -> PreparedCalibrationTask: ...

    def admit_saved_calibration_capture_source(
        self,
        source_path: str | Path,
        *,
        expected_camera_role: str,
    ) -> CaptureArtifactRef: ...

    def write_calibration_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        folder: str | Path,
        frame_export_policy: str,
        expected_camera_role: str | None,
    ) -> None: ...

    def admit_calibration_capture(self, source: CaptureArtifactRef): ...

    def start_calibration_request(
        self,
        request: CalibrationArtifactRequest,
    ) -> RunHandle: ...

    def run_calibration_request(
        self,
        request: CalibrationArtifactRequest,
    ) -> CalibrationArtifactRef: ...

    def open_calibration_request_gui(self, request: CalibrationArtifactRequest): ...

    def open_calibration_edit_gui(self, reference: CalibrationArtifactRef): ...

    def admit_calibration_artifact(
        self,
        reference: CalibrationArtifactRef,
    ) -> ResolvedCalibration: ...

    def admit_saved_calibration_pointer(
        self,
        path: str | Path,
    ) -> ResolvedCalibration: ...

    def read_calibration_computation(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationComputation: ...

    def open_calibration_report_gui(self, reference: CalibrationArtifactRef): ...


class CalibrationNotebookAdapter:
    __slots__ = ()

    @property
    def _calibration_notebook_host(self) -> CalibrationNotebookHost:
        raise NotImplementedError

    def prepare_calibration_task(
        self,
        intent: CalibrationTaskIntent,
    ) -> PreparedCalibrationTask:
        return self._calibration_notebook_host.bind_calibration_task(intent, self)

    def sitemap_camera_roles(self) -> tuple[str, ...]:
        return self._calibration_notebook_host.available_sitemap_camera_roles()

    def sitemap_analysis_request(
        self,
        *,
        camera_role: str | None = None,
        threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
        roi_radius: int | None = None,
    ) -> CalibrationAnalysisRequest:
        _selected_camera, profile = (
            self._calibration_notebook_host.resolve_sitemap_profile(camera_role)
        )
        return build_sitemap_analysis_request(
            profile,
            threshold_method=threshold_method,
            roi_radius=roi_radius,
        )

    def sitemap_request(
        self,
        *,
        frames: int = 20,
        camera_role: str | None = None,
        pulse: PulseDocument | str | Path | None = None,
        reference_exposure_s: float | None = None,
        readout_exposure_s: float | None = None,
        threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
        roi_radius: int | None = None,
    ) -> SitemapCalibrationRequest:
        host = self._calibration_notebook_host
        selected_camera, profile = host.resolve_sitemap_profile(camera_role)
        selected_pulse = None if pulse is None else host.load_calibration_pulse(pulse)
        return build_sitemap_calibration_request(
            profile,
            camera_ref=host.calibration_camera_ref(selected_camera),
            sequencer_ref=host.calibration_sequencer_ref(profile.sequencer_role),
            repeat_groups=frames,
            pulse_document=selected_pulse,
            reference_exposure_s=reference_exposure_s,
            readout_exposure_s=readout_exposure_s,
            threshold_method=threshold_method,
            roi_radius=roi_radius,
        )

    def sitemap(
        self,
        *,
        frames: int = 20,
        camera_role: str | None = None,
    ) -> CalibrationArtifactRef:
        sequence = self.sitemap_request(frames=frames, camera_role=camera_role)
        source = self._calibration_notebook_host.run_calibration_capture(
            sequence.capture_request
        )
        try:
            return self.calibrate(self.calibration_request(source, sequence.analysis))
        except KeyboardInterrupt as error:
            raise SitemapCalibrationInterrupted(source) from error
        except Exception as error:
            raise SitemapCalibrationFailed(source) from error

    def admit_saved_calibration_capture(
        self,
        source_path: str | Path,
        *,
        expected_camera_role: str,
    ) -> CaptureArtifactRef:
        return self._calibration_notebook_host.admit_saved_calibration_capture_source(
            source_path,
            expected_camera_role=expected_camera_role,
        )

    def write_calibration_task_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        folder: str | Path,
        frame_export_policy: str,
        expected_camera_role: str | None = None,
    ) -> None:
        self._calibration_notebook_host.write_calibration_outputs(
            source,
            calibration,
            folder=folder,
            frame_export_policy=frame_export_policy,
            expected_camera_role=expected_camera_role,
        )

    def calibration_request(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
    ) -> CalibrationArtifactRequest:
        request = build_calibration_artifact_request(
            self._calibration_notebook_host.admit_calibration_capture(source),
            analysis,
        )
        self._require_binding(request.readout_binding)
        return request

    def start_calibration(self, request: CalibrationArtifactRequest) -> RunHandle:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        return self._calibration_notebook_host.start_calibration_request(request)

    def start_calibration_analysis(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
    ) -> RunHandle:
        return self.start_calibration(self.calibration_request(source, analysis))

    def calibrate(
        self,
        request: CalibrationArtifactRequest,
    ) -> CalibrationArtifactRef:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        return self._calibration_notebook_host.run_calibration_request(request)

    def calibration_gui(self, request: CalibrationArtifactRequest):
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        return self._calibration_notebook_host.open_calibration_request_gui(request)

    def calibration_edit_gui(self, reference: CalibrationArtifactRef):
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        return self._calibration_notebook_host.open_calibration_edit_gui(reference)

    def load_calibration(
        self,
        reference: CalibrationArtifactRef,
    ) -> ResolvedCalibration:
        return self._calibration_notebook_host.admit_calibration_artifact(reference)

    def load_saved_calibration(
        self,
        calibration_ref_file: str | Path,
    ) -> ResolvedCalibration:
        return self._calibration_notebook_host.admit_saved_calibration_pointer(
            calibration_ref_file
        )

    def load_calibration_computation(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationComputation:
        return self._calibration_notebook_host.read_calibration_computation(reference)

    def load_calibration_report(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationReport:
        return self.load_calibration_computation(reference).report

    def calibration_report_gui(self, reference: CalibrationArtifactRef):
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        return self._calibration_notebook_host.open_calibration_report_gui(reference)


__all__ = [
    "CalibrationNotebookAdapter",
    "CalibrationNotebookHost",
    "SitemapCalibrationFailed",
    "SitemapCalibrationInterrupted",
]
