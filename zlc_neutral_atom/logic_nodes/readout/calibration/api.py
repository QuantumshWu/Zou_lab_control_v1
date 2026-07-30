"""Public Experiment API owned by readout Calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING

from zlc_neutral_atom.capture.application import CaptureRequest, PreparedFiniteCapture
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument

from .application import (
    CalibrationArtifactRequest,
    build_calibration_artifact_request,
)
from .calibration import (
    CalibrationAnalysisRequest,
    ResolvedCalibration,
    ThresholdMethod,
)
from .reference import CalibrationArtifactRef
from .sitemap import (
    DEFAULT_CALIBRATION_PULSE_PATH,
    SitemapAcquisitionProfile,
    SitemapCalibrationRequest,
    build_sitemap_analysis_request,
    build_sitemap_calibration_request,
)
from .task import (
    CalibrationTaskIntent,
    PreparedCalibrationTask,
    prepare_calibration_task,
)

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


class CalibrationApi:
    __slots__ = (
        "_bind_capture",
        "_calibrations_root",
        "_camera_roles",
        "_captures_root",
        "_closed",
        "_current_calibration_ref",
        "_lock",
        "_load_pulse",
        "_open_ui",
        "_operation_guard",
        "_profiles",
        "_resolve_camera_ref",
        "_resolve_camera_role",
        "_resolve_sequencer_ref",
        "_start_calibration_operation",
        "_wait_run",
        "_write_outputs",
    )

    def __init__(
        self,
        *,
        captures_root: Path,
        calibrations_root: Path,
        profiles: Mapping[str, SitemapAcquisitionProfile],
        camera_roles: tuple[str, ...],
        resolve_camera_role: Callable,
        resolve_camera_ref: Callable,
        resolve_sequencer_ref: Callable,
        load_pulse: Callable,
        bind_capture: Callable,
        wait_run: Callable,
        operation_guard: Callable,
        write_outputs: Callable,
        start_calibration: Callable,
        open_ui: Callable,
    ) -> None:
        if not isinstance(captures_root, Path) or not captures_root.is_absolute():
            raise ValueError("captures_root must be an absolute Path")
        if not isinstance(calibrations_root, Path) or not calibrations_root.is_absolute():
            raise ValueError("calibrations_root must be an absolute Path")
        operations = (
            resolve_camera_role,
            resolve_camera_ref,
            resolve_sequencer_ref,
            load_pulse,
            bind_capture,
            wait_run,
            operation_guard,
            write_outputs,
            start_calibration,
            open_ui,
        )
        if any(not callable(operation) for operation in operations):
            raise TypeError("Calibration API operations must be callable")
        self._captures_root = captures_root.resolve()
        self._calibrations_root = calibrations_root.resolve()
        self._profiles = MappingProxyType(dict(profiles))
        self._camera_roles = tuple(camera_roles)
        self._lock = threading.RLock()
        self._closed = False
        self._current_calibration_ref: CalibrationArtifactRef | None = None
        self._resolve_camera_role = resolve_camera_role
        self._resolve_camera_ref = resolve_camera_ref
        self._resolve_sequencer_ref = resolve_sequencer_ref
        self._load_pulse = load_pulse
        self._bind_capture = bind_capture
        self._wait_run = wait_run
        self._operation_guard = operation_guard
        self._write_outputs = write_outputs
        self._start_calibration_operation = start_calibration
        self._open_ui = open_ui

    @property
    def current_calibration_ref(self) -> CalibrationArtifactRef | None:
        """Visible application default; authoritative requests still freeze a ref."""

        with self._operation_guard():
            with self._lock:
                if self._closed:
                    raise RuntimeError("Calibration API is closed")
                return self._current_calibration_ref

    @current_calibration_ref.setter
    def current_calibration_ref(
        self,
        reference: CalibrationArtifactRef | None,
    ) -> None:
        with self._operation_guard():
            with self._lock:
                if self._closed:
                    raise RuntimeError("Calibration API is closed")
                if reference is not None:
                    if not isinstance(reference, CalibrationArtifactRef):
                        raise TypeError(
                            "current_calibration_ref must be "
                            "CalibrationArtifactRef or None"
                        )
                    self.load_calibration(reference)
                self._current_calibration_ref = reference

    def _remember_committed_calibration(
        self,
        reference: CalibrationArtifactRef,
    ) -> None:
        """Record the typed ref after ``calibration.json`` becomes visible."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Calibration API is closed")
            self._current_calibration_ref = reference

    def close(self) -> tuple[Exception, ...]:
        with self._lock:
            self._closed = True
            self._current_calibration_ref = None
        return ()

    def prepare_calibration_task(
        self,
        intent: CalibrationTaskIntent,
    ) -> PreparedCalibrationTask:
        return prepare_calibration_task(intent, self)

    def prepare_capture(
        self,
        request: CaptureRequest,
    ) -> PreparedFiniteCapture:
        """Bind Calibration's declared live capture through the installed host."""

        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be CaptureRequest")
        return self._bind_capture(request)

    def sitemap_camera_roles(self) -> tuple[str, ...]:
        cameras = set(self._camera_roles)
        roles = tuple(self._profiles)
        if any(role not in cameras for role in roles):
            raise RuntimeError(
                "installation sitemap capabilities differ from its camera catalog"
            )
        return roles

    def _resolve_sitemap_profile(
        self,
        camera_role: str | None,
    ) -> tuple[str, SitemapAcquisitionProfile]:
        selected = self._resolve_camera_role(camera_role)
        try:
            profile = self._profiles[selected]
        except KeyError as error:
            raise ValueError(
                f"camera role {selected!r} has no sitemap profile"
            ) from error
        if profile.readout_binding != ReadoutBindingKey(selected):
            raise ValueError("composed sitemap profile differs from selected camera")
        return selected, profile

    def sitemap_analysis_request(
        self,
        *,
        camera_role: str | None = None,
        threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
        roi_radius: int | None = None,
    ) -> CalibrationAnalysisRequest:
        _selected_camera, profile = self._resolve_sitemap_profile(camera_role)
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
        selected_camera, profile = self._resolve_sitemap_profile(camera_role)
        selected_pulse = self._load_pulse(
            DEFAULT_CALIBRATION_PULSE_PATH if pulse is None else pulse
        )
        camera_ref = self._resolve_camera_ref(selected_camera)
        sequencer_ref = self._resolve_sequencer_ref(profile.sequencer_role)
        return build_sitemap_calibration_request(
            profile,
            camera_ref=camera_ref,
            sequencer_ref=sequencer_ref,
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
        source = self._wait_run(
            self._bind_capture(sequence.capture_request).start()
        )
        try:
            return self.calibrate(self.calibration_request(source, sequence.analysis))
        except KeyboardInterrupt as error:
            raise SitemapCalibrationInterrupted(source) from error
        except Exception as error:
            raise SitemapCalibrationFailed(source) from error

    def write_calibration_post_final_exports(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        save_frames: bool,
        expected_camera_role: str | None = None,
    ) -> None:
        binding = (
            None
            if expected_camera_role is None
            else ReadoutBindingKey(expected_camera_role)
        )
        if binding is not None:
            self._resolve_camera_ref(binding.value)
        self._write_outputs(
            source,
            calibration,
            save_frames=save_frames,
            expected_camera_role=None if binding is None else binding.value,
        )

    def calibration_request(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
    ) -> CalibrationArtifactRequest:
        request = build_calibration_artifact_request(
            self._load_calibration_capture(source),
            analysis,
        )
        return request

    def _load_calibration_capture(self, source: CaptureArtifactRef):
        from zlc_neutral_atom.capture.artifact import load_capture_artifact

        return load_capture_artifact(self._captures_root, source, materialize=False)

    def start_calibration(
        self,
        request: CalibrationArtifactRequest,
        *,
        lifecycle_owner: object | None = None,
    ) -> RunHandle:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        return self._start_calibration_operation(
            request,
            lifecycle_owner,
            self._remember_committed_calibration,
        )

    def start_calibration_analysis(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
        *,
        lifecycle_owner: object | None = None,
    ) -> RunHandle:
        return self.start_calibration(
            self.calibration_request(source, analysis),
            lifecycle_owner=lifecycle_owner,
        )

    def calibrate(
        self,
        request: CalibrationArtifactRequest,
    ) -> CalibrationArtifactRef:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        return self._wait_run(self.start_calibration(request))

    def calibration_gui(self, request: CalibrationArtifactRequest):
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        return self._open_ui(
            "create",
            self.load_calibration_computation,
            self.start_calibration,
            request=request,
        )

    def calibration_edit_gui(self, reference: CalibrationArtifactRef):
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        return self._open_ui(
            "create",
            self.load_calibration_computation,
            self.start_calibration,
            reference=reference,
        )

    def load_calibration(
        self,
        reference: CalibrationArtifactRef,
    ) -> ResolvedCalibration:
        from .repository import load_calibration_artifact

        return load_calibration_artifact(
            self._calibrations_root,
            self._captures_root,
            reference,
        )

    def load_calibration_computation(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationComputation:
        from .repository import load_calibration_computation

        return load_calibration_computation(
            self._calibrations_root,
            self._captures_root,
            reference,
        )

    def load_calibration_report(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationReport:
        return self.load_calibration_computation(reference).report

    def calibration_report_gui(self, reference: CalibrationArtifactRef):
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        return self._open_ui(
            "report",
            self.load_calibration_computation,
            reference,
        )


__all__ = [
    "CalibrationApi",
    "SitemapCalibrationFailed",
    "SitemapCalibrationInterrupted",
]
