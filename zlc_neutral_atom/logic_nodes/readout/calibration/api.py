"""Public Experiment API owned by readout Calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
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
from .repository import CalibrationRepository
from .sitemap import (
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
        "_admit_capture",
        "_admit_saved_capture",
        "_admit_saved_calibration",
        "_bind_capture",
        "_camera_roles",
        "_load_calibration",
        "_load_pulse",
        "_open_ui",
        "_profiles",
        "_repository",
        "_repository_path",
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
        repository_path: Path,
        profiles: Mapping[str, SitemapAcquisitionProfile],
        camera_roles: tuple[str, ...],
        resolve_camera_role: Callable,
        resolve_camera_ref: Callable,
        resolve_sequencer_ref: Callable,
        load_pulse: Callable,
        bind_capture: Callable,
        wait_run: Callable,
        admit_capture: Callable,
        admit_saved_capture: Callable,
        write_outputs: Callable,
        start_calibration: Callable,
        load_calibration: Callable,
        admit_saved_calibration: Callable,
        open_ui: Callable,
    ) -> None:
        if not isinstance(repository_path, Path):
            raise TypeError("repository_path must be Path")
        operations = (
            resolve_camera_role,
            resolve_camera_ref,
            resolve_sequencer_ref,
            load_pulse,
            bind_capture,
            wait_run,
            admit_capture,
            admit_saved_capture,
            write_outputs,
            start_calibration,
            load_calibration,
            admit_saved_calibration,
            open_ui,
        )
        if any(not callable(operation) for operation in operations):
            raise TypeError("Calibration API operations must be callable")
        self._repository_path = repository_path
        self._profiles = MappingProxyType(dict(profiles))
        self._camera_roles = tuple(camera_roles)
        self._resolve_camera_role = resolve_camera_role
        self._resolve_camera_ref = resolve_camera_ref
        self._resolve_sequencer_ref = resolve_sequencer_ref
        self._load_pulse = load_pulse
        self._bind_capture = bind_capture
        self._wait_run = wait_run
        self._admit_capture = admit_capture
        self._admit_saved_capture = admit_saved_capture
        self._write_outputs = write_outputs
        self._start_calibration_operation = start_calibration
        self._load_calibration = load_calibration
        self._admit_saved_calibration = admit_saved_calibration
        self._open_ui = open_ui
        self._repository: CalibrationRepository | None = None

    def _calibration_repository(self) -> CalibrationRepository:
        repository = self._repository
        if repository is None:
            repository = CalibrationRepository(self._repository_path)
            self._repository = repository
        return repository

    def _repository_for_readout_family(self) -> CalibrationRepository:
        """Share the family-owned repository with dependent readout leaves."""

        return self._calibration_repository()

    def close(self) -> tuple[Exception, ...]:
        repository = self._repository
        if repository is None:
            return ()
        try:
            repository.close()
        except Exception as error:
            return (error,)
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
        selected_pulse = None if pulse is None else self._load_pulse(pulse)
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

    def admit_saved_calibration_capture(
        self,
        source_path: str | Path,
        *,
        expected_camera_role: str,
    ) -> CaptureArtifactRef:
        binding = ReadoutBindingKey(expected_camera_role)
        self._resolve_camera_ref(binding.value)
        return self._admit_saved_capture(source_path, binding.value)

    def write_calibration_task_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        folder: str | Path,
        frame_export_policy: str,
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
            self._calibration_repository(),
            folder=folder,
            frame_export_policy=frame_export_policy,
            expected_camera_role=None if binding is None else binding.value,
        )

    def calibration_request(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
    ) -> CalibrationArtifactRequest:
        request = build_calibration_artifact_request(
            self._admit_calibration_capture(source),
            analysis,
        )
        return request

    def _admit_calibration_capture(self, source: CaptureArtifactRef):
        return self._admit_capture(source)

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
            self._calibration_repository(),
            lifecycle_owner,
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
        resolved = self._load_calibration(
            reference,
            self._calibration_repository(),
        )
        return resolved

    def load_saved_calibration(
        self,
        calibration_ref_file: str | Path,
    ) -> ResolvedCalibration:
        resolved = self._admit_saved_calibration(
            calibration_ref_file,
            self._calibration_repository(),
        )
        return resolved

    def load_calibration_computation(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationComputation:
        computation = self._calibration_repository().load_computation(reference)
        return computation

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
