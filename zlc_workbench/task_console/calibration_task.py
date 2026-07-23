"""One concrete Capture -> Calibration coordinator for TaskConsole."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import uuid
from typing import Callable, Mapping

from zlc_data.param_decl import ParamDecl
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.readout.calibration import (
    CalibrationAnalysisRequest,
    ThresholdMethod,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.sitemap import SitemapCalibrationRequest
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunCancelled,
    RunFailed,
    RunHandle,
    RunId,
    RunSnapshot,
    RunState,
)

CALIBRATION_SOURCE_MODES = ("live", "saved frames")
CALIBRATION_THRESHOLD_METHODS = tuple(item.value for item in ThresholdMethod)
DEFAULT_CALIBRATION_FOLDER = "calibrations"
DEFAULT_CALIBRATION_PULSE_PATH = (
    "zlc_neutral_atom/assets/imaging_template.json"
)


def _nonempty_text(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be blank")
    return text


def _finite_nonnegative(value: object, field: str) -> float:
    number = float(value)
    if number < 0.0 or number == float("inf") or number != number:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _finite_positive(value: object, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


@dataclass(frozen=True)
class CalibrationTaskIntent:
    """TaskConsole-owned user intent before installation services bind a run.

    This record deliberately retains the complete Main calibration surface.
    It is not a partial ``SitemapCalibrationRequest``: live acquisition and
    saved-frame calibration are two branches of the same task, while filesystem
    output and raw-frame retention are task orchestration concerns rather than
    fields on the numeric calibration request.
    """

    source_mode: str
    folder: str
    save_frames: bool
    pulse: str
    threshold_method: str
    reference_exposure_s: float
    readout_exposure_s: float
    threshold_frames: int
    roi_radius: int
    camera_role: str

    def __post_init__(self) -> None:
        source_mode = str(self.source_mode).strip().lower()
        if source_mode not in CALIBRATION_SOURCE_MODES:
            raise ValueError(
                f"source_mode must be one of {CALIBRATION_SOURCE_MODES}"
            )
        folder = _nonempty_text(self.folder, "folder")
        if type(self.save_frames) is not bool:
            raise TypeError("save_frames must be bool")
        pulse = _nonempty_text(self.pulse, "pulse")
        threshold_method = str(self.threshold_method).strip().lower()
        if threshold_method not in CALIBRATION_THRESHOLD_METHODS:
            raise ValueError(
                "threshold_method must be one of "
                f"{CALIBRATION_THRESHOLD_METHODS}"
            )
        reference_exposure_s = _finite_positive(
            self.reference_exposure_s,
            "reference_exposure_s",
        )
        readout_exposure_s = _finite_positive(
            self.readout_exposure_s,
            "readout_exposure_s",
        )
        if not isinstance(self.threshold_frames, int) or isinstance(
            self.threshold_frames,
            bool,
        ):
            raise TypeError("threshold_frames must be an integer")
        if self.threshold_frames < 2:
            raise ValueError("threshold_frames must be at least 2")
        if not isinstance(self.roi_radius, int) or isinstance(self.roi_radius, bool):
            raise TypeError("roi_radius must be an integer")
        roi_radius = self.roi_radius
        if roi_radius < 1:
            raise ValueError("roi_radius must be positive")
        camera_role = _nonempty_text(self.camera_role, "camera_role")
        object.__setattr__(self, "source_mode", source_mode)
        object.__setattr__(self, "folder", folder)
        object.__setattr__(self, "pulse", pulse)
        object.__setattr__(self, "threshold_method", threshold_method)
        object.__setattr__(
            self,
            "reference_exposure_s",
            reference_exposure_s,
        )
        object.__setattr__(self, "readout_exposure_s", readout_exposure_s)
        object.__setattr__(self, "roi_radius", roi_radius)
        object.__setattr__(self, "camera_role", camera_role)


def calibration_task_params(
    camera_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    """Main calibration controls, declared once for the TaskConsole form."""

    choices = tuple(str(role) for role in camera_roles)
    if not choices:
        raise ValueError("calibration task requires a configured camera role")
    camera_default = "camera" if "camera" in choices else choices[0]
    return (
        ParamDecl(
            "source_mode",
            "Source",
            "choice",
            default="live",
            required=True,
            choices=CALIBRATION_SOURCE_MODES,
            tooltip="Acquire live frames now or calibrate from saved raw frames.",
        ),
        ParamDecl(
            "folder",
            "Output folder",
            "path",
            default=DEFAULT_CALIBRATION_FOLDER,
            required=True,
            path_mode="dir",
            base_dir=DEFAULT_CALIBRATION_FOLDER,
            tooltip=(
                "The one calibration directory: live writes the result and optional "
                "raw frames here; saved frames reads this directory's frames/ export."
            ),
        ),
        ParamDecl(
            "save_frames",
            "Save live frames",
            "bool",
            default=True,
            tooltip="Keep raw live frames so the same acquisition can be recalibrated.",
        ),
        ParamDecl(
            "pulse",
            "Pulse template",
            "path",
            default=DEFAULT_CALIBRATION_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="zlc_neutral_atom/assets",
            file_filter="Pulse program (*.json);;All files (*)",
            tooltip="Live only: imaging pulse used for each long-short-long bracket.",
        ),
        ParamDecl(
            "threshold_method",
            "Threshold",
            "choice",
            default="otsu",
            required=True,
            choices=CALIBRATION_THRESHOLD_METHODS,
            tooltip="Per-site threshold estimator.",
        ),
        ParamDecl(
            "reference_exposure_s",
            "Reference exposure",
            "float",
            default=0.020,
            unit="s",
            lo=0.0,
            hi=10.0,
            required=True,
            optional=False,
            tooltip="Live only: long exposure for the two outer reference frames.",
        ),
        ParamDecl(
            "readout_exposure_s",
            "Readout exposure",
            "float",
            default=0.005,
            unit="s",
            lo=0.0,
            hi=10.0,
            required=True,
            optional=False,
            tooltip="Live only: short exposure for the middle readout frame.",
        ),
        ParamDecl(
            "threshold_frames",
            "Reference brackets",
            "int",
            default=100,
            lo=2,
            hi=20_000,
            required=True,
            optional=False,
            tooltip="Number of long-short-long calibration shots.",
        ),
        ParamDecl(
            "roi_radius",
            "ROI radius",
            "int",
            default=1,
            unit="px",
            lo=1.0,
            hi=64.0,
            required=True,
            optional=False,
            tooltip="Per-site square ROI half-width in pixels.",
        ),
        ParamDecl(
            "camera_role",
            "Camera",
            "choice",
            default=camera_default,
            required=True,
            choices=choices,
            tooltip="Camera used for live calibration acquisition.",
        ),
    )


def build_calibration_task_intent(
    values: Mapping[str, object],
) -> CalibrationTaskIntent:
    """Freeze all form values without pretending they already form a Run."""

    return CalibrationTaskIntent(
        source_mode=str(values.get("source_mode", "live")),
        folder=str(values.get("folder", DEFAULT_CALIBRATION_FOLDER)),
        save_frames=values.get("save_frames", True),
        pulse=str(values.get("pulse", DEFAULT_CALIBRATION_PULSE_PATH)),
        threshold_method=str(values.get("threshold_method", "otsu")),
        reference_exposure_s=float(values.get("reference_exposure_s", 0.020)),
        readout_exposure_s=float(values.get("readout_exposure_s", 0.005)),
        threshold_frames=values.get("threshold_frames", 100),
        roi_radius=values.get("roi_radius", 1),
        camera_role=str(values.get("camera_role", "")),
    )


__all__ = [
    "CALIBRATION_SOURCE_MODES",
    "CALIBRATION_THRESHOLD_METHODS",
    "CalibrationTaskHandle",
    "CalibrationTaskExecution",
    "CalibrationTaskIntent",
    "build_calibration_task_intent",
    "calibration_task_params",
]


class _CancelledBetweenStages(Exception):
    pass


class _ChildEnded(Exception):
    def __init__(self, stage: str, snapshot: RunSnapshot) -> None:
        self.stage = stage
        self.snapshot = snapshot


def _summary(error: BaseException) -> str:
    text = str(error).strip()
    return type(error).__name__ if not text else f"{type(error).__name__}: {text}"


@dataclass(frozen=True)
class CalibrationTaskExecution:
    """Fully prepared live or saved-source calibration task."""

    intent: CalibrationTaskIntent
    analysis: CalibrationAnalysisRequest
    sequence: SitemapCalibrationRequest | None = None
    source_capture_ref: CaptureArtifactRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intent, CalibrationTaskIntent):
            raise TypeError("intent must be CalibrationTaskIntent")
        if not isinstance(self.analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        if self.intent.source_mode == "live":
            if not isinstance(self.sequence, SitemapCalibrationRequest):
                raise TypeError("live calibration requires SitemapCalibrationRequest")
            if self.sequence.analysis != self.analysis:
                raise ValueError("live calibration sequence differs from frozen analysis")
            if self.source_capture_ref is not None:
                raise ValueError("live calibration cannot preselect a source capture")
        else:
            if not isinstance(self.source_capture_ref, CaptureArtifactRef):
                raise TypeError("saved calibration requires CaptureArtifactRef")
            if self.sequence is not None:
                raise ValueError("saved calibration cannot contain a capture request")


class CalibrationTaskHandle:
    """Run-like owner of one analysis Run after an optional live Capture Run.

    Live input commits its Capture independently; saved input is an already
    admitted exact CaptureArtifactRef.  Both branches enter the same analysis
    Run and only that Run can produce ``CalibrationArtifactRef``.
    """

    def __init__(
        self,
        request: CalibrationTaskExecution,
        *,
        start_capture: Callable[[object], RunHandle],
        build_calibration_request: Callable[
            [CaptureArtifactRef, object], object
        ],
        start_calibration: Callable[[object], RunHandle],
        write_outputs: Callable[
            [CaptureArtifactRef, CalibrationArtifactRef, CalibrationTaskIntent], None
        ],
    ) -> None:
        if not isinstance(request, CalibrationTaskExecution):
            raise TypeError("request must be CalibrationTaskExecution")
        for name, callback in (
            ("start_capture", start_capture),
            ("build_calibration_request", build_calibration_request),
            ("start_calibration", start_calibration),
            ("write_outputs", write_outputs),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        self.run_id = RunId(f"calibration-task-{uuid.uuid4().hex}")
        self._request = request
        self._start_capture = start_capture
        self._build_calibration_request = build_calibration_request
        self._start_calibration = start_calibration
        self._write_outputs = write_outputs
        self._condition = threading.Condition(threading.RLock())
        self._phase = "capture-starting"
        self._active: RunHandle | None = None
        self._stage: str | None = None
        self._cancel_requested = False
        self._cancel_reason = "user requested stop"
        self._terminal: RunSnapshot | None = None
        self._result: CalibrationArtifactRef | None = None
        self._source: CaptureArtifactRef | None = None
        self._thread = threading.Thread(
            target=self._coordinate,
            name=f"zlc-calibration-task-{self.run_id.value[-12:]}",
            daemon=False,
        )
        self._thread.start()

    @property
    def source_capture_ref(self) -> CaptureArtifactRef | None:
        with self._condition:
            return self._source

    def _checkpoint(self) -> None:
        with self._condition:
            if self._cancel_requested:
                raise _CancelledBetweenStages

    def _run_child(self, stage: str, handle: RunHandle):
        if not isinstance(handle, RunHandle):
            raise TypeError(f"{stage} starter returned a non-RunHandle")
        with self._condition:
            self._active = handle
            self._stage = stage
            self._phase = f"{stage}-running"
            cancelled = self._cancel_requested
            reason = self._cancel_reason
            self._condition.notify_all()
        if cancelled:
            handle.cancel(reason)
        try:
            return handle.result()
        except (RunCancelled, RunFailed) as error:
            raise _ChildEnded(stage, error.snapshot) from None
        finally:
            with self._condition:
                if self._active is handle:
                    self._active = None
                    self._stage = None

    def _finish(
        self,
        state: RunState,
        phase: str,
        *,
        child: RunSnapshot | None = None,
        error: str | None = None,
    ) -> None:
        with self._condition:
            self._terminal = RunSnapshot(
                self.run_id,
                state,
                phase,
                self._result is not None,
                None if child is None else child.commit_recovery_warning,
                (
                    error
                    if error is not None
                    else None if child is None else child.primary_error
                ),
                () if child is None else child.cleanup_errors,
                None if child is None else child.recovery_instruction,
            )
            self._active = None
            self._stage = None
            self._condition.notify_all()

    def _coordinate(self) -> None:
        try:
            sequence = self._request.sequence
            if sequence is None:
                source = self._request.source_capture_ref
                assert isinstance(source, CaptureArtifactRef)
            else:
                source = self._run_child(
                    "capture",
                    self._start_capture(sequence.capture_request),
                )
                if not isinstance(source, CaptureArtifactRef):
                    raise TypeError("capture Run returned a non-CaptureArtifactRef")
            with self._condition:
                self._source = source
                self._phase = "calibration-preparing"
            self._checkpoint()
            request = self._build_calibration_request(
                source,
                self._request.analysis,
            )
            self._checkpoint()
            handle = self._start_calibration(request)
            result = self._run_child("calibration", handle)
            if not isinstance(result, CalibrationArtifactRef):
                raise TypeError(
                    "calibration Run returned a non-CalibrationArtifactRef"
                )
            self._result = result
            with self._condition:
                self._phase = "writing-task-outputs"
            self._write_outputs(source, result, self._request.intent)
            self._finish(
                RunState.SUCCEEDED,
                "calibration-committed",
                child=handle.snapshot(),
            )
        except _CancelledBetweenStages:
            self._finish(RunState.CANCELLED, "cancelled")
        except _ChildEnded as ended:
            source_note = (
                None
                if (
                    ended.snapshot.state is not RunState.FAILED
                    or ended.stage != "calibration"
                    or self._source is None
                )
                else (
                    f"{ended.snapshot.primary_error or 'calibration Run failed'}; "
                    f"source capture remains {self._source!r}"
                )
            )
            self._finish(
                ended.snapshot.state,
                "cancelled"
                if ended.snapshot.state is RunState.CANCELLED
                else "failed",
                child=ended.snapshot,
                error=source_note,
            )
        except BaseException as error:
            with self._condition:
                cancelled = self._cancel_requested
                source = self._source
            failure = _summary(error)
            if source is not None:
                failure += f"; source capture remains {source!r}"
            if self._result is not None:
                failure += f"; calibration remains {self._result!r}"
            self._finish(
                RunState.CANCELLED if cancelled else RunState.FAILED,
                "cancelled" if cancelled else "failed",
                error=None if cancelled else failure,
            )

    def snapshot(self) -> RunSnapshot:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            active = self._active
            stage = self._stage
            phase = self._phase
            cancelling = self._cancel_requested
        if active is None:
            child = None
        else:
            child = active.snapshot()
            phase = f"{stage}/{child.phase}"
        return RunSnapshot(
            self.run_id,
            RunState.CANCELLING if cancelling else RunState.RUNNING,
            phase,
            bool(
                child is not None
                and stage == "calibration"
                and child.final_committed
            ),
            None if child is None else child.commit_recovery_warning,
            None if child is None else child.primary_error,
            () if child is None else child.cleanup_errors,
            None if child is None else child.recovery_instruction,
        )

    def cancel(self, reason: str = "user requested stop") -> CancelOutcome:
        text = str(reason).strip() or "user requested stop"
        with self._condition:
            if self._terminal is not None:
                return CancelOutcome.ALREADY_TERMINAL
            if self._cancel_requested:
                return CancelOutcome.ALREADY_REQUESTED
            self._cancel_requested = True
            self._cancel_reason = text
            active = self._active
            self._condition.notify_all()
        return (
            CancelOutcome.REQUESTED
            if active is None
            else active.cancel(text)
        )

    def wait(self, timeout: float | None = None) -> RunSnapshot:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout < 0
        ):
            raise ValueError("wait timeout must be a non-negative real or None")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._terminal is None:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"calibration task {self.run_id} is active")
                self._condition.wait(remaining)
            snapshot = self._terminal
        remaining = (
            None
            if deadline is None
            else max(0.0, deadline - time.monotonic())
        )
        self._thread.join(remaining)
        if self._thread.is_alive():
            raise TimeoutError(
                f"calibration task {self.run_id} is terminal but not reaped"
            )
        return snapshot

    def result(self, timeout: float | None = None) -> CalibrationArtifactRef:
        snapshot = self.wait(timeout)
        if snapshot.state is RunState.SUCCEEDED:
            assert self._result is not None
            return self._result
        if snapshot.state is RunState.CANCELLED:
            raise RunCancelled(snapshot)
        raise RunFailed(snapshot)
