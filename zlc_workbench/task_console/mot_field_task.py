"""TaskConsole presentation for the typed MOT-field request."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Mapping
import uuid

import numpy as np

from zlc_data.param_decl import ParamDecl
from zlc_neutral_atom.pulse_programs import DEFAULT_MOT_FIELD_PULSE_PATH
from zlc_neutral_atom.mot_field import (
    MotFieldRequest,
    MotFieldResult,
    analyze_mot_scan,
)
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunCancelled,
    RunFailed,
    RunHandle,
    RunId,
    RunSnapshot,
    RunState,
)
from zlc_neutral_atom.scan import ScanArtifactRef
from zlc_neutral_atom.scan.repository import MaterializedScanData
from zlc_storage.paths import resolve_under_project


def mot_field_params(
    camera_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    """Return the familiar one-click MOT controls, with no generic timeout."""

    camera_roles = tuple(
        role for role in camera_roles if role == "mot_camera"
    ) or ("mot_camera",)
    return (
        ParamDecl(
            "pulse",
            "Pulse template",
            "path",
            default=DEFAULT_MOT_FIELD_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
            tooltip="Autonomous SCAN_SLOT template declaring da_x, da_y and da_z",
        ),
        ParamDecl(
            "center_x",
            "Bx centre",
            "float",
            default=0.0,
            unit="code",
            lo=-512.0,
            hi=511.0,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "center_y",
            "By centre",
            "float",
            default=0.0,
            unit="code",
            lo=-512.0,
            hi=511.0,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "center_z",
            "Bz centre",
            "float",
            default=0.0,
            unit="code",
            lo=-512.0,
            hi=511.0,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "span",
            "Span (+/-)",
            "float",
            default=12.0,
            unit="code",
            lo=0.0,
            hi=511.0,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "points",
            "Points per axis",
            "int",
            default=7,
            lo=2,
            hi=15,
            required=True,
            optional=False,
            tooltip="Total autonomous scan cells are points^3",
        ),
        ParamDecl(
            "roi_cx",
            "ROI centre x",
            "float",
            default=0.0,
            unit="px",
            lo=0.0,
            hi=1_000_000.0,
            required=True,
            optional=False,
            tooltip="0 uses the frame centre, matching the established MOT task",
        ),
        ParamDecl(
            "roi_cy",
            "ROI centre y",
            "float",
            default=0.0,
            unit="px",
            lo=0.0,
            hi=1_000_000.0,
            required=True,
            optional=False,
            tooltip="0 uses the frame centre, matching the established MOT task",
        ),
        ParamDecl(
            "roi_radius",
            "ROI radius",
            "float",
            default=8.0,
            unit="px",
            lo=0.1,
            hi=1_000_000.0,
            required=True,
            optional=False,
            tooltip="The 1x..2x annulus supplies the local background",
        ),
        ParamDecl(
            "folder",
            "Report folder",
            "path",
            default="mot_field",
            required=True,
            path_mode="dir",
            tooltip=(
                "Raw intensity block, exact Bx/By/Bz axes, and refined "
                "optimum are written to mot_field_scan.npz"
            ),
        ),
        ParamDecl(
            "camera_role",
            "Camera role",
            "choice",
            default=camera_roles[0],
            required=True,
            choices=camera_roles,
            tooltip=(
                "Must be an external-trigger-capable camera physically observing "
                "the MOT; a free-running monitor cannot prove point association"
            ),
        ),
    )


def _finite(value: object, field: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return number


@dataclass(frozen=True)
class MotFieldTaskIntent:
    """Complete user-facing MOT task intent before hardware binding."""

    pulse: str
    center_x: float
    center_y: float
    center_z: float
    span: float
    points: int
    roi_cx: float
    roi_cy: float
    roi_radius: float
    folder: str
    camera_role: str

    def __post_init__(self) -> None:
        pulse = str(self.pulse).strip()
        if not pulse:
            raise ValueError("pulse must not be blank")
        center_x = _finite(self.center_x, "center_x")
        center_y = _finite(self.center_y, "center_y")
        center_z = _finite(self.center_z, "center_z")
        span = _finite(self.span, "span")
        if span < 0.0:
            raise ValueError("span must be non-negative")
        if not isinstance(self.points, int) or isinstance(self.points, bool):
            raise TypeError("points must be an integer")
        if self.points < 2:
            raise ValueError("points must be at least 2")
        roi_cx = _finite(self.roi_cx, "roi_cx")
        roi_cy = _finite(self.roi_cy, "roi_cy")
        if roi_cx < 0.0 or roi_cy < 0.0:
            raise ValueError("ROI centre coordinates must be non-negative")
        roi_radius = _finite(self.roi_radius, "roi_radius")
        if roi_radius <= 0.0:
            raise ValueError("roi_radius must be positive")
        folder = str(self.folder).strip()
        if not folder:
            raise ValueError("folder must not be blank")
        camera_role = str(self.camera_role).strip()
        if camera_role != "mot_camera":
            raise ValueError("MOT field task requires the mot_camera role")
        object.__setattr__(self, "pulse", pulse)
        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", center_y)
        object.__setattr__(self, "center_z", center_z)
        object.__setattr__(self, "span", span)
        object.__setattr__(self, "roi_cx", roi_cx)
        object.__setattr__(self, "roi_cy", roi_cy)
        object.__setattr__(self, "roi_radius", roi_radius)
        object.__setattr__(self, "folder", folder)
        object.__setattr__(self, "camera_role", camera_role)


def build_mot_field_intent(
    values: Mapping[str, object],
) -> MotFieldTaskIntent:
    """Freeze the visible task form without exposing hardware wiring knobs."""

    return MotFieldTaskIntent(
        pulse=str(values.get("pulse") or DEFAULT_MOT_FIELD_PULSE_PATH),
        center_x=float(values.get("center_x", 0.0)),
        center_y=float(values.get("center_y", 0.0)),
        center_z=float(values.get("center_z", 0.0)),
        span=float(values.get("span", 12.0)),
        points=values.get("points", 7),
        roi_cx=float(values.get("roi_cx", 0.0)),
        roi_cy=float(values.get("roi_cy", 0.0)),
        roi_radius=float(values.get("roi_radius", 8.0)),
        folder=str(values.get("folder", "mot_field")),
        camera_role=str(values.get("camera_role", "mot_camera")),
    )


class _ScanEnded(Exception):
    def __init__(self, snapshot: RunSnapshot) -> None:
        self.snapshot = snapshot


class _CancelledAfterScan(Exception):
    pass


def _summary(error: BaseException) -> str:
    text = str(error).strip()
    return type(error).__name__ if not text else f"{type(error).__name__}: {text}"


def write_mot_field_report(
    result: MotFieldResult,
    folder: str | Path,
) -> Path:
    """Atomically write the Task Console's authoritative MOT report data.

    The exact scan repository remains the owner of camera frames and run
    lineage.  This small operator-facing report preserves the analyzed 3-D
    intensity block, all three physical DAC coordinate axes, and the refined
    optimum.  Plot panels consume :class:`MotFieldResult` separately; report
    persistence therefore has no frontend or matplotlib dependency.
    """

    if not isinstance(result, MotFieldResult):
        raise TypeError("result must be MotFieldResult")
    directory = resolve_under_project(folder)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "mot_field_scan.npz"
    axes = tuple(
        np.asarray(
            tuple(axis.coordinate_at(index) for index in range(axis.size))
        )
        for axis in result.point_axes
    )
    temporary = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                intensity=np.asarray(result.intensity),
                bx=axes[0],
                by=axes[1],
                bz=axes[2],
                best=np.asarray(result.best_field, dtype=np.float64),
                best_intensity=np.asarray(
                    result.best_intensity,
                    dtype=np.float64,
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class MotFieldTaskHandle:
    """Run-like owner of one exact Scan Run followed by pure MOT analysis."""

    def __init__(
        self,
        request: MotFieldRequest,
        *,
        report_folder: str | Path,
        start_scan,
        materialize_scan,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not callable(start_scan) or not callable(materialize_scan):
            raise TypeError("MOT task callbacks must be callable")
        self.run_id = RunId(f"mot-field-task-{uuid.uuid4().hex}")
        self._request = request
        self._report_folder = resolve_under_project(report_folder)
        self._start_scan = start_scan
        self._materialize_scan = materialize_scan
        self._condition = threading.Condition(threading.RLock())
        self._active: RunHandle | None = None
        self._phase = "scan-starting"
        self._cancel_requested = False
        self._cancel_reason = "user requested stop"
        self._terminal: RunSnapshot | None = None
        self._scan_ref: ScanArtifactRef | None = None
        self._result: MotFieldResult | None = None
        self._report_path: Path | None = None
        self._thread = threading.Thread(
            target=self._coordinate,
            name=f"zlc-mot-field-{self.run_id.value[-12:]}",
            daemon=False,
        )
        self._thread.start()

    @property
    def source_scan_ref(self) -> ScanArtifactRef | None:
        with self._condition:
            return self._scan_ref

    @property
    def report_path(self) -> Path | None:
        """Return the committed report path, or ``None`` before success."""

        with self._condition:
            return self._report_path

    def _checkpoint(self) -> None:
        with self._condition:
            if self._cancel_requested:
                raise _CancelledAfterScan

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
                error if error is not None else (
                    None if child is None else child.primary_error
                ),
                () if child is None else child.cleanup_errors,
                None if child is None else child.recovery_instruction,
            )
            self._active = None
            self._condition.notify_all()

    def _coordinate(self) -> None:
        child: RunHandle | None = None
        try:
            child = self._start_scan(self._request)
            if not isinstance(child, RunHandle):
                raise TypeError("MOT scan starter returned a non-RunHandle")
            with self._condition:
                self._active = child
                self._phase = "scan-running"
                cancelled = self._cancel_requested
                reason = self._cancel_reason
                self._condition.notify_all()
            if cancelled:
                child.cancel(reason)
            try:
                source = child.result()
            except (RunCancelled, RunFailed) as error:
                raise _ScanEnded(error.snapshot) from None
            finally:
                with self._condition:
                    if self._active is child:
                        self._active = None
            if not isinstance(source, ScanArtifactRef):
                raise TypeError("MOT scan Run returned a non-ScanArtifactRef")
            with self._condition:
                self._scan_ref = source
                self._phase = "analyzing-final-scan"
            self._checkpoint()
            materialized = self._materialize_scan(source)
            if not isinstance(materialized, MaterializedScanData):
                raise TypeError("MOT materializer returned a non-MaterializedScanData")
            self._checkpoint()
            result = analyze_mot_scan(self._request, materialized)
            self._checkpoint()
            with self._condition:
                self._phase = "writing-report"
            report_path = write_mot_field_report(
                result,
                self._report_folder,
            )
            with self._condition:
                self._result = result
                self._report_path = report_path
            self._finish(
                RunState.SUCCEEDED,
                "mot-field-complete",
                child=child.snapshot(),
            )
        except _CancelledAfterScan:
            self._finish(RunState.CANCELLED, "cancelled")
        except _ScanEnded as ended:
            self._finish(
                ended.snapshot.state,
                "cancelled"
                if ended.snapshot.state is RunState.CANCELLED
                else "failed",
                child=ended.snapshot,
            )
        except BaseException as error:
            with self._condition:
                cancelled = self._cancel_requested
            self._finish(
                RunState.CANCELLED if cancelled else RunState.FAILED,
                "cancelled" if cancelled else "failed",
                child=None if child is None else child.snapshot(),
                error=None if cancelled else _summary(error),
            )

    def snapshot(self) -> RunSnapshot:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            child = self._active
            phase = self._phase
            cancelling = self._cancel_requested
        child_snapshot = None if child is None else child.snapshot()
        if child_snapshot is not None:
            phase = f"scan/{child_snapshot.phase}"
        return RunSnapshot(
            self.run_id,
            RunState.CANCELLING if cancelling else RunState.RUNNING,
            phase,
            False,
            None
            if child_snapshot is None
            else child_snapshot.commit_recovery_warning,
            None if child_snapshot is None else child_snapshot.primary_error,
            () if child_snapshot is None else child_snapshot.cleanup_errors,
            None if child_snapshot is None else child_snapshot.recovery_instruction,
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
            child = self._active
            self._condition.notify_all()
        return (
            CancelOutcome.REQUESTED
            if child is None
            else child.cancel(text)
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
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"MOT field task {self.run_id} is active")
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
                f"MOT field task {self.run_id} is terminal but not reaped"
            )
        return snapshot

    def result(self, timeout: float | None = None) -> MotFieldResult:
        snapshot = self.wait(timeout)
        if snapshot.state is RunState.SUCCEEDED:
            assert self._result is not None
            return self._result
        if snapshot.state is RunState.CANCELLED:
            raise RunCancelled(snapshot)
        raise RunFailed(snapshot)


__all__ = [
    "MotFieldTaskIntent",
    "MotFieldTaskHandle",
    "build_mot_field_intent",
    "mot_field_params",
    "write_mot_field_report",
]
