"""TaskConsole presentation for the typed MOT-field request."""

from __future__ import annotations

import threading
import time
from typing import Mapping
import uuid

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


def _preferred(
    roles: tuple[str, ...],
    *candidates: str,
) -> str:
    if not roles:
        raise ValueError("MOT task requires at least one configured role")
    for candidate in candidates:
        if candidate in roles:
            return candidate
    return roles[0]


def mot_field_params(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    """Return the familiar one-click MOT controls, with no generic timeout."""

    camera_roles = tuple(
        role for role in camera_roles if role == "mot_camera"
    ) or ("mot_camera",)
    sequencer_roles = tuple(sequencer_roles)
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
            "camera_role",
            "Camera role",
            "choice",
            default=_preferred(camera_roles, "mot_camera"),
            required=True,
            choices=camera_roles,
            tooltip=(
                "Must be an external-trigger-capable camera physically observing "
                "the MOT; a free-running monitor cannot prove point association"
            ),
        ),
        ParamDecl(
            "sequencer_role",
            "Sequencer role",
            "choice",
            default=_preferred(sequencer_roles, "sequencer"),
            required=True,
            choices=sequencer_roles,
        ),
        ParamDecl(
            "trigger_channel",
            "Trigger channel",
            "text",
            default=None,
            required=False,
            tooltip="Leave blank to use the camera/pulse binding's declared trigger",
        ),
    )


def build_mot_field_request(experiment, values: Mapping[str, object]):
    """Freeze the form through the notebook facade's typed request owner."""

    roi_cx = float(values.get("roi_cx", 0.0))
    roi_cy = float(values.get("roi_cy", 0.0))
    options = {
        "center_x": float(values.get("center_x", 0.0)),
        "center_y": float(values.get("center_y", 0.0)),
        "center_z": float(values.get("center_z", 0.0)),
        "span": float(values.get("span", 12.0)),
        "points": int(values.get("points", 7)),
        "roi_cx": None if roi_cx == 0.0 else roi_cx,
        "roi_cy": None if roi_cy == 0.0 else roi_cy,
        "roi_radius": float(values.get("roi_radius", 8.0)),
        "camera_role": str(values["camera_role"]),
        "sequencer_role": str(values["sequencer_role"]),
    }
    trigger = values.get("trigger_channel")
    if trigger not in (None, ""):
        options["trigger_channel"] = str(trigger)
    pulse = values.get("pulse") or DEFAULT_MOT_FIELD_PULSE_PATH
    return experiment.readout.mot_field_request(pulse, **options)


class _ScanEnded(Exception):
    def __init__(self, snapshot: RunSnapshot) -> None:
        self.snapshot = snapshot


class _CancelledAfterScan(Exception):
    pass


def _summary(error: BaseException) -> str:
    text = str(error).strip()
    return type(error).__name__ if not text else f"{type(error).__name__}: {text}"


class MotFieldTaskHandle:
    """Run-like owner of one exact Scan Run followed by pure MOT analysis."""

    def __init__(
        self,
        request: MotFieldRequest,
        *,
        start_scan,
        materialize_scan,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not callable(start_scan) or not callable(materialize_scan):
            raise TypeError("MOT task callbacks must be callable")
        self.run_id = RunId(f"mot-field-task-{uuid.uuid4().hex}")
        self._request = request
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
                self._result = result
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


def start_mot_field_task(
    request: MotFieldRequest,
    *,
    start_scan,
    materialize_scan,
) -> MotFieldTaskHandle:
    return MotFieldTaskHandle(
        request,
        start_scan=start_scan,
        materialize_scan=materialize_scan,
    )


__all__ = [
    "MotFieldTaskHandle",
    "build_mot_field_request",
    "mot_field_params",
    "start_mot_field_task",
]
