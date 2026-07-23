"""One concrete Capture -> Calibration coordinator for TaskConsole."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Callable

from zlc_neutral_atom.capture_reference import CaptureArtifactRef
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

__all__ = ["CalibrationTaskHandle"]


class _CancelledBetweenStages(Exception):
    pass


class _ChildEnded(Exception):
    def __init__(self, stage: str, snapshot: RunSnapshot) -> None:
        self.stage = stage
        self.snapshot = snapshot


def _summary(error: BaseException) -> str:
    text = str(error).strip()
    return type(error).__name__ if not text else f"{type(error).__name__}: {text}"


class CalibrationTaskHandle:
    """Run-like owner of exactly two ordinary, sequential Runs.

    Capture commits independently.  Only the second child can produce this
    handle's successful ``CalibrationArtifactRef``.
    """

    def __init__(
        self,
        request: SitemapCalibrationRequest,
        *,
        start_capture: Callable[[object], RunHandle],
        build_calibration_request: Callable[
            [CaptureArtifactRef, object, float], object
        ],
        start_calibration: Callable[[object], RunHandle],
    ) -> None:
        if not isinstance(request, SitemapCalibrationRequest):
            raise TypeError("request must be SitemapCalibrationRequest")
        for name, callback in (
            ("start_capture", start_capture),
            ("build_calibration_request", build_calibration_request),
            ("start_calibration", start_calibration),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        self.run_id = RunId(f"calibration-task-{uuid.uuid4().hex}")
        self._request = request
        self._start_capture = start_capture
        self._build_calibration_request = build_calibration_request
        self._start_calibration = start_calibration
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
            source = self._run_child(
                "capture",
                self._start_capture(self._request.capture_request),
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
                self._request.calibration_timeout_seconds,
            )
            self._checkpoint()
            handle = self._start_calibration(request)
            result = self._run_child("calibration", handle)
            if not isinstance(result, CalibrationArtifactRef):
                raise TypeError(
                    "calibration Run returned a non-CalibrationArtifactRef"
                )
            self._result = result
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
