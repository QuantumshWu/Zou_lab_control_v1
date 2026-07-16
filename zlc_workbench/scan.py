"""Headless lifecycle owner for one final-only scan panel.

The controller deliberately knows neither how a scan is compiled nor where its
artifact is stored.  A composition-owned application freezes those decisions,
starts one Run, and projects an already FINAL artifact into immutable display
bytes.  Qt is only expected to provide a no-payload owner wake and to paint the
returned view model.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Protocol, runtime_checkable

from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunHandle,
    RunId,
    RunSnapshot,
    RunState,
)
from zlc_neutral_atom.scan.reference import ScanArtifactRef


_DEFAULT_PROJECTION_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _error_summary(error: BaseException) -> str:
    text = str(error).strip()
    return type(error).__name__ if not text else f"{type(error).__name__}: {text}"


@dataclass(frozen=True, slots=True)
class FinalScanPresentation:
    """Detached final-only raster plus the artifact identity that justified it."""

    source_ref: ScanArtifactRef
    png_bytes: bytes
    projection_summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, ScanArtifactRef):
            raise TypeError("source_ref must be ScanArtifactRef")
        if not isinstance(self.png_bytes, bytes):
            raise TypeError("png_bytes must be immutable bytes")
        if not self.png_bytes.startswith(_PNG_SIGNATURE):
            raise ValueError("png_bytes must contain a PNG raster")
        if len(self.png_bytes) < 24 or self.png_bytes[12:16] != b"IHDR":
            raise ValueError("png_bytes must contain a canonical PNG IHDR")
        width, height = self.raster_size
        if width <= 0 or height <= 0:
            raise ValueError("PNG raster dimensions must be positive")
        if not isinstance(self.projection_summary, str):
            raise TypeError("projection_summary must be str")
        summary = self.projection_summary.strip()
        if not summary:
            raise ValueError("projection_summary cannot be empty")
        object.__setattr__(self, "projection_summary", summary)

    @property
    def raster_size(self) -> tuple[int, int]:
        return (
            int.from_bytes(self.png_bytes[16:20], "big"),
            int.from_bytes(self.png_bytes[20:24], "big"),
        )

    @property
    def gui_decode_peak_nbytes(self) -> int:
        width, height = self.raster_size
        # One decoded source pixmap plus one no-upscale presentation pixmap.
        return len(self.png_bytes) + 2 * width * height * 4


@runtime_checkable
class ScanPanelApplication(Protocol):
    """Narrow application port over one already-frozen scan request."""

    def start(self) -> RunHandle[ScanArtifactRef]: ...

    def project_final(
        self,
        source_ref: ScanArtifactRef,
        *,
        memory_limit_bytes: int,
    ) -> FinalScanPresentation: ...


@dataclass(frozen=True, slots=True)
class ScanPanelViewModel:
    """Owner-thread snapshot consumed by a thin GUI shell."""

    generation: int
    status: str
    run_id: str | None
    artifact_ref: ScanArtifactRef | None
    presentation: FinalScanPresentation | None
    diagnostic: str | None
    can_start: bool
    can_stop: bool
    worker_idle: bool
    closing: bool
    closed: bool
    final_only: bool = True


@dataclass(frozen=True, slots=True)
class _WorkToken:
    generation: int
    run_id: RunId | None = None
    artifact_ref: ScanArtifactRef | None = None


class ScanPanelController:
    """One-Run controller with a worker mailbox and owner-thread state changes.

    ``owner_cycle`` must be called by the GUI owner whenever
    ``request_owner_wake`` fires and periodically while a Run is active.  The
    periodic call is only a nonblocking ``RunHandle.snapshot`` poll.  A terminal
    handle is reaped with ``result``/``wait`` on a worker, never on the GUI
    thread.
    """

    def __init__(
        self,
        application: ScanPanelApplication,
        request_owner_wake: object,
        *,
        projection_memory_limit_bytes: int = _DEFAULT_PROJECTION_MEMORY_LIMIT_BYTES,
        executor: Executor | None = None,
    ) -> None:
        if not isinstance(application, ScanPanelApplication):
            raise TypeError("application must implement ScanPanelApplication")
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError("executor must implement concurrent.futures.Executor")

        self._owner_thread = threading.get_ident()
        self._application: ScanPanelApplication | None = application
        self._request_owner_wake = request_owner_wake
        self._projection_memory_limit_bytes = _positive_integer(
            projection_memory_limit_bytes,
            "projection_memory_limit_bytes",
        )
        self._executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="zlc-scan-panel")
            if executor is None
            else executor
        )
        self._owns_executor = executor is None
        self._executor_closed = False

        self._lock = threading.Lock()
        self._tracked: set[Future] = set()
        self._mailbox: list[tuple[str, _WorkToken, Future]] = []
        self._wake_failure: str | None = None

        self._generation = 0
        self._starting = False
        self._cancel_when_started = False
        self._handle: RunHandle[ScanArtifactRef] | None = None
        self._terminal_work_inflight = False
        self._owner_reaped = True
        self._projection_inflight = False
        self._artifact_ref: ScanArtifactRef | None = None
        self._presentation: FinalScanPresentation | None = None
        self._diagnostic: str | None = None
        self._status = "IDLE · FINAL-ONLY"
        self._closing = False
        self._closed = False
        self._view_model = self._build_view_model()

    @property
    def view_model(self) -> ScanPanelViewModel:
        self._require_owner()
        return self._view_model

    @property
    def worker_idle(self) -> bool:
        with self._lock:
            return not self._tracked and not self._mailbox

    @property
    def closed(self) -> bool:
        self._require_owner()
        return self._closed

    def start(self) -> int:
        """Start the frozen request without blocking the owner thread."""

        self._require_owner()
        if not self._can_start():
            raise RuntimeError("scan panel is not ready to start another Run")
        self._generation += 1
        token = _WorkToken(self._generation)
        self._starting = True
        self._cancel_when_started = False
        self._handle = None
        self._terminal_work_inflight = False
        self._owner_reaped = False
        self._projection_inflight = False
        self._artifact_ref = None
        self._presentation = None
        self._diagnostic = None
        self._status = "STARTING · NOT FINAL"
        application = self._application
        if application is None:
            raise RuntimeError("scan panel application is detached")
        self._submit("start", token, application.start)
        self._publish_model()
        return token.generation

    def stop(self) -> CancelOutcome | None:
        """Request cancellation, including the admission race before a handle exists."""

        self._require_owner()
        if self._closing or self._closed:
            return None
        if self._starting and self._handle is None:
            if self._cancel_when_started:
                return CancelOutcome.ALREADY_REQUESTED
            self._cancel_when_started = True
            self._status = "CANCELLING BEFORE ADMISSION · NOT FINAL"
            self._publish_model()
            return CancelOutcome.REQUESTED
        handle = self._handle
        if handle is None:
            return None
        snapshot = handle.snapshot()
        self._validate_snapshot(handle, snapshot)
        if snapshot.state.terminal:
            self._poll_active_handle()
            return CancelOutcome.ALREADY_TERMINAL
        outcome = handle.cancel("scan panel user requested stop")
        snapshot = handle.snapshot()
        self._status = f"{snapshot.state.value} · NOT FINAL"
        self._publish_model()
        return outcome

    def owner_cycle(self) -> ScanPanelViewModel:
        """Drain worker completions and poll the active Run without blocking."""

        self._require_owner()
        if self._closed:
            return self._view_model
        self._drain_mailbox()
        if not self._closing:
            self._poll_active_handle()
        self._maybe_finish_close()
        self._publish_model()
        return self._view_model

    def close(self) -> None:
        """Begin nonblocking shutdown; completion is observed through owner_cycle."""

        self._require_owner()
        if self._closed or self._closing:
            return
        self._closing = True
        self._generation += 1  # revoke every current completion token
        self._cancel_when_started = True
        self._status = "CLOSING"

        handle = self._handle
        terminal_inflight = self._terminal_work_inflight
        self._handle = None
        if handle is not None and not self._owner_reaped and not terminal_inflight:
            try:
                handle.cancel("scan panel is closing")
            except BaseException as error:
                self._record_diagnostic(
                    f"close cancellation failed: {_error_summary(error)}"
                )
            self._submit_stale_reap(handle)
        elif handle is not None and not self._owner_reaped:
            try:
                handle.cancel("scan panel is closing")
            except BaseException as error:
                self._record_diagnostic(
                    f"close cancellation failed: {_error_summary(error)}"
                )

        self._publish_model()
        self._request_wake()
        self._maybe_finish_close()

    def _submit(self, kind: str, token: _WorkToken, work: object) -> Future:
        if self._executor_closed or not callable(work):
            raise RuntimeError("scan panel worker is unavailable")
        future = self._executor.submit(work)
        with self._lock:
            self._tracked.add(future)

        def done(completed: Future) -> None:
            with self._lock:
                self._tracked.discard(completed)
                self._mailbox.append((kind, token, completed))
            self._request_wake()

        future.add_done_callback(done)
        return future

    def _request_wake(self) -> None:
        wake = self._request_owner_wake
        if wake is None:
            return
        try:
            wake()
        except BaseException as error:
            with self._lock:
                self._wake_failure = _error_summary(error)

    def _drain_mailbox(self) -> None:
        with self._lock:
            pending, self._mailbox = self._mailbox, []
            wake_failure, self._wake_failure = self._wake_failure, None
        if wake_failure is not None:
            self._record_diagnostic(f"owner wake failed: {wake_failure}")
        for kind, token, future in pending:
            if kind == "start":
                self._accept_start(token, future)
            elif kind == "terminal-result":
                self._accept_terminal_result(token, future)
            elif kind == "terminal-reap":
                self._accept_terminal_reap(token, future)
            elif kind == "project-final":
                self._accept_projection(token, future)
            elif kind == "stale-reap":
                self._accept_stale_reap(future)
            else:
                self._record_diagnostic(f"unknown worker completion kind {kind!r}")

    def _accept_start(self, token: _WorkToken, future: Future) -> None:
        try:
            handle = future.result()
            self._validate_handle(handle)
        except BaseException as error:
            if token.generation == self._generation and not self._closing:
                self._starting = False
                self._owner_reaped = True
                self._status = "FAILED BEFORE ADMISSION · NOT FINAL"
                self._record_diagnostic(f"scan start failed: {_error_summary(error)}")
            return

        if token.generation != self._generation or self._closing:
            try:
                handle.cancel("stale scan panel start result")
            except BaseException as error:
                self._record_diagnostic(
                    f"stale Run cancellation failed: {_error_summary(error)}"
                )
            self._submit_stale_reap(handle)
            return

        self._starting = False
        self._handle = handle
        snapshot = handle.snapshot()
        self._validate_snapshot(handle, snapshot)
        if self._cancel_when_started:
            handle.cancel("scan panel stop requested before admission returned")
            snapshot = handle.snapshot()
            self._validate_snapshot(handle, snapshot)
        self._status = f"{snapshot.state.value} / {snapshot.phase} · NOT FINAL"

    def _poll_active_handle(self) -> None:
        handle = self._handle
        if handle is None or self._owner_reaped:
            return
        snapshot = handle.snapshot()
        self._validate_snapshot(handle, snapshot)
        if not snapshot.state.terminal:
            self._status = f"{snapshot.state.value} / {snapshot.phase} · NOT FINAL"
            return
        if self._terminal_work_inflight:
            return
        self._terminal_work_inflight = True
        token = _WorkToken(self._generation, handle.run_id)
        if snapshot.state is RunState.SUCCEEDED and snapshot.final_committed:
            self._status = "FINAL COMMITTED · RETRIEVING RESULT"
            self._submit("terminal-result", token, handle.result)
        else:
            self._status = f"{snapshot.state.value} · NOT FINAL · REAPING"
            self._submit("terminal-reap", token, handle.wait)

    def _accept_terminal_result(self, token: _WorkToken, future: Future) -> None:
        try:
            reference = future.result()
            if not isinstance(reference, ScanArtifactRef):
                raise TypeError("successful scan Run must return ScanArtifactRef")
        except BaseException as error:
            if self._matches_active(token):
                self._terminal_work_inflight = False
                self._owner_reaped = True
                self._status = "FINAL RESULT RETRIEVAL FAILED"
                self._record_diagnostic(
                    f"final result retrieval failed: {_error_summary(error)}"
                )
            return
        if not self._matches_active(token):
            return

        self._terminal_work_inflight = False
        self._owner_reaped = True
        self._artifact_ref = reference
        self._presentation = None
        self._projection_inflight = True
        self._status = "FINAL · BUILDING DISPLAY"
        application = self._application
        if application is None:
            self._projection_inflight = False
            self._status = "FINAL · DISPLAY FAILED"
            self._record_diagnostic("final display application is detached")
            return
        project_token = _WorkToken(token.generation, token.run_id, reference)
        self._submit(
            "project-final",
            project_token,
            lambda: application.project_final(
                reference,
                memory_limit_bytes=self._projection_memory_limit_bytes,
            ),
        )

    def _accept_terminal_reap(self, token: _WorkToken, future: Future) -> None:
        error: BaseException | None = None
        snapshot: RunSnapshot | None = None
        try:
            snapshot = future.result()
            if not isinstance(snapshot, RunSnapshot):
                raise TypeError("RunHandle.wait must return RunSnapshot")
        except BaseException as caught:
            error = caught
        if not self._matches_active(token):
            return
        self._terminal_work_inflight = False
        self._owner_reaped = True
        if error is not None:
            self._status = "RUN REAP FAILED · NOT FINAL"
            self._record_diagnostic(f"Run reap failed: {_error_summary(error)}")
            return
        assert snapshot is not None
        if snapshot.run_id != token.run_id:
            self._status = "RUN REAP FAILED · NOT FINAL"
            self._record_diagnostic("Run reap returned another run_id")
            return
        self._status = f"{snapshot.state.value} · NOT FINAL"
        self._record_snapshot_diagnostic(snapshot)

    def _accept_projection(self, token: _WorkToken, future: Future) -> None:
        error: BaseException | None = None
        presentation: FinalScanPresentation | None = None
        try:
            presentation = future.result()
            if not isinstance(presentation, FinalScanPresentation):
                raise TypeError("project_final must return FinalScanPresentation")
            if presentation.source_ref != token.artifact_ref:
                raise ValueError("final presentation belongs to another artifact")
            if (
                presentation.gui_decode_peak_nbytes
                > self._projection_memory_limit_bytes
            ):
                raise MemoryError(
                    "final PNG decode exceeds the projection memory limit"
                )
        except BaseException as caught:
            error = caught
        if not self._matches_active(token, require_artifact=True):
            return
        self._projection_inflight = False
        if error is not None:
            # The durable artifact remains authoritative even when display work fails.
            self._presentation = None
            self._status = "FINAL · DISPLAY FAILED"
            self._record_diagnostic(f"final display failed: {_error_summary(error)}")
            return
        self._presentation = presentation
        self._status = "FINAL"

    def _submit_stale_reap(self, handle: RunHandle) -> None:
        token = _WorkToken(self._generation)
        self._submit("stale-reap", token, handle.wait)

    def _accept_stale_reap(self, future: Future) -> None:
        try:
            future.result()
        except BaseException as error:
            self._record_diagnostic(f"detached Run reap failed: {_error_summary(error)}")

    def _matches_active(
        self,
        token: _WorkToken,
        *,
        require_artifact: bool = False,
    ) -> bool:
        if self._closing or token.generation != self._generation:
            return False
        handle = self._handle
        if handle is None or token.run_id != handle.run_id:
            return False
        if require_artifact:
            return (
                token.artifact_ref is not None
                and token.artifact_ref == self._artifact_ref
            )
        return True

    @staticmethod
    def _validate_handle(handle: object) -> None:
        run_id = getattr(handle, "run_id", None)
        if not isinstance(run_id, RunId):
            raise TypeError("ScanPanelApplication.start must return a RunHandle")
        for method in ("snapshot", "cancel", "wait", "result"):
            if not callable(getattr(handle, method, None)):
                raise TypeError("ScanPanelApplication.start must return a RunHandle")

    @staticmethod
    def _validate_snapshot(handle: RunHandle, snapshot: object) -> None:
        if not isinstance(snapshot, RunSnapshot):
            raise TypeError("RunHandle.snapshot must return RunSnapshot")
        if snapshot.run_id != handle.run_id:
            raise RuntimeError("RunHandle snapshot belongs to another run_id")

    def _record_snapshot_diagnostic(self, snapshot: RunSnapshot) -> None:
        parts = []
        if snapshot.primary_error:
            parts.append(snapshot.primary_error)
        parts.extend(snapshot.cleanup_errors)
        if snapshot.commit_recovery_warning:
            parts.append(snapshot.commit_recovery_warning)
        if snapshot.recovery_instruction:
            parts.append(snapshot.recovery_instruction)
        if parts:
            self._record_diagnostic(" · ".join(parts))

    def _record_diagnostic(self, message: str) -> None:
        message = str(message).strip()
        if not message:
            return
        self._diagnostic = (
            message
            if self._diagnostic is None
            else f"{self._diagnostic}\n{message}"
        )

    def _can_start(self) -> bool:
        return (
            not self._closing
            and not self._closed
            and not self._starting
            and not self._terminal_work_inflight
            and not self._projection_inflight
            and (self._handle is None or self._owner_reaped)
        )

    def _can_stop(self) -> bool:
        if self._closing or self._closed:
            return False
        if self._starting and self._handle is None:
            return not self._cancel_when_started
        handle = self._handle
        if handle is None or self._owner_reaped:
            return False
        return not handle.snapshot().state.terminal

    def _build_view_model(self) -> ScanPanelViewModel:
        with self._lock:
            worker_idle = not self._tracked and not self._mailbox
        return ScanPanelViewModel(
            generation=self._generation,
            status=self._status,
            run_id=None if self._handle is None else self._handle.run_id.value,
            artifact_ref=self._artifact_ref,
            presentation=self._presentation,
            diagnostic=self._diagnostic,
            can_start=self._can_start(),
            can_stop=self._can_stop(),
            worker_idle=worker_idle,
            closing=self._closing,
            closed=self._closed,
        )

    def _publish_model(self) -> None:
        self._view_model = self._build_view_model()

    def _maybe_finish_close(self) -> None:
        if not self._closing or self._closed:
            return
        with self._lock:
            if self._tracked or self._mailbox:
                return
        if self._owns_executor and not self._executor_closed:
            self._executor.shutdown(wait=False, cancel_futures=False)
        self._executor_closed = True
        self._application = None
        self._request_owner_wake = None
        self._closed = True
        self._status = "CLOSED"
        self._publish_model()

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("ScanPanelController methods require its owner thread")


__all__ = [
    "FinalScanPresentation",
    "ScanPanelApplication",
    "ScanPanelController",
    "ScanPanelViewModel",
]
