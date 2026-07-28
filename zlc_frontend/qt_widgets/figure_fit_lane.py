"""Figure-owned Fit preparation and one reusable Qt-owned solver lane.

The Figure owns the fit request and publishes its parameters.  This module has
no QWidget, Measurement, ROI processor, artifact window, or persistence
authority.  Qt freezes an exact source/spec request; the worker executes the
existing named-axis ``zlc_data`` fit; Qt may accept the immutable result only
if the panel still shows that source revision.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading

from zlc_data import (
    FitResultBatch,
    FitSpec,
    OwnedSnapshot,
    bind_fit,
)
from .owner_wake import QtOwnerWake


__all__ = [
    "FigureFitLane",
    "FigureFitRequest",
]


@dataclass(frozen=True, slots=True)
class FigureFitRequest:
    panel_id: str
    request_revision: int
    source: object
    spec: FitSpec
    cancelled: threading.Event = field(
        default_factory=threading.Event,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        identity = str(self.panel_id).strip()
        if not identity:
            raise ValueError("panel_id must not be empty")
        object.__setattr__(self, "panel_id", identity)
        if (
            isinstance(self.request_revision, bool)
            or not isinstance(self.request_revision, int)
            or self.request_revision <= 0
        ):
            raise ValueError("fit request_revision must be positive int")
        snapshot = getattr(self.source, "snapshot", None)
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("panel Fit source must own an immutable snapshot")
        if not isinstance(self.spec, FitSpec):
            raise TypeError("panel Fit request requires FitSpec")
        if (
            self.spec.committed_transform.source_schema_fingerprint
            != snapshot.ref.schema_fingerprint
        ):
            raise ValueError("panel Fit spec belongs to another source schema")


class FigureFitLane:
    """One serial worker for explicit/continuous Figure fit requests.

    A panel may replace its not-yet-started request with a newer source
    revision.  Work already executing is allowed to finish; the panel's exact
    revision check rejects it if the visible source has advanced.  This avoids
    starvation when a camera publishes faster than a fit while never showing
    or publishing a stale result.
    """

    def __init__(
        self,
        qt_parent,
        *,
        accept_completion,
        request_shutdown_wake,
    ) -> None:
        if not callable(accept_completion):
            raise TypeError("accept_completion must be callable")
        if not callable(request_shutdown_wake):
            raise TypeError("request_shutdown_wake must be callable")
        self._accept_completion = accept_completion
        self._request_shutdown_wake = request_shutdown_wake
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="zlc-task-console-fit",
        )
        self._lock = threading.Lock()
        self._future: Future | None = None
        self._active: tuple[FigureFitRequest, object] | None = None
        self._completion = None
        self._pending: dict[str, tuple[FigureFitRequest, object]] = {}
        self._closing = False
        self._shutdown_complete = False
        self._shutdown_notified = False
        self._shutdown_failures: tuple[str, ...] = ()
        self._wake = QtOwnerWake(qt_parent)
        self._wake.bind(self._owner_cycle)

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def shutdown_complete(self) -> bool:
        with self._lock:
            return self._shutdown_complete

    @property
    def shutdown_failures(self) -> tuple[str, ...]:
        with self._lock:
            return self._shutdown_failures

    def enqueue(self, request: FigureFitRequest, completion_token: object) -> None:
        if not isinstance(request, FigureFitRequest):
            raise TypeError("fit lane requires FigureFitRequest")
        with self._lock:
            if self._closing:
                return
            busy = self._future is not None or self._completion is not None
            if busy:
                previous = self._pending.get(request.panel_id)
                self._pending[request.panel_id] = (request, completion_token)
                if previous is not None:
                    previous[0].cancelled.set()
                return
        self._start(request, completion_token)

    def forget(self, panel_id: str) -> None:
        identity = str(panel_id).strip()
        with self._lock:
            pending = self._pending.pop(identity, None)
            active = self._active
        if pending is not None:
            pending[0].cancelled.set()
        if active is not None and active[0].panel_id == identity:
            active[0].cancelled.set()

    def _start(self, request: FigureFitRequest, completion_token: object) -> None:
        with self._lock:
            if self._closing:
                request.cancelled.set()
                return
            if self._future is not None:
                raise RuntimeError("Fit lane admitted overlapping work")
            future = self._pool.submit(self._execute, request)
            self._active = (request, completion_token)
            self._future = future
        # Preserve the immutable request identity outside the Future result.
        # Even an unexpected executor/Future failure must still be routable to
        # the one panel whose exact live surface is pinned by this command.
        future.add_done_callback(
            lambda completed, submitted=request, token=completion_token: self._finished(
                submitted,
                token,
                completed,
            )
        )

    @staticmethod
    def _execute(request: FigureFitRequest):
        snapshot = request.source.snapshot
        try:
            result = bind_fit(request.spec, snapshot.block.schema).run(
                snapshot,
                cancel_check=request.cancelled.is_set,
            )
        except BaseException as error:
            return None, f"{type(error).__name__}: {error}"
        if not isinstance(result, FitResultBatch):
            return None, "TypeError: fit engine returned another result type"
        return result, None

    def _finished(
        self,
        request: FigureFitRequest,
        completion_token: object,
        future: Future,
    ) -> None:
        try:
            result, error = future.result()
        except BaseException as error:
            result = None
            error = f"{type(error).__name__}: {error}"
        completion = (completion_token, request, result, error)
        with self._lock:
            if self._future is not future:
                raise RuntimeError("Figure Fit future identity changed")
            active = self._active
            if (
                active is None
                or active[0] is not request
                or active[1] is not completion_token
            ):
                raise RuntimeError("Figure Fit request identity changed")
            self._future = None
            self._active = None
            if self._closing:
                self._completion = None
                # Cancellation and an in-flight solver diagnostic are both
                # already represented by the discarded completion.  The
                # ownership fact needed by close is that the worker returned
                # and no longer retains the request/snapshot.
                self._shutdown_failures = ()
                self._shutdown_complete = True
                notify_shutdown = True
            else:
                self._completion = completion
                notify_shutdown = False
        if notify_shutdown:
            self._wake.request_owner_wake()
            return
        self._wake.request_owner_wake()

    def _owner_cycle(self) -> None:
        if self._closing:
            with self._lock:
                ready = self._shutdown_complete and not self._shutdown_notified
                if ready:
                    self._shutdown_notified = True
            if ready:
                self._wake.detach()
                self._request_shutdown_wake()
            return
        with self._lock:
            completion = self._completion
            if completion is None:
                return
            self._completion = None
        self._accept_completion(completion)
        with self._lock:
            if self._closing or not self._pending:
                return
            panel_id = next(iter(self._pending))
            request, completion_token = self._pending.pop(panel_id)
        self._start(request, completion_token)

    def shutdown(self) -> bool:
        """Cancel admission and report only after active solver work retires."""

        with self._lock:
            if self._closing:
                return self._shutdown_complete
            self._closing = True
            active = self._active
            pending = tuple(self._pending.values())
            self._pending.clear()
            self._completion = None
            active_future = self._future
        if active is not None:
            active[0].cancelled.set()
        for request, _completion_token in pending:
            request.cancelled.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
        if active_future is not None:
            return False
        with self._lock:
            self._active = None
            self._shutdown_complete = True
            self._shutdown_notified = True
        self._wake.detach()
        return True
