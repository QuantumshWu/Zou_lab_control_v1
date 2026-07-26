"""The TaskConsole's single worker-owned raster lane.

Qt freezes immutable requests and accepts completed fronts.  This owner keeps
every ``PanelComposer`` and Agg object on one worker thread, coalesces requests
that have not started, and never reads a card widget from that worker.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading

from zlc_data import FitResultBatch
from zlc_frontend import (
    FigureOutputFront,
    FigureOutputRequest,
    FigureOutputSession,
    PlotPanelComposeRequest,
    PlotPanelContract,
    FigureSource,
    PanelProvenance,
    PlotPanelSession,
)
from zlc_frontend.panel_render import FacetedPanelFocus
from zlc_frontend.qt_widgets import QtOwnerWake


@dataclass(frozen=True, slots=True)
class PanelRenderRequest:
    panel_id: str
    request_revision: int
    signature: object
    source_key: object
    frame_key: object
    value: object
    contract: PlotPanelContract
    source: FigureSource
    display: object
    provenance: PanelProvenance
    focus: FacetedPanelFocus | None
    fit_result: FitResultBatch | None = None
    fit_result_identity: str | None = None
    # One Figure may be presented by the live card and by an Edit tab frozen
    # at an older accepted input.  They share this lane and the same renderer
    # implementation, but each surface needs its own persistent Agg state.
    # ``panel_id`` remains the Figure's semantic identity; ``surface_id`` is
    # only the composition/presentation route.
    surface_id: str | None = None

    @property
    def render_surface_id(self) -> str:
        return self.panel_id if self.surface_id is None else self.surface_id


@dataclass(frozen=True, slots=True)
class PanelFigureOutputRequest:
    """One Figure's immutable derived-output intent and publication source."""

    panel_id: str
    owner_token: object
    request_revision: int
    source_value: object | None
    request: FigureOutputRequest | None

    def __post_init__(self) -> None:
        identity = str(self.panel_id).strip()
        if not identity:
            raise ValueError("Figure output panel_id must not be empty")
        if int(self.request_revision) < 1:
            raise ValueError("Figure output request_revision must be positive")
        if self.request is None:
            if self.source_value is not None:
                raise ValueError("empty Figure output request cannot carry a source")
        elif not isinstance(self.request, FigureOutputRequest):
            raise TypeError("Figure output work requires FigureOutputRequest")
        object.__setattr__(self, "panel_id", identity)
        object.__setattr__(self, "request_revision", int(self.request_revision))


@dataclass(frozen=True, slots=True)
class ConsoleRenderCompletion:
    renders: tuple[tuple[object, object, object, object, str | None], ...]
    figure_outputs: tuple[
        tuple[PanelFigureOutputRequest, FigureOutputFront | None, str | None], ...
    ]


class ConsoleRenderLane:
    """Serialize panel composition and return completions on the Qt owner."""

    def __init__(
        self,
        qt_parent,
        *,
        accept_completion: Callable[[object], set[str]],
        request_shutdown_wake: Callable[[], None],
    ) -> None:
        if not callable(accept_completion):
            raise TypeError("accept_completion must be callable")
        if not callable(request_shutdown_wake):
            raise TypeError("request_shutdown_wake must be callable")
        self._accept_completion = accept_completion
        self._request_shutdown_wake = request_shutdown_wake
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="zlc-task-console-raster",
        )
        self._lock = threading.Lock()
        self._future: Future | None = None
        self._completion = None
        self._pending: dict[str, PanelRenderRequest] = {}
        self._pending_outputs: dict[str, PanelFigureOutputRequest] = {}
        self._reset_pending: set[str] = set()
        self._worker_composers: dict[str, tuple[object, object]] = {}
        self._worker_output_sessions: dict[str, FigureOutputSession] = {}
        self._accepting_completion = False
        self._closing = False
        self._shutdown_future: Future | None = None
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

    @property
    def idle(self) -> bool:
        with self._lock:
            return (
                self._future is None
                and self._completion is None
                and not self._pending
                and not self._pending_outputs
            )

    def enqueue(self, requests: tuple[PanelRenderRequest, ...]) -> None:
        if not requests or self._closing:
            return
        with self._lock:
            if (
                self._future is not None
                or self._completion is not None
                or self._accepting_completion
            ):
                for request in requests:
                    self._pending[request.render_surface_id] = request
                return
        self._start(requests, (), ())

    def enqueue_outputs(
        self,
        requests: tuple[PanelFigureOutputRequest, ...],
    ) -> None:
        """Evaluate Figure-derived datasets on this lane's worker owner."""

        if not requests or self._closing:
            return
        with self._lock:
            if (
                self._future is not None
                or self._completion is not None
                or self._accepting_completion
            ):
                for request in requests:
                    self._pending_outputs[request.panel_id] = request
                return
        self._start((), requests, ())

    def forget(self, surface_id: str) -> None:
        """Dispose worker state for one retired presentation surface."""

        start_reset = False
        with self._lock:
            self._pending.pop(surface_id, None)
            self._pending_outputs.pop(surface_id, None)
            if (
                self._future is None
                and self._completion is None
                and not self._accepting_completion
            ):
                start_reset = True
            else:
                self._reset_pending.add(surface_id)
        if start_reset:
            self._start((), (), (surface_id,))

    def _start(
        self,
        requests: tuple[PanelRenderRequest, ...],
        output_requests: tuple[PanelFigureOutputRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ) -> None:
        if self._closing:
            return
        future = self._pool.submit(
            self._compose,
            requests,
            output_requests,
            reset_panel_ids,
        )
        with self._lock:
            if self._future is not None:
                raise RuntimeError("TaskConsole render lane admitted overlapping batches")
            self._future = future
        future.add_done_callback(self._finished)

    def _compose(
        self,
        requests: tuple[PanelRenderRequest, ...],
        output_requests: tuple[PanelFigureOutputRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ):
        from zlc_frontend.panel_render import PanelRenderError

        for surface_id in reset_panel_ids:
            owned = self._worker_composers.pop(surface_id, None)
            if owned is not None:
                owned[1].close()
            output_session = self._worker_output_sessions.pop(surface_id, None)
            if output_session is not None:
                output_session.close()

        results = []
        for request in requests:
            surface_id = request.render_surface_id
            owned = self._worker_composers.get(surface_id)
            if owned is None or owned[0] != request.source_key:
                if owned is not None:
                    owned[1].close()
                composer = PlotPanelSession(request.contract)
                self._worker_composers[surface_id] = (
                    request.source_key,
                    composer,
                )
            else:
                composer = owned[1]
            try:
                composed = composer.compose(
                    PlotPanelComposeRequest(
                        request.source,
                        request.display,
                        request.provenance,
                        focus=request.focus,
                        fit_result=request.fit_result,
                        fit_result_identity=request.fit_result_identity,
                    )
                )
                frame = composed.frame
                faceted_result = composed.faceted
                figure = composed.figure
            except PanelRenderError as error:
                results.append((request, None, None, None, str(error)))
            except BaseException as error:
                # Never retain a traceback that may own the dataset and Agg graph.
                results.append(
                    (
                        request,
                        None,
                        None,
                        None,
                        f"{type(error).__name__}: {error}",
                    )
                )
            else:
                results.append((request, frame, faceted_result, figure, None))
        output_results = []
        for request in output_requests:
            if request.request is None:
                output_results.append((request, FigureOutputFront({}), None))
                continue
            session = self._worker_output_sessions.get(request.panel_id)
            if session is None:
                session = FigureOutputSession()
                self._worker_output_sessions[request.panel_id] = session
            try:
                front = session.evaluate(request.request)
            except BaseException as error:
                output_results.append(
                    (request, None, f"{type(error).__name__}: {error}")
                )
            else:
                output_results.append((request, front, None))
        return ConsoleRenderCompletion(tuple(results), tuple(output_results))

    def _finished(self, future: Future) -> None:
        try:
            completion = future.result()
        except BaseException as error:
            completion = f"{type(error).__name__}: {error}"
        with self._lock:
            if self._future is not future:
                raise RuntimeError("TaskConsole render future identity changed")
            self._future = None
            if self._closing:
                return
            self._completion = completion
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
            self._accepting_completion = True

        try:
            reset = set(self._accept_completion(completion))
        finally:
            with self._lock:
                self._accepting_completion = False
                reset_pending = set(self._reset_pending)
                self._reset_pending.clear()
                pending = tuple(self._pending.values())
                self._pending.clear()
                pending_outputs = tuple(self._pending_outputs.values())
                self._pending_outputs.clear()
        reset.update(reset_pending)
        # A result superseded by a newer request for the same panel is stale
        # to Qt, not stale to the worker-owned renderer.  The pending request's
        # ``source_key`` below is the authority: an equal key reuses the Agg
        # surface/blit cache; a different key replaces it in ``_compose``.
        # Closing the composer merely because a fast wheel gesture overtook one
        # raster answer made every zoom burst pay first-frame construction and
        # permanently prevented the steady blit path from warming up.
        reset.difference_update(
            request.render_surface_id for request in pending
        )
        reset.difference_update(request.panel_id for request in pending_outputs)
        if pending or pending_outputs or reset:
            self._start(pending, pending_outputs, tuple(sorted(reset)))

    def _release_worker_state(self) -> tuple[str, ...]:
        """Release every frontend session on the serial worker that owns it."""

        failures: list[str] = []
        for _source_key, composer in tuple(self._worker_composers.values()):
            try:
                composer.close()
            except BaseException as error:
                failures.append(f"{type(error).__name__}: {error}")
        self._worker_composers.clear()
        for session in tuple(self._worker_output_sessions.values()):
            try:
                session.close()
            except BaseException as error:
                failures.append(f"{type(error).__name__}: {error}")
        self._worker_output_sessions.clear()
        return tuple(failures)

    def _shutdown_finished(self, future: Future) -> None:
        try:
            failures = future.result()
        except BaseException as error:
            failures = (f"{type(error).__name__}: {error}",)
        with self._lock:
            if self._shutdown_future is not future:
                raise RuntimeError("TaskConsole render shutdown future identity changed")
            self._shutdown_future = None
            self._shutdown_failures = tuple(failures)
            self._shutdown_complete = True
        self._wake.request_owner_wake()

    def shutdown(self) -> bool:
        """Begin serial worker retirement and report when release has completed.

        The Qt owner never waits for a compose.  The release task is queued on
        the same single-worker executor, so it necessarily runs after an
        already-started compose and closes every Agg/session on its creating
        thread.  ``True`` is returned only after that task has finished.
        """

        with self._lock:
            if self._closing:
                return self._shutdown_complete
            self._closing = True
            self._pending.clear()
            self._pending_outputs.clear()
            self._reset_pending.clear()
            self._completion = None
            has_worker_state = bool(
                self._future is not None
                or self._worker_composers
                or self._worker_output_sessions
            )

        if not has_worker_state:
            # ThreadPoolExecutor is lazy; with no submitted/current work there
            # is no worker-owned graph to retire and no thread to join.
            self._pool.shutdown(wait=False)
            with self._lock:
                self._shutdown_complete = True
                self._shutdown_notified = True
            self._wake.detach()
            return True

        future = self._pool.submit(self._release_worker_state)
        with self._lock:
            self._shutdown_future = future
        future.add_done_callback(self._shutdown_finished)
        self._pool.shutdown(wait=False)
        return False
