"""Reusable Figure-surface worker lanes for raster and selector outputs.

Qt freezes immutable requests and accepts completed fronts.  This owner keeps
every ``PanelComposer`` and Agg object on one render thread, evaluates selector
materializations on an independent worker, coalesces work that has not started,
and never reads a card widget from either worker.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from types import MappingProxyType

from zlc_data import FitResultBatch
from zlc_frontend import (
    FigureAreaCommit,
    FigureCrossCommit,
    FigureSource,
    PlotPanelComposeRequest,
    PlotPanelContract,
    PanelProvenance,
    PlotPanelSession,
    materialize_area_outputs,
    materialize_cross_outputs,
)
from zlc_frontend.panel_render import FacetedPanelFocus
from .owner_wake import QtOwnerWake


__all__ = [
    "FigureSurfaceCompletion",
    "FigureSurfaceLane",
    "FigureSurfaceRenderRequest",
]


@dataclass(frozen=True, slots=True)
class FigureSurfaceRenderRequest:
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
class _SelectorWork:
    route_key: object
    operation_token: object
    source: FigureSource
    commit: FigureAreaCommit | FigureCrossCommit


@dataclass(frozen=True, slots=True)
class FigureSurfaceCompletion:
    renders: tuple[tuple[object, object, object, object, str | None], ...]
    selector_outputs: tuple[tuple[object, object | None, str | None], ...]


class FigureSurfaceLane:
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
        self._selector_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="zlc-task-console-selector",
        )
        self._lock = threading.Lock()
        self._future: Future | None = None
        self._completion = None
        self._pending: deque[tuple[FigureSurfaceRenderRequest, ...]] = deque()
        self._reset_pending: set[str] = set()
        self._worker_composers: dict[str, tuple[object, object]] = {}
        self._selector_future: Future | None = None
        self._selector_active: _SelectorWork | None = None
        self._selector_completion = None
        self._selector_pending: dict[object, _SelectorWork] = {}
        self._selector_order: deque[object] = deque()
        self._accepting_completion = False
        self._closing = False
        self._shutdown_future: Future | None = None
        self._render_retired = False
        self._selector_retired = False
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
                and self._selector_future is None
                and self._selector_completion is None
                and not self._selector_pending
            )

    def enqueue(self, requests: tuple[FigureSurfaceRenderRequest, ...]) -> None:
        if not requests or self._closing:
            return
        surface_ids = tuple(request.render_surface_id for request in requests)
        if len(surface_ids) != len(set(surface_ids)):
            raise ValueError("one render group cannot repeat a presentation surface")
        with self._lock:
            if (
                self._future is not None
                or self._completion is not None
                or self._accepting_completion
            ):
                replaced = set(surface_ids)
                self._pending = deque(
                    group
                    for group in self._pending
                    if replaced.isdisjoint(
                        request.render_surface_id for request in group
                    )
                )
                self._pending.append(requests)
                return
        self._start(requests, ())

    def enqueue_selector(
        self,
        route_key: object,
        operation_token: object,
        source: FigureSource,
        commit: FigureAreaCommit | FigureCrossCommit,
    ) -> None:
        """Evaluate one Area or Cross commit independently from raster work.

        ``operation_token`` is opaque to the frontend.  The composition root
        freezes the exact publication/generation facts inside that token and
        is solely responsible for admitting or rejecting the completion.
        Pending work is latest-only per route while distinct routes retain
        first-arrival order.
        """

        if not isinstance(source, FigureSource):
            raise TypeError("selector source must be FigureSource")
        if not isinstance(commit, (FigureAreaCommit, FigureCrossCommit)):
            raise TypeError("selector work requires exactly one Area or Cross commit")
        try:
            hash(route_key)
        except TypeError as error:
            raise TypeError("selector route_key must be hashable") from error
        work = _SelectorWork(route_key, operation_token, source, commit)
        with self._lock:
            if self._closing:
                return
            active = self._selector_active
            if (
                active is not None
                and active.route_key == route_key
                and active.operation_token == operation_token
            ):
                return
            queued = self._selector_pending.get(route_key)
            if queued is not None and queued.operation_token == operation_token:
                return
            busy = (
                self._selector_future is not None
                or self._selector_completion is not None
                or self._accepting_completion
            )
            if busy:
                if route_key not in self._selector_pending:
                    self._selector_order.append(route_key)
                self._selector_pending[route_key] = work
                return
        self._start_selector(work)

    def forget(self, surface_id: str) -> None:
        """Dispose worker state for one retired presentation surface."""

        start_reset = False
        with self._lock:
            self._pending = deque(
                group
                for group in self._pending
                if all(
                    request.render_surface_id != surface_id
                    for request in group
                )
            )
            if (
                self._future is None
                and self._completion is None
                and not self._accepting_completion
            ):
                start_reset = True
            else:
                self._reset_pending.add(surface_id)
        if start_reset:
            self._start((), (surface_id,))

    def forget_selector(self, route_key: object) -> None:
        """Discard selector work that has not started for one retired route."""

        with self._lock:
            self._selector_pending.pop(route_key, None)

    def _start(
        self,
        requests: tuple[FigureSurfaceRenderRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ) -> None:
        with self._lock:
            if self._closing:
                return
            if self._future is not None:
                raise RuntimeError("Figure surface lane admitted overlapping batches")
            future = self._pool.submit(
                self._compose,
                requests,
                reset_panel_ids,
            )
            self._future = future
        future.add_done_callback(self._finished)

    def _compose(
        self,
        requests: tuple[FigureSurfaceRenderRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ):
        from zlc_frontend.panel_render import PanelRenderError

        for surface_id in reset_panel_ids:
            owned = self._worker_composers.pop(surface_id, None)
            if owned is not None:
                owned[1].close()

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
        return tuple(results)

    def _start_selector(self, work: _SelectorWork) -> None:
        with self._lock:
            if self._closing:
                return
            if self._selector_future is not None:
                raise RuntimeError("selector lane admitted overlapping work")
            future = self._selector_pool.submit(self._evaluate_selector, work)
            self._selector_active = work
            self._selector_future = future
        future.add_done_callback(
            lambda completed, submitted=work: self._selector_finished(
                submitted,
                completed,
            )
        )

    @staticmethod
    def _evaluate_selector(work: _SelectorWork):
        try:
            if isinstance(work.commit, FigureAreaCommit):
                outputs = materialize_area_outputs(work.source, work.commit)
            else:
                outputs = materialize_cross_outputs(work.source, work.commit)
        except BaseException as error:
            # Do not retain a traceback that may own a large source snapshot.
            return (
                work.operation_token,
                None,
                f"{type(error).__name__}: {error}",
            )
        return (
            work.operation_token,
            MappingProxyType(dict(outputs)),
            None,
        )

    def _selector_finished(self, work: _SelectorWork, future: Future) -> None:
        try:
            completion = future.result()
        except BaseException as error:
            completion = (
                work.operation_token,
                None,
                f"{type(error).__name__}: {error}",
            )
        with self._lock:
            if self._selector_future is not future:
                raise RuntimeError("selector future identity changed")
            if self._selector_active is not work:
                raise RuntimeError("selector work identity changed")
            self._selector_future = None
            self._selector_active = None
            if self._closing:
                self._selector_completion = None
                self._selector_retired = True
                self._complete_shutdown_if_ready_locked()
            else:
                self._selector_completion = completion
        self._wake.request_owner_wake()

    def _finished(self, future: Future) -> None:
        try:
            completion = future.result()
        except BaseException as error:
            completion = f"{type(error).__name__}: {error}"
        with self._lock:
            if self._future is not future:
                raise RuntimeError("Figure surface render future identity changed")
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
            render_completion = self._completion
            selector_completion = self._selector_completion
            if render_completion is None and selector_completion is None:
                return
            self._completion = None
            self._selector_completion = None
            self._accepting_completion = True

        payloads: list[object] = []
        if isinstance(render_completion, str):
            payloads.append(render_completion)
            render_results = ()
        else:
            render_results = () if render_completion is None else render_completion
        if render_results or selector_completion is not None:
            payloads.append(
                FigureSurfaceCompletion(
                    tuple(render_results),
                    (
                        ()
                        if selector_completion is None
                        else (selector_completion,)
                    ),
                )
            )

        reset: set[str] = set()
        try:
            for payload in payloads:
                accepted = self._accept_completion(payload)
                if accepted is not None:
                    reset.update(accepted)
        finally:
            with self._lock:
                self._accepting_completion = False
                # A selector may finish while raster composition is still in
                # flight (and vice versa).  Drain only the lane whose worker
                # and unaccepted completion are both absent.
                if self._future is None and self._completion is None:
                    reset_pending = set(self._reset_pending)
                    self._reset_pending.clear()
                    pending = self._pending.popleft() if self._pending else ()
                else:
                    reset_pending = set()
                    pending = ()
                selector_work = self._pop_selector_pending_locked()
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
        if pending or reset:
            self._start(pending, tuple(sorted(reset)))
        if selector_work is not None:
            self._start_selector(selector_work)

    def _pop_selector_pending_locked(self) -> _SelectorWork | None:
        if (
            self._selector_future is not None
            or self._selector_completion is not None
        ):
            return None
        while self._selector_order:
            route_key = self._selector_order.popleft()
            work = self._selector_pending.pop(route_key, None)
            if work is not None:
                return work
        return None

    def _release_worker_state(self) -> tuple[str, ...]:
        """Release every frontend session on the serial worker that owns it."""

        failures: list[str] = []
        for _source_key, composer in tuple(self._worker_composers.values()):
            try:
                composer.close()
            except BaseException as error:
                failures.append(f"{type(error).__name__}: {error}")
        self._worker_composers.clear()
        return tuple(failures)

    def _complete_shutdown_if_ready_locked(self) -> None:
        if self._render_retired and self._selector_retired:
            self._shutdown_complete = True

    def _shutdown_finished(self, future: Future) -> None:
        try:
            failures = future.result()
        except BaseException as error:
            failures = (f"{type(error).__name__}: {error}",)
        with self._lock:
            if self._shutdown_future is not future:
                raise RuntimeError("Figure surface shutdown future identity changed")
            self._shutdown_future = None
            self._shutdown_failures = tuple(failures)
            self._render_retired = True
            self._complete_shutdown_if_ready_locked()
        self._wake.request_owner_wake()

    def shutdown(self) -> bool:
        """Begin worker retirement and report when release has completed.

        The Qt owner never waits for a compose.  The release task is queued on
        the same single-worker executor, so it necessarily runs after an
        already-started compose and closes every Agg session on its creating
        thread.  Stateless selector work retires independently. ``True`` is
        returned only after both workers have released their references.
        """

        with self._lock:
            if self._closing:
                return self._shutdown_complete
            self._closing = True
            self._pending.clear()
            self._reset_pending.clear()
            self._completion = None
            self._selector_pending.clear()
            self._selector_order.clear()
            self._selector_completion = None
            has_render_state = bool(
                self._future is not None
                or self._worker_composers
            )
            has_selector_work = self._selector_future is not None
            self._render_retired = not has_render_state
            self._selector_retired = not has_selector_work

        if has_render_state:
            future = self._pool.submit(self._release_worker_state)
            with self._lock:
                self._shutdown_future = future
            future.add_done_callback(self._shutdown_finished)
        else:
            # ThreadPoolExecutor is lazy; with no submitted/current work there
            # is no worker-owned graph to retire and no thread to join.
            self._pool.shutdown(wait=False)
        if has_render_state:
            self._pool.shutdown(wait=False)
        self._selector_pool.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            self._complete_shutdown_if_ready_locked()
            complete = self._shutdown_complete
            if complete:
                self._shutdown_notified = True
        if complete:
            self._wake.detach()
        return complete
