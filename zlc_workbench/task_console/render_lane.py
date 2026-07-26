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

from zlc_frontend.qt_widgets import QtOwnerWake


@dataclass(frozen=True, slots=True)
class PanelRenderRequest:
    panel_id: str
    kind: str
    request_revision: int
    signature: object
    source_key: object
    frame_key: object
    value: object
    display: object
    intent: object
    label: str
    value_label: str
    size: tuple[int, int]
    size_name: str
    pixel_ratio: float
    provenance: object
    view: object
    faceted: bool
    focus: object
    fit_result: object | None = None
    fit_result_identity: str | None = None
    rolling_distribution: bool = False
    # One Figure may be presented by the live card and by an Edit tab frozen
    # at an older accepted input.  They share this lane and the same renderer
    # implementation, but each surface needs its own persistent Agg state.
    # ``panel_id`` remains the Figure's semantic identity; ``surface_id`` is
    # only the composition/presentation route.
    surface_id: str | None = None

    @property
    def render_surface_id(self) -> str:
        return self.panel_id if self.surface_id is None else self.surface_id


class ConsoleRenderLane:
    """Serialize panel composition and return completions on the Qt owner."""

    def __init__(
        self,
        qt_parent,
        *,
        accept_completion: Callable[[object], set[str]],
    ) -> None:
        self._accept_completion = accept_completion
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="zlc-task-console-raster",
        )
        self._lock = threading.Lock()
        self._future: Future | None = None
        self._completion = None
        self._pending: dict[str, PanelRenderRequest] = {}
        self._reset_pending: set[str] = set()
        self._worker_composers: dict[str, tuple[object, object]] = {}
        self._closing = False
        self._wake = QtOwnerWake(qt_parent)
        self._wake.bind(self._owner_cycle)

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def idle(self) -> bool:
        with self._lock:
            return (
                self._future is None
                and self._completion is None
                and not self._pending
            )

    def enqueue(self, requests: tuple[PanelRenderRequest, ...]) -> None:
        if not requests or self._closing:
            return
        with self._lock:
            if self._future is not None or self._completion is not None:
                for request in requests:
                    self._pending[request.render_surface_id] = request
                return
        self._start(requests, ())

    def forget(self, surface_id: str) -> None:
        """Dispose worker state for one retired presentation surface."""

        start_reset = False
        with self._lock:
            self._pending.pop(surface_id, None)
            if self._future is None and self._completion is None:
                start_reset = True
            else:
                self._reset_pending.add(surface_id)
        if start_reset:
            self._start((), (surface_id,))

    def _start(
        self,
        requests: tuple[PanelRenderRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ) -> None:
        if self._closing:
            return
        future = self._pool.submit(self._compose, requests, reset_panel_ids)
        with self._lock:
            if self._future is not None:
                raise RuntimeError("TaskConsole render lane admitted overlapping batches")
            self._future = future
        future.add_done_callback(self._finished)

    def _compose(
        self,
        requests: tuple[PanelRenderRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ):
        from zlc_frontend.site_map import SiteMapPresentation
        from zlc_frontend.site_map_render import SiteMapComposer
        from zlc_frontend.panel_render import PanelComposer, PanelRenderError

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
                if request.kind == "sites":
                    composer = SiteMapComposer(
                        request.panel_id,
                        size=request.size,
                        size_name=request.size_name,
                        pixel_ratio=request.pixel_ratio,
                        title=request.label,
                        value_label=request.value_label,
                    )
                else:
                    composer = PanelComposer(
                        request.panel_id,
                        intent=request.intent,
                        size=request.size,
                        size_name=request.size_name,
                        pixel_ratio=request.pixel_ratio,
                        label=request.label,
                        value_label=request.value_label,
                        view=request.view,
                        rolling_trace=request.kind == "monitor",
                        rolling_distribution=request.rolling_distribution,
                    )
                self._worker_composers[surface_id] = (
                    request.source_key,
                    composer,
                )
            else:
                composer = owned[1]
            try:
                if request.kind == "sites":
                    presentation = getattr(request.value, "presentation", None)
                    if not isinstance(presentation, SiteMapPresentation):
                        raise PanelRenderError(
                            "Site map requires one typed physical SiteMap view"
                        )
                    frame = composer.compose(
                        presentation,
                        display=request.display,
                    )
                    faceted_result = None
                    figure = None
                elif request.faceted:
                    faceted_result = composer.compose_faceted(
                        request.value.snapshot,
                        display=request.display,
                        provenance=request.provenance,
                        focus=request.focus,
                        fit_result=request.fit_result,
                        fit_result_identity=request.fit_result_identity,
                    )
                    frame = None
                    figure = faceted_result.figure
                else:
                    frame, figure = composer.compose_with_figure(
                        request.value.snapshot,
                        display=request.display,
                        provenance=request.provenance,
                        fit_result=request.fit_result,
                        fit_result_identity=request.fit_result_identity,
                    )
                    faceted_result = None
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

    def _finished(self, future: Future) -> None:
        try:
            completion = future.result()
        except BaseException as error:
            completion = f"{type(error).__name__}: {error}"
        with self._lock:
            if self._closing:
                return
            self._completion = completion
        self._wake.request_owner_wake()

    def _owner_cycle(self) -> None:
        if self._closing:
            return
        with self._lock:
            completion = self._completion
            if completion is None:
                return
            self._completion = None
            self._future = None

        reset = set(self._accept_completion(completion))
        with self._lock:
            reset.update(self._reset_pending)
            self._reset_pending.clear()
            pending = tuple(self._pending.values())
            self._pending.clear()
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

    def shutdown(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._wake.detach()
        with self._lock:
            self._pending.clear()
            self._completion = None

        def release_worker_state() -> None:
            for _source_key, composer in tuple(self._worker_composers.values()):
                composer.close()
            self._worker_composers.clear()

        self._pool.submit(release_worker_state)
        self._pool.shutdown(wait=False)
