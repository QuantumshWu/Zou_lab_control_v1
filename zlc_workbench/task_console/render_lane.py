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
    request_revision: int
    signature: object
    source_key: object
    frame_key: object
    value: object
    display: object
    intent: object
    label: str
    size: tuple[int, int]
    provenance: object
    view: object
    faceted: bool
    focus: object


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
                    self._pending[request.panel_id] = request
                return
        self._start(requests, ())

    def forget(self, panel_id: str) -> None:
        """Dispose worker state for one removed/rebound panel in lane order."""

        start_reset = False
        with self._lock:
            self._pending.pop(panel_id, None)
            if self._future is None and self._completion is None:
                start_reset = True
            else:
                self._reset_pending.add(panel_id)
        if start_reset:
            self._start((), (panel_id,))

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
        from zlc_frontend.panel_render import PanelComposer, PanelRenderError

        for panel_id in reset_panel_ids:
            owned = self._worker_composers.pop(panel_id, None)
            if owned is not None:
                owned[1].close()

        results = []
        for request in requests:
            owned = self._worker_composers.get(request.panel_id)
            if owned is None or owned[0] != request.source_key:
                if owned is not None:
                    owned[1].close()
                composer = PanelComposer(
                    request.panel_id,
                    intent=request.intent,
                    size=request.size,
                    label=request.label,
                    view=request.view,
                )
                self._worker_composers[request.panel_id] = (
                    request.source_key,
                    composer,
                )
            else:
                composer = owned[1]
            try:
                if request.faceted:
                    faceted_result = composer.compose_faceted(
                        request.value.snapshot,
                        display=request.display,
                        provenance=request.provenance,
                        focus=request.focus,
                    )
                    frame = None
                    document = faceted_result.figure.document
                else:
                    frame = composer.compose(
                        request.value.snapshot,
                        display=request.display,
                        provenance=request.provenance,
                    )
                    faceted_result = None
                    document = composer.document_for(request.value.snapshot.block.schema)
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
                results.append((request, frame, faceted_result, document, None))
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
