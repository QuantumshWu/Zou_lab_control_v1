"""Finite exact-capture live preview composition for the target Workbench."""

from __future__ import annotations

import threading
from typing import Callable

from zlc_frontend.figure import (
    DatasetId,
    EvaluatedImage,
    EvaluatedInput,
    FigureDocument,
    FigureEvaluationPolicy,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    validate_view_spec,
)
from zlc_frontend.image_raster import rasterize_image_gray8
from zlc_frontend.render import (
    BoardFrame,
    CoherenceStamp,
    PanelFrame,
    PanelPresentationIdentity,
    SourceIdentity,
    detached_render_fault,
)
from zlc_neutral_atom.runtime.dataset import MonitorDataset, MonitorDatasetSnapshot
from zlc_neutral_atom.runtime.pipeline import CapturePreviewSpec
from zlc_storage import canonical_text

from .workspace import BoardController, BoardPublishPort, PanelSourceBinding


class LiveDatasetSlot:
    """One capacity-one materializer handle plus coalesced revision notices."""

    def __init__(
        self,
        spec: CapturePreviewSpec,
        *,
        dataset_id: DatasetId,
        evaluation_policy: FigureEvaluationPolicy,
    ) -> None:
        if not isinstance(spec, CapturePreviewSpec):
            raise TypeError("spec must be CapturePreviewSpec")
        if not isinstance(dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if not isinstance(evaluation_policy, FigureEvaluationPolicy):
            raise TypeError("evaluation_policy must be FigureEvaluationPolicy")
        self.spec = spec
        self.dataset_id = dataset_id
        self.evaluation_policy = evaluation_policy
        self._lock = threading.Lock()
        self._dataset: MonitorDataset | None = None
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._listener: Callable[[], None] | None = None
        self._listener_claimed = False
        self._pending_change = False
        self._failure: str | None = None
        self._terminal = False
        self._closed = False

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    @property
    def terminal(self) -> bool:
        with self._lock:
            return self._terminal

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        replay = False
        with self._lock:
            if self._listener_claimed:
                raise RuntimeError("live slot already has a change listener")
            if self._closed:
                raise RuntimeError("live slot is closed")
            self._listener_claimed = True
            self._listener = listener
            replay, self._pending_change = self._pending_change, False
        if replay:
            listener()

    def bind(
        self,
        dataset: MonitorDataset,
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None:
        if not isinstance(dataset, MonitorDataset):
            raise TypeError("dataset must be MonitorDataset")
        run_id = canonical_text(run_id, "run_id")
        causation_domain_id = canonical_text(
            causation_domain_id,
            "causation_domain_id",
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("live slot is closed")
            if self._terminal:
                raise RuntimeError("live slot is terminal")
            if self._dataset is not None:
                raise RuntimeError("live slot already owns a materializer")
            self._dataset = dataset
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id

    def updated(self) -> None:
        with self._lock:
            if self._closed or self._dataset is None:
                raise RuntimeError("live slot has no active materializer")
            listener = self._listener
            if listener is None:
                self._pending_change = True
                return
        if listener is not None:
            listener()

    def freeze_current(
        self,
    ) -> tuple[str, str, MonitorDatasetSnapshot]:
        """Atomically materialize current state; never resolve a prior notice ref."""

        with self._lock:
            if self._closed or self._dataset is None:
                raise RuntimeError("live slot has no active materializer")
            dataset = self._dataset
            run_id = self._run_id
            causation = self._causation_domain_id
        snapshot = dataset.materialize(None)
        with self._lock:
            if self._closed or self._dataset is not dataset:
                raise RuntimeError("live slot lifetime ended while freezing a snapshot")
        assert run_id is not None and causation is not None
        return run_id, causation, snapshot

    def fail(self, message: str) -> None:
        message = canonical_text(message, "preview failure")
        dataset, listener = self._detach(message)
        try:
            if dataset is not None:
                dataset.close()
        finally:
            if listener is not None:
                try:
                    listener()
                except BaseException:
                    pass

    def source_terminal(self) -> None:
        with self._lock:
            self._terminal = True

    def close(self) -> None:
        dataset, _listener = self._detach(None, closed=True)
        if dataset is not None:
            dataset.close()

    def _detach(
        self,
        failure: str | None,
        *,
        closed: bool = False,
    ) -> tuple[MonitorDataset | None, Callable[[], None] | None]:
        with self._lock:
            if failure is not None and self._failure is not None:
                return None, None
            dataset, self._dataset = self._dataset, None
            if failure is not None and self._failure is None:
                self._failure = failure
            self._terminal = True
            self._closed = self._closed or closed
            listener, self._listener = self._listener, None
            if failure is not None and listener is None and not self._listener_claimed:
                self._pending_change = True
            return dataset, listener


class LiveImageBoardController:
    """One-snapshot coalescer: worker freeze, owner admit, worker render."""

    def __init__(
        self,
        slot: LiveDatasetSlot,
        document: FigureDocument,
        board: BoardController,
        *,
        submit_worker: Callable[[Callable[[], None]], object],
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(slot, LiveDatasetSlot):
            raise TypeError("slot must be LiveDatasetSlot")
        if not isinstance(document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if not isinstance(board, BoardController):
            raise TypeError("board must be BoardController")
        if not callable(submit_worker) or not callable(request_owner_wake):
            raise TypeError("worker submission and owner wake must be callable")
        _validate_single_image_document(slot, document)
        model = board.model
        if len(model.panels) != 1:
            raise ValueError("first live-image controller requires one board panel")
        panel = model.panels[0]
        self._slot = slot
        self._document = document
        self._board = board
        self._panel = panel
        self._presentation = PanelPresentationIdentity(
            panel.panel_id,
            document.document_id,
            document.revision,
            0,
            0,
        )
        self._evaluator = FigureEvaluator(slot.evaluation_policy)
        self._submit_worker = submit_worker
        self._request_owner_wake = request_owner_wake
        self._lock = threading.Lock()
        self._worker_gate = threading.Lock()
        self._dirty = False
        self._active = False
        self._candidate: tuple[str, str, MonitorDatasetSnapshot] | None = None
        self._port: BoardPublishPort | None = None
        self._source: SourceIdentity | None = None
        self._sequence = 0
        self._fault: BaseException | None = None
        self._front_invalidated = False
        self._closed = False
        self._close_complete = False
        slot.set_change_listener(self._source_changed)

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    def _source_changed(self) -> None:
        if self._slot.failure is not None:
            self._request_owner_wake()
        else:
            self._request_snapshot()

    def _request_snapshot(self) -> None:
        with self._lock:
            if self._closed or self._fault is not None:
                return
            self._dirty = True
            if self._active:
                return
            self._active = True
            self._dirty = False
        self._submit(self._freeze_latest)

    def _freeze_latest(self) -> None:
        try:
            candidate = self._slot.freeze_current()
            with self._lock:
                if self._closed:
                    self._active = False
                    return
                self._candidate = candidate
            self._request_owner_wake()
        except BaseException as error:
            self._set_fault(error)

    def admit_pending(self) -> bool:
        failure = self._slot.failure
        if failure is not None:
            with self._lock:
                if self._closed:
                    return False
                invalidate = not self._front_invalidated
                self._front_invalidated = True
                if self._fault is None:
                    self._fault = RuntimeError(failure)
                self._candidate = None
                self._active = False
                self._port = None
                self._source = None
            if invalidate:
                try:
                    self._board.invalidate()
                except BaseException:
                    with self._lock:
                        self._front_invalidated = False
                    raise
            return False
        with self._lock:
            if self._closed or self._fault is not None:
                return False
            candidate, self._candidate = self._candidate, None
        if candidate is None:
            return False
        try:
            run_id, causation, snapshot = candidate
            ref = snapshot.snapshot.ref
            head = snapshot.head
            if head is None:
                raise RuntimeError("live snapshot has no head event")
            source = SourceIdentity(
                self._slot.dataset_id,
                ref.block_id,
                ref.stream_generation,
                ref.schema_fingerprint,
            )
            stamp = CoherenceStamp(
                run_id,
                causation,
                "single-source-event-payload",
                ref.schema_fingerprint,
                head.payload_digest,
                (EvaluatedInput(self._slot.dataset_id, ref),),
                (self._presentation,),
            )
            if self._port is None or source != self._source:
                self._port = self._board.open_publish_port(
                    (PanelSourceBinding(source, self._presentation),)
                )
                self._source = source
            sequence = self._sequence
            self._sequence += 1
            token = self._port.admit(
                sequence,
                ((self._panel.coherence_group, stamp),),
            )
            self._submit(
                lambda: self._render(candidate, source, stamp, sequence, token),
                ends_cycle=True,
            )
            return True
        except BaseException as error:
            self._set_fault(error)
            return False

    def _render(
        self,
        candidate: tuple[str, str, MonitorDatasetSnapshot],
        source: SourceIdentity,
        stamp: CoherenceStamp,
        sequence: int,
        token: object,
    ) -> None:
        try:
            snapshot = candidate[2].snapshot
            evaluated = self._evaluator.evaluate(
                self._document,
                ResolvedDatasetMap(
                    (ResolvedDataset(self._slot.dataset_id, snapshot),)
                ),
                cancel_requested=lambda: self._closed,
            )
            if evaluated.inputs != stamp.inputs:
                raise RuntimeError("figure evaluation changed the admitted input revision")
            if (
                len(evaluated.layers) != 1
                or len(evaluated.layers[0].cells) != 1
                or len(evaluated.layers[0].cells[0].series) != 1
            ):
                raise RuntimeError("live IMAGE document must evaluate to one image")
            layer = evaluated.layers[0]
            data = layer.cells[0].series[0].data
            if not isinstance(data, EvaluatedImage):
                raise RuntimeError("live IMAGE document did not evaluate to an image")
            frame = BoardFrame(
                self._board.model.board_id,
                self._board.model.layout_generation,
                sequence,
                (
                    PanelFrame(
                        self._panel.panel_id,
                        self._panel.coherence_group,
                        source,
                        stamp,
                        rasterize_image_gray8(data),
                    ),
                ),
            )
            port = self._port
            if port is not None:
                port.publish(token, frame)
        except BaseException as error:
            self._set_fault(error)
            return
    def _submit(
        self,
        work: Callable[[], None],
        *,
        ends_cycle: bool = False,
    ) -> None:
        pending_work: Callable[[], None] | None = work

        def serialized() -> None:
            nonlocal pending_work
            restart = False
            with self._worker_gate:
                callback = pending_work
                if callback is None:
                    return
                try:
                    callback()
                except BaseException as error:
                    self._set_fault(error)
                finally:
                    # A render callback may own the only full snapshot.  Drop
                    # it before this gate admits another worker callback.
                    callback = None
                    pending_work = None
                if ends_cycle:
                    with self._lock:
                        restart = (
                            self._dirty
                            and not self._closed
                            and self._fault is None
                        )
                        if restart:
                            self._dirty = False
                        else:
                            self._active = False
            if restart:
                self._submit(self._freeze_latest)

        try:
            self._submit_worker(serialized)
        except BaseException as error:
            self._set_fault(error)

    def _set_fault(self, error: BaseException) -> None:
        with self._lock:
            if self._closed:
                self._candidate = None
                self._active = False
                return
            if self._fault is None:
                self._fault = detached_render_fault(error)
            self._candidate = None
            self._active = False
        try:
            self._slot.fail(f"{type(error).__name__}: {error}")
        except BaseException:
            pass

    def close(self) -> None:
        with self._lock:
            if self._close_complete:
                return
            self._closed = True
            self._active = False
            self._dirty = False
            self._candidate = None
            self._front_invalidated = True
        try:
            self._slot.close()
        finally:
            self._board.close()
        with self._lock:
            self._fault = None
            self._close_complete = True


def _validate_single_image_document(
    slot: LiveDatasetSlot,
    document: FigureDocument,
) -> None:
    schema = slot.spec.dataset_edge.schema
    if (
        len(document.datasets) != 1
        or document.datasets[0].dataset_id != slot.dataset_id
        or document.datasets[0].schema_fingerprint != schema.fingerprint
        or len(document.layers) != 1
        or document.layers[0].dataset_id != slot.dataset_id
    ):
        raise ValueError("live document must contain the slot's one dataset and layer")
    view = document.layers[0].view
    validate_view_spec(schema, view)
    if view.intent is not ViewIntent.IMAGE:
        raise ValueError("live image controller requires IMAGE intent")


__all__ = ["LiveDatasetSlot", "LiveImageBoardController"]
