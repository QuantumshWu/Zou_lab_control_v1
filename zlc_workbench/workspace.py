"""Headless workspace and coherent board ownership for the Qt composition root."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Protocol, runtime_checkable
import weakref

from zlc_frontend import (
    BoardFrame,
    BoardPresenter,
    CoherenceStamp,
    PanelPresentationIdentity,
    RenderSurface,
    SourceIdentity,
)
from zlc_storage import (
    canonical_text as _text,
    nonnegative_integer,
)


@dataclass(frozen=True)
class PanelSlot:
    """Persisted placement identity; view semantics remain frontend-owned."""

    panel_id: str
    controller_key: str
    coherence_group: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _text(self.panel_id, "panel_id"))
        object.__setattr__(
            self, "controller_key", _text(self.controller_key, "controller_key")
        )
        object.__setattr__(
            self,
            "coherence_group",
            _text(self.coherence_group, "coherence_group"),
        )


@dataclass(frozen=True)
class BoardModel:
    board_id: str
    layout_generation: int
    surface: RenderSurface
    panels: tuple[PanelSlot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "board_id", _text(self.board_id, "board_id"))
        object.__setattr__(
            self,
            "layout_generation",
            nonnegative_integer(self.layout_generation, "layout_generation"),
        )
        if self.surface is not RenderSurface.WORKER_RASTER_LIVE:
            raise ValueError("BoardController boards require WORKER_RASTER_LIVE")
        panels = tuple(self.panels)
        if any(not isinstance(panel, PanelSlot) for panel in panels):
            raise TypeError("panels must contain PanelSlot values")
        ids = tuple(panel.panel_id for panel in panels)
        if len(set(ids)) != len(ids):
            raise ValueError("panel ids must be unique within a board")
        object.__setattr__(self, "panels", panels)

    @property
    def panel_ids(self) -> tuple[str, ...]:
        return tuple(panel.panel_id for panel in self.panels)

    def replace_panels(self, panels: tuple[PanelSlot, ...]) -> "BoardModel":
        return BoardModel(
            self.board_id,
            self.layout_generation + 1,
            self.surface,
            tuple(panels),
        )


@dataclass(frozen=True)
class WorkspaceModel:
    """Revisioned value; controllers never mutate persisted workspace state in place."""

    workspace_id: str
    revision: int
    boards: tuple[BoardModel, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "workspace_id", _text(self.workspace_id, "workspace_id")
        )
        object.__setattr__(
            self,
            "revision",
            nonnegative_integer(self.revision, "revision"),
        )
        boards = tuple(self.boards)
        if any(not isinstance(board, BoardModel) for board in boards):
            raise TypeError("boards must contain BoardModel values")
        ids = tuple(board.board_id for board in boards)
        if len(set(ids)) != len(ids):
            raise ValueError("board ids must be unique within a workspace")
        object.__setattr__(self, "boards", boards)

    def replace_board(self, board: BoardModel) -> "WorkspaceModel":
        if not isinstance(board, BoardModel):
            raise TypeError("board must be BoardModel")
        found = False
        updated = []
        for current in self.boards:
            if current.board_id == board.board_id:
                found = True
                if board.layout_generation <= current.layout_generation:
                    raise ValueError(
                        "workspace board replacement requires a newer layout_generation"
                    )
                updated.append(board)
            else:
                updated.append(current)
        if not found:
            updated.append(board)
        return replace(self, revision=self.revision + 1, boards=tuple(updated))


@runtime_checkable
class PanelHost(Protocol):
    """Qt-side panel shell; it owns widgets but never acquisition or run lifecycle."""

    @property
    def panel_id(self) -> str: ...

    def clear(self, reason: str) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PanelSourceBinding:
    """Expected producer identity for one panel in the active board layout."""

    source_identity: SourceIdentity
    presentation: PanelPresentationIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be SourceIdentity")
        if not isinstance(self.presentation, PanelPresentationIdentity):
            raise TypeError("presentation must be PanelPresentationIdentity")

    @property
    def panel_id(self) -> str:
        return self.presentation.panel_id


class BoardPublishPort:
    """Revocable board-work capability.

    ``admit`` is owner-thread-only and runs before raster work is dispatched;
    ``publish`` is worker-safe and consumes the returned one-shot token.
    """

    __slots__ = ("_controller_ref", "_token")

    def __init__(self, controller: "BoardController", token: object) -> None:
        self._controller_ref = weakref.ref(controller)
        self._token = token

    def admit(
        self,
        sequence: int,
        coherence_stamps: tuple[tuple[str, CoherenceStamp], ...],
    ) -> object:
        controller = self._controller_ref()
        if controller is None:
            raise RuntimeError("board controller no longer exists")
        return controller._admit_work(
            self._token,
            sequence,
            coherence_stamps,
        )

    def publish(self, work_token: object, frame: BoardFrame) -> bool:
        controller = self._controller_ref()
        return (
            False
            if controller is None
            else controller._publish(self._token, work_token, frame)
        )


class BoardController:
    """Latest-only board mailbox with one owner-thread coherent present point."""

    def __init__(
        self,
        model: BoardModel,
        presenter: BoardPresenter,
        post_to_owner: object,
    ) -> None:
        if not isinstance(model, BoardModel):
            raise TypeError("model must be BoardModel")
        if not isinstance(presenter, BoardPresenter):
            raise TypeError("presenter must implement BoardPresenter")
        if not callable(post_to_owner):
            raise TypeError("post_to_owner must be callable")
        self._owner_thread = threading.get_ident()
        self._lock = threading.Lock()
        self._model = model
        self._presenter: BoardPresenter | None = presenter
        self._post_to_owner = post_to_owner
        self._pending: BoardFrame | None = None
        self._requested_sequence = -1
        self._work_token: object | None = None
        self._expected_stamps: dict[str, CoherenceStamp] = {}
        self._wake_queued = False
        self._publish_token: object | None = None
        self._source_bindings: dict[str, PanelSourceBinding] = {}
        self._closed = False
        self._fault: BaseException | None = None

    @property
    def model(self) -> BoardModel:
        with self._lock:
            return self._model

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    def reconfigure(self, model: BoardModel) -> None:
        self._require_owner()
        if not isinstance(model, BoardModel):
            raise TypeError("model must be BoardModel")
        if model.board_id != self._model.board_id:
            raise ValueError("BoardController cannot change board identity")
        if model.layout_generation <= self._model.layout_generation:
            raise ValueError("reconfigure requires a newer layout_generation")
        with self._lock:
            self._ensure_usable()
            self._model = model
            self._pending = None
            self._requested_sequence = -1
            self._work_token = None
            self._expected_stamps = {}
            self._wake_queued = False
            self._publish_token = None
            self._source_bindings = {}
        self._clear_presenter()

    def open_publish_port(
        self,
        bindings: tuple[PanelSourceBinding, ...],
    ) -> BoardPublishPort:
        self._require_owner()
        bindings = tuple(bindings)
        if any(not isinstance(value, PanelSourceBinding) for value in bindings):
            raise TypeError("bindings must contain PanelSourceBinding values")
        by_panel = {value.panel_id: value for value in bindings}
        if len(by_panel) != len(bindings):
            raise ValueError("panel source bindings must have unique panel ids")
        expected = set(self._model.panel_ids)
        if set(by_panel) != expected:
            raise ValueError("source bindings must cover every active panel exactly")
        token = object()
        with self._lock:
            self._ensure_usable()
            self._publish_token = token
            self._source_bindings = by_panel
            self._pending = None
            self._requested_sequence = -1
            self._work_token = None
            self._expected_stamps = {}
            self._wake_queued = False
        self._clear_presenter()
        return BoardPublishPort(self, token)

    def _admit_work(
        self,
        token: object,
        sequence: int,
        coherence_stamps: tuple[tuple[str, CoherenceStamp], ...],
    ) -> object:
        """Freeze the latest requested board job before raster work begins."""

        self._require_owner()
        sequence = nonnegative_integer(sequence, "board work sequence")
        try:
            pairs = tuple(coherence_stamps)
        except TypeError as exc:
            raise TypeError("coherence_stamps must be a tuple of pairs") from exc
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[1], CoherenceStamp)
            for pair in pairs
        ):
            raise TypeError(
                "coherence_stamps must contain (group, CoherenceStamp) pairs"
            )
        by_group = {
            _text(group, "coherence group"): stamp
            for group, stamp in pairs
        }
        if len(by_group) != len(pairs):
            raise ValueError("coherence_stamps must have unique groups")
        with self._lock:
            if (
                self._closed
                or self._fault is not None
                or token is not self._publish_token
            ):
                raise RuntimeError("board publish port is revoked")
            if sequence <= self._requested_sequence:
                raise ValueError("board work sequence must increase")
            model_groups: dict[str, list[str]] = {}
            for panel in self._model.panels:
                model_groups.setdefault(panel.coherence_group, []).append(
                    panel.panel_id
                )
            if set(by_group) != set(model_groups):
                raise ValueError(
                    "coherence_stamps must cover every active group exactly"
                )
            for group, panel_ids in model_groups.items():
                stamp = by_group[group]
                presentations = {
                    value.panel_id: value for value in stamp.presentations
                }
                if set(presentations) != set(panel_ids):
                    raise ValueError(
                        "coherence stamp presentations differ from its board group"
                    )
                inputs = {value.dataset_id: value.ref for value in stamp.inputs}
                for panel_id in panel_ids:
                    binding = self._source_bindings[panel_id]
                    if presentations[panel_id] != binding.presentation:
                        raise ValueError(
                            "coherence stamp presentation differs from the publish port"
                        )
                    source = binding.source_identity
                    try:
                        ref = inputs[source.dataset_id]
                    except KeyError as exc:
                        raise ValueError(
                            "coherence stamp omits a bound panel source"
                        ) from exc
                    if (
                        ref.block_id != source.block_id
                        or ref.stream_generation != source.stream_generation
                        or ref.schema_fingerprint != source.schema_fingerprint
                    ):
                        raise ValueError(
                            "coherence stamp input differs from the publish port"
                        )
            work_token = object()
            self._requested_sequence = sequence
            self._work_token = work_token
            self._expected_stamps = by_group
            self._pending = None
            return work_token

    def _publish(
        self,
        token: object,
        work_token: object,
        frame: BoardFrame,
    ) -> bool:
        """Worker-safe replace-pending operation; stale frames are rejected, never mixed."""

        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        schedule = False
        with self._lock:
            if (
                self._closed
                or self._fault is not None
                or token is not self._publish_token
                or work_token is not self._work_token
            ):
                return False
            model = self._model
            if frame.board_id != model.board_id:
                raise ValueError("frame belongs to another board")
            if frame.layout_generation != model.layout_generation:
                return False
            if tuple(panel.panel_id for panel in frame.panels) != model.panel_ids:
                raise ValueError("frame panel order/set does not match the active board layout")
            expected_groups = tuple(panel.coherence_group for panel in model.panels)
            actual_groups = tuple(panel.coherence_group for panel in frame.panels)
            if actual_groups != expected_groups:
                raise ValueError("frame coherence groups do not match the active board layout")
            frame_stamps: dict[str, CoherenceStamp] = {}
            for panel in frame.panels:
                frame_stamps.setdefault(
                    panel.coherence_group,
                    panel.coherence_stamp,
                )
            if frame_stamps != self._expected_stamps:
                return False
            if frame.sequence != self._requested_sequence:
                return False
            self._work_token = None
            self._pending = frame
            if not self._wake_queued:
                self._wake_queued = True
                schedule = True
        if schedule:
            try:
                post_to_owner = self._post_to_owner
                if post_to_owner is None:
                    return False
                post_to_owner(self.present_pending)
            except BaseException as exc:
                with self._lock:
                    if token is self._publish_token:
                        self._fault = exc
                        self._pending = None
                        self._publish_token = None
                raise
        return True

    def present_pending(self) -> bool:
        """Owner-thread atomic board flip; never presents individual panels separately."""

        self._require_owner()
        with self._lock:
            if self._closed:
                return False
            self._ensure_usable()
            self._wake_queued = False
            frame, self._pending = self._pending, None
        if frame is None:
            return False
        try:
            presenter = self._presenter
            if presenter is None:
                return False
            presenter.present(frame)
        except BaseException as exc:
            with self._lock:
                self._fault = exc
                self._pending = None
            raise
        return True

    def close(self) -> None:
        self._require_owner()
        presenter: BoardPresenter | None
        with self._lock:
            if self._closed:
                return
            self._pending = None
            self._publish_token = None
            self._source_bindings = {}
            self._work_token = None
            self._expected_stamps = {}
            self._wake_queued = False
            self._closed = True
            presenter = self._presenter
            self._post_to_owner = None
        if presenter is None:
            with self._lock:
                self._fault = None
            return
        try:
            presenter.clear()
        except BaseException as exc:
            with self._lock:
                if self._presenter is presenter:
                    self._fault = exc
                    self._closed = False
            raise
        with self._lock:
            if self._presenter is presenter:
                self._presenter = None
                self._fault = None

    def _clear_presenter(self) -> None:
        presenter = self._presenter
        if presenter is None:
            return
        try:
            presenter.clear()
        except BaseException as exc:
            with self._lock:
                self._fault = exc
                self._pending = None
                self._publish_token = None
            raise

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("BoardController presentation/layout is owner-thread affine")

    def _ensure_usable(self) -> None:
        if self._closed:
            raise RuntimeError("BoardController is closed")
        if self._fault is not None:
            raise RuntimeError("BoardController is faulted") from self._fault


__all__ = [
    "BoardController",
    "BoardModel",
    "BoardPublishPort",
    "PanelSourceBinding",
    "PanelHost",
    "PanelSlot",
    "WorkspaceModel",
]
