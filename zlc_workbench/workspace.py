"""Headless workspace and coherent board ownership for the Qt composition root."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Protocol, runtime_checkable

from zlc_frontend import BoardFrame, BoardPresenter, RenderSurface
from zlc_storage import (
    canonical_text as _text,
    nonnegative_integer,
    normalized_text,
    sha256_text,
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
class CoherenceSourceBinding:
    coherence_group: str
    producer_generation: int
    schema_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coherence_group",
            _text(self.coherence_group, "coherence_group"),
        )
        object.__setattr__(
            self,
            "producer_generation",
            nonnegative_integer(self.producer_generation, "producer_generation"),
        )
        object.__setattr__(
            self,
            "schema_fingerprint",
            sha256_text(self.schema_fingerprint, "schema_fingerprint"),
        )


class BoardPublishPort:
    """Revocable worker capability bound to one board layout and source generation."""

    __slots__ = ("_controller", "_token")

    def __init__(self, controller: "BoardController", token: object) -> None:
        self._controller = controller
        self._token = token

    def publish(self, frame: BoardFrame) -> bool:
        return self._controller._publish(self._token, frame)


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
        self._presenter = presenter
        self._post_to_owner = post_to_owner
        self._pending: BoardFrame | None = None
        self._accepted_sequence = -1
        self._last_sequence = -1
        self._wake_queued = False
        self._publish_token: object | None = None
        self._source_bindings: dict[str, CoherenceSourceBinding] = {}
        self._closed = False
        self._fault: BaseException | None = None

    @property
    def model(self) -> BoardModel:
        return self._model

    @property
    def fault(self) -> BaseException | None:
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
            self._accepted_sequence = -1
            self._last_sequence = -1
            self._wake_queued = False
            self._publish_token = None
            self._source_bindings = {}

    def open_publish_port(
        self,
        bindings: tuple[CoherenceSourceBinding, ...],
    ) -> BoardPublishPort:
        self._require_owner()
        bindings = tuple(bindings)
        if any(not isinstance(value, CoherenceSourceBinding) for value in bindings):
            raise TypeError("bindings must contain CoherenceSourceBinding values")
        by_group = {value.coherence_group: value for value in bindings}
        if len(by_group) != len(bindings):
            raise ValueError("coherence source bindings must have unique groups")
        expected = {panel.coherence_group for panel in self._model.panels}
        if set(by_group) != expected:
            raise ValueError("source bindings must cover every active coherence group exactly")
        token = object()
        with self._lock:
            self._ensure_usable()
            self._publish_token = token
            self._source_bindings = by_group
            self._pending = None
            self._accepted_sequence = -1
            self._last_sequence = -1
            self._wake_queued = False
        return BoardPublishPort(self, token)

    def _publish(self, token: object, frame: BoardFrame) -> bool:
        """Worker-safe replace-pending operation; stale frames are rejected, never mixed."""

        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        schedule = False
        with self._lock:
            if self._closed or self._fault is not None or token is not self._publish_token:
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
            for group, identity in frame.coherence_stamps:
                binding = self._source_bindings[group]
                if identity.producer_generation != binding.producer_generation:
                    return False
                if identity.schema_fingerprint != binding.schema_fingerprint:
                    return False
            if frame.sequence <= self._accepted_sequence:
                return False
            self._accepted_sequence = frame.sequence
            self._pending = frame
            if not self._wake_queued:
                self._wake_queued = True
                schedule = True
        if schedule:
            try:
                self._post_to_owner(self._scheduled_present)
            except BaseException as exc:
                with self._lock:
                    if token is self._publish_token:
                        self._fault = exc
                        self._pending = None
                        self._publish_token = None
                raise
        return True

    def _scheduled_present(self) -> None:
        self.present_pending()

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
            self._presenter.present(frame)
        except BaseException as exc:
            with self._lock:
                self._fault = exc
                self._pending = None
            raise
        with self._lock:
            self._last_sequence = frame.sequence
        return True

    def close(self) -> None:
        self._require_owner()
        with self._lock:
            self._pending = None
            self._publish_token = None
            self._source_bindings = {}
            self._wake_queued = False
            self._closed = True

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("BoardController presentation/layout is owner-thread affine")

    def _ensure_usable(self) -> None:
        if self._closed:
            raise RuntimeError("BoardController is closed")
        if self._fault is not None:
            raise RuntimeError("BoardController is faulted") from self._fault


@dataclass(frozen=True)
class RunStatusView:
    run_id: str
    revision: int
    state: str
    phase: str
    final_committed: bool
    primary_error: str | None
    cleanup_errors: tuple[str, ...]


class RunHandleStatusBinding:
    """Read-only projection of a runtime handle; cancellation remains handle-owned."""

    def __init__(self, handle: object) -> None:
        if not callable(getattr(handle, "snapshot", None)) or not callable(
            getattr(handle, "cancel", None)
        ):
            raise TypeError("handle must provide snapshot() and cancel()")
        self._handle = handle

    def snapshot(self) -> RunStatusView:
        value = self._handle.snapshot()
        run_id = _text(str(value.run_id), "run_id")
        raw_revision = getattr(value.revision, "value", value.revision)
        if not isinstance(raw_revision, int) or isinstance(raw_revision, bool) or raw_revision < 0:
            raise TypeError("run snapshot revision must be a non-negative integer value")
        state = getattr(value.state, "value", value.state)
        return RunStatusView(
            run_id=run_id,
            revision=raw_revision,
            state=_text(str(state), "state"),
            phase=_text(value.phase, "phase"),
            final_committed=bool(value.final_committed),
            primary_error=value.primary_error,
            cleanup_errors=tuple(value.cleanup_errors),
        )

    def cancel(self, reason: str = "user requested stop") -> object:
        return self._handle.cancel(normalized_text(reason, "cancel reason"))


__all__ = [
    "BoardController",
    "BoardModel",
    "BoardPublishPort",
    "CoherenceSourceBinding",
    "PanelHost",
    "PanelSlot",
    "RunHandleStatusBinding",
    "RunStatusView",
    "WorkspaceModel",
]
