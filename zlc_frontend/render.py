"""Headless render hand-off values owned by the target frontend package.

The renderer may use Matplotlib, Qt, or neither, but the worker/GUI boundary is
always an immutable :class:`BoardFrame`.  No live Figure, Artist, ndarray view,
or QImage storage crosses this module's boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Protocol, runtime_checkable


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


def _nonnegative(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} cannot be negative")
    return value


class RenderSurface(Enum):
    """The three deliberately supported rendering ownership modes."""

    GUI_ARTIST = "gui-artist"
    WORKER_RASTER_LIVE = "worker-raster-live"
    WORKER_HEADLESS_EXPORT = "worker-headless-export"


class PixelFormat(Enum):
    """Canonical owned raster layouts accepted at the presentation boundary."""

    RGBA8888 = "rgba8888"
    RGB888 = "rgb888"
    GRAY8 = "gray8"

    @property
    def channels(self) -> int:
        return {
            PixelFormat.RGBA8888: 4,
            PixelFormat.RGB888: 3,
            PixelFormat.GRAY8: 1,
        }[self]


@dataclass(frozen=True)
class FrameIdentity:
    """Complete identity needed to reject mixed-shot or stale board frames."""

    run_id: str
    dataset_id: str
    producer_generation: int
    schema_fingerprint: str
    revision: int
    coherence_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, "dataset_id"))
        object.__setattr__(
            self,
            "producer_generation",
            _nonnegative(self.producer_generation, "producer_generation"),
        )
        object.__setattr__(
            self,
            "schema_fingerprint",
            _text(self.schema_fingerprint, "schema_fingerprint"),
        )
        object.__setattr__(self, "revision", _nonnegative(self.revision, "revision"))
        object.__setattr__(
            self, "coherence_key", _text(self.coherence_key, "coherence_key")
        )


@dataclass(frozen=True)
class RasterBuffer:
    """An owned immutable raster; ``pixels`` can never alias a worker buffer."""

    width: int
    height: int
    stride_bytes: int
    pixel_format: PixelFormat
    pixels: bytes

    def __post_init__(self) -> None:
        width = _nonnegative(self.width, "width")
        height = _nonnegative(self.height, "height")
        stride = _nonnegative(self.stride_bytes, "stride_bytes")
        if width == 0 or height == 0:
            raise ValueError("raster width and height must be positive")
        if not isinstance(self.pixel_format, PixelFormat):
            raise TypeError("pixel_format must be PixelFormat")
        minimum_stride = width * self.pixel_format.channels
        if stride < minimum_stride:
            raise ValueError("stride_bytes is too small for width and pixel format")
        if not isinstance(self.pixels, bytes):
            raise TypeError("pixels must be owned immutable bytes")
        if len(self.pixels) != stride * height:
            raise ValueError("pixels length must equal stride_bytes * height")


@dataclass(frozen=True)
class PanelFrame:
    panel_id: str
    coherence_group: str
    identity: FrameIdentity
    raster: RasterBuffer

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _text(self.panel_id, "panel_id"))
        object.__setattr__(
            self,
            "coherence_group",
            _text(self.coherence_group, "coherence_group"),
        )
        if not isinstance(self.identity, FrameIdentity):
            raise TypeError("identity must be FrameIdentity")
        if not isinstance(self.raster, RasterBuffer):
            raise TypeError("raster must be RasterBuffer")


@dataclass(frozen=True)
class BoardFrame:
    """One atomic, shot-coherent presentation for a complete board layout."""

    board_id: str
    layout_generation: int
    sequence: int
    panels: tuple[PanelFrame, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "board_id", _text(self.board_id, "board_id"))
        object.__setattr__(
            self,
            "layout_generation",
            _nonnegative(self.layout_generation, "layout_generation"),
        )
        object.__setattr__(self, "sequence", _nonnegative(self.sequence, "sequence"))
        panels = tuple(self.panels)
        if not panels:
            raise ValueError("BoardFrame must contain at least one panel")
        if any(not isinstance(panel, PanelFrame) for panel in panels):
            raise TypeError("panels must contain PanelFrame values")
        ids = tuple(panel.panel_id for panel in panels)
        if len(set(ids)) != len(ids):
            raise ValueError("BoardFrame panel ids must be unique")
        identity_by_group: dict[str, FrameIdentity] = {}
        for panel in panels:
            existing = identity_by_group.setdefault(
                panel.coherence_group, panel.identity
            )
            if existing != panel.identity:
                raise ValueError(
                    "panels in one coherence group must carry one exact FrameIdentity"
                )
        object.__setattr__(self, "panels", panels)

    @property
    def coherence_stamps(self) -> tuple[tuple[str, FrameIdentity], ...]:
        by_group: dict[str, FrameIdentity] = {}
        for panel in self.panels:
            by_group.setdefault(panel.coherence_group, panel.identity)
        return tuple(sorted(by_group.items()))


@runtime_checkable
class BoardPresenter(Protocol):
    """GUI-side sink; one call presents the entire board coherently."""

    def present(self, frame: BoardFrame) -> None: ...


class AtomicBoardFront:
    """Concrete old-or-new front mapping read by a GUI board paint adapter.

    The swap is atomic at the model transaction boundary.  Separate native widgets
    can still be painted by the OS at different instants and must not advertise
    pixel-clock simultaneity; they all nevertheless read one immutable BoardFrame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: BoardFrame | None = None

    def present(self, frame: BoardFrame) -> None:
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        with self._lock:
            self._current = frame

    def current(self) -> BoardFrame | None:
        with self._lock:
            return self._current


__all__ = [
    "BoardFrame",
    "BoardPresenter",
    "AtomicBoardFront",
    "FrameIdentity",
    "PanelFrame",
    "PixelFormat",
    "RasterBuffer",
    "RenderSurface",
]
