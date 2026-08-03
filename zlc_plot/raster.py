"""Immutable Agg-to-frontend hand-off and serial plot worker.

The Qt adapter consumes only :class:`RasterFront` values.  A live Matplotlib
Figure, Artist, mutable array view, or Qt object never crosses this boundary.
All plot mutations and raster capture run on the single worker owned by
``RasterPlotHost``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from numbers import Integral
from pathlib import Path
from threading import Condition, Lock, Thread, current_thread
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import uuid4

import numpy as np
from PIL import Image
from ._axis_transform import AxisTransform
from ._validation import finite_real as _finite
from ._validation import integer, nonempty_text, optional_nonempty_text, readonly_copy
from .config import DEFAULTS, PlotLibraryDefaults
from .selectors import (
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
)

if TYPE_CHECKING:
    from .fit import (
        FacetFitBatchResult,
        FitEngine,
        FitModelSpec,
        FitOptions,
        FitResult,
    )
    from .layout import SurfacePlan
    from .live import LiveContract, LivePlotController
    from .primitives import ImageFrame, ImagePointOverlay, PlotInput
    from .session import (
        DisplayDescription,
        FitEvent,
        PlotSession,
        PlotSessionConfig,
        SelectionEvent,
        SelectorData,
    )
    from .state import DisplayState
    from .specs import PlotSpec


ValueT = TypeVar("ValueT")
EventT = TypeVar("EventT")
_UNSET = object()


def _positive_integer(value: object, name: str) -> int:
    result = integer(value, name, minimum=1)
    assert result is not None
    return result


@dataclass(frozen=True, slots=True)
class RasterBuffer:
    """One tightly packed, owned, immutable RGBA8888 image."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        width = _positive_integer(self.width, "raster width")
        height = _positive_integer(self.height, "raster height")
        if not isinstance(self.pixels, bytes):
            raise TypeError("raster pixels must be owned immutable bytes")
        if len(self.pixels) != width * height * 4:
            raise ValueError("RGBA8888 byte length does not match raster dimensions")

    def as_rgba(self, *, copy: bool = False) -> np.ndarray:
        """Return this exact RGBA8888 buffer as a read-only array.

        The default view shares the immutable byte storage.  ``copy=True``
        returns independent storage while retaining the read-only contract.
        Neither path renders, rescales, or color-converts the image.
        """

        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean")
        rgba = np.frombuffer(self.pixels, dtype=np.uint8).reshape(
            self.height,
            self.width,
            4,
        )
        if copy:
            rgba = readonly_copy(rgba)
        return rgba

    def encode(self, format: str = "PNG", **options: object) -> bytes:
        """Encode the current physical pixels without rendering or resizing."""

        selected_format = nonempty_text(format, "image format").upper()
        output = BytesIO()
        Image.frombytes(
            "RGBA",
            (self.width, self.height),
            self.pixels,
        ).save(output, format=selected_format, **options)
        return output.getvalue()

    def save(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        **options: object,
    ) -> None:
        """Encode the current physical pixels to ``path`` without rerendering."""

        target = Path(path)
        selected = format
        if selected is None:
            selected = target.suffix.removeprefix(".")
        target.write_bytes(
            self.encode(nonempty_text(selected, "image format"), **options)
        )


@dataclass(frozen=True, slots=True)
class RasterIdentity:
    """Exact session state represented by one accepted raster."""

    host_id: str
    sequence: int
    data_generation: str | None
    data_revision: int
    image_overlay_revision: int | None
    display_revision: int
    layout_revision: int
    kind: str
    preset: str

    def __post_init__(self) -> None:
        for name in (
            "sequence",
            "data_revision",
            "display_revision",
            "layout_revision",
        ):
            value = integer(getattr(self, name), name, minimum=0)
            assert value is not None
            object.__setattr__(self, name, value)
        for name in ("host_id", "kind", "preset"):
            object.__setattr__(self, name, nonempty_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "data_generation",
            optional_nonempty_text(self.data_generation, "data_generation"),
        )
        object.__setattr__(
            self,
            "image_overlay_revision",
            integer(
                self.image_overlay_revision,
                "image_overlay_revision",
                minimum=0,
                optional=True,
            ),
        )

    def same_surface(self, other: object) -> bool:
        """Return whether two fronts share one current interactive surface."""

        return isinstance(other, RasterIdentity) and (
            self.host_id,
            self.display_revision,
            self.layout_revision,
            self.kind,
            self.preset,
        ) == (
            other.host_id,
            other.display_revision,
            other.layout_revision,
            other.kind,
            other.preset,
        )


@dataclass(frozen=True, slots=True)
class RasterInteractionMap:
    """Everything pointer handling may read from the exact painted front."""

    axes: tuple[AxisTransform, ...]
    selectors: tuple[SelectorState, ...]
    color_limits: NumericRange | None = None
    facet_focus_index: int | None = None

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes or any(not isinstance(value, AxisTransform) for value in axes):
            raise ValueError("interaction map requires AxisTransform values")
        selectors = SelectorSnapshot(tuple(self.selectors)).committed
        color_limits = self.color_limits
        if color_limits is not None and not isinstance(color_limits, NumericRange):
            raise TypeError("color_limits must be NumericRange or None")
        focus = integer(
            self.facet_focus_index,
            "facet_focus_index",
            minimum=0,
            optional=True,
        )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "selectors", selectors)
        object.__setattr__(self, "color_limits", color_limits)
        object.__setattr__(self, "facet_focus_index", focus)

@dataclass(frozen=True, slots=True)
class RasterFront:
    """One complete immutable frontend value promoted atomically."""

    identity: RasterIdentity
    source_revisions: tuple[int, ...]
    buffer: RasterBuffer
    logical_size: tuple[int, int]
    logical_dpi: float
    device_pixel_ratio: float
    interaction: RasterInteractionMap

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RasterIdentity):
            raise TypeError("front identity must be RasterIdentity")
        revisions = tuple(self.source_revisions)
        if not revisions:
            raise ValueError("front source_revisions cannot be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in revisions
        ):
            raise TypeError("front source_revisions must be non-negative integers")
        revisions = tuple(map(int, revisions))
        if any(right <= left for left, right in zip(revisions, revisions[1:])):
            raise ValueError("front source_revisions must increase")
        if revisions[-1] != self.identity.data_revision:
            raise ValueError("front latest source revision must equal data_revision")
        object.__setattr__(self, "source_revisions", revisions)
        if not isinstance(self.buffer, RasterBuffer):
            raise TypeError("front buffer must be RasterBuffer")
        width, height = tuple(self.logical_size)
        object.__setattr__(
            self,
            "logical_size",
            (
                _positive_integer(width, "logical width"),
                _positive_integer(height, "logical height"),
            ),
        )
        logical_dpi = _finite(self.logical_dpi, "logical dpi")
        if logical_dpi <= 0.0:
            raise ValueError("logical dpi must be positive")
        ratio = _finite(self.device_pixel_ratio, "device pixel ratio")
        if ratio <= 0.0:
            raise ValueError("device pixel ratio must be positive")
        if not isinstance(self.interaction, RasterInteractionMap):
            raise TypeError("front interaction must be RasterInteractionMap")


@dataclass(frozen=True, slots=True)
class RasterOperation(Generic[ValueT]):
    """A worker result and the exact front painted after that result."""

    value: ValueT
    front: RasterFront


class _DispatchMode(str, Enum):
    CONTROL = "control"
    ADAPTIVE = "adaptive"
    PUBLISH = "publish"
    PRESENTATION = "presentation"

    @property
    def publishes(self) -> bool:
        return self in {_DispatchMode.PUBLISH, _DispatchMode.PRESENTATION}


class _PresentationRollbackError(RuntimeError):
    """Both publication and the required session rollback failed."""

    def __init__(self, primary: BaseException, rollback: BaseException) -> None:
        super().__init__("raster presentation failed and could not be rolled back")
        self.primary = primary
        self.rollback = rollback


_CoalesceResolver = Callable[[tuple[Any, ...], Mapping[str, Any]], object | None]


def _parameter_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object:
    return ("parameter", str(args[0]))


def _parameters_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object:
    return ("parameters", tuple(sorted(args[0])))


def _axis_unit_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object:
    return ("axis-unit", args[0])


def _labels_coalesce(
    _args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> object:
    return ("labels", tuple(sorted(kwargs)))


def _pointer_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object | None:
    return "pointer-motion" if args[0] == "move" else None


class _SessionCommand(Enum):
    """Single dispatch-policy catalogue for the worker-side session surface."""

    UPDATE_DATA = "update_data", _DispatchMode.PUBLISH, "data"
    UPDATE_IMAGE_FRAME = "update_image_frame", _DispatchMode.PUBLISH, "data"
    UPDATE_IMAGE_OVERLAY = "update_image_overlay", _DispatchMode.PUBLISH, "image-overlay"
    SET_PARAMETER = "set_parameter", _DispatchMode.PUBLISH, _parameter_coalesce
    SET_PARAMETERS = "set_parameters", _DispatchMode.PUBLISH, _parameters_coalesce
    SET_LABELS = "set_labels", _DispatchMode.PUBLISH, _labels_coalesce
    DESCRIBE_DISPLAY = "describe_display", _DispatchMode.CONTROL
    CONFIGURATION = "configuration", _DispatchMode.CONTROL, None, True
    REPLACE_SPEC = "replace_spec", _DispatchMode.PUBLISH, "plot-spec"
    RESOLVED_COLOR_LIMITS = "resolved_color_limits", _DispatchMode.CONTROL
    SET_RELIM_MODE = "set_relim_mode", _DispatchMode.PUBLISH, "relim-mode"
    SET_Y_LIMITS = "set_y_limits", _DispatchMode.PUBLISH, "y-limits"
    RESET_Y_LIMITS = "reset_y_limits", _DispatchMode.PUBLISH, "y-limits"
    SET_COLOR_LIMITS = "set_color_limits", _DispatchMode.PUBLISH, "color-limits"
    RESET_COLOR_LIMITS = "reset_color_limits", _DispatchMode.PUBLISH, "color-limits"
    SET_X_LIMITS = "set_x_limits", _DispatchMode.PUBLISH, "viewport"
    SET_VIEW_LIMITS = "set_view_limits", _DispatchMode.PUBLISH, "viewport"
    SET_SIZE = "set_size", _DispatchMode.PUBLISH, "size"
    SET_DEVICE_PIXEL_RATIO = "set_device_pixel_ratio", _DispatchMode.PUBLISH, "device-pixel-ratio"
    SET_AXIS_UNIT = "set_axis_unit", _DispatchMode.PUBLISH, _axis_unit_coalesce
    SET_VALUE_UNIT = "set_value_unit", _DispatchMode.PUBLISH, "value-unit"
    SET_TIME_UNIT = "set_time_unit", _DispatchMode.PUBLISH, "time-unit"
    SAVE = "save", _DispatchMode.CONTROL
    CLEAR_FIT = "clear_fit", _DispatchMode.PUBLISH, "fit-control"
    FIT_MODELS = "fit_models", _DispatchMode.CONTROL, None, True
    SELECTORS = "selectors", _DispatchMode.CONTROL, None, True
    SELECTOR_STATE = "selector_state", _DispatchMode.CONTROL
    SELECTOR_DATA = "selector_data", _DispatchMode.CONTROL
    REMOVE_SELECTOR = "remove_selector", _DispatchMode.PUBLISH
    SET_SELECTOR_VALUE = "set_selector_value", _DispatchMode.PUBLISH
    SET_AREA_SELECTOR = "set_area_selector", _DispatchMode.PUBLISH
    SET_X_SELECTOR = "set_x_selector", _DispatchMode.PUBLISH
    SET_THRESHOLD_SELECTOR = "set_threshold_selector", _DispatchMode.PUBLISH
    SET_FACET_THRESHOLDS = "set_facet_thresholds", _DispatchMode.PUBLISH, "facet-thresholds"
    SET_CROSSHAIR_SELECTOR = "set_crosshair_selector", _DispatchMode.PUBLISH
    FIT = "fit_async", _DispatchMode.CONTROL
    SET_VIEWPORT = "set_viewport", _DispatchMode.PUBLISH, "viewport"
    SHOW_FACET_OVERVIEW = "show_facet_overview", _DispatchMode.PUBLISH, "facet-presentation"
    RESET_VIEWPORT = "reset_viewport", _DispatchMode.PUBLISH, "viewport"
    POINTER = "_raster_pointer_event", _DispatchMode.ADAPTIVE, _pointer_coalesce
    FOCUS_FACET = "focus_facet", _DispatchMode.PUBLISH, "facet-presentation"

    def __init__(
        self,
        operation: str,
        mode: _DispatchMode,
        coalesce: object | _CoalesceResolver = None,
        attribute: bool = False,
    ) -> None:
        self.operation = operation
        self.mode = mode
        self.coalesce = coalesce
        self.attribute = attribute

    def coalesce_key(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> object | None:
        return self.coalesce(args, kwargs) if callable(self.coalesce) else self.coalesce


@dataclass(slots=True)
class _WorkerTask:
    callback: Callable[[], Any]
    completion: Future[RasterOperation[Any]]
    mode: _DispatchMode
    coalesce_key: object | None
    after_publish: Callable[[], None] | None
    on_abort: Callable[[], None] | None


class _WorkerSessionAdapter:
    """Concentrate worker-only PlotSession presentation details."""

    def __init__(self, host: "RasterPlotHost") -> None:
        self._host = host

    def _session(self) -> "PlotSession":
        if current_thread() is not self._host._thread:
            raise RuntimeError("plot session access must run on the raster worker")
        return self._host._require_session()

    def invoke(self, command: _SessionCommand, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self._session(), command.operation, _UNSET)
        if command.attribute:
            if target is _UNSET:
                raise TypeError(f"session is missing attribute {command.operation!r}")
            if args or kwargs:
                raise TypeError(f"session attribute {command.operation!r} takes no arguments")
            return target
        if not callable(target):
            raise TypeError(f"session is missing callable {command.operation!r}")
        return target(*args, **kwargs)

    @property
    def data_revision(self) -> int | None:
        front = self._host.front
        return None if front is None else front.identity.data_revision

    @property
    def defaults(self) -> PlotLibraryDefaults:
        return self._host.defaults

    def update_data(
        self,
        data: object,
        *,
        revision: int | None = None,
    ) -> object:
        return self.invoke(_SessionCommand.UPDATE_DATA, data, revision=revision)

    def capture_rgba(
        self,
        *,
        redraw: bool = False,
    ) -> np.ndarray:
        return np.asarray(
            self._session()._raster_capture_rgba(redraw=redraw),
            dtype=np.uint8,
        )

    def interaction_state(
        self,
    ) -> tuple[SelectorState, ...]:
        return tuple(self._session()._raster_interaction_snapshot())

    def color_limits(self) -> NumericRange | None:
        return self._session()._raster_color_limits_snapshot()

class RasterPlotHost:
    """Serialize one ``PlotSession`` and publish immutable latest-only fronts.

    The callable factory is invoked on the owned worker, and closing the host
    always closes that session.  A live Figure is never exposed to callers.
    :meth:`from_plot` is the standard immutable-data/spec construction path;
    the raw factory remains available for deliberate ``PlotSession`` subclasses.
    Public mutation methods never run plot work on the caller.  Pending data and
    high-frequency controls coalesce only with the same semantic key; selector,
    fit, unit, size and other distinct commands retain their ordering.
    """

    def __init__(
        self,
        session_factory: Callable[[], "PlotSession"],
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory: Callable[[], PlotSession] | None = session_factory
        self._session: PlotSession | None = None
        self._session_defaults: PlotLibraryDefaults | None = None
        self._release_session_host: Callable[[], None] | None = None
        self._worker_adapter = _WorkerSessionAdapter(self)
        self._condition = Condition(Lock())
        self._pending: deque[_WorkerTask] = deque()
        self._closing = False
        self._closed = False
        self._ready = False
        self._startup_error: BaseException | None = None
        self._host_id = uuid4().hex
        self._sequence = 0
        self._front: RasterFront | None = None
        self._front_callbacks: list[Callable[[RasterFront], None]] = []
        self._thread = Thread(
            target=self._run,
            name="zlc-raster-plot",
            daemon=True,
        )
        self._thread.start()
        self._initial_front = self._submit(
            lambda: None,
            mode=_DispatchMode.PUBLISH,
            coalesce_key="initial-front",
        )

    @classmethod
    def from_plot(
        cls,
        data: "PlotInput",
        spec: "PlotSpec",
        *,
        size: str | None = None,
        parameters: Mapping[str, object] | None = None,
        defaults: PlotLibraryDefaults = DEFAULTS,
        device_pixel_ratio: float = 1.0,
        fit_engine: "FitEngine | None" = None,
    ) -> "RasterPlotHost":
        """Create the session on this host's worker from immutable plot inputs."""

        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        initial_parameters = None if parameters is None else dict(parameters)

        def create_session() -> "PlotSession":
            from .session import PlotSession

            return PlotSession(
                data,
                spec,
                size=size,
                parameters=initial_parameters,
                defaults=defaults,
                device_pixel_ratio=device_pixel_ratio,
                fit_engine=fit_engine,
            )

        return cls(create_session)

    def _require_session(self) -> "PlotSession":
        with self._condition:
            while (
                not self._ready
                and self._startup_error is None
                and not self._closed
            ):
                self._condition.wait()
            if self._startup_error is not None:
                raise RuntimeError("raster plot session initialization failed") from self._startup_error
            if self._session is None:
                raise RuntimeError("raster plot host closed before session initialization")
            return self._session

    @property
    def front(self) -> RasterFront | None:
        with self._condition:
            return self._front

    @property
    def host_id(self) -> str:
        """Stable opaque identity stamped into every front from this host."""

        return self._host_id

    @property
    def defaults(self) -> PlotLibraryDefaults:
        """The immutable configuration owned by the worker-side session."""

        with self._condition:
            while (
                self._session_defaults is None
                and self._startup_error is None
                and not self._closed
            ):
                self._condition.wait()
            if self._startup_error is not None:
                raise RuntimeError(
                    "raster plot session initialization failed"
                ) from self._startup_error
            if self._session_defaults is None:
                raise RuntimeError("raster plot host is closed")
            return self._session_defaults

    def wait_for_front(self, timeout: float | None = None) -> RasterFront:
        """Wait for the first complete raster before exposing a new window."""

        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative or None")
        with self._condition:
            front = self._front
            initial = self._initial_front
            closing = self._closing
        if front is not None:
            return front
        if closing or initial is None:
            raise RuntimeError("raster plot host closed before its first front")
        return initial.result(timeout=timeout).front

    def subscribe_front(
        self,
        callback: Callable[[RasterFront], None],
    ) -> Callable[[], None]:
        if not callable(callback):
            raise TypeError("front callback must be callable")
        with self._condition:
            if self._closing:
                raise RuntimeError("raster plot host is closing")
            self._front_callbacks.append(callback)

        def unsubscribe() -> None:
            with self._condition:
                if callback in self._front_callbacks:
                    self._front_callbacks.remove(callback)

        return unsubscribe

    def _subscribe_session_event(
        self,
        callback: Callable[[EventT], object],
        install: Callable[
            ["PlotSession", Callable[[EventT], object]],
            Callable[[], None],
        ],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        if not callable(callback):
            raise TypeError("event callback must be callable")
        if not callable(install):
            raise TypeError("event subscription installer must be callable")

        def subscribe() -> Callable[[], Future[RasterOperation[None]]]:
            release = install(self._require_session(), callback)
            if not callable(release):
                raise TypeError("session subscription must return an unsubscribe callback")

            def unsubscribe() -> Future[RasterOperation[None]]:
                return self._submit(release, mode=_DispatchMode.CONTROL)

            return unsubscribe

        return self._submit(subscribe, mode=_DispatchMode.CONTROL)

    def subscribe_display(
        self,
        callback: Callable[["DisplayState"], object],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a display callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_display(listener),
        )

    def subscribe_fit(
        self,
        callback: Callable[["FitEvent"], object],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a fit callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_fit(listener),
        )

    def subscribe_selection(
        self,
        callback: Callable[["SelectionEvent"], object],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a selection callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_selection(listener),
        )

    def live_controller(
        self,
        contract: "LiveContract",
        *,
        refresh_interval_ms: int | None = None,
    ) -> "LivePlotController":
        """Create a live producer bridge whose mutations stay on this worker."""

        from .live import LivePlotController

        self.wait_for_front()
        return LivePlotController(
            self._worker_adapter,
            contract,
            refresh_interval_ms=refresh_interval_ms,
            dispatch=self.dispatch,
        )

    def _dispatch_callback(
        self,
        callback: Callable[[], ValueT],
        mode: _DispatchMode,
        *,
        after_publish: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> Future[RasterOperation[ValueT]]:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if mode is _DispatchMode.PRESENTATION:
            if not callable(after_publish) or not callable(on_abort):
                raise TypeError("presentation requires finalization and abort callbacks")
        elif after_publish is not None or on_abort is not None:
            raise TypeError("finalization/abort require presentation dispatch")
        if current_thread() is not self._thread:
            return self._submit(
                callback,
                mode=mode,
                after_publish=after_publish,
                on_abort=on_abort,
            )
        completion: Future[RasterOperation[ValueT]] = Future()
        if not completion.set_running_or_notify_cancel():
            return completion
        try:
            operation = self._execute_worker_task(
                callback,
                mode,
                after_publish=after_publish,
                on_abort=on_abort,
            )
        except BaseException as error:
            self._record_terminal_rollback_failure(error)
            completion.set_exception(error)
        else:
            completion.set_result(operation)
        return completion

    def _execute_worker_task(
        self,
        callback: Callable[[], ValueT],
        mode: _DispatchMode,
        *,
        after_publish: Callable[[], None] | None,
        on_abort: Callable[[], None] | None,
    ) -> RasterOperation[ValueT]:
        """Run the only callback-to-front presentation state machine."""

        promoted = False
        try:
            value = callback()
            publishes = mode.publishes or (
                mode is _DispatchMode.ADAPTIVE
                and bool(getattr(value, "publish_front", False))
            )
            front = self._capture_front() if publishes else self.front
            if front is None:
                front = self._capture_front()
            if publishes:
                if not self._promote(front):
                    raise RuntimeError("raster host closed before front promotion")
                promoted = True
            if mode is _DispatchMode.PRESENTATION:
                assert after_publish is not None
                after_publish()
            return RasterOperation(value, front)
        except BaseException as error:
            if mode is _DispatchMode.PRESENTATION and not promoted:
                assert on_abort is not None
                try:
                    on_abort()
                except BaseException as rollback:
                    raise _PresentationRollbackError(error, rollback) from error
            raise

    def _record_terminal_rollback_failure(self, error: BaseException) -> None:
        if not isinstance(error, _PresentationRollbackError):
            return
        with self._condition:
            self._closing = True
            pending = tuple(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        for task in pending:
            if not task.completion.done():
                task.completion.set_exception(error)

    def _dispatch_session(
        self,
        command: _SessionCommand,
        *args: Any,
        _execute: Callable[[], Any] | None = None,
        _after_publish: Callable[[], None] | None = None,
        _on_abort: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Future[RasterOperation[Any]]:
        callback = (
            (lambda: self._worker_adapter.invoke(command, *args, **kwargs))
            if _execute is None
            else _execute
        )
        if not callable(callback):
            raise TypeError("session command executor must be callable")
        return self._submit(
            callback,
            mode=command.mode,
            coalesce_key=command.coalesce_key(args, kwargs),
            after_publish=_after_publish,
            on_abort=_on_abort,
        )

    def dispatch(self, callback: Callable[[], None]) -> Future[RasterOperation[None]]:
        """Marshal a session completion onto the serial render worker."""

        return self._dispatch_callback(callback, _DispatchMode.PUBLISH)

    def dispatch_control(
        self,
        callback: Callable[[], None],
    ) -> Future[RasterOperation[None]]:
        """Run owner-affine control work without publishing a raster front."""

        return self._dispatch_callback(callback, _DispatchMode.CONTROL)

    def dispatch_presentation(
        self,
        callback: Callable[[], None],
        after_publish: Callable[[], None],
        on_abort: Callable[[], None],
    ) -> Future[RasterOperation[None]]:
        """Queue one non-reentrant publish/finalize presentation transaction."""

        return self._submit(
            callback,
            mode=_DispatchMode.PRESENTATION,
            after_publish=after_publish,
            on_abort=on_abort,
        )

    def update_data(
        self,
        data: object,
        *,
        revision: int | None = None,
    ) -> Future[RasterOperation[None]]:
        return self._dispatch_session(_SessionCommand.UPDATE_DATA, data, revision=revision)

    def update_image_overlay(
        self,
        overlay: "ImagePointOverlay",
    ) -> Future[RasterOperation["ImagePointOverlay"]]:
        """Coalesce independent Image point revisions on the render worker."""

        return self._dispatch_session(_SessionCommand.UPDATE_IMAGE_OVERLAY, overlay)

    def update_image_frame(
        self,
        frame: "ImageFrame",
    ) -> Future[RasterOperation["ImageFrame"]]:
        """Coalesce complete image frames on the serial render worker."""

        return self._dispatch_session(_SessionCommand.UPDATE_IMAGE_FRAME, frame)

    def set_parameter(
        self,
        name: str,
        value: object,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.SET_PARAMETER, name, value)

    def set_parameters(
        self,
        values: Mapping[str, object],
    ) -> Future[RasterOperation["DisplayState"]]:
        prepared = dict(values)
        return self._dispatch_session(_SessionCommand.SET_PARAMETERS, prepared)

    def describe_display(self) -> Future[RasterOperation["DisplayDescription"]]:
        """Return the worker session's immutable control-plane description."""

        return self._dispatch_session(_SessionCommand.DESCRIBE_DISPLAY)

    def configuration(self) -> Future[RasterOperation["PlotSessionConfig"]]:
        """Return immutable spec, size and display parameters for persistence."""

        return self._dispatch_session(_SessionCommand.CONFIGURATION)

    def replace_spec(
        self,
        spec: "PlotSpec",
        *,
        parameters: Mapping[str, object] | None = None,
    ) -> Future[RasterOperation["PlotSessionConfig"]]:
        """Replace semantic roles on the existing raster surface."""

        prepared = None if parameters is None else dict(parameters)
        return self._dispatch_session(
            _SessionCommand.REPLACE_SPEC,
            spec,
            parameters=prepared,
        )

    def resolved_color_limits(
        self,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[NumericRange]]:
        """Return the effective clim painted by the worker session."""

        return self._dispatch_session(
            _SessionCommand.RESOLVED_COLOR_LIMITS,
            display=display,
        )

    def set_labels(
        self,
        *,
        title: str | None | object = _UNSET,
        x: str | None | object = _UNSET,
        y: str | None | object = _UNSET,
        value: str | None | object = _UNSET,
    ) -> Future[RasterOperation["DisplayState"]]:
        """Update title/axis/value text through the serial render worker."""

        updates: dict[str, object] = {}
        for name, selected in (
            ("title", title),
            ("x", x),
            ("y", y),
            ("value", value),
        ):
            if selected is not _UNSET:
                updates[name] = selected
        return self._dispatch_session(_SessionCommand.SET_LABELS, **updates)

    def set_relim_mode(self, mode: str) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.SET_RELIM_MODE, mode)

    def set_y_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.SET_Y_LIMITS, low, high, fixed=fixed)

    def reset_y_limits(
        self,
        *,
        mode: str = "normal",
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.RESET_Y_LIMITS, mode=mode)

    def set_color_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            _SessionCommand.SET_COLOR_LIMITS, low, high, fixed=fixed
        )

    def reset_color_limits(
        self,
        *,
        mode: str = "tight",
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.RESET_COLOR_LIMITS, mode=mode)

    def set_x_limits(
        self,
        low: float,
        high: float,
    ) -> Future[RasterOperation[RectangleRange]]:
        return self._dispatch_session(_SessionCommand.SET_X_LIMITS, low, high)

    def set_view_limits(
        self,
        *,
        x: tuple[float, float] | NumericRange | None = None,
        y: tuple[float, float] | NumericRange | None = None,
    ) -> Future[RasterOperation[RectangleRange]]:
        return self._dispatch_session(_SessionCommand.SET_VIEW_LIMITS, x=x, y=y)

    def set_size(self, preset: str) -> Future[RasterOperation["SurfacePlan"]]:
        return self._dispatch_session(_SessionCommand.SET_SIZE, preset)

    def set_device_pixel_ratio(self, ratio: float) -> Future[RasterOperation["SurfacePlan"]]:
        selected = _finite(ratio, "device pixel ratio")
        if selected <= 0.0:
            raise ValueError("device pixel ratio must be positive")
        return self._dispatch_session(_SessionCommand.SET_DEVICE_PIXEL_RATIO, selected)

    def set_axis_unit(
        self,
        axis: object,
        unit: str | None,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.SET_AXIS_UNIT, axis, unit)

    def set_value_unit(self, unit: str | None) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.SET_VALUE_UNIT, unit)

    def set_time_unit(self, unit: str | None) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(_SessionCommand.SET_TIME_UNIT, unit)

    def save(
        self,
        path: str | Path,
        *,
        dpi: float | None = None,
        export_scale: float | None = None,
        **kwargs: Any,
    ) -> Future[RasterOperation[None]]:
        options = dict(kwargs)
        return self._dispatch_session(
            _SessionCommand.SAVE,
            path,
            dpi=dpi,
            export_scale=export_scale,
            **options,
        )

    def clear_fit(self) -> Future[RasterOperation[None]]:
        return self._dispatch_session(_SessionCommand.CLEAR_FIT)

    def fit_models(self) -> Future[RasterOperation[tuple["FitModelSpec", ...]]]:
        """Return the fit catalogue owned by this host's session."""

        return self._dispatch_session(_SessionCommand.FIT_MODELS)

    def selectors(self) -> Future[RasterOperation[tuple[SelectorState, ...]]]:
        """Return the current canonical selector geometry without slicing data."""

        return self._dispatch_session(_SessionCommand.SELECTORS)

    def selector_state(
        self,
        kind: SelectorKind,
        *,
        display: bool = False,
    ) -> Future[RasterOperation[SelectorState]]:
        """Return one selector in canonical or current display coordinates."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            _SessionCommand.SELECTOR_STATE,
            kind,
            display=display,
        )

    def selector_data(self, kind: SelectorKind) -> Future[RasterOperation["SelectorData"]]:
        """Materialize selector output only for this explicit application call."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(_SessionCommand.SELECTOR_DATA, kind)

    def remove_selector(self, kind: SelectorKind) -> Future[RasterOperation[SelectorState]]:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(_SessionCommand.REMOVE_SELECTOR, kind)

    def set_selector_value(
        self,
        kind: SelectorKind,
        value: object,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            _SessionCommand.SET_SELECTOR_VALUE,
            kind,
            value,
            display=display,
        )

    def set_area_selector(
        self,
        x: NumericRange,
        y: NumericRange,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            _SessionCommand.SET_AREA_SELECTOR,
            x,
            y,
            display=display,
        )

    def set_x_selector(
        self,
        low: float,
        high: float,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            _SessionCommand.SET_X_SELECTOR,
            low,
            high,
            display=display,
        )

    def set_threshold_selector(
        self,
        value: float,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            _SessionCommand.SET_THRESHOLD_SELECTOR,
            value,
            display=display,
        )

    def set_facet_thresholds(
        self,
        thresholds: tuple[float | None, ...] | list[float | None],
        *,
        display: bool = True,
    ) -> Future[RasterOperation[tuple[float | None, ...]]]:
        return self._dispatch_session(
            _SessionCommand.SET_FACET_THRESHOLDS,
            tuple(thresholds),
            display=display,
        )

    def set_crosshair_selector(
        self,
        x: float,
        y: float,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            _SessionCommand.SET_CROSSHAIR_SELECTOR,
            x,
            y,
            display=display,
        )

    def fit(
        self,
        model: str | "FitModelSpec",
        *,
        selector_kind: SelectorKind | None = None,
        initial: Mapping[str, float] | tuple[float, ...] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: "FitOptions | None" = None,
        live: bool = True,
        fit_all_facets: bool = False,
    ) -> Future[RasterOperation["FitResult | FacetFitBatchResult"]]:
        """Fit asynchronously and complete after the accepted front is promoted."""

        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        completion: Future[RasterOperation[FitResult | FacetFitBatchResult]] = Future()
        started = self._dispatch_session(
            _SessionCommand.FIT,
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
            fit_all_facets=fit_all_facets,
        )

        def start_finished(
            value: Future[
                RasterOperation[Future[FitResult | FacetFitBatchResult]]
            ],
        ) -> None:
            if completion.cancelled():
                return
            try:
                analysis = value.result().value
                if not isinstance(analysis, Future):
                    raise TypeError("session.fit_async must return Future")
            except BaseException as error:
                if not completion.done():
                    completion.set_exception(error)
                return

            def analysis_finished(
                result_future: Future[FitResult | FacetFitBatchResult],
            ) -> None:
                if completion.cancelled():
                    return
                try:
                    result = result_future.result()
                except BaseException as error:
                    if not completion.done():
                        completion.set_exception(error)
                    return
                if current_thread() is not self._thread:
                    completion.set_exception(
                        RuntimeError(
                            "session fit completion escaped the raster owner thread"
                        )
                    )
                    return
                front = self.front
                if front is None:
                    completion.set_exception(
                        RuntimeError("accepted fit has no promoted raster front")
                    )
                    return
                result_revision = getattr(
                    result,
                    "data_revision",
                    getattr(result, "source_revision", None),
                )
                if result_revision != front.identity.data_revision:
                    completion.set_exception(
                        RuntimeError(
                            "accepted fit result does not match its promoted front"
                        )
                    )
                    return
                completion.set_result(RasterOperation(result, front))

            analysis.add_done_callback(analysis_finished)

        started.add_done_callback(start_finished)
        return completion

    def _pointer_event(
        self,
        action: str,
        x: float,
        y: float,
        *,
        button: int | None = None,
        double: bool = False,
        step: float = 0.0,
        key: str | None = None,
        identity: RasterIdentity | None = None,
        axes: AxisTransform | None = None,
        interaction: RasterInteractionMap | None = None,
    ) -> Future[RasterOperation[object]]:
        """Route Qt raster input through PlotSession's interaction engine."""

        selected_action = nonempty_text(action, "pointer action").lower()
        if selected_action not in {
            "press",
            "move",
            "release",
            "scroll",
            "key",
            "cancel",
        }:
            raise ValueError(f"unknown pointer action {action!r}")
        x_value = _finite(x, "pointer x")
        y_value = _finite(y, "pointer y")
        if identity is not None and not isinstance(identity, RasterIdentity):
            raise TypeError("pointer identity must be RasterIdentity or None")
        if axes is not None and not isinstance(axes, AxisTransform):
            raise TypeError("pointer axes must be AxisTransform or None")
        if axes is not None and identity is None:
            raise ValueError("pointer axes require their painted front identity")
        if interaction is not None and not isinstance(
            interaction, RasterInteractionMap
        ):
            raise TypeError("pointer interaction must be RasterInteractionMap or None")
        if interaction is not None and identity is None:
            raise ValueError("pointer interaction requires its painted front identity")

        def apply() -> object:
            session = self._require_session()
            if identity is not None:
                revisions = session.revisions
                plan = session.surface_plan
                if (
                    identity.host_id != self._host_id
                    or int(revisions.display) != identity.display_revision
                    or int(revisions.layout) != identity.layout_revision
                    or str(plan.kind) != identity.kind
                    or str(plan.preset) != identity.preset
                ):
                    raise RuntimeError(
                        "the painted pointer front is no longer layout-compatible"
                    )
                # Data and selector revisions are deliberately *not* part of
                # the pointer-surface validity contract.  A live plot may
                # promote a newer frame between the Qt press and the worker
                # command (or during a drag).  The pixel-to-data transform is
                # captured for the gesture, while the session owns the latest
                # selector state.  Rejecting that normal publication race
                # turned continuous interaction into a fatal error.  Only a
                # display/layout change invalidates the captured surface; a
                # new data revision is rebased onto the same surface.
            return session._raster_pointer_event(
                selected_action,
                x_value,
                y_value,
                button=button,
                double=double,
                step=step,
                key=key,
                axes_snapshot=axes,
            )

        return self._dispatch_session(
            _SessionCommand.POINTER,
            selected_action,
            _execute=apply,
        )

    def set_viewport(
        self,
        x: NumericRange,
        y: NumericRange,
    ) -> Future[RasterOperation[RectangleRange]]:
        """Set visible ranges in the current display units."""

        return self._dispatch_session(_SessionCommand.SET_VIEWPORT, x, y)

    def focus_facet(
        self,
        identity: RasterIdentity,
        axes: AxisTransform,
    ) -> Future[RasterOperation[None]]:
        """Open one cell from the exact compatible FacetGrid overview."""

        if not isinstance(identity, RasterIdentity):
            raise TypeError("identity must be RasterIdentity")
        if not isinstance(axes, AxisTransform):
            raise TypeError("axes must be AxisTransform")
        if identity.kind != "facet_grid" or axes.role != "facet_cell":
            raise TypeError("facet focus requires a FacetGrid cell")
        if axes.cell_index is None:
            raise ValueError("facet focus requires a cell index")

        def apply() -> None:
            session = self._require_session()
            revisions = session.revisions
            if (
                identity.host_id != self._host_id
                or revisions.display != identity.display_revision
                or revisions.layout != identity.layout_revision
            ):
                raise RuntimeError(
                    "the overview display/layout changed before facet focus"
                )
            session.focus_facet(axes.cell_index)

        return self._dispatch_session(_SessionCommand.FOCUS_FACET, _execute=apply)

    def show_facet_overview(self) -> Future[RasterOperation[None]]:
        """Return a focused FacetGrid front to its complete overview."""

        return self._dispatch_session(_SessionCommand.SHOW_FACET_OVERVIEW)

    def reset_viewport(self) -> Future[RasterOperation[None]]:
        return self._dispatch_session(_SessionCommand.RESET_VIEWPORT)

    def _submit(
        self,
        callback: Callable[[], ValueT],
        *,
        mode: _DispatchMode,
        coalesce_key: object | None = None,
        after_publish: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> Future[RasterOperation[ValueT]]:
        if not isinstance(mode, _DispatchMode):
            raise TypeError("task mode must be _DispatchMode")
        if mode is _DispatchMode.PRESENTATION:
            if not callable(after_publish) or not callable(on_abort):
                raise TypeError("presentation task requires finalize and abort callbacks")
        elif after_publish is not None or on_abort is not None:
            raise TypeError("finalize/abort are valid only for a presentation task")
        completion: Future[RasterOperation[ValueT]] = Future()
        task = _WorkerTask(
            callback,
            completion,
            mode,
            coalesce_key,
            after_publish,
            on_abort,
        )
        superseded: Future[RasterOperation[Any]] | None = None
        with self._condition:
            if self._closing:
                completion.set_exception(RuntimeError("raster plot host is closing"))
                return completion
            if (
                coalesce_key is not None
                and self._pending
                and self._pending[-1].coalesce_key == coalesce_key
            ):
                pending = self._pending[-1]
                superseded = pending.completion
                self._pending[-1] = task
            else:
                self._pending.append(task)
            self._condition.notify()
        # Future.cancel() invokes done callbacks synchronously.  Never run
        # application callbacks while holding the non-reentrant queue lock.
        if superseded is not None:
            superseded.cancel()
        return completion

    def _run(self) -> None:
        try:
            try:
                from .session import PlotSession

                factory = self._session_factory
                if factory is None:
                    raise RuntimeError("raster session factory was already released")
                session = factory()
                if not isinstance(session, PlotSession):
                    raise TypeError("session_factory must create PlotSession")
                self._session = session
                defaults = session.defaults
                release = session.attach_host(
                    self,
                    self.dispatch_control,
                    presentation_dispatch=self.dispatch_presentation,
                )
                if not callable(release):
                    raise TypeError("session.attach_host must return a release callback")
                self._session_defaults = defaults
                self._release_session_host = release
            except BaseException as error:
                with self._condition:
                    self._startup_error = error
                    pending = tuple(self._pending)
                    self._pending.clear()
                    self._closing = True
                    self._condition.notify_all()
                for task in pending:
                    if not task.completion.done():
                        task.completion.set_exception(error)
                return
            else:
                with self._condition:
                    self._ready = True
                    self._condition.notify_all()
            while True:
                with self._condition:
                    while not self._pending and not self._closing:
                        self._condition.wait()
                    if not self._pending and self._closing:
                        return
                    task = self._pending.popleft()
                if not task.completion.set_running_or_notify_cancel():
                    with self._condition:
                        self._condition.notify_all()
                    continue
                try:
                    operation = self._execute_worker_task(
                        task.callback,
                        task.mode,
                        after_publish=task.after_publish,
                        on_abort=task.on_abort,
                    )
                except BaseException as error:
                    self._record_terminal_rollback_failure(error)
                    task.completion.set_exception(error)
                else:
                    task.completion.set_result(operation)
                finally:
                    with self._condition:
                        self._condition.notify_all()
        finally:
            try:
                session = self._session
                if session is not None:
                    release, self._release_session_host = (
                        self._release_session_host,
                        None,
                    )
                    if release is not None:
                        release()
                    session.close()
            finally:
                with self._condition:
                    self._session = None
                    self._session_defaults = None
                    self._session_factory = None
                    self._front = None
                    self._front_callbacks.clear()
                    self._initial_front = None
                    self._closed = True
                    self._condition.notify_all()

    def _capture_front(self) -> RasterFront:
        session = self._require_session()
        plan = session.surface_plan
        width, height = tuple(plan.raster_size)
        pixels = self._worker_adapter.capture_rgba()
        if pixels.shape != (height, width, 4):
            raise RuntimeError(
                "session RGBA shape does not match its surface raster size"
            )
        raw = pixels.tobytes(order="C")
        buffer = RasterBuffer(width, height, raw)
        axes_maps = session._raster_axes_snapshot()
        selectors = self._worker_adapter.interaction_state()
        color_limits = self._worker_adapter.color_limits()
        facet_focus_index = session.facet_focus_index
        source_revisions = session._raster_source_revisions_snapshot()
        revisions = session.revisions
        self._sequence += 1
        identity = RasterIdentity(
            host_id=self._host_id,
            sequence=self._sequence,
            data_generation=session.data_generation,
            data_revision=int(revisions.data),
            image_overlay_revision=session.image_overlay_revision,
            display_revision=int(revisions.display),
            layout_revision=int(revisions.layout),
            kind=str(plan.kind),
            preset=str(plan.preset),
        )
        return RasterFront(
            identity=identity,
            source_revisions=source_revisions,
            buffer=buffer,
            logical_size=tuple(plan.logical_size),
            logical_dpi=float(plan.logical_dpi),
            device_pixel_ratio=float(plan.device_pixel_ratio),
            interaction=RasterInteractionMap(
                axes=axes_maps,
                selectors=selectors,
                color_limits=color_limits,
                facet_focus_index=facet_focus_index,
            ),
        )

    def _promote(self, front: RasterFront) -> bool:
        with self._condition:
            if self._closing:
                return False
            self._front = front
            callbacks = tuple(self._front_callbacks)
        for callback in callbacks:
            try:
                callback(front)
            except BaseException:
                continue
        return True

    def close(self, *, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative or None")
        with self._condition:
            if self._closing:
                thread = self._thread
                pending: tuple[_WorkerTask, ...] = ()
            else:
                self._closing = True
                pending = tuple(self._pending)
                self._pending.clear()
                self._front_callbacks.clear()
                self._condition.notify_all()
                thread = self._thread
        # See _submit(): cancellation callbacks may re-enter this host.
        for task in pending:
            task.completion.cancel()
        if thread is not None and thread is not current_thread():
            thread.join(timeout)
        stopped = thread is None or not thread.is_alive()
        if stopped:
            with self._condition:
                self._initial_front = None
                if self._thread is thread:
                    self._thread = None
        return stopped

    def __enter__(self) -> "RasterPlotHost":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


__all__ = [
    "RasterBuffer",
    "RasterFront",
    "RasterIdentity",
    "RasterInteractionMap",
    "RasterOperation",
    "RasterPlotHost",
]
