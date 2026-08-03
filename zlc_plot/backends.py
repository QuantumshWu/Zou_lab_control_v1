"""Optional notebook and PyQt5 adapters for PlotSession.

Qt and ipympl are loaded only when their adapters are materialised. Notebook
views construct their own widget canvas, so users need no backend magic and the
package does not alter unrelated pyplot figures.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
import importlib
import math
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable

from ._axis_transform import AxisTransform
from ._selector_scene import (
    ColorLimitCandidate,
    SceneKind,
    SelectorLine,
    SelectorMarkers,
    SelectorScene,
    SelectorTarget,
    SelectorText,
)
from .assets import HELVETICA_LIGHT_FAMILY, helvetica_light_path
from .raster import (
    RasterFront,
    RasterPlotHost,
)
from .selectors import SelectorState

if TYPE_CHECKING:
    from .live import LiveContract, LivePlotController
    from .session import PlotSession


class BackendUnavailableError(RuntimeError):
    """An explicitly requested optional frontend is not installed/configured."""


def _complete_cleanup(callbacks: tuple[Callable[[], object], ...]) -> None:
    """Run every cleanup edge, then re-raise the first failure unchanged."""

    first_error: BaseException | None = None
    for callback in callbacks:
        try:
            callback()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _surface_logical_size(plan: object) -> tuple[int, int]:
    value = getattr(plan, "logical_size", None)
    try:
        width, height = tuple(value)
    except (TypeError, ValueError) as error:
        raise TypeError("session.surface_plan does not expose logical_size") from error
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("surface logical size must contain positive integers")
    return (width, height)


def _surface_device_pixel_ratio(plan: object) -> float:
    value = getattr(plan, "device_pixel_ratio", None)
    if isinstance(value, bool):
        raise TypeError("surface device pixel ratio must be a real number")
    try:
        ratio = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "session.surface_plan does not expose device_pixel_ratio"
        ) from error
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("surface device pixel ratio must be positive and finite")
    return ratio


def _surface_logical_dpi(plan: object) -> float:
    value = getattr(plan, "logical_dpi", None)
    if isinstance(value, bool):
        raise TypeError("surface logical dpi must be a real number")
    try:
        dpi = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("surface logical dpi must be a real number") from error
    if not math.isfinite(dpi) or dpi <= 0.0:
        raise ValueError("surface logical dpi must be positive and finite")
    return dpi


def _surface_figure_size_inches(plan: object) -> tuple[float, float]:
    value = getattr(plan, "figure_size_inches", None)
    try:
        width, height = (float(component) for component in tuple(value))
    except (TypeError, ValueError) as error:
        raise TypeError(
            "session.surface_plan does not expose figure_size_inches"
        ) from error
    if any(
        not math.isfinite(component) or component <= 0.0
        for component in (width, height)
    ):
        raise ValueError("surface figure size must be positive and finite")
    return (width, height)


def _prepare_native_canvas(canvas: object, plan: object) -> None:
    """Bind a native canvas to the plan's logical/physical DPI pair.

    Matplotlib canvas constructors replace ``Figure._original_dpi`` with the
    Figure's current (already physical) DPI.  Restoring the logical value before
    the backend applies its own DPR prevents a second multiplication on Retina
    and mixed-DPI screens.
    """

    figure = getattr(canvas, "figure", None)
    if figure is None:
        raise TypeError("native Matplotlib canvas has no Figure")
    figure._original_dpi = _surface_logical_dpi(plan)
    setter = getattr(canvas, "_set_device_pixel_ratio", None)
    if callable(setter):
        setter(_surface_device_pixel_ratio(plan))
    _restore_native_canvas_geometry(canvas, plan)


def _restore_native_canvas_geometry(canvas: object, plan: object) -> None:
    """Restore exact plan geometry after a native widget resize."""

    figure = getattr(canvas, "figure", None)
    if figure is None:
        raise TypeError("native Matplotlib canvas has no Figure")
    figure.set_size_inches(_surface_figure_size_inches(plan), forward=False)
    physical_dpi = float(getattr(plan, "dpi", figure.dpi))
    if not math.isfinite(physical_dpi) or physical_dpi <= 0.0:
        raise ValueError("surface physical dpi must be positive and finite")
    figure._set_dpi(physical_dpi, forward=False)


def _known_notebook_pixel_ratio(modules: "_NotebookModules") -> float:
    value = getattr(modules.canvas_type, "current_dpi_ratio", 1.0)
    if isinstance(value, bool):
        return 1.0
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return 1.0
    return ratio if math.isfinite(ratio) and ratio > 0.0 else 1.0


def _disable_matplotlib_navigation(canvas: object, manager: object) -> None:
    """Remove Matplotlib's default key/mouse navigation from a package canvas."""

    disconnect = getattr(canvas, "mpl_disconnect", None)
    for name in ("key_press_handler_id", "button_press_handler_id"):
        connection = getattr(manager, name, None)
        if connection is not None and callable(disconnect):
            disconnect(connection)
        if hasattr(manager, name):
            setattr(manager, name, None)
    toolbar = getattr(manager, "toolbar", None)
    for name in ("_id_press", "_id_release", "_id_drag"):
        connection = None if toolbar is None else getattr(toolbar, name, None)
        if connection is not None and callable(disconnect):
            disconnect(connection)
        if toolbar is not None and hasattr(toolbar, name):
            setattr(toolbar, name, None)


def _attach_dispatch(
    session: "PlotSession",
    owner: object,
    dispatch: Callable[[Callable[[], None]], Future[None]],
) -> Callable[[], None]:
    """Attach one interactive owner to the session's host edge."""

    release = session.attach_host(owner, dispatch)
    if not callable(release):
        raise TypeError("session.attach_host must return a release callback")
    return release


class _InteractionGate:
    """One adapter-owned switch and generation for native input transport."""

    __slots__ = ("_enabled", "_generation")

    def __init__(self) -> None:
        self._enabled = True
        self._generation = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def generation(self) -> int:
        return self._generation

    def set_enabled(self, enabled: bool) -> bool:
        if not isinstance(enabled, bool):
            raise TypeError("interaction enabled state must be bool")
        if enabled is self._enabled:
            return False
        self._enabled = enabled
        self._generation += 1
        return True


@dataclass(frozen=True, slots=True)
class _SurfaceSnapshot:
    figure: object
    plan: object


def _capture_surface(session: "PlotSession") -> _SurfaceSnapshot:
    try:
        figure, plan = session._backend_surface()
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("session backend surface is invalid") from error
    return _SurfaceSnapshot(figure, plan)


@dataclass(frozen=True, slots=True)
class _NotebookModules:
    canvas_type: type[Any]
    manager_type: type[Any]
    display: Callable[..., object]
    get_ipython: Callable[[], object]


def _load_notebook_modules() -> _NotebookModules:
    try:
        ipympl_backend = importlib.import_module("ipympl.backend_nbagg")
        display_module = importlib.import_module("IPython.display")
        ipython_module = importlib.import_module("IPython")
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailableError(
            "NotebookView requires ipympl and IPython; "
            "install them with `pip install zlc-plot[notebook]`."
        ) from error
    canvas_type = getattr(ipympl_backend, "Canvas", None)
    manager_type = getattr(ipympl_backend, "FigureManager", None)
    display = getattr(display_module, "display", None)
    get_ipython = getattr(ipython_module, "get_ipython", None)
    if (
        not isinstance(canvas_type, type)
        or not isinstance(manager_type, type)
        or not callable(display)
        or not callable(get_ipython)
    ):
        raise BackendUnavailableError(
            "the installed notebook stack does not provide the ipympl widget canvas"
        )
    return _NotebookModules(canvas_type, manager_type, display, get_ipython)


def notebook_available() -> bool:
    """Return whether the complete optional notebook stack can be loaded."""

    try:
        _load_notebook_modules()
    except BackendUnavailableError:
        return False
    return True


def _close_notebook_canvas(canvas: object | None, manager: object | None) -> None:
    toolbar = None if manager is None else getattr(manager, "toolbar", None)
    destroy = None if manager is None else getattr(manager, "destroy", None)
    close_canvas = None if canvas is None else getattr(canvas, "close", None)
    close_toolbar = None if toolbar is None else getattr(toolbar, "close", None)
    callbacks: list[Callable[[], object]] = []
    if callable(destroy):
        callbacks.append(destroy)
    elif callable(close_canvas):
        callbacks.append(close_canvas)
    if callable(close_toolbar):
        callbacks.append(close_toolbar)
    _complete_cleanup(tuple(callbacks))


class NotebookView:
    """A canvas-only ipympl view over one PlotSession.

    The actual ipympl canvas is placed in the widget tree; no PNG proxy is
    introduced, so the package's selectors and ``mpl_connect`` transport stay
    active.  Matplotlib's default navigation is removed and its
    toolbar/header/footer are not exposed. Fixed-size layout changes retain the
    same Figure and widget model. The application owns every button, parameter
    editor and status readout.
    """

    def __init__(
        self,
        session: "PlotSession",
        *,
        close_session_on_close: bool = False,
    ) -> None:
        from .session import PlotSession

        if not isinstance(session, PlotSession):
            raise TypeError("session must be PlotSession")
        if not isinstance(close_session_on_close, bool):
            raise TypeError("close_session_on_close must be bool")
        self._session = session
        self._close_session = close_session_on_close
        self._modules: _NotebookModules | None = None
        self._canvas: object | None = None
        self._manager: object | None = None
        self._canvas_dpr_callback: Callable[..., object] | None = None
        self._surface_apply_scheduled = False
        self._displayed = False
        self._closed = False
        self._interaction_gate = _InteractionGate()
        self._dispatch_lock = threading.RLock()
        self._pending_dispatch: set[Future[None]] = set()
        self._release_dispatch: Callable[[], None] | None = None
        self._pending_surface = _capture_surface(session)
        unsubscribe = session.subscribe_surface(self._on_session_surface)
        if not callable(unsubscribe):
            raise TypeError("session.subscribe_surface must return an unsubscribe callback")
        self._unsubscribe: Callable[[], object] | None = unsubscribe

    def display(self) -> None:
        """Materialise and display the reusable notebook surface exactly once."""

        canvas = self._ensure_canvas()
        if self._displayed:
            return None
        assert self._modules is not None
        try:
            self._modules.display(canvas)
            self._displayed = True
            # Rebuild dynamic artists for the newly attached non-blitting canvas.
            # The real ipympl view owns its own refresh/initialized handshake.
            self._session.redraw_surface()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise
        return None

    @property
    def session(self) -> "PlotSession":
        """The public session controlled by this notebook canvas."""

        return self._session

    @property
    def interaction_enabled(self) -> bool:
        """Whether this adapter forwards native input to the plot session."""

        return self._interaction_gate.enabled

    def set_interaction_enabled(self, enabled: bool) -> None:
        """Enable or suspend plot input without replacing the widget or Figure."""

        if self._closed:
            raise RuntimeError("NotebookView is closed")
        if not self._interaction_gate.set_enabled(enabled):
            return
        canvas = self._canvas
        if enabled:
            self._session._connect_mpl_events()
        else:
            self._session.cancel_interaction()
            self._session._disconnect_mpl_events()
        if canvas is not None and hasattr(canvas, "capture_scroll"):
            canvas.capture_scroll = enabled

    def live_controller(
        self,
        contract: "LiveContract",
        *,
        refresh_interval_ms: int | None = None,
    ) -> "LivePlotController":
        """Create a live bridge already bound to the notebook owner loop."""

        self._ensure_canvas()
        from .live import LivePlotController

        return LivePlotController(
            self._session,
            contract,
            refresh_interval_ms=refresh_interval_ms,
            dispatch=self.dispatch,
            control_dispatch=self.dispatch,
        )

    def close(self) -> None:
        with self._dispatch_lock:
            if self._closed:
                return
            self._closed = True
            pending_dispatch = tuple(self._pending_dispatch)
            self._pending_dispatch.clear()
            self._surface_apply_scheduled = False
            unsubscribe, self._unsubscribe = self._unsubscribe, None
            release, self._release_dispatch = self._release_dispatch, None
            canvas, self._canvas = self._canvas, None
            manager, self._manager = self._manager, None
            dpr_callback, self._canvas_dpr_callback = (
                self._canvas_dpr_callback,
                None,
            )
            self._displayed = False
        for completion in pending_dispatch:
            completion.cancel()
        callbacks: list[Callable[[], object]] = []
        if unsubscribe is not None:
            callbacks.append(unsubscribe)
        if not self._close_session and not self._interaction_gate.enabled:
            # The gate belongs to this adapter.  Leave a surviving session's
            # native Matplotlib transport ready for its next owner.
            callbacks.append(self._session._connect_mpl_events)
        if release is not None:
            callbacks.append(release)
        if canvas is not None and dpr_callback is not None:
            def remove_dpr_callback() -> None:
                on_msg = getattr(canvas, "on_msg", None)
                if callable(on_msg):
                    on_msg(dpr_callback, remove=True)

            callbacks.append(remove_dpr_callback)
        if canvas is not None or manager is not None:
            callbacks.append(lambda: _close_notebook_canvas(canvas, manager))
        if self._close_session:
            callbacks.append(self._session.close)
        _complete_cleanup(tuple(callbacks))

    def _ensure_canvas(self) -> object:
        if self._closed:
            raise RuntimeError("NotebookView is closed")
        if self._canvas is not None:
            return self._canvas
        try:
            modules = _load_notebook_modules()
            self._modules = modules
            self._kernel_add_callback()
            self._session.adopt_native_device_pixel_ratio(
                _known_notebook_pixel_ratio(modules)
            )
            self._pending_surface = _capture_surface(self._session)
            if self._release_dispatch is None:
                self._release_dispatch = _attach_dispatch(
                    self._session,
                    self,
                    self.dispatch,
                )
            self._apply_surface(self._pending_surface)
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise
        assert self._canvas is not None
        return self._canvas

    def _kernel_add_callback(self) -> Callable[[Callable[[], None]], object]:
        if self._modules is None:
            raise BackendUnavailableError("NotebookView has no loaded kernel modules")
        shell = self._modules.get_ipython()
        loop = getattr(getattr(shell, "kernel", None), "io_loop", None)
        add_callback = getattr(loop, "add_callback", None)
        if not callable(add_callback):
            raise BackendUnavailableError(
                "NotebookView requires an active Jupyter kernel owner loop"
            )
        return add_callback

    def dispatch(self, callback: Callable[[], None]) -> Future[None]:
        """Marshal a worker completion onto the notebook kernel event loop."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        completion: Future[None] = Future()
        with self._dispatch_lock:
            if self._closed:
                completion.set_exception(RuntimeError("NotebookView is closed"))
                return completion
            if self._modules is None:
                completion.set_exception(
                    RuntimeError(
                        "call NotebookView.display() before dispatching worker updates"
                    )
                )
                return completion
            self._pending_dispatch.add(completion)

        def invoke() -> None:
            with self._dispatch_lock:
                if self._closed:
                    self._pending_dispatch.discard(completion)
                    completion.cancel()
                    return
                if not completion.set_running_or_notify_cancel():
                    self._pending_dispatch.discard(completion)
                    return
                try:
                    callback()
                except BaseException as error:
                    completion.set_exception(error)
                else:
                    completion.set_result(None)
                finally:
                    self._pending_dispatch.discard(completion)

        try:
            self._kernel_add_callback()(invoke)
        except BaseException as error:
            with self._dispatch_lock:
                self._pending_dispatch.discard(completion)
                if not completion.done():
                    completion.set_exception(error)
        return completion

    def _on_session_surface(self) -> None:
        with self._dispatch_lock:
            if self._closed:
                return
        snapshot = _capture_surface(self._session)
        with self._dispatch_lock:
            if self._closed:
                return
            self._pending_surface = snapshot
            if self._canvas is None or self._surface_apply_scheduled:
                return
            self._surface_apply_scheduled = True
        try:
            self._kernel_add_callback()(self._apply_pending_surface)
        except BaseException:
            with self._dispatch_lock:
                self._surface_apply_scheduled = False
            raise

    def _apply_pending_surface(self) -> None:
        """Promote only the newest complete surface on the notebook owner loop."""

        with self._dispatch_lock:
            self._surface_apply_scheduled = False
            if self._closed or self._canvas is None:
                return
            snapshot = self._pending_surface
        self._apply_surface(snapshot)

    def _interactive_canvas(
        self,
        figure: object,
        plan: object,
    ) -> tuple[object, object]:
        assert self._modules is not None
        try:
            canvas = self._modules.canvas_type(figure)
            manager = self._modules.manager_type(canvas, id(figure))
        except Exception as error:
            raise TypeError("session backend surface is not a Matplotlib Figure") from error
        if hasattr(canvas, "_closed"):
            canvas._closed = False
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is None:
            raise BackendUnavailableError("the ipympl manager has no toolbar model")
        if hasattr(canvas, "toolbar"):
            canvas.toolbar = toolbar
        if getattr(canvas, "toolbar", toolbar) is not toolbar:
            raise BackendUnavailableError(
                "the ipympl canvas and manager toolbar models disagree"
            )
        _prepare_native_canvas(canvas, plan)
        _disable_matplotlib_navigation(canvas, manager)
        return canvas, manager

    def _observe_notebook_pixel_ratio(self, canvas: object) -> None:
        on_msg = getattr(canvas, "on_msg", None)
        if not callable(on_msg):
            raise BackendUnavailableError(
                "the ipympl canvas cannot report browser device pixel ratio"
            )

        def receive_ratio(
            widget: object,
            content: object,
            _buffers: object,
        ) -> None:
            if self._closed or widget is not self._canvas:
                return
            if not isinstance(content, dict) or content.get("type") != "set_dpi_ratio":
                return
            value = content.get("dpi_ratio")
            if isinstance(value, bool):
                return
            try:
                ratio = float(value)
            except (TypeError, ValueError):
                return
            if not math.isfinite(ratio) or ratio <= 0.0:
                return
            self._session.adopt_native_device_pixel_ratio(ratio)

        on_msg(receive_ratio)
        self._canvas_dpr_callback = receive_ratio

    def _apply_surface(self, snapshot: _SurfaceSnapshot) -> None:
        if self._closed or self._modules is None:
            return
        canvas = self._canvas
        manager = self._manager
        existing_canvas = canvas is not None
        if canvas is None:
            canvas, manager = self._interactive_canvas(
                snapshot.figure,
                snapshot.plan,
            )
            self._canvas = canvas
            self._manager = manager
            self._observe_notebook_pixel_ratio(canvas)
        else:
            if getattr(canvas, "figure", None) is not snapshot.figure:
                raise RuntimeError(
                    "NotebookView requires one stable Figure and canvas model"
                )
            _restore_native_canvas_geometry(canvas, snapshot.plan)
        if existing_canvas:
            assert manager is not None
            figure = getattr(canvas, "figure", None)
            ratio = _surface_device_pixel_ratio(snapshot.plan)
            physical_size = tuple(float(value) for value in figure.bbox.size)
            expected_size = tuple(value / ratio for value in physical_size)
            current_size = tuple(float(value) for value in getattr(canvas, "_size", ()))
            if len(current_size) != 2 or any(
                not math.isclose(current, expected, rel_tol=0.0, abs_tol=0.01)
                for current, expected in zip(
                    current_size,
                    expected_size,
                    strict=True,
                )
            ):
                manager.resize(*physical_size)
        width, height = _surface_logical_size(snapshot.plan)
        widget_layout = getattr(canvas, "layout", None)
        if widget_layout is not None:
            # The same SurfacePlan logical pixels size both the notebook widget
            # and the Qt QImage widget. Browser/container resize adds whitespace.
            widget_layout.width = f"{width}px"
            widget_layout.height = f"{height}px"
            widget_layout.min_width = f"{width}px"
            widget_layout.max_width = f"{width}px"
            widget_layout.min_height = f"{height}px"
            widget_layout.max_height = f"{height}px"
        for name, value in (
            ("toolbar_visible", False),
            ("header_visible", False),
            ("footer_visible", False),
            ("resizable", False),
            ("capture_scroll", self._interaction_gate.enabled),
        ):
            if hasattr(canvas, name):
                setattr(canvas, name, value)

    def __enter__(self) -> "NotebookView":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


@dataclass(frozen=True, slots=True)
class _QtModules:
    QtCore: Any
    QtGui: Any
    QtWidgets: Any


_QT_MODULES: _QtModules | None = None
_QT_WIDGET_CLASS: type[Any] | None = None
_QT_APPLICATION: object | None = None
_QT_FONT_FAMILY: str | None = None
_IPYTHON_QT_LOOP_ENABLED = False
_QT_WINDOW_BIND_RETRY_INTERVAL_MS = 16
_QT_WINDOW_BIND_MAX_ATTEMPTS = 8


def _style_points_to_logical_pixels(points: float, logical_dpi: float) -> float:
    """Convert package style points at one front's logical DPI."""

    return float(points) * float(logical_dpi) / 72.0


def _register_qt5_font(qt_gui: object) -> str:
    """Register and verify the package font once after QApplication exists."""

    global _QT_FONT_FAMILY
    if _QT_FONT_FAMILY is not None:
        return _QT_FONT_FAMILY
    expected = HELVETICA_LIGHT_FAMILY
    font_id = int(
        qt_gui.QFontDatabase.addApplicationFont(str(helvetica_light_path()))
    )
    if font_id < 0:
        raise RuntimeError("Qt could not register the packaged Helvetica Light font")
    families = tuple(qt_gui.QFontDatabase.applicationFontFamilies(font_id))
    if expected not in families:
        qt_gui.QFontDatabase.removeApplicationFont(font_id)
        raise RuntimeError(
            f"canonical Qt font asset identifies as {families!r}, not {expected!r}"
        )
    _QT_FONT_FAMILY = expected
    return expected


def _enable_ipython_qt_loop() -> None:
    """Install IPython's Qt pump once when a widget is created in a notebook."""

    global _IPYTHON_QT_LOOP_ENABLED
    if _IPYTHON_QT_LOOP_ENABLED:
        return
    if os.environ.get("QT_QPA_PLATFORM", "").strip().lower() in {
        "offscreen",
        "minimal",
    }:
        return
    try:
        ipython_module = importlib.import_module("IPython")
        get_ipython = getattr(ipython_module, "get_ipython")
        shell = get_ipython()
    except (AttributeError, ImportError, ModuleNotFoundError):
        return
    if shell is None:
        return
    run_line_magic = getattr(shell, "run_line_magic", None)
    if not callable(run_line_magic):
        return
    try:
        run_line_magic("gui", "qt5")
    except Exception:
        return
    _IPYTHON_QT_LOOP_ENABLED = True


def _configure_qt5_high_dpi(qt_core: object, qt_widgets: object) -> None:
    """Set process-wide Qt5 density attributes before QApplication exists."""

    application_type = getattr(qt_widgets, "QApplication", None)
    qt = getattr(qt_core, "Qt", None)
    if application_type is None or qt is None or application_type.instance() is not None:
        return
    for name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attribute = getattr(qt, name, None)
        if attribute is not None:
            application_type.setAttribute(attribute, True)


def _missing_qt5_high_dpi_attributes(qt_core: object, application: object) -> tuple[str, ...]:
    qt = getattr(qt_core, "Qt", None)
    test_attribute = getattr(application, "testAttribute", None)
    if qt is None or not callable(test_attribute):
        return ()
    return tuple(
        name
        for name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps")
        if getattr(qt, name, None) is not None
        and not bool(test_attribute(getattr(qt, name)))
    )


def _load_qt5_modules() -> _QtModules:
    global _QT_MODULES
    if _QT_MODULES is not None:
        return _QT_MODULES
    try:
        qt_core = importlib.import_module("PyQt5.QtCore")
        qt_gui = importlib.import_module("PyQt5.QtGui")
        qt_widgets = importlib.import_module("PyQt5.QtWidgets")
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailableError(
            "Qt5PlotWidget requires PyQt5; "
            "install them with `pip install zlc-plot[qt]`."
        ) from error
    _configure_qt5_high_dpi(qt_core, qt_widgets)
    _QT_MODULES = _QtModules(qt_core, qt_gui, qt_widgets)
    return _QT_MODULES


def ensure_qt5_application(
    argv: list[str] | tuple[str, ...] | None = None,
) -> object:
    """Return the owner-thread QApplication with Qt5 DPR support enabled.

    Density attributes must be selected before the first QApplication is
    created.  Calling this entry point instead of constructing QApplication
    directly gives both standalone programs and notebook-launched Qt windows
    the same per-monitor DPR contract.
    """

    global _QT_APPLICATION
    modules = _load_qt5_modules()
    QtCore, QtWidgets = modules.QtCore, modules.QtWidgets
    application = QtWidgets.QApplication.instance()
    if application is not None:
        if not isinstance(application, QtWidgets.QApplication):
            raise RuntimeError("the active Qt application is not a QApplication")
        if QtCore.QThread.currentThread() != application.thread():
            raise RuntimeError("QApplication must be accessed from its owner thread")
        missing = _missing_qt5_high_dpi_attributes(QtCore, application)
        if missing:
            raise RuntimeError(
                "the existing QApplication was created without Qt5 High-DPI "
                f"attributes ({', '.join(missing)}); restart the process and call "
                "ensure_qt5_application() before constructing QApplication"
            )
        _register_qt5_font(modules.QtGui)
        _QT_APPLICATION = application
        _enable_ipython_qt_loop()
        return application
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("QApplication must be created on the Python main thread")
    if argv is not None and not isinstance(argv, (list, tuple)):
        raise TypeError("QApplication argv must be a list, tuple, or None")
    selected_argv = sys.argv if argv is None else argv
    arguments = list(selected_argv)
    if any(not isinstance(value, str) for value in arguments):
        raise TypeError("QApplication arguments must be strings")
    _configure_qt5_high_dpi(QtCore, QtWidgets)
    _QT_APPLICATION = QtWidgets.QApplication(arguments)
    _register_qt5_font(modules.QtGui)
    _enable_ipython_qt_loop()
    return _QT_APPLICATION


def qt5_available() -> bool:
    """Return whether the PyQt5-only adapter can be loaded."""

    try:
        _load_qt5_modules()
    except BackendUnavailableError:
        return False
    return True


def _qt5_plot_widget_class() -> type[Any]:
    """Lazily construct and cache the concrete PyQt5 QWidget subclass."""

    global _QT_WIDGET_CLASS
    if _QT_WIDGET_CLASS is not None:
        return _QT_WIDGET_CLASS
    modules = _load_qt5_modules()
    QtCore, QtWidgets = modules.QtCore, modules.QtWidgets
    pointer_cancel_events = frozenset(
        event_type
        for event_type in (
            getattr(QtCore.QEvent, "Hide", None),
            getattr(QtCore.QEvent, "WindowDeactivate", None),
            getattr(QtCore.QEvent, "ApplicationDeactivate", None),
            getattr(QtCore.QEvent, "UngrabMouse", None),
        )
        if event_type is not None
    )
    interaction_events = frozenset(
        event_type
        for event_type in (
            getattr(QtCore.QEvent, "MouseButtonPress", None),
            getattr(QtCore.QEvent, "MouseButtonRelease", None),
            getattr(QtCore.QEvent, "MouseButtonDblClick", None),
            getattr(QtCore.QEvent, "MouseMove", None),
            getattr(QtCore.QEvent, "Wheel", None),
            getattr(QtCore.QEvent, "KeyPress", None),
            getattr(QtCore.QEvent, "KeyRelease", None),
        )
        if event_type is not None
    )

    class _RasterPixelRatioObserver(QtCore.QObject):
        """Observe the actual DPR of one visible QWidget hierarchy."""

        def __init__(
            self,
            host: object,
            on_change: Callable[[float], None],
        ) -> None:
            super().__init__(host)
            self._host = host
            self._on_change: Callable[[float], None] | None = on_change
            self._window_handle = None
            self._observed_screen_ids: set[int] = set()
            self._observed_screens: list[object] = []
            self._last_ratio: float | None = None
            self._bind_scheduled = False
            self._bind_attempts = 0
            self._detached = False
            host.installEventFilter(self)

        @property
        def current_ratio(self) -> float:
            if self._detached or self._host is None:
                raise RuntimeError("pixel-ratio observer is detached")
            top = self._host.window()
            source = self._host if top is None else top
            ratio = float(source.devicePixelRatioF())
            return ratio if math.isfinite(ratio) and ratio > 0.0 else 1.0

        def refresh(self, *, force: bool = False) -> None:
            if self._detached:
                return
            ratio = self.current_ratio
            if not force and ratio == self._last_ratio:
                return
            self._last_ratio = ratio
            callback = self._on_change
            if callback is not None:
                callback(ratio)

        def schedule_bind(self, *, reset: bool = False, delay_ms: int = 0) -> None:
            if self._detached or self._bind_scheduled:
                return
            if reset:
                self._bind_attempts = 0
            self._bind_scheduled = True
            QtCore.QTimer.singleShot(delay_ms, self._bind)

        def _native_window_handle(self, host: object) -> object | None:
            top = host.window()
            if top is None:
                return None
            try:
                top.ensurePolished()
                if top.windowHandle() is None:
                    top.winId()
                return top.windowHandle()
            except RuntimeError:
                return None

        def _bind(self) -> None:
            self._bind_scheduled = False
            host = self._host
            if self._detached or host is None or not host.isVisible():
                return
            handle = self._native_window_handle(host)
            if handle is None:
                if self._bind_attempts >= _QT_WINDOW_BIND_MAX_ATTEMPTS:
                    return
                self._bind_attempts += 1
                self.schedule_bind(delay_ms=_QT_WINDOW_BIND_RETRY_INTERVAL_MS)
                return
            self._bind_attempts = 0
            if handle is not self._window_handle:
                previous = self._window_handle
                self._window_handle = handle
                if previous is not None:
                    try:
                        previous.screenChanged.disconnect(self._screen_changed)
                    except (RuntimeError, TypeError):
                        pass
                handle.screenChanged.connect(self._screen_changed)
            self._observe_screen(handle.screen())
            self.refresh()

        def _observe_screen(self, screen: object) -> None:
            if screen is None or id(screen) in self._observed_screen_ids:
                return
            self._observed_screen_ids.add(id(screen))
            self._observed_screens.append(screen)
            for name in (
                "logicalDotsPerInchChanged",
                "physicalDotsPerInchChanged",
            ):
                signal = getattr(screen, name, None)
                if signal is not None:
                    signal.connect(self._screen_metric_changed)

        def _screen_metric_changed(self, *_args: object) -> None:
            self.refresh()

        def _screen_changed(self, screen: object) -> None:
            if self._detached:
                return
            self._observe_screen(screen)
            QtCore.QTimer.singleShot(0, self.refresh)

        def eventFilter(self, watched: object, event: object) -> bool:
            if (
                not self._detached
                and watched is self._host
                and event.type() == QtCore.QEvent.Show
            ):
                self.schedule_bind(reset=True)
            return super().eventFilter(watched, event)

        def detach(self) -> None:
            if self._detached:
                return
            host = self._host
            if host is not None and QtCore.QThread.currentThread() != host.thread():
                raise RuntimeError("pixel-ratio observer is GUI-thread affine")
            self._detached = True
            self._bind_scheduled = False
            self._bind_attempts = 0
            handle, self._window_handle = self._window_handle, None
            if handle is not None:
                try:
                    handle.screenChanged.disconnect(self._screen_changed)
                except (RuntimeError, TypeError):
                    pass
            for screen in self._observed_screens:
                for name in (
                    "logicalDotsPerInchChanged",
                    "physicalDotsPerInchChanged",
                ):
                    signal = getattr(screen, name, None)
                    if signal is not None:
                        try:
                            signal.disconnect(self._screen_metric_changed)
                        except (RuntimeError, TypeError):
                            pass
            self._observed_screens.clear()
            self._observed_screen_ids.clear()
            if host is not None:
                try:
                    host.removeEventFilter(self)
                except RuntimeError:
                    pass
            self._host = None
            self._on_change = None

    class _Qt5PlotWidget(QtWidgets.QWidget):
        """QImage-only frontend over a serial worker-owned plot session."""

        _front_ready = QtCore.pyqtSignal()
        _gesture_ready = QtCore.pyqtSignal(object)
        _invoke_ready = QtCore.pyqtSignal(object)
        _error_ready = QtCore.pyqtSignal(str)
        errorOccurred = QtCore.pyqtSignal(str)
        surfaceChanged = QtCore.pyqtSignal(object)

        def __init__(
            self,
            host: RasterPlotHost,
            parent: object | None = None,
            *,
            auto_present: bool = True,
        ) -> None:
            if not isinstance(host, RasterPlotHost):
                raise TypeError(
                    "Qt5PlotWidget requires a RasterPlotHost constructed from "
                    "a worker-side session factory"
                )
            if not isinstance(auto_present, bool):
                raise TypeError("auto_present must be a boolean")
            application = QtWidgets.QApplication.instance()
            if application is None:
                raise RuntimeError(
                    "call ensure_qt5_application() before creating Qt5PlotWidget"
                )
            if QtCore.QThread.currentThread() != application.thread():
                raise RuntimeError("Qt5PlotWidget must be created on the Qt owner thread")
            missing = _missing_qt5_high_dpi_attributes(QtCore, application)
            if missing:
                raise RuntimeError(
                    "Qt5PlotWidget requires a High-DPI QApplication; restart the "
                    "process and create it with ensure_qt5_application()"
            )
            _register_qt5_font(modules.QtGui)
            _enable_ipython_qt_loop()
            ready_timeout = host.defaults.runtime.shutdown_timeout_ms / 1000.0
            initial_front = host.wait_for_front(timeout=ready_timeout)
            super().__init__(parent)
            self._host = host
            self._closed = False
            self._interaction_gate = _InteractionGate()
            self._dispatch_lock = threading.RLock()
            self._pending_dispatch: set[Future[None]] = set()
            self._front: RasterFront | None = None
            self._front_handoff_lock = threading.Lock()
            self._queued_front: RasterFront | None = None
            self._front_signal_pending = False
            self._prepared_image: tuple[bytes, object] | None = None
            self._interaction_window = None
            self._gesture_front: RasterFront | None = None
            self._gesture_axes: AxisTransform | None = None
            self._gesture_kind: SceneKind | None = None
            self._candidate: SelectorState | ColorLimitCandidate | None = None
            self._candidate_scene: SelectorScene | None = None
            self._pointer_button: int | None = None
            self._requested_dpr: float | None = None
            self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setMouseTracking(True)
            self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._front_ready.connect(
                self._accept_queued_front,
                type=QtCore.Qt.QueuedConnection,
            )
            self._gesture_ready.connect(
                self._finish_pointer,
                type=QtCore.Qt.QueuedConnection,
            )
            self._invoke_ready.connect(
                self._invoke_callback,
                type=QtCore.Qt.QueuedConnection,
            )
            self._error_ready.connect(
                self._emit_error,
                type=QtCore.Qt.QueuedConnection,
            )
            self._unsubscribe: Callable[[], None] | None = None
            self._pixel_ratio_observer: _RasterPixelRatioObserver | None = None
            try:
                selected_front = self._host.front or initial_front
                if auto_present:
                    self._unsubscribe = self._host.subscribe_front(self._on_front)
                    if not self._install_front(selected_front):
                        raise RuntimeError("Qt rejected the initial raster front")
                    assert self._front is not None
                self._requested_dpr = selected_front.device_pixel_ratio
                self._pixel_ratio_observer = _RasterPixelRatioObserver(
                    self,
                    self._apply_device_pixel_ratio,
                )
                self._apply_device_pixel_ratio(
                    self._pixel_ratio_observer.current_ratio
                )
            except BaseException:
                try:
                    self.close_adapter()
                except BaseException:
                    pass
                raise

        @property
        def host(self) -> RasterPlotHost:
            return self._host

        @property
        def presented_front(self) -> RasterFront | None:
            """The immutable front currently visible in this widget."""

            return self._front

        @property
        def interaction_enabled(self) -> bool:
            """Whether this adapter forwards native input to the plot worker."""

            return self._interaction_gate.enabled

        def set_interaction_enabled(self, enabled: bool) -> None:
            """Enable or suspend plot input without rebuilding the raster view."""

            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError(
                    "set_interaction_enabled must run on the Qt widget owner thread"
                )
            if self._closed:
                raise RuntimeError("Qt5PlotWidget is closed")
            if not self._interaction_gate.set_enabled(enabled):
                return
            if enabled:
                self.setFocusPolicy(QtCore.Qt.StrongFocus)
            else:
                self._cancel_active_interaction()
                self.clearFocus()
                self.setFocusPolicy(QtCore.Qt.NoFocus)

        def present_front(self, front: RasterFront) -> bool:
            """Atomically install one compatible front on the Qt owner thread.

            A manually staged front must originate from this widget's host and
            still match its current display/layout surface.  Data revision may
            lag the worker's latest revision so an application can coordinate
            presentation across independent hosts.
            """

            if not isinstance(front, RasterFront):
                raise TypeError("front must be a RasterFront")
            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError(
                    "present_front must run on the Qt widget owner thread"
                )
            if self._closed:
                raise RuntimeError("Qt5PlotWidget is closed")
            if front.identity.host_id != self._host.host_id:
                raise ValueError("front belongs to a different RasterPlotHost")
            latest = self._host.front
            if latest is None:
                raise RuntimeError("RasterPlotHost has no current front")
            if not front.identity.same_surface(latest.identity):
                raise RuntimeError(
                    "front no longer matches the host's current display/layout surface"
                )
            return self._install_front(front)

        def close_adapter(self) -> None:
            if self._closed:
                return
            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError("close_adapter must run on the Qt owner thread")
            with self._dispatch_lock:
                if self._closed:
                    return
                self._closed = True
                pending_dispatch = tuple(self._pending_dispatch)
                self._pending_dispatch.clear()
                self._interaction_gate.set_enabled(False)
            observer, self._pixel_ratio_observer = self._pixel_ratio_observer, None
            interaction_window, self._interaction_window = self._interaction_window, None
            unsubscribe, self._unsubscribe = self._unsubscribe, None
            for completion in pending_dispatch:
                completion.cancel()
            callbacks: list[Callable[[], object]] = [self._cancel_active_interaction]
            if observer is not None:
                callbacks.append(observer.detach)
            if interaction_window is not None and interaction_window is not self:
                callbacks.append(lambda: interaction_window.removeEventFilter(self))
            if unsubscribe is not None:
                callbacks.append(unsubscribe)
            try:
                _complete_cleanup(tuple(callbacks))
            finally:
                with self._front_handoff_lock:
                    self._queued_front = None
                    self._front_signal_pending = False
                self._front = None
                self._prepared_image = None

        def dispatch(self, callback: Callable[[], None]) -> Future[None]:
            """Marshal an application callback to the Qt owner thread."""

            if not callable(callback):
                raise TypeError("callback must be callable")
            completion: Future[None] = Future()
            with self._dispatch_lock:
                if self._closed:
                    completion.set_exception(RuntimeError("Qt5PlotWidget is closed"))
                    return completion
                self._pending_dispatch.add(completion)
            try:
                self._invoke_ready.emit((callback, completion))
            except BaseException as error:
                with self._dispatch_lock:
                    self._pending_dispatch.discard(completion)
                    if not completion.done():
                        completion.set_exception(error)
            return completion

        @QtCore.pyqtSlot(object)
        def _invoke_callback(self, request: object) -> None:
            callback, completion = request
            with self._dispatch_lock:
                if self._closed:
                    self._pending_dispatch.discard(completion)
                    completion.cancel()
                    return
                if not completion.set_running_or_notify_cancel():
                    self._pending_dispatch.discard(completion)
                    return
                try:
                    callback()
                except BaseException as error:
                    completion.set_exception(error)
                else:
                    completion.set_result(None)
                finally:
                    self._pending_dispatch.discard(completion)

        def _on_front(self, front: RasterFront) -> None:
            emit = False
            with self._front_handoff_lock:
                if self._closed:
                    return
                self._queued_front = front
                if not self._front_signal_pending:
                    self._front_signal_pending = True
                    emit = True
            if emit:
                self._front_ready.emit()

        @QtCore.pyqtSlot()
        def _accept_queued_front(self) -> None:
            with self._front_handoff_lock:
                front = self._queued_front
                self._queued_front = None
                self._front_signal_pending = False
            if front is not None:
                latest = self._host.front
                if (
                    latest is not None
                    and front.identity.host_id == self._host.host_id
                    and front.identity.same_surface(latest.identity)
                ):
                    self._install_front(front)

        def _install_front(self, front: RasterFront) -> bool:
            if self._closed:
                return False
            current = self._front
            if (
                current is not None
                and front.identity.sequence <= current.identity.sequence
            ):
                return False
            gesture_front = self._gesture_front
            gesture_axes = self._gesture_axes
            if gesture_front is not None and gesture_axes is not None:
                surface_compatible = (
                    front.identity.kind == gesture_front.identity.kind
                    and front.identity.preset == gesture_front.identity.preset
                    and front.identity.display_revision
                    == gesture_front.identity.display_revision
                    and front.identity.layout_revision
                    == gesture_front.identity.layout_revision
                    and front.logical_size == gesture_front.logical_size
                    and math.isclose(
                        front.device_pixel_ratio,
                        gesture_front.device_pixel_ratio,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
                replacement_axes = next(
                    (
                        candidate
                        for candidate in front.interaction.axes
                        if candidate.role == gesture_axes.role
                        and candidate.cell_index == gesture_axes.cell_index
                    ),
                    None,
                )
                if not surface_compatible or replacement_axes is None:
                    self._cancel_active_interaction()
                elif replacement_axes != gesture_axes:
                    if self._pointer_button == 2 and self._candidate is None:
                        # Pan input remains mapped through the immutable press
                        # transform, while each transient viewport front is
                        # allowed to become visible immediately.
                        return self._promote_front(front)
                    return False
                elif isinstance(self._candidate, ColorLimitCandidate):
                    limits = front.interaction.color_limits
                    value = self._candidate.value
                    if limits is None or not (
                        math.isclose(
                            limits.low,
                            value.low,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-15,
                        )
                        and math.isclose(
                            limits.high,
                            value.high,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-15,
                        )
                    ):
                        return False
                if surface_compatible and replacement_axes is not None:
                    self._gesture_front = front
                    self._gesture_axes = replacement_axes
            return self._promote_front(front)

        def _promote_front(self, front: RasterFront) -> bool:
            current = self._front
            pixels = front.buffer.pixels
            image = modules.QtGui.QImage(
                pixels,
                front.buffer.width,
                front.buffer.height,
                front.buffer.width * 4,
                modules.QtGui.QImage.Format_RGBA8888,
            )
            if image.isNull():
                self.errorOccurred.emit("Qt rejected the immutable RGBA raster front")
                return False
            # QImage borrows the Python buffer.  Promote them as one owner so
            # replacing a front can never briefly leave the old QImage backed
            # by already-released bytes.
            self._prepared_image = (pixels, image)
            self._front = front
            width, height = front.logical_size
            if self.width() != width or self.height() != height:
                self.setFixedSize(width, height)
            self.update()
            surface_changed = (
                current is None
                or current.identity.kind != front.identity.kind
                or current.identity.preset != front.identity.preset
                or current.identity.layout_revision
                != front.identity.layout_revision
                or current.logical_size != front.logical_size
                or not math.isclose(
                    current.device_pixel_ratio,
                    front.device_pixel_ratio,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
            if surface_changed:
                self.surfaceChanged.emit(front.identity)
            return True

        def _track(self, future: Future[object]) -> None:
            def completed(value: Future[object]) -> None:
                try:
                    value.result()
                except CancelledError:
                    return
                except BaseException as error:
                    if not self._closed:
                        self._error_ready.emit(str(error))

            future.add_done_callback(completed)

        @QtCore.pyqtSlot(str)
        def _emit_error(self, message: str) -> None:
            if not self._closed:
                self.errorOccurred.emit(message)

        def closeEvent(self, event: object) -> None:
            self.close_adapter()
            super().closeEvent(event)

        def showEvent(self, event: object) -> None:
            super().showEvent(event)
            self.update()
            window = self.window()
            if window is self._interaction_window:
                return
            previous, self._interaction_window = self._interaction_window, window
            if previous is not None and previous is not self:
                try:
                    previous.removeEventFilter(self)
                except RuntimeError:
                    pass
            if window is not self:
                window.installEventFilter(self)

        def event(self, event: object) -> bool:
            if event.type() in pointer_cancel_events:
                self._cancel_active_interaction()
            if (
                not self._interaction_gate.enabled
                and event.type() in interaction_events
            ):
                event.ignore()
                return False
            return super().event(event)

        def eventFilter(self, watched: object, event: object) -> bool:
            if (
                watched is self._interaction_window
                and event.type() in pointer_cancel_events
            ):
                self._cancel_active_interaction()
            return super().eventFilter(watched, event)

        def _apply_device_pixel_ratio(self, ratio: float) -> None:
            if self._closed:
                return
            selected = float(ratio)
            if self._requested_dpr == selected:
                return
            self._requested_dpr = selected
            if self._pointer_button is not None or self._gesture_front is not None:
                self._cancel_active_interaction()
            self._track(self._host.set_device_pixel_ratio(selected))

        def paintEvent(self, event: object) -> None:
            painter = modules.QtGui.QPainter(self)
            try:
                painter.fillRect(self.rect(), modules.QtGui.QColor("white"))
                prepared = self._prepared_image
                if prepared is not None:
                    image = prepared[1]
                    painter.drawImage(
                        QtCore.QRectF(self.rect()),
                        image,
                        QtCore.QRectF(
                            0.0,
                            0.0,
                            float(image.width()),
                            float(image.height()),
                        ),
                    )
                if self._candidate is not None and self._candidate_scene is not None:
                    self._paint_candidate(
                        painter,
                        self._candidate,
                        self._candidate_scene,
                    )
            finally:
                painter.end()

        def _paint_candidate(
            self,
            painter: object,
            candidate: SelectorState | ColorLimitCandidate,
            scene: SelectorScene,
        ) -> None:
            front = self._gesture_front
            if front is None:
                return
            primitives = scene.primitives_for(candidate.kind)
            if not primitives:
                return
            axes_by_target = {
                SelectorTarget(item.role, item.cell_index): item
                for item in front.interaction.axes
            }
            logical_dpi = front.logical_dpi

            def color(rgba: tuple[float, float, float, float]) -> object:
                return modules.QtGui.QColor.fromRgbF(*rgba)

            def plot_box(target: SelectorTarget) -> tuple[AxisTransform, object]:
                axes = axes_by_target[target]
                left = axes.bounds[0] * self.width()
                top = axes.bounds[1] * self.height()
                right = axes.bounds[2] * self.width()
                bottom = axes.bounds[3] * self.height()
                return axes, QtCore.QRectF(left, top, right - left, bottom - top)

            def point(axes: AxisTransform, value: tuple[float, float]) -> object:
                x, y = axes.display_to_normalized(*value)
                return QtCore.QPointF(x * self.width(), y * self.height())

            def line_pen(primitive: SelectorLine) -> object:
                pen = modules.QtGui.QPen(color(primitive.color))
                pen.setWidthF(
                    _style_points_to_logical_pixels(
                        primitive.width_pt,
                        logical_dpi,
                    )
                )
                pen.setStyle(
                    {
                        "--": QtCore.Qt.DashLine,
                        ":": QtCore.Qt.DotLine,
                        "-.": QtCore.Qt.DashDotLine,
                    }.get(primitive.linestyle, QtCore.Qt.SolidLine)
                )
                return pen

            painter.setRenderHint(modules.QtGui.QPainter.Antialiasing, True)
            for primitive in sorted(primitives, key=lambda item: item.zorder):
                try:
                    axes, box = plot_box(primitive.target)
                except KeyError:
                    continue
                painter.save()
                try:
                    painter.setClipRect(box)
                    if isinstance(primitive, SelectorLine):
                        painter.setPen(line_pen(primitive))
                        points = tuple(point(axes, item) for item in primitive.points)
                        if len(points) == 2:
                            painter.drawLine(points[0], points[1])
                        else:
                            painter.drawPolyline(modules.QtGui.QPolygonF(points))
                    elif isinstance(primitive, SelectorMarkers):
                        size = _style_points_to_logical_pixels(
                            primitive.size_pt,
                            logical_dpi,
                        )
                        if primitive.edgecolor is None:
                            painter.setPen(QtCore.Qt.NoPen)
                        else:
                            pen = modules.QtGui.QPen(color(primitive.edgecolor))
                            pen.setWidthF(
                                _style_points_to_logical_pixels(
                                    primitive.edge_width_pt,
                                    logical_dpi,
                                )
                            )
                            painter.setPen(pen)
                        painter.setBrush(color(primitive.facecolor))
                        for item in primitive.points:
                            center = point(axes, item)
                            bounds = QtCore.QRectF(
                                center.x() - size / 2.0,
                                center.y() - size / 2.0,
                                size,
                                size,
                            )
                            if primitive.shape == "circle":
                                painter.drawEllipse(bounds)
                            else:
                                painter.drawRect(bounds)
                    elif isinstance(primitive, SelectorText):
                        font = modules.QtGui.QFont(primitive.font_family)
                        font.setPixelSize(max(1, round(
                            _style_points_to_logical_pixels(
                                primitive.font_size_pt,
                                logical_dpi,
                            )
                        )))
                        painter.setFont(font)
                        painter.setRenderHint(
                            modules.QtGui.QPainter.TextAntialiasing,
                            True,
                        )
                        painter.setPen(color(primitive.color))
                        metrics = modules.QtGui.QFontMetricsF(font)
                        first_line = primitive.text.splitlines()[0] or " "
                        ink = metrics.tightBoundingRect(first_line)
                        top_padding = metrics.ascent() + ink.top()
                        anchor_x = box.left() + primitive.position[0] * box.width()
                        anchor_y = (
                            box.top()
                            + (1.0 - primitive.position[1]) * box.height()
                            - top_padding
                        )
                        if primitive.horizontal_alignment == "right":
                            area = QtCore.QRectF(
                                box.left(),
                                anchor_y,
                                anchor_x - box.left(),
                                box.bottom() - anchor_y,
                            )
                            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignRight
                        else:
                            area = QtCore.QRectF(
                                anchor_x,
                                anchor_y,
                                box.right() - anchor_x,
                                box.bottom() - anchor_y,
                            )
                            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
                        painter.drawText(area, alignment, primitive.text)
                finally:
                    painter.restore()

        def _normalized(self, event: object) -> tuple[float, float]:
            position = getattr(event, "position", None)
            if callable(position):
                point = position()
                x, y = float(point.x()), float(point.y())
            else:
                x, y = float(event.x()), float(event.y())
            return (
                x / max(1, self.width()),
                y / max(1, self.height()),
            )

        @staticmethod
        def _button_number(button: object) -> int | None:
            return {
                QtCore.Qt.LeftButton: 1,
                QtCore.Qt.MiddleButton: 2,
                QtCore.Qt.RightButton: 3,
            }.get(button)

        def _submit_pointer(
            self,
            action: str,
            event: object | None = None,
            *,
            button: int | None = None,
            double: bool = False,
            step: float = 0.0,
            key: str | None = None,
        ) -> None:
            if not self._interaction_gate.enabled:
                return
            gate_generation = self._interaction_gate.generation
            x, y = (0.0, 0.0) if event is None else self._normalized(event)
            source_front = (
                self._gesture_front
                if action in {"move", "release"}
                and self._gesture_front is not None
                else self._front
            )
            source_axes = (
                self._gesture_axes
                if action in {"move", "release"}
                and self._gesture_axes is not None
                else None
            )
            if source_axes is None and source_front is not None and event is not None:
                source_axes = next(
                    (
                        item
                        for item in source_front.interaction.axes
                        if item.bounds[0] <= x <= item.bounds[2]
                        and item.bounds[1] <= y <= item.bounds[3]
                    ),
                    None,
                )
            if action == "press":
                self._gesture_front = source_front
                self._gesture_axes = source_axes
            future = self._host._pointer_event(
                action,
                x,
                y,
                button=button,
                double=double,
                step=step,
                key=key,
                identity=(None if source_front is None else source_front.identity),
                axes=source_axes,
                interaction=(
                    None if source_front is None else source_front.interaction
                ),
            )

            def completed(value: Future[object]) -> None:
                try:
                    operation = value.result()
                except CancelledError:
                    return
                except BaseException as error:
                    self._gesture_ready.emit(
                        (gate_generation, action, None, None, str(error))
                    )
                else:
                    self._gesture_ready.emit(
                        (
                            gate_generation,
                            action,
                            operation.value,
                            operation.front,
                            None,
                        )
                    )

            future.add_done_callback(completed)

        @QtCore.pyqtSlot(object)
        def _finish_pointer(self, result: object) -> None:
            gate_generation, action, state, operation_front, error = result
            if gate_generation != self._interaction_gate.generation:
                return
            if error is not None:
                active = (
                    self._pointer_button is not None
                    or self._gesture_front is not None
                    or self._candidate is not None
                )
                self._clear_interaction()
                if active:
                    self._track(
                        self._host._pointer_event(
                            "cancel",
                            0.0,
                            0.0,
                        )
                    )
                self.errorOccurred.emit(str(error))
                return
            candidate = state.candidate
            role = state.role
            cell_index = state.cell_index
            active_pan = state.active_pan
            if action in {"release", "cancel", "key"}:
                self._clear_interaction()
                if state.publish_front and operation_front is not None:
                    self._install_front(operation_front)
                return
            if candidate is not None:
                self._candidate = candidate
                self._candidate_scene = state.scene
                self._gesture_kind = candidate.kind
            if state.publish_front and operation_front is not None:
                self._install_front(operation_front)
            if candidate is None or role is None:
                if action in {"release", "cancel", "key"} or not active_pan:
                    self._clear_interaction()
                return
            front = self._gesture_front or self._front
            if front is None:
                self._clear_interaction()
                return
            axes = self._gesture_axes
            if (
                axes is None
                or axes.role != role
                or axes.cell_index != cell_index
            ):
                axes = next(
                    (
                        item
                        for item in front.interaction.axes
                        if item.role == role and item.cell_index == cell_index
                    ),
                    None,
                )
            if axes is None:
                self._cancel_active_interaction()
                return
            self._gesture_front = front
            self._gesture_axes = axes
            self._gesture_kind = candidate.kind
            self.update()

        def mousePressEvent(self, event: object) -> None:
            if self._closed:
                event.ignore()
                return
            button = self._button_number(event.button())
            if button is None:
                super().mousePressEvent(event)
                return
            self._pointer_button = button
            if button in (1, 2):
                self.grabMouse()
            self._submit_pointer("press", event, button=button)
            event.accept()

        def mouseDoubleClickEvent(self, event: object) -> None:
            if self._closed:
                event.ignore()
                return
            button = self._button_number(event.button())
            if button is None:
                super().mouseDoubleClickEvent(event)
                return
            self._pointer_button = button
            if button in (1, 2):
                self.grabMouse()
            self._submit_pointer(
                "press",
                event,
                button=button,
                double=True,
            )
            event.accept()

        def mouseMoveEvent(self, event: object) -> None:
            button = self._pointer_button
            if self._closed or button is None:
                super().mouseMoveEvent(event)
                return
            self._submit_pointer(
                "move",
                event,
                button=button,
            )
            event.accept()

        def mouseReleaseEvent(self, event: object) -> None:
            button = self._button_number(event.button())
            if self._closed or button is None:
                super().mouseReleaseEvent(event)
                return
            self._submit_pointer("release", event, button=button)
            self._pointer_button = None
            try:
                if self.mouseGrabber() is self:
                    self.releaseMouse()
            except RuntimeError:
                pass
            event.accept()

        def wheelEvent(self, event: object) -> None:
            if self._closed or self._pointer_button is not None:
                event.ignore()
                return
            delta = float(event.angleDelta().y())
            if delta == 0.0:
                event.ignore()
                return
            self._submit_pointer("scroll", event, step=delta)
            event.accept()

        def keyPressEvent(self, event: object) -> None:
            if event.key() == QtCore.Qt.Key_Escape:
                self._submit_pointer("key", key="escape")
                self._clear_interaction()
                event.accept()
                return
            super().keyPressEvent(event)

        def _clear_interaction(self) -> None:
            try:
                if self.mouseGrabber() is self:
                    self.releaseMouse()
            except RuntimeError:
                pass
            self._gesture_front = None
            self._gesture_axes = None
            self._gesture_kind = None
            self._candidate = None
            self._candidate_scene = None
            self._pointer_button = None
            self.update()

        def _cancel_active_interaction(self) -> None:
            active = (
                self._pointer_button is not None
                or self._gesture_front is not None
                or self._candidate is not None
            )
            self._clear_interaction()
            if active:
                self._track(
                    self._host._pointer_event(
                        "cancel",
                        0.0,
                        0.0,
                    )
                )

    _Qt5PlotWidget.__name__ = "Qt5PlotWidget"
    _Qt5PlotWidget.__qualname__ = "Qt5PlotWidget"
    _Qt5PlotWidget.__module__ = __name__
    _QT_WIDGET_CLASS = _Qt5PlotWidget
    return _QT_WIDGET_CLASS


def Qt5PlotWidget(
    host: RasterPlotHost,
    parent: object | None = None,
    *,
    auto_present: bool = True,
) -> object:
    """Create a PyQt5 QImage frontend over a serial raster worker.

    Set ``auto_present=False`` to stage host fronts externally and install a
    coordinated cohort with the widget's owner-thread ``present_front`` method.
    """

    if not isinstance(host, RasterPlotHost):
        raise TypeError("host must be RasterPlotHost")
    if not isinstance(auto_present, bool):
        raise TypeError("auto_present must be bool")
    ensure_qt5_application()
    widget_type = _qt5_plot_widget_class()
    return widget_type(
        host,
        parent,
        auto_present=auto_present,
    )


__all__ = [
    "BackendUnavailableError",
    "NotebookView",
    "Qt5PlotWidget",
    "RasterPlotHost",
    "ensure_qt5_application",
    "notebook_available",
    "qt5_available",
]
