"""Qt-owner observation of the screen pixel ratio for raster surfaces."""

from __future__ import annotations

from collections.abc import Callable
import math

from PyQt5 import QtCore, QtWidgets


class RasterPixelRatioObserver(QtCore.QObject):
    """Publish changes in the actual screen DPR of one QWidget hierarchy.

    The native :class:`QWindow` does not reliably exist during widget
    construction, and its screen can change after the widget is shown.  This
    object is the sole Qt lifecycle owner for that observation.  Product
    windows receive only a normalized ratio and retain responsibility for
    authoring their own presentation revision and worker request.
    """

    def __init__(
        self,
        host: QtWidgets.QWidget,
        on_change: Callable[[float], None],
    ) -> None:
        if not isinstance(host, QtWidgets.QWidget):
            raise TypeError("raster pixel-ratio host must be a QWidget")
        if not callable(on_change):
            raise TypeError("raster pixel-ratio callback must be callable")
        super().__init__(host)
        self._host: QtWidgets.QWidget | None = host
        self._on_change: Callable[[float], None] | None = on_change
        self._window_handle = None
        self._observed_screen_ids: set[int] = set()
        self._observed_screens: list[object] = []
        self._last_ratio: float | None = None
        self._bind_scheduled = False
        self._detached = False
        host.installEventFilter(self)

    @property
    def current_ratio(self) -> float:
        if self._detached or self._host is None:
            raise RuntimeError("raster pixel-ratio observer is detached")
        top = self._host.window()
        source = self._host if top is None else top
        ratio = float(source.devicePixelRatioF())
        return ratio if math.isfinite(ratio) and ratio > 0.0 else 1.0

    def refresh(self, *, force: bool = False) -> None:
        """Publish the current ratio once, or again when explicitly forced."""

        if self._detached:
            return
        ratio = self.current_ratio
        if not force and ratio == self._last_ratio:
            return
        self._last_ratio = ratio
        callback = self._on_change
        if callback is not None:
            callback(ratio)

    def schedule_bind(self) -> None:
        """Bind after Qt has had an owner turn to create the native window."""

        if self._detached or self._bind_scheduled:
            return
        self._bind_scheduled = True
        QtCore.QTimer.singleShot(0, self._bind)

    def _bind(self) -> None:
        self._bind_scheduled = False
        host = self._host
        if self._detached or host is None or not host.isVisible():
            return
        top = host.window()
        handle = None if top is None else top.windowHandle()
        if handle is None:
            self.schedule_bind()
            return
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

    def _observe_screen(self, screen) -> None:
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

    def _screen_metric_changed(self, *_args) -> None:
        self.refresh()

    def _screen_changed(self, screen) -> None:
        if self._detached:
            return
        self._observe_screen(screen)
        # Qt finalizes per-monitor DPR after emitting QWindow.screenChanged.
        QtCore.QTimer.singleShot(0, self.refresh)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
        if (
            not self._detached
            and watched is self._host
            and event.type() == QtCore.QEvent.Show
        ):
            self.schedule_bind()
        return super().eventFilter(watched, event)

    def detach(self) -> None:
        """Retire every Qt callback before the host begins asynchronous close."""

        if self._detached:
            return
        host = self._host
        if host is not None and QtCore.QThread.currentThread() != host.thread():
            raise RuntimeError("raster pixel-ratio observer is GUI-thread affine")
        self._detached = True
        self._bind_scheduled = False
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


__all__ = ["RasterPixelRatioObserver"]
