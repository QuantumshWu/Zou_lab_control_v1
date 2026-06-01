"""Acquisition session that keeps data collection away from the UI thread."""

from __future__ import annotations

import inspect
import itertools
import threading
from collections.abc import Iterable
from typing import Any, Callable, Sequence

import numpy as np

from .live import _as_data_x, _as_data_y, plot


def _is_integer_scalar(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _strict_bool(value, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean.")


def _positive_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite, not a boolean.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and > 0.")
    return result


def _positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer, not a boolean.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _call_accepts(func: Callable, args: tuple[Any, ...]) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(*args)
    except TypeError:
        return False
    return True


def _source_callable(source) -> Callable | None:
    if callable(source):
        return source
    for name in ("measure", "read", "acquire", "next"):
        func = getattr(source, name, None)
        if callable(func):
            return func
    return None


def _put_row(data_y: np.ndarray, index: int, value) -> None:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return
    stop = min(arr.size, data_y.shape[1])
    data_y[index, :stop] = arr[:stop]


class RunSession:
    """Own one acquisition worker, one shared array, and one front-end plot."""

    def __init__(
        self,
        data_x,
        source,
        *,
        data_y=None,
        kind: str | None = "auto",
        mode: str | None = None,
        labels: Sequence[str] | None = None,
        update_time: float = 0.05,
        max_points: int | None = None,
        autostart: bool = True,
        display: bool = True,
        stop_when_full: bool | None = None,
        copy_on_refresh: bool = True,
        plot_options: dict[str, Any] | None = None,
        stop_hint: bool | str = True,
        **plot_kwargs,
    ):
        self.source = source
        self.kind = "hist" if kind is None and _is_integer_scalar(data_x) else (kind or "auto")
        self.mode = mode or ("roll" if self.kind in {"monitor", "rolling", "live-distribution"} else "append")
        self.labels = labels
        self.update_time = _positive_float(update_time, "update_time")
        self.max_points = None if max_points is None else _positive_int(max_points, "max_points")
        self.autostart = _strict_bool(autostart, "autostart")
        self.display = _strict_bool(display, "display")
        self.copy_on_refresh = _strict_bool(copy_on_refresh, "copy_on_refresh")
        if isinstance(stop_hint, (bool, np.bool_, str)):
            self.stop_hint = stop_hint
        else:
            raise TypeError("stop_hint must be a boolean or string.")
        self.plot_kwargs = {**dict(plot_options or {}), **dict(plot_kwargs)}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.running = False
        self.points_done = 0

        self._prepare_arrays(data_x, data_y)
        if stop_when_full is None:
            stop_when_full = self.mode != "roll"
        self.stop_when_full = _strict_bool(stop_when_full, "stop_when_full")

        self.plot = plot(
            self._plot_x,
            self._plot_y,
            kind=self.kind,
            update="watch",
            labels=self.labels,
            update_time=self.update_time,
            watch_interval=self.update_time,
            stop_when_full=self.stop_when_full,
            done=self._done_event.is_set,
            points_done=lambda: self.points_done,
            display=self.display,
            copy=self.copy_on_refresh,
            lock=self._lock,
            **self.plot_kwargs,
        )
        if self.kind == "hist":
            self.plot.values = self.data_y[:, 0]
            self.plot.data_y = self.data_y
            if getattr(self.plot, "_watcher", None) is not None:
                self.plot._watcher.data_y = self.data_y

        if self.autostart:
            self.start()
            self._print_stop_hint()

    def _prepare_arrays(self, data_x, data_y) -> None:
        normalized_kind = str(self.kind).lower().replace("_", "-")
        if normalized_kind in {"hist", "histogram", "distribution"}:
            if _is_integer_scalar(data_x) or isinstance(data_x, (bool, np.bool_)):
                count = _positive_int(data_x, "histogram count")
                self.data_y = np.full((count, 1), np.nan, dtype=float)
            else:
                values = np.asarray(data_x if data_y is None else data_y, dtype=float).reshape(-1)
                self.data_y = values.reshape(-1, 1)
            self.data_x = _as_data_x(np.arange(len(self.data_y)))
            self._plot_x = self.data_y[:, 0]
            self._plot_y = None
            self.kind = "hist"
            return

        self.data_x = _as_data_x(data_x)
        self.data_y = _as_data_y(data_y, len(self.data_x))
        self._plot_x = self.data_x
        self._plot_y = self.data_y

    def __getattr__(self, name: str):
        return getattr(self.plot, name)

    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def refresh(self, *, draw: bool = True):
        """Refresh the attached frontend plot from the shared data array."""

        watcher = getattr(self.plot, "_watcher", None)
        if watcher is not None:
            watcher.refresh(draw=draw)
        else:
            self.plot.refresh(draw=draw)
        return self

    def start(self):
        """Start acquisition in the session-owned worker."""
        if self.running:
            return self
        self._stop_event.clear()
        self._done_event.clear()
        self._error = None
        self.running = True
        self._thread = threading.Thread(target=self._worker, name="ZLCFrontEndRun", daemon=True)
        self._thread.start()
        return self

    def _print_stop_hint(self) -> None:
        if not self.display or not self.stop_hint:
            return
        if isinstance(self.stop_hint, str):
            message = self.stop_hint
        else:
            message = "Live measurement started. Call the returned object's .stop() to stop measurement and plot."
        print(message)

    def stop(self):
        """Request acquisition stop and stop front-end refresh."""
        self._stop_event.set()
        stop = getattr(self.source, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass
        self._done_event.set()
        try:
            self.refresh(draw=True)
        except Exception:
            pass
        self.plot.stop()
        self.running = False
        return self

    def _worker(self) -> None:
        start = getattr(self.source, "start", None)
        if callable(start):
            start()
        try:
            if self.mode == "roll":
                self._run_roll()
            else:
                self._run_append()
        except BaseException as exc:
            self._error = exc
        finally:
            stop = getattr(self.source, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
            self.running = False
            self._done_event.set()

    def _run_append(self) -> None:
        limit = len(self.data_y) if self.max_points is None else min(self.max_points, len(self.data_y))
        for index in range(limit):
            if self._stop_event.is_set():
                break
            value = self._read_value(index)
            with self._lock:
                _put_row(self.data_y, index, value)
                self.points_done = index + 1

    def _run_roll(self) -> None:
        counter = itertools.count() if self.max_points is None else range(self.max_points)
        for index in counter:
            if self._stop_event.is_set():
                break
            value = self._read_value(index)
            with self._lock:
                self.data_y[:] = np.roll(self.data_y, shift=1, axis=0)
                _put_row(self.data_y, 0, value)
                self.points_done = min(len(self.data_y), self.points_done + 1)

    def _read_value(self, index: int):
        if isinstance(self.source, Iterable) and not callable(self.source):
            if not hasattr(self, "_source_iter"):
                self._source_iter = iter(self.source)
            return next(self._source_iter)

        func = _source_callable(self.source)
        if func is None:
            raise TypeError("source must be callable, iterable, or expose read/measure/acquire/next.")

        point = None
        if len(self.data_x):
            point_arr = self.data_x[min(index, len(self.data_x) - 1)]
            point = float(point_arr[0]) if point_arr.size == 1 else point_arr

        if isinstance(point, np.ndarray):
            expanded = tuple(float(v) for v in point)
            candidates = (
                (point, index),
                (*expanded, index),
                expanded,
                (point,),
                (index,),
                (),
            )
        else:
            candidates = (
                (point, index),
                (point,),
                (index,),
                (),
            )

        for args in candidates:
            if _call_accepts(func, args):
                return func(*args)
        raise TypeError("source signature is not compatible with point/index acquisition.")


def run(data_x, source, **kwargs) -> RunSession:
    """Create and start a plot-backed acquisition session."""
    return RunSession(data_x, source, **kwargs)


__all__ = ["RunSession", "run"]
