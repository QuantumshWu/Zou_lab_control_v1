"""Internal UI-timer watcher for shared arrays."""

from __future__ import annotations

from typing import Callable

import numpy as np


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


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-negative integer, not a boolean.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


class ArrayWatcher:
    """Refresh a plot from a shared array on the active Matplotlib backend."""

    def __init__(
        self,
        live_plot,
        data_y=None,
        *,
        interval: float | None = None,
        points_done: Callable[[], int] | int | None = None,
        done: Callable[[], bool] | None = None,
        stop_when_full: bool = True,
        finalize: bool = True,
        auto_show: bool = True,
        copy: bool = False,
        lock=None,
    ):
        self.live_plot = live_plot
        self.data_y = live_plot.data_y if data_y is None else data_y
        self.interval = _positive_float(interval if interval is not None else getattr(live_plot, "update_time", 0.1), "watch_interval")
        self.points_done_source = points_done
        self.done_source = done
        self.stop_when_full = _strict_bool(stop_when_full, "stop_when_full")
        self.finalize = _strict_bool(finalize, "finalize")
        self.auto_show = _strict_bool(auto_show, "auto_show")
        self.copy = _strict_bool(copy, "copy")
        self.lock = lock
        self.timer = None
        self.running = False
        self.live_plot._watcher = self

    def _points_done(self, data=None) -> int:
        if callable(self.points_done_source):
            return _non_negative_int(self.points_done_source(), "points_done")
        if self.points_done_source is not None:
            return _non_negative_int(self.points_done_source, "points_done")
        arr = np.asarray(self.data_y if data is None else data)
        if arr.ndim == 1:
            return int(np.count_nonzero(np.isfinite(arr)))
        return int(np.count_nonzero(np.isfinite(arr[:, 0])))

    def _is_done(self, points_done: int) -> bool:
        if self.done_source is not None:
            if _strict_bool(self.done_source(), "done callback return"):
                return True
        return self.stop_when_full and points_done >= len(self.live_plot.data_x)

    def _snapshot(self):
        if self.lock is None:
            return np.array(self.data_y, copy=True) if self.copy else self.data_y
        with self.lock:
            return np.array(self.data_y, copy=True) if self.copy else self.data_y

    def _snapshot_and_points(self):
        if self.lock is None:
            data = np.array(self.data_y, copy=True) if self.copy else self.data_y
            return data, self._points_done(data)
        with self.lock:
            data = np.array(self.data_y, copy=True) if self.copy else self.data_y
            return data, self._points_done(data)

    def refresh(self, *, draw: bool = True):
        data, points_done = self._snapshot_and_points()
        done_now = self._is_done(points_done)
        if done_now and self.copy:
            data, points_done = self._snapshot_and_points()
        self.live_plot.update(data_y=data, points_done=points_done, draw=draw)
        if done_now or self._is_done(points_done):
            self.stop()
        return self.live_plot

    def start(self):
        if self.running:
            return self.live_plot
        if self.auto_show and not getattr(self.live_plot, "_shown", False):
            self.live_plot.show()
        interval_ms = max(1, int(self.interval * 1000))
        self.timer = self.live_plot.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.refresh)
        self.running = True
        self.timer.start()
        return self.live_plot

    def stop(self):
        if self.timer is not None:
            try:
                self.timer.stop()
            except Exception:
                pass
        self.running = False
        if self.finalize and getattr(self.live_plot, "data_figure", None) is None:
            self.live_plot.after_plot()
        return self.live_plot


__all__ = ["ArrayWatcher"]
