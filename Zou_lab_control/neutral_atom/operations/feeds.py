"""Experiment feeds: loops that publish per-shot signals into a SignalHub.

A feed is the producer half of the task-console contract (the console is the
consumer).  ``VirtualLoadingFeed`` is the self-contained virtual source used to
develop and test console layouts without hardware: it owns a ``VirtualTrapArray``,
self-calibrates (site centers from all-sites frames, per-site Otsu thresholds from
sample frames -- the same core/analysis primitives the real pipeline uses), then
publishes one atom-loading shot per ``step()``.

Feeds run either synchronously (call ``step()`` yourself: deterministic tests) or
in a background thread (``start(rate_hz)``); the hub is the only shared state.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..core.analysis import estimate_thresholds, find_site_centers, roi_counts
from ..core.signals import SignalHub
from ..devices.virtual import VirtualTrapArray


class ExperimentFeed:
    """Base feed: owns the publish loop; subclasses implement one ``shot()``."""

    def __init__(self, hub: SignalHub, *, prefix: str = ""):
        self.hub = hub
        self.prefix = str(prefix)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.shots = 0

    # ------------------------------------------------------------------ loop
    def shot(self) -> dict[str, object]:  # pragma: no cover - abstract
        """Produce one shot's worth of {name: value} signals."""
        raise NotImplementedError

    def step(self) -> dict[str, object]:
        """Run ONE shot synchronously and publish it (test/notebook friendly)."""
        values = self.shot()
        named = {self.prefix + key: value for key, value in values.items()}
        self.hub.publish(named)
        self.shots += 1
        return named

    def start(self, *, rate_hz: float = 5.0) -> "ExperimentFeed":
        """Publish shots from a daemon thread at ``rate_hz`` until ``stop()``."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self.rate_hz = float(rate_hz)   # remembered so a paused feed can resume at the same rate
        self._stop.clear()
        period = 1.0 / max(0.1, float(rate_hz))

        def _loop() -> None:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    self.step()
                except Exception:
                    # A wedged source must not kill the daemon silently mid-run;
                    # back off and keep trying (the console shows stale data).
                    time.sleep(period)
                    continue
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)

        self._thread = threading.Thread(target=_loop, name=f"zlc-feed-{self.prefix or 'main'}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class VirtualLoadingFeed(ExperimentFeed):
    """Virtual atom-loading experiment publishing the standard console signals.

    Published per shot (all behind ``prefix``):

    ``frame``       HxW camera image (latest shot)
    ``counts``      (N_sites,) per-site ROI counts
    ``occupied``    (N_sites,) 0/1 occupancy from per-site thresholds
    ``rate``        scalar running loading rate (EMA over shots)
    ``rate_sites``  (N_sites,) per-site running loading rate (EMA)
    ``rate_grid``   grid-shaped per-site running loading rate (2D map)
    ``centers``     (N_sites, 2) calibrated site centers in camera px (x, y)
    ``thresholds``  (N_sites,) per-site Otsu thresholds (counts)
    ``shot``        scalar shot counter
    """

    def __init__(
        self,
        hub: SignalHub,
        *,
        prefix: str = "",
        grid_shape: tuple[int, int] = (5, 7),
        loading_probability: float = 0.55,
        exposure: float = 0.02,
        roi_radius: int = 1,
        ema: float = 0.05,
        seed: int | None = None,
        calibration_frames: int = 4,
        threshold_frames: int = 24,
        trap_array: VirtualTrapArray | None = None,
    ):
        super().__init__(hub, prefix=prefix)
        self.exposure = float(exposure)
        self.roi_radius = int(roi_radius)
        self.ema = float(ema)
        self.trap = trap_array if trap_array is not None else VirtualTrapArray(
            grid_shape=tuple(grid_shape),
            loading_probability=float(loading_probability),
            seed=seed,
        )
        # --- self-calibration with the SAME primitives as the real pipeline ---
        # site centers: average a few all-sites frames (every trap rendered occupied)
        self.trap.reload()
        all_sites = np.mean(
            [self.trap.render_image(exposure=self.exposure, all_sites=True).astype(float)
             for _ in range(max(1, int(calibration_frames)))],
            axis=0,
        )
        self.centers = find_site_centers(all_sites, self.trap.grid_shape)
        # per-site thresholds: Otsu over sample frames with fresh random loading
        samples = []
        for _ in range(max(2, int(threshold_frames))):
            self.trap.reload()
            samples.append(self.trap.render_image(exposure=self.exposure).astype(float))
        self.thresholds = estimate_thresholds(samples, self.centers, radius=self.roi_radius)
        self._rate_sites = np.full(self.trap.n_sites, np.nan)
        self._rate = float("nan")

    def shot(self) -> dict[str, object]:
        self.trap.reload()
        frame = self.trap.render_image(exposure=self.exposure)
        counts = roi_counts(frame.astype(float), self.centers, radius=self.roi_radius)
        occupied = (counts > self.thresholds).astype(float)
        # EMA running rates (first shot seeds the average)
        if np.isnan(self._rate):
            self._rate_sites = occupied.copy()
            self._rate = float(occupied.mean())
        else:
            self._rate_sites = (1.0 - self.ema) * self._rate_sites + self.ema * occupied
            self._rate = (1.0 - self.ema) * self._rate + self.ema * float(occupied.mean())
        return {
            "frame": frame,
            "counts": counts,
            "occupied": occupied,
            "rate": self._rate,
            "rate_sites": self._rate_sites.copy(),
            "rate_grid": self._rate_sites.reshape(self.trap.grid_shape).copy(),
            "centers": np.asarray(self.centers, dtype=float).copy(),
            "thresholds": np.asarray(self.thresholds, dtype=float).reshape(-1).copy(),
            "shot": float(self.shots + 1),
        }


__all__ = ["ExperimentFeed", "VirtualLoadingFeed"]
