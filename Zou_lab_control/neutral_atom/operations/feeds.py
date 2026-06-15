"""Experiment feeds: loops that publish per-shot signals into a SignalHub.

A feed is the producer half of the task-console contract (the console is the
consumer).  :class:`LoadingFeed` is the standard atom-loading producer: it pulls
frames through the :class:`CameraDevice` CONTRACT, self-calibrates with the SAME
``core``/``operations`` primitives the real readout uses (site centers from
all-sites frames, per-site Otsu thresholds from sample frames), then publishes
one atom-loading shot per ``step()``.

Because it only ever touches the camera CONTRACT (``camera.acquire(...)``) and the
backend-neutral ``imaging_sequence`` helper, the feed is backend-agnostic: pass
``exp.camera`` (+ ``exp.devices.sequencer``) for real hardware, or a
``VirtualCamera`` for offline development -- it is the SAME feed, only the data
source changes.  That is the "virtual == real" core principle (AGENTS.md §2): the
console/feed validated on virtual data runs verbatim on the real machine.  For
the offline convenience wrapper that builds a ``VirtualCamera`` see
``neutral_atom.devices.virtual.virtual_loading_feed``.

Feeds run either synchronously (call ``step()`` yourself: deterministic tests) or
in a background thread (``start(rate_hz)``); the hub is the only shared state.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..core.analysis import estimate_thresholds, find_site_centers, grid_shape_tuple, positive_int, roi_counts
from ..core.signals import SignalHub
from ..devices.base import CameraDevice
from ..timing import imaging_sequence


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

    def published_signals(self) -> frozenset:
        """The signal names this feed publishes (behind ``prefix``).  Lets a
        consumer (e.g. the task console) map a panel back to the feed that
        produces its data, then expose THAT feed's parameters.  Subclasses with
        a fixed signal set override this; the base publishes nothing structured."""
        return frozenset()


class LoadingFeed(ExperimentFeed):
    """Atom-loading experiment producer over the :class:`CameraDevice` contract.

    Backend-agnostic: identical with a real camera or a ``VirtualCamera`` -- only
    the ``camera`` you pass differs.  It self-calibrates exactly as the real
    readout does (site centers detected from an all-sites template via
    ``find_site_centers``; per-site Otsu thresholds learned from sample frames via
    ``estimate_thresholds``), then images one loading shot per ``shot()``.

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
        camera: CameraDevice,
        *,
        sequencer: object | None = None,
        prefix: str = "",
        grid_shape: tuple[int, int] = (5, 7),
        exposure: float = 0.02,
        roi_radius: int = 1,
        ema: float = 0.05,
        calibration_frames: int = 4,
        threshold_frames: int = 24,
    ):
        super().__init__(hub, prefix=prefix)
        self.camera = camera
        self.sequencer = sequencer
        self.grid_shape = grid_shape_tuple(grid_shape)
        self.exposure = float(exposure)
        self.roi_radius = int(roi_radius)
        self.ema = float(ema)
        # Kept as attributes (not just __init__ locals) so the task-console Edit
        # tab can auto-discover them via inspect.signature + getattr and rebuild
        # the feed with edited acquisition parameters.
        self.calibration_frames = int(calibration_frames)
        self.threshold_frames = int(threshold_frames)
        # Two backend-neutral sequences, built ONCE: an all-sites template (the
        # virtual camera renders every trap occupied for ``name="sitemap"``; real
        # hardware runs its deterministic-fill template) and a per-shot readout
        # with fresh loading (``load=True`` -> the camera reloads each frame).
        self._sitemap_seq = imaging_sequence(exposure=self.exposure, load=True, name="sitemap")
        self._readout_seq = imaging_sequence(exposure=self.exposure, load=True, name="readout")
        # --- self-calibration through the contract, SAME primitives as the real readout ---
        template = self._acquire(max(1, self.calibration_frames), self._sitemap_seq)
        average = np.mean([np.asarray(img, dtype=float) for img in template], axis=0)
        self.centers = find_site_centers(average, self.grid_shape)
        self.n_sites = len(self.centers)
        samples = [np.asarray(img, dtype=float) for img in self._acquire(max(2, self.threshold_frames), self._readout_seq)]
        self.thresholds = np.asarray(estimate_thresholds(samples, self.centers, radius=self.roi_radius), dtype=float)
        self._rate_sites = np.full(self.n_sites, np.nan)
        self._rate = float("nan")

    def _acquire(self, frames: int, sequence) -> list[np.ndarray]:
        return self.camera.acquire(positive_int(frames, "frames"), sequence=sequence, sequencer=self.sequencer)

    def shot(self) -> dict[str, object]:
        frame = self._acquire(1, self._readout_seq)[-1]
        counts = roi_counts(np.asarray(frame, dtype=float), self.centers, radius=self.roi_radius)
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
            "rate_grid": self._rate_sites.reshape(self.grid_shape).copy(),
            "centers": np.asarray(self.centers, dtype=float).copy(),
            "thresholds": np.asarray(self.thresholds, dtype=float).reshape(-1).copy(),
            "shot": float(self.shots + 1),
        }

    def published_signals(self) -> frozenset:
        keys = ("frame", "counts", "occupied", "rate", "rate_sites", "rate_grid",
                "centers", "thresholds", "shot")
        return frozenset(self.prefix + key for key in keys)


class ScannedMeasurementFeed(ExperimentFeed):
    """Drive a :class:`ScannedMeasurement` one scan point per ``shot()``, into a hub.

    Wraps a swept measurement as a console feed (``start``/``stop``/``running``)
    so a finite scan grows a live curve in the task console exactly as the
    free-running loading feed grows a rate trace.  Each ``shot()`` advances ONE
    scan point through the measurement's contract path
    (``measurement.measure(value, index)`` -> ``camera.acquire`` + ``calibration``)
    and publishes the CUMULATIVE curve so far, so a Monitor 1-D panel
    (``value = <y_key>`` vs ``x_key``) fills in as the scan runs.

    It touches ONLY ``measurement.measure`` (the camera/sequencer/calibration
    contract) and ``hub.publish`` -- it imports no concrete backend and reads no
    simulation ground truth, so it is guarded by
    ``tests/test_virtual_equals_real_contract.py`` like the rest of the analysis
    layer.

    Published per shot (behind ``prefix``):

    ``<x_key>``         (k,) cumulative scan x values (the points done so far)
    ``<y_key>``         (k,) cumulative scalar curve (series 0 of the reducer)
    ``<y_key>_sites``   (n_series,) the LATEST point's per-site vector (per-site only)
    ``<y_key>_grid``    grid-shaped latest per-site vector (per-site only, if grid_shape)
    ``scan_done``       0 while running, 1 once the final point has been published
    ``shot``            scalar shot counter

    Finite-scan semantics: after the last point is published the feed sets its
    own stop event, so a background ``start()`` thread exits on its own once the
    sweep completes; ``finished`` reports completion and ``run_to_completion()``
    runs every remaining point synchronously (tests / headless).
    """

    def __init__(
        self,
        hub: SignalHub,
        measurement,
        *,
        x_key: str = "x",
        y_key: str = "y",
        grid_shape: tuple[int, int] | None = None,
        prefix: str = "",
    ):
        super().__init__(hub, prefix=prefix)
        self.measurement = measurement
        self.x_key = str(x_key)
        self.y_key = str(y_key)
        self.grid_shape = None if grid_shape is None else grid_shape_tuple(grid_shape)
        # The measurement owns the swept values (single source of truth); the feed
        # only walks its index and accumulates the (x, y) seen so far.
        self._values = np.asarray(measurement.axis.values, dtype=float).reshape(-1)
        self._index = 0
        self._x_done: list[float] = []
        self._y_done: list[float] = []

    @property
    def n_points(self) -> int:
        return int(self._values.size)

    @property
    def points_done(self) -> int:
        return int(self._index)

    @property
    def finished(self) -> bool:
        """True once every scan point has been measured and published."""

        return self._index >= self.n_points

    def shot(self) -> dict[str, object]:
        """Advance ONE scan point and return the cumulative curve so far.

        Raises ``StopIteration`` if called after the sweep is finished -- the
        ``start()`` loop never does (the feed stops itself), and tests check
        ``finished`` first.
        """

        if self.finished:
            raise StopIteration("ScannedMeasurementFeed: scan already complete.")
        index = self._index
        value = float(self._values[index])
        row = np.atleast_1d(np.asarray(self.measurement.measure(value, index), dtype=float))
        self._index += 1
        self._x_done.append(value)
        self._y_done.append(float(row[0]))

        out: dict[str, object] = {
            self.x_key: np.asarray(self._x_done, dtype=float).copy(),
            self.y_key: np.asarray(self._y_done, dtype=float).copy(),
            "scan_done": 1.0 if self.finished else 0.0,
            "shot": float(self._index),
        }
        if row.size > 1:
            # A per-site reducer: publish the latest point's per-site vector (and a
            # grid map when a shape is known) alongside the scalar curve.
            out[self.y_key + "_sites"] = row.copy()
            if self.grid_shape is not None and row.size == int(np.prod(self.grid_shape)):
                out[self.y_key + "_grid"] = row.reshape(self.grid_shape).copy()
        return out

    def step(self) -> dict[str, object]:
        """Run one point, publish it, and self-stop once the final point lands.

        Reuses the base publish path; the only addition is the finite-scan stop:
        after the last point's signals are on the hub, the stop event is set so a
        background ``start()`` thread's loop exits cleanly.
        """

        named = super().step()
        if self.finished:
            self._stop.set()
        return named

    def run_to_completion(self) -> "ScannedMeasurementFeed":
        """Synchronously run + publish every remaining scan point (test/headless)."""

        while not self.finished:
            self.step()
        return self

    def published_signals(self) -> frozenset:
        keys = (self.x_key, self.y_key, self.y_key + "_sites", self.y_key + "_grid",
                "scan_done", "shot")
        return frozenset(self.prefix + key for key in keys)


__all__ = ["ExperimentFeed", "LoadingFeed", "ScannedMeasurementFeed"]
