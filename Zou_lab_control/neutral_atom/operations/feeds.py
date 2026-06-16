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
from ..timing import imaging_channel_kwargs, imaging_sequence


class ExperimentFeed:
    """Base feed: owns the publish loop; subclasses implement one ``shot()``."""

    def __init__(self, hub: SignalHub, *, prefix: str = ""):
        self.hub = hub
        self.prefix = str(prefix)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.shots = 0
        # Health: a wedged source must not fail silently.  The loop records the
        # last exception + a consecutive-failure count and publishes them as
        # signals so the console can raise a visible banner (M7).
        self.last_error: str | None = None
        self.consecutive_errors = 0
        # OWNER-THREAD parameter requests: while the acquisition loop runs, it is
        # the SOLE owner of the source (the only thread that calls acquire/
        # configure).  An edited parameter from another thread (the GUI) is QUEUED
        # here and applied by the loop BETWEEN shots -- never a stop/start of the
        # thread from outside, which would block the GUI and could run two
        # acquire() calls on one camera at once (a deadlock/freeze).
        self._pending_params: dict[str, object] | None = None
        self._params_lock = threading.Lock()
        # Increments each time a QUEUED edit has been applied AND its fresh frame
        # published (see acquisition_epoch): the GUI polls this to re-snapshot the
        # Edit panel on the FIRST post-edit frame instead of the stale pre-edit one.
        self._apply_epoch = 0

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
                    applied = self._apply_pending_params()   # owner thread applies edits BETWEEN shots
                    self.step()
                    if applied:
                        # the just-published frame is the FIRST computed with the edited
                        # params -- mark the epoch so a waiting GUI re-snapshots THIS frame
                        self._apply_epoch += 1
                except Exception as exc:
                    if self._stop.is_set():
                        return  # asked to stop mid-shot -- a clean exit, not a fault
                    # A wedged source must not kill the daemon silently mid-run, but
                    # it must NOT fail silently either: record the error + a running
                    # count and publish the count so the console raises a banner
                    # (the operator otherwise just sees frozen, stale data).
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.consecutive_errors += 1
                    self.hub.publish({self.prefix + "feed_error": float(self.consecutive_errors)})
                    self._stop.wait(period)
                    continue
                if self.consecutive_errors:
                    # Recovered: clear the banner so a transient hiccup doesn't stick.
                    self.last_error = None
                    self.consecutive_errors = 0
                    self.hub.publish({self.prefix + "feed_error": 0.0})
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

    # --------------------------------------------------- acquisition parameters
    # A panel is a VIEW; the feed is the producer; behind the feed sits the data
    # SOURCE (a camera, or the feed's own analysis).  A panel's Edit tab edits the
    # source through these two methods, so e.g. a raw-frame panel can tune the
    # camera's exposure/ROI.  The source -- not __init__ reflection -- decides
    # what is editable.
    def acquisition_parameters(self) -> dict[str, object]:
        """Editable parameters of the data SOURCE behind this feed, as
        ``{name: current_value}`` (e.g. a camera's ``exposure``/``roi``, or this
        feed's analysis settings).  Default: nothing editable."""
        return {}

    def set_acquisition_parameters(self, **values) -> None:
        """Apply edited acquisition parameters to the source NOW, on the CALLING
        thread (re-configure the camera, or re-calibrate).  This must only be
        called by the OWNER of the source -- the acquisition loop (via
        ``_apply_pending_params``) when running, or the caller of
        ``apply_acquisition_parameters`` when idle -- never concurrently with a
        ``shot()``.  Default: no editable parameters, nothing to do."""

    def apply_acquisition_parameters(self, **values) -> None:
        """The SAFE entry point for an edit coming from another thread (the GUI).

        While the acquisition loop is running it OWNS the source, so the edit is
        QUEUED and the loop applies it between shots (``_apply_pending_params``):
        the source re-arms in the owning thread with no second ``acquire()`` ever
        running on it, and the GUI never blocks on a join.  The live Monitor keeps
        streaming and the next published frame reflects the change.  When the feed
        is idle, the edit is applied immediately and one fresh shot is published so
        a viewer reflects it at once."""
        if self.running:
            with self._params_lock:
                self._pending_params = {**(self._pending_params or {}), **values}
        else:
            self.set_acquisition_parameters(**values)
            self.step()
            self._apply_epoch += 1   # idle: applied + published synchronously, right here

    def acquisition_epoch(self) -> int:
        """A counter incremented each time a QUEUED acquisition-parameter edit has
        been applied AND its resulting frame published (or, when idle, applied +
        published synchronously by ``apply_acquisition_parameters``).

        A GUI that queued an edit can poll this to learn WHEN the first frame
        computed with the new params is on the hub, and re-snapshot its Edit panel
        then -- instead of reading the stale pre-edit frame the instant it queues
        (the queue-then-apply-between-shots latency)."""
        return self._apply_epoch

    def _apply_pending_params(self) -> bool:
        """Drain and apply queued acquisition-parameter edits.  Called by the
        acquisition loop BETWEEN shots, so the source is reconfigured in its sole
        owner thread (a streaming camera re-arms cleanly, no concurrent acquire).
        Returns whether anything was applied (so the loop can bump the epoch)."""
        with self._params_lock:
            pending = self._pending_params
            self._pending_params = None
        if pending:
            self.set_acquisition_parameters(**pending)
            return True
        return False

    # ----------------------------------------------- plot region -> source params
    def region_to_acquisition_parameters(self, x_min, x_max, y_min, y_max) -> dict[str, object]:
        """Convert a region the user marked on the plot into THIS source's
        acquisition parameters.

        The plot's selector / zoom is a GENERIC interface: it always yields a
        rectangle as four endpoints ``(x_min, x_max, y_min, y_max)`` in the panel's
        axis units (the same coordinates the axes show) -- it knows nothing about
        cameras, ROIs or scan grids, and it serves every 2-D panel (a camera frame,
        a 2-D parameter scan, ...).  Each source OWNS the conversion of that
        rectangle into its own parameter format here, so the frontend never bakes
        in a device-specific shape.  Default: a source with no spatial region
        returns ``{}`` (the selection is a no-op for it)."""
        return {}


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
        self._calibrate()

    def _calibrate(self) -> None:
        """(Re)build the imaging sequences and self-calibrate from fresh frames:
        site centers from an all-sites template, per-site Otsu thresholds from
        sample frames -- the SAME primitives the real readout uses.  Run at
        construction and whenever the acquisition parameters change."""
        # Two backend-neutral sequences: an all-sites template (the virtual camera
        # renders every trap occupied for ``name="sitemap"``; real hardware runs
        # its deterministic-fill template) and a per-shot readout with fresh
        # loading (``load=True`` -> the camera reloads each frame).
        # Target whatever channels the bound sequencer actually exposes (real
        # configs name them ch00..chNN): without this, the imaging sequence would
        # reference the trap/cooling/probe/emCCD placeholders and every pulse on a
        # real streamer would hit a non-existent channel.  Same single-source
        # mapping the session uses; {} on a virtual/notebook sequencer -> defaults.
        channel_kwargs = imaging_channel_kwargs(self.sequencer)
        self._sitemap_seq = imaging_sequence(exposure=self.exposure, load=True, name="sitemap", **channel_kwargs)
        self._readout_seq = imaging_sequence(exposure=self.exposure, load=True, name="readout", **channel_kwargs)
        template = self._acquire(max(1, self.calibration_frames), self._sitemap_seq)
        average = np.mean([np.asarray(img, dtype=float) for img in template], axis=0)
        self.centers = find_site_centers(average, self.grid_shape)
        self.n_sites = len(self.centers)
        samples = [np.asarray(img, dtype=float) for img in self._acquire(max(2, self.threshold_frames), self._readout_seq)]
        self.thresholds = np.asarray(estimate_thresholds(samples, self.centers, radius=self.roi_radius), dtype=float)
        self._rate_sites = np.full(self.n_sites, np.nan)
        self._rate = float("nan")

    def acquisition_parameters(self) -> dict[str, object]:
        """The atom-loading analysis settings this feed applies to camera frames."""
        return {
            "grid_shape": self.grid_shape,
            "exposure": self.exposure,
            "roi_radius": self.roi_radius,
            "ema": self.ema,
            "calibration_frames": self.calibration_frames,
            "threshold_frames": self.threshold_frames,
        }

    def set_acquisition_parameters(self, **values) -> None:
        """Update the analysis settings and re-calibrate in place (the running
        feed keeps publishing under the same signal names)."""
        if "grid_shape" in values:
            self.grid_shape = grid_shape_tuple(values["grid_shape"])
        if "exposure" in values:
            self.exposure = float(values["exposure"])
        if "roi_radius" in values:
            self.roi_radius = int(values["roi_radius"])
        if "ema" in values:
            self.ema = float(values["ema"])
        if "calibration_frames" in values:
            self.calibration_frames = int(values["calibration_frames"])
        if "threshold_frames" in values:
            self.threshold_frames = int(values["threshold_frames"])
        self._calibrate()

    def _acquire(self, frames: int, sequence) -> list[np.ndarray]:
        # Pass the feed's stop event so a Stop interrupts a wedged trigger wait
        # promptly (a camera that cannot interrupt simply ignores it).
        return self.camera.acquire(positive_int(frames, "frames"), sequence=sequence,
                                   sequencer=self.sequencer, stop=self._stop)

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


class CameraFrameFeed(ExperimentFeed):
    """Stream RAW camera frames into the hub, no site analysis.  The data SOURCE is
    the camera itself, so the editable acquisition parameters ARE the camera's own
    settings -- ``exposure`` and ``roi`` -- applied live (``camera.configure``).
    Backend-agnostic: identical with a real camera or a ``VirtualCamera``.

    MULTI-TRIGGER per cycle (``frames_per_cycle``).  One ``shot()`` reads
    ``frames_per_cycle`` frames -- ONE per camera (emCCD) trigger in the running
    pulse -- and publishes each as its own signal ``frame_0``, ``frame_1``, ... (the
    first is also published as ``frame`` for back-compat / the default 2D panel).  A
    pulse that triggers the camera TWICE (e.g. a release-recapture / two-readout "T"
    sequence) must set ``frames_per_cycle=2``; otherwise ``acquire(1)`` reads only
    the FIRST trigger's frame each cycle and the second is dropped -- which is why a
    single-frame feed always shows the first emCCD image.  Put ``value = frame_0`` on
    one panel and ``value = frame_1`` on another to watch the two triggers side by
    side.  (``frames_per_cycle`` must match the camera-trigger count per cycle so the
    per-trigger assignment stays phase-aligned; for a feed that FIRES the sequence,
    ``acquire`` enforces frames == trigger count.)"""

    def __init__(self, hub: SignalHub, camera: CameraDevice, *, sequencer: object | None = None,
                 frames_per_cycle: int = 1, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.camera = camera
        self.sequencer = sequencer
        self.frames_per_cycle = max(1, int(frames_per_cycle))

    def shot(self) -> dict[str, object]:
        n = max(1, int(self.frames_per_cycle))
        frames = self.camera.acquire(n, sequencer=self.sequencer, stop=self._stop)
        out: dict[str, object] = {f"frame_{i}": np.asarray(f, dtype=float) for i, f in enumerate(frames)}
        out["frame"] = out["frame_0"]   # back-compat + default 2D panel: first trigger
        return out

    def published_signals(self) -> frozenset:
        n = max(1, int(self.frames_per_cycle))
        keys = ["frame"] + [f"frame_{i}" for i in range(n)]
        return frozenset(self.prefix + key for key in keys)

    def acquisition_parameters(self) -> dict[str, object]:
        """The acquisition layer speaks PLOT coordinates: the spatial parameter is
        ``region`` = the rectangle's ENDPOINTS ``[x_min, x_max, y_min, y_max]`` in
        sensor pixels -- the SAME shape the plot's selector/zoom yields -- NOT the
        camera's device-format ``[x, w, y, h]`` sub-array (that conversion is hidden
        in ``set_acquisition_parameters``).  ``region`` reflects the camera's
        ACTUALLY-applied window (``camera.roi`` read-back), expressed as endpoints.
        ``frames_per_cycle`` = camera triggers (frames) to read per cycle."""
        params: dict[str, object] = {
            "exposure": float(self.camera.exposure),
            "frames_per_cycle": int(self.frames_per_cycle),
        }
        roi = self.camera.roi   # device window (x, w, y, h) or None; the snapped read-back
        if roi is not None:
            x, w, y, h = (int(v) for v in roi)
            params["region"] = [x, x + w, y, y + h]   # -> endpoints (plot coords)
        return params

    def set_acquisition_parameters(self, **values) -> None:
        if "frames_per_cycle" in values:
            self.frames_per_cycle = max(1, int(values["frames_per_cycle"]))   # feed-side, not a camera prop
        kw: dict[str, object] = {}
        if "exposure" in values:
            kw["exposure"] = float(values["exposure"])
        if "region" in values:
            region = values["region"]
            if region in (None, "", "None"):
                kw["roi"] = None
            else:
                # endpoints (plot coords) -> the camera's device ROI rect; the
                # camera then snaps to its sub-array grid.  This is the ONLY place
                # the endpoint<->device-ROI conversion lives.
                x0, x1, y0, y1 = (float(v) for v in region)
                x0, x1 = sorted((x0, x1))
                y0, y1 = sorted((y0, y1))
                kw["roi"] = [int(round(x0)), int(round(x1 - x0)), int(round(y0)), int(round(y1 - y0))]
        if kw:
            self.camera.configure(**kw)   # live on the camera -- no rebuild

    def region_to_acquisition_parameters(self, x_min, x_max, y_min, y_max) -> dict[str, object]:
        """A camera's spatial parameter IS the plot region.  Just normalise the
        rectangle to sorted ENDPOINTS ``[x_min, x_max, y_min, y_max]`` -- no device
        shape here; the endpoint->ROI->sub-array conversion happens deeper
        (``set_acquisition_parameters`` -> ``camera.configure`` -> snap)."""
        x0, x1 = sorted((float(x_min), float(x_max)))
        y0, y1 = sorted((float(y_min), float(y_max)))
        return {"region": [int(round(x0)), int(round(x1)), int(round(y0)), int(round(y1))]}


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
        # Share the feed's stop event so a Stop interrupts a wedged trigger
        # MID-scan-point (the engine's per-point camera.acquire honours it), not
        # only between points.
        try:
            self.measurement.stop_event = self._stop
        except AttributeError:
            pass
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


class ProcessorFeed(ExperimentFeed):
    """One-shot DATA-PROCESSING action feed: runs a :class:`ProcessorSpec` ONCE,
    publishes its result dict to the hub, and self-stops -- the discrete sibling of
    :class:`ScannedMeasurementFeed` (a finite scan).  It DRIVES the spec's
    ``run(ctx)`` and owns no analysis itself.

    The cooperative-stop event is shared with the run via the context, so a long
    camera grab inside ``run`` cancels cleanly on ``stop()`` (the SOLE-camera-owner
    invariant: the run executes on this feed's own thread, never a second acquire)."""

    def __init__(self, hub: SignalHub, spec, *, readout, camera=None,
                 sequencer: object | None = None, params: dict | None = None, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.spec = spec
        self._readout = readout
        self._camera = camera
        self._sequencer = sequencer
        self._params = dict(params or {})
        self.finished = False
        self.result: dict = {}

    def shot(self) -> dict[str, object]:
        from .processor import ProcessorContext

        # One-shot: stop the loop after THIS publish no matter what.  Setting the
        # stop up front means a run() that raises (reported by the loop as
        # feed_error) is NOT retried -- a deterministic processing action runs once.
        self._stop.set()
        ctx = ProcessorContext(
            readout=self._readout, params=self._params,
            camera=self._camera, sequencer=self._sequencer, stop=self._stop)
        result = self.spec.run(ctx)
        self.result = {str(key): value for key, value in dict(result).items()}
        self.finished = True
        out = dict(self.result)
        out["processor_done"] = 1.0
        return out

    def run_to_completion(self) -> "ProcessorFeed":
        """Run the action once synchronously and publish its result (test/headless)."""

        if not self.finished:
            self.step()
        return self

    def published_signals(self) -> frozenset:
        keys = tuple(self.spec.result_keys) + ("processor_done",)
        return frozenset(self.prefix + key for key in keys)


__all__ = ["CameraFrameFeed", "ExperimentFeed", "LoadingFeed", "ProcessorFeed", "ScannedMeasurementFeed"]
