"""Experiment logic nodes: loops that publish per-shot signals into a SignalHub.

A logic node is the upstream half of the task-console contract (the console is the
consumer).  There are three KINDs, all sharing the :class:`LogicNode` loop:

* :class:`Measurement` -- drives a device acquisition loop and publishes named
  signals (e.g. a camera :class:`CameraMeasurement` publishing ``frame``, or a swept
  :class:`ScannedMeasurementNode`);
* :class:`Processor` -- a reactive TRANSFORM node with no acquisition of its own
  (the "func" layer): it consumes hub signals and republishes derived ones, e.g.
  :class:`DetectProcessor` running the SAME ``calibration.detect`` contract the
  real readout uses, frame -> occupancy/counts/rate;
* :class:`Task` -- a one-shot orchestration (e.g. :class:`CalibrateReadoutTask`,
  which produces a ``TrapCalibration`` + an npz artifact and streams its template
  frames to a mid-run output panel).

The loading readout is COMPOSED by the user from these primitives -- a camera
Measurement publishing ``frame`` + a DetectProcessor turning ``frame`` into
occupancy/counts/rate, with calibration produced by a CalibrateReadoutTask.  No
monolithic node fabricates every signal: each layer is independent and explicitly
wired by the notebook or task console.  Every logic node touches only the camera
CONTRACT (``camera.acquire(...)``) and backend-neutral helpers, so the SAME nodes
run on a ``VirtualCamera`` offline and on a real qCMOS -- only the data source
changes.  That is the "virtual == real" core principle (AGENTS.md §2).

Logic nodes run either synchronously (call ``step()`` yourself: deterministic tests)
or in a background thread (``start(rate_hz)``); the hub is the only shared state.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..core.analysis import estimate_thresholds, find_site_centers, grid_shape_tuple, positive_int, roi_counts
from ..core.signals import SignalHub
from ..devices.base import CameraDevice
from ..timing import imaging_channel_kwargs, imaging_sequence


class LogicNode:
    """Base logic node: owns the publish loop; subclasses implement one ``shot()``.

    ``layer`` names which of the five architecture layers this node IS
    (measurement / processor / task) and ``node_label`` is its short human name
    (``camera`` / ``detect`` / ``calibrate`` / a measurement's own name).  The GUI
    shows ``display_label`` (``<prefix><node_label>``) + ``layer`` in a panel's
    signal-flow legend, so the dashboard speaks in LAYER terms -- never the Python
    class name."""

    layer = "node"
    node_label = "node"

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

    @property
    def display_label(self) -> str:
        """Short human name for this logic node in the GUI -- its LAYER node name
        (``camera`` / ``detect`` / ``calibrate`` / a measurement's curve), NEVER the
        Python class name.  The hub prefix is a signal-namespacing detail, not part of
        the label (the namespaced signal names shown alongside disambiguate A/B)."""
        return str(self.node_label)

    # ------------------------------------------------------------------ loop
    def shot(self) -> dict[str, object]:  # pragma: no cover - abstract
        """Produce one shot's worth of {name: value} signals."""
        raise NotImplementedError

    def step(self) -> dict[str, object]:
        """Run ONE shot synchronously and publish it (test/notebook friendly).

        A shot that returns an EMPTY mapping is a no-op: nothing is published and the
        shot counter does not advance.  This lets a REACTIVE logic node (a
        :class:`Processor` that transforms another signal) tick at its own rate yet
        only emit when its input actually advanced."""
        values = self.shot()
        if not values:
            return {}
        values = self._postprocess(values)
        if not values:
            return {}   # a logic node (e.g. a repeat-averaging measurement) may suppress this tick
        named = {self.prefix + key: value for key, value in values.items()}
        self.hub.publish(named)
        self.shots += 1
        return named

    def _postprocess(self, values: dict[str, object]) -> dict[str, object]:
        """Hook to transform a shot's raw values before publish.  Identity in the
        base; :class:`Measurement` applies its ``update_mode`` accumulation here."""
        return values

    def start(self, *, rate_hz: float = 5.0) -> "LogicNode":
        """Publish shots from a daemon thread at ``rate_hz`` until ``stop()``."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self.rate_hz = float(rate_hz)   # remembered so a paused node can resume at the same rate
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
                    self.hub.publish({self.prefix + "node_error": float(self.consecutive_errors)})
                    self._stop.wait(period)
                    continue
                if self.consecutive_errors:
                    # Recovered: clear the banner so a transient hiccup doesn't stick.
                    self.last_error = None
                    self.consecutive_errors = 0
                    self.hub.publish({self.prefix + "node_error": 0.0})
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)

        self._thread = threading.Thread(target=_loop, name=f"zlc-node-{self.prefix or 'main'}", daemon=True)
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
        """The signal names this logic node publishes (behind ``prefix``).  Lets a
        consumer (e.g. the task console) map a panel back to the node that
        produces its data, then expose THAT node's parameters.  Subclasses with
        a fixed signal set override this; the base publishes nothing structured."""
        return frozenset()

    # --------------------------------------------------- acquisition parameters
    # A panel is a VIEW; a logic node produces the data; behind the node sits the data
    # SOURCE (a camera, or the node's own analysis).  A panel's Edit tab edits the
    # source through these two methods, so e.g. a raw-frame panel can tune the
    # camera's exposure/ROI.  The source -- not __init__ reflection -- decides
    # what is editable.
    def acquisition_parameters(self) -> dict[str, object]:
        """Editable parameters of the data SOURCE behind this logic node, as
        ``{name: current_value}`` (e.g. a camera's ``exposure``/``roi``, or this
        node's analysis settings).  Default: nothing editable."""
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
        streaming and the next published frame reflects the change.  When the node
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


# ===================================================================== logic node kinds
# The three concrete logic node KINDS that compose a task console.  All share the
# LogicNode worker-loop/publish/param-queue/cancel infrastructure above; they differ
# only in WHERE their per-shot values come from:
#   Measurement -- ACQUIRES from devices (camera/sequencer), update_mode-driven.
#   Processor   -- TRANSFORMS hub signals into derived signals (no acquisition).
#   Task        -- ORCHESTRATES the others over a multi-step flow, with mid-run output.
class Measurement(LogicNode):
    """A logic node that ACQUIRES data from devices and publishes named signals.

    Continuous vs swept is just ``update_mode`` (how successive shots accumulate),
    not a separate class.  Concrete measurements implement ``shot()``."""

    layer = "measurement"
    node_label = "measurement"
    UPDATE_MODES = ("single", "replace", "roll", "average", "repeat")

    def __init__(self, hub: SignalHub, *, prefix: str = "", update_mode: str = "roll", repeats: int = 1):
        super().__init__(hub, prefix=prefix)
        self.update_mode = self._coerce_update_mode(update_mode)
        self.repeats = max(1, int(repeats))
        self._accum: dict[str, np.ndarray] | None = None   # running sum per numeric key
        self._accum_n = 0

    @classmethod
    def _coerce_update_mode(cls, mode: str) -> str:
        mode = str(mode)
        if mode not in cls.UPDATE_MODES:
            raise ValueError(f"update_mode {mode!r} must be one of {cls.UPDATE_MODES}.")
        return mode

    @staticmethod
    def _is_numeric(value) -> bool:
        return not isinstance(value, str) and isinstance(value, (int, float, np.number, np.ndarray))

    def _postprocess(self, values: dict[str, object]) -> dict[str, object]:
        """Apply ``update_mode`` to a raw shot before publish.

        ``roll`` / ``replace`` pass through (per-shot publish; the rolling-vs-overwrite
        VIEW is the plot's relim, not the node's job).  ``single`` publishes once
        then self-stops.  ``average`` publishes the cumulative running mean of every
        numeric signal.  ``repeat`` accumulates ``repeats`` shots and publishes only
        their mean (suppressing the in-between ticks)."""
        mode = self.update_mode
        if mode in ("roll", "replace"):
            return values
        if mode == "single":
            self._stop.set()
            return values
        numeric = {k: np.asarray(v, dtype=float) for k, v in values.items() if self._is_numeric(v)}
        if self._accum is None or set(self._accum) != set(numeric):
            self._accum = {k: np.zeros_like(val) for k, val in numeric.items()}
            self._accum_n = 0
        for key, val in numeric.items():
            self._accum[key] = self._accum[key] + val
        self._accum_n += 1
        if mode == "average":
            return {**values, **{k: s / self._accum_n for k, s in self._accum.items()}}
        # repeat: hold until `repeats` accumulated, then emit the mean and reset
        if self._accum_n < self.repeats:
            return {}
        mean = {**values, **{k: s / self._accum_n for k, s in self._accum.items()}}
        self._accum = None
        self._accum_n = 0
        return mean


class Processor(LogicNode):
    """A logic node that TRANSFORMS hub signals into derived signals (the "func" layer).

    It consumes one or more named signals, computes, and publishes -- with NO device
    acquisition of its own.  REACTIVE: it only emits when a consumed signal advanced
    since the last tick (tracked via the hub's per-signal version), so it runs as a
    live graph node beside the measurement that produces its input, at its own poll
    rate, and no-ops (``shot`` returns ``{}``) when there is nothing new."""

    layer = "processor"
    node_label = "processor"
    provides: tuple[str, ...] = ()

    def __init__(self, hub: SignalHub, *, consumes, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.consumes = tuple(str(c) for c in consumes)
        self._seen_version: dict[str, int] = {}

    def new_inputs(self) -> dict[str, object] | None:
        """Latest values of the consumed signals IF any advanced since last seen,
        else None (so ``step`` no-ops this tick)."""
        versions = self.hub.signal_versions()
        if not any(versions.get(n, 0) > self._seen_version.get(n, 0) for n in self.consumes):
            return None
        self._seen_version = {n: versions.get(n, 0) for n in self.consumes}
        try:
            return {n: self.hub.latest(n) for n in self.consumes}
        except KeyError:
            return None

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:  # pragma: no cover - abstract
        raise NotImplementedError

    def shot(self) -> dict[str, object]:
        inputs = self.new_inputs()
        return {} if inputs is None else self.transform(inputs)

    def published_signals(self) -> frozenset:
        return frozenset(self.prefix + key for key in self.provides)


class DetectProcessor(Processor):
    """Per-frame atom detection as a live graph node -- the REAL readout pipeline.

    Consumes a camera ``frame`` signal and runs the SAME ``calibration.detect``
    contract the notebook/real readout uses, publishing per-site occupancy/counts +
    a running loading rate.  THIS is the virtual==real split: the camera produces
    frames (a Measurement); detection is a SEPARATE node here -- not one node
    fabricating every signal.  The calibration (site centers + per-site thresholds)
    comes from a prior calibrate-readout Task, exactly as on real hardware.

    Published per new frame (behind ``prefix``): ``occupied`` (N,), ``counts`` (N,),
    ``rate`` scalar EMA loading rate, ``rate_sites`` (N,) per-site EMA, ``rate_grid``
    grid map (when grid known), ``centers`` (N,2), ``thresholds`` (N,)."""

    node_label = "detect"
    provides = ("occupied", "counts", "rate", "rate_sites", "rate_grid", "centers", "thresholds")

    def __init__(self, hub: SignalHub, *, calibration=None, calibration_source=None,
                 source: str = "frame", grid_shape: tuple[int, int] | None = None,
                 ema: float = 0.05, prefix: str = ""):
        super().__init__(hub, consumes=(source,), prefix=prefix)
        self.calibration = calibration
        # Optional lazy source: a callable -> calibration (or None while a calibrate
        # task is still running on its own thread).  Lets the live readout stream
        # WITHOUT blocking the GUI on calibration -- the detector simply no-ops until
        # the calibration is ready, then picks it up.
        self.calibration_source = calibration_source
        self.source = str(source)
        self.grid_shape = None if grid_shape is None else grid_shape_tuple(grid_shape)
        self.ema = float(ema)
        self._rate = float("nan")
        self._rate_sites: np.ndarray | None = None

    def _resolve_calibration(self):
        if self.calibration is None and self.calibration_source is not None:
            self.calibration = self.calibration_source()
        return self.calibration

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        calibration = self._resolve_calibration()
        if calibration is None:
            return {}                                       # not calibrated yet -> no-op (non-blocking)
        frame = np.asarray(inputs[self.source], dtype=float)
        detection = calibration.detect(frame)               # the single readout contract
        occupied = np.asarray(detection.occupied, dtype=float).reshape(-1)
        counts = np.asarray(detection.counts, dtype=float).reshape(-1)
        if self._rate_sites is None or np.isnan(self._rate):
            self._rate_sites = occupied.copy()
            self._rate = float(occupied.mean())
        else:
            self._rate_sites = (1.0 - self.ema) * self._rate_sites + self.ema * occupied
            self._rate = (1.0 - self.ema) * self._rate + self.ema * float(occupied.mean())
        out: dict[str, object] = {
            "occupied": occupied,
            "counts": counts,
            "rate": self._rate,
            "rate_sites": self._rate_sites.copy(),
            "centers": np.asarray(self.calibration.centers, dtype=float),
            "thresholds": np.asarray(self.calibration.thresholds, dtype=float).reshape(-1),
        }
        if self.grid_shape is not None and occupied.size == int(np.prod(self.grid_shape)):
            out["rate_grid"] = self._rate_sites.reshape(self.grid_shape).copy()
        return out


class TaskOutput:
    """The MID-RUN output channel handed to a :class:`Task`'s ``run``.

    A task publishes intermediate numeric signals (a template frame, a progress
    fraction, partial counts) through this channel as it runs, so a DEDICATED task
    panel shows the work in progress -- like a confocal task's live plot.  Signals go
    on the hub behind the task's ``prefix`` (the panel subscribes to e.g.
    ``<prefix>frame`` / ``<prefix>progress``).  The hub stores numbers/arrays only, so
    a textual stage label is surfaced separately (``progress`` 0..1) -- not here."""

    def __init__(self, hub: SignalHub, *, prefix: str = ""):
        self.hub = hub
        self.prefix = str(prefix)
        self.progress = 0.0

    def publish(self, **signals) -> None:
        if "progress" in signals:
            self.progress = float(signals["progress"])
        self.hub.publish({self.prefix + str(key): value for key, value in signals.items()})


class Task(LogicNode):
    """A logic node that ORCHESTRATES devices/measurements/processors over a multi-step
    flow and may emit MID-RUN output to a dedicated panel (confocal-style).

    One-shot: ``step()`` runs the whole ``run(out)`` flow once, publishes its final
    result + ``task_done``, then self-stops.  ``run`` receives a :class:`TaskOutput`
    for intermediate frames/progress.  A task's heavy artifact (e.g. a calibration
    object, saved files) lives on the task instance, not the hub."""

    layer = "task"
    node_label = "task"
    provides: tuple[str, ...] = ()
    # Signals the task streams to its dedicated MID-RUN output panel (via TaskOutput)
    # while it runs -- listed here so the console maps that panel back to this task.
    mid_run: tuple[str, ...] = ()

    def __init__(self, hub: SignalHub, *, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.finished = False
        self.result: dict = {}

    def run(self, out: "TaskOutput") -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def shot(self) -> dict[str, object]:
        # One-shot: stop the loop after THIS publish (a run() that raises is reported
        # via node_error and NOT retried), exactly like a finite scan / processor.
        self._stop.set()
        out = TaskOutput(self.hub, prefix=self.prefix)
        self.result = {str(key): value for key, value in dict(self.run(out)).items()}
        self.finished = True
        return {**self.result, "task_done": 1.0}

    def run_to_completion(self) -> "Task":
        if not self.finished:
            self.step()
        return self

    def published_signals(self) -> frozenset:
        return frozenset(self.prefix + key
                         for key in (tuple(self.provides) + tuple(self.mid_run) + ("task_done",)))


class CalibrateReadoutTask(Task):
    """Acquire frames and run the REAL sitemap + per-site threshold calibration,
    producing a ``TrapCalibration`` as a first-class Task.  Mid-run it publishes the
    template + a sample frame and a
    progress fraction to a dedicated panel.  The resulting calibration is held on
    ``self.calibration`` (and optionally saved to ``save_path`` as an ``npz``) for a
    downstream :class:`DetectProcessor`.  Same primitives as the real readout
    (``calibrate_sitemap_from_images`` / ``calibrate_threshold_from_images``), so it
    is identical on real hardware -- only the camera frames differ."""

    node_label = "calibrate"
    provides = ("centers", "thresholds", "n_sites")
    mid_run = ("frame", "progress")    # streamed to the dedicated mid-run panel

    # The user-facing mode names map to the calibration's (sitemap_method,
    # threshold_method) pair: box+otsu, per-site/uniform PSF + (otsu|bimodal).
    MODE_TO_SITEMAP_METHOD = {"box": "box", "per-site PSF": "psf", "uniform PSF": "uniform_psf"}

    def __init__(self, hub: SignalHub, camera: CameraDevice, *, sequencer: object | None = None,
                 grid_shape: tuple[int, int] = (5, 7), exposure: float = 0.02, roi_radius: int = 1,
                 calibration_frames: int = 4, threshold_frames: int = 24, method: str = "box",
                 mode: str | None = None, threshold_method: str = "otsu",
                 source: str = "live", data_dir: str | None = None,
                 save_path: str | None = None, load_path: str | None = None, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.camera = camera
        self.sequencer = sequencer
        self.grid_shape = grid_shape_tuple(grid_shape)
        self.exposure = float(exposure)
        self.roi_radius = int(roi_radius)
        self.calibration_frames = max(1, int(calibration_frames))
        self.threshold_frames = max(2, int(threshold_frames))
        # ``mode`` is the user-facing name (box / per-site PSF / uniform PSF); the
        # legacy ``method`` (box|psf|uniform_psf) is the resolved sitemap method.
        if mode is not None:
            self.mode = str(mode)
        else:
            self.mode = next((m for m, v in self.MODE_TO_SITEMAP_METHOD.items() if v == str(method)), "box")
        self.threshold_method = str(threshold_method)
        self.source = str(source)
        self.data_dir = data_dir
        self.save_path = save_path
        self.load_path = load_path
        self.calibration = None

    @property
    def method(self) -> str:
        """Resolved sitemap method (box | psf | uniform_psf) for the chosen mode."""
        return self.MODE_TO_SITEMAP_METHOD.get(self.mode, "box")

    def acquisition_parameters(self) -> dict[str, object]:
        """The calibrate task's tunable parameters (source + sitemap grid + frame
        counts + mode + threshold + save/load paths), as ``{name: current}`` --
        shown in the panel's Edit and applied before the next Run."""
        return {
            "source": str(self.source),
            "data_dir": "" if self.data_dir is None else str(self.data_dir),
            "grid_shape": tuple(self.grid_shape),
            "exposure": float(self.exposure),
            "roi_radius": int(self.roi_radius),
            "calibration_frames": int(self.calibration_frames),
            "threshold_frames": int(self.threshold_frames),
            "mode": str(self.mode),
            "threshold_method": str(self.threshold_method),
            "save_path": "" if self.save_path is None else str(self.save_path),
            "load_path": "" if self.load_path is None else str(self.load_path),
        }

    def set_acquisition_parameters(self, **values) -> None:
        if "source" in values:
            self.source = str(values["source"])
        if "data_dir" in values:
            self.data_dir = str(values["data_dir"]) or None
        if "grid_shape" in values:
            self.grid_shape = grid_shape_tuple(values["grid_shape"])
        if "exposure" in values:
            self.exposure = float(values["exposure"])
        if "roi_radius" in values:
            self.roi_radius = int(values["roi_radius"])
        if "calibration_frames" in values:
            self.calibration_frames = max(1, int(values["calibration_frames"]))
        if "threshold_frames" in values:
            self.threshold_frames = max(2, int(values["threshold_frames"]))
        if "mode" in values:
            self.mode = str(values["mode"])
        if "threshold_method" in values:
            self.threshold_method = str(values["threshold_method"])
        if "save_path" in values:
            self.save_path = str(values["save_path"]) or None
        if "load_path" in values:
            self.load_path = str(values["load_path"]) or None

    def run(self, out: "TaskOutput") -> dict:
        # Load an existing calibration outright (no acquisition) when a load path
        # is given -- the user reuses a saved centers+thresholds artifact.
        if self.load_path:
            from ..core.calibration import TrapCalibration
            self.calibration = TrapCalibration.load(self.load_path)
            out.publish(progress=1.0)
            centers = np.asarray(self.calibration.centers, dtype=float)
            thr = np.asarray(self.calibration.thresholds, dtype=float).reshape(-1)
            return {"centers": centers, "thresholds": thr, "n_sites": float(len(centers))}
        if str(self.source) == "folder":
            calibration = self._run_from_folder(out)
        else:
            calibration = self._run_live(out)
        self.calibration = calibration
        out.publish(progress=1.0)
        centers = np.asarray(self.calibration.centers, dtype=float)
        thr = np.asarray(self.calibration.thresholds, dtype=float).reshape(-1)
        if self.save_path:
            # A .npz / .json path saves the full calibration (so load_path can
            # restore it); a bare path falls back to the centers+thresholds npz.
            from pathlib import Path as _Path
            if _Path(self.save_path).suffix.lower() in (".npz", ".json"):
                self.calibration.save(self.save_path)
            else:
                np.savez(self.save_path, centers=centers, thresholds=thr)
        return {"centers": centers, "thresholds": thr, "n_sites": float(len(centers))}

    def _run_live(self, out: "TaskOutput"):
        """Acquire frames now (camera + pulse) and calibrate the sitemap + thresholds."""
        from .calibration import calibrate_sitemap_from_images, calibrate_threshold_from_images
        channel_kwargs = imaging_channel_kwargs(self.sequencer)
        sitemap_seq = imaging_sequence(exposure=self.exposure, load=True, name="sitemap", **channel_kwargs)
        readout_seq = imaging_sequence(exposure=self.exposure, load=True, name="readout", **channel_kwargs)
        template = self.camera.acquire(self.calibration_frames, sequence=sitemap_seq,
                                       sequencer=self.sequencer, stop=self._stop)
        out.publish(frame=np.asarray(template[-1], dtype=float), progress=0.4)
        sitemap = calibrate_sitemap_from_images(
            template, grid_shape=self.grid_shape, roi_radius=self.roi_radius, method=self.method, display=False)
        samples = self.camera.acquire(self.threshold_frames, sequence=readout_seq,
                                      sequencer=self.sequencer, stop=self._stop)
        out.publish(frame=np.asarray(samples[-1], dtype=float), progress=0.85)
        thresholds = calibrate_threshold_from_images(
            samples, sitemap.calibration, method=self.threshold_method, display=False)
        return thresholds.calibration

    def _run_from_folder(self, out: "TaskOutput"):
        """Calibrate from frames SAVED IN A FOLDER (the real-data flow): index the
        run, fit the sitemap from the reference template, then per-site thresholds."""
        from .calibration import calibrate_sitemap_from_images, calibrate_threshold_from_images
        from .imageio import index_run
        if not self.data_dir:
            raise ValueError("source='folder' needs a data_dir folder path.")
        run = index_run(self.data_dir, "img")
        template_frames = list(run.template_frames())
        if template_frames:
            out.publish(frame=np.asarray(template_frames[-1], dtype=float), progress=0.4)
        sitemap = calibrate_sitemap_from_images(
            template_frames, grid_shape=self.grid_shape, roi_radius=self.roi_radius,
            method=self.method, display=False)
        samples = list(run.short_frames())
        if samples:
            out.publish(frame=np.asarray(samples[-1], dtype=float), progress=0.85)
        thresholds = calibrate_threshold_from_images(
            samples, sitemap.calibration, method=self.threshold_method, display=False)
        return thresholds.calibration


class CameraMeasurement(Measurement):
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
    single-frame measurement always shows the first emCCD image.  Put ``value = frame_0`` on
    one panel and ``value = frame_1`` on another to watch the two triggers side by
    side.  (``frames_per_cycle`` must match the camera-trigger count per cycle so the
    per-trigger assignment stays phase-aligned; for a measurement that FIRES the sequence,
    ``acquire`` enforces frames == trigger count.)"""

    node_label = "camera"

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
            self.frames_per_cycle = max(1, int(values["frames_per_cycle"]))   # measurement-side, not a camera prop
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


class ScannedMeasurementNode(Measurement):
    """Drive a :class:`ScannedMeasurement` one scan point per ``shot()``, into a hub.

    Wraps a swept measurement as a console logic node (``start``/``stop``/``running``)
    so a finite scan grows a live curve in the task console exactly as the
    free-running loading rate trace grows.  Each ``shot()`` advances ONE
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

    Finite-scan semantics: after the last point is published the node sets its
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
        # Share the node's stop event so a Stop interrupts a wedged trigger
        # MID-scan-point (the engine's per-point camera.acquire honours it), not
        # only between points.
        try:
            self.measurement.stop_event = self._stop
        except AttributeError:
            pass
        self.x_key = str(x_key)
        self.y_key = str(y_key)
        self.node_label = str(y_key)   # GUI flow legend shows the curve it produces
        self.grid_shape = None if grid_shape is None else grid_shape_tuple(grid_shape)
        # The measurement owns the swept values (single source of truth); the node
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
        ``start()`` loop never does (the node stops itself), and tests check
        ``finished`` first.
        """

        if self.finished:
            raise StopIteration("ScannedMeasurementNode: scan already complete.")
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

    def run_to_completion(self) -> "ScannedMeasurementNode":
        """Synchronously run + publish every remaining scan point (test/headless)."""

        while not self.finished:
            self.step()
        return self

    def published_signals(self) -> frozenset:
        keys = (self.x_key, self.y_key, self.y_key + "_sites", self.y_key + "_grid",
                "scan_done", "shot")
        return frozenset(self.prefix + key for key in keys)


class ProcessorRun(LogicNode):
    """One-shot DATA-PROCESSING logic node: runs a :class:`ProcessorSpec` ONCE,
    publishes its result dict to the hub, and self-stops -- the discrete sibling of
    :class:`ScannedMeasurementNode` (a finite scan).  It DRIVES the spec's
    ``run(ctx)`` and owns no analysis itself.

    The cooperative-stop event is shared with the run via the context, so a long
    camera grab inside ``run`` cancels cleanly on ``stop()`` (the SOLE-camera-owner
    invariant: the run executes on this node's own thread, never a second acquire)."""

    layer = "processor"
    node_label = "processor"

    def __init__(self, hub: SignalHub, spec, *, readout, camera=None,
                 sequencer: object | None = None, params: dict | None = None, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.spec = spec
        self.node_label = getattr(spec, "name", "processor")
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
        # node_error) is NOT retried -- a deterministic processing action runs once.
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

    def run_to_completion(self) -> "ProcessorRun":
        """Run the action once synchronously and publish its result (test/headless)."""

        if not self.finished:
            self.step()
        return self

    def published_signals(self) -> frozenset:
        keys = tuple(self.spec.result_keys) + ("processor_done",)
        return frozenset(self.prefix + key for key in keys)


__all__ = [
    "CalibrateReadoutTask",
    "CameraMeasurement",
    "DetectProcessor",
    "Measurement",
    "Processor",
    "ProcessorRun",
    "LogicNode",
    "ScannedMeasurementNode",
    "Task",
    "TaskOutput",
]
