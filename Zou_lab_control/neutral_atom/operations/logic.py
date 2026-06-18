"""Experiment logic nodes: loops that publish per-shot signals into a SignalHub.

A logic node is the upstream half of the task-console contract (the console is the
consumer).  There are three KINDs, all sharing the :class:`LogicNode` loop:

* :class:`Measurement` -- drives a device acquisition loop and publishes named
  signals (e.g. a camera :class:`CameraMeasurement` publishing ``frame``, or a swept
  :class:`ScannedMeasurementNode`);
* :class:`Processor` -- a reactive TRANSFORM node with no acquisition of its own
  (the "func" layer): it consumes hub signals and republishes derived ones, e.g.
  :class:`OccupancyProcessor` running the SAME ``calibration.detect`` contract the
  real readout uses, frame -> occupancy/counts/rate;
* :class:`Task` -- a one-shot orchestration (e.g. :class:`CalibrateReadoutTask`,
  which produces a ``TrapCalibration`` + an npz artifact and streams its template
  frames to a mid-run output panel).

The loading readout is COMPOSED by the user from these primitives -- a camera
Measurement publishing ``frame`` + an OccupancyProcessor turning ``frame`` into
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.analysis import grid_shape_tuple
from ..core.signals import SignalHub
from ..devices.base import CameraDevice


@dataclass(frozen=True)
class SignalSpec:
    """What ONE output of a logic node MEANS -- the human label + unit + one-line
    description for a signal it publishes (or, for a task, produces off-hub).

    A node declares these ONCE (``output_specs``); the GUI reads them so a plot can
    set its axis label/unit from the producing measurement (not a hard-coded per-kind
    string) and a node's "publishes" legend reads as ``occupied  (35,)  per-site 0/1
    occupancy`` -- every output named, shaped and explained.  ``name`` is the FULL hub
    signal name (with the node's prefix), so a consumer maps a signal straight to its
    meaning."""

    name: str               # full published signal name (incl. the node's prefix)
    label: str              # axis / legend label, e.g. "loading rate"
    unit: str = ""          # physical unit, e.g. "s" / "K" (blank = dimensionless)
    description: str = ""    # one-line human meaning for the publishes legend

    @property
    def axis_label(self) -> str:
        """``label (unit)`` for a plot axis, or just ``label`` when dimensionless."""
        return f"{self.label} ({self.unit})" if self.unit else self.label


def describe_shape(value) -> str:
    """A standardized shape string read straight from a published VALUE -- the SINGLE
    way the GUI says what a signal looks like, AUTO-EXTRACTED from real data rather
    than a hand-typed name->format map (which silently drifts from what a node really
    emits).  ``scalar`` for a 0-d / Python number, else the numpy shape tuple verbatim
    (``(35,)`` / ``(35, 2)`` / ``(96, 128)``).  ``None`` -> ``"—"`` (no value yet)."""
    if value is None:
        return "—"
    shape = np.shape(value)
    if shape == ():
        return "scalar"
    if len(shape) == 1:                  # numpy 1-D repr keeps the trailing comma: (35,)
        return f"({int(shape[0])},)"
    return "(" + ", ".join(str(int(n)) for n in shape) + ")"


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

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """One :class:`SignalSpec` per published signal -- the LABEL / unit / one-line
        meaning the GUI shows (plot axis label, the "publishes" legend).  The base
        derives a bare spec (label = name) for every :meth:`published_signals` entry;
        a measurement/processor OVERRIDES this to give each output a real label, unit
        and description so a plot reads its axis from the producing node, not a
        hard-coded per-kind string.  Single source -- the node owns what its outputs
        mean."""
        return tuple(SignalSpec(str(name), str(name)) for name in sorted(self.published_signals()))

    def signal_spec(self, name: str) -> SignalSpec | None:
        """The :class:`SignalSpec` for one published signal name, or ``None``."""
        for spec in self.output_specs():
            if spec.name == str(name):
                return spec
        return None

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


class OccupancyProcessor(Processor):
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

    node_label = "occupancy"
    # ``frame_judged`` = the EXACT frame this occupancy was computed from, republished so
    # the site map's underlay is the SAME shot as the rings (the camera keeps streaming
    # newer frames on its own thread; using the live camera frame would offset the rings).
    provides = ("occupied", "counts", "rate", "rate_sites", "rate_grid", "centers",
                "thresholds", "frame_judged")
    # The site map takes ONE signal (an occupancy vector this node publishes) and resolves
    # its ring CENTRES + frame UNDERLAY from the SAME node: these name the two outputs that
    # carry them.  THIS is the single source -- the panel layer (ProcessorSpec.metadata) and
    # the console's site-map resolver both read these, so "one signal" wiring never drifts.
    sitemap_centers_key = "centers"
    sitemap_image_key = "frame_judged"

    def __init__(self, hub: SignalHub, *, calibration=None, calibration_source=None,
                 source: str = "frame", grid_shape: tuple[int, int] | None = None,
                 ema: float = 0.05, method: str | None = None, prefix: str = ""):
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
        # The READOUT method (box / per-site PSF / ...) is chosen HERE, not at calibration
        # time: one calibration carries every method's geometry + thresholds, and the
        # processor picks which to read with (None = the calibration's default).
        self.method = None if method in (None, "") else str(method)
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
        detection = calibration.detect(frame, method=self.method)   # the single readout contract
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
            "thresholds": np.asarray(detection.thresholds, dtype=float).reshape(-1),
            # publish the judged frame ATOMICALLY with the occupancy -> the site map's
            # underlay + rings are always the same shot (root fix for the misalignment).
            "frame_judged": frame,
        }
        if self.grid_shape is not None and occupied.size == int(np.prod(self.grid_shape)):
            out["rate_grid"] = self._rate_sites.reshape(self.grid_shape).copy()
        return out

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """Label + meaning of each detection signal (the readout pipeline's outputs)."""
        p = self.prefix
        return (
            SignalSpec(p + "occupied", "occupancy", "", "per-site single-shot occupancy (0 / 1)"),
            SignalSpec(p + "counts", "readout counts", "", "per-site integrated readout signal"),
            SignalSpec(p + "rate", "loading rate", "", "running-mean loading rate over all sites"),
            SignalSpec(p + "rate_sites", "loading rate", "", "per-site running-mean loading rate"),
            SignalSpec(p + "rate_grid", "loading rate", "", "per-site loading rate as a site grid"),
            SignalSpec(p + "centers", "site centre", "px", "site centres in camera pixels (N, 2)"),
            SignalSpec(p + "thresholds", "threshold", "counts", "per-site bright/dark count threshold"),
            SignalSpec(p + "frame_judged", "camera image", "counts",
                       "the exact camera frame this occupancy was judged from (site-map underlay)"),
        )


class TaskOutput:
    """The MID-RUN output channel handed to a :class:`Task`'s ``run`` -- a per-task
    BUFFER, NOT the SignalHub.

    A task publishes intermediate numeric signals (a template frame, a progress
    fraction) here as it runs, so a DEDICATED task panel shows the work in progress --
    like a confocal task's live plot.  Task output deliberately does **not** go on the
    hub: the hub carries ONLY measurement + processor outputs, so a one-shot task's
    transient frames never collide with -- or masquerade as -- live readout signals.
    The console reads ``latest`` / ``version`` to render the task's dedicated panel; a
    textual stage is surfaced via the numeric ``progress`` 0..1."""

    def __init__(self, *, prefix: str = ""):
        self.prefix = str(prefix)
        self.progress = 0.0
        self._latest: dict[str, object] = {}
        self._version = 0

    def publish(self, **signals) -> None:
        if "progress" in signals:
            self.progress = float(signals["progress"])
        # buffer (task-local, so raw names -- no prefix collision to guard against)
        self._latest.update({str(key): value for key, value in signals.items()})
        self._version += 1

    def latest(self, name: str):
        """The most recent value buffered under ``name`` (or ``None``)."""
        return self._latest.get(str(name))

    def names(self) -> list:
        """Names buffered so far (the task's declared ``mid_run`` keys as they arrive)."""
        return list(self._latest)

    @property
    def version(self) -> int:
        """Bumped on every ``publish`` so a viewer can poll for fresh mid-run data."""
        return self._version


class Task(LogicNode):
    """A logic node that ORCHESTRATES devices/measurements/processors over a multi-step
    flow and may emit MID-RUN output to a dedicated panel (confocal-style).

    One-shot: ``step()`` runs the whole ``run(out)`` flow once, then self-stops.  A
    task publishes NOTHING to the hub: its result lives on ``self.result`` and its
    heavy artifact (e.g. a calibration object, saved files) on the task instance,
    while ``run`` writes intermediate frames/progress to its own :class:`TaskOutput`
    buffer (``self.output``, NOT the hub) for the dedicated mid-run panel."""

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
        # Mid-run output is a per-task BUFFER (NOT the hub) -- created up front so the
        # console can bind the task's dedicated panel to it before/while it runs.
        self.output = TaskOutput(prefix=self.prefix)

    def run(self, out: "TaskOutput") -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def shot(self) -> dict[str, object]:
        # One-shot: stop the loop AFTER this run (a run() that raises is reported via
        # node_error and NOT retried), exactly like a finite scan / processor.  The stop
        # event is set in a ``finally`` -- AFTER ``run`` -- so that DURING the run it
        # stays clear and means "cancel": ``run`` can poll ``self._stop`` (and pass it to
        # ``camera.acquire``) to interrupt a long acquisition the moment Stop is pressed.
        # The result + mid-run output stay on the INSTANCE (self.result / self.output); a
        # task publishes NOTHING to the hub -- the hub is measurements + processors only.
        try:
            self.result = {str(key): value for key, value in dict(self.run(self.output)).items()}
            self.finished = True
        finally:
            self._stop.set()
        return {}

    def run_to_completion(self) -> "Task":
        if not self.finished:
            self.step()
        return self

    def published_signals(self) -> frozenset:
        # A task publishes nothing to the hub (its result lives on the instance, its
        # mid-run output in self.output) -- so it provides no hub signal name.  What it
        # PRODUCES (``provides`` result keys + ``mid_run`` stream keys) is documented by
        # the console from those public attrs, with shapes read off real values.
        return frozenset()


class CalibrateReadoutTask(Task):
    """Acquire frames and run the REAL sitemap + per-site threshold calibration,
    producing a ``TrapCalibration`` as a first-class Task.  Mid-run it publishes the
    template + a sample frame and a
    progress fraction to a dedicated panel.  The resulting calibration is held on
    ``self.calibration`` (and optionally saved to ``save_path`` as an ``npz``) for a
    downstream :class:`OccupancyProcessor`.  Same primitives as the real readout
    (``calibrate_sitemap_from_images`` / ``calibrate_threshold_from_images``), so it
    is identical on real hardware -- only the camera frames differ."""

    node_label = "calibrate"
    provides = ("centers", "thresholds", "n_sites")
    mid_run = ("frame", "progress", "stage")   # streamed to the dedicated mid-run panel + banner

    # The imaging pulse TEMPLATE the cali loads -- a REAL, inspectable program (no opaque
    # "built-in" sentinel).  A bare name resolves to the shipped configs template; an
    # absolute path to the user's own PulseTableState .json.  Each cali pass LOADS it and
    # SETS its imaging exposure (with_imaging_exposure) -- "load a template, set the
    # duration, on/off, run".  The cali no longer chooses a readout METHOD: it computes
    # ALL methods (box / per-site PSF / uniform PSF) and the OccupancyProcessor picks one.
    DEFAULT_PULSE_TEMPLATE = "imaging_template.json"

    @classmethod
    def _resolve_template(cls, pulse_template):
        """Load the imaging template: the given path if it is a real file, else the shipped
        template of that name in the repo ``pulses/`` folder (where the pulse GUI saves and
        the Browse dialog opens -- so the default ``imaging_template.json`` is a REAL,
        inspectable file the experimenter can find), else the in-memory default."""
        from ..timing import PulseTableState, default_imaging_template
        text = str(pulse_template or "").strip() or cls.DEFAULT_PULSE_TEMPLATE
        path = Path(text)
        if path.is_file():
            return PulseTableState.load(path)
        name = path.name
        for base in (Path("pulses"), Path(__file__).resolve().parents[3] / "pulses"):
            shipped = base / name
            if shipped.is_file():
                return PulseTableState.load(shipped)
        return default_imaging_template()

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """What the calibration PRODUCES (off the hub) + streams mid-run -- keyed by the
        bare ``provides`` / ``mid_run`` names so the console legend reads e.g.
        ``centers (result)  (35, 2)  fitted site coordinates``."""
        return (
            SignalSpec("frame", "reference frame", "counts", "long-exposure template frame (streamed live)"),
            SignalSpec("centers", "site centres", "px", "fitted site coordinates (N, 2)"),
            SignalSpec("thresholds", "threshold", "counts", "per-site bright/dark count threshold (N,)"),
            SignalSpec("n_sites", "site count", "", "number of trap sites found"),
        )

    def __init__(self, hub: SignalHub, camera: CameraDevice, *, sequencer: object | None = None,
                 grid_shape: tuple[int, int] = (5, 7), roi_radius: int = 1,
                 sitemap_exposure: float = 0.05, readout_exposure: float = 0.02,
                 pulse_template: str = "imaging_template.json",
                 calibration_frames: int = 30, threshold_frames: int = 100,
                 threshold_method: str = "otsu",
                 source: str = "live", folder: str = "calibrations",
                 calibration_sink=None, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.camera = camera
        self.sequencer = sequencer
        # Where the finished calibration is HANDED BACK to: set by readout.calibrate_task
        # to write the session calibration (``readout.current``), so a decoupled live
        # OccupancyProcessor picks it up the instant this task completes -- the
        # "cali -> occupancy" wiring with NO path to type (the bug this fixes: the task
        # produced a calibration but never published it, so occupancy stayed on a stale /
        # empty calibration and showed no sites).
        self.calibration_sink = calibration_sink
        self.grid_shape = grid_shape_tuple(grid_shape)
        self.roi_radius = int(roi_radius)
        # TWO exposures (set on the SAME loaded template per pass): the SITEMAP/reference
        # pass uses a LONGER duration (more photons -> cleaner site centroids + PSF fit);
        # the THRESHOLD pass uses the ACTUAL readout duration (thresholds learnt under real
        # readout conditions).  The template is a real PulseTableState; ``with_imaging_exposure``
        # sets its imaging window per pass -- "load a template, set the duration, run".
        self.sitemap_exposure = float(sitemap_exposure)
        self.readout_exposure = float(readout_exposure)
        self.pulse_template = str(pulse_template or self.DEFAULT_PULSE_TEMPLATE)
        self.calibration_frames = max(1, int(calibration_frames))
        self.threshold_frames = max(2, int(threshold_frames))
        self.threshold_method = str(threshold_method)
        # ONE folder for input + output (no blank paths); ``source`` decides how it's used.
        self.source = str(source)
        self.folder = str(folder or "calibrations")
        self.calibration = None

    def acquisition_parameters(self) -> dict[str, object]:
        """The calibrate task's tunable parameters (source + folder + pulse template +
        exposures + grid + frame counts + threshold), as ``{name: current}`` -- shown in
        the panel's Edit and applied before the next Run.  Every value is concrete (no
        blank): the pulse template + folder read back as their paths.  NOTE the cali no
        longer has a readout ``mode`` -- it computes every method; the OccupancyProcessor
        chooses box / per-site PSF / uniform PSF."""
        return {
            "source": str(self.source),
            "folder": str(self.folder),
            "pulse_template": str(self.pulse_template),
            "grid_shape": tuple(self.grid_shape),
            "sitemap_exposure": float(self.sitemap_exposure),
            "readout_exposure": float(self.readout_exposure),
            "roi_radius": int(self.roi_radius),
            "calibration_frames": int(self.calibration_frames),
            "threshold_frames": int(self.threshold_frames),
            "threshold_method": str(self.threshold_method),
        }

    def set_acquisition_parameters(self, **values) -> None:
        if "source" in values:
            self.source = str(values["source"])
        if "folder" in values:
            self.folder = str(values["folder"]) or "calibrations"
        if "pulse_template" in values:
            self.pulse_template = str(values["pulse_template"]) or self.DEFAULT_PULSE_TEMPLATE
        if "grid_shape" in values:
            self.grid_shape = grid_shape_tuple(values["grid_shape"])
        if "sitemap_exposure" in values:
            self.sitemap_exposure = float(values["sitemap_exposure"])
        if "readout_exposure" in values:
            self.readout_exposure = float(values["readout_exposure"])
        if "roi_radius" in values:
            self.roi_radius = int(values["roi_radius"])
        if "calibration_frames" in values:
            self.calibration_frames = max(1, int(values["calibration_frames"]))
        if "threshold_frames" in values:
            self.threshold_frames = max(2, int(values["threshold_frames"]))
        if "mode" in values:
            self.mode = self._coerce_mode(values["mode"])
        if "threshold_method" in values:
            self.threshold_method = str(values["threshold_method"])

    def run(self, out: "TaskOutput") -> dict:
        # "saved calibration": reload a finished calibration.json from the folder (no
        # acquisition) -- the user reuses a saved centers+thresholds artifact.
        if str(self.source) == "saved calibration":
            from ..core.calibration import TrapCalibration
            self.calibration = TrapCalibration.load(self._saved_calibration_path())
            self._adopt()
            out.publish(progress=1.0, stage=f"loaded calibration from {self.folder}")
            centers = np.asarray(self.calibration.centers, dtype=float)
            thr = np.asarray(self.calibration.thresholds, dtype=float).reshape(-1)
            return {"centers": centers, "thresholds": thr, "n_sites": float(len(centers)),
                    "report_dir": ""}
        if str(self.source) == "saved frames":
            calibration = self._run_from_folder(out)
        else:
            calibration = self._run_live(out)
        self.calibration = calibration
        self._adopt()                            # hand the calibration to the session NOW
        centers = np.asarray(self.calibration.centers, dtype=float)
        thr = np.asarray(self.calibration.thresholds, dtype=float).reshape(-1)
        # ALWAYS write the rb87-style report (per-site distribution + fidelity + site map
        # + a loadable calibration.json/npz) to a timestamped sub-folder of ``folder``, so
        # a calibration leaves reviewable + reloadable artifacts on disk.
        out.publish(progress=0.98, stage="writing distribution + fidelity report")
        self.report = self._write_report(self.report_dir())
        folder = self.report.get("folder") if isinstance(self.report, dict) else None
        out.publish(progress=1.0, stage=(f"saved report -> {folder}" if folder else "done"))
        return {"centers": centers, "thresholds": thr, "n_sites": float(len(centers)),
                "report_dir": str(folder or "")}

    def _saved_calibration_path(self) -> "Path":
        """Locate the saved calibration to reload from ``folder``: the folder itself if it
        is a .json/.npz file, else ``folder/calibration.json`` (the report's artifact)."""
        p = Path(self.folder)
        if p.suffix.lower() in (".json", ".npz"):
            return p
        return p / "calibration.json"

    def _adopt(self) -> None:
        """Hand the just-produced calibration to the session via ``calibration_sink`` so a
        decoupled OccupancyProcessor (reading ``readout.current``) starts judging sites
        immediately -- the cali->occupancy connection, no path to type."""
        if self.calibration_sink is not None and self.calibration is not None:
            self.calibration_sink(self.calibration)

    def report_dir(self) -> "Path":
        """A timestamped run sub-folder under ``folder`` (e.g.
        ``calibrations/calibration_20260617_213000/``), so each calibration's figures +
        loadable calibration land in their own findable place and never overwrite a prior
        run.  ``folder`` always has a real default ("calibrations"), so this is never
        ambiguous."""
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.folder) / f"calibration_{stamp}"

    def _write_report(self, folder) -> dict:
        """Write the per-site distribution / fidelity / site-map report for THIS run's
        readout frames (no-op-safe if a folder run kept no frames)."""
        from datetime import datetime
        from .calibration_report import write_calibration_report
        frames = list(getattr(self, "_readout_samples", []) or [])
        if not frames:
            return {}
        # the reference brackets (if acquired) give the report ground-truth labels, so the
        # per-method fidelity is held-out classification accuracy (distinct box / PSF / uniform),
        # not the affine-invariant self-consistent estimate.
        return write_calibration_report(
            folder, calibration=self.calibration, readout_frames=frames,
            template=getattr(self, "_reference_template", None),
            threshold_method=self.threshold_method,
            reference_groups=getattr(self, "_reference_groups", None),
            readout_by_group=getattr(self, "_readout_by_group", None),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _imaging_seq(self, exposure: float, name: str):
        """The acquisition sequence for one cali pass: LOAD the pulse template and SET its
        imaging window to ``exposure`` -- the user's "load a template, set the duration,
        run" workflow.  ONE template is reused for both passes (only the exposure differs).
        Same path on real hardware (only the camera frames differ)."""
        return self._resolve_template(self.pulse_template).with_imaging_exposure(exposure).to_sequence(name=name)

    def _collect_frames(self, seq, n_frames: int, out: "TaskOutput", *, stage: str,
                        progress_lo: float, progress_hi: float) -> list:
        """Acquire ``n_frames`` ONE AT A TIME, streaming each to the task's mid-run
        panel (frame + stage text + a progress fraction mapped into
        ``[progress_lo, progress_hi]``) so the operator watches the data come in --
        exactly the frames the extraction below runs on.  Honours the Stop event
        between frames (a long calibration is interruptible)."""
        frames: list = []
        n = max(1, int(n_frames))
        for i in range(n):
            if self._stop.is_set():
                break
            batch = self.camera.acquire(1, sequence=seq, sequencer=self.sequencer, stop=self._stop)
            frame = np.asarray(batch[-1], dtype=float)
            frames.append(frame)
            frac = progress_lo + (progress_hi - progress_lo) * (i + 1) / n
            out.publish(frame=frame, progress=frac, stage=f"{stage} {i + 1}/{n}")
        return frames

    def _run_live(self, out: "TaskOutput"):
        """Acquire frames now and run the SAME sitemap + per-site-threshold extraction
        the real readout uses -- the rb87-style flow.

        TWO pulses: the REFERENCE pass uses the LONGER readout duration
        (``sitemap_pulse`` / ``sitemap_exposure``) so the averaged template has high SNR
        site centroids; the READOUT pass uses the ACTUAL (short) readout duration
        (``readout_pulse`` / ``readout_exposure``) so per-site thresholds are learnt
        under real readout conditions.  Both stream frame-by-frame to the mid-run panel
        (the operator sees ~N reference then ~M readout frames accumulate), then the
        centers come from averaging the reference frames and the thresholds from the
        per-site count distribution of the readout frames."""
        from .calibration import calibrate_all_methods_from_images
        # name="reference" (NOT "sitemap"): the virtual camera images a REAL ~50% loading
        # at the long exposure, and averaging many such frames reveals every site -- the
        # authentic template, not a synthetic all-bright frame.
        sitemap_seq = self._imaging_seq(self.sitemap_exposure, "reference")
        out.publish(progress=0.0, stage="loading reference frames")
        template = self._collect_frames(sitemap_seq, self.calibration_frames, out,
                                        stage="reference frame", progress_lo=0.0, progress_hi=0.3)
        # READOUT BRACKETS (the Rb87 fidelity flow): each shot images the SAME atoms
        # long-short-long, so the two long frames vote a ground-truth occupancy label for the
        # short readout (a shot where they disagree is an atom-loss event -> ambiguous).  The
        # short readout frames also feed the box/PSF otsu thresholds (cali once, read many ways).
        ref_groups, readout_by_group = self._collect_bracket_groups(
            out, progress_lo=0.35, progress_hi=0.9)
        samples = list(readout_by_group)
        out.publish(progress=0.95, stage="finding sites + thresholds (box / PSF)")
        calibration = calibrate_all_methods_from_images(
            template, samples, grid_shape=self.grid_shape, roi_radius=self.roi_radius,
            threshold_method=self.threshold_method)
        # Keep the readout frames + reference brackets + averaged template so run() can write
        # the rb87-style report: per-site distribution + HELD-OUT per-method fidelity scored
        # against the bracket-voted ground truth + the site map.
        self._readout_samples = list(samples)
        self._reference_groups = ref_groups
        self._readout_by_group = list(readout_by_group)
        self._reference_template = (np.mean(np.asarray(template, dtype=float), axis=0)
                                    if template else None)
        return calibration

    #: The number of long reference frames bracketing each short readout (a "20-5-20" shot):
    #: two long images, one before and one after, vote ground truth by strict consensus.
    REFERENCE_FRAMES_PER_BRACKET = 2

    def _collect_bracket_groups(self, out: "TaskOutput", *, progress_lo: float, progress_hi: float):
        """Acquire ``threshold_frames`` reference BRACKETS.  Each bracket is ONE correlated
        shot -- a long-short-long camera sequence imaging the SAME atom loading (the trap is
        held on, no re-cooling between triggers, so a following frame sees the previous
        frame's survivors).  Returns ``(reference_groups, readout_per_group)``:
        ``reference_groups[g]`` is the long frames that vote ground truth, and
        ``readout_per_group[g]`` is the short readout frame scored against them.  Honours the
        Stop event between brackets (interruptible).  Identical on real hardware -- only the
        camera frames' author differs (``virtual == real``)."""
        from ..timing import imaging_channel_kwargs, reference_bracket_sequence
        n_ref = int(self.REFERENCE_FRAMES_PER_BRACKET)
        readout_index = n_ref // 2
        bracket = reference_bracket_sequence(
            ref_exposure=self.sitemap_exposure, readout_exposure=self.readout_exposure,
            n_ref=n_ref, **imaging_channel_kwargs(self.sequencer))
        n_groups = max(2, int(self.threshold_frames))
        reference_groups: list = []
        readout_per_group: list = []
        for g in range(n_groups):
            if self._stop.is_set():
                break
            batch = self.camera.acquire(n_ref + 1, sequence=bracket, sequencer=self.sequencer,
                                        stop=self._stop)
            frames = [np.asarray(f, dtype=float) for f in batch]
            readout_per_group.append(frames[readout_index])
            reference_groups.append([f for i, f in enumerate(frames) if i != readout_index])
            frac = progress_lo + (progress_hi - progress_lo) * (g + 1) / n_groups
            out.publish(frame=frames[readout_index], progress=frac, stage=f"readout bracket {g + 1}/{n_groups}")
        return reference_groups, readout_per_group

    def _run_from_folder(self, out: "TaskOutput"):
        """Calibrate from frames SAVED IN A FOLDER (the real-data flow): index the run,
        find the sites from the reference template, then per-site thresholds for every
        readout method (the OccupancyProcessor picks which to use)."""
        from .calibration import calibrate_all_methods_from_images
        from .imageio import index_run
        if not self.folder:
            raise ValueError("source='saved frames' needs a folder of saved frames.")
        run = index_run(self.folder, "img")
        template_frames = list(run.template_frames())
        if template_frames:
            out.publish(frame=np.asarray(template_frames[-1], dtype=float), progress=0.4)
        samples = list(run.short_frames())
        if samples:
            out.publish(frame=np.asarray(samples[-1], dtype=float), progress=0.85)
        calibration = calibrate_all_methods_from_images(
            template_frames, samples, grid_shape=self.grid_shape, roi_radius=self.roi_radius,
            threshold_method=self.threshold_method)
        self._readout_samples = list(samples)
        self._reference_template = (np.mean(np.asarray(template_frames, dtype=float), axis=0)
                                    if template_frames else None)
        return calibration


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

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """Each camera output is a raw 2-D image (the newest ``frame`` + one
        ``frame_i`` per per-cycle trigger)."""
        specs = []
        for name in sorted(self.published_signals()):
            bare = name[len(self.prefix):] if self.prefix and name.startswith(self.prefix) else name
            desc = "newest camera frame" if bare == "frame" else f"camera frame {bare.split('_')[-1]} of the cycle"
            specs.append(SignalSpec(name, "camera image", "counts", desc))
        return tuple(specs)

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
    ``scan_done``       0 while running, 1 once the final point has been published

    Nothing DERIVABLE is published: the panel namespace already carries the global
    ``shot`` counter, and a site-grid view of the per-site vector is a reshape EXPRESSION
    (``value = <y_key>_sites.reshape(ny, nx)``) -- so neither is a separate signal.

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
        # GUI flow legend / signal namespace: the measurement's slug (its prefix, e.g.
        # ``temperature``) -- the SAME token its hub signals carry -- not the raw y key.
        self.node_label = str(prefix).rstrip("_") or str(y_key)
        # Full hub names of the scan's x axis + y curve.  A 1d plot wired to the y curve
        # resolves its companion x signal (and that x's axis label/unit) from THIS node, so
        # the curve is drawn vs the swept parameter with the right x-axis -- one signal pick.
        self.x_signal = self.prefix + self.x_key
        self.y_signal = self.prefix + self.y_key
        # The measurement owns the swept values (single source of truth); they are the
        # x AXIS, known UP FRONT.  Mirroring Confocal_GUIv2's BaseMeasurement: the curve
        # is a PRE-ALLOCATED ``np.full((n_points, n_series), nan)`` array filled IN PLACE
        # by scan index -- NEVER an append-and-grow list.  So the published curve has its
        # FINAL length from the first shot (a stable x axis; unmeasured points are NaN
        # gaps that fill in), exactly like the reference measurement.
        self._values = np.asarray(measurement.axis.values, dtype=float).reshape(-1)
        self._index = 0
        n_series = max(1, int(getattr(measurement.reducer, "n_series", 1)))
        self.data_y = np.full((self._values.size, n_series), np.nan, dtype=float)

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
        """Measure ONE scan point and FILL it into the pre-allocated curve by index.

        Publishes the FULL-LENGTH x axis + curve every shot (NaN where not yet
        measured), so the plot shows the final x range from the first point and fills
        in -- never a growing array.  Raises ``StopIteration`` if called after the sweep
        is finished (the ``start()`` loop self-stops; tests check ``finished`` first).
        """

        if self.finished:
            raise StopIteration("ScannedMeasurementNode: scan already complete.")
        index = self._index
        value = float(self._values[index])
        row = np.atleast_1d(np.asarray(self.measurement.measure(value, index), dtype=float))
        self.data_y[index, :row.size] = row          # fill IN PLACE by scan index (no append)
        self._index += 1

        out: dict[str, object] = {
            self.x_key: self._values.copy(),           # the FULL x axis, stable from shot 1
            self.y_key: self.data_y[:, 0].copy(),      # full-length curve; NaN = not-yet-measured
            "scan_done": 1.0 if self.finished else 0.0,
        }
        if self.data_y.shape[1] > 1:
            # A per-site reducer: publish the LATEST point's per-site VECTOR alongside the
            # scalar curve.  A site-grid view is a reshape expression on this vector
            # (``value = <y_key>_sites.reshape(ny, nx)``), not a separate signal.
            out[self.y_key + "_sites"] = self.data_y[index, :].copy()
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

    @property
    def _per_site(self) -> bool:
        return self.data_y.shape[1] > 1

    def published_signals(self) -> frozenset:
        keys = [self.x_key, self.y_key, "scan_done"]
        if self._per_site:                              # only a per-site reducer emits a vector
            keys.append(self.y_key + "_sites")
        return frozenset(self.prefix + key for key in keys)

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """x / y axis labels + units come from the swept measurement itself -- the
        scan AXIS (``axis.label``/``axis.unit``) for x and the REDUCER labels for the
        curve -- so a plot of this node reads its axes from the measurement, not a
        hard-coded string."""
        p = self.prefix
        axis = self.measurement.axis
        rlabels = tuple(self.measurement.reducer.labels)          # (xlabel, ylabel, zlabel)
        ylabel = rlabels[1] if len(rlabels) > 1 else self.y_key
        xlabel = str(getattr(axis, "label", "x"))
        xunit = str(getattr(axis, "unit", ""))
        specs = [
            SignalSpec(p + self.x_key, xlabel, xunit, "scan x axis (the swept parameter)"),
            SignalSpec(p + self.y_key, ylabel, "", "measured curve vs the scan x axis"),
            SignalSpec(p + "scan_done", "scan complete", "", "1 once the final point is measured"),
        ]
        if self._per_site:
            specs.insert(2, SignalSpec(p + self.y_key + "_sites", ylabel, "", "latest scan point, per site"))
        return tuple(specs)


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
    "describe_shape",
    "SignalSpec",
    "CalibrateReadoutTask",
    "CameraMeasurement",
    "OccupancyProcessor",
    "Measurement",
    "Processor",
    "ProcessorRun",
    "LogicNode",
    "ScannedMeasurementNode",
    "Task",
    "TaskOutput",
]
