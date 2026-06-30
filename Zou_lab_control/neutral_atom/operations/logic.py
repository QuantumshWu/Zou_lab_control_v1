"""Experiment logic nodes: loops that publish per-shot signals into a SignalHub.

A logic node is the upstream half of the task-console contract (the console is the
consumer).  There are three KINDs, all sharing the :class:`LogicNode` loop:

* :class:`Measurement` -- drives a device acquisition loop and publishes named
  signals (e.g. a camera :class:`CameraMeasurement` publishing one ``frame_i`` per
  emCCD event, or a swept :class:`ScannedMeasurementNode`);
* :class:`Processor` -- a reactive TRANSFORM node with no acquisition of its own
  (the "func" layer): it consumes hub signals and republishes derived ones, e.g.
  :class:`OccupancyProcessor` running the SAME ``calibration.detect`` contract the
  real readout uses, frame_0 -> occupancy/counts/rate;
* :class:`Task` -- a one-shot orchestration (e.g. :class:`CalibrateReadoutTask`,
  which produces a ``TrapCalibration`` + an npz artifact and streams its template
  frames to a mid-run output panel).

The loading readout is COMPOSED by the user from these primitives -- a camera
Measurement publishing ``frame_0`` + an OccupancyProcessor turning ``frame_0`` into
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

from Zou_lab_control._paths import CALIBRATION_DIR
from ..core.analysis import grid_shape_tuple
from ..core.signals import NO_LINEAGE, SignalHub
from ..devices.base import CameraDevice, arm_then_fire
from .signal_expr import DEFAULT_SOURCE, SignalExpr, hub_namespace


@dataclass(frozen=True)
class SignalSpec:
    """What ONE output of a logic node MEANS -- the human label + unit + one-line
    description for a signal it publishes (or, for a task, produces off-hub).

    A node declares these ONCE (``output_specs``); the GUI reads them so a plot can
    set its axis label/unit from the producing measurement (not a hard-coded per-kind
    string) and a node's "publishes" legend reads as ``occupied  (35,)  per-site 0/1
    occupancy`` -- every output named, shaped and explained.  ``name`` is the FULL hub
    signal name (with the node's prefix), so a consumer maps a signal straight to its
    meaning.

    ``points_shape`` / ``data_shape`` declare THIS signal's OWN slot in the
    ``(repeat, *points_shape, *data_shape)`` contract -- because one node publishes signals
    of DIFFERENT structure (occupancy judges a frame's repeat axis into a per-site vector,
    while its ``centers`` is static geometry with NO repeat axis).  Both ``None`` (the
    default) means "this signal does NOT follow the repeat contract" -- a consumer prints its
    raw shape and falls back to the node-level points/data triple.  A signal that DOES carry
    the repeat axis declares ``points_shape``/``data_shape`` so ``core_ndim =
    len(points_shape) + len(data_shape)`` tells the plot exactly which leading axis is the
    repeat to collapse -- structure-driven, never an ndim guess."""

    name: str               # full published signal name (incl. the node's prefix)
    label: str              # axis / legend label, e.g. "loading rate"
    unit: str = ""          # physical unit, e.g. "s" / "K" (blank = dimensionless)
    description: str = ""    # one-line human meaning for the publishes legend
    points_shape: tuple | None = None   # this signal's swept-parameter (data_points) axes (None = no contract slot)
    data_shape: tuple | None = None     # this signal's per-point data axes (None = no contract slot)

    def __post_init__(self):
        # IRON-LAW guard: a signal that follows the repeat contract is ALWAYS (repeat, data_points, *data_dim)
        # with the data_points axis PRESENT (a no-scan acquisition = exactly ONE point -> points_shape=(1,)).
        # The empty tuple () means "repeat-carrying but ZERO points" -- a contradiction that historically
        # crept in (dropping the middle 1) and broke the uniform shape.  points_shape is therefore EITHER
        # None (no repeat contract: a static constant / a derived scalar) OR a NON-EMPTY tuple.  data_shape
        # follows the same rule so a declared contract slot is never half-specified.
        if self.points_shape == () or self.data_shape == ():
            raise ValueError(
                f"SignalSpec({self.name!r}): points_shape/data_shape must be None (no repeat contract) or a "
                f"NON-EMPTY tuple -- got points_shape={self.points_shape!r}, data_shape={self.data_shape!r}.  "
                "A no-scan, single-point signal is (repeat, 1, *data) with points_shape=(1,); the empty () "
                "drops the mandatory data_points axis (#iron-law).")

    @property
    def axis_label(self) -> str:
        """``label (unit)`` for a plot axis, or just ``label`` when dimensionless."""
        return f"{self.label} ({self.unit})" if self.unit else self.label

    @property
    def has_structure(self) -> bool:
        """True when this signal declares its OWN repeat-contract slot (points/data shape),
        so a consumer reads the per-signal structure instead of the node-level triple."""
        return self.points_shape is not None or self.data_shape is not None

    @property
    def core_ndim(self) -> int | None:
        """``len(points_shape) + len(data_shape)`` -- the dimensionality of ONE repeat slice of
        this signal, so ``reduce_repeat`` knows the block carries a repeat axis exactly when its
        ndim is ``1 + core_ndim``.  ``None`` when the signal declares no structure."""
        if not self.has_structure:
            return None
        return len(self.points_shape or ()) + len(self.data_shape or ())


def grid_for_points(grid, points) -> tuple:
    """The SINGLE validity precondition for un-flattening a flat swept-points axis into a 2-D map.

    A 2-D scan publishes its swept points FLAT (``points_shape=(P,)``) and declares ``grid_shape=(n0,n1)``
    so a 2-D panel can reshape them back into an image.  That reshape is valid IFF the points exist AND the
    grid divides them exactly -- ``points`` non-empty and ``prod(grid) == prod(points)``.  Returns ``grid``
    when valid, else ``()``.

    This is the ONE rule EVERY consumer shares -- :func:`describe_shape` (the legend), the console's
    ``_signal_structure`` (what a panel reshapes) and ``_coerce`` (the actual reshape) -- so the rule can
    never drift between them.  Crucially, a node may carry a ``grid_shape`` that is NOT a swept-points grid:
    an :class:`OccupancyProcessor` stores ``grid_shape=(5,7)`` as the physical TRAP LAYOUT,
    which has nothing to do with reshaping points.  Because reshape eligibility is a dimensional FACT about
    the signal (``prod(grid)==prod(points)``), such a trap grid is never smeared onto a per-site signal
    (occupancy ``points=()`` -> ``prod=1`` != ``35``) and imshow'd as a (5x7) heatmap -- no per-node
    special-casing, the same fact decides it everywhere."""
    pts = tuple(int(n) for n in (points or ()))
    g = tuple(int(n) for n in (grid or ()))
    if pts and g and int(np.prod(g)) == int(np.prod(pts)):
        return g
    return ()


def describe_shape(value, *, points_shape=None, data_shape=None, grid_shape=None) -> str:
    """A standardized shape string read straight from a published VALUE -- the SINGLE
    way the GUI says what a signal looks like, AUTO-EXTRACTED from real data rather
    than a hand-typed name->format map (which silently drifts from what a node really
    emits).  ``scalar`` for a 0-d / Python number; ``None`` -> ``"—"`` (no value yet).

    When the value IS a measurement/processor's contract block (its shape matches the declared
    ``(repeat, *points_shape, *data_shape)``, #H3o) it is shown in CONTRACT form
    ``repeat × points × (data)`` -- ALWAYS all three groups (the data_points axis is mandatory, never
    dropped): a 1-D scan ``5 × 8 × (3)``; a 2-D scan ``5 × (4×5) × (1)`` via ``grid_shape`` reshaping the
    SWEPT POINTS; a no-scan single-point signal is ``5 × 1 × (...)`` -- a camera/judged frame
    ``5 × 1 × (96×128)``, per-site occupancy ``5 × 1 × (35)``.  ``grid_shape`` is ONLY a 2-D SCAN's
    points reshape -- it is NEVER applied to the DATA (the 35 sites read ``(35)``, not ``5×7``: that
    layout is the sitemap's display concern, #H3v-3).  Otherwise the raw numpy shape (``(35,)`` /
    ``(96, 128)``) -- e.g. static ``centers`` (35, 2) carries no repeat axis."""
    if value is None:
        return "—"
    shape = tuple(int(n) for n in np.shape(value))
    if shape == ():
        return "scalar"
    ps = tuple(int(n) for n in (points_shape or ()))
    dsh = tuple(int(n) for n in (data_shape or ()))
    if (points_shape is not None or data_shape is not None) and len(shape) >= 1 \
            and tuple(shape[1:]) == ps + dsh:
        gs = grid_for_points(grid_shape, ps)   # ONE rule: a grid shows only when it divides the points
        dstr = "×".join(str(n) for n in dsh) or "1"
        pstr = "×".join(str(n) for n in (gs or ps)) or "1"   # data_points ALWAYS shown (no-scan = 1)
        return f"{shape[0]} × {pstr} × ({dstr})"              # repeat × data_points × (data)
    if len(shape) == 1:                  # numpy 1-D repr keeps the trailing comma: (35,)
        return f"({shape[0]},)"
    return "(" + ", ".join(str(n) for n in shape) + ")"


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
    # UNIFORM measurement output contract (#H3n): an ACQUIRING node publishes its primary data block
    # with shape ``(repeat, *points_shape, *data_shape)`` -- ``points_shape`` is the swept parameter
    # space (a camera = ``(1,)``, a 1-D scan = ``(n_points,)``, a 2-D scan = ``(n0*n1,)``) and
    # ``data_shape`` is the per-point data (a scan scalar = ``(dim,)``, a camera frame = ``(H, W)``).
    # Defaults are empty (a processor / task publishes no such block); camera + scan nodes set them,
    # and ``tests/test_measurement_output_contract.py`` MECHANICALLY enforces the published shape.
    points_shape: tuple = ()
    data_shape: tuple = ()
    # The INTENDED 2-D display geometry of the points axis when it is a flattened grid -- a 2-D scan
    # declares ``grid_shape=(n0, n1)`` (prod == points_shape[0]) so a 2-D panel can reshape the
    # flattened points back into an image, since ``data_shape`` stays ``(1,)`` for a scan (#H3o).
    # Empty for a camera (its image is in ``data_shape``) and a 1-D scan.
    grid_shape: tuple = ()

    def __init__(self, hub: SignalHub, *, prefix: str = ""):
        self.hub = hub
        self.prefix = str(prefix)
        # Per-INSTANCE human name (the console sets it from the node's row title), so two
        # nodes of the SAME kind (e.g. two occupancy judges) are told apart in the Logic
        # rows, the source combobox and every legend.  Blank -> fall back to the LAYER
        # node_label (camera / occupancy / ...); never the Python class name.
        self.instance_label = ""
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
        # The SOURCE-shot id the NEXT publish belongs to (set within shot(): an acquiring node mints a fresh
        # one after it acquires; a reactive Processor inherits its consumed input's).  step() reads it, tags
        # the publish, then clears it -- so a no-op / frozen shot publishes nothing with a stale id.  Set and
        # read on the node's OWN thread only (shot()->step() are same-thread), so no extra lock (#shot-clock).
        self._current_source_shot: int | None = None

    @property
    def display_label(self) -> str:
        """Short human name for this logic node in the GUI -- a per-instance
        ``instance_label`` (set from the node's row title) if given, else its LAYER node
        name (``camera`` / ``occupancy`` / ``calibrate`` / a measurement's curve), NEVER the
        Python class name.  The instance label lets two same-kind nodes (e.g. two occupancy
        judges) be told apart; the hub prefix is a signal-namespacing detail, not the label
        (the namespaced signal names shown alongside also disambiguate A/B)."""
        return str(self.instance_label or self.node_label)

    # ------------------------------------------------------------------ loop
    def shot(self) -> dict[str, object]:  # pragma: no cover - abstract
        """Produce one shot's worth of {name: value} signals."""
        raise NotImplementedError

    def _assert_declared(self, out: dict, declared) -> dict:
        """Publish-time conformance, enforced for EVERY ``shot()``-publisher on the common
        ancestor (#E2): a node may publish ONLY signals it declared (its bare output / result
        keys, before ``step`` prefixes them).  An undeclared key would become a silent,
        unlegended hub signal the flow graph never shows, so fail loud at the boundary instead
        of letting one node type carry the guard while a sibling silently drops it."""
        extra = set(out) - set(declared)
        if extra:
            raise ValueError(
                f"{type(self).__name__} published undeclared signal(s) {sorted(extra)}; "
                "declare them in the node's output keys (the single output-key source).")
        return out

    def step(self) -> dict[str, object]:
        """Run ONE shot synchronously and publish it (test/notebook friendly).

        A shot that returns an EMPTY mapping is a no-op: nothing is published and the
        shot counter does not advance.  This lets a REACTIVE logic node (a
        :class:`Processor` that transforms another signal) tick at its own rate yet
        only emit when its input actually advanced."""
        values = self.shot()
        if not values:
            return {}
        named = {self.prefix + key: value for key, value in values.items()}
        # Tag every signal of this shot with the SOURCE-shot id shot() set (mint for an acquiring node,
        # inherited for a reactive processor; None -> NO_LINEAGE for a free-running publish), so a derived
        # signal and the frame it came from share ONE id and the console can show them as one shot.  Clear
        # after, so a later no-op/frozen shot never re-uses a stale id (#shot-clock).
        self.hub.publish(named, provenance=self._current_source_shot)
        self._current_source_shot = None
        self.shots += 1
        return named

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
                    # A wedged source must not kill the daemon silently mid-run, but it must NOT fail
                    # silently either: record the error + a running count ON THE INSTANCE.  The console's
                    # _update_summary reads these (last_error / consecutive_errors) EVERY tick (timer-driven,
                    # not version-gated) and raises the banner -- so the error is surfaced WITHOUT putting a
                    # synthetic "node_error" value on the hub (which had no reader, yet showed up as an
                    # "(unbound)" signal in every picker, #5).
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.consecutive_errors += 1
                    self._stop.wait(period)
                    continue
                if self.consecutive_errors:
                    # Recovered: clear the banner so a transient hiccup doesn't stick.
                    self.last_error = None
                    self.consecutive_errors = 0
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
#   Measurement -- ACQUIRES from devices (camera/sequencer); OWNS the repeat axis (fills a block).
#   Processor   -- TRANSFORMS hub signals into derived signals (no acquisition).
#   Task        -- ORCHESTRATES the others over a multi-step flow, with mid-run output.
class Measurement(LogicNode):
    """A logic node that ACQUIRES data from devices and publishes named signals.

    A Measurement OWNS the repeat axis: each concrete measurement FILLS a
    ``(repeat, *points_shape, *data_shape)`` BLOCK every shot (the camera's depth-``repeat``
    ring, a scan's raw block) and publishes it whole.  It does NOT collapse the repeat axis --
    HOW the repeats are combined for viewing (average / add / roll / create) is the PLOT's
    ``repeat_mode`` (display-only), the SINGLE place a repeat axis is collapsed.  So there are
    exactly two repeat knobs in the whole pipeline: ``repeat`` here (how many shots to keep, 0 = ∞
    rolls forever) and the plot's ``repeat_mode`` (how to show them).  Concrete measurements
    implement ``shot()``; ``repeat`` is the ONE acquisition param auto-injected by the console."""

    layer = "measurement"
    node_label = "measurement"
    # The published key that carries the (repeat,*points_shape,*data_shape) CONTRACT block.  Each
    # concrete measurement sets it (camera -> "frame_0"; a scan -> its y_key) so the base can verify the
    # block shape at publish time and the output-contract test can find it generically.
    primary_signal: str = ""

    def _assert_primary_shape(self, out: dict) -> dict:
        """Publish-time contract guard: the primary block MUST be ``(repeat, *points_shape, *data_shape)``.

        A new measurement that mis-sizes its block (forgets the repeat axis, sets only one of
        points/data shape, publishes an un-repeated 2-D array) fails LOUD here instead of silently
        producing a wrong plot.  Returns ``out`` so a subclass can ``return self._assert_primary_shape(out)``."""

        key = self.primary_signal
        if key and key in out and out[key] is not None:
            block = np.asarray(out[key])
            # IRON-LAW guard: a measurement that publishes a primary block ALWAYS has the data_points
            # axis -- a no-scan acquisition is exactly ONE point (points_shape=(1,)), never the empty ()
            # that drops the axis.  Catch the omission LOUD here so a new measurement cannot silently
            # publish a (repeat, *data) block missing data_points (the drift the iron law forbids).
            if not tuple(self.points_shape):
                raise ValueError(
                    f"{type(self).__name__} publishes primary block {key!r} but points_shape is empty: every "
                    "measurement block is (repeat, data_points, *data_dim) with the data_points axis PRESENT "
                    "-- set points_shape=(1,) for a no-scan acquisition, or (n_points,) for a scan (#iron-law).")
            # The block's first axis is the RING DEPTH actually published = ``max(1, repeat)`` (a
            # finite run keeps ``repeat`` slices; ``repeat=0`` = ∞ keeps a 1-deep rolling ring), NOT
            # the raw ``repeat`` -- so 0 = ∞ does not false-trip this contract guard.
            # Resolve the ring depth defensively: a new measurement subclass that sets primary_signal
            # but forgot _ring/repeat should still get this guard's CLEAR shape message, not a cryptic
            # AttributeError from an eagerly-evaluated default (getattr's default arg always evaluates).
            ring = int(getattr(self, "_ring", 0)) or max(1, int(getattr(self, "repeat", 1)))
            expected = (ring, *tuple(self.points_shape), *tuple(self.data_shape))
            if block.shape != expected:
                raise ValueError(
                    f"{type(self).__name__} primary block {key!r} has shape {block.shape}, but the "
                    f"measurement output contract requires {expected} = (ring, *points_shape, *data_shape) "
                    "with ring = max(1, repeat); set points_shape/data_shape (and fill the ring axis) to "
                    "match what you publish.")
        return out


class Processor(LogicNode):
    """A logic node that TRANSFORMS hub signals into derived signals (the "func" layer).

    It consumes one or more named signals, computes, and publishes -- with NO device
    acquisition of its own.  REACTIVE: it only emits when a consumed signal advanced
    since the last tick (tracked via the hub's per-signal version), so it runs as a
    live graph node beside the measurement that produces its input, at its own poll
    rate, and no-ops (``shot`` returns ``{}``) when there is nothing new.

    A processor is a PURE TYPED TRANSFORM with NO user-facing mode (that would be a third
    repeat knob on top of the measurement's ``repeat`` (0 = ∞) and the plot's
    ``repeat_mode`` -- the tangle we deliberately do not have).  Its relationship to the repeat
    axis is a STATIC class fact, ``repeat_contract``, NOT a runtime knob and NEVER shown in a form:
      * ``"reduce"`` (default, the common case): emits derived signals that carry NO repeat axis
        (a per-shot judgement / a statistic over a shot set).  There is nothing left for the plot
        to collapse, so it never collides with ``repeat_mode``.
      * ``"preserve"``: maps each repeat slice 1:1 and emits a >=3-D block whose axis 0 IS the
        repeat, so the SAME plot ``reduce_repeat`` machinery collapses it (a future per-slice
        image filter -- no console instance yet).
    Enforced by ``tests/test_processor_repeat_contract.py``."""

    layer = "processor"
    node_label = "processor"
    provides: tuple[str, ...] = ()
    repeat_contract = "reduce"   # see class docstring; static, never a user knob
    # Provenance inheritance (#shot-clock): a single-input processor inherits its input's source-shot id so
    # its derived signals group with that frame.  With MULTIPLE inputs, by default the processor no-ops until
    # they share one id (they are not one coherent shot); set True to instead fuse them at max(ids) -- only
    # for a processor that deliberately combines several acquisitions into one derived result.
    fuse_across_shots = False

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
            inputs = {n: self.hub.latest(n) for n in self.consumes}
        except KeyError:
            return None
        # INHERIT the consumed input's source-shot id so the derived signals (occupied / frame_judged)
        # carry the SAME id as the frame they were computed from -> the console groups them into one
        # display shot (#shot-clock).  Single input -> that id.  Multiple inputs from DIFFERENT shots are
        # not one coherent acquisition -> no-op until they line up (or max-fuse if fuse_across_shots).
        ids = {self.hub.latest_provenance(n) for n in self.consumes}
        ids.discard(NO_LINEAGE)
        if len(ids) == 1:
            self._current_source_shot = next(iter(ids))
        elif not ids:
            self._current_source_shot = None                  # inputs carry no lineage -> derived is NO_LINEAGE too
        elif self.fuse_across_shots:
            self._current_source_shot = max(ids)
        else:
            return None                                       # different acquisitions -> wait for one coherent shot
        return inputs

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:  # pragma: no cover - abstract
        raise NotImplementedError

    def output_keys(self) -> tuple[str, ...]:
        """The bare signal names this processor publishes -- the SINGLE source.

        ``provides`` (a class fact) IS this declaration; ``published_signals`` (the
        prefixed hub names) and the spec's ``result_keys`` both derive from it, so the
        output names are typed ONCE.  A subclass with dynamic outputs overrides this."""

        return tuple(self.provides)

    def shot(self) -> dict[str, object]:
        inputs = self.new_inputs()
        if inputs is None:
            return {}
        out = self.transform(inputs)
        return self._assert_declared(out, self.output_keys())

    def published_signals(self) -> frozenset:
        return frozenset(self.prefix + key for key in self.output_keys())


class OccupancyProcessor(Processor):
    """Per-frame atom detection as a live graph node -- the REAL readout pipeline.

    Consumes a camera ``frame`` BLOCK and runs the SAME ``calibration.detect``
    contract the notebook/real readout uses, judging EACH repeat slice.  THIS is the
    virtual==real split: the camera produces frames (a Measurement); detection is a
    SEPARATE node here -- not one node fabricating every signal.  The calibration (site
    centers + per-site thresholds) comes from a prior calibrate-readout Task, exactly
    as on real hardware.

    ``repeat_contract == "preserve"`` (#H3q): the repeat axis flows THROUGH the processor, and so
    does the UNIFORM ``(repeat, data_points, *data_dim)`` shape -- occupancy judges ONE readout per
    shot (no parameter sweep), so data_points = 1, the SAME single-point axis the camera frame carries.
    Fed the camera's ``(repeat, 1, H, W)`` block it judges every slice and publishes ``occupied`` /
    ``counts`` as ``(repeat, 1, n_sites)`` (points_shape=(1,), data_shape=(n_sites,)) and
    ``frame_judged`` as the SAME ``(repeat, 1, H, W)`` it was fed (a frame keeps ONE shape raw or
    judged).  Each repeat-collapse is STRUCTURE-driven, not an ndim guess: the signal's declared
    ``core_ndim = len(points)+len(data)`` tells the plot that axis 0 is the repeat
    (``ndim == 1 + core_ndim``), so a sites/2d panel with ``repeat_mode=average`` over ``occupied``
    averages ``(repeat, 1, n_sites) -> (1, n_sites) -> (n_sites,)`` = the per-site LOADING PROBABILITY
    (averaging N shots recovers every ~50%-loaded site) -- ONE mechanism (the plot's repeat collapse),
    not a private in-node accumulator.  The user sets
    how many shots to average via the camera's ``repeat``.  ``centers`` (N, 2) and
    ``thresholds`` (N,) are STATIC calibration geometry -- they declare NO contract slot (their
    SignalSpec leaves points/data ``None``), so they print their raw shape and carry no repeat
    axis a consumer could mistake for one."""

    node_label = "occupancy"
    repeat_contract = "preserve"        # judges each repeat slice -> a clean (repeat, n_sites) block
    points_shape: tuple = ()            # one frame per shot sweeps no parameter (n_sites is the data)
    # ``frame_judged`` = the EXACT block this occupancy was computed from, republished so
    # the site map's underlay is the SAME shots as the rings (the camera keeps streaming
    # newer frames on its own thread; using the live camera frame would offset the rings).
    # ``rate`` (scalar) = the loading fraction of THIS block (mean occupancy over its sites x
    # shots) -- a single number per tick, so a Rolling-trace monitor draws the loading rate vs
    # time and a pulse scan reads it as the swept y.  Per-site loading PROBABILITY is NOT a
    # separate signal: it is ``repeat_mode=average`` over ``occupied`` (and the 2-D site map via
    # grid_shape) -- one mechanism, not a duplicated in-node accumulator.
    provides = ("occupied", "counts", "rate", "centers", "thresholds", "frame_judged")
    # The site map takes ONE signal (an occupancy vector this node publishes) and resolves
    # its ring CENTRES + frame UNDERLAY from the SAME node: these name the two outputs that
    # carry them.  THIS is the single source -- the panel layer (ProcessorSpec.metadata) and
    # the console's site-map resolver both read these, so "one signal" wiring never drifts.
    sitemap_centers_key = "centers"
    sitemap_image_key = "frame_judged"

    def __init__(self, hub: SignalHub, *, calibration=None, calibration_source=None,
                 session_calibration=None, source_expr=None,
                 grid_shape: tuple[int, int] | None = None,
                 method: str | None = None, prefix: str = ""):
        # The frame to judge is a signal expression -- the SAME universal multi-slot signal +
        # ``value = ...`` mechanism every source field uses (default = the camera's first emCCD
        # event, ``frame_0``).  ``consumes`` (what makes the node reactive) is the picked input
        # names, so the node re-judges when any of them advances.  Its ``value`` must evaluate to
        # ONE (H×W) frame; an empty pick falls back to ``frame_0``.
        expr = source_expr if isinstance(source_expr, SignalExpr) else SignalExpr.from_value(source_expr)
        if not expr.inputs:
            expr = SignalExpr(["frame_0"], DEFAULT_SOURCE)
        super().__init__(hub, consumes=tuple(expr.inputs), prefix=prefix)
        self.source_expr = expr
        self.calibration = calibration
        # Optional lazy source: a callable -> calibration (or None while a calibrate
        # task is still running on its own thread).  Lets the live readout stream
        # WITHOUT blocking the GUI on calibration -- the detector simply no-ops until
        # the calibration is ready, then picks it up.
        self.calibration_source = calibration_source
        # Optional getter for the LIVE session calibration (built from THIS camera's ROI, so it
        # matches the live frame).  Used as a fallback when the loaded FILE calibration's ROI does
        # NOT match the frame (a stale calibration.json from a different camera shape): rather than
        # raise every shot and wedge the whole readout, the node switches to the matching session
        # calibration so occupancy / rate keep flowing.
        self.session_calibration = session_calibration
        self.grid_shape = None if grid_shape is None else grid_shape_tuple(grid_shape)
        # The READOUT method (box / per-site PSF / ...) is chosen HERE, not at calibration
        # time: one calibration carries every method's geometry + thresholds, and the
        # processor picks which to read with (None = the calibration's default).
        self.method = None if method in (None, "") else str(method)
        # The per-site DATA width -- set on the first judged block so structure_provider can
        # drive the plot's reshape (the (repeat, n_sites) contract).  () until the first shot.
        self.data_shape: tuple = ()
        # The judged frame's (H, W) -- the data_shape of the ``frame_judged`` signal's per-signal
        # structure, set on the first judged block (() until then).
        self.frame_shape: tuple = ()

    def _resolve_calibration(self):
        if self.calibration is None and self.calibration_source is not None:
            self.calibration = self.calibration_source()
        return self.calibration

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        calibration = self._resolve_calibration()
        if calibration is None:
            return {}                                       # not calibrated yet -> no-op (non-blocking)
        # The frame to judge = the source expression over the consumed signals (default
        # ``value = signal`` on one input IS that frame; an expression may combine several,
        # e.g. ``value = (signal[0] + signal[1]) / 2``).  The value must be ONE (H×W) frame.
        try:
            value = np.asarray(self.source_expr.evaluate(inputs), dtype=float)
        except Exception:
            return {}                                       # malformed expression -> no-op (don't wedge)
        # Normalize whatever the source expression yields to a (repeat, 1, H, W) BLOCK -- the camera's
        # own ``(repeat,1,H,W)`` block, a bare ``(H,W)`` frame, or a ``(repeat,H,W)`` stack all become
        # one uniform block.  Occupancy is then judged PER repeat slice so the repeat axis flows
        # THROUGH the node (repeat_contract='preserve'): a sites/2d panel's repeat_mode=average over
        # ``occupied`` IS the per-site loading probability (averaging N shots recovers every site).
        if value.ndim == 2:
            block = value[None, None]                       # (1, 1, H, W)
        elif value.ndim == 3:
            block = value[:, None]                          # (repeat, H, W) -> (repeat, 1, H, W)
        elif value.ndim == 4:
            block = value
        else:
            return {}                                       # not an image / image block -> no-op

        occ_rows: list = []
        cnt_rows: list = []
        n_sites = None
        thresholds = None
        centers = None
        for r in range(int(block.shape[0])):
            img = np.asarray(block[r, 0], dtype=float)
            if not np.isfinite(img).any():
                occ_rows.append(None); cnt_rows.append(None); continue   # unfilled ring slice
            try:
                detection = calibration.detect(img, method=self.method)  # the single readout contract
            except ValueError as exc:
                # The loaded calibration does not fit this frame (e.g. a calibration.json from a
                # different camera ROI/shape, or one that lacks the chosen readout method).  Try the
                # live SESSION calibration (built from THIS camera, so it matches); if there is none
                # that fits, RAISE a clear text error -- the node surfaces it like a signal-size
                # mismatch (red node-error banner) instead of silently producing no data forever.
                fallback = self.session_calibration() if callable(self.session_calibration) else None
                if fallback is None or fallback is calibration:
                    raise ValueError(
                        f"Judge occupancy: the loaded calibration does not fit this frame "
                        f"(frame {img.shape}, method {self.method or 'default'}): {exc}.  It is likely "
                        f"a calibration for a different camera ROI/shape or missing this method -- "
                        f"recalibrate for THIS camera, or pick the matching calibration file.") from exc
                detection = fallback.detect(img, method=self.method)
                self.calibration = calibration = fallback     # adopt the matching one for next shots
            occ_rows.append(np.asarray(detection.occupied, dtype=float).reshape(-1))
            cnt_rows.append(np.asarray(detection.counts, dtype=float).reshape(-1))
            n_sites = occ_rows[-1].size
            thresholds = np.asarray(detection.thresholds, dtype=float).reshape(-1)
            centers = np.asarray(calibration.centers, dtype=float)
        if n_sites is None:
            return {}                                       # no filled slice judged this tick -> no-op
        nan = np.full(int(n_sites), np.nan)
        # Every per-shot BLOCK obeys the UNIFORM contract (repeat, data_points, *data_dim): occupancy
        # judges ONE readout per shot (no parameter sweep), so data_points = 1 -- the SAME single-point
        # axis the camera frame carries.  occupied/counts are (repeat, 1, n_sites); frame_judged is the
        # camera block (repeat, 1, H, W) unchanged.  This middle 1 is NEVER dropped: a frame keeps one
        # shape raw or judged, and occupancy/sites/scan/camera all read as (repeat, points, data).
        # unfilled slices are NaN so the plot's nanmean ignores them.
        occupied = np.stack([o if o is not None else nan for o in occ_rows], axis=0)[:, None, :]
        counts = np.stack([c if c is not None else nan for c in cnt_rows], axis=0)[:, None, :]
        self.data_shape = (int(n_sites),)                   # the per-site DATA width (points axis is the (1,))
        self.frame_shape = (int(block.shape[2]), int(block.shape[3]))   # (H, W) of frame_judged
        return {
            "occupied": occupied,                           # (repeat, 1, n_sites) per-shot occupancy
            "counts": counts,                               # (repeat, 1, n_sites) per-shot readout signal
            "rate": float(np.nanmean(occupied)),            # per-cycle loading fraction (derived scalar, no repeat)
            "centers": centers,                             # (N, 2) static site geometry -- no repeat axis
            "thresholds": thresholds,                       # (N,) calibration constant -- no repeat axis
            # the judged BLOCK, published ATOMICALLY with the occupancy -> the site map's underlay +
            # rings are always the SAME shots (the site-map path reduces it to one (H,W) underlay).
            "frame_judged": block,                          # (repeat, 1, H, W) -- the SAME single-point frame
        }

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """Label + meaning of each detection signal (the readout pipeline's outputs), each with its
        OWN repeat-contract structure (#H3s-F3) -- so a consumer reads each signal's points/data slot,
        not one node-level triple that can only be right for one of them:

          * ``occupied`` / ``counts`` -- ``(repeat, 1, n_sites)``: points=(1,), data=(n_sites,) (core_ndim 2)
            -- one judged readout per shot (data_points=1, never dropped), n_sites per point.
          * ``frame_judged`` -- ``(repeat, 1, H, W)``: points=(1,), data=(H, W) (core_ndim 3) -- the SAME
            shape the camera frame carries (a frame keeps ONE shape raw or judged).
          * ``rate`` -- a per-cycle loading fraction, a DERIVED SCALAR (already reduced over the block),
            NOT a per-shot block: points/data ``None`` (no repeat contract), consumed by a rolling monitor.
          * ``centers`` / ``thresholds`` -- STATIC geometry with NO repeat axis: they leave points/data
            ``None`` (no contract slot), so a consumer prints their raw shape ``(N, 2)`` / ``(N,)``."""
        p = self.prefix
        # (n_sites,) / (H, W) once judged, else None -- BEFORE the first judge the data width is unknown,
        # so the signal declares NO contract slot yet (None, not the illegal empty ()); once judged it is
        # the uniform (repeat, 1, *data) block.  points_shape pairs with data_shape: both known or both None.
        sites = tuple(self.data_shape) if self.data_shape else None
        frame = tuple(self.frame_shape) if self.frame_shape else None
        return (
            SignalSpec(p + "occupied", "occupancy", "", "per-site per-shot occupancy (0 / 1); average = loading probability",
                       points_shape=((1,) if sites else None), data_shape=sites),
            SignalSpec(p + "counts", "readout counts", "", "per-site per-shot integrated readout signal",
                       points_shape=((1,) if sites else None), data_shape=sites),
            SignalSpec(p + "rate", "loading rate", "", "loading fraction of this block (scalar) -> rolling-trace monitor / scan y"),
            SignalSpec(p + "centers", "site centre", "px", "site centres in camera pixels (N, 2)"),
            SignalSpec(p + "thresholds", "threshold", "counts", "per-site bright/dark count threshold"),
            SignalSpec(p + "frame_judged", "camera image", "counts",
                       "the EXACT frame this occupancy was judged from -- shot-locked to occupied/centers "
                       "(one atomic publish).  Use THIS for a 2D readout image that matches the site map "
                       "(same shot); the camera's live `frame` advances independently and is NOT shot-locked.",
                       points_shape=((1,) if frame else None), data_shape=frame),
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
        # One-shot: stop the loop AFTER this run (a run() that raises is NOT retried),
        # exactly like a finite scan / processor.  ``finished`` means "this one-shot has
        # RUN ONCE and will not retry" -- success OR failure terminates it -- so BOTH it
        # and the stop event are set in ``finally``.  Keeping them together is what lets
        # the console release its lock on either outcome (a run() that raises otherwise
        # leaves finished=False forever -> dashboard locked, finding 7).  During the run
        # the stop event stays clear and means "cancel": ``run`` can poll ``self._stop``
        # (and pass it to ``camera.acquire``) to interrupt a long acquisition the moment
        # Stop is pressed.  The exception still propagates (finally does not swallow), so a
        # headless step() caller sees it; failure is expressed by ``result`` staying empty.
        # The result + mid-run output stay on the INSTANCE (self.result / self.output); a
        # task publishes NOTHING to the hub -- the hub is measurements + processors only.
        try:
            self.result = {str(key): value for key, value in dict(self.run(self.output)).items()}
        finally:
            self.finished = True
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

    # The imaging pulse TEMPLATE the cali loads -- a REAL, inspectable program that IS the
    # long-short-long bracket (3 emCCD frames in one cooling cycle), not a single window the
    # task secretly unrolls.  A bare name resolves to the shipped ``pulses/`` template; an
    # absolute path to the user's own PulseTableState .json.  Each cali pass LOADS it and sets
    # ONLY the two exposures BY NAME -- API slot a1 = the long reference frame(s), a2 = the
    # short readout -- so what is fired == the template file.  The cali does not choose a
    # readout METHOD: it computes ALL methods (box / per-site PSF / uniform PSF) and the
    # OccupancyProcessor picks one.
    # The ONE canonical default imaging-template path (the cali task spec + the generic
    # Pulse-scan measurement both reference THIS, so every GUI form shows the same real,
    # project-relative ``pulses/imaging_template.json`` -- never a bare name that the path
    # widget would anchor to the project ROOT and display as a non-existent file).
    DEFAULT_PULSE_TEMPLATE = "pulses/imaging_template.json"

    @classmethod
    def _resolve_template(cls, pulse_template):
        """Load the imaging template via the ONE shared resolver (the given path if real, else the
        same-named file shipped under the project ``pulses/`` folder -- where the pulse GUI saves and
        the Browse dialog opens, so the default ``pulses/imaging_template.json`` is a REAL inspectable
        file -- else the in-memory long-short-long default)."""
        from ..timing import default_imaging_template, resolve_pulse_template
        return resolve_pulse_template(pulse_template, default_name=cls.DEFAULT_PULSE_TEMPLATE,
                                      default_factory=default_imaging_template)

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
                 reference_exposure: float = 0.020,
                 readout_exposure: float = 0.005,
                 pulse_template: str = DEFAULT_PULSE_TEMPLATE,
                 threshold_frames: int = 100,
                 threshold_method: str = "otsu",
                 source: str = "live", folder: str = CALIBRATION_DIR,
                 save_frames: bool = True,
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
        # The Rb87 reference bracket comes from the TEMPLATE (a literal long-short-long), and these
        # two operator-set exposures only set its frame DURATIONS by name (exactly as the real rb87
        # readout sets a 20 ms reference + a 5 ms readout): ``reference_exposure`` is the LONG frame
        # (high-SNR -> votes ground truth + builds the site map / PSF; written to API slots ``a1``
        # and ``a3``, the two long reference exposures of the long-short-long bracket) and
        # ``readout_exposure`` is the SHORT readout frame (its per-site thresholds are learnt; API
        # slot ``a2``).  ``_collect_bracket_groups`` does set_api(a1, long) / set_api(a2, short) /
        # set_api(a3, long) -- it changes ONLY those durations, never the structure, so what is
        # fired == the template file.
        self.reference_exposure = float(reference_exposure)
        self.readout_exposure = float(readout_exposure)
        self.pulse_template = str(pulse_template or self.DEFAULT_PULSE_TEMPLATE)
        self.threshold_frames = max(2, int(threshold_frames))
        self.threshold_method = str(threshold_method)
        # ONE folder for input + output (no blank paths); ``source`` decides how it's used.
        # Anchor a relative folder to the PROJECT root (not the volatile CWD) so the data +
        # report land where the GUI field says they do, wherever Python was launched.
        from Zou_lab_control._paths import resolve_under_project
        self.source = str(source)
        self.folder = str(resolve_under_project(folder or "calibrations"))
        # source=live: also SAVE the acquired raw frames to ``folder`` (img1.npy, ...) so a
        # later source="saved frames" run re-calibrates from them without re-acquiring.
        self.save_frames = bool(save_frames)
        self.calibration = None

    def acquisition_parameters(self) -> dict[str, object]:
        """The calibrate task's tunable parameters (source + folder + pulse template +
        exposures + grid + frame counts + threshold), as ``{name: current}`` -- shown in
        the panel's Edit and applied before the next Run.  Every value is concrete (no
        blank): the pulse template + folder read back as their paths.  The cali computes
        every readout method into one calibration; the OccupancyProcessor chooses which to
        read with (box / per-site PSF / uniform PSF), so there is no readout-method param here."""
        return {
            "source": str(self.source),
            "folder": str(self.folder),
            "save_frames": bool(self.save_frames),
            "pulse_template": str(self.pulse_template),
            "grid_shape": tuple(self.grid_shape),
            "reference_exposure": float(self.reference_exposure),
            "readout_exposure": float(self.readout_exposure),
            "roi_radius": int(self.roi_radius),
            "threshold_frames": int(self.threshold_frames),
            "threshold_method": str(self.threshold_method),
        }

    def set_acquisition_parameters(self, **values) -> None:
        if "source" in values:
            self.source = str(values["source"])
        if "folder" in values:
            self.folder = str(values["folder"]) or "calibrations"
        if "save_frames" in values:
            self.save_frames = bool(values["save_frames"])
        if "pulse_template" in values:
            self.pulse_template = str(values["pulse_template"]) or self.DEFAULT_PULSE_TEMPLATE
        if "grid_shape" in values:
            self.grid_shape = grid_shape_tuple(values["grid_shape"])
        if "reference_exposure" in values:
            self.reference_exposure = float(values["reference_exposure"])
        if "readout_exposure" in values:
            self.readout_exposure = float(values["readout_exposure"])
        if "roi_radius" in values:
            self.roi_radius = int(values["roi_radius"])
        if "threshold_frames" in values:
            self.threshold_frames = max(2, int(values["threshold_frames"]))
        if "threshold_method" in values:
            self.threshold_method = str(values["threshold_method"])

    def run(self, out: "TaskOutput") -> dict:
        # TWO sources, both of which BUILD a calibration: a LIVE acquisition (camera + pulse
        # template) or SAVED FRAMES on disk.  Reusing an already-saved calibration is NOT a
        # calibration run -- the Judge-occupancy processor loads its calibration.json directly,
        # so there is no "saved calibration" source here (that was a category error).
        if str(self.source) == "saved frames":
            calibration = self._run_from_folder(out)
        else:
            calibration = self._run_live(out)
        self.calibration = calibration
        self._adopt()                            # hand the calibration to the session NOW
        centers = np.asarray(self.calibration.centers, dtype=float)
        thr = np.asarray(self.calibration.thresholds, dtype=float).reshape(-1)
        # Write the CANONICAL latest calibration to ``<folder>/calibration.json`` -- the stable,
        # named file the Judge-occupancy processor defaults to, so calibrate-then-judge wires up
        # with NO path typed yet the file in use is always named.
        try:
            Path(self.folder).mkdir(parents=True, exist_ok=True)
            self.calibration.save(Path(self.folder) / "calibration.json")
        except Exception:
            pass
        # ALWAYS write the rb87-style report (per-site distribution + fidelity + site map +
        # the loadable calibration.json/npz) DIRECTLY into the user's EXPLICIT ``folder`` --
        # ONE place, alongside calibration.json + any saved raw frames, with NO hidden
        # timestamped sub-folder (the user picks the folder; re-running overwrites it, and a
        # different run goes in a different folder -- explicit, never magic).
        out.publish(progress=0.98, stage="writing distribution + fidelity report")
        self.report = self._write_report(Path(self.folder))
        folder = self.report.get("folder") if isinstance(self.report, dict) else None
        out.publish(progress=1.0, stage=(f"saved report -> {folder or self.folder}"))
        return {"centers": centers, "thresholds": thr, "n_sites": float(len(centers)),
                "report_dir": str(folder or self.folder)}

    def _adopt(self) -> None:
        """Hand the just-produced calibration to the session via ``calibration_sink`` so the
        notebook API's ``readout.current`` reflects the latest calibration immediately (the
        live OccupancyProcessor itself loads the canonical ``calibration.json`` this run also
        writes -- an explicit named file, not an implicit session handoff)."""
        if self.calibration_sink is not None and self.calibration is not None:
            self.calibration_sink(self.calibration)

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
        # Report the calibration's ACTUAL threshold method: after the held-out writeback it is
        # "per_site_reference", not the pre-train otsu label -- so summary.json matches reality.
        cal_method = self.calibration.metadata.get("threshold_method", self.threshold_method)
        return write_calibration_report(
            folder, calibration=self.calibration, readout_frames=frames,
            template=getattr(self, "_reference_template", None),
            threshold_method=cal_method,
            reference_groups=getattr(self, "_reference_groups", None),
            readout_by_group=getattr(self, "_readout_by_group", None),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


    def _run_live(self, out: "TaskOutput"):
        """Acquire the Rb87 long-short-long reference brackets NOW and run the SAME sitemap +
        per-site-threshold extraction the real readout uses.

        ONE acquisition, ONE correlated dataset -- this mirrors the saved-frames flow exactly
        (:meth:`_run_from_folder`), so live == saved.  Every shot fires the loaded imaging
        template imaged long-short-long (e.g. 20ms-5ms-20ms): the two LONG reference frames
        serve DOUBLE duty -- (a) they vote, by strict consensus, the ground-truth occupancy
        that LABELS the middle short readout (a shot where the two long frames disagree is a
        mid-readout atom-loss event, so it is discarded -- the "data cleaning" the long frames
        exist for), AND (b) every long frame averages into the high-SNR template the site
        centres + PSF are fitted from.  The short readout frames drive the box/PSF otsu
        thresholds (calibrate once, read many ways).  There is NO separate site-finding pass:
        the site map comes from the SAME bracket frames that vote the labels.  Identical on
        real hardware (only the camera frames' author differs: virtual == real)."""
        from .calibration import calibrate_all_methods_from_images
        out.publish(progress=0.0, stage="loading reference brackets (long-short-long)")
        # The rb87 calibration shot IS the long-short-long bracket (built from the loaded
        # template -- its load + image structure, imaged at the long/short exposures).
        ref_groups, readout_by_group = self._collect_bracket_groups(
            out, progress_lo=0.0, progress_hi=0.9)
        # The site map is built from EVERY long reference frame of the brackets (group-major) --
        # the same frames that vote the per-site ground truth, exactly like the saved-frames flow.
        template = [f for grp in ref_groups for f in grp]
        samples = list(readout_by_group)
        # Optionally SAVE the acquired raw frames so a later source="saved frames" run
        # re-calibrates from them without re-acquiring (the "don't re-run every time" ask).
        if self.save_frames:
            self._save_live_frames(ref_groups, readout_by_group)
        out.publish(progress=0.95, stage="finding sites + thresholds (box / PSF)")
        calibration = calibrate_all_methods_from_images(
            template, samples, grid_shape=self.grid_shape, roi_radius=self.roi_radius,
            threshold_method=self.threshold_method, exposure=self.readout_exposure)
        # Use the bracket-voted ground truth to train + write back each method's per-site
        # boundary (so detect reads on it), and stash the groups for the held-out report.
        calibration = self._apply_reference_thresholds(calibration, ref_groups, readout_by_group)
        # Keep the readout frames + averaged template so run() can write the rb87-style report.
        self._readout_samples = list(samples)
        self._reference_template = (np.mean(np.asarray(template, dtype=float), axis=0)
                                    if template else None)
        return calibration

    def _apply_reference_thresholds(self, calibration, ref_groups, readout_by_group):
        """Score each readout method's HELD-OUT fidelity against the bracket-voted ground
        truth and write the reference-trained per-site thresholds back into the calibration,
        so a downstream ``OccupancyProcessor.detect`` reads on the trained boundary (not the
        otsu quick split) -- the Rb87 "use the true labels to set where the boundary is" step.
        A NaN (a site with too few labelled shots to train) falls back to the otsu threshold;
        if there is no usable ground truth at all, ``_held_out_by_method`` returns ``{}`` and
        the calibration keeps its otsu thresholds.  Stashes the groups + per-method report so
        ``run()`` can write the held-out report.  Shared by the live and saved-frames flows."""
        from .calibration_report import _held_out_by_method
        self._reference_groups = list(ref_groups or [])
        self._readout_by_group = list(readout_by_group or [])
        self._method_fidelity = _held_out_by_method(calibration, self._reference_groups, self._readout_by_group)
        if not self._method_fidelity:
            return calibration
        trained: dict[str, np.ndarray] = {}
        for m, data in self._method_fidelity.items():
            thr = np.asarray(data["thresholds"], dtype=float).reshape(-1)
            fallback = np.asarray(calibration.thresholds_for(m), dtype=float).reshape(-1)
            bad = ~np.isfinite(thr)
            if np.any(bad):
                thr = thr.copy()
                thr[bad] = fallback[bad]
            trained[m] = thr
        return calibration.with_method_thresholds(trained, threshold_method="per_site_reference")

    def _imaging_layout(self, state) -> int:
        """WHICH captured frame is the short readout (the ``readout_index``), read from the loaded
        imaging template: the camera-trigger periods IN FIRE ORDER are the frames, and the one
        tagged API slot ``a2`` (the short readout) is the readout -- the rest are the long
        references that vote ground truth (else the middle frame).  This is the cali's own
        INTERPRETATION of its template; it does NOT tell the camera how many frames to take (the
        camera captures one per trigger).  The cali images exactly the long-short-long the FILE
        defines -- it does not invent a bracket."""
        from ..devices.camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS
        # WHICH line gates a frame is the CAMERA's knowledge (it is wired to it), not the sequencer's.
        trig = [c for c in (getattr(self.camera, "capture_trigger_channels", None) or DEFAULT_CAMERA_TRIGGER_CHANNELS)
                if c in state.channels]
        bits = [state.channels.index(c) for c in trig]
        frame_periods = [i for i, p in enumerate(state.periods) if any(p.states[b] for b in bits)]
        if len(frame_periods) < 2:
            raise ValueError(
                "the imaging template must trigger the camera at least twice (>=1 long reference "
                "frame + 1 short readout) -- a long-short-long bracket. Open the template in the "
                "pulse GUI and add the camera-trigger frames.")
        a2 = {int(s.target) for s in state.api_slots if s.name == "a2" and s.kind == "duration"}
        return next((frame_periods.index(i) for i in frame_periods if i in a2), len(frame_periods) // 2)

    def _collect_bracket_groups(self, out: "TaskOutput", *, progress_lo: float, progress_hi: float):
        """Acquire ``threshold_frames`` reference BRACKETS.  Each bracket is ONE correlated
        shot -- a long-short-long camera sequence imaging the SAME atom loading (the trap is
        held on, no re-cooling between triggers, so a following frame sees the previous
        frame's survivors).  Returns ``(reference_groups, readout_per_group)``:
        ``reference_groups[g]`` is the long frames that vote ground truth, and
        ``readout_per_group[g]`` is the short readout frame scored against them.  Honours the
        Stop event between brackets (interruptible).  Identical on real hardware -- only the
        camera frames' author differs (``virtual == real``)."""
        # The imaging template IS the long-short-long bracket: ONE cooling/load cycle, then the
        # camera-trigger frames back-to-back with trap-held gaps (no re-cooling between them, so
        # the long frames bracket and label the SAME atoms).  The cali ONLY sets the exposures
        # BY NAME -- api slot a1 = the long reference frame(s), a2 = the short readout -- so
        # editing reference_exposure/readout_exposure changes those durations and NOTHING else.
        # What is fired == the template the operator chose: file == fired.
        template = self._resolve_template(self.pulse_template)
        try:
            # Each exposure cell carries its OWN api handle (names are unique, like the GUI
            # allocates a fresh a<N> per click): a1 = first long, a2 = short readout, a3 =
            # second long.  Cali sets all three by name; structure stays as loaded.
            template.set_api("a1", self.reference_exposure)
            template.set_api("a2", self.readout_exposure)
            template.set_api("a3", self.reference_exposure)
        except ValueError as exc:
            raise ValueError(
                f"{exc}  The Calibrate task sets the imaging template's exposures by API slot: tag "
                "the three exposure cells as a1 (first long), a2 (short readout), a3 (second long) "
                "in the pulse GUI (click each duration cell to its API state).") from exc
        readout_index = self._imaging_layout(template)        # WHICH frame is the short readout (a2)
        self._readout_index = readout_index                    # shared with _save_live_frames
        bracket = template.to_sequence(name="reference_bracket")
        from ..devices.camera_trigger import count_trigger_pulses
        cam_trig = getattr(self.camera, "capture_trigger_channels", None)
        n_frames = max(1, count_trigger_pulses(bracket, **({"trigger_channels": cam_trig} if cam_trig else {})))
        n_groups = max(2, int(self.threshold_frames))
        reference_groups: list = []
        readout_per_group: list = []
        for g in range(n_groups):
            if self._stop.is_set():
                break
            # Decoupled shot: the camera arms for the bracket's N frames, then (on_armed) WE fire the
            # sequencer -- the camera never drives it.  The bracket (the fired pulse) is handed in only
            # so the virtual camera can simulate; a real qCMOS reads the frames its trigger gates.
            # Without a bound sequencer (a notebook-composed readout that leans on the virtual atom
            # array) there is nothing to fire -- the camera still simulates from the bracket sequence.
            batch = self.camera.acquire(
                n_frames, sequence=bracket, on_armed=arm_then_fire(self.sequencer, bracket), stop=self._stop)
            frames = [np.asarray(f, dtype=float) for f in batch]
            if len(frames) <= readout_index:
                break                                          # stopped mid-bracket -> no full shot
            readout_per_group.append(frames[readout_index])
            reference_groups.append([f for i, f in enumerate(frames) if i != readout_index])
            frac = progress_lo + (progress_hi - progress_lo) * (g + 1) / n_groups
            # Stream the FIRST long reference frame (the high-SNR image the site map is built from)
            # so the operator watches the site-finding data accumulate, not just the short readout.
            out.publish(frame=frames[0], progress=frac, stage=f"reference bracket {g + 1}/{n_groups} (long-short-long)")
        return reference_groups, readout_per_group

    # Raw camera frames are an INPUT-format artifact (the round-trip data for a later
    # source="saved frames" re-cali), NOT a "save action".  They live in this explicit
    # semantic sub-folder of ``folder`` so the cali folder root stays clean -- only the
    # canonical artifacts (calibration.json / .npz / summary.json / run_schema.json) and
    # the paired (.png + .npz) figure saves live at the root.  An explicit named sub-folder
    # is NOT the "hidden timestamp" the user banned in #3 -- this name is fixed, predictable,
    # and the GUI's "folder" field still points at the user-chosen root.
    FRAMES_SUBDIR = "frames"

    def _save_live_frames(self, ref_groups, readout_by_group) -> None:
        """Write the live brackets to ``<folder>/frames/`` as a contiguous ``img<n>.npy`` run --
        the round-trip DATA -- each paired with an ``img<n>.png`` 2D-image render of the SAME frame,
        plus a ``run_schema.json`` at the cali folder root describing the grouping (and the frames
        sub-folder), so a later source="saved frames" run re-indexes them with ``index_run`` and
        re-calibrates WITHOUT re-acquiring.  Each group is written in trigger order (the short
        readout in the middle of its reference frames); the schema records ``shots_per_group`` /
        ``short_shot`` / ``ref_shots`` / ``frames_subdir`` so the reader reconstructs the exact
        bracket -- no frame duplication, no hard-coded layout.

        The ``.png`` is the operator's eyeball companion to the emCCD bracket frames the cali
        ACTUALLY used: it is rendered through the frontend viewer seam (the same 2D-image imshow the
        live console uses), so na never imports the frontend.  Headless (no viewer registered) ->
        the ``.npy`` data is written exactly the same and the png is simply skipped; a render hiccup
        never costs the saved ``.npy``."""
        import json
        from Zou_lab_control._viewer_registry import active_plotter
        from .imageio import save_frame
        folder = Path(self.folder)
        frames_dir = folder / self.FRAMES_SUBDIR
        frames_dir.mkdir(parents=True, exist_ok=True)
        # png companion: rendered by the frontend's ``save_frame_image`` when a viewer is registered.
        _plotter = active_plotter()
        _render_png = getattr(_plotter, "save_frame_image", None) if _plotter is not None else None
        # The readout frame's position within the bracket is the template-derived layout the
        # live acquisition used (``_collect_bracket_groups`` stored it) -- so the saved run
        # re-reads with the SAME grouping the template defines (any frame count, not a hard 3).
        readout_index = int(getattr(self, "_readout_index", len(ref_groups[0]) // 2 if ref_groups else 1))
        n = 0
        shots_per_group = 1
        for refs, short in zip(ref_groups, readout_by_group):
            refs = list(refs)
            group = refs[:readout_index] + [short] + refs[readout_index:]   # trigger order
            shots_per_group = len(group)
            for fr in group:
                n += 1
                arr = np.asarray(fr, dtype=float)
                save_frame(frames_dir / f"img{n}.npy", arr)
                if _render_png is not None:
                    try:
                        _render_png(frames_dir / f"img{n}.png", arr)   # 2D image of this exact frame
                    except Exception:
                        pass   # a viewer render hiccup must NEVER lose the round-trip .npy
        short_shot = readout_index + 1                       # 1-based position of the short frame
        ref_shots = [i for i in range(1, shots_per_group + 1) if i != short_shot]
        schema = {"prefix": "img", "shots_per_group": shots_per_group, "short_shot": short_shot,
                  "ref_shots": ref_shots, "n_groups": int(len(readout_by_group)),
                  "frames_subdir": self.FRAMES_SUBDIR}
        (folder / "run_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def _index_saved_run(self):
        """Index the saved frames, honouring ``run_schema.json`` (the live run wrote it) so a
        live-saved bracket re-reads with its real grouping + its ``frames_subdir``.  If no
        schema, fall back to the index_run defaults (the write_virtual_run / 4-shot convention)
        at the folder root."""
        import json
        from .imageio import index_run
        root = Path(self.folder)
        schema_path = root / "run_schema.json"
        if schema_path.is_file():
            try:
                s = json.loads(schema_path.read_text(encoding="utf-8"))
                sub = str(s.get("frames_subdir", "") or "")
                data_dir = root / sub if sub else root
                return index_run(data_dir, str(s.get("prefix", "img")),
                                 shots_per_group=int(s["shots_per_group"]),
                                 short_shot=int(s["short_shot"]),
                                 ref_shots=tuple(int(v) for v in s["ref_shots"]))
            except Exception:
                pass
        # No schema: try the named sub-folder first (cleaner layout), else the root (legacy /
        # write_virtual_run convention -- a flat folder of img<n>.npy).
        sub = root / self.FRAMES_SUBDIR
        if sub.is_dir() and (any(sub.glob("img*.npy")) or any(sub.glob("img*.tif*"))):
            return index_run(sub, "img")
        return index_run(root, "img")

    def _run_from_folder(self, out: "TaskOutput"):
        """Calibrate from frames SAVED IN A FOLDER (the real-data flow): index the run,
        find the sites from the reference template, then per-site thresholds for every
        readout method (the OccupancyProcessor picks which to use).

        A saved run is grouped exactly like the live bracket -- each group holds the long
        REFERENCE frames (``ref_shots``) that vote ground truth around its short readout
        (``short_shot``).  Re-grouping the run's reference frames back into per-group
        brackets lets the saved-frames flow take the SAME held-out training path as the
        live flow (:meth:`_apply_reference_thresholds`): distinct box / PSF / uniform
        held-out fidelity + reference-trained per-site thresholds, NOT the affine-invariant
        self-consistent estimate.  If the run is too short to vote ground truth, the helper
        no-ops and the calibration keeps its otsu thresholds (graceful fallback)."""
        from .calibration import calibrate_all_methods_from_images
        if not self.folder:
            raise ValueError("source='saved frames' needs a folder of saved frames.")
        run = self._index_saved_run()
        n_ref = len(run.ref_shots)
        # The long reference frames (group-major) build the all-sites template AND, regrouped,
        # vote the per-group ground truth -- read the files ONCE for both uses.
        reference_flat = [np.asarray(f, dtype=float) for f in run.reference_frames()]
        if reference_flat:
            out.publish(frame=reference_flat[-1], progress=0.4)
        samples = [np.asarray(f, dtype=float) for f in run.short_frames()]   # one readout per group
        if samples:
            out.publish(frame=samples[-1], progress=0.85)
        calibration = calibrate_all_methods_from_images(
            reference_flat, samples, grid_shape=self.grid_shape, roi_radius=self.roi_radius,
            threshold_method=self.threshold_method, exposure=self.readout_exposure)
        n_groups = run.n_groups
        ref_groups = ([reference_flat[g * n_ref:(g + 1) * n_ref] for g in range(n_groups)]
                      if n_ref and len(reference_flat) >= n_groups * n_ref
                      and len(samples) == n_groups else [])
        calibration = self._apply_reference_thresholds(calibration, ref_groups, samples)
        self._readout_samples = list(samples)
        self._reference_template = (np.mean(np.asarray(reference_flat, dtype=float), axis=0)
                                    if reference_flat else None)
        return calibration


def camera_frame_keys(frames_per_cycle, prefix=""):
    """The SINGLE source of a camera's published signal names: ONE ``frame_i`` per emCCD event,
    ``frame_0 .. frame_{N-1}`` (NO lumped ``frame``).  Used by BOTH ``CameraMeasurement.published_signals``
    (live, with the node prefix) and the console's declared-signal picker (bare, before the node starts) so
    the two can never drift -- a declared 'waiting' name always equals what the running camera will emit."""
    n = max(1, int(frames_per_cycle or 1))
    return [f"{prefix}frame_{i}" for i in range(n)]


class CameraMeasurement(Measurement):
    """Stream RAW camera frames into the hub, no site analysis.  The data SOURCE is
    the camera itself, so the editable acquisition parameters ARE the camera's own
    settings -- ``exposure`` and ``roi`` -- applied live (``camera.configure``).
    Backend-agnostic: identical with a real camera or a ``VirtualCamera``.

    MULTI-TRIGGER per cycle (``frames_per_cycle``).  One ``shot()`` reads
    ``frames_per_cycle`` frames -- ONE per camera (emCCD) trigger in the running
    pulse -- and publishes each as its own signal ``frame_0``, ``frame_1``, ... and
    NOTHING else (no lumped ``frame``; the default 2D panel binds ``frame_0``).  A
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
                 frames_per_cycle: int = 1, prefix: str = "", repeat: int = 0):
        # A cycle fires ``frames_per_cycle`` emCCD events (triggers); the camera publishes ONE signal
        # PER event -- ``frame_0, frame_1, … frame_{N-1}`` -- and NOTHING else (no lumped ``frame``).
        # Each ``frame_i`` is that event's OWN repeat block obeying the UNIFORM contract (#H3n)
        # ``(ring, *points_shape, *data_shape)`` = ``(ring, 1, H, W)``: the i-th emCCD image STACKED
        # across the cycle's repeats.  So a panel bound to ``frame_i`` reduces ITS repeat axis
        # (repeat_mode: average = the long-exposure mean of THAT specific emCCD event) -- the only way
        # a 3-event cycle can show the repeat_mode effect for a chosen event.  ONE knob ``repeat``,
        # 0 = ∞ (no free-run toggle): repeat=K keeps each event's K-deep block then STOPS; repeat=0
        # rolls a 1-deep ring forever (live monitor).  The camera never averages at the measurement
        # (that was the live stutter) -- it FILLS and publishes the WHOLE per-event block every cycle.
        super().__init__(hub, prefix=prefix)
        self.camera = camera
        self.sequencer = sequencer
        self.frames_per_cycle = max(1, int(frames_per_cycle))
        self.points_shape: tuple = (1,)                  # one frame = one data point (no swept param)
        self.primary_signal = "frame_0"                  # event 0's (repeat,1,H,W) block (#H3r-F4)
        self.data_shape: tuple = ()                      # set to (H, W) on the first frame
        self._rings = None                               # list of N (ring,1,H,W) blocks; None until 1st frame
        self.set_repeat(repeat)

    def set_repeat(self, repeat: int = 0) -> None:
        """How many photos to KEEP & AVERAGE then STOP, or ``0`` = ∞ (roll forever, a live monitor
        showing the latest frame).  ONE knob, 0 = infinite -- there is NO separate free-run toggle:
        repeat=K fills a K-deep block once and stops; repeat=0 rolls a 1-deep ring forever.  Resets
        the (partly filled) block."""
        self.repeat = max(0, int(repeat))
        self._ring = max(1, self.repeat)                 # block depth: K for finite, 1 for the rolling live view
        self._rings = None
        self._filled = 0

    @property
    def total_points(self) -> int:
        return 0 if self.repeat <= 0 else int(self.repeat)

    @property
    def points_done(self) -> int:
        return int(self._filled)

    @property
    def finished(self) -> bool:
        return self.repeat > 0 and self._filled >= self.repeat

    def shot(self) -> dict[str, object]:
        n = max(1, int(self.frames_per_cycle))
        # Live monitor: the streamer is already firing continuously; the camera images whatever it is
        # firing now (we hand it that pulse).  None / not firing a camera trigger -> no frame (freeze).
        frames = self.camera.acquire(n, sequence=self.sequencer.firing, stop=self._stop)
        if not frames:
            # The streamer is not firing a camera-triggering pulse (e.g. the user hit "Stop
            # Pulse") -> no trigger -> no frame.  Publish nothing: the live view holds its last
            # image and FREEZES, exactly as a real externally-triggered camera does.  The gate
            # lives in the DATA SOURCE (only the lowest layer is faked): the virtual camera reads
            # the in-process sequencer's firing state; a real qCMOS learns it directly from the
            # absence of hardware trigger edges -- so this is correct for a real streamer driven
            # from another process too (the camera sees the actual triggers, not a local flag).
            return {}
        # A real cycle WAS acquired -> mint this shot's SOURCE-shot id; every ``frame_i`` published this
        # cycle carries it, and the OccupancyProcessor consuming a chosen ``frame_i`` inherits it, so the
        # judged image / occupancy group with the exact frame they came from (#shot-clock).
        self._current_source_shot = self.hub.next_source_shot()
        frames = [np.asarray(f, dtype=float) for f in frames]
        n = len(frames)
        f0 = frames[0]
        if self._rings is None or len(self._rings) != n or self.data_shape != f0.shape:
            self.data_shape = tuple(f0.shape)            # (H, W) -- the per-point DATA shape
            self._rings = [np.full((self._ring, 1, *self.data_shape), np.nan, dtype=float)
                           for _ in range(n)]            # ONE repeat block per emCCD event of the cycle
            self._filled = 0
        out: dict[str, object] = {}
        for i, fi in enumerate(frames):                  # fill EACH event's own ring, publish its block
            if self.repeat <= 0:                         # ∞: roll the newest in at the end (live monitor)
                self._rings[i] = np.roll(self._rings[i], -1, axis=0)
                self._rings[i][-1, 0] = fi
            else:                                        # finite: FILL the next slot of the K-deep block
                self._rings[i][min(self._filled, self._ring - 1), 0] = fi
            # ``frame_i`` IS event i's (repeat, 1, H, W) block (NaN = not-yet-taken) -- a panel reduces
            # its repeat axis (average = long exposure of THAT emCCD event).  No lumped ``frame``.
            out[f"frame_{i}"] = self._rings[i].copy()
        self._filled = (self._filled + 1) if self.repeat <= 0 else min(self._filled + 1, self.repeat)
        if self.finished:                                # take exactly K cycles, then stop
            self._stop.set()
        return self._assert_primary_shape(out)           # frame_0 == (repeat, 1, H, W) -- contract guard

    def published_signals(self) -> frozenset:
        # ONE block per emCCD event; no lumped `frame`.  Same source as the console's declared picker.
        return frozenset(camera_frame_keys(self.frames_per_cycle, self.prefix))

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """Camera outputs: ONE ``frame_i`` per emCCD event of the cycle, each a ``(repeat, 1, H, W)``
        block (repeat × ONE point × the H×W image) -- a panel reduces ITS repeat axis (repeat_mode:
        average = the long exposure of THAT specific emCCD event).  There is NO lumped ``frame``."""
        specs = []
        for name in sorted(self.published_signals()):
            bare = name[len(self.prefix):] if self.prefix and name.startswith(self.prefix) else name
            i = bare.split("_")[-1]
            desc = (f"emCCD event {i} of the cycle: (repeat, 1, H, W) block -- plot reduces repeats "
                    f"(average = long exposure of event {i}).  LIVE camera, advances independently of the "
                    "readout: for a 2D image shot-locked to the site map, bind Judge-occupancy `frame_judged`.")
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
        else:
            # No sub-array set (full frame): still expose ``region`` as the FULL sensor window,
            # so the Edit always shows an ROI field (the operator can crop by editing it or by
            # area-selecting on the plot) and the selector writeback has a field to fill.  Falls
            # back to omitting it only if the camera cannot report its sensor size.
            shape = getattr(self.camera, "sensor_shape", None)
            if shape is not None:
                h, w = (int(v) for v in shape)
                params["region"] = [0, w, 0, h]
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


class _SweptBlockMeasurement(Measurement):
    """Shared swept-block ring contract for EVERY scan node (#E4).

    A scan node fills a RAW ``(repeat, *points_shape, *data_shape)`` block point-by-point and
    publishes it WHOLE -- the node never combines the repeats; the PLOT reduces the repeat axis per
    its ``repeat_mode``.  ONE knob ``repeat``: 0 = ∞ (roll a 1-deep ring forever, a live scan showing
    the latest sweep); K = keep K whole passes (averageable) then STOP.  A subclass sets the swept x
    values + per-point data width once via :meth:`_init_swept_block`, implements only its per-point
    body (calling :meth:`_fill_point`), and reuses the ring state / progress properties / publish /
    finite-scan stop here -- so neither twin (a :class:`ScannedMeasurement` driver or a
    :class:`PulseScanNode`) can drift the contract: the law lives on the common ancestor, not retyped
    per node."""

    def _init_swept_block(self, *, values, data_shape: tuple, repeat: int) -> None:
        """Set up the swept x values + the pre-allocated RAW block.  ``repeat`` (0 = ∞) fixes the ring
        depth; ``data_shape`` is the per-point data (``(n_series,)`` for a reducer curve, ``(1,)`` for
        a scalar)."""
        self.repeat = max(0, int(repeat))
        self._ring = max(1, self.repeat)              # block depth: K passes for finite, 1 (rolling) for ∞
        self._index = 0                               # within-pass point index (0..n_points-1)
        self._pass = 0                                # 0-based pass currently being filled
        self._values = np.asarray(values, dtype=float).reshape(-1)
        self.points_shape: tuple = (int(self._values.size),)   # the swept parameter points
        self.data_shape: tuple = tuple(data_shape)             # per-point data
        # RAW (repeat, *points_shape, *data_shape), NaN = not-yet-measured; filled in place by
        # (pass, point) and published whole (the plot reduces the leading repeat axis).
        self._raw = np.full((self._ring, *self.points_shape, *self.data_shape), np.nan, dtype=float)

    @property
    def n_points(self) -> int:
        return int(self._values.size)

    @property
    def points_done(self) -> int:
        """Total points measured so far ACROSS all passes (monotonic over the whole repeat run)."""
        return int(self._pass * self.n_points + self._index)

    @property
    def total_points(self) -> int:
        """All points over all passes (n_points x repeat); 0 (open-ended) while ∞ (repeat=0)."""
        return 0 if self.repeat <= 0 else int(self.n_points * int(self.repeat))

    @property
    def finished(self) -> bool:
        """True once every point of every pass has been measured (never, while ∞ / repeat=0)."""
        return self.repeat > 0 and (self._pass >= int(self.repeat))

    def _publish_raw(self) -> np.ndarray:
        """The raw block to publish.  Finite: as-is (slot == pass, already chronological).  ∞ rolling
        ring: rolled so the most-recently-written slice is LAST (oldest->newest), so the plot's
        replace/roll/create read the newest correctly."""
        if self.repeat > 0:
            return self._raw.copy()
        last = (self._pass if self._index > 0 else self._pass - 1) % self._ring
        return np.roll(self._raw, self._ring - 1 - last, axis=0).copy()

    def _fill_point(self, index: int, row) -> None:
        """Fill measured ``row`` into the raw block at the current ``(pass, point)`` and advance the
        within-pass index / pass counter.  Clears a reused ring slot on a pass's first point so two
        passes never mix.  The node only FILLS; HOW the repeats combine for display is the plot's job."""
        row = np.atleast_1d(np.asarray(row, dtype=float))
        slot = self._pass % self._ring
        if index == 0:                               # first point of a pass -> clear its slice
            self._raw[slot] = np.nan                 # (so a reused ring slot never mixes 2 passes)
        self._raw[slot, index, :row.size] = row
        self._index += 1
        if self._index >= self.n_points:             # this pass complete -> start the next one
            self._index = 0
            self._pass += 1

    def step(self) -> dict[str, object]:
        """Run one point, publish it, and self-stop once the final point lands (so a background
        ``start()`` thread's loop exits cleanly on a finite scan)."""
        named = super().step()
        if self.finished:
            self._stop.set()
        return named

    def run_to_completion(self):
        """Synchronously run + publish every remaining scan point (test/headless)."""
        while not self.finished:
            self.step()
        return self


class ScannedMeasurementNode(_SweptBlockMeasurement):
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

    ``<x_key>``         (points,) the full swept x axis, stable from shot 1 (NaN-free)
    ``<y_key>``         (repeat, points, dim) the RAW output block -- the node FILLS it point by
                        point and does NOT combine the repeats; a PLOT reduces the repeat axis per
                        its ``repeat_mode`` (average / add / replace / roll / new).  ``dim`` is the
                        reducer's series count (a per-site reducer makes ``dim = n_sites``, so a
                        1-D plot draws one line per site and a grid view reshapes a reduced point).
    ``scan_done``       0 while running, 1 once the final point has been published

    Nothing DERIVABLE is published: the panel namespace already carries the global ``shot`` counter,
    and any combine / per-site grid view is a PLOT-side reduction + reshape, never a separate signal.

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
        repeat: int = 1,
    ):
        super().__init__(hub, prefix=prefix)
        self.measurement = measurement
        # UNIFORM contract (#H3n): the block is ``(repeat, *points_shape, *data_shape)`` = a 1-D scan's
        # ``(ring, n_points, dim)``.  ONE knob ``repeat``, 0 = ∞ (no free-run toggle): repeat=K keeps a
        # K-deep block (K passes averaged) then STOPS; repeat=0 rolls a 1-deep ring forever (a live scan
        # showing the latest sweep).  The node only FILLS point-by-point; HOW the repeats are combined
        # for display is the PLOT's ``repeat_mode``.
        # Share the node's stop event so a Stop interrupts a wedged trigger MID-scan-point.
        try:
            self.measurement.stop_event = self._stop
        except AttributeError:
            pass
        self.x_key = str(x_key)
        self.y_key = str(y_key)
        self.node_label = str(prefix).rstrip("_") or str(y_key)
        self.x_signal = self.prefix + self.x_key
        self.y_signal = self.prefix + self.y_key
        # The swept-block contract (ring depth, RAW (repeat, n_points, dim) buffer, progress
        # properties, finite-scan stop) is owned by _SweptBlockMeasurement; we supply the swept x
        # values (the measurement's axis = the single source of truth) and the per-point data width
        # (the reducer's series count, so a per-site reducer makes dim = n_sites).
        n_series = max(1, int(getattr(measurement.reducer, "n_series", 1)))
        self._init_swept_block(values=measurement.axis.values, data_shape=(n_series,), repeat=repeat)
        self.primary_signal = self.y_key                       # the (repeat,n_points,dim) block (#H3r-F4)

    def shot(self) -> dict[str, object]:
        """Measure ONE scan point and FILL it into the raw ``(repeat, points, dim)`` block at
        ``(pass, point)`` -- the node only fills, it does NOT combine the repeats (the PLOT's
        ``repeat_mode`` decides how to reduce the repeat axis).  Publishes the FULL raw block + the
        stable x axis every shot (NaN = not-yet-measured).  Raises ``StopIteration`` once finished."""

        if self.finished:
            raise StopIteration("ScannedMeasurementNode: scan already complete.")
        index = self._index
        value = float(self._values[index])
        row = self.measurement.measure(value, index)
        self._fill_point(index, row)                      # base owns slot/pass-clear/advance (#E4)

        self._current_source_shot = self.hub.next_source_shot()   # one SOURCE-shot per scan point (#shot-clock)
        return self._assert_primary_shape({
            self.x_key: self._values.copy(),             # the FULL x axis, stable from shot 1
            self.y_key: self._publish_raw(),             # RAW (repeat, points, dim) -- plot reduces it
            "scan_done": 1.0 if self.finished else 0.0,
        })

    def published_signals(self) -> frozenset:
        return frozenset(self.prefix + key for key in (self.x_key, self.y_key, "scan_done"))

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """x / y axis labels + units come from the swept measurement itself (the scan AXIS for x,
        the REDUCER labels for the curve).  ``y_key`` is the RAW ``(repeat, points, dim)`` block --
        a plot reduces its repeat axis per ``repeat_mode``."""
        p = self.prefix
        axis = self.measurement.axis
        rlabels = tuple(self.measurement.reducer.labels)          # (xlabel, ylabel, zlabel)
        ylabel = rlabels[1] if len(rlabels) > 1 else self.y_key
        xlabel = str(getattr(axis, "label", "x"))
        xunit = str(getattr(axis, "unit", ""))
        return (
            SignalSpec(p + self.x_key, xlabel, xunit, "scan x axis (the swept parameter)"),
            SignalSpec(p + self.y_key, ylabel, "", "raw (repeat, points, dim) block; plot reduces repeats"),
            SignalSpec(p + "scan_done", "scan complete", "", "1 once the final point is measured"),
        )


class PulseScanNode(_SweptBlockMeasurement):
    """Drive a pulse-template scan one point per ``shot()``, with a DECOUPLED y.

    Unlike :class:`ScannedMeasurementNode` (which reduces its OWN frames through a calibration),
    pulse-scan is a DEVICE driver whose y comes from ANOTHER running node: per point it resolves
    the bound scan slots to that row (``with_slots_resolved`` -- the SAME named-slot resolver the
    hardware scan + pulse GUI use; api slots stay FIXED on the base state), FIRES + acquires the
    camera ``frame``(s), PUBLISHES them, lets the consumers (e.g. a Judge-occupancy processor)
    recompute, then evaluates a SOURCE EXPRESSION over the hub for y.  So x = the scan points and
    y = a signal published by a decoupled producer (e.g. occupancy ``rate``) combined by a
    ``value = ...`` expression.

    Because the device lockout allows only ONE device driver, pulse-scan owns the streamer +
    camera and publishes the per-event ``frame_i`` itself; the reactive processors it subscribes to keep running
    (processors are not device-driving), so the user starts the producer FIRST, then pulse-scan.

    Settling y to THIS point's frame (the riskiest part) is race-free:
      * GUI / live: the consumer runs on its OWN thread, so the node WAITS for the picked y
        signals' per-signal version to advance past the pre-publish snapshot (it only READS the
        hub -- never steps another node's thread -- so there is no cross-thread step() race).
      * headless / notebook / tests: an optional ``settle`` callback steps the consumer INLINE
        (single-threaded, deterministic) -- the consumer's thread is not running, so settle is
        the only thing touching it.
    Both paths read y through the SAME expression once the consumer is fresh.

    It touches only ``with_slots_resolved`` + ``to_sequence`` + ``camera.acquire`` + the hub
    (publish / signal_versions / latest), reads no simulation ground truth and imports no
    concrete backend -- guarded by ``tests/test_virtual_equals_real_contract.py`` like the rest
    of the analysis layer.
    """

    node_label = "pulse_scan"
    #: How long (s) to wait for the subscribed y signals to refresh for THIS point before
    #: reading them anyway (so a mis-wired y never wedges the scan; a real consumer ticks well
    #: under this).
    SETTLE_TIMEOUT_S = 5.0

    def __init__(self, hub: SignalHub, plan, *, x_key: str = "param", y_key: str = "signal",
                 prefix: str = "", repeat: int = 1):
        super().__init__(hub, prefix=prefix)
        self.plan = plan
        # The swept-block ring contract -- ``(repeat, n_points, 1)`` buffer, ring depth, progress
        # properties, finite-scan stop -- is owned by _SweptBlockMeasurement (#E4); set up below via
        # _init_swept_block once the swept x values are known.  Pulse-scan fires ONCE per point per
        # pass, so a "pass" IS a whole re-sweep (repeat=K keeps K averageable sweeps then STOPS;
        # repeat=0 rolls a 1-deep ring forever); the PLOT combines the repeats + (2-D) reshapes by
        # ``scan_shape``.
        self.base_state = plan.base_state
        self.scan_names = list(plan.scan_names)
        self.scan_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in plan.scan_arrays]
        # SOFTWARE api-slot sweep (the analogue of the hardware scan table): per point we set_api
        # each api column on a deep copy, fire, then wait extra_delay_s -- the device-owned
        # inter-point settle (load -> on_pulse -> wait pulse done -> settle -> next).
        self.api_names = list(getattr(plan, "api_names", ()))
        self.api_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in getattr(plan, "api_arrays", ())]
        self.extra_delay_s = max(0.0, float(getattr(plan, "extra_delay_s", 0.0)))
        self.camera = plan.camera
        self.sequencer = plan.sequencer
        self.y_expr = plan.y_expr if isinstance(plan.y_expr, SignalExpr) else SignalExpr.from_value(plan.y_expr)
        # Optional inline settle (headless single-threaded determinism); None in the GUI, where
        # the node instead waits for the consumer's own thread to republish the y signals.
        self.settle = getattr(plan, "settle", None)
        self.x_key = str(x_key)
        # The OUTPUT signal name (user-set #7) comes from the plan; the constructor default is the
        # fallback for callers that don't carry it.
        self.y_key = str(getattr(plan, "y_key", "") or y_key)
        # Optional (n0, n1) grid shape for a 2-D scan: a 2-D panel reduces the raw y block's repeat
        # axis then reshapes the (points,) curve into this map (the node itself never reshapes).
        self.scan_shape = getattr(plan, "scan_shape", None)
        self.node_label = str(prefix).rstrip("_") or str(y_key)
        self.x_signal = self.prefix + self.x_key
        self.y_signal = self.prefix + self.y_key
        # The x dimension is the hardware scan slots if any, else the software api sweep; the base
        # pre-allocates the RAW (repeat, n_points, 1) NaN block + ring/progress state from it (a stable
        # x axis from shot 1).  One scalar per point (data_shape=(1,)); a 2-D scan flattens its n0*n1
        # grid into n_points here and the 2-D panel reshapes by scan_shape.
        if self.scan_arrays:
            swept_values = self.scan_arrays[0].astype(float)
        elif self.api_arrays:
            swept_values = self.api_arrays[0].astype(float)
        else:
            swept_values = np.array([0.0])
        self._init_swept_block(values=swept_values, data_shape=(1,), repeat=repeat)
        self.primary_signal = self.y_key                       # the (repeat,n_points,1) block (#H3r-F4)
        # 2-D scan: the flattened points reshape to this (n0, n1) image on a 2-D panel (#H3o); the
        # data stays a scalar, so the 2-D-ness is HERE, not in data_shape.
        self.grid_shape: tuple = tuple(self.scan_shape) if self.scan_shape else ()
        self._axis_label = str(plan.axis_label)
        self._axis_unit = str(plan.axis_unit)

    def _publish_grid(self) -> np.ndarray:
        """For a 2-D scan: the SAME raw block reshaped to ``(repeat, n0, n1)`` -- a pure reshape of
        this node's own data, NO cross-repeat combine.  The 2-D panel reduces the repeat axis (per
        ``repeat_mode``) then shows the (n0, n1) map, exactly as a 1-D panel reduces (repeat,points)."""
        n0, n1 = self.scan_shape
        return self._publish_raw()[:, :, 0].reshape(self._ring, int(n0), int(n1))

    def published_signals(self) -> frozenset:
        """The signals this scan publishes (behind ``prefix``): the swept x axis, the RAW y block,
        the scan-done flag, the frame it fires, plus (2-D scan) the raw block reshaped into the grid.
        Declaring them lets the console map a panel back to THIS node (so the 1-D frame-title reads
        ``<y> <- pulse_scan``) and resolve x.  Both y signals are RAW (repeat-axis kept) -- the PLOT
        reduces them; the node never combines repeats."""
        p = self.prefix
        # ONE source of the camera-frame names (no lumped ``frame``): pulse-scan fires ONE trigger per
        # point, so it publishes the same ``frame_0`` a single-cycle CameraMeasurement does (#E1).
        keys = [p + self.x_key, p + self.y_key, p + "scan_done", *camera_frame_keys(1)]
        if self.scan_shape is not None:
            keys.append(p + self.y_key + "_grid")
        return frozenset(keys)

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """LABEL / unit / meaning per published signal -- so a 1-D panel wired to the y curve reads
        its x-axis label+unit (the swept parameter) and y label from THIS node.  ``y_key`` is the RAW
        ``(repeat, points, 1)`` block; ``<y>_grid`` the RAW ``(repeat, n0, n1)`` block -- a plot
        reduces the repeat axis per ``repeat_mode``."""
        p = self.prefix
        specs = [
            SignalSpec(p + self.x_key, self._axis_label, self._axis_unit, "scan x axis (the swept parameter)"),
            SignalSpec(p + self.y_key, self.y_key, "", "raw (repeat, points, 1) block; plot reduces repeats"),
            SignalSpec(p + "scan_done", "scan done", "", "1.0 when the finite scan has completed"),
        ]
        if self.scan_shape is not None:
            specs.append(SignalSpec(p + self.y_key + "_grid", self.y_key, "",
                                    "raw (repeat, n0, n1) block; plot reduces repeats then shows the map"))
        return tuple(specs)

    def shot(self) -> dict[str, object]:
        if self.finished:
            raise StopIteration("PulseScanNode: scan already complete.")
        index = self._index
        # Resolve THIS point's pulse: the hardware scan slots through the named-slot resolver, AND
        # (software) any swept API slots through set_api on the deep copy.  Either or both may be
        # present; an api-only sweep has no scan slots.
        resolved = self.base_state
        if self.scan_names:
            slots = {name: float(arr[index]) for name, arr in zip(self.scan_names, self.scan_arrays)}
            resolved = resolved.with_slots_resolved(slots)
        if self.api_names:
            api_row = {name: float(arr[index]) for name, arr in zip(self.api_names, self.api_arrays)}
            resolved = resolved.with_api_resolved(api_row)
        sequence = resolved.to_sequence(name="pulse_scan")
        # ONE trigger per point: pulse-scan FIRES the pulse and reads the UPSTREAM signal (y) -- it
        # is decoupled from the camera's exposure/averaging (no n_frames knob; temporal averaging
        # belongs to the upstream camera/processor).  The camera never drives the sequencer; WE fire
        # it from on_armed.  No bound sequencer -> nothing to fire (the virtual camera still simulates).
        frames = self.camera.acquire(
            1, sequence=sequence, on_armed=arm_then_fire(self.sequencer, sequence), stop=self._stop)
        if not frames:
            # Streamer not firing (e.g. Stop) -> no trigger -> no frame: freeze, do not advance
            # (the SAME data-source gate the camera measurement uses).
            return {}
        # 1) publish the camera frame(s) under BARE names so the reactive consumers (e.g. a
        #    Judge-occupancy processor consuming ``frame_0``) pick them up -- this is the SAME
        #    ``frame_0`` signal a single-cycle CameraMeasurement publishes (default 2D = frame_0).
        before = self.hub.signal_versions()
        # Mint ONE source-shot id for this scan point: the camera frame, any consumer that inherits it
        # (a Judge-occupancy processor on `frame`), AND the scan y-block (published via step()) all share
        # it, so this point's frame + derived occupancy + y are one coherent shot.  Order matters: the
        # frame publish (with sid) must precede the consumer recompute (_await_y_inputs) so it inherits sid
        # before _read_y -- a reorder would break lineage coherence (#shot-clock).
        sid = self.hub.next_source_shot()
        frame_pub = {f"frame_{k}": np.asarray(f, dtype=float) for k, f in enumerate(frames)}
        self.hub.publish(frame_pub, provenance=sid)
        self._current_source_shot = sid                  # the y-block (base step()) carries the same id
        # 2) make the y signals FRESH for this frame, then 3) read y from the source expression.
        self._await_y_inputs(before)
        y = self._read_y()
        # 4) device-owned inter-point settle: load -> on_pulse -> wait pulse done (camera.acquire)
        #    -> settle (extra_delay_s) -> next.  The sequencer owns the wait (the caller just sets
        #    the adjustable extra delay); honours Stop so a long settle does not wedge teardown.
        if self.extra_delay_s > 0.0 and not self._stop.is_set():
            self.sequencer.settle(self.extra_delay_s, stop=self._stop)
        self._fill_point(index, y)                        # base owns slot/pass-clear/advance (#E4)
        out = {self.x_key: self._values.copy(),
               self.y_key: self._publish_raw(),          # RAW (repeat, points, 1) -- plot reduces it
               "scan_done": 1.0 if self.finished else 0.0}
        if self.scan_shape is not None:
            out[self.y_key + "_grid"] = self._publish_grid()   # RAW (repeat, n0, n1) -- plot reduces it
        return self._assert_primary_shape(out)

    def _await_y_inputs(self, before: dict) -> None:
        """Block until the picked y signals have advanced past ``before`` (a per-signal version
        bump = the consumer republished them from the frame just published), honouring Stop.

        Inline ``settle`` (headless) short-circuits this: it steps the consumer once, single-
        threaded, so the version is fresh immediately."""
        names = [n for n in self.y_expr.inputs if n]
        if not names:
            return
        if self.settle is not None:
            try:
                self.settle()
            except Exception:
                pass
            return
        deadline = time.monotonic() + self.SETTLE_TIMEOUT_S
        while not self._stop.is_set():
            now = self.hub.signal_versions()
            if all(now.get(n, 0) > before.get(n, 0) for n in names):
                return
            if time.monotonic() >= deadline:
                return                                       # timeout: read what is there, never wedge
            self._stop.wait(0.01)

    def _read_y(self) -> float:
        """y = the source expression over the live hub (the decoupled subscription)."""
        try:
            value = self.y_expr.evaluate(hub_namespace(self.hub))
            arr = np.asarray(value, dtype=float).reshape(-1)
            return float(arr[0]) if arr.size else float("nan")
        except Exception:
            return float("nan")


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
        # stop up front means a run() that raises is NOT retried -- a deterministic
        # processing action runs once.  ``finished`` means "this one-shot has RUN ONCE"
        # (success OR failure), so it is set in ``finally`` alongside the stop event --
        # otherwise a spec.run() that raises leaves finished=False forever (mirrors the
        # Task fix, finding 7).  The publish below runs only on success (after run returns).
        self._stop.set()
        ctx = ProcessorContext(
            readout=self._readout, params=self._params,
            camera=self._camera, sequencer=self._sequencer, stop=self._stop)
        try:
            result = self.spec.run(ctx)
        finally:
            self.finished = True
        self.result = {str(key): value for key, value in dict(result).items()}
        out = dict(self.result)
        out["processor_done"] = 1.0
        self._current_source_shot = self.hub.next_source_shot()   # one SOURCE-shot for this discrete result (#shot-clock)
        return self._assert_declared(out, tuple(self.spec.result_keys) + ("processor_done",))

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
    "grid_for_points",
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
