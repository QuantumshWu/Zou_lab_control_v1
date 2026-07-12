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
or in a background thread (``start()``); the hub is the only shared state.  The
background loop is DATA-paced, NEVER rate-capped: a shot that acquires publishes and
loops straight into the next (a blocking device read / the hardware sets the real
cadence), so a scan or a fast readout runs at full hardware speed.  Display refresh
throttling belongs to the CONSUMER (the task console's per-panel ``update_ms``), not
here -- the acquisition loop must not be slowed to a display rate.
"""

from __future__ import annotations

import ast
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from Zou_lab_control._paths import CALIBRATION_DIR
from ..devices.base import EXCLUSIVE, OBSERVE
from ..core.analysis import grid_shape_tuple
from ..core.signals import (
    NO_LINEAGE,
    SignalHub,
    SignalSchema,
    SignalTensor,
    TensorPatch,
)
from ..devices.base import CameraDevice
from .measurement import (
    SWEEP_API_SLOT,
    SWEEP_SCAN_SLOT,
    fire_api_sweep_point,
    prepare_hardware_scan,
    program_completion_timeout,
    reducer_data_shape,
    triggered_frames,
)
from .signal_expr import DEFAULT_SOURCE, ProcessorSignalSnapshot, SignalExpr, hub_namespace

# The background loop's ONLY two time constants -- and NEITHER is an acquisition-rate cap.  A pass that
# published (an acquiring measurement, a reactive processor whose input advanced) loops immediately, so
# its cadence comes from the blocking device read / the hardware, never from a throttle here.
_IDLE_POLL_S = 0.05     # FALLBACK only: a pass that published NOTHING waits on the hub EVENT (wakes the
                        # instant any signal is published) with this as the max idle before a re-check, so a
                        # truly-idle node does not hot-spin -- it never throttles a reactive consumer, which
                        # wakes on its input, not on this timeout.
_ERROR_BACKOFF_S = 0.2  # after a shot raises, wait this before retrying, so a wedged source does not spin
                        # the error banner.
IMAGE_STREAM_HISTORY = 8   # short shot-coherence buffer for full-frame image streams (producer-declared)


@dataclass(frozen=True)
class SignalSpec:
    """The complete, authoritative contract for one logic-node output.

    A node declares these ONCE (``output_specs``); the GUI reads them so a plot can
    set its axis label/unit from the producing measurement (not a hard-coded per-kind
    string) and a node's "publishes" legend reads as ``occupied  (35,)  per-site 0/1
    occupancy`` -- every output named, shaped and explained.  ``name`` is the FULL hub
    signal name (with the node's prefix), so a consumer maps a signal straight to its
    meaning.

    Every physical signal has exactly ``(R, P, *data_shape)`` axes.
    ``point_shape`` is the logical scan geometry whose product is the one physical
    P axis; ``data_shape`` is retained verbatim.  Both are non-empty,
    including scalars (``(1,)`` + ``(1,)``).  There is no unstructured/``None``
    branch and no dimensionality heuristic.

    This object is the producer-side single source of truth.  :meth:`to_schema`
    creates the transport-level :class:`SignalSchema` used by :class:`SignalHub`;
    UI consumers read that registered schema rather than reconstructing structure.

    ``history`` declares the hub storage depth for THIS signal.  ``None`` means
    "use the hub default" (deep histories for small scalar/count streams);
    high-throughput image streams use a compact buffer because their own repeat
    block already carries averaging/history semantics and the hub only needs a
    short shot-coherence backlog."""

    name: str               # full published signal name (incl. the node's prefix)
    label: str              # axis / legend label, e.g. "loading rate"
    unit: str = ""          # physical unit, e.g. "s" / "K" (blank = dimensionless)
    description: str = ""    # one-line human meaning for the publishes legend
    points_shape: tuple[int, ...] = (1,)
    data_shape: tuple[int, ...] = (1,)
    dtype: Any = None
    repeat_capacity: int | None = None
    history: int | None = None          # per-signal hub ring depth override (None = hub default)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        points = tuple(int(n) for n in self.points_shape)
        data = tuple(int(n) for n in self.data_shape)
        if not points or any(n < 1 for n in points):
            raise ValueError(
                f"SignalSpec({self.name!r}): points_shape must be a non-empty tuple of "
                f"positive sizes, got {self.points_shape!r}.")
        if not data or any(n < 1 for n in data):
            raise ValueError(
                f"SignalSpec({self.name!r}): data_shape must be a non-empty tuple of "
                f"positive sizes, got {self.data_shape!r}.")
        object.__setattr__(self, "points_shape", points)
        object.__setattr__(self, "data_shape", data)
        if self.repeat_capacity is not None and int(self.repeat_capacity) < 1:
            raise ValueError(
                f"SignalSpec({self.name!r}): repeat_capacity must be >= 1 or None, "
                f"got {self.repeat_capacity!r}.")
        if self.repeat_capacity is not None:
            object.__setattr__(self, "repeat_capacity", int(self.repeat_capacity))
        if self.history is not None and int(self.history) < 1:
            raise ValueError(f"SignalSpec({self.name!r}): history must be >= 1 or None, got {self.history!r}.")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

        # Reuse SignalSchema's dtype/shape validation at declaration time.  The
        # returned object is discarded; to_schema() adds the final runtime dtype.
        self.to_schema()

    @property
    def axis_label(self) -> str:
        """``label (unit)`` for a plot axis, or just ``label`` when dimensionless."""
        return f"{self.label} ({self.unit})" if self.unit else self.label

    def to_schema(self, *, dtype=None) -> SignalSchema:
        """Create the one transport schema represented by this declaration.

        ``dtype`` is supplied by the unified publish boundary when this spec left
        it unbound.  Producers that publish patches declare a dtype up front so
        the hub can initialize their store before applying the first patch.
        """

        resolved_dtype = self.dtype if self.dtype is not None else dtype
        return SignalSchema(
            point_shape=self.points_shape,
            data_shape=self.data_shape,
            dtype=resolved_dtype,
            repeat_capacity=self.repeat_capacity,
            label=self.label,
            unit=self.unit,
            description=self.description,
            metadata=self.metadata,
        )


def grid_for_points(grid, points) -> tuple:
    """Validate an optional display grid against logical point geometry.

    ``SignalSchema.point_shape`` is authoritative and the physical P axis is its
    product.  A separate ``grid`` is only a presentation alias and is accepted
    when it has the same point count; it never reshapes ``data_shape``.
    """
    pts = tuple(int(n) for n in (points or ()))
    g = tuple(int(n) for n in (grid or ()))
    if pts and g and int(np.prod(g)) == int(np.prod(pts)):
        return g
    return ()


def format_dims(dims) -> str:
    """The ONE spelling of a dims tuple for the GUI: axes joined by the ``×`` glyph
    (``40×20``), and ``"1"`` for an empty/scalar shape.  EVERY surface that turns a
    signal's shape into a display string routes through here, so ``(40, 20)`` (numpy-tuple
    spelling) can never appear beside ``40×20`` again -- the single source :func:`describe_shape`
    and the flow-graph / picker labels all share."""
    parts = tuple(str(int(n)) for n in (dims or ()))
    return "×".join(parts) if parts else "1"


def contract_shape_label(repeat, points_shape, data_shape, grid_shape=None) -> str:
    """The ONE spelling of the canonical ``repeat × points × (data)`` contract shape.  A ``grid_shape``
    reshapes ONLY the swept points (never the data), and only when it divides them (:func:`grid_for_points`).
    Shared by :func:`describe_shape` (value-driven, R = the real block's leading axis) and the console's
    schema-driven declared path (R = the schema's repeat capacity), so a signal reads IDENTICALLY whether
    a value is buffered yet -- the grammar literal lives here alone and can never drift between surfaces."""
    ps = tuple(int(n) for n in (points_shape or ()))
    ds = tuple(int(n) for n in (data_shape or ()))
    gs = grid_for_points(grid_shape, ps)
    return f"{int(repeat)} × {format_dims(gs or ps)} × ({format_dims(ds)})"


def describe_shape(value, *, points_shape=None, data_shape=None, grid_shape=None) -> str:
    """A standardized shape string read straight from a published VALUE -- the SINGLE
    way the GUI says what a signal looks like, AUTO-EXTRACTED from real data rather
    than a hand-typed name->format map (which silently drifts from what a node really
    emits).  ``scalar`` for a 0-d / Python number; ``None`` -> ``"—"`` (no value yet).

    When the value is a registered signal tensor (its shape matches the declared
    physical ``(repeat, prod(points_shape), *data_shape)``) it is shown in contract form
    ``repeat × points × (data)`` -- ALWAYS all three groups (the physical P axis is mandatory, never
    dropped): a 1-D scan ``5 × 8 × (3)``; a 2-D scan ``5 × (4×5) × (1)`` via ``grid_shape`` reshaping the
    SWEPT POINTS; a no-scan single-point signal is ``5 × 1 × (...)`` -- a camera/judged frame
    ``5 × 1 × (96×128)``, per-site occupancy ``5 × 1 × (35)``.  ``grid_shape`` is ONLY a 2-D SCAN's
    points reshape -- it is NEVER applied to the DATA (the 35 sites read ``(35)``, not ``5×7``: that
    layout is the sitemap's display concern, #H3v-3).  Otherwise the raw numpy shape (``(35,)`` /
    ``(96, 128)``)."""
    if value is None:
        return "—"
    shape = tuple(int(n) for n in np.shape(value))
    if shape == ():
        return "scalar"
    ps = tuple(int(n) for n in (points_shape or ()))
    dsh = tuple(int(n) for n in (data_shape or ()))
    if ps and dsh and len(shape) == 2 + len(dsh) \
            and shape[1] == int(np.prod(ps, dtype=np.int64)) \
            and tuple(shape[2:]) == dsh:
        return contract_shape_label(shape[0], ps, dsh, grid_shape)   # R × P × (data), one grammar
    # Raw shape (schema unknown): the SAME ``×`` spelling as the contract form and the flow-graph
    # labels -- never the numpy-tuple ``(96, 128)`` that made the same signal read two different ways.
    return f"({format_dims(shape)})"


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
    # The node's device-ACCESS declaration: ``{attribute name: EXCLUSIVE | OBSERVE}``.
    # EXCLUSIVE = the node DRIVES that hardware (arms the camera / prepares+fires the
    # streamer): these instances are what the console's mutual exclusion intersects -- nodes
    # on disjoint hardware coexist (the monitor camera's live view keeps running while the
    # main camera's calibration starts).  OBSERVE = the node only READS state: the base's
    # ``__setattr__`` narrows the assigned device to its read-only view
    # (:func:`devices.base.read_only`), so the declaration IS the capability -- a later edit
    # that tries to drive an observed device raises instead of silently fighting the
    # exclusive owner.  Declared on the NODE (not a GUI kind-string table) so a
    # notebook-injected ``running_nodes=`` node obeys the SAME rules as a GUI row.  Empty
    # for a reactive Processor (reads only hub signals).  ``start()`` verifies every
    # declared name is a real attribute, so a declaration can never silently point at
    # nothing (the drift a private-name rename once caused).
    _devices: dict = {}
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
        # PER-RUN provenance: the device BASE-STATE + acquisition params captured ONCE at the start of a
        # measurement run, so a saved figure records "what the apparatus was doing when this data was
        # taken".  Captured in ``start()`` (a fresh run = a fresh base state) and held CONSTANT for the
        # whole run -- NOT re-snapshotted per shot (a scanned parameter's point-to-point change IS the
        # scan, already in the data).  ``provenance_snapshot()`` returns this cached dict, computing +
        # caching it once on first read for a notebook ``step()`` loop that never called ``start()``.
        self._provenance: dict[str, object] | None = None
        # Schemas installed by THIS instance.  Besides avoiding duplicate work,
        # this is the ownership proof required for an explicit dynamic-layout
        # replacement (for example a camera ROI or calibration-site change).
        self._registered_output_schemas: dict[str, SignalSchema] = {}
        self._registered_output_history: dict[str, int | None] = {}
        self._runtime_start_authority: object | None = None

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
        self._register_output_schemas(named)
        # Tag every signal of this shot with the SOURCE-shot id shot() set (mint for an acquiring node,
        # inherited for a reactive processor; None -> NO_LINEAGE for a free-running publish), so a derived
        # signal and the frame it came from share ONE id and the console can show them as one shot.  Clear
        # after, so a later no-op/frozen shot never re-uses a stale id (#shot-clock).
        self.hub.publish(named, provenance=self._current_source_shot)
        self._current_source_shot = None
        self.shots += 1
        # Provenance is the device BASE-STATE of ONE measurement run -- captured ONCE at ``start()``,
        # NOT re-snapshotted every shot: within a run the apparatus base state is constant, and what
        # DOES change point-to-point (a scanned parameter) is the scan itself, already in the data.  A
        # notebook ``step()`` loop with no ``start()`` still gets provenance via the lazy fallback in
        # ``provenance_snapshot()`` (captured on first read, then cached), so this hot path stays clean.
        return named

    @staticmethod
    def _payload_dtype(value) -> np.dtype:
        if isinstance(value, SignalTensor):
            return value.data.dtype
        if isinstance(value, TensorPatch):
            return np.asarray(value.values).dtype
        return np.asarray(value).dtype

    def _register_output_schemas(self, values: Mapping[str, object]) -> None:
        """Resolve ``SignalSpec`` and register every value before atomic publish.

        A declaration may leave dtype unbound until hardware returns its first
        value.  Structural changes are allowed only when this same node instance
        installed the previous schema; the replacement starts a new hub schema
        version and discards incompatible history.  Shape is never inferred from
        ndarray rank.
        """

        # LOUD by design: swallowing output_specs errors once downgraded image
        # streams to the deep default ring and consumed multiple GiB.
        specs = {str(spec.name): spec for spec in self.output_specs()}
        missing = sorted(str(name) for name in values if str(name) not in specs)
        if missing:
            raise ValueError(
                f"{type(self).__name__} has no SignalSpec for published signal(s) {missing}.")

        for raw_name, payload in values.items():
            name = str(raw_name)
            spec = specs[name]
            schema = spec.to_schema(dtype=self._payload_dtype(payload))
            previous = self._registered_output_schemas.get(name)
            try:
                current = self.hub.schema(name)
            except KeyError:
                current = None

            if previous is not None and previous.same_definition(schema):
                if current is None or not current.same_definition(previous):
                    raise RuntimeError(
                        f"{type(self).__name__} lost ownership of unchanged schema {name!r}.")
                if self._registered_output_history.get(name) != spec.history:
                    self.hub.configure_signal(name, history=spec.history)
                    self._registered_output_history[name] = spec.history
                # Hot path: a scan may publish thousands of point patches.  Its
                # immutable schema was registered once; avoid repeated hub locks
                # and metadata fingerprints for every point.
                continue

            replace_schema = previous is not None and not previous.same_definition(schema)
            if replace_schema:
                if current is None or not current.same_definition(previous):
                    raise RuntimeError(
                        f"{type(self).__name__} cannot replace schema for {name!r}: the hub no "
                        "longer carries the definition this node installed.")
            elif current is not None and not current.same_definition(schema):
                raise ValueError(
                    f"signal {name!r} already has an incompatible schema from another "
                    "producer; namespace one of the nodes instead of replacing it.")

            self.hub.register_signal(
                name,
                schema,
                history=spec.history,
                replace=replace_schema,
                initialize=isinstance(payload, TensorPatch),
            )
            self._registered_output_schemas[name] = schema
            self._registered_output_history[name] = spec.history

    def _inherit_output_schema_ownership(self, previous: "LogicNode") -> None:
        """Transfer the schema proof for outputs retained by a rebuilt node instance.

        A Task-console row may be stopped, rebuilt with different acquisition geometry, and
        restarted under the SAME signal names.  The lingering hub schema belongs to that logical
        row, but a fresh Python node has no local proof that it installed it; without an explicit
        handoff its first changed-shape frame is correctly rejected as an attempted overwrite by an
        unrelated producer.  Once ``previous`` is stopped, the row owner calls this method before
        starting the replacement.  Only schemas that (a) were installed by ``previous``, (b) are
        still byte-for-byte the hub's current definitions, and (c) remain declared by this node are
        inherited.  The replacement can then use the normal ``replace=True`` path on its first
        changed-shape publish, which atomically starts a new schema version and discards incompatible
        history.  No canonical validation is bypassed and no unrelated producer can be adopted.
        """

        if previous is self:
            return
        if not isinstance(previous, LogicNode) or previous.hub is not self.hub:
            raise ValueError("schema ownership can transfer only between LogicNodes on the same SignalHub.")
        if previous.running:
            raise RuntimeError("stop the previous LogicNode before transferring its output schemas.")

        retained = {str(name) for name in self.published_signals()}
        for name, schema in previous._registered_output_schemas.items():
            if name not in retained:
                continue
            try:
                current = self.hub.schema(name)
            except KeyError:
                continue
            if not current.same_definition(schema):
                raise RuntimeError(
                    f"cannot transfer schema ownership for {name!r}: the hub no longer carries "
                    "the definition installed by the previous node instance.")
            self._registered_output_schemas[name] = schema
            self._registered_output_history[name] = previous._registered_output_history.get(name)

    def _bind_runtime_start_authority(self, authority: object) -> None:
        """Bind this instance to the one migration runtime allowed to start it."""

        if authority is None:
            raise ValueError("runtime start authority cannot be None")
        current = self._runtime_start_authority
        if current is not None and current is not authority:
            raise RuntimeError("LogicNode is already bound to another runtime authority")
        self._runtime_start_authority = authority

    def _start_from_runtime(self, authority: object) -> "LogicNode":
        if self._runtime_start_authority is not authority:
            raise RuntimeError("LogicNode start capability does not match its runtime authority")
        return self._start_impl()

    def start(self) -> "LogicNode":
        if tuple(self.referenced_devices()):
            raise RuntimeError(
                "device-bearing LogicNode.start() requires LegacyRuntimeFence authority"
            )
        return self._start_impl()

    def _start_impl(self) -> "LogicNode":
        """Publish shots from a daemon thread AS FAST AS THE HARDWARE ALLOWS until ``stop()``.

        The loop is DATA-paced, never rate-capped: an acquiring shot blocks on the device read and loops
        straight into the next, so a scan / fast readout runs at full hardware speed.  A pass that
        publishes nothing (a reactive processor whose input has not advanced, or a self-finished node)
        idles a short :data:`_IDLE_POLL_S` slice so it does not hot-spin.  Display-rate throttling is the
        CONSUMER's job (the console's per-panel ``update_ms``), NOT this loop's."""
        if self._thread is not None and self._thread.is_alive():
            return self
        # The _devices declaration must point at REAL attributes: a declared name that does
        # not exist (e.g. after a rename to a private name) would silently drop the device
        # from the occupancy exclusion / the observe narrowing -- fail loud instead.
        missing = [n for n in self._devices if not hasattr(self, n)]
        if missing:
            raise AttributeError(
                f"{type(self).__name__}._devices declares {missing} but the attribute(s) do "
                "not exist -- the declaration must name the node's real device attributes.")
        # Capture the run's provenance ONCE, at the moment this run begins -- the device base-state a
        # saved figure records.  Re-captured on every ``start()`` (a fresh run = a fresh base state),
        # then held CONSTANT for the whole run (start -> stop), so every figure saved during the run
        # carries the same "what the apparatus was doing" record, and a stopped run keeps its last one.
        self.refresh_provenance()
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                seen = self.hub.version                       # version BEFORE this pass -> wake if ANY publish lands past it
                try:
                    applied = self._apply_pending_params()   # owner thread applies edits BETWEEN shots
                    did_work = bool(self.step())             # {} -> reactive no-op / finished: nothing published
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
                    self._stop.wait(_ERROR_BACKOFF_S)
                    continue
                if self.consecutive_errors:
                    # Recovered: clear the banner so a transient hiccup doesn't stick.
                    self.last_error = None
                    self.consecutive_errors = 0
                if not did_work:
                    # Nothing to publish this pass: a reactive processor whose input has not advanced, or a
                    # node that has finished but not yet been reaped.  Wait EVENT-DRIVEN -- wake the instant
                    # ANY signal is published (so a reactive processor reacts to its input with ~zero latency,
                    # and a decoupled scan whose y comes from that processor is NOT throttled by this cap),
                    # else after _IDLE_POLL_S so a truly-idle node does not hot-spin.  A pass that DID publish
                    # never waits here: the blocking device read paces it (hardware speed).
                    self.hub.wait_for_change(seen, timeout=_IDLE_POLL_S, stop=self._stop)

        self._thread = threading.Thread(target=_loop, name=f"zlc-node-{self.prefix or 'main'}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> bool:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("stop timeout must be a non-negative number")
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=float(timeout))
        if thread is not None and thread.is_alive():
            return False
        self._thread = None
        return True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __setattr__(self, name, value):
        # The one enforcement point of the OBSERVE capability: a device assigned to an
        # attribute the class declared OBSERVE is stored as its read-only view, so the code
        # that "only records" a device physically cannot drive it.  Everything else passes
        # straight through (the ``_devices`` lookup is a class-dict read -- negligible).
        if value is not None and self._devices.get(name) == OBSERVE:
            from ..devices.base import ReadOnlyDevice, read_only
            if not isinstance(value, ReadOnlyDevice):
                value = read_only(value)
        object.__setattr__(self, name, value)

    def occupied_devices(self) -> tuple:
        """The hardware device INSTANCES this node DRIVES while running -- the ``EXCLUSIVE``
        entries of the node's ``_devices`` declaration, resolved in ONE place (a declared
        name whose attribute is ``None`` simply contributes nothing, so a sequencer-less
        scan occupies only what it really holds; OBSERVE entries never occupy -- any number
        of observers coexist with the one driver).  Two nodes CONFLICT iff these sets
        intersect by identity; the console's mutual exclusion stops exactly the conflicting
        nodes and leaves everyone on disjoint hardware running."""
        return tuple(d for d in (getattr(self, name, None)
                                 for name, mode in self._devices.items() if mode == EXCLUSIVE)
                     if d is not None)

    def referenced_devices(self) -> tuple:
        """Real devices whose host API this node accesses while running.

        EXCLUSIVE entries drive; OBSERVE entries use an explicitly read-only host surface.
        Wiring or generation dependencies that the node never calls belong in
        :meth:`lifecycle_devices`, not in an artificial OBSERVE claim.
        """
        from ..devices.base import underlying_device
        out: list = []
        seen: set[int] = set()
        for name in self._devices:
            dev = underlying_device(getattr(self, name, None))
            if dev is not None and id(dev) not in seen:
                seen.add(id(dev))
                out.append(dev)
        return tuple(out)

    def lifecycle_devices(self) -> tuple:
        """Devices whose replacement invalidates this node without being host API access."""

        return ()

    def _bare_published_signals(self) -> frozenset:
        """The SHORT (un-prefixed) signal names this node emits -- exactly the keys its
        :meth:`shot` returns.  Subclasses declare THEIR bare names here and nowhere else:
        the base owns the ONE prefix rule (:meth:`published_signals` == ``prefix + bare``),
        the same joining :meth:`step` applies at publish time -- so a node's declaration and
        its publication can never drift, and no subclass ever hand-spells the prefix."""
        return frozenset()

    def _unprefixed_published_signals(self) -> frozenset:
        """Signals this node RELAYS under their canonical bare names, prefix NOT applied --
        an explicit, documented adapter, never a default.  The one user is the pulse scan
        republishing the camera's ``frame_i`` (#E1): the scan REPLACES the camera as the
        device driver, so a frame consumer (the occupancy judge) keeps its binding whichever
        one drives.  Empty everywhere else."""
        return frozenset()

    def published_signals(self) -> frozenset:
        """The FULL hub names this node publishes: the bare declaration behind ``prefix``
        (the base's one joining rule, mirroring :meth:`step`'s publish-time join) plus any
        declared unprefixed relays.  FINAL by convention -- subclasses declare bare names via
        :meth:`_bare_published_signals`, they never re-spell the join (contract-tested)."""
        names = {self.prefix + str(n) for n in self._bare_published_signals()}
        names.update(str(n) for n in self._unprefixed_published_signals())
        return frozenset(names)

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        """One :class:`SignalSpec` per BARE output name -- the LABEL / unit / one-line meaning
        the GUI shows.  The base derives a plain spec (label = name) for every declared bare
        name; a measurement/processor overrides this to give each output a real label, unit
        and description -- spelled on the SHORT name, the prefix join stays the base's."""
        return tuple(SignalSpec(str(name), str(name))
                     for name in sorted({*self._bare_published_signals(),
                                         *self._unprefixed_published_signals()}))

    def output_specs(self) -> tuple[SignalSpec, ...]:
        """The published-name specs: each bare spec re-keyed by the SAME prefix rule
        :meth:`published_signals` uses (an unprefixed relay keeps its bare name), so a
        consumer's ``signal_spec(full_hub_name)`` lookup always matches.  FINAL by
        convention -- subclasses override :meth:`_bare_output_specs` only."""
        from dataclasses import replace as _dc_replace
        unprefixed = {str(n) for n in self._unprefixed_published_signals()}
        return tuple(spec if str(spec.name) in unprefixed
                     else _dc_replace(spec, name=self.prefix + str(spec.name))
                     for spec in self._bare_output_specs())

    def signal_spec(self, name: str) -> SignalSpec | None:
        """The :class:`SignalSpec` for one published (FULL) signal name, or ``None``."""
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

    # ----------------------------------------------------- acquisition provenance
    # "What was the apparatus doing when this data was taken" -- the device BASE-STATE of ONE run,
    # captured ONCE at ``start()`` so a SAVED figure can record it (frontend reads it off the producing
    # node).  It reads ONLY each device's PUBLIC ``.snapshot()`` (the backend-neutral contract on
    # devices/base.py) plus the node's own acquisition params + calibration fingerprint + (for a
    # derived node with no device) the signals it CONSUMES -- never a simulation ground-truth, never a
    # concrete backend -- so it is the SAME provenance a real qCMOS run records (virtual == real,
    # guarded by test_virtual_equals_real_contract).  Every layer's node is handled GENERICALLY (probe
    # for held devices / a ``consumes`` list / a ``session`` -- no per-node-type if/else): a measurement
    # yields device snapshots, a processor yields its consumed inputs, a task yields whatever it holds.
    def _snapshot_devices(self) -> dict[str, object]:
        """The public ``.snapshot()`` of every device this node HOLDS, keyed by role.

        GENERIC (no per-node-type ``if``): it probes the device attributes a producing node MAY own --
        a directly-held ``camera`` / ``sequencer`` (a measurement / camera task) and a held
        ``devices`` bundle (a task that carries the whole session) -- and includes each one that is
        present AND exposes the ``.snapshot()`` device contract.  A pure processor holds none of these
        so it yields an empty dict.  A snapshot that raises maps its role to the error text rather than
        wedging a save; a ``devices`` bundle's snapshot (already a role-keyed dict) is merged flat so a
        task and a measurement record the SAME per-role device keys."""
        out: dict[str, object] = {}
        for role in ("camera", "sequencer"):
            device = getattr(self, role, None)
            snap = getattr(device, "snapshot", None)
            if device is None or not callable(snap):
                continue
            try:
                out[role] = snap()
            except Exception as exc:                     # a device must never break a figure save
                out[role] = f"<snapshot failed: {type(exc).__name__}: {exc}>"
        # A node that instead carries the whole session's device BUNDLE (``self.devices`` with its own
        # ``.snapshot()`` returning a role-keyed dict, e.g. a session-holding task) contributes every
        # role it does not already have directly -- so a task records the same per-role device state a
        # measurement does, still generically (probe the attribute, no node-type branch).
        bundle_snap = getattr(getattr(self, "devices", None), "snapshot", None)
        if callable(bundle_snap):
            try:
                bundle = bundle_snap()
            except Exception:
                bundle = None
            if isinstance(bundle, dict):
                for role, snap in bundle.items():
                    out.setdefault(str(role), snap)
        return out

    def _calibration_fingerprint(self) -> dict[str, object] | None:
        """A small, human-readable fingerprint of the calibration this node reads with, or ``None``.

        Records WHICH calibration the data was judged against (site count + the readout method +
        the source file if the calibration knows one) without dumping the whole object -- enough to
        answer "was this the same calibration".  Reads only public attributes; absent for a node
        that carries no calibration."""
        cal = getattr(self, "calibration", None)
        if cal is None:
            return None
        fp: dict[str, object] = {}
        centers = getattr(cal, "centers", None)
        if centers is not None:
            try:
                fp["n_sites"] = int(np.asarray(centers).shape[0])
            except Exception:
                pass
        meta = getattr(cal, "metadata", None)
        if isinstance(meta, dict):
            for key in ("threshold_method", "source_path", "thresholds_calibrated"):
                if key in meta:
                    fp[key] = meta[key]
        return fp or None

    def refresh_provenance(self) -> None:
        """Re-capture this run's provenance NOW and cache it.  Called once by :meth:`start` at the
        beginning of a run (a fresh run = a fresh device base-state); a node with no ``start()`` gets
        the same one-shot capture lazily via :meth:`provenance_snapshot`."""
        self._provenance = self._collect_provenance()

    def _collect_provenance(self) -> dict[str, object]:
        """Assemble the provenance dict GENERICALLY for any layer's node: the held devices' snapshots
        (measurement / camera task) + the consumed input signals (a derived processor with no device)
        + this node's acquisition params + calibration fingerprint + node/layer identity + a
        wall-clock timestamp.  Every section is included only when the generic probe finds it, so a
        measurement, a processor and a task each record what they actually have -- no node-type
        branch."""
        prov: dict[str, object] = {
            "node": self.display_label,
            "layer": self.layer,
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        devices = self._snapshot_devices()
        if devices:
            prov["devices"] = devices
        # A DERIVED node (a processor) holds no device, so its provenance is otherwise empty and
        # meaningless.  Give it substance GENERICALLY: if the node declares the signals it CONSUMES
        # (the ``consumes`` attribute every Processor carries), record them -- so a processor's
        # provenance says WHICH upstream signals it transformed, and the console can walk them to the
        # producing measurement's device state (upstream chaining).  Probed by attribute, not by type.
        consumes = getattr(self, "consumes", None)
        if consumes:
            prov["consumes"] = [str(c) for c in consumes]
        try:
            params = self.acquisition_parameters()
        except Exception:
            params = {}
        if params:
            prov["acquisition_parameters"] = dict(params)
        fingerprint = self._calibration_fingerprint()
        if fingerprint is not None:
            prov["calibration_fingerprint"] = fingerprint
        return prov

    def provenance_snapshot(self) -> dict[str, object]:
        """The device base-state + acquisition context of the run that produced this node's data --
        the record a saved figure keeps so a reader sees "what the apparatus was doing when this was
        taken".

        Returns the per-run snapshot captured ONCE at :meth:`start` (constant for the whole run, kept
        after stop).  A notebook ``step()`` loop that never called ``start()`` has no cached snapshot
        yet, so the FIRST read computes it and caches it -- from then on it is constant too (never
        re-snapshotted per shot).  Always a plain dict of picklable values (device ``.snapshot()``
        dicts + scalars), so it round-trips through the saved-figure npz."""
        if self._provenance is None:
            self._provenance = self._collect_provenance()
        return dict(self._provenance)


# ===================================================================== logic node kinds
# The three concrete logic node KINDS that compose a task console.  All share the
# LogicNode worker-loop/publish/param-queue/cancel infrastructure above; they differ
# only in WHERE their per-shot values come from:
#   Measurement -- ACQUIRES from devices (camera/sequencer); OWNS the repeat axis (fills a block).
#   Processor   -- TRANSFORMS hub signals into derived signals (no acquisition).
#   Task        -- ORCHESTRATES the others over a multi-step flow, with mid-run output.
class Measurement(LogicNode):
    """A logic node that ACQUIRES data from devices and publishes named signals.

    A Measurement OWNS the repeat axis: each concrete measurement FILLS a physical
    ``(R,P,*data_shape)`` BLOCK every shot (the camera's depth-``repeat``
    ring, a scan's raw block) and publishes it whole.  It does NOT collapse the repeat axis --
    HOW the repeats are combined for viewing (average / add / roll / create) is the PLOT's
    ``repeat_mode`` (display-only), the SINGLE place a repeat axis is collapsed.  So there are
    exactly two repeat knobs in the whole pipeline: ``repeat`` here (how many shots to keep, 0 = ∞
    rolls forever) and the plot's ``repeat_mode`` (how to show them).  Concrete measurements
    implement ``shot()``; ``repeat`` is the ONE acquisition param auto-injected by the console."""

    layer = "measurement"
    node_label = "measurement"

    def _set_repeat_ring(self, repeat: int) -> None:
        """The ONE (repeat -> ring depth) law: ``repeat`` 0 = INFINITE (roll a 1-deep ring
        forever, a live view) / K = keep a K-deep block then stop.  Every measurement's ring
        depth derives here -- camera and swept twins once retyped these two lines each."""
        self.repeat = max(0, int(repeat))
        self._ring = max(1, self.repeat)

    @property
    def ring_depth(self) -> int:
        """The declared repeat-ring CAP of this node's primary block: ``max(1, repeat)`` (a finite
        run keeps up to ``repeat`` slices; ``repeat=0`` = ∞ keeps a 1-deep rolling ring).  The
        leading axis of a published block grows 1..ring_depth as shots land -- the ONE source the
        publish-shape guard and the GUI's structure channel (facet=repeat cell count) both read.
        Resolved defensively: a subclass that forgot ``_ring``/``repeat`` still gets a sane 1."""
        return int(getattr(self, "_ring", 0)) or max(1, int(getattr(self, "repeat", 1)))


class Processor(LogicNode):
    """A logic node that TRANSFORMS hub signals into derived signals (the "func" layer).

    It consumes one or more named signals, computes, and publishes -- with NO device
    acquisition of its own.  REACTIVE: construction opens one stateful Hub
    subscription and every shot consumes at most one retained update.  A single
    input is replayed in publication order; multiple inputs advance only at their
    earliest shared provenance.  The Processor therefore never collapses a burst
    to ``latest`` or assembles values from different acquisitions.  If its bounded
    input journal is overrun, the Hub's ``SignalHistoryGap`` propagates loudly.

    Every output independently declares its complete ``SignalSpec``.  Whether a
    transform preserves or reduces an axis is therefore visible in the output
    schema itself, not in a coarse node-level mode string."""

    layer = "processor"
    node_label = "processor"
    provides: tuple[str, ...] = ()
    def __init__(self, hub: SignalHub, *, consumes, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.consumes = tuple(str(c) for c in consumes)
        if not self.consumes:
            raise ValueError(f"{type(self).__name__} requires at least one consumed signal.")
        if len(set(self.consumes)) != len(self.consumes):
            raise ValueError(
                f"{type(self).__name__} consumes contains duplicate signal names: "
                f"{self.consumes!r}.")
        self._input_tensors: dict[str, SignalTensor] = {}
        # SELF-LOOP guard, at the base and at construction (the single source, covering the
        # console AND a bare notebook node).  A processor consuming its OWN published signal is
        # never meaningful: with no lingering value it silently starves forever (its input only
        # advances if it publishes), and with one it becomes a full-CPU self-sustained republish
        # loop whose inherited source-shot id never advances (the shot clock freezes).  Both are
        # far worse to diagnose than this loud reject.  Indirect rings (A->B->A) span nodes this
        # instance cannot see; the console's start-time graph walk catches those.
        own = frozenset(self.published_signals()) & frozenset(self.consumes)
        if own:
            raise ValueError(
                f"{type(self).__name__} would consume its own output signal(s) "
                f"{sorted(own)} -- a self-feedback loop (silent starvation or a runaway "
                "republish spiral).  Pick an upstream signal as the source instead.")

        # Subscribe at construction, before this node can run.  Existing values
        # define the starting boundary; every publication after that boundary is
        # replayed exactly once.  Rebuilding a Processor intentionally starts a
        # new subscription rather than re-processing an arbitrary old latest.
        self._input_subscription = self.hub.signal_cursors(self.consumes)

    def new_inputs(self) -> ProcessorSignalSnapshot | None:
        """Consume the next exact retained input transaction, if one is ready.

        ``None`` is only the ordinary reactive idle state.  A missing journal
        entry, removed stream, or schema epoch change is not idle: the Hub raises
        ``SignalHistoryGap`` and the LogicNode worker records it in ``last_error``.
        """

        update = self.hub.next_coherent_update(
            self.consumes, self._input_subscription, timeout=0)
        if update is None:
            return None
        self._input_subscription = update.cursors
        self._input_tensors = dict(update.tensors)
        self._current_source_shot = (
            None if update.provenance == NO_LINEAGE else int(update.provenance))
        return ProcessorSignalSnapshot(
            self._input_tensors, provenance=int(update.provenance))

    def input_validity(self, physical_shape: tuple[int, int]) -> np.ndarray:
        """Intersection of consumed ``(R,P)`` validity masks.

        A transform may combine several signals only when their physical cells
        align.  Mismatched masks are a schema/wiring error, never something to
        broadcast or infer from ndarray dimensions.
        """

        expected = tuple(int(n) for n in physical_shape)
        masks = []
        for name in self.consumes:
            tensor = self._input_tensors.get(name)
            if tensor is None:
                continue
            if tuple(tensor.valid.shape) != expected:
                raise ValueError(
                    f"processor input {name!r} has R/P shape {tensor.valid.shape}, but the "
                    f"transform result uses {expected}; preserve axes explicitly in the source expression.")
            masks.append(tensor.valid)
        if not masks:
            return np.ones(expected, dtype=bool)
        return np.logical_and.reduce(masks)

    def input_repeat_capacity(self) -> int:
        """Declared upstream repeat capacity for shape-preserving outputs."""

        capacities = {
            int(tensor.schema.repeat_capacity or tensor.data.shape[0])
            for tensor in self._input_tensors.values()
        }
        if len(capacities) > 1:
            raise ValueError(
                f"processor inputs have incompatible repeat capacities {sorted(capacities)}.")
        return next(iter(capacities), 1)

    def input_point_shape(self, point_count: int) -> tuple[int, ...]:
        """Logical point geometry shared by all consumed tensors."""

        shapes = {tuple(tensor.schema.point_shape) for tensor in self._input_tensors.values()}
        if len(shapes) > 1:
            raise ValueError(f"processor inputs have incompatible point shapes {sorted(shapes)}.")
        shape = next(iter(shapes), (int(point_count),))
        if int(np.prod(shape, dtype=np.int64)) != int(point_count):
            raise ValueError(
                f"processor input point_shape {shape} does not flatten to physical P={point_count}.")
        return shape

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

    def _bare_published_signals(self) -> frozenset:
        return frozenset(str(key) for key in self.output_keys())


class OccupancyProcessor(Processor):
    """Per-frame atom detection as a live graph node -- the REAL readout pipeline.

    Consumes a camera ``frame`` BLOCK and runs the SAME ``calibration.detect``
    contract the notebook/real readout uses, judging EACH repeat slice.  THIS is the
    virtual==real split: the camera produces frames (a Measurement); detection is a
    SEPARATE node here -- not one node fabricating every signal.  The calibration (site
    centers + per-site thresholds) comes from a prior calibrate-readout Task, exactly
    as on real hardware.

    It preserves every valid physical ``(R,P)`` input cell and the canonical
    ``(R,P,*data_shape)`` layout.  Camera ``data_shape=(H,W)`` becomes occupancy/count/rate
    ``data_shape=(N,)``, ``(N,)``, and ``(1,)``; ``frame_judged`` retains ``(H,W)``.
    Logical multi-dimensional point geometry is carried by SignalSchema and only
    flattened on the physical P axis.  Static calibration geometry is still a
    physical tensor with no scan: centers ``(1,1,N,2)`` and thresholds
    ``(1,1,N)``.  Site is data, never a sampling-point axis."""

    node_label = "occupancy"
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
            expr = SignalExpr([FRAME_0], DEFAULT_SOURCE)
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
        # Dynamic dimensions are owned by this producer instance.  The calibration
        # usually supplies N/H/W immediately; the first valid input binds anything
        # not known at construction before the hub sees a publication.
        centers = np.asarray(getattr(calibration, "centers", ()))
        n_sites = int(centers.shape[0]) if centers.ndim == 2 and centers.shape[1:] == (2,) else 1
        image_shape = tuple(int(n) for n in (getattr(calibration, "metadata", {}) or {}).get(
            "image_shape", ()))
        self._point_shape: tuple[int, ...] = (1,)
        self._site_shape: tuple[int, ...] = (max(1, n_sites),)
        self._frame_shape: tuple[int, ...] = image_shape if len(image_shape) == 2 else (1, 1)
        self._output_repeat_capacity = 1

    def _resolve_calibration(self):
        if self.calibration is None and self.calibration_source is not None:
            self.calibration = self.calibration_source()
        return self.calibration

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        calibration = self._resolve_calibration()
        if calibration is None:
            return {}
        block = np.asarray(self.source_expr.evaluate(hub_namespace(self.hub, inputs)))
        if block.ndim != 4:
            raise ValueError(
                f"occupancy source must preserve canonical (R,P,*data_shape) axes with "
                f"data_shape=(H,W); got {block.shape}.  "
                "Indexing away R/P in a signal expression is not allowed.")

        repeats, points, height, width = (int(n) for n in block.shape)
        valid = self.input_validity((repeats, points))
        if not np.any(valid):
            return {}

        centers = np.asarray(calibration.centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1:] != (2,) or centers.shape[0] < 1:
            raise ValueError(
                f"occupancy calibration centers must have shape (N,2), got {centers.shape}.")
        n_sites = int(centers.shape[0])
        occupied = np.full((repeats, points, n_sites), np.nan, dtype=float)
        counts = np.full_like(occupied, np.nan)
        thresholds = None

        for repeat_index in range(repeats):
            for point_index in range(points):
                if not valid[repeat_index, point_index]:
                    continue
                image = np.asarray(block[repeat_index, point_index], dtype=float)
                try:
                    detection = calibration.detect(image, method=self.method)
                except ValueError as exc:
                    fallback = self.session_calibration() if callable(self.session_calibration) else None
                    if fallback is None or fallback is calibration:
                        raise ValueError(
                            f"Judge occupancy: the loaded calibration does not fit this frame "
                            f"(frame {image.shape}, method {self.method or 'default'}): {exc}.  "
                            "Recalibrate for this camera or select the matching calibration.") from exc
                    detection = fallback.detect(image, method=self.method)
                    self.calibration = calibration = fallback
                    centers = np.asarray(calibration.centers, dtype=float)
                    if centers.shape != (n_sites, 2):
                        raise ValueError(
                            "fallback calibration changes the site count within one tensor update; "
                            "restart the processor so the schema change is explicit.")

                occ = np.asarray(detection.occupied, dtype=float).reshape(-1)
                cnt = np.asarray(detection.counts, dtype=float).reshape(-1)
                thr = np.asarray(detection.thresholds, dtype=float).reshape(-1)
                if occ.shape != (n_sites,) or cnt.shape != (n_sites,) or thr.shape != (n_sites,):
                    raise ValueError(
                        "calibration.detect returned a site vector inconsistent with centers: "
                        f"occupied={occ.shape}, counts={cnt.shape}, thresholds={thr.shape}, N={n_sites}.")
                occupied[repeat_index, point_index] = occ
                counts[repeat_index, point_index] = cnt
                thresholds = thr

        if thresholds is None:
            return {}
        finite = np.isfinite(occupied)
        denominator = np.sum(finite, axis=-1, keepdims=True)
        rate = np.divide(
            np.nansum(occupied, axis=-1, keepdims=True),
            denominator,
            out=np.full((repeats, points, 1), np.nan, dtype=float),
            where=denominator > 0,
        )

        self._point_shape = self.input_point_shape(points)
        self._site_shape = (n_sites,)
        self._frame_shape = (height, width)
        self._output_repeat_capacity = self.input_repeat_capacity()
        specs = {spec.name: spec for spec in self._bare_output_specs()}

        def tensor(name: str, data: np.ndarray, mask: np.ndarray) -> SignalTensor:
            schema = specs[name].to_schema(dtype=data.dtype)
            return SignalTensor(data, schema, valid=mask)

        static_valid = np.ones((1, 1), dtype=bool)
        return {
            "occupied": tensor("occupied", occupied, valid),
            "counts": tensor("counts", counts, valid),
            "rate": tensor("rate", rate, valid),
            "centers": tensor("centers", centers.reshape(1, 1, n_sites, 2), static_valid),
            "thresholds": tensor(
                "thresholds", thresholds.reshape(1, 1, n_sites), static_valid),
            "frame_judged": tensor("frame_judged", block, valid),
        }

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        """Complete typed schemas for every occupancy output."""
        sites = self._site_shape
        frame = self._frame_shape
        points = self._point_shape
        repeat_capacity = self._output_repeat_capacity
        return (
            SignalSpec("occupied", "occupancy", "",
                       "per-site occupancy (0/1) for every valid repeat/point cell",
                       points_shape=points, data_shape=sites, dtype=np.float64,
                       repeat_capacity=repeat_capacity),
            SignalSpec("counts", "readout counts", "",
                       "per-site integrated readout signal for every valid repeat/point cell",
                       points_shape=points, data_shape=sites, dtype=np.float64,
                       repeat_capacity=repeat_capacity),
            SignalSpec("rate", "loading rate", "",
                       "per-cell mean occupancy across sites",
                       points_shape=points, data_shape=(1,), dtype=np.float64,
                       repeat_capacity=repeat_capacity),
            SignalSpec("centers", "site centre", "px", "site centres in camera pixels",
                       points_shape=(1,), data_shape=(*sites, 2), dtype=np.float64,
                       repeat_capacity=1),
            SignalSpec("thresholds", "threshold", "counts", "per-site bright/dark threshold",
                       points_shape=(1,), data_shape=sites, dtype=np.float64,
                       repeat_capacity=1),
            SignalSpec("frame_judged", "camera image", "counts",
                       "the EXACT frame this occupancy was judged from -- shot-locked to occupied/centers "
                       "(one atomic publish).  Use THIS for a 2D readout image that matches the site map "
                       "(same shot); the camera's live `frame` advances independently and is NOT shot-locked.",
                       points_shape=points, data_shape=frame,
                       repeat_capacity=repeat_capacity,
                       history=IMAGE_STREAM_HISTORY),
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

    def __init__(self, *, prefix: str = "", spec_provider=None):
        self.prefix = str(prefix)
        self.progress = 0.0
        self._latest: dict[str, object] = {}
        self._tensors: dict[str, SignalTensor] = {}
        self._schemas: dict[str, SignalSchema] = {}
        self._spec_provider = spec_provider
        self._version = 0

    def publish(self, **signals) -> None:
        # Human stage text is control-plane status for the task banner, not an
        # object-dtype signal.  Keep latest("stage") for the banner API.
        if "stage" in signals:
            self._latest["stage"] = str(signals.pop("stage"))

        specs = ({str(spec.name): spec for spec in self._spec_provider()}
                 if callable(self._spec_provider) else {})
        for raw_key, value in signals.items():
            key = str(raw_key)
            spec = specs.get(key)
            if spec is None:
                raise ValueError(f"TaskOutput has no SignalSpec for numeric output {key!r}.")
            dtype = value.data.dtype if isinstance(value, SignalTensor) else np.asarray(value).dtype
            schema = spec.to_schema(dtype=dtype)
            tensor = value if isinstance(value, SignalTensor) else SignalTensor.from_value(value, schema)
            schema.assert_compatible(tensor.schema, check_repeat=True)
            self._schemas[key] = schema
            self._tensors[key] = tensor
            self._latest[key] = tensor.data.copy()
            if key == "progress":
                self.progress = float(tensor.data[0, 0, 0])
        self._version += 1

    def latest(self, name: str):
        """The most recent value buffered under ``name`` (or ``None``)."""
        return self._latest.get(str(name))

    def latest_tensor(self, name: str) -> SignalTensor:
        try:
            return self._tensors[str(name)]
        except KeyError as exc:
            raise KeyError(f"no numeric task output {name!r}") from exc

    def schema(self, name: str) -> SignalSchema:
        try:
            return self._schemas[str(name)]
        except KeyError as exc:
            raise KeyError(f"no task output schema {name!r}") from exc

    def names(self) -> list:
        """Names buffered so far (the task's declared ``mid_run`` keys as they arrive)."""
        return list(self._latest)

    @property
    def version(self) -> int:
        """Bumped on every ``publish`` so a viewer can poll for fresh mid-run data."""
        return self._version


class OneShotNode(LogicNode):
    """A logic node whose action runs exactly ONCE, then the node self-stops.

    The one-shot law lives HERE, not retyped per subclass (Task and ProcessorRun once
    each carried their own copy and drifted on the stop semantics):

    * ``_run_once()`` (the subclass body) executes with the stop event CLEAR, so the
      run can poll ``self._stop`` (and hand it to ``camera.acquire``) as a cooperative
      CANCEL -- pressing Stop interrupts a long acquisition mid-run.
    * ``finished`` means "this one-shot has RUN ONCE and will not retry" -- success OR
      failure terminates it -- so it and the stop event are BOTH set in ``finally``.
      Keeping them together is what releases the console's lock on either outcome
      (a body that raises would otherwise leave finished=False forever, finding 7);
      the exception still propagates (finally does not swallow), so a headless
      ``step()`` caller sees it, and failure is expressed by ``result`` staying empty.
    * ``_publish_result()`` (success only -- an exception propagates past it) says what
      the node hands the hub; the default publishes NOTHING.
    """

    def __init__(self, hub: SignalHub, *, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.finished = False
        self.result: dict = {}

    def _run_once(self) -> dict:  # pragma: no cover - abstract
        """The node's single action; its dict return value becomes ``self.result``."""
        raise NotImplementedError

    def _publish_result(self) -> dict[str, object]:
        """Hub signals to publish after a SUCCESSFUL run (default: nothing)."""
        return {}

    def shot(self) -> dict[str, object]:
        try:
            self.result = {str(key): value for key, value in dict(self._run_once()).items()}
        finally:
            self.finished = True
            self._stop.set()
        return self._publish_result()

    def run_to_completion(self):
        """Run the action once synchronously (tests / headless / notebook)."""
        if not self.finished:
            self.step()
        return self


class Task(OneShotNode):
    """A logic node that ORCHESTRATES devices/measurements/processors over a multi-step
    flow and may emit MID-RUN output to a dedicated panel (confocal-style).

    One-shot (see :class:`OneShotNode`): ``step()`` runs the whole ``run(out)`` flow
    once, then self-stops.  A task publishes NOTHING to the hub: its result lives on
    ``self.result`` and its heavy artifact (e.g. a calibration object, saved files) on
    the task instance, while ``run`` writes intermediate frames/progress to its own
    :class:`TaskOutput` buffer (``self.output``, NOT the hub) for the dedicated
    mid-run panel."""

    layer = "task"
    node_label = "task"
    # A task orchestrates the full shot flow (arm the camera, fire the sequencer) -- both
    # shipped tasks (calibrate readout, MOT-field optimize) drive both.  A future task that
    # merely RECORDS a device declares it OBSERVE instead (holding is not occupying, and
    # the base then narrows it to the read-only view).
    _devices = {"camera": EXCLUSIVE, "sequencer": EXCLUSIVE}
    provides: tuple[str, ...] = ()
    # Signals the task streams to its dedicated MID-RUN output panel (via TaskOutput)
    # while it runs -- listed here so the console maps that panel back to this task.
    mid_run: tuple[str, ...] = ()

    def __init__(self, hub: SignalHub, *, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        # Mid-run output is a per-task BUFFER (NOT the hub) -- created up front so the
        # console can bind the task's dedicated panel to it before/while it runs.
        self.output = TaskOutput(prefix=self.prefix, spec_provider=self._bare_output_specs)

    def run(self, out: "TaskOutput") -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _run_once(self) -> dict:
        # The result + mid-run output stay on the INSTANCE (self.result / self.output); a
        # task publishes NOTHING to the hub -- the hub is measurements + processors only
        # (the base's default _publish_result already returns {}).
        return self.run(self.output)

    def output_specs(self) -> tuple[SignalSpec, ...]:
        # A task publishes nothing to the hub (its result lives on the instance, its mid-run
        # output in self.output), so the base's hub-prefix join does not apply: its output
        # specs stay keyed by the BARE ``provides`` / ``mid_run`` names the console documents
        # them under.  (published_signals is the base's empty bare set -- no hub names.)
        return self._bare_output_specs()


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

    # The imaging bracket's three exposure cells, tagged by API slot -- the ONE source for the
    # role<->slot-name convention.  BOTH the exposure WRITER (_collect_bracket_groups sets each cell
    # by name) and the short-readout-frame FINDER (_imaging_layout) read it, so renaming a slot can
    # never desync the two sides (was: "a1"/"a2"/"a3" hand-typed on both sides + in the error text).
    _EXPOSURE_SLOTS = {"reference_1": "a1", "readout": "a2", "reference_2": "a3"}

    # The imaging pulse TEMPLATE the cali loads -- a REAL, inspectable program that IS the
    # long-short-long bracket (3 emCCD frames in one cooling cycle), not a single window the
    # task secretly unrolls.  A bare name resolves to the shipped ``pulses/`` template; an
    # absolute path to the user's own PulseTableState .json.  Each cali pass LOADS it and sets
    # ONLY the three exposure cells BY NAME -- API slots a1/a3 receive the long reference value,
    # and a2 receives the short readout value -- so what is fired == the template file.  The cali does not choose a
    # readout METHOD: it computes ALL methods (box / per-site PSF / uniform PSF) and the
    # OccupancyProcessor picks one.
    # The ONE canonical default imaging-template path (the cali task spec + the generic
    # Pulse-scan measurement both reference THIS, so every GUI form shows the same real,
    # project-relative ``pulses/imaging_template.json`` -- never a bare name that the path
    # widget would anchor to the project ROOT and display as a non-existent file).
    DEFAULT_PULSE_TEMPLATE = "pulses/imaging_template.json"

    @classmethod
    def _resolve_template(cls, pulse_template, sequencer=None):
        """Load the imaging template via the ONE shared FIREABLE resolver (the given path if real,
        else the same-named file shipped under the project ``pulses/`` folder -- where the pulse GUI
        saves and the Browse dialog opens, so the default ``pulses/imaging_template.json`` is a REAL
        inspectable file -- else the in-memory long-short-long default).  Fireable = the authoring
        grid is snapped to the HARDWARE tick, read from the connected ``sequencer`` when given (the
        cali run passes its own) -- an old save with a finer ``time_step_ns`` would otherwise fail
        the clock-grid validation the moment the exposures are set.  The GUI slot preview passes no
        sequencer and gets the streamer-config default grid, same as every other fire path."""
        from ..timing import default_imaging_template, resolve_fireable_template
        return resolve_fireable_template(pulse_template, default_name=cls.DEFAULT_PULSE_TEMPLATE,
                                         default_factory=default_imaging_template, sequencer=sequencer)

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        """What the calibration PRODUCES (off the hub) + streams mid-run -- keyed by the
        bare ``provides`` / ``mid_run`` names."""
        if str(getattr(self, "source", "")) == "saved frames":
            # The selected run, not the connected camera, owns the saved-frame schema.  RunIndex
            # reads this shape while indexing paths, before TaskOutput registers its immutable
            # signal contract, so a 96x128 run cannot be mislabeled as the live camera's 48x60.
            frame_shape = tuple(int(n) for n in self._index_saved_run().image_shape)
        else:
            frame_shape = tuple(int(n) for n in (
                getattr(getattr(self, "camera", None), "frame_shape", ()) or ()))
        if len(frame_shape) != 2 or any(n < 1 for n in frame_shape):
            frame_shape = (1, 1)
        grid = tuple(int(n) for n in getattr(self, "grid_shape", (1, 1)))
        n_sites = max(1, int(np.prod(grid, dtype=np.int64)))
        return (
            SignalSpec(
                "frame", "reference frame", "counts",
                "long-exposure template frame (streamed live)",
                points_shape=(1,), data_shape=frame_shape,
                repeat_capacity=1, history=IMAGE_STREAM_HISTORY),
            SignalSpec(
                "progress", "progress", "", "task completion fraction",
                dtype=np.float64, repeat_capacity=1),
            SignalSpec(
                "centers", "site centres", "px", "fitted site coordinates",
                points_shape=(1,), data_shape=(n_sites, 2),
                dtype=np.float64, repeat_capacity=1),
            SignalSpec(
                "thresholds", "threshold", "counts", "per-site bright/dark threshold",
                points_shape=(1,), data_shape=(n_sites,),
                dtype=np.float64, repeat_capacity=1),
            SignalSpec(
                "n_sites", "site count", "", "number of trap sites found",
                dtype=np.float64, repeat_capacity=1),
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
        from ..devices.camera_trigger import resolve_camera_trigger_channels
        # WHICH line gates a frame is the CAMERA's knowledge (it is wired to it), not the sequencer's
        # -- resolved through the one camera->channels rule (never a re-spelled fallback).
        raw_lanes = state.port_catalog.raw_lanes
        trig = [c for c in resolve_camera_trigger_channels(self.camera) if c in raw_lanes]
        bits = [raw_lanes.index(c) for c in trig]
        frame_periods = [i for i, p in enumerate(state.periods) if any(p.states[b] for b in bits)]
        if len(frame_periods) < 2:
            raise ValueError(
                "the imaging template must trigger the camera at least twice (>=1 long reference "
                "frame + 1 short readout) -- a long-short-long bracket. Open the template in the "
                "pulse GUI and add the camera-trigger frames.")
        readout_slot = self._EXPOSURE_SLOTS["readout"]
        a2 = {int(s.target) for s in state.api_slots if s.name == readout_slot and s.kind == "duration"}
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
        # BY NAME -- each exposure cell has its own unique handle: a1 = first long reference,
        # a2 = short readout, a3 = second long reference.  The two reference handles receive the
        # same configured value; editing either exposure setting changes only those durations.
        # What is fired == the template the operator chose: file == fired.
        template = self._resolve_template(self.pulse_template, self.sequencer)
        try:
            # Each exposure cell carries its OWN api handle (names are unique, like the GUI
            # allocates a fresh a<N> per click): a1 = first long, a2 = short readout, a3 =
            # second long.  Cali sets all three by name; structure stays as loaded.
            template.set_api(self._EXPOSURE_SLOTS["reference_1"], self.reference_exposure)
            template.set_api(self._EXPOSURE_SLOTS["readout"], self.readout_exposure)
            template.set_api(self._EXPOSURE_SLOTS["reference_2"], self.reference_exposure)
        except ValueError as exc:
            raise ValueError(
                f"{exc}  The Calibrate task sets the imaging template's exposures by API slot: tag "
                "the three exposure cells as a1 (first long), a2 (short readout), a3 (second long) "
                "in the pulse GUI (click each duration cell to its API state).") from exc
        readout_index = self._imaging_layout(template)        # WHICH frame is the short readout (a2)
        self._readout_index = readout_index                    # shared with _save_live_frames
        bracket = template.to_sequence(name="reference_bracket")
        from ..devices.camera_trigger import count_camera_trigger_pulses
        n_frames = max(1, count_camera_trigger_pulses(self.camera, bracket))
        n_groups = max(2, int(self.threshold_frames))
        reference_groups: list = []
        readout_per_group: list = []
        for g in range(n_groups):
            if self._stop.is_set():
                break
            # Decoupled shot through the ONE measurement-layer helper: arm the camera for the
            # bracket's N frames, fire the sequencer, read the frames back -- the camera never
            # drives (or even sees) the sequencer; its trigger input gates the readout, real or
            # virtual alike.
            batch = triggered_frames(self.camera, self.sequencer, bracket, n_frames, stop=self._stop)
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


#: The camera's FIRST emCCD event signal -- the default frame every consumer binds.  Derived
#: from :func:`camera_frame_keys` so the name has exactly one spelling in the project.
FRAME_0 = camera_frame_keys(1)[0]


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
    per-trigger assignment stays phase-aligned.)

    The node is PASSIVE: the streamer runs independently (the pulse GUI's On Pulse) and the
    camera just reads what its trigger input gates -- no frames means the view freezes.
    ``sequencer`` records the physical/virtual trigger wiring only.  The node never calls its
    host API, so it is a lifecycle dependency rather than an OBSERVE resource claim."""

    node_label = "camera"
    # Drives only its sensor.  Sequencer is the trigger-wire lifecycle dependency below,
    # not a host-side observation permission.
    _devices = {"camera": EXCLUSIVE}

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
        # publishes the latest native frame as a 1-deep block forever (live monitor).  The camera never
        # averages at the measurement (that was the live stutter) -- finite mode FILLS and publishes the
        # WHOLE per-event block, live mode publishes the latest per-event block.
        super().__init__(hub, prefix=prefix)
        self.camera = camera
        self.sequencer = sequencer
        self.frames_per_cycle = max(1, int(frames_per_cycle))
        # Declare (H, W) at BUILD time from the camera's own contract (``frame_shape`` = ROI else full
        # sensor): the node's structure must be TRUE the moment it is built, not after the first frame
        # -- a restarted measurement whose trigger is not firing yet (e.g. right after a calibration
        # task parked the sequencer on a finished finite program) would otherwise declare data_shape=()
        # and every 2-D panel bound to frame_0 would refuse the hub's perfectly good lingering block.
        # A camera driver must expose this before acquisition: without it a producer cannot register a
        # truthful schema and consumers cannot allocate safely while waiting for the first trigger.
        self._frame_shape: tuple[int, int] = tuple(int(n) for n in (camera.frame_shape or ()))
        if len(self._frame_shape) != 2 or any(n < 1 for n in self._frame_shape):
            raise ValueError(
                f"{type(camera).__name__}.frame_shape must declare positive (H, W) before "
                f"CameraMeasurement is built; got {camera.frame_shape!r}.")
        self._rings = None                               # list of N (ring,1,H,W) blocks; None until 1st frame
        self.set_repeat(repeat)

    def set_repeat(self, repeat: int = 0) -> None:
        """How many photos to KEEP & AVERAGE then STOP, or ``0`` = ∞ (latest-frame live monitor).

        ONE knob, 0 = infinite -- there is NO separate free-run toggle: repeat=K fills a K-deep
        block once and stops; repeat=0 publishes a native-dtype 1-deep latest block forever.  Resets
        the (partly filled) block."""
        self._set_repeat_ring(repeat)                    # block depth: K for finite, 1 for the live view
        self._rings = None
        self._filled = 0

    def lifecycle_devices(self) -> tuple:
        from ..devices.base import underlying_device

        sequencer = underlying_device(self.sequencer)
        return () if sequencer is None else (sequencer,)

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
        # PASSIVE live monitor: the camera is armed and reads whatever the (independently
        # running) streamer triggers -- the node never fires, never reads the streamer's state.
        frames = self.camera.acquire(n, stop=self._stop)
        if not frames:
            # The streamer is not firing a camera-triggering pulse (e.g. the user hit "Stop
            # Pulse") -> no trigger -> no frame.  Publish nothing: the live view holds its last
            # image and FREEZES, exactly as a real externally-triggered camera does.  The gate
            # lives in the DATA SOURCE (only the lowest layer is faked): the virtual camera
            # senses its own trigger wire's firing state; a real qCMOS learns it directly from
            # the absence of hardware trigger edges -- the node sees only "frames or no frames".
            return {}
        # A real cycle WAS acquired -> mint this shot's SOURCE-shot id; every ``frame_i`` published this
        # cycle carries it, and the OccupancyProcessor consuming a chosen ``frame_i`` inherits it, so the
        # judged image / occupancy group with the exact frame they came from (#shot-clock).
        self._current_source_shot = self.hub.next_source_shot()
        frames = [np.asarray(f) for f in frames]
        n = len(frames)
        f0 = frames[0]
        if self.repeat <= 0:
            self._frame_shape = tuple(f0.shape)          # an explicit ROI/schema-version change
            keys = camera_frame_keys(n)
            out = {keys[i]: np.asarray(fi)[None, None] for i, fi in enumerate(frames)}
            self._filled += 1
            return self._assert_declared(out, self._bare_published_signals())

        if (self._rings is None or len(self._rings) != n or self._frame_shape != f0.shape
                or self._rings[0].dtype != f0.dtype):
            self._frame_shape = tuple(f0.shape)          # an explicit ROI/schema-version change
            self._rings = [np.zeros((self._ring, 1, *self._frame_shape), dtype=f0.dtype)
                           for _ in range(n)]            # ONE repeat block per emCCD event of the cycle
            self._filled = 0
        out: dict[str, object] = {}
        keys = camera_frame_keys(n)                      # the single source of the per-event names
        filled_after = min(self._filled + 1, self._ring)
        for i, fi in enumerate(frames):                  # fill EACH event's own ring, publish its block
            # finite: FILL the next slot of the K-deep block
            self._rings[i][min(self._filled, self._ring - 1), 0] = fi
            # ``frame_i`` IS event i's block SO FAR: a NATIVE-dtype (filled, 1, H, W) view -- only the
            # repeats that exist are published (no NaN padding, no float64 expansion, no copy here:
            # the hub's ``_stored_array`` makes the one defensive copy).  A panel reduces the repeat
            # axis (average = long exposure of THAT emCCD event).  No lumped ``frame``.
            out[keys[i]] = self._rings[i][:filled_after]
        self._filled = filled_after
        if self.finished:                                # take exactly K cycles, then stop
            self._stop.set()
        return self._assert_declared(out, self._bare_published_signals())

    def _bare_published_signals(self) -> frozenset:
        # ONE block per emCCD event; no lumped `frame`.  Same source as the console's declared picker.
        return frozenset(camera_frame_keys(self.frames_per_cycle))

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        """Camera outputs: ONE ``frame_i`` per emCCD event of the cycle, each a ``(repeat, 1, H, W)``
        block (repeat × ONE point × the H×W image) -- a panel reduces ITS repeat axis (repeat_mode:
        average = the long exposure of THAT specific emCCD event).  There is NO lumped ``frame``."""
        specs = []
        for bare in sorted(self._bare_published_signals()):
            i = bare.split("_")[-1]
            desc = (f"emCCD event {i} of the cycle: (repeat, 1, H, W) block -- plot reduces repeats "
                    f"(average = long exposure of event {i}).  LIVE camera, advances independently of the "
                    "readout: for a 2D image shot-locked to the site map, bind Judge-occupancy `frame_judged`.")
            specs.append(SignalSpec(
                bare, "camera image", "counts", desc,
                points_shape=(1,), data_shape=self._frame_shape,
                repeat_capacity=self.ring_depth,
                history=IMAGE_STREAM_HISTORY,
            ))
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
                # A cleared region box means "back to the FULL sensor" -- send the one FULL_FRAME
                # sentinel every camera configure() accepts.  Passing None here meant "leave the ROI
                # unchanged" at the backend gate, so clearing the ROI from the GUI was impossible (#B1).
                from ..devices.base import FULL_FRAME

                kw["roi"] = FULL_FRAME
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

    A scan node fills a physical ``(R,P,*data_shape)`` tensor point-by-point.
    It publishes one :class:`TensorPatch` per new point, so a P-point scan stores O(P*D)
    payload instead of P cumulative O(P*D) snapshots.  ONE knob ``repeat``: 0 = ∞
    (roll a 1-deep ring forever); K = keep K passes then STOP.  A subclass sets the swept x
    values + per-point data width once via :meth:`_init_swept_block`, implements only its per-point
    body (calling :meth:`_fill_point`), and reuses the ring state / progress properties / publish /
    finite-scan stop here -- so neither twin (a :class:`ScannedMeasurement` driver or a
    :class:`PulseScanNode`) can drift the contract: the law lives on the common ancestor, not retyped
    per node."""

    @property
    def x_signal(self) -> str:
        """Full hub name of the swept x axis -- ALWAYS ``prefix + x_key``, derived live so it can
        never drift from the prefix the node runs under (a stored copy once did)."""
        return self.prefix + self.x_key

    @property
    def y_signal(self) -> str:
        """Full hub name of the swept y block -- ALWAYS ``prefix + y_key`` (see ``x_signal``)."""
        return self.prefix + self.y_key

    def _init_swept_block(self, *, values, data_shape: tuple, repeat: int,
                          point_shape: tuple | None = None) -> None:
        """Set up the swept x values + the pre-allocated RAW block.  ``repeat`` (0 = ∞) fixes the ring
        depth; ``data_shape`` is the complete per-point data tensor (``(1,)`` for a scalar).
        Requires ``x_key``/``y_key`` already set: the node's short label derives here
        (the instance prefix stem, else the y key) so neither twin retypes the fallback."""
        self.node_label = str(self.prefix).rstrip("_") or str(self.y_key)
        self._set_repeat_ring(repeat)                 # block depth: K passes for finite, 1 (rolling) for INF
        self._index = 0                               # within-pass point index (0..n_points-1)
        self._pass = 0                                # 0-based pass currently being filled
        self._values = np.asarray(values, dtype=float).reshape(-1)
        proposed = tuple(int(n) for n in (point_shape or (self._values.size,)))
        if not proposed or any(n < 1 for n in proposed) \
                or int(np.prod(proposed, dtype=np.int64)) != int(self._values.size):
            raise ValueError(
                f"point_shape {proposed} must be positive and flatten to P={self._values.size}.")
        self._point_shape = proposed
        self._data_shape = tuple(int(n) for n in data_shape)
        if not self._data_shape or any(n < 1 for n in self._data_shape):
            raise ValueError(f"data_shape must be non-empty and positive, got {data_shape!r}.")
        # Local state mirrors the hub store only to build patches and implement a
        # rolling pass reset.  Its point axis is always physically flattened.
        self._raw = np.full(
            (self._ring, self._values.size, *self._data_shape), np.nan, dtype=float)
        self._coordinates: dict[str, np.ndarray] = {self.x_key: self._values.copy()}
        self._x_published = False

    def _swept_publish(self, patch: TensorPatch) -> dict[str, object]:
        """Publish one y patch and publish immutable x coordinates only once."""

        out: dict[str, object] = {self.y_key: patch}
        if not self._x_published:
            out.update({
                name: values.reshape(1, self.n_points, 1)
                for name, values in self._coordinates.items()
            })
            self._x_published = True
        return self._assert_declared(out, self._bare_published_signals())

    def _bare_published_signals(self) -> frozenset:
        """The swept node's data outputs.  Completion is node control state, not a signal."""
        return frozenset((*self._coordinates, self.y_key))

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

    def _fill_point(self, index: int, row) -> TensorPatch:
        """Fill measured ``row`` into the raw block at the current ``(pass, point)`` and advance the
        within-pass index / pass counter.  Clears a reused ring slot on a pass's first point so two
        passes never mix.  The node only FILLS; HOW the repeats combine for display is the plot's job."""
        row = np.asarray(row, dtype=float)
        if row.ndim == 0 and self._data_shape == (1,):
            row = row.reshape(1)
        if row.shape != self._data_shape:
            raise ValueError(
                f"scan point produced shape {row.shape}; declared data_shape is {self._data_shape}.")
        slot = self._pass % self._ring
        reset_rolling_slot = index == 0 and self.repeat <= 0
        if reset_rolling_slot:
            self._raw[slot] = np.nan
        self._raw[slot, index] = row
        if reset_rolling_slot:
            valid = np.zeros(self.n_points, dtype=bool)
            valid[index] = True
            patch = TensorPatch(
                (slot, slice(None)), self._raw[slot].copy(), valid=valid)
        else:
            patch = TensorPatch.point(slot, index, row.copy(), valid=True)
        self._index += 1
        if self._index >= self.n_points:             # this pass complete -> start the next one
            self._index = 0
            self._pass += 1
        return patch

    def step(self) -> dict[str, object]:
        """Publish one complete scan-point transaction and apply the shared stop law.

        Acquisition, schema registration and Hub publication are one transaction from a
        swept node's point cursor.  A subclass may release strategy-specific resources in
        :meth:`_on_step_failure`, but it must not replace this cursor/publish boundary.
        """
        try:
            named = super().step()
        except Exception as exc:
            self._on_step_failure(exc)
            raise
        if self.finished:
            self._stop.set()
        return named

    def _on_step_failure(self, exc: BaseException) -> None:
        """Release resources after a failed point transaction.

        Most swept measurements own no persistent execution resource, so stopping their
        loop is sufficient.  Hardware-backed scans extend this hook to drive their device
        safe without reimplementing :meth:`step`.
        """

        self._stop.set()

    def run_to_completion(self):
        """Synchronously run + publish every remaining scan point (test/headless).

        ``stop()`` is a cancellation boundary for the previous background run,
        not a permanent poison pill for this explicit synchronous resume.  A
        live owner thread may never be resumed concurrently; once it has been
        joined, clear its consumed cancellation token before driving the
        remaining points on the caller thread.
        """
        if self.finished:
            return self
        if self.running:
            raise RuntimeError(
                "cannot run_to_completion while the scan's owner thread is still running; "
                "stop it first.")
        self._stop.clear()
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

    ``<x_key>``         ``(1,P,1)`` full swept x tensor, stable from shot 1 (NaN-free)
    ``<y_key>``         ``(R,P,*data_shape)`` RAW output block -- the node FILLS it point by
                        point and does NOT combine the repeats; a PLOT reduces the repeat axis per
                        its ``repeat_mode`` (average / add / replace / roll / create).  For this
                        reducer's complete declared ``data_shape`` (a per-site reducer uses
                        ``(n_sites,)``; higher-rank output retains every trailing axis).
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
        # A notebook ScannedMeasurement fences each point itself.  This LogicNode is
        # already the long-lived fenced owner, so disable the nested adapter and, when
        # a notebook PulseController carries a managed sequencer facade, clone only that
        # controller with its authority-owned raw endpoint.  The raw endpoint never escapes
        # this node; LegacyRuntimeFence still owns the complete lifetime and claims.
        if hasattr(self.measurement, "run_hardware"):
            self.measurement.run_hardware = None
        pulse = getattr(self.measurement, "pulse", None)
        pulse_sequencer = getattr(pulse, "sequencer", None)
        raw_sequencer = getattr(pulse_sequencer, "_authority_device", None)
        if raw_sequencer is not None:
            import copy

            node_pulse = copy.copy(pulse)
            node_pulse.sequencer = raw_sequencer
            self.measurement.pulse = node_pulse
            self.measurement.sequencer = raw_sequencer
        # UNIFORM contract (#H3n): the physical block is ``(R,P,*data_shape)``; for this reducer
        # the reducer's complete ``data_shape``.  ONE knob ``repeat``, 0 = ∞ (no free-run toggle): repeat=K keeps a
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
        # The swept-block contract (ring depth, RAW (R,P,*data_shape) buffer, progress
        # properties, finite-scan stop) is owned by _SweptBlockMeasurement; we supply the swept x
        # values (the measurement's axis = the single source of truth) and the complete per-point
        # tensor shape declared by its reducer.  There is no scalar-width fallback.
        self._init_swept_block(
            values=measurement.axis.values,
            data_shape=reducer_data_shape(measurement.reducer),
            repeat=repeat,
        )

    def _wrapped_devices(self) -> tuple:
        """The hardware the wrapped :class:`ScannedMeasurement` drives -- this node is a WRAPPER
        (it touches only ``measurement.measure``), so both the occupancy claim and the reference
        set live on the measurement, never mirrored onto the wrapper as a second copy."""
        m = self.measurement
        from ..devices.base import underlying_device
        sequencer = getattr(m, "sequencer", None)
        sequencer = getattr(sequencer, "_authority_device", sequencer)
        return tuple(d for d in (underlying_device(getattr(m, "camera", None)),
                                 underlying_device(sequencer))
                     if d is not None)

    def occupied_devices(self) -> tuple:
        return self._wrapped_devices()

    def referenced_devices(self) -> tuple:
        # A scan drives both wrapped devices, so its referenced set == its occupied set.
        return self._wrapped_devices()

    def shot(self) -> dict[str, object]:
        """Measure ONE scan point and FILL it into the raw ``(R,P,*data_shape)`` block at
        ``(pass, point)`` -- the node only fills, it does NOT combine the repeats (the PLOT's
        ``repeat_mode`` decides how to reduce the repeat axis).  Publishes the FULL raw block + the
        stable x axis every shot (NaN = not-yet-measured).  Raises ``StopIteration`` once finished."""

        if self.finished:
            raise StopIteration("ScannedMeasurementNode: scan already complete.")
        index = self._index
        value = float(self._values[index])
        row = self.measurement.measure(value, index)
        patch = self._fill_point(index, row)              # O(data_shape) point update

        self._current_source_shot = self.hub.next_source_shot()   # one SOURCE-shot per scan point (#shot-clock)
        return self._swept_publish(patch)

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        """x / y axis labels + units come from the swept measurement itself (the scan AXIS for x,
        the REDUCER labels for the curve).  ``y_key`` is the RAW ``(R,P,*data_shape)`` block --
        a plot reduces its repeat axis per ``repeat_mode``."""
        axis = self.measurement.axis
        rlabels = tuple(self.measurement.reducer.labels)          # (xlabel, ylabel, zlabel)
        ylabel = rlabels[1] if len(rlabels) > 1 else self.y_key
        xlabel = str(getattr(axis, "label", "x"))
        xunit = str(getattr(axis, "unit", ""))
        return (
            SignalSpec(
                self.x_key, xlabel, xunit, "scan coordinate",
                points_shape=self._point_shape, data_shape=(1,),
                dtype=np.float64, repeat_capacity=1,
                metadata={"role": "coordinate"},
            ),
            SignalSpec(
                self.y_key, ylabel, "", "measured value at each scan point",
                points_shape=self._point_shape, data_shape=self._data_shape,
                dtype=np.float64, repeat_capacity=self.ring_depth,
            ),
        )


class PulseScanNode(_SweptBlockMeasurement):
    """Run a scan-slot or API-slot pulse sweep and collect an external y stream.

    The node owns only the sequencer.  A scan-slot sweep uploads one complete hardware table and
    fires once; an API-slot sweep resolves and submits one finite pulse per row.  Neither path arms
    a camera or relays a frame.  A separately running measurement/processor pipeline publishes y,
    and each step consumes one fresh, lineage-coherent sample for the next point.
    """

    _devices = {"sequencer": EXCLUSIVE}

    #: Maximum wait for the next ordered y update.  A missing producer aborts the scan loudly;
    #: reusing a previous value would silently associate the wrong observation with this point.
    Y_UPDATE_TIMEOUT_S = 5.0

    def __init__(self, hub: SignalHub, plan, *, x_key: str = "param", y_key: str = "signal",
                 prefix: str = "", repeat: int = 1):
        super().__init__(hub, prefix=prefix)
        self.plan = plan
        self.pulse_state = plan.pulse_state
        self.sweep_kind = str(plan.sweep_kind)
        if self.sweep_kind not in (SWEEP_SCAN_SLOT, SWEEP_API_SLOT):
            raise ValueError(f"unsupported PulseScan sweep kind {self.sweep_kind!r}.")
        # Public coordinates describe both execution strategies.  API mutation handles stay
        # private so opaque a1/a2 names never leak into the Hub namespace.
        self.scan_names = [str(name) for name in plan.scan_names]
        self.scan_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in plan.scan_arrays]
        self.api_handles = [str(name) for name in getattr(plan, "api_handles", ())]
        if not self.scan_names or any(not name for name in self.scan_names):
            raise ValueError("PulseScan requires one non-empty semantic name per scan axis.")
        if len(set(self.scan_names)) != len(self.scan_names):
            raise ValueError(f"PulseScan semantic coordinate names must be unique: {self.scan_names}.")
        if len(self.scan_arrays) != len(self.scan_names):
            raise ValueError("PulseScan scan_names and scan_arrays must have the same length.")
        if self.sweep_kind == SWEEP_API_SLOT and len(self.api_handles) != len(self.scan_arrays):
            raise ValueError("PulseScan API sweep needs one mutation handle per coordinate.")
        if self.sweep_kind == SWEEP_SCAN_SLOT and self.api_handles:
            raise ValueError("PulseScan scan-slot sweep cannot carry API mutation handles.")
        point_counts = {int(array.size) for array in self.scan_arrays}
        if len(point_counts) != 1 or next(iter(point_counts), 0) < 1:
            raise ValueError(
                f"PulseScan coordinate arrays must be non-empty and aligned; sizes={sorted(point_counts)}.")
        point_count = next(iter(point_counts))
        if self.sweep_kind == SWEEP_SCAN_SLOT and int(repeat) > 0 and point_count < 2:
            raise ValueError(
                "a finite PulseScan scan-slot sweep needs at least two table rows; the hardware "
                "counts completed sweeps when its point cursor wraps.  Add another scan point, "
                "use repeat=0 for a continuous one-point stream, or fire a fixed pulse instead.")
        self.sequencer = plan.sequencer
        self.y_expr = plan.y_expr if isinstance(plan.y_expr, SignalExpr) else SignalExpr.from_value(plan.y_expr)
        self._run_started = False
        self._run_program = None
        self._point_program = None
        self._y_cursors = {}
        self._subscribed_y_names: tuple[str, ...] | None = None
        self._selected_y_tensors: dict[str, SignalTensor] = {}
        self._last_y_provenance: int | None = None
        self._fatal_scan_error = False
        self._y_history_reservation = None
        self.y_history_reservation_bytes = 0
        # x_key used to be the generic duplicate ``param``.  The first semantic
        # ScanSlot.name is now the primary x binding; every axis is published once.
        del x_key
        self.x_key = self.scan_names[0]
        # The OUTPUT signal name (user-set #7) comes from the plan; the constructor default is the
        # fallback for callers that don't carry it.
        self.y_key = str(getattr(plan, "y_key", "") or y_key)
        if self.y_key in self.scan_names:
            raise ValueError(
                f"PulseScan y signal {self.y_key!r} collides with a semantic coordinate; "
                "choose a distinct y name.")
        # Optional (n0, n1) grid shape for a 2-D scan: a 2-D panel reduces the raw y block's repeat
        # axis then reshapes the (points,) curve into this map (the node itself never reshapes).
        self.scan_shape = getattr(plan, "scan_shape", None)
        # One scalar per hardware scan-table row.
        swept_values = self.scan_arrays[0].astype(float)
        self._init_swept_block(
            values=swept_values,
            data_shape=(1,),
            repeat=repeat,
            point_shape=(tuple(self.scan_shape) if self.scan_shape else None),
        )
        self._coordinates = {
            name: array.copy() for name, array in zip(self.scan_names, self.scan_arrays)
        }
        self._axis_label = str(plan.axis_label)
        self._axis_unit = str(plan.axis_unit)

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        """One schema per semantic coordinate plus the unique physical y tensor."""
        coordinate_specs = tuple(
            SignalSpec(
                name,
                self._axis_label if index == 0 else name,
                self._axis_unit if index == 0 else "",
                f"semantic scan coordinate {name}",
                points_shape=self._point_shape,
                data_shape=(1,),
                dtype=np.float64,
                repeat_capacity=1,
                metadata={"role": "coordinate", "axis_index": index},
            )
            for index, name in enumerate(self.scan_names)
        )
        return (*coordinate_specs,
            SignalSpec(
                self.y_key, self.y_key, "", "external value at each hardware scan-table row",
                points_shape=self._point_shape, data_shape=(1,),
                dtype=np.float64, repeat_capacity=self.ring_depth,
                metadata={
                    "coordinate_signals": tuple(self.prefix + name for name in self.scan_names),
                    "axis_order": tuple(self.scan_names),
                },
            ),
        )

    def shot(self) -> dict[str, object]:
        if self.finished:
            raise StopIteration("PulseScanNode: scan already complete.")
        try:
            index = self._index
            self._start_execution(index)
            fresh, lineage, snapshot = self._await_y_sample()
            if not fresh:
                raise TimeoutError(
                    f"PulseScan y inputs {self._y_input_names()} produced no next ordered update "
                    f"within {self.Y_UPDATE_TIMEOUT_S:.1f}s; scan aborted to avoid assigning a late "
                    "sample to the wrong point.")
            y = self._read_y(snapshot)
            if self.sweep_kind == SWEEP_API_SLOT:
                self._finish_api_point()
            # The result inherits the external acquisition's lineage; this node mints no camera shot.
            self._current_source_shot = lineage
            patch = self._fill_point(index, y)
            out = self._swept_publish(patch)
            if self.finished and self.sweep_kind == SWEEP_SCAN_SLOT:
                self._finish_scan_slot_run()
        except Exception as exc:
            self._fail_scan(exc)
            raise
        if self.finished:
            self._release_y_history()
        return out

    def _on_step_failure(self, exc: BaseException) -> None:
        """Drive the sequencer safe when any part of a point transaction fails."""

        self._fail_scan(exc)

    def _fail_scan(self, exc: BaseException) -> None:
        """Record one terminal fault, release buffers, and safe the sequencer."""

        # Match the base loop's cooperative-stop semantics: an operator Stop
        # racing a blocked read is a clean cancellation, not a red error banner.
        # A real PulseScan fault sets this flag before setting _stop, so the
        # outer step() boundary can encounter the same exception idempotently.
        if self._stop.is_set() and not self._fatal_scan_error:
            return
        if not self._fatal_scan_error:
            self._fatal_scan_error = True
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.consecutive_errors += 1
        self._stop.set()
        self._release_y_history()
        # Idempotent across the shot() and step() exception boundaries: shot()
        # may already have safed and cleared _run_started before step() sees
        # the same exception.
        if self._run_started:
            try:
                self._abort_execution(force=True)
            except Exception as safe_exc:
                # Preserve the data-alignment fault as the raised/root error,
                # but make a failed hardware-safe transition impossible to
                # miss in health state and logs.
                self.last_error += (
                    f"; SAFE-STATE FAILURE: {type(safe_exc).__name__}: {safe_exc}")
                logging.getLogger(__name__).exception(
                    "PulseScan failed to put the sequencer in safe state")

    def _ensure_y_subscription(self) -> None:
        """Reserve and position every y cursor before the first pulse is fired."""

        if self._subscribed_y_names is not None:
            return
        names = self._y_input_names()
        self._reserve_y_history(names)
        self._y_cursors = self.hub.signal_cursors(names)
        disappeared = [
            name for name, cursor in self._y_cursors.items()
            if cursor.schema_version == 0
        ]
        if disappeared:
            self._release_y_history()
            raise ValueError(
                f"PulseScan y input(s) {disappeared} disappeared before hardware fire.")
        # The plan's expression wiring is immutable for this run.  Cache the
        # validated subscription so a long hardware scan does not parse Python
        # or walk the schema registry once per point.
        self._subscribed_y_names = tuple(names)

    def _start_execution(self, index: int) -> None:
        """Start the selected execution strategy for ``index``.

        A scan-slot run starts once and then streams all rows.  An API-slot run starts one finite
        pulse for every row; the shared subscription remains positioned across those pulses.
        """

        if self.sweep_kind == SWEEP_SCAN_SLOT and self._run_started:
            return
        self._ensure_y_subscription()
        try:
            if self.sweep_kind == SWEEP_SCAN_SLOT:
                self._run_program = prepare_hardware_scan(
                    self.sequencer, self.pulse_state, scan_repeats=self.repeat)
            else:
                api_row = {
                    handle: float(column[index])
                    for handle, column in zip(self.api_handles, self.scan_arrays)
                }
                self._point_program = fire_api_sweep_point(
                    self.sequencer, self.pulse_state, api_row)
        except Exception as exc:
            self._release_y_history()
            try:
                self._abort_execution(force=True)
            except Exception as safe_exc:
                raise RuntimeError(
                    f"pulse scan start failed ({type(exc).__name__}: {exc}) and the "
                    f"sequencer safe-state transition also failed "
                    f"({type(safe_exc).__name__}: {safe_exc})") from exc
            raise
        self._run_started = True

    def _finish_api_point(self) -> None:
        """Wait for the finite API pulse tail before another row can replace it."""

        program = self._point_program
        waiter = getattr(self.sequencer, "wait_done", None)
        if callable(waiter):
            budget = program_completion_timeout(program)
            if not waiter(timeout=budget, stop=self._stop):
                if self._stop.is_set():
                    raise RuntimeError("PulseScan API sweep cancelled while waiting for pulse completion.")
                raise TimeoutError(
                    f"PulseScan API point did not finish within {budget:.1f}s; refusing to "
                    "replace a still-running pulse with the next row.")
        self._point_program = None
        self._run_started = False

    def _finish_scan_slot_run(self) -> None:
        """Wait for the finite hardware-scan tail before releasing the sequencer."""

        program = self._run_program
        waiter = getattr(self.sequencer, "wait_done", None)
        if callable(waiter):
            budget = program_completion_timeout(program)
            if not waiter(timeout=budget, stop=self._stop):
                if self._stop.is_set():
                    raise RuntimeError(
                        "PulseScan scan-slot sweep cancelled while waiting for hardware completion.")
                raise TimeoutError(
                    f"PulseScan scan-slot sweep did not finish within {budget:.1f}s; refusing "
                    "to release a sequencer that may still be playing its tail.")
        self._run_program = None
        self._run_started = False

    def _reserve_y_history(self, names: list[str]) -> None:
        """Preflight y schemas and reserve every finite scan-point update."""

        if not names:
            raise ValueError(
                "PulseScan requires at least one explicit y input; a constant or live helper "
                "expression has no cursor event to associate with a scan point.")
        identity = self._identity_y_source(names)
        if identity and len(names) != 1:
            raise ValueError(
                "PulseScan with multiple y inputs requires an explicit scalar SignalExpr.")

        capacity = self.total_points if self.repeat > 0 else 0
        estimated = 0
        for name in names:
            try:
                schema = self.hub.schema(name)
            except KeyError as exc:
                raise ValueError(
                    f"PulseScan y input {name!r} has no registered SignalSchema.  Start the "
                    "measurement/processor pipeline before firing the hardware scan.") from exc
            if identity and schema.data_shape != (1,):
                raise ValueError(
                    f"PulseScan y input {name!r} has data_shape={schema.data_shape}, not a scalar "
                    "cell.  Select a component or reduce explicitly in SignalExpr before firing.")
            if capacity:
                dtype_bytes = int(schema.dtype.itemsize) if schema.dtype is not None else 8
                cells = int(schema.repeat_capacity or 1) * schema.point_count
                data_items = int(np.prod(schema.data_shape, dtype=np.int64))
                estimated += capacity * cells * data_items * dtype_bytes

        self.y_history_reservation_bytes = estimated
        if capacity:
            self._y_history_reservation = self.hub.reserve_history(names, capacity)
            if estimated >= 1024 * 1024:
                logging.getLogger(__name__).warning(
                    "PulseScan reserves %d ordered updates for %s (estimated payload %.1f MiB).  "
                    "Reducing camera/image data in an upstream processor before the scan avoids "
                    "retaining full frames.",
                    capacity, names, estimated / (1024.0 * 1024.0))

    def _expression_dependencies(self) -> set[str]:
        """Return direct signal names in the y expression, before hardware fire.

        PulseScan deliberately accepts only a pure expression over its selected
        event snapshot.  Hub lookup helpers (``latest/history/tensor/...``)
        would read outside the cursor transaction and can therefore assign a
        newer frame to an older scan point; they are rejected even when they
        happen to return a scalar.
        """

        try:
            tree = ast.parse(self.y_expr.source, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"invalid PulseScan y expression: {exc.msg}") from exc
        loaded = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        stored = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        forbidden = {
            "history", "history_logical", "latest", "tensor", "logical", "valid",
            "schema", "names", "shot",
        }
        bypass = sorted(loaded & forbidden)
        if bypass:
            raise ValueError(
                "PulseScan y expression cannot use live/history Hub helper(s) "
                f"{bypass}; select every signal as an input so reservation and cursors cover it.")
        pure_names = {
            "signal", "np", "numpy", "math", "float", "int", "bool", "abs",
            "min", "max", "sum", "len", "round",
        }
        registered = set(self.hub.registered_names())
        candidates = loaded - pure_names
        direct = candidates & registered
        # Locals introduced inside the expression (e.g. a temporary or a
        # comprehension variable) are not signal dependencies.  A registered
        # name wins over that exemption so shadowing cannot evade reservation.
        unknown = sorted(candidates - direct - stored)
        if unknown:
            raise ValueError(
                f"PulseScan y expression references unregistered signal(s) {unknown}; "
                "start and register every producer before firing the hardware scan.")
        return direct

    def _identity_y_source(self, names: list[str]) -> bool:
        """Whether the expression semantically passes through one input unchanged."""

        tree = ast.parse(self.y_expr.source, mode="exec")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            return False
        assignment = tree.body[0]
        if len(assignment.targets) != 1 \
                or not isinstance(assignment.targets[0], ast.Name) \
                or assignment.targets[0].id != "value" \
                or not isinstance(assignment.value, ast.Name):
            return False
        source_name = assignment.value.id
        if source_name == "signal":
            return len(self.y_expr.inputs) == 1
        return len(names) == 1 and source_name == names[0]

    def _release_y_history(self) -> None:
        reservation = self._y_history_reservation
        self._y_history_reservation = None
        if reservation is not None:
            reservation.release()

    def _request_execution_abort(self, *, force: bool = False) -> None:
        """Request hardware safe without mutating state still owned by the live node thread."""

        should_abort = force or (self._run_started and not self.finished)
        if should_abort and self.sequencer is not None:
            callbacks = []
            for method in ("set_safe_state", "abort"):
                callback = getattr(self.sequencer, method, None)
                if callable(callback) and callback not in callbacks:
                    callbacks.append(callback)
            last_error = None
            for callback in callbacks:
                try:
                    callback()
                    return
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise RuntimeError(
                    "every sequencer safe-state callback failed") from last_error

    def _clear_execution_state(self) -> None:
        self._run_started = False
        self._run_program = None
        self._point_program = None

    def _abort_execution(self, *, force: bool = False) -> None:
        """Immediately put the sequencer safe, then clear owner-local execution state."""

        try:
            self._request_execution_abort(force=force)
        finally:
            self._clear_execution_state()

    def _y_input_names(self) -> list[str]:
        """Hub signals whose next coherent publish constitutes one scan point."""

        if self._subscribed_y_names is not None:
            return list(self._subscribed_y_names)
        names = [str(name) for name in self.y_expr.inputs if str(name)]
        for name in sorted(self._expression_dependencies()):
            if name not in names:
                names.append(name)
        return names

    def _await_y_sample(self) -> tuple[bool, int | None, dict[str, object] | None]:
        """Consume exactly the next retained lineage-coherent y update."""

        names = self._y_input_names()
        if not names:
            self._selected_y_tensors = {}
            return True, None, {}
        update = self.hub.next_coherent_update(
            names,
            self._y_cursors,
            timeout=self.Y_UPDATE_TIMEOUT_S,
            stop=self._stop,
        )
        if update is None:
            return False, None, None
        lineage = int(update.provenance)
        if lineage != NO_LINEAGE and self._last_y_provenance is not None \
                and lineage <= self._last_y_provenance:
            raise ValueError(
                f"PulseScan y provenance must be strictly increasing; got {lineage} after "
                f"{self._last_y_provenance}.  Duplicate/out-of-order source shots cannot be "
                "assigned safely to successive scan points.")
        self._y_cursors = dict(update.cursors)
        self._selected_y_tensors = dict(update.tensors)
        if lineage != NO_LINEAGE:
            self._last_y_provenance = lineage
        return True, lineage, update.values()

    def _read_y(self, snapshot: dict[str, object] | None) -> float:
        """Evaluate one explicit scalar without flattening or choosing a component."""

        # Snapshot-only namespace: no helper in this expression can reach a
        # newer Hub value than the exact cursor-selected tensors above.
        namespace = dict(snapshot or {})
        namespace.update({"np": np, "numpy": np, "math": math})
        value = self.y_expr.evaluate(namespace)
        array = np.asarray(value, dtype=float)
        # A Python/NumPy scalar means the expression explicitly selected or
        # reduced its inputs; accepting it cannot discard an implicit axis.
        if array.ndim == 0:
            return float(array.item())
        if array.ndim != 3 or tuple(array.shape[2:]) != (1,):
            raise ValueError(
                "PulseScan y expression must return a scalar or canonical (R,P,1) tensor; "
                f"got {array.shape}.  Select a component or reduce explicitly in SignalExpr.")

        masks = []
        for name, tensor in self._selected_y_tensors.items():
            if tensor.data.shape[:2] != array.shape[:2]:
                raise ValueError(
                    f"PulseScan y result has R/P={array.shape[:2]}, but input {name!r} has "
                    f"R/P={tensor.data.shape[:2]}; reduce explicitly in SignalExpr.")
            masks.append(tensor.valid)
        valid = np.logical_and.reduce(masks) if masks else np.ones(array.shape[:2], dtype=bool)
        if int(np.count_nonzero(valid)) != 1:
            raise ValueError(
                f"PulseScan y has {int(np.count_nonzero(valid))} valid scalar cells; exactly one "
                "is required per scan point.  Select a repeat/point or reduce explicitly in SignalExpr.")
        return float(array[..., 0][valid][0])

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop collection and stop unfinished pulse playback."""

        self._stop.set()
        try:
            self._request_execution_abort(force=True)
        except BaseException:
            # Still join the software owner, but retain all execution/reservation state so
            # a later recovery can retry safe without a fabricated clean terminal state.
            super().stop(timeout=timeout)
            raise
        if not super().stop(timeout=timeout):
            return False
        self._clear_execution_state()
        self._release_y_history()
        return True


class ProcessorRun(OneShotNode):
    """One-shot DATA-PROCESSING logic node (see :class:`OneShotNode`): runs a
    :class:`ProcessorSpec` ONCE, publishes its result dict to the hub, and self-stops --
    the discrete sibling of :class:`ScannedMeasurementNode` (a finite scan).  It DRIVES
    the spec's ``run(ctx)`` and owns no analysis itself.

    The cooperative-stop event is shared with the run via the context, so a long
    camera grab inside ``run`` cancels cleanly on ``stop()`` (the SOLE-camera-owner
    invariant: the run executes on this node's own thread, never a second acquire)."""

    # One-shot over live-grabbed frames: drives the camera (and fires the sequencer when
    # one is bound); a saved-frames run holds None for both -> occupies nothing.
    _devices = {"camera": EXCLUSIVE, "sequencer": EXCLUSIVE}

    layer = "processor"
    node_label = "processor"

    def __init__(self, hub: SignalHub, spec, *, readout, camera=None,
                 sequencer: object | None = None, params: dict | None = None, prefix: str = ""):
        super().__init__(hub, prefix=prefix)
        self.spec = spec
        self.node_label = getattr(spec, "name", "processor")
        self._readout = readout
        self.camera = camera
        self.sequencer = sequencer
        self._params = dict(params or {})

    def _run_once(self) -> dict:
        from .processor import ProcessorContext

        # The base keeps the stop event CLEAR during the run and hands it to the context,
        # so a long camera grab inside ``spec.run`` cancels cleanly the moment Stop is
        # pressed (setting it up front, as this node once did, would make any run that
        # polls the stop cancel itself instantly).
        ctx = ProcessorContext(
            readout=self._readout, params=self._params,
            camera=self.camera, sequencer=self.sequencer, stop=self._stop)
        return self.spec.run(ctx)

    def _publish_result(self) -> dict[str, object]:
        out = dict(self.result)
        self._current_source_shot = self.hub.next_source_shot()   # one SOURCE-shot for this discrete result (#shot-clock)
        return self._assert_declared(out, tuple(self.spec.result_keys))

    def _bare_published_signals(self) -> frozenset:
        return frozenset(self.spec.result_keys)

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        """Use typed one-shot results as the dynamic schema source.

        Scalar raw results use SignalSpec's formal scalar default.  A non-scalar
        result must be a SignalTensor, forcing the concrete processor to state
        point/data semantics instead of making this host infer them from rank.
        """

        specs = []
        for key in self.spec.result_keys:
            value = self.result.get(key)
            if isinstance(value, SignalTensor):
                schema = value.schema
                specs.append(SignalSpec(
                    str(key), schema.label or str(key), schema.unit, schema.description,
                    points_shape=schema.point_shape,
                    data_shape=schema.data_shape,
                    dtype=schema.dtype,
                    repeat_capacity=schema.repeat_capacity,
                    metadata=schema.metadata,
                ))
            else:
                specs.append(SignalSpec(str(key), str(key)))
        return tuple(specs)


__all__ = [
    "describe_shape",
    "FRAME_0",
    "OneShotNode",
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
