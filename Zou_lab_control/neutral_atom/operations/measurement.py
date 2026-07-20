"""Generic scanned-measurement abstraction (the engine behind every live scan).

A "scanned measurement" sweeps one bound pulse parameter, acquires a few camera
frames per point, and reduces those frames to one declared per-point data tensor.
Detection-time/fidelity and
release-recapture temperature are both this same shape; the only difference is
WHICH slot is scanned (``ScanAxis``), HOW the per-point sequence is built
(``ShotPlan``), and HOW the frames become a y value (``PointReducer``).  Holding
those three roles apart -- and sharing one engine -- means a new measurement is a
new reducer/plan, not a new live-scan loop.

The engine runs through the SAME contract real hardware uses:
``camera.acquire(...)`` for frames, the bound ``PulseController`` for setting the
scanned slot, ``calibration.signals``/``detect`` for per-site quantities, and the
viewer registry (``active_plotter().run``) for the live plot.  It imports no
concrete camera backend and reads no simulation ground truth, so a virtual run
exercises the identical path a real run does (``tests/
test_virtual_equals_real_contract.py`` guards this).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable

import numpy as np

from zlc_data.shape_text import measurement_slug  # one spelling of a node's token

from Zou_lab_control._readout_math import finite_mean
from Zou_lab_control._viewer_registry import active_plotter

from ._spec import REQUIRED, CatalogSpec

from ..core.analysis import estimate_threshold_fidelity, otsu_threshold, positive_int
from ..core.params import ParamDecl
from ..core.calibration import TrapCalibration
from ..core.results import MeasurementTaskResult
from ..core.utils import site_index
from ..devices.base import AcquisitionCancelled
from ..timing.sequence import snap_seconds_to_clock
from ..views.plots import plot_detection_scan


#: Spec-metadata key + the value that routes :meth:`MeasurementSpec.make_node` to the DECOUPLED
#: pulse-scan node (:class:`~.logic.PulseScanNode`) instead of the frame-reducing
#: :class:`~.logic.ScannedMeasurementNode` -- the ONE source read by BOTH the spec that SETS it
#: (``measurements/pulse_scan.py``) and ``make_node`` that DISPATCHES on it, so the tag can't drift.
NODE_META_KEY = "node"
PULSE_SCAN_NODE = "pulse_scan"

#: The two execution semantics supported by PulseScan.  A hardware scan slot is uploaded as one
#: complete table; an API slot is resolved and submitted as one finite pulse per point.  Keeping
#: this discriminator beside the shared execution helpers gives the measurement factory, logic
#: node, and frontend one source instead of parallel mode vocabularies.
SWEEP_SCAN_SLOT = "scan_slot"
SWEEP_API_SLOT = "api_slot"
PULSE_SWEEP_KINDS: tuple[str, str] = (SWEEP_SCAN_SLOT, SWEEP_API_SLOT)

#: The metadata marker every COUPLED-tier measurement (temperature / fidelity-vs-duration /
#: grey-molasses-detuning -- the pulse-scan special cases whose y is reduced INLINE over a loading's
#: frames) tags its spec with -- the ONE source (was hand-typed ``"coupled"`` in each spec), read by
#: the docs and ``tests/test_scan_tier_boundary.py``.
SCAN_TIER_KEY = "scan_tier"
SCAN_TIER_COUPLED = "coupled"


def _replace_running_pulse(sequencer, payload):
    """Stop prior playback, prepare ``payload``, and fire it exactly once."""

    if sequencer is None:
        raise RuntimeError("a pulse scan requires a sequencer")
    stop = getattr(sequencer, "set_safe_state", None)
    if not callable(stop):
        stop = getattr(sequencer, "abort", None)
    if callable(stop):
        stop()
    program = sequencer.prepare(payload)
    sequencer.fire()
    return program


def prepare_hardware_scan(sequencer, pulse_state, *, scan_repeats: int):
    """Upload and fire one complete hardware scan, returning its runtime program.

    The caller supplies a ``PulseTableState`` whose slots and whole scan table are final.  This
    shared execution seam clones it, applies the whole-table repeat count (``0`` = continuous),
    stops prior playback, then performs exactly one ``prepare(state)`` and one ``fire()``.  It
    never resolves individual rows and knows nothing about cameras or reducers.
    """

    if not getattr(pulse_state, "scan_slots", None) or not getattr(pulse_state, "scan_table", None):
        raise ValueError("hardware scan state must contain both scan_slots and scan_table")
    payload = pulse_state.to_dict()
    payload["scan_repeats"] = max(0, int(scan_repeats))
    payload["repeat_forever"] = True
    submitted = type(pulse_state).from_dict(payload)

    return _replace_running_pulse(sequencer, submitted)


def program_completion_timeout(program) -> float:
    """Conservative wait budget for one finite submitted runtime program.

    ``RuntimeSequenceProgram.duration`` covers one complete streamed-table sweep,
    so a finite ``scan_repeats=K`` run needs ``K`` times that duration.  Ordinary
    and API-resolved programs have no scan points and use their duration once.
    Every tail wait shares the same five-second transport slack.
    """

    duration = max(0.0, float(getattr(program, "duration", 0.0) or 0.0))
    scan_points = getattr(program, "scan_points", None)
    repeats = max(1, int(getattr(program, "scan_repeats", 0) or 0)) if scan_points else 1
    return 2.0 * duration * repeats + 5.0


def fire_api_sweep_point(sequencer, pulse_state, values):
    """Resolve and fire one finite API-slot sweep point.

    Any hardware scan slots in the same template are first resolved to their nominal values and
    removed, so selecting an API sweep can never accidentally start the template's persisted FPGA
    table.  ``values`` then overrides the named API handles on a fresh state.  The returned runtime
    program is used by the caller to derive a finite completion timeout before submitting the next
    point.
    """

    base = pulse_state.with_slots_resolved({})
    submitted = base.with_api_resolved(values)
    payload = submitted.to_dict()
    payload["repeat_forever"] = False
    payload["scan_repeats"] = 1
    submitted = type(submitted).from_dict(payload)
    return _replace_running_pulse(sequencer, submitted)


def triggered_frames(camera, sequencer, sequence, frames: int = 1, *, stop=None) -> list:
    """THE arm-before-fire shot: arm the camera, fire N trigger edges, read the frames back.

    This is the SINGLE place the measurement layer pairs a camera with a sequencer --
    every readout / calibration / scan / capture path calls it (never a hand-rolled
    ``prepare(...); fire(...)`` around a camera; ``tests/test_camera_measurement_multitrigger.py``
    greps for offenders).  The ordering is the whole point: the camera is armed FIRST
    (:meth:`~..devices.base.CameraDevice.arm` returns only once the hardware waits for
    triggers), so the fire can never outrun it and the first trigger edge is never lost;
    the frames are then consumed from the camera's own lossless buffer and the camera is
    stood down.

    N FRAMES = N TRIGGER EDGES.  A camera reads out exactly one frame per capture-trigger edge,
    so to read ``frames`` frames the sequencer must EMIT ``frames`` edges on the camera's own
    trigger line.  A sequence's ONE base cycle already carries ``base_triggers`` edges (1 for a
    plain imaging pulse; 2/3 for a release-recapture / long-short-long bracket -- one loading read
    several times); this REPEATS that base cycle just enough that the fired program's total edge
    count reaches ``frames`` (a single-trigger imaging pulse -> ``sequence.repeated(frames)``, N
    independent shots; a bracket that already carries ``frames`` edges fires once, unchanged).  On
    real hardware the FPGA emits those N edges the same way; the virtual streamer's wired camera
    counts them and renders N frames.  Firing a single-trigger pulse ONCE and hoping for N frames
    is the bug this closes -- a real camera would time out waiting for edge 2.

    ``sequencer=None`` degrades to the camera's free-run ``acquire`` (nothing to fire: a
    self-triggering sensor such as a Basler in ``Software`` mode still yields frames; an
    externally triggered camera then honestly reads nothing).  ``stop`` cancels a blocking
    wait cooperatively.  Returns what arrived (backends with a loud fault model raise
    ``TimeoutError`` / ``AcquisitionCancelled`` instead of returning short)."""
    frames = positive_int(frames, "frames")
    if sequencer is None:
        return camera.acquire(frames, stop=stop)
    program = _program_for_frames(camera, sequence, frames)
    camera.arm(frames, stop=stop)          # cancellable arm-before-fire (a Stop breaks a lock wait)
    try:
        sequencer.prepare(program)
        sequencer.fire(program)
        out = camera.read_frames(frames, stop=stop)
        # The camera has its frames, but a FINITE program may still be PLAYING past its last
        # camera window (post-imaging reload / repump / reset segments carry no trigger edge).
        # Returning here would let the caller's next prepare land on a still-RUNNING program --
        # the AXI session treats that as an explicit switch and aborts it, silently truncating
        # the tail on real hardware.  So wait for the program to actually finish: wait_done is
        # fire-anchored, so the frame readout above already consumed most of the play time and
        # this costs only the remainder.  The budget derives from the program's own duration
        # (plus slack for readout/transport); a finite program that STILL is not done then is a
        # wedged streamer -- fail LOUD, never quietly truncate the next shot instead.
        if not getattr(program, "repeat_forever", False):
            waiter = getattr(sequencer, "wait_done", None)
            if callable(waiter):
                budget = program_completion_timeout(program)
                # Thread ``stop`` so this program-tail wait is cooperatively cancellable like the frame read
                # above: without it a Stop pressed here blocks up to ``2*duration+5`` s while the node still
                # reports "running" and holds the camera armed -- the double-acquire wedge the sole-owner
                # invariant exists to prevent.
                if not waiter(timeout=budget, stop=stop):
                    if stop is not None and stop.is_set():
                        raise AcquisitionCancelled(     # a Stop mid-tail is a clean cancel, NOT a wedged streamer
                            "cancelled while waiting for the fired program to finish.")
                    raise TimeoutError(
                        f"sequencer did not finish the fired finite program within {budget:.1f} s "
                        "after its frames were read -- the streamer is wedged (a truncated fire "
                        "would silently cut the program's tail on the next prepare).")
        return out
    except BaseException:
        # A prepare / fire / read / wait fault (an RPyC EOF mid-fire, a STATUS_UNDERFLOW stall, a
        # cancel, a wedged finite program) must NOT leave the FPGA armed / loaded / firing -- drive
        # the sequencer SAFE before re-raising, so a failed shot can never strand outputs high.
        # (The camera is stood down by the finally below.)  On the SUCCESS path the sequencer is
        # left as the caller set it -- wait_done above has proven the finite program completed.
        try:
            sequencer.set_safe_state()
        except Exception:
            pass
        raise
    finally:
        camera.disarm()


def _program_for_frames(camera, sequence, frames: int):
    """Repeat ``sequence``'s base cycle just enough to emit ``frames`` camera-trigger edges.

    ``base_triggers`` = how many capture edges the ONE base cycle carries on the camera's own
    counting line (``count_camera_trigger_pulses``: the camera's ``effective_trigger_channels``,
    ``()`` -> 0 edges for a free-running sensor, whose frames come off its own clock -- the
    program is then fired as-is, never repeated for it).
    ``repeats = ceil(frames / base_triggers)`` reaches ``frames`` edges: a single-trigger imaging
    pulse (base_triggers=1) becomes ``repeated(frames)`` (N independent shots); a bracket already
    carrying >= ``frames`` edges (base_triggers >= frames) fires once, untouched.  A sequence that
    carries NO camera trigger, or one that is already a repeat program / lacks ``repeated`` (a raw
    device payload), is fired as-is -- the edge-faithful count then flows from what actually fires."""
    from ..devices.camera_trigger import count_camera_trigger_pulses

    if frames <= 1 or not hasattr(sequence, "repeated"):
        return sequence
    if getattr(sequence, "repeat_forever", False) or int(getattr(sequence, "repeat_count", 1)) > 1:
        return sequence                      # already a multi-cycle / continuous program
    base_triggers = count_camera_trigger_pulses(camera, sequence)
    if base_triggers <= 0 or base_triggers >= frames:
        return sequence                      # no camera trigger, or the base already carries enough edges
    repeats = -(-frames // base_triggers)    # ceil: reach at least `frames` edges
    return sequence.repeated(repeats) if repeats > 1 else sequence


# --------------------------------------------------------------- declarative spec


# ParamDecl itself lives in the dependency-free leaf ``core.params`` (imported above): the
# devices layer declares its config forms with the SAME record, which a devices -> operations
# import could not do (cycle).  The measurement-layer helpers below stay here -- they read the
# device registry / a DeviceSet, knowledge the leaf must not have.


def device_param(role, device_set, *, key: str | None = None, label: str | None = None,
                 tooltip: str | None = None, default: str | None = None) -> "ParamDecl":
    """The ONE source for a device-selection control on ANY measurement/task form: a ``choice``
    ``ParamDecl`` whose choices are the devices of ``role``'s type in ``device_set`` and whose
    default is the conventional device for that role.  Every spec that uses a device declares its
    role via this -- never a hand-rolled ``choices=s.devices.camera_names()`` and never a hardcoded
    ``s.devices.camera`` -- so a measurement runs on ANY device of the right type by picking it in
    the GUI, and a NEW device domain (an RF source, a DAQ, ...) needs no edit here or in any spec.

    ``role`` is a :class:`~..devices.registry.DeviceDomain` or its key string (``"camera"`` /
    ``"sequencer"`` / a registered ``"rf"``); ``key`` overrides the param key when a spec binds two
    devices of the SAME domain (e.g. ``pump_camera`` and ``probe_camera``)."""
    from ..devices.registry import DEVICE_DOMAINS

    domain = role if hasattr(role, "base_type") else DEVICE_DOMAINS[str(role)]
    names = device_set.device_names(domain.base_type)
    if default is not None and default in names:            # a spec's PREFERRED device for this role
        chosen = default                                    # (e.g. MOT optimise -> "monitor_camera")
    else:
        chosen = device_set.default_device_name(domain.base_type, conventional=domain.key) if names else None
    return ParamDecl(
        key=str(key or domain.key), label=str(label or domain.label), kind="choice",
        choices=names, default=chosen,
        tooltip=str(tooltip or f"Which {domain.label.lower()} this node uses."))


def normalize_device_roles(devices):
    """Normalise a spec's ``devices=[...]`` declaration to ``(param_key, domain, opts)`` triples.

    A role entry is either a bare string / :class:`DeviceDomain` (``"camera"``) or a
    ``(role, opts)`` pair, where ``opts`` may set ``key=`` (two devices of the SAME domain --
    e.g. ``pump_camera`` / ``probe_camera``), ``default=``, ``label=`` or ``tooltip=``.  ONE
    normaliser so the decorator, the param appender (:func:`device_params_for`) and the
    build-wrapper all read a role the same way -- no per-spec spelling of the role shape."""

    from ..devices.registry import DEVICE_DOMAINS

    out = []
    for entry in (devices or ()):
        if isinstance(entry, (tuple, list)):
            role, opts = entry[0], dict(entry[1] if len(entry) > 1 else {})
        else:
            role, opts = entry, {}
        domain = role if hasattr(role, "base_type") else DEVICE_DOMAINS[str(role)]
        key = str(opts.get("key") or domain.key)      # the param key AND the **values key
        out.append((key, domain, opts))
    return tuple(out)


def device_params_for(roles, device_set):
    """The choice :class:`ParamDecl`s for a spec's declared device ``roles`` (one per role) --
    the SINGLE bridge :meth:`CatalogSpec.with_devices_bound` uses to turn ``devices=[...]`` into
    form controls, each built by the ONE :func:`device_param` source."""

    return tuple(
        device_param(domain, device_set, key=key,
                     label=opts.get("label"), tooltip=opts.get("tooltip"),
                     default=opts.get("default"))
        for key, domain, opts in normalize_device_roles(roles))




def axis_range_tuple(value, name: str) -> tuple[float, float, int]:
    """Coerce an ``axis_range`` param value to ``(min, max, points)``.

    Accepts a 3-tuple/list ``(min, max, points)`` (the GUI form's three boxes) and
    validates it -- ``points`` >= 2, ``max`` > ``min``.  Kept dependency-free so
    both the spec build closures and the GUI consumer share ONE interpreter of an
    axis-range value (the ``axis_range`` ParamDecl contract owns its coercion; no
    free-text eval)."""

    try:
        lo, hi, points = value
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an (min, max, points) axis range, got {value!r}.")
    lo, hi, points = float(lo), float(hi), int(points)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name} range bounds must be finite.")
    if points < 2:
        raise ValueError(f"{name} needs at least 2 points.")
    if hi <= lo:
        raise ValueError(f"{name} max ({hi}) must exceed min ({lo}).")
    return lo, hi, points


@dataclass(frozen=True)
class MeasurementSpec(CatalogSpec):
    """A named measurement + its declared parameters + a build closure.

    A spec is the dependency-free contract between a measurement and a GUI/API:
    it lists the parameters ONCE (``params``, from :class:`CatalogSpec`), names the
    live curve's axes (``result_labels`` + ``x_key``/``y_key`` published-signal keys),
    and exposes ``build(**param_values) -> ScannedMeasurement`` returning an UNRUN
    measurement a logic node can drive point-by-point.  ``metadata`` carries
    spec-specific extras (e.g. the capture radius the temperature fit needs) without
    widening the call signature.  (A per-site result lives in the raw ``<y_key>`` block's
    ``data_shape=(n_sites,)`` -- the node publishes ONE canonical ``(R,P,*data_shape)`` signal; a per-site
    grid view is a PLOT-side reduce + reshape on it, using the trap array's grid shape, not a
    separate stored field.)
    """

    result_labels: tuple[str, str] = REQUIRED
    x_key: str = REQUIRED
    y_key: str = REQUIRED
    build: Callable[..., "ScannedMeasurement"] = REQUIRED
    # Canonical machine slug (the node prefix + signal names derive from it).  Defaults
    # to ``measurement_slug(name)`` so a measurement is named ONCE; ``x_key``/``y_key``
    # are the BARE quantity tokens (``t_off``/``survival``) -- the full hub signal is
    # ``f"{key}_{x_key}"`` (e.g. ``temperature_t_off``), so the signal names match the
    # display name automatically instead of a hand-typed abbreviation.
    key: str = ""

    collision_advice: ClassVar[str] = (
        "give each measurement a unique x_key/y_key (e.g. a per-measurement prefix).")

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.key:
            object.__setattr__(self, "key", measurement_slug(self.name))

    def collision_key(self) -> tuple:
        # Two measurements publishing the same bare x_key/y_key would clobber on the hub.
        return (self.x_key, self.y_key)

    def make_node(self, hub, *, prefix: str = "", repeat: int = 1, **values):
        """Build the LIVE logic node this measurement drives -- the ``ProcessorSpec.make_node``
        counterpart, so a decoupled console (or the notebook) asks the SPEC for its node instead of
        knowing which concrete node class a measurement uses.

        ``build(**values)`` returns the per-point ENGINE (a :class:`ScannedMeasurement`, or -- for the
        generic pulse scan -- a :class:`PulseScanPlan` carrier); this wraps it in the matching swept
        node behind ``prefix`` with the acquisition ``repeat`` knob (0 = ∞).  WHICH node is the scan
        TIER, declared ONCE on the spec (``metadata['node']``): ``"pulse_scan"`` is the DECOUPLED tier
        (a device driver whose y is a ``signal_expr`` off another node -> :class:`PulseScanNode`);
        anything else (incl. the COUPLED temperature/fidelity tier, whose y is reduced inline over a
        loading's frames) is a frame-reducing :class:`ScannedMeasurementNode`.  The tier's physics
        (acquire(1) vs the 2-trigger single loading) is pinned by ``tests/test_scan_tier_boundary.py``;
        this method only routes to the node that tier implies, it does NOT change what is fired/read.

        The concrete node classes are imported HERE (lazily) so the GUI never imports
        ``operations.logic`` to pick a node by a metadata string -- the spec owns the assembly."""
        from .logic import PulseScanNode, ScannedMeasurementNode

        built = self.build(**values)
        prefix = str(prefix)
        repeat = max(0, int(repeat))
        if self.metadata.get(NODE_META_KEY) == PULSE_SCAN_NODE:
            return PulseScanNode(hub, built, x_key=self.x_key, y_key=self.y_key,
                                 prefix=prefix, repeat=repeat)
        return ScannedMeasurementNode(hub, built, x_key=self.x_key, y_key=self.y_key,
                                      prefix=prefix, repeat=repeat)


@dataclass(frozen=True)
class ScanAxis:
    """Declares the swept parameter: which bound slot, which values, units.

    ``slot`` is the semantic :class:`ScanSlot.name`; positional ``sN`` tokens
    never cross this boundary.  ``values`` are the scan points in the axis's own unit: a time
    (``duration``) slot is set with ``pulse.set_slot`` and takes NANOSECONDS at
    the wire, but ``values``/``unit`` are kept in the user-facing unit (seconds by
    default, like ``detection_time``) and converted by ``scale_to_ns``.  A
    ``dac`` slot is set with ``pulse.set_slot`` and takes the SIGNED user value
    untouched.  ``label``/``unit`` only annotate the plot.
    """

    slot: str
    values: np.ndarray
    label: str = "x"
    unit: str = "s"
    kind: str = "duration"
    scale_to_ns: float = 1e9

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("ScanAxis.values must be a non-empty finite 1-D array.")
        object.__setattr__(self, "values", values)
        kind = str(self.kind).lower()
        if kind not in ("duration", "dac"):
            raise ValueError("ScanAxis.kind must be 'duration' or 'dac'.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "slot", str(self.slot))

    @property
    def is_time(self) -> bool:
        return self.kind == "duration"

    def apply(self, pulse, value: float) -> None:
        """Push one scan value into the bound pulse (contract-only)."""

        if self.is_time:
            pulse.set_slot(self.slot, float(value) * float(self.scale_to_ns))
        else:
            pulse.set_slot(self.slot, float(value))


@dataclass(frozen=True)
class DeviceControlAxis:
    """Declares a swept DEVICE runtime control (not a pulse slot).

    ``ScanAxis`` writes the bound PULSE each point; a great many experiments instead sweep a device
    SET-POINT -- an RF source's two-photon detuning for a grey-molasses scan, a laser's saturation, a
    field.  This axis carries a ``write(value)`` callable (a device control's setter, bound to the
    device) and its ``apply`` writes THAT, leaving the pulse fixed; the ``ShotPlan`` then fires a
    constant sequence and the reducer reads the frames.  It DUCK-TYPES the surface the scan engine uses
    (``values`` / ``label`` / ``unit`` / ``apply``), so :class:`ScannedMeasurement` sweeps a device knob
    with no change; ``kind='device'`` and ``is_time=False`` make the engine skip the pulse-slot
    validation and the clock-grid snap that only pulse axes need.  Pairs with a fixed-sequence plan
    (e.g. :class:`~..operations.temperature.FixedReleaseRecapturePlan`)."""

    values: np.ndarray
    write: Callable[[float], None]
    label: str = "x"
    unit: str = ""
    kind: str = "device"
    scale_to_ns: float = 1.0        # unused (no pulse write); present so the ScanAxis surface matches

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("DeviceControlAxis.values must be a non-empty finite 1-D array.")
        if not callable(self.write):
            raise TypeError("DeviceControlAxis.write must be a callable (value) -> None.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "kind", "device")

    @property
    def is_time(self) -> bool:
        return False

    def apply(self, pulse, value: float) -> None:
        """Write one scan value to the DEVICE control; the bound pulse is untouched."""

        del pulse
        self.write(float(value))


@runtime_checkable
class ShotPlan(Protocol):
    """How many frames a point acquires and how its sequence is built.

    ``sequence_for`` returns the ``PulseSequence`` for one scan value using ONLY
    the bound ``PulseController`` contract (``frame_sequence`` etc.); it never
    touches a concrete backend.
    """

    n_frames: int

    def sequence_for(self, pulse, axis: "ScanAxis", value: float):
        ...


@dataclass(frozen=True)
class NFramePlan:
    """Acquire ``n_frames`` independent frames; one trigger per frame.

    The scanned value is written into the pulse's time slot and a finite
    ``n_frames``-trigger sequence is generated -- the detection-time scan's plan.
    """

    n_frames: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_frames", positive_int(self.n_frames, "n_frames"))

    def sequence_for(self, pulse, axis: "ScanAxis", value: float):
        if axis.is_time:
            resolved = float(value) * float(axis.scale_to_ns)
        else:
            resolved = float(value)
        return pulse.frame_sequence(self.n_frames, slots={axis.slot: resolved})


@runtime_checkable
class PointReducer(Protocol):
    """Reduce one point's frames to exactly one declared data tensor.

    ``data_shape`` is the complete, non-empty shape returned by :meth:`reduce`.
    A scalar quantity therefore declares and returns ``(1,)``; a per-site
    quantity declares ``(n_sites,)``; higher-rank data retains every axis.
    """

    labels: tuple[str, str, str]
    data_shape: tuple[int, ...]

    def reduce(self, frames, calibration: TrapCalibration) -> np.ndarray:
        ...


def reducer_data_shape(reducer: PointReducer) -> tuple[int, ...]:
    """Return the reducer's complete positive ``data_shape`` or fail loudly.

    This is the single validation seam shared by the acquisition engine and
    its hub adapter.  There is deliberately no scalar-width fallback: every
    reducer must declare the actual tensor shape it produces.
    """

    try:
        raw_shape = reducer.data_shape
    except AttributeError as exc:
        raise TypeError(
            f"{type(reducer).__name__} must declare a non-empty data_shape tuple.") from exc
    try:
        shape = tuple(
            positive_int(size, f"{type(reducer).__name__}.data_shape[{axis}]")
            for axis, size in enumerate(raw_shape)
        )
    except TypeError as exc:
        raise TypeError(
            f"{type(reducer).__name__}.data_shape must be a non-empty iterable of "
            f"positive integers, got {raw_shape!r}.") from exc
    if not shape:
        raise ValueError(
            f"{type(reducer).__name__}.data_shape must be non-empty; "
            "a scalar reducer declares (1,).")
    return shape


def otsu_fidelity_from_frames(frames, calibration: TrapCalibration, site: int | None):
    """Stack per-frame site signals, select ``site`` (or pool ALL when ``None``), Otsu-split the
    distribution and score the two-Gaussian threshold fidelity -- the ONE
    ``signals -> (counts, threshold, FidelityEstimate)`` pipeline, shared by the live
    :class:`OtsuFidelityReducer` and the held-out reference path in the readout subsystem so the
    detection-time scan's y and its reference fidelity are computed identically (#C3).  Returns the
    stacked per-site ``counts`` too (the reference path records them).

    A point that produced NO frames -- a pulse that never triggered the camera, a cancelled / short
    read, or a self-triggering sensor that yielded nothing (``triggered_frames`` is documented to
    return whatever arrived) -- has no readout to threshold: it returns empty counts + a NaN-fidelity
    estimate.  The single-shot reducer's existing 'non-finite fidelity -> 0.5 (no discrimination)'
    fallback then fills that scan point, so a frameless point can never ``np.vstack`` an empty list and
    crash the whole scan / reference characterization (both funnel through here)."""
    frames = list(frames)
    if not frames:
        return np.empty((0, 0), dtype=float), float("nan"), estimate_threshold_fidelity(np.empty(0), 0.0)
    counts = np.vstack([calibration.signals(image) for image in frames])
    values = counts.reshape(-1) if site is None else counts[:, site_index(site, counts.shape[1])]
    threshold = otsu_threshold(values)
    return counts, threshold, estimate_threshold_fidelity(values, threshold)


@dataclass
class OtsuFidelityReducer:
    """Per-point single-shot readout fidelity (the detection-time scan's y).

    Stacks ``calibration.signals`` over the point's frames (so a PSF calibration
    uses PSF signals -- the same quantity ``detect`` compares), otsu-splits the
    pooled distribution, and returns the gaussian-split fidelity.  ``site=None``
    pools all sites; an int restricts to one site.  Each reduce records its
    threshold/fidelity in ``thresholds``/``fidelities`` so the detection-time
    wrapper can surface them.

    Scope: this is a LIVE quick-look estimate -- a Gaussian two-population MODEL
    fidelity from the same shots it thresholds (no train/test split, no truth
    labels), which is the right cheap signal to optimize a detection time on a
    rolling curve.  For a rigorous HELD-OUT, per-site empirical fidelity (truth
    from reference shots, threshold trained on a split, fidelity scored on the
    held-out split), run ``exp.readout.characterize_from_dir(...)`` -- a separate
    characterization that acquires the reference frames the live scan does not.
    """

    site: int | None = None
    labels: tuple[str, str, str] = ("Detection time (s)", "Fidelity", "Fidelity")
    data_shape: ClassVar[tuple[int, ...]] = (1,)
    thresholds: list[float] = field(default_factory=list)
    fidelities: list[float] = field(default_factory=list)

    def reduce(self, frames, calibration: TrapCalibration) -> np.ndarray:
        _counts, threshold, model = otsu_fidelity_from_frames(frames, calibration, self.site)
        fidelity = float(model.fidelity)
        if not np.isfinite(fidelity):
            fidelity = 0.5
        self.thresholds.append(float(threshold))
        self.fidelities.append(fidelity)
        return np.asarray([fidelity], dtype=float)


@dataclass
class ScanResult(MeasurementTaskResult):
    """Live or completed generic scan: x axis + per-point y rows.

    Inherits ``stop()``/``points_done``/``running``/``measurement_done`` from
    :class:`MeasurementTaskResult` (so the GUI gets the same cooperative stop the
    detection-time scan has).  ``data_y`` is ``(P,*data_shape)`` and retains
    every axis declared by the reducer.
    """

    x: np.ndarray
    data_y: np.ndarray
    labels: tuple[str, str, str] = ("x", "y", "y")
    measurement: Any = None
    plot: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        """First value of a one-axis data shape (the scalar/site-0 curve).

        Higher-rank data has no unambiguous implicit selection; callers use
        :attr:`data_y` and explicitly select or reduce the desired axes.
        """

        if self.data_y.ndim != 2:
            raise ValueError(
                f"ScanResult.y is only defined for a one-axis data_shape; got "
                f"data_shape={self.data_y.shape[1:]}. Use data_y and explicitly "
                "select or reduce its data axes.")
        return self.data_y[:, 0]

    @property
    def points_done(self) -> int:
        """Count completed P rows without treating any data axis as points."""

        if self.measurement is not None and hasattr(self.measurement, "points_done"):
            return int(self.measurement.points_done)
        if not self.data_y.size:
            return 0
        finite = np.isfinite(self.data_y)
        if finite.ndim == 1:
            has_data = finite
        else:
            has_data = np.any(finite, axis=tuple(range(1, finite.ndim)))
        return int(np.count_nonzero(has_data))

    @property
    def finished(self) -> bool:
        return bool(len(self.x) and self.points_done == len(self.x))

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.labels[0],
            "points": int(len(self.x)),
            "points_done": self.points_done,
            "data_shape": tuple(int(n) for n in self.data_y.shape[1:]),
            "running": self.running,
            "finished": self.finished,
        }


def require_pulse_controller(pulse):
    """The ONE "is this a fireable pulse handle" gate: anything a scan is about to FIRE must
    be a ``PulseController`` (it exposes a callable ``frame_sequence``) -- the object
    ``exp.timing.bind_pulse(...)`` returns.  Every entry point that
    accepts an operator-supplied ``pulse`` (the scan builders in ``subsystems.readout`` and
    :class:`ScannedMeasurement` itself) calls THIS, so the predicate and its user guidance
    are typed exactly once and a raw sequence/path always fails with the same clear text.
    Returns ``pulse`` so call sites can gate-and-use in one expression."""
    if not callable(getattr(pulse, "frame_sequence", None)):
        raise TypeError(
            "pulse must be a PulseController returned by exp.timing.bind_pulse(...)."
        )
    return pulse


@dataclass
class ScannedMeasurement:
    """One swept measurement: ``axis`` x ``plan`` x ``reducer`` over an engine.

    ``run`` sweeps ``axis.values``; per point it sets the slot, acquires
    ``plan.n_frames`` frames (``shots_per_point`` times, averaged), and reduces
    them to the point's y.  With a viewer registered it drives the live curve
    through ``active_plotter().run`` (background worker, UI-thread refresh,
    cooperative ``stop()``); headless it runs the same callback synchronously and
    still returns a complete :class:`ScanResult`.
    """

    pulse: Any
    camera: Any
    sequencer: Any
    calibration: TrapCalibration
    axis: ScanAxis
    plan: ShotPlan
    reducer: PointReducer
    shots_per_point: int = 1
    run_hardware: Callable[..., object] | None = None
    data_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        # The fireable pulse handle is the outer orchestration boundary.  Reject it
        # before inspecting collaborators so every entry point reports the one shared
        # controller error instead of whichever invalid helper happens to be visited first.
        require_pulse_controller(self.pulse)
        self.shots_per_point = positive_int(self.shots_per_point, "shots_per_point")
        self.data_shape = reducer_data_shape(self.reducer)
        # Validate the exact semantic field for BOTH duration and DAC axes.  A
        # duration axis must not silently mutate the first duration slot when it
        # declared another one, and an sN token is never a public identity.
        # A DeviceControlAxis carries no pulse slot (kind='device', is_time=False -- see its
        # docstring); only a ScanAxis names a semantic ScanSlot to validate, so skip the pulse-slot
        # check when the axis has no `slot` rather than crashing on the missing attribute.
        slot = getattr(self.axis, "slot", None)
        underlying = getattr(self.pulse, "pulse", None)
        if slot is not None and underlying is not None and hasattr(underlying, "scan_names"):
            known = list(underlying.scan_names)
            if slot not in known:
                raise ValueError(
                    f"ScanAxis.slot {slot!r} is not a semantic scan slot of the "
                    f"bound pulse (known slots: {known}).")
            actual_kind = underlying.scan_slots[known.index(slot)].kind
            if actual_kind != self.axis.kind:
                raise ValueError(
                    f"ScanAxis {slot!r} declares kind={self.axis.kind!r}, but the "
                    f"bound ScanSlot kind is {actual_kind!r}.")
        # Snap a DURATION axis to the sequencer clock grid: the hardware only lands on
        # whole ticks, so a continuous / linspace sweep (readout-duration -> fidelity,
        # release-recapture hold, ...) is quantized HERE, at the one place every duration
        # scan builds its per-point sequence.  The plotted x then equals the exposures
        # actually run and no downstream sequence trips the clock-grid validator -- the
        # "repeat period not on the clock grid" failure.  (DAC axes are integer codes.)
        if self.axis.is_time:
            clock = float(getattr(self._sequencer(), "clock_hz", 0.0) or 0.0)
            if clock > 0.0:
                scale = float(self.axis.scale_to_ns) / 1e9      # axis unit -> seconds
                snapped = np.array(
                    [snap_seconds_to_clock(float(v) * scale, clock) / scale for v in self.axis.values],
                    dtype=float)
                self.axis = replace(self.axis, values=snapped)
        # A driver (e.g. ScannedMeasurementNode) sets this to its stop Event so a
        # Stop interrupts a wedged trigger DURING a scan point, not just between
        # points; None = the synchronous notebook path (no mid-point cancel).
        self.stop_event = None

    def _sequencer(self):
        # Prefer the sequencer the pulse is bound to (single source of truth);
        # fall back to the one passed in (e.g. the session's default device).
        return getattr(self.pulse, "_sequencer", self.sequencer) or self.sequencer

    def measure(self, value: float, index: int | None = None) -> np.ndarray:
        """Acquire + reduce one scan point.  The live engine calls this per point."""

        if self.run_hardware is not None:
            sequencer = self._sequencer()
            authority_sequencer = getattr(sequencer, "_authority_device", sequencer)
            return np.asarray(
                self.run_hardware(
                    (self.camera, authority_sequencer),
                    lambda: self._measure_owned(value, index),
                    name="scanned-measurement-point",
                ),
                dtype=float,
            )
        return self._measure_owned(value, index)

    def _measure_owned(self, value: float, index: int | None = None) -> np.ndarray:
        """Measure one point after the caller has established device ownership."""

        self.axis.apply(self.pulse, float(value))
        sequence = self.plan.sequence_for(self.pulse, self.axis, float(value))
        sequencer = self._sequencer()
        rows: list[np.ndarray] = []
        for shot in range(self.shots_per_point):
            frames = triggered_frames(
                self.camera, sequencer, sequence, self.plan.n_frames, stop=self.stop_event)
            row = np.asarray(self.reducer.reduce(frames, self.calibration), dtype=float)
            if row.shape != self.data_shape:
                raise ValueError(
                    f"{type(self.reducer).__name__}.reduce returned shape {row.shape} "
                    f"for shot {shot}; declared data_shape is {self.data_shape}.")
            rows.append(row)
        # Average over the shots where each site HAD an atom: a per-site reducer returns NaN for a
        # site that was empty on a given shot (no atom -> survival undefined), and a plain mean would
        # let one NaN shot poison that site's whole scan point.  ``finite_mean`` (the ONE gap-safe
        # average, shared with the live plots) averages the occupied shots; a site empty across ALL
        # shots stays NaN (correctly: no data) -- computed directly as sum/count, so it neither warns
        # nor needs a catch_warnings that would blanket-ignore every other RuntimeWarning here.
        return finite_mean(np.stack(rows, axis=0), axis=0)

    def run(self, *, live: bool = True, update_time: float = 0.05, display: bool = True,
            stop_hint: str | bool = True) -> ScanResult:
        values = self.axis.values
        data_y = np.full((len(values), *self.data_shape), np.nan, dtype=float)
        result = ScanResult(x=values, data_y=data_y, labels=tuple(self.reducer.labels))

        plotter = active_plotter()
        if live and plotter is not None and len(self.data_shape) > 1:
            raise ValueError(
                f"live curve rendering needs an explicit selection or reduction for "
                f"data_shape={self.data_shape}; use ScannedMeasurementNode with a facet, "
                "or run with live=False and inspect ScanResult.data_y.")
        if live and plotter is not None:
            result.measurement = plotter.run(
                values.reshape(-1, 1),
                self.measure,
                data_y=data_y,
                labels=tuple(self.reducer.labels),
                update_time=update_time,
                display=display,
                stop_hint=stop_hint,
            )
            result.plot = result.measurement.plot
        else:
            # No viewer registered (headless / frontend not imported): run the scan
            # synchronously and still return a complete result.
            for index, value in enumerate(values):
                data_y[index, ...] = self.measure(float(value), index)
            if len(self.data_shape) == 1:
                result.plot = plot_detection_scan(
                    values, data_y[:, 0], labels=tuple(self.reducer.labels), display=display)
        return result


__all__ = [
    "DeviceControlAxis",
    "MeasurementSpec",
    "NFramePlan",
    "OtsuFidelityReducer",
    "ParamDecl",
    "PointReducer",
    "program_completion_timeout",
    "reducer_data_shape",
    "ScanAxis",
    "ScanResult",
    "ScannedMeasurement",
    "ShotPlan",
    "prepare_hardware_scan",
    "require_pulse_controller",
    "triggered_frames",
]
