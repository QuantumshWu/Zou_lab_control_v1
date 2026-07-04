"""Generic scanned-measurement abstraction (the engine behind every live scan).

A "scanned measurement" sweeps one bound pulse parameter, acquires a few camera
frames per point, and reduces those frames to a number (or one number per site)
that becomes the live curve's y value.  Detection-time/fidelity and
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

from Zou_lab_control._readout_math import finite_mean
from Zou_lab_control._viewer_registry import active_plotter

from ._spec import REQUIRED, CatalogSpec

from ..core.analysis import estimate_threshold_fidelity, otsu_threshold, positive_int
from ..core.calibration import TrapCalibration
from ..core.results import MeasurementTaskResult
from ..core.utils import site_index
from ..timing.sequence import snap_seconds_to_clock
from ..views.plots import plot_detection_scan


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
    camera.arm(frames)
    try:
        sequencer.prepare(program)
        sequencer.fire(program)
        return camera.read_frames(frames, stop=stop)
    except BaseException:
        # A prepare / fire / read fault (an RPyC EOF mid-fire, a STATUS_UNDERFLOW stall, a cancel) must
        # NOT leave the FPGA armed / loaded / firing -- drive the sequencer SAFE before re-raising, so a
        # failed shot can never strand outputs high.  (The camera is stood down by the finally below.)
        # On the SUCCESS path the sequencer is left as the caller set it -- a finite program has run to
        # completion -- so this touches only the error path.
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
    trigger line (via ``count_trigger_pulses`` and the camera's ``capture_trigger_channels``).
    ``repeats = ceil(frames / base_triggers)`` reaches ``frames`` edges: a single-trigger imaging
    pulse (base_triggers=1) becomes ``repeated(frames)`` (N independent shots); a bracket already
    carrying >= ``frames`` edges (base_triggers >= frames) fires once, untouched.  A sequence that
    carries NO camera trigger, or one that is already a repeat program / lacks ``repeated`` (a raw
    device payload), is fired as-is -- the edge-faithful count then flows from what actually fires."""
    from ..devices.camera_trigger import count_trigger_pulses

    if frames <= 1 or not hasattr(sequence, "repeated"):
        return sequence
    if getattr(sequence, "repeat_forever", False) or int(getattr(sequence, "repeat_count", 1)) > 1:
        return sequence                      # already a multi-cycle / continuous program
    trig = getattr(camera, "capture_trigger_channels", None)
    base_triggers = count_trigger_pulses(sequence, **({"trigger_channels": trig} if trig else {}))
    if base_triggers <= 0 or base_triggers >= frames:
        return sequence                      # no camera trigger, or the base already carries enough edges
    repeats = -(-frames // base_triggers)    # ceil: reach at least `frames` edges
    return sequence.repeated(repeats) if repeats > 1 else sequence


# --------------------------------------------------------------- declarative spec


@dataclass(frozen=True)
class ParamDecl:
    """Declarative description of ONE tunable/scannable measurement parameter.

    Dependency-free: a measurement declares each of its parameters once
    (``key``/``label``/``kind``/``default``/bounds/...) and BOTH the Python API
    default AND the GUI control derive from this single declaration (the single
    source of truth -- explicit declaration, not signature reflection or AST
    inspection).  ``kind`` selects how a caller/GUI interprets the value:

    ``"float"``       a scalar in ``[lo, hi]`` (``unit`` annotates it)
    ``"int"``         an integer in ``[lo, hi]``
    ``"axis_range"``  a swept range; the value is ``(min, max, points)`` (the GUI
                      renders three boxes).  ``default`` is that 3-tuple, in
                      ``unit``.
    ``"bool"``        a flag (checkbox)
    ``"choice"``      one of ``choices`` (a combo box)
    ``"text"``        a free string (a line edit) -- e.g. a label; taken verbatim,
                      NEVER ``eval``'d (lo/hi/choices ignored)
    ``"path"``        a filesystem path: a line edit + a Browse button (native dialog).
                      ``path_mode='file'`` picks a file (filtered by ``file_filter``),
                      ``path_mode='dir'`` picks a folder.  Taken verbatim, never eval'd.
    ``"signal"``      the NAME of a hub signal to consume (a processor's input): a combo
                      box of the live hub signals, like a plot's input picker.
    ``"signal_expr"`` a MULTI-slot signal picker + a ``value = ...`` expression (the same
                      one a plot panel's source uses): pick one or more hub signals (read as
                      ``signal`` / ``signal[i]``) and combine them.  The value is a
                      ``{"inputs": [name, ...], "source": "value = ..."}`` dict -- so a
                      processor/measurement "source" can subscribe to several running nodes'
                      signals and combine them, never just one bare name.
    ``"pulse_param"`` a parameter of a pulse template to sweep: a combo box whose choices are
                      introspected from the template FILE named in the ``depends_on`` field
                      (its periods / channels / DAC buses), so picking the template repopulates
                      it.  The value is a ``"kind:target"`` token (e.g. ``"duration:2"``).

    ``required`` marks a parameter a GUI must highlight when missing.  ``depends_on`` (kind
    ``pulse_param``) names the sibling ``path`` field whose pulse template is introspected to
    populate this control.  ``display`` is a pure DATA placement flag (not an art/geometry knob)
    that splits a plot panel's params between its two surfaces:
      * ``display=True``  -- a pure DISPLAY knob (how the SAME data is drawn): bins, history length,
                             fit chooser, log axis, colormap, relim.  Rendered in the lightweight
                             Setting popup, alongside size / relim, where an operator reaches for them.
      * ``display=False`` -- an ACQUISITION / measurement-API param (what data is taken).  Rendered in
                             the panel's Edit tab instead, so Setting and Edit never duplicate.
    A measurement param ignores it (defaults True).  No value here is ever ``eval``'d -- the spec
    consumer validates / coerces by ``kind``.
    """

    key: str
    label: str
    kind: str
    default: Any = None
    unit: str = ""
    lo: float = 0.0
    hi: float = 1e12
    required: bool = False
    choices: tuple = ()
    tooltip: str = ""
    path_mode: str = "file"          # kind="path": "file" (open-file dialog) | "dir" (folder dialog)
    file_filter: str = "All files (*)"  # kind="path", path_mode="file": the open-file filter
    base_dir: str = ""               # kind="path": the folder a Browse dialog opens in when the
                                     # field doesn't resolve to an existing path (e.g. "pulses"
                                     # for a pulse template, "calibrations" for a data folder)
    depends_on: str = ""             # kind="pulse_param": the sibling kind="path" field whose
                                     # pulse template is introspected to populate this combo
    display: bool = True             # plot-panel placement flag (DATA, not art): True = a pure DISPLAY
                                     # knob (bins / history / fit / colormap …) in the Setting popup;
                                     # False = an acquisition / measurement-API param in the Edit tab.
                                     # Ignored by measurements.
    segmented: bool = False          # kind="choice" RENDER hint (DATA, not art): True = a capsule
                                     # tri/multi-state toggle (confocal TriStateToggleSwitch) instead of
                                     # a combo box; same value semantics (one of ``choices``).

    def __post_init__(self) -> None:
        kind = str(self.kind).lower()
        if kind not in ("float", "int", "axis_range", "bool", "choice", "text", "path",
                        "signal", "signal_expr", "pulse_param", "pulse_slots"):
            raise ValueError(
                "ParamDecl.kind must be one of "
                "float/int/axis_range/bool/choice/text/path/signal/signal_expr/pulse_param/pulse_slots, "
                f"got {self.kind!r}."
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "choices", tuple(self.choices))


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


def measurement_slug(name: str) -> str:
    """Canonical machine token for a measurement, derived from its display ``name``
    (lower-case, non-alphanumeric runs -> single ``_``, trimmed).  ONE source: the
    node prefix + every published signal name derive from this, so the measurement is
    called the same thing in the Add-Panel list, the signal-flow legend, and the hub
    signal names -- never a separately hand-typed abbreviation that drifts."""
    import re
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(name).lower())).strip("_")


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
    dimension axis -- the node publishes ONE raw ``(repeat, points, dim)`` signal; a per-site
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
        if self.metadata.get("node") == "pulse_scan":
            return PulseScanNode(hub, built, x_key=self.x_key, y_key=self.y_key,
                                 prefix=prefix, repeat=repeat)
        return ScannedMeasurementNode(hub, built, x_key=self.x_key, y_key=self.y_key,
                                      prefix=prefix, repeat=repeat)


@dataclass(frozen=True)
class ScanAxis:
    """Declares the swept parameter: which bound slot, which values, units.

    ``slot`` is the ``PulseController`` slot key (``"s0"``..., or an int slot
    index).  ``values`` are the scan points in the axis's own unit: a time
    (``duration``) slot is set with ``pulse.set_time`` and takes NANOSECONDS at
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
            pulse.set_time(float(value) * float(self.scale_to_ns))
        else:
            pulse.set_slot(self.slot, float(value))


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
            return pulse.frame_sequence(self.n_frames, time_ns=float(value) * float(axis.scale_to_ns))
        # A DAC-axis scan still needs N triggers; the slot was already set via axis.apply.
        return pulse.frame_sequence(self.n_frames, slots={axis.slot: float(value)})


@runtime_checkable
class PointReducer(Protocol):
    """Reduce one point's frames (+ calibration) to its y value(s)."""

    labels: tuple[str, str, str]
    n_series: int

    def reduce(self, frames, calibration: TrapCalibration):
        ...


def otsu_fidelity_from_frames(frames, calibration: TrapCalibration, site: int | None):
    """Stack per-frame site signals, select ``site`` (or pool ALL when ``None``), Otsu-split the
    distribution and score the two-Gaussian threshold fidelity -- the ONE
    ``signals -> (counts, threshold, FidelityEstimate)`` pipeline, shared by the live
    :class:`OtsuFidelityReducer` and the held-out reference path in the readout subsystem so the
    detection-time scan's y and its reference fidelity are computed identically (#C3).  Returns the
    stacked per-site ``counts`` too (the reference path records them)."""
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
    n_series: int = 1
    thresholds: list[float] = field(default_factory=list)
    fidelities: list[float] = field(default_factory=list)

    def reduce(self, frames, calibration: TrapCalibration) -> float:
        _counts, threshold, model = otsu_fidelity_from_frames(frames, calibration, self.site)
        fidelity = float(model.fidelity)
        if not np.isfinite(fidelity):
            fidelity = 0.5
        self.thresholds.append(float(threshold))
        self.fidelities.append(fidelity)
        return fidelity


@dataclass
class ScanResult(MeasurementTaskResult):
    """Live or completed generic scan: x axis + per-point y rows.

    Inherits ``stop()``/``points_done``/``running``/``measurement_done`` from
    :class:`MeasurementTaskResult` (so the GUI gets the same cooperative stop the
    detection-time scan has).  ``data_y`` is ``(n_points, n_series)``; a scalar
    reducer fills column 0, a per-site reducer fills one column per site.
    """

    x: np.ndarray
    data_y: np.ndarray
    labels: tuple[str, str, str] = ("x", "y", "y")
    measurement: Any = None
    plot: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        """First series (the scalar curve, or site 0 of a per-site scan)."""

        return self.data_y[:, 0]

    @property
    def finished(self) -> bool:
        return bool(self.data_y.size and np.all(np.isfinite(self.data_y[:, 0])))

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.labels[0],
            "points": int(len(self.x)),
            "points_done": self.points_done,
            "series": int(self.data_y.shape[1]),
            "running": self.running,
            "finished": self.finished,
        }


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

    def __post_init__(self) -> None:
        self.shots_per_point = positive_int(self.shots_per_point, "shots_per_point")
        if not callable(getattr(self.pulse, "frame_sequence", None)):
            raise TypeError(
                "pulse must be a PulseController returned by exp.timing.bind_pulse(...) "
                "or na.bind_pulse(...)."
            )
        # Fail early on a DAC axis whose slot the bound pulse doesn't have: a bad
        # slot key is otherwise silently stored by set_slot and never applied, so
        # the scan runs but moves nothing -- a wrong experiment with no error.
        # (e.g. slot=0 stringifies to "0", not "s0"; a typo "s5" on a 3-slot pulse.)
        if self.axis.kind == "dac":
            underlying = getattr(self.pulse, "pulse", None)
            # PulseTableState exposes its scan-slot keys as the `scan_var_names`
            # property ("s0".."sN"); validate against it when present.
            known = getattr(underlying, "scan_var_names", None)
            if known and self.axis.slot not in known:
                raise ValueError(
                    f"ScanAxis.slot {self.axis.slot!r} is not a scan slot of the bound pulse "
                    f"(known slots: {list(known)}); a DAC scan must target one of them.")
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
        return getattr(self.pulse, "sequencer", self.sequencer) or self.sequencer

    def measure(self, value: float, index: int | None = None) -> np.ndarray:
        """Acquire + reduce one scan point.  The live engine calls this per point."""

        self.axis.apply(self.pulse, float(value))
        sequence = self.plan.sequence_for(self.pulse, self.axis, float(value))
        sequencer = self._sequencer()
        rows = []
        for _ in range(self.shots_per_point):
            frames = triggered_frames(
                self.camera, sequencer, sequence, self.plan.n_frames, stop=self.stop_event)
            rows.append(np.atleast_1d(np.asarray(self.reducer.reduce(frames, self.calibration), dtype=float)))
        # Average over the shots where each site HAD an atom: a per-site reducer returns NaN for a
        # site that was empty on a given shot (no atom -> survival undefined), and a plain mean would
        # let one NaN shot poison that site's whole scan point.  ``finite_mean`` (the ONE gap-safe
        # average, shared with the live plots) averages the occupied shots; a site empty across ALL
        # shots stays NaN (correctly: no data) -- computed directly as sum/count, so it neither warns
        # nor needs a catch_warnings that would blanket-ignore every other RuntimeWarning here.
        return finite_mean(np.vstack(rows), axis=0)

    def run(self, *, live: bool = True, update_time: float = 0.05, display: bool = True,
            stop_hint: str | bool = True) -> ScanResult:
        values = self.axis.values
        n_series = int(self.reducer.n_series)
        data_y = np.full((len(values), n_series), np.nan, dtype=float)
        result = ScanResult(x=values, data_y=data_y, labels=tuple(self.reducer.labels))

        plotter = active_plotter()
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
                data_y[index, :] = self.measure(float(value), index)
            result.plot = plot_detection_scan(values, data_y[:, 0], labels=tuple(self.reducer.labels), display=display)
        return result


__all__ = [
    "MeasurementSpec",
    "NFramePlan",
    "OtsuFidelityReducer",
    "ParamDecl",
    "PointReducer",
    "ScanAxis",
    "ScanResult",
    "ScannedMeasurement",
    "ShotPlan",
    "triggered_frames",
]
