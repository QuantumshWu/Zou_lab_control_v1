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

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable

import numpy as np

from Zou_lab_control._viewer_registry import active_plotter

from ._spec import REQUIRED, CatalogSpec

from ..core.analysis import estimate_threshold_fidelity, otsu_threshold, positive_int
from ..devices.base import arm_then_fire
from ..core.calibration import TrapCalibration
from ..core.results import MeasurementTaskResult
from ..core.utils import site_index
from ..timing.sequence import snap_seconds_to_clock
from ..views.plots import plot_detection_scan


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
    populate this control.  ``display`` is a pure DATA placement flag (not an art/geometry knob):
    a plot-panel param with ``display=True`` is a BASIC display knob rendered in the lightweight
    Setting popup (e.g. the colormap chooser); ``display=False`` is a FUNCTIONAL plot-API param
    rendered in the panel's Edit tab instead, so Setting and Edit never duplicate.  A measurement
    param ignores it (defaults True).  No value here is ever ``eval``'d -- the spec consumer
    validates / coerces by ``kind``.
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
    display: bool = True             # plot-panel placement flag (DATA, not art): True = a basic
                                     # display knob in the Setting popup; False = a functional
                                     # plot-API param in the Edit tab.  Ignored by measurements.
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
    held-out split), run ``exp.readout.characterize(...)`` -- a separate
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
            frames = self.camera.acquire(
                self.plan.n_frames, sequence=sequence,
                on_armed=arm_then_fire(sequencer, sequence),
                stop=self.stop_event)
            rows.append(np.atleast_1d(np.asarray(self.reducer.reduce(frames, self.calibration), dtype=float)))
        # nanmean, not mean: a per-site reducer returns NaN for a site that was
        # empty on a given shot (no atom -> survival undefined).  Plain mean would
        # let one NaN shot poison that site's entire scan point to NaN.  nanmean
        # averages the shots where the site was occupied; a site empty across ALL
        # shots stays NaN (correctly: no data), with the all-NaN warning silenced.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(np.vstack(rows), axis=0)

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
]
