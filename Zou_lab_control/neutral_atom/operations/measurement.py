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

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from Zou_lab_control._viewer_registry import active_plotter

from ..core.analysis import estimate_threshold_fidelity, otsu_threshold, positive_int
from ..core.calibration import TrapCalibration
from ..core.results import MeasurementTaskResult
from ..core.utils import site_index
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

    ``required`` marks a parameter a GUI must highlight when missing.  No value
    here is ever ``eval``'d -- the spec consumer validates/coerces by ``kind``.
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

    def __post_init__(self) -> None:
        kind = str(self.kind).lower()
        if kind not in ("float", "int", "axis_range", "bool", "choice"):
            raise ValueError(
                f"ParamDecl.kind must be one of float/int/axis_range/bool/choice, got {self.kind!r}."
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "choices", tuple(self.choices))


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
class MeasurementSpec:
    """A named measurement + its declared parameters + a build closure.

    A spec is the dependency-free contract between a measurement and a GUI/API:
    it lists the parameters ONCE (``params``), names the live curve's axes
    (``result_labels`` + ``x_key``/``y_key`` published-signal keys), and exposes
    ``build(**param_values) -> ScannedMeasurement`` returning an UNRUN measurement
    a feed can drive point-by-point.  ``grid_shape`` is set for per-site
    measurements so a consumer can reshape a per-site result vector into a 2-D map.
    ``metadata`` carries spec-specific extras (e.g. the capture radius the
    temperature fit needs) without widening the call signature.
    """

    name: str
    params: tuple[ParamDecl, ...]
    result_labels: tuple[str, str]
    x_key: str
    y_key: str
    build: Callable[..., "ScannedMeasurement"]
    grid_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def param(self, key: str) -> ParamDecl:
        """Return the declaration for ``key`` (raises ``KeyError`` if absent)."""

        for decl in self.params:
            if decl.key == key:
                return decl
        raise KeyError(key)

    def defaults(self) -> dict[str, Any]:
        """The declared default value for every parameter, keyed by ``key``."""

        return {decl.key: decl.default for decl in self.params}


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


@dataclass
class OtsuFidelityReducer:
    """Per-point single-shot readout fidelity (the detection-time scan's y).

    Stacks ``calibration.signals`` over the point's frames (so a PSF calibration
    uses PSF signals -- the same quantity ``detect`` compares), otsu-splits the
    pooled distribution, and returns the gaussian-split fidelity.  ``site=None``
    pools all sites; an int restricts to one site.  Each reduce records its
    threshold/fidelity in ``thresholds``/``fidelities`` so the detection-time
    wrapper can surface them.
    """

    site: int | None = None
    labels: tuple[str, str, str] = ("Detection time (s)", "Fidelity", "Fidelity")
    n_series: int = 1
    thresholds: list[float] = field(default_factory=list)
    fidelities: list[float] = field(default_factory=list)

    def reduce(self, frames, calibration: TrapCalibration) -> float:
        counts = np.vstack([calibration.signals(image) for image in frames])
        if self.site is None:
            values = counts.reshape(-1)
        else:
            values = counts[:, site_index(self.site, counts.shape[1])]
        threshold = otsu_threshold(values)
        model = estimate_threshold_fidelity(values, threshold)
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
            frames = self.camera.acquire(self.plan.n_frames, sequence=sequence, sequencer=sequencer)
            rows.append(np.atleast_1d(np.asarray(self.reducer.reduce(frames, self.calibration), dtype=float)))
        return np.mean(np.vstack(rows), axis=0)

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
