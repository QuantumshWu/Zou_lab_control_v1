"""Headless fitting shared by DataFigure and live FitProcessor.

This module owns model definitions, selection, seeds/bounds, deterministic
sampling, the solver, and fit quality.  Frontends only adapt plot data into
``SelectedData`` and draw ``FitResult``; processors only adapt signal tensors
and publish the same result.  There is no Qt/Matplotlib dependency here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np


#: Opt-in env var (or the conftest fixture) that arms :func:`_assert_off_gui_thread`.  Off by default so
#: a legitimate notebook main-thread solve (no console) stays legal; the per-kind fit-thread tests set it
#: to mechanically prove a console fit NEVER solves on the Qt event loop (#6).
_FIT_THREAD_GUARD_ENV = "ZLC_FIT_THREAD_GUARD"

#: The Qt-thread check, REGISTERED by the frontend at import (:func:`register_solve_thread_guard`).  The
#: headless core keeps ZERO Qt dependency (devices/base contract, sealed seam) -- the frontend, which
#: already owns Qt, supplies the "am I on the app thread?" test; a pure-headless/notebook session never
#: registers one, so the guard is inert there (a hub-less solve is legal).
_solve_thread_guard = None


def register_solve_thread_guard(guard) -> None:
    """Install (or clear with ``None``) the Qt-thread check the frontend supplies -- see
    :func:`_assert_off_gui_thread`.  Kept here so ``core.fitting`` never imports Qt itself."""
    global _solve_thread_guard
    _solve_thread_guard = guard


def _assert_off_gui_thread() -> None:
    """When armed (``ZLC_FIT_THREAD_GUARD``) AND a Qt-thread check is registered, raise if an unbounded
    curve-fit solve is about to run ON the Qt application thread -- the mechanical enforcement of 'no fit
    ever solves on the Qt event loop' (#6).  A worker-node solve, and a headless/notebook solve (no guard
    registered), both pass."""
    if not os.environ.get(_FIT_THREAD_GUARD_ENV):
        return
    guard = _solve_thread_guard
    if guard is not None:
        guard()

from zlc_data.readout_math import (
    bimodal_jacobian,
    bimodal_model,
    gaussian,
    gaussian2d_center,
    gaussian_jacobian,
)
from zlc_data.plot_region import SelectedData, Selection, select_rows
from zlc_data.raster import RegularRaster


ModelFunction = Callable[..., np.ndarray]
Initializer = Callable[[object, np.ndarray], tuple[list[list[float]], tuple[np.ndarray, np.ndarray]]]


@dataclass(frozen=True)
class FitRequest:
    model: str
    selection: Selection = Selection()
    initial: tuple[float, ...] | None = None
    fixed: Mapping[str, float] = field(default_factory=dict)
    sample_budget: int = 12_000
    max_evaluations: int = 4_000
    coordinate_frame: str = "data"
    solve: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", str(self.model).strip().lower())
        object.__setattr__(self, "fixed", {str(k): float(v) for k, v in dict(self.fixed).items()})
        if self.initial is not None:
            object.__setattr__(self, "initial", tuple(float(v) for v in self.initial))
        if int(self.sample_budget) < 8:
            raise ValueError("fit sample_budget must be at least 8")
        if int(self.max_evaluations) < 1:
            raise ValueError("fit max_evaluations must be positive")
        object.__setattr__(self, "sample_budget", int(self.sample_budget))
        object.__setattr__(self, "max_evaluations", int(self.max_evaluations))
        object.__setattr__(self, "coordinate_frame", str(self.coordinate_frame or "data"))
        object.__setattr__(self, "solve", bool(self.solve))

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "selection": self.selection.to_dict(),
            "initial": None if self.initial is None else list(self.initial),
            "fixed": dict(self.fixed),
            "sample_budget": self.sample_budget,
            "max_evaluations": self.max_evaluations,
            "coordinate_frame": self.coordinate_frame,
            "solve": self.solve,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FitRequest":
        data = dict(payload or {})
        return cls(
            model=str(data.get("model", "")),
            selection=Selection.from_dict(data.get("selection")),
            initial=(None if data.get("initial") is None
                     else tuple(float(value) for value in data["initial"])),
            fixed={str(key): float(value) for key, value in dict(data.get("fixed", {})).items()},
            sample_budget=int(data.get("sample_budget", 12_000)),
            max_evaluations=int(data.get("max_evaluations", 4_000)),
            coordinate_frame=str(data.get("coordinate_frame", "data")),
            solve=bool(data.get("solve", True)),
        )


@dataclass(frozen=True)
class FitResult:
    model: str
    names: tuple[str, ...]
    parameters: np.ndarray
    covariance: np.ndarray | None
    valid: bool
    status: str
    quality: Mapping[str, float]
    n_points: int
    coordinate_frame: str = "data"

    @classmethod
    def invalid(
        cls,
        model: "FitModel | str",
        status: str,
        *,
        n_points: int = 0,
        coordinate_frame: str = "data",
    ) -> "FitResult":
        """Create the one explicit invalid result used by every adapter.

        A failed selection/model/solver never degrades to ``None`` and never
        reuses a previous parameter vector.  The parameter slots remain present
        as NaN so a processor can publish a complete, non-stale result atomically.
        """

        spec = fit_model(model) if isinstance(model, str) else model
        return cls(
            spec.key,
            spec.names,
            np.full(len(spec.names), np.nan),
            None,
            False,
            str(status),
            {"rmse": np.nan, "r2": np.nan},
            int(n_points),
            str(coordinate_frame or "data"),
        )

    def __post_init__(self) -> None:
        params = np.asarray(self.parameters, dtype=float).reshape(-1)
        if params.size != len(self.names):
            raise ValueError(
                f"fit result has {params.size} values for {len(self.names)} parameter names")
        params.setflags(write=False)
        object.__setattr__(self, "parameters", params)
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float)
            covariance.setflags(write=False)
            object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "quality", {str(k): float(v) for k, v in self.quality.items()})

    @property
    def popt(self) -> np.ndarray | None:
        return self.parameters if self.valid else None

    @property
    def function(self) -> str:
        """Model key, matching the public figure-fit vocabulary."""
        return self.model

    @property
    def pcov(self) -> np.ndarray | None:
        return self.covariance

    def parameter_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, self.parameters)}


@dataclass(frozen=True)
class HistogramFitResult:
    """Typed result of the readout-distribution fit.

    A histogram fit is not a general :class:`FitRequest`: its independent
    values are bin centres, its dependent values are bin populations, and a
    two-component result additionally owns the readout threshold/separation
    decision.  It nevertheless belongs in this headless module because model
    seeds, bounds and solver policy must not live in a plotting class.
    """

    requested_mode: str
    component_count: int
    parameters: np.ndarray
    covariance: np.ndarray | None
    valid: bool
    separated: bool
    threshold: float | None
    status: str

    def __post_init__(self) -> None:
        mode = str(self.requested_mode).strip().lower()
        if mode not in {"none", "single", "double"}:
            raise ValueError(f"unknown histogram fit mode {self.requested_mode!r}")
        count = int(self.component_count)
        expected = {0: 0, 1: 3, 2: 6}
        if count not in expected:
            raise ValueError("histogram fit component_count must be 0, 1, or 2")
        parameters = np.asarray(self.parameters, dtype=float).reshape(-1)
        if parameters.size != expected[count]:
            raise ValueError(
                f"{count}-component histogram fit needs {expected[count]} parameters; "
                f"got {parameters.size}")
        parameters.setflags(write=False)
        object.__setattr__(self, "requested_mode", mode)
        object.__setattr__(self, "component_count", count)
        object.__setattr__(self, "parameters", parameters)
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float)
            covariance.setflags(write=False)
            object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "separated", bool(self.separated))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "status", str(self.status))

    @classmethod
    def invalid(cls, mode: str, status: str) -> "HistogramFitResult":
        return cls(str(mode), 0, np.empty(0), None, False, False, None, str(status))

    @property
    def single_parameters(self) -> tuple[float, float, float] | None:
        if not self.valid or self.component_count != 1:
            return None
        return tuple(float(value) for value in self.parameters)

    @property
    def bimodal_parameters(self) -> np.ndarray | None:
        return self.parameters if self.valid and self.component_count == 2 else None

    def curves(self, x) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        """Return left/right/total model curves at ``x``.

        The frontend chooses only the display sampling.  Model evaluation stays
        here with the fitted result, so no plotting surface reimplements model
        math or has to know the parameter layout.
        """

        if not self.valid:
            raise ValueError(f"cannot evaluate invalid histogram fit: {self.status}")
        x = np.asarray(x, dtype=float)
        if self.component_count == 1:
            total = gaussian(x, *self.parameters)
            return None, None, total
        left = gaussian(x, *self.parameters[:3])
        right = gaussian(x, *self.parameters[3:])
        return left, right, left + right

    def evaluate(self, x) -> np.ndarray:
        return self.curves(x)[2]


@dataclass(frozen=True)
class FitModel:
    key: str
    family: str
    names: tuple[str, ...]
    formula: str
    function: ModelFunction
    initialize: Initializer
    min_points: int


def _span(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(abs(np.nanmax(values) - np.nanmin(values)) or 1.0)


def _lorentzian(x, center, full_width, height, offset):
    half2 = (full_width / 2.0) ** 2
    return height * half2 / ((x - center) ** 2 + half2) + offset


def _gaussian(x, amplitude, offset, sigma, center):
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset


def _lorentzian_zeeman(x, center, full_width, height, offset, split):
    return (_lorentzian(x - split / 2.0, center, full_width, height, 0.0)
            + _lorentzian(x + split / 2.0, center, full_width, height, 0.0) + offset)


def _rabi(x, amplitude, offset, frequency, decay, phase):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) * np.exp(-x / decay) + offset


def _decay(x, amplitude, offset, decay):
    return amplitude * np.exp(-x / decay) + offset


def _init_lorentz(x, y):
    width = _span(x) / 4.0
    yrange = _span(y)
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    seeds = [
        [float(x[np.nanargmax(y)]), width, yrange, lo],
        [float(x[np.nanargmin(y)]), width, -yrange, hi],
    ]
    bounds = (
        np.array([np.nanmin(x), max(width / 10, 1e-15), -10 * yrange, lo - 10 * yrange]),
        np.array([np.nanmax(x), width * 10, 10 * yrange, hi + 10 * yrange]),
    )
    return seeds, bounds


def _init_gaussian(x, y):
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    amp, sigma = hi - lo, _span(x) / 6.0
    center = float(x[np.nanargmax(y)])
    seeds = [[amp, lo, sigma, center], [-amp, hi, sigma, center]]
    yrange = _span(y)
    bounds = (
        np.array([-10 * yrange, lo - 10 * yrange, max(sigma / 20, 1e-15), np.nanmin(x)]),
        np.array([10 * yrange, hi + 10 * yrange, sigma * 20, np.nanmax(x)]),
    )
    return seeds, bounds


def _init_zeeman(x, y):
    from scipy.signal import find_peaks

    yrange = _span(y)
    peaks, props = find_peaks(y, width=1, prominence=yrange / 8)
    step = float(abs(np.nanmedian(np.diff(x))) or 1.0) if len(x) > 1 else 1.0
    if len(peaks):
        order = peaks[np.argsort(y[peaks])[::-1]]
        first = int(order[0])
        widths = props.get("widths", np.ones(len(peaks)))
        lookup = {int(p): float(widths[i] * step) for i, p in enumerate(peaks)}
        width = max(lookup.get(first, step), step)
        seconds = order[: min(4, len(order))]
        seeds = []
        for second in seconds:
            center = float((x[first] + x[int(second)]) / 2)
            split = float(abs(x[int(second)] - x[first]))
            seeds.append([center, width, yrange, float(np.nanmin(y)), split])
    else:
        width = _span(x) / 8.0
        seeds = [[float(x[np.nanargmax(y)]), width, yrange, float(np.nanmin(y)), width * 2]]
    lo, hi, xrange = float(np.nanmin(y)), float(np.nanmax(y)), _span(x)
    bounds = (
        np.array([np.nanmin(x), max(width / 10, 1e-15), -10 * yrange, lo - 10 * yrange, 0]),
        np.array([np.nanmax(x), width * 10, 10 * yrange, hi + 10 * yrange, 2 * xrange]),
    )
    return seeds, bounds


def _init_rabi(x, y):
    amp = _span(y) / 2.0
    offset = float(np.nanmean(y))
    if len(x) > 1:
        dx = float(np.nanmedian(np.diff(x)))
        frequencies = np.fft.rfftfreq(len(y), d=abs(dx) or 1.0)
        spectrum = np.abs(np.fft.rfft(y - offset))
        frequency = float(frequencies[1 + np.argmax(spectrum[1:])]) if len(frequencies) > 1 else 1.0
    else:
        frequency = 1.0
    decay = _span(x)
    seeds = [[amp, offset, frequency, decay, np.pi / 2],
             [-amp, offset, frequency, decay, np.pi / 2]]
    yrange = _span(y)
    bounds = (
        np.array([-5 * abs(amp), offset - 2 * yrange, max(abs(frequency) / 10, 1e-15),
                  max(decay / 20, 1e-15), -np.pi]),
        np.array([5 * abs(amp), offset + 2 * yrange, max(abs(frequency) * 10, 1e-15),
                  decay * 20, 2 * np.pi]),
    )
    return seeds, bounds


def _init_decay(x, y):
    amp, offset, decay = _span(y), float(np.nanmean(y)), _span(x) / 2.0
    seeds = [[amp, offset, decay], [-amp, offset, decay]]
    yrange = _span(y)
    bounds = (
        np.array([-4 * yrange, offset - yrange, max(decay / 10, 1e-15)]),
        np.array([4 * yrange, offset + yrange, decay * 10]),
    )
    return seeds, bounds


def _init_center(coords, z):
    x, y = (np.asarray(v, dtype=float) for v in coords)
    offset = float(np.nanmedian(z))
    weights = np.clip(np.asarray(z, dtype=float) - offset, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("2D center fit needs non-flat finite data")
    x0 = float(np.sum(weights * x) / total)
    y0 = float(np.sum(weights * y) / total)
    radius = float(np.sqrt(np.sum(weights * ((x - x0) ** 2 + (y - y0) ** 2)) / total) or 1.0)
    amplitude = float(np.nanmax(z) - offset) or 1.0
    seeds = [[amplitude, offset, radius, x0, y0]]
    yrange = _span(z)
    bounds = (
        np.array([-5 * yrange, np.nanmin(z) - yrange, max(radius / 20, 1e-15),
                  np.nanmin(x), np.nanmin(y)]),
        np.array([5 * yrange, np.nanmax(z) + yrange, radius * 20,
                  np.nanmax(x), np.nanmax(y)]),
    )
    return seeds, bounds


_FIT_MODELS: dict[str, FitModel] = {
    model.key: model for model in (
        FitModel("lorent", "1d", ("x0", "fwhm", "amplitude", "offset"),
                 "Lorentzian", _lorentzian, _init_lorentz, 5),
        FitModel("gaussian", "1d", ("amplitude", "offset", "sigma", "x0"),
                 "Gaussian", _gaussian, _init_gaussian, 5),
        FitModel("lorent_zeeman", "1d", ("x0", "fwhm", "amplitude", "offset", "split"),
                 "Zeeman-split Lorentzian", _lorentzian_zeeman, _init_zeeman, 6),
        FitModel("rabi", "1d", ("amplitude", "offset", "frequency", "decay", "phase"),
                 "Damped Rabi", _rabi, _init_rabi, 7),
        FitModel("decay", "1d", ("amplitude", "offset", "decay"),
                 "Exponential decay", _decay, _init_decay, 4),
        FitModel("center", "2d", ("amplitude", "offset", "radius", "x0", "y0"),
                 "2D Gaussian center", gaussian2d_center, _init_center, 6),
    )
}


def fit_model(key: str) -> FitModel:
    try:
        return _FIT_MODELS[str(key).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown fit model {key!r}; choose from {tuple(_FIT_MODELS)}") from exc


def fit_models(*, family: str | None = None) -> tuple[FitModel, ...]:
    """Return the immutable ordered model catalogue, optionally by family."""

    models = tuple(_FIT_MODELS.values())
    if family is None:
        return models
    wanted = str(family).strip().lower()
    return tuple(model for model in models if model.family.lower() == wanted)


def _sample(selected: SelectedData, budget: int) -> SelectedData:
    """Deterministic even sampling; selection is applied before the budget."""

    if selected.count <= budget:
        return selected
    take = np.linspace(0, selected.count - 1, budget, dtype=np.int64)
    return SelectedData(
        {key: value[take] for key, value in selected.coordinates.items()},
        selected.values[take], selected.indices[take], selected.selection, selected.value_shape)


def _apply_request_overrides(model: FitModel, request: FitRequest, seeds, bounds):
    low, high = (np.array(v, dtype=float, copy=True) for v in bounds)
    if request.initial is not None:
        if len(request.initial) != len(model.names):
            raise ValueError(
                f"{model.key} initial has {len(request.initial)} values; expected {len(model.names)}")
        seeds = [np.asarray(request.initial, dtype=float)]
    else:
        seeds = [np.asarray(seed, dtype=float) for seed in seeds]
    for name, fixed in request.fixed.items():
        if name not in model.names:
            raise ValueError(f"{model.key} has no parameter {name!r}; choose from {model.names}")
        index = model.names.index(name)
        eps = max(abs(float(fixed)) * 1e-9, 1e-12)
        low[index], high[index] = float(fixed) - eps, float(fixed) + eps
        for seed in seeds:
            seed[index] = float(fixed)
    for seed in seeds:
        np.clip(seed, low + np.finfo(float).eps, high - np.finfo(float).eps, out=seed)
    return seeds, (low, high)


def _invalid(model: FitModel, request: FitRequest, status: str, n_points: int = 0) -> FitResult:
    return FitResult.invalid(
        model, status, n_points=n_points, coordinate_frame=request.coordinate_frame)


def _solve_candidates(
    function: ModelFunction,
    xdata,
    ydata,
    seeds,
    bounds,
    *,
    max_evaluations: int,
    jacobian=None,
):
    """One candidate-selection policy for every least-squares fit in this module."""

    from scipy.optimize import curve_fit

    ydata = np.asarray(ydata, dtype=float)
    best = None
    best_loss = np.inf
    failures: list[str] = []
    for seed in seeds:
        try:
            kwargs = {
                "p0": np.asarray(seed, dtype=float),
                "bounds": bounds,
                "maxfev": int(max_evaluations),
            }
            if jacobian is not None:
                kwargs["jac"] = jacobian
            popt, pcov = curve_fit(function, xdata, ydata, **kwargs)
            prediction = np.asarray(function(xdata, *popt), dtype=float)
            residual = prediction - ydata
            loss = float(np.dot(residual, residual))
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best = (np.asarray(popt, dtype=float), np.asarray(pcov, dtype=float), residual)
        except Exception as exc:  # every seed is an independent solver attempt
            failures.append(f"{type(exc).__name__}: {exc}")
    return best, best_loss, failures


def fit_selected(selected: SelectedData, request: FitRequest) -> FitResult:
    """Fit already-selected rows and always return a validity-bearing result."""

    model = fit_model(request.model)
    sampled = _sample(selected, request.sample_budget)
    values = np.asarray(sampled.values)
    if values.ndim != 2 or values.shape[1] != 1:
        return _invalid(
            model,
            request,
            "fit requires exactly one scalar value per selected row; "
            f"got value shape {values.shape}. Select a data component or reduce "
            "the complete data_shape explicitly before fitting.",
            sampled.count,
        )
    if sampled.count < model.min_points:
        return _invalid(
            model, request,
            f"selection contains {sampled.count} finite points; {model.key} needs at least {model.min_points}",
            sampled.count)
    y = np.asarray(values[:, 0], dtype=float)
    try:
        if model.family == "1d":
            if "x" not in sampled.coordinates:
                raise KeyError("1D fit requires x coordinates")
            xdata = np.asarray(sampled.coordinates["x"], dtype=float)
        else:
            if "x" not in sampled.coordinates or "y" not in sampled.coordinates:
                raise KeyError("2D fit requires x and y coordinates")
            xdata = (np.asarray(sampled.coordinates["x"], dtype=float),
                     np.asarray(sampled.coordinates["y"], dtype=float))
        seeds, bounds = model.initialize(xdata, y)
        seeds, bounds = _apply_request_overrides(model, request, seeds, bounds)
    except Exception as exc:
        return _invalid(model, request, f"fit initialization failed: {exc}", sampled.count)

    if not request.solve:
        popt = np.asarray(seeds[0], dtype=float)
        prediction = np.asarray(model.function(xdata, *popt), dtype=float)
        residual = prediction - y
        loss = float(np.dot(residual, residual))
        rmse = float(np.sqrt(np.mean(residual**2)))
        total = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - loss / total) if total > 0 else float("nan")
        return FitResult(
            model.key, model.names, popt, None, True, "initial values",
            {"rmse": rmse, "r2": r2}, sampled.count, request.coordinate_frame)

    # Only a REAL solve is gated: the ``solve=False`` reconstruction above (draw a curve from
    # already-known params, e.g. a grid cell redrawing published node params) never runs curve_fit,
    # so it is legal on any thread.  ``fit_histogram`` (the bounded bimodal micro-solve) does not route
    # through here and is exempt by construction (#6c).
    _assert_off_gui_thread()
    best, best_loss, failures = _solve_candidates(
        model.function,
        xdata,
        y,
        seeds,
        bounds,
        max_evaluations=request.max_evaluations,
    )
    if best is None:
        detail = failures[-1] if failures else "no converged seed"
        return _invalid(model, request, f"fit failed: {detail}", sampled.count)

    popt, pcov, residual = best
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - best_loss / total) if total > 0 else float("nan")
    return FitResult(model.key, model.names, popt, pcov, True, "ok",
                     {"rmse": rmse, "r2": r2}, sampled.count, request.coordinate_frame)


def fit_histogram(edges, counts, mode: str) -> HistogramFitResult:
    """Fit a binned readout distribution with one or two zero-offset Gaussians.

    This is deliberately separate from :class:`FitRequest`: it is the domain
    fit behind histogram/readout presentation, not a generic curve fit offered
    by the plot-family model picker.  Both the rolling monitor side-band and
    standalone/grid histograms call this exact seed/bounds/solver path.
    """

    requested = str(mode).strip().lower()
    if requested not in {"none", "single", "double"}:
        raise ValueError("histogram fit mode must be 'none', 'single', or 'double'")
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if requested == "none":
        return HistogramFitResult.invalid(requested, "fit disabled")
    if edges.ndim != 1 or counts.ndim != 1 or edges.size != counts.size + 1:
        return HistogramFitResult.invalid(
            requested,
            f"histogram edges/counts mismatch: edges={edges.shape}, counts={counts.shape}",
        )
    if counts.size < 2 or not np.all(np.isfinite(edges)) or not np.all(np.isfinite(counts)):
        return HistogramFitResult.invalid(requested, "histogram needs finite bins and counts")
    if np.any(np.diff(edges) <= 0) or np.any(counts < 0):
        return HistogramFitResult.invalid(
            requested, "histogram edges must increase and counts must be non-negative")
    n_samples = float(counts.sum())
    if n_samples < 6 or np.count_nonzero(counts) < 2:
        return HistogramFitResult.invalid(requested, "histogram has insufficient occupied bins")

    centers = (edges[:-1] + edges[1:]) / 2.0
    span = float(edges[-1] - edges[0]) or 1.0

    def weighted_stats(c, w):
        total = float(w.sum())
        mean = float((w * c).sum() / total)
        sigma = float(np.sqrt(max((w * (c - mean) ** 2).sum() / total, 0.0)))
        return mean, sigma

    def solve_single(*, fallback_status: str | None = None) -> HistogramFitResult:
        mean, sigma = weighted_stats(centers, counts)
        sigma = max(sigma, span / 40.0, 1e-9)
        peak = max(float(np.max(counts)), 1.0)
        seed = [peak, mean, sigma]
        bounds = (
            np.array([0.0, float(edges[0]), max(span / 200.0, 1e-12)]),
            np.array([max(peak * 5.0, 1.0), float(edges[-1]), max(span * 2.0, 1e-12)]),
        )
        best, _loss, failures = _solve_candidates(
            gaussian,
            centers,
            counts,
            [seed],
            bounds,
            max_evaluations=20_000,
            jacobian=gaussian_jacobian,
        )
        if best is None:
            detail = failures[-1] if failures else "no converged seed"
            return HistogramFitResult.invalid(requested, f"single histogram fit failed: {detail}")
        popt, pcov, _residual = best
        status = fallback_status or "ok"
        return HistogramFitResult(requested, 1, popt, pcov, True, False, None, status)

    if requested == "single":
        return solve_single()

    # Otsu's between-class variance supplies stable dark/bright seeds directly
    # from the binned populations, so this remains O(number of bins).
    w0 = np.cumsum(counts)[:-1]
    m0 = np.cumsum(counts * centers)[:-1]
    w1 = n_samples - w0
    total_moment = float((counts * centers).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(
            (w0 > 0) & (w1 > 0),
            w0 * w1 * (m0 / w0 - (total_moment - m0) / w1) ** 2,
            0.0,
        )
    split = int(np.argmax(score))
    left_c, left_w = centers[: split + 1], counts[: split + 1]
    right_c, right_w = centers[split + 1:], counts[split + 1:]
    if float(left_w.sum()) < 2 or float(right_w.sum()) < 2:
        half = int(np.searchsorted(np.cumsum(counts), n_samples / 2.0))
        left_c, left_w = centers[: half + 1], counts[: half + 1]
        right_c, right_w = centers[half + 1:], counts[half + 1:]
        if float(left_w.sum()) <= 0 or float(right_w.sum()) <= 0:
            return solve_single(fallback_status="double seed split failed; single fallback")
    mu0, sigma0 = weighted_stats(left_c, left_w)
    mu1, sigma1 = weighted_stats(right_c, right_w)
    if mu0 > mu1:
        mu0, mu1 = mu1, mu0
        sigma0, sigma1 = sigma1, sigma0
    sigma0 = max(sigma0, span / 40.0, 1e-9)
    sigma1 = max(sigma1, span / 40.0, 1e-9)

    def amplitude_near(mean):
        index = int(np.clip(np.searchsorted(centers, mean), 0, len(counts) - 1))
        return max(float(counts[index]), 1.0)

    amp0, amp1 = amplitude_near(mu0), amplitude_near(mu1)
    seed = [amp0, mu0, sigma0, amp1, mu1, sigma1]
    bounds = (
        np.array([0.0, edges[0], span / 200.0, 0.0, edges[0], span / 200.0]),
        np.array([max(amp0 * 5.0, 1.0), edges[-1], span * 2.0,
                  max(amp1 * 5.0, 1.0), edges[-1], span * 2.0]),
    )
    best, _loss, failures = _solve_candidates(
        bimodal_model,
        centers,
        counts,
        [seed],
        bounds,
        max_evaluations=20_000,
        jacobian=bimodal_jacobian,
    )
    if best is None:
        detail = failures[-1] if failures else "no converged seed"
        return solve_single(fallback_status=f"double fit failed ({detail}); single fallback")
    popt, pcov, _residual = best
    if popt[1] > popt[4]:
        order = np.array([3, 4, 5, 0, 1, 2])
        popt = popt[order]
        pcov = pcov[np.ix_(order, order)]
    fitted_separation = abs(popt[4] - popt[1]) / (
        abs(popt[2]) + abs(popt[5]) + 1e-12)
    separated = bool(fitted_separation >= 1.5)
    threshold = None
    if separated and popt[4] > popt[1]:
        between = np.linspace(float(popt[1]), float(popt[4]), 400)
        difference = np.abs(
            gaussian(between, *popt[:3]) - gaussian(between, *popt[3:]))
        threshold = float(between[int(np.nanargmin(difference))])
    return HistogramFitResult(
        requested, 2, popt, pcov, True, separated, threshold, "ok")


def fit_data(
    coordinates: Mapping[str, object],
    values,
    request: FitRequest,
) -> FitResult:
    """Select and fit raw coordinate/value rows through the shared pipeline."""

    model = fit_model(request.model)
    try:
        selected = select_rows(coordinates, values, request.selection, finite=True)
    except Exception as exc:
        return _invalid(model, request, f"selection failed: {exc}")
    return fit_selected(selected, request)


def fit_image(
    image,
    request: FitRequest,
    *,
    origin: Sequence[float] | None = None,
    grid: RegularRaster | None = None,
) -> FitResult:
    """Fit a regular 2-D image without materialising full-frame float coordinate grids.

    Selection and finite filtering happen before deterministic sampling, exactly as
    in :func:`fit_data`.  Only the sampled coordinates are allocated as floats; a
    multi-megapixel frame therefore costs one boolean mask rather than two equally
    large ``x``/``y`` arrays.  ``origin`` maps local array indices into the request's
    coordinate frame and defaults to ``selection.metadata['origin']`` when present.  ``grid`` is
    the authoritative compact geometry when coordinates have non-default origin or spacing.
    """

    model = fit_model(request.model)
    if model.family != "2d":
        return _invalid(model, request, f"{model.key} is not a 2D image model")
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        return _invalid(model, request, f"2D fit requires a non-empty image; got {array.shape}")
    if array.dtype.kind not in "buif":
        return _invalid(model, request, f"2D fit requires a real numeric image; got {array.dtype}")

    if grid is not None:
        if not isinstance(grid, RegularRaster):
            return _invalid(model, request, f"image grid must be RegularRaster, got {type(grid).__name__}")
        if tuple(array.shape) != tuple(grid.shape):
            return _invalid(
                model, request,
                f"image shape {array.shape} does not match regular raster {grid.shape}")
        ox, oy = grid.origin
        dx, dy = grid.spacing
    else:
        if origin is None:
            origin = request.selection.metadata.get("origin", (0.0, 0.0))
        try:
            ox, oy = (float(value) for value in origin)
        except Exception:
            return _invalid(model, request, f"image origin must be a two-number sequence, got {origin!r}")
        dx = dy = 1.0

    x_start, x_stop = 0, int(array.shape[1])
    y_start, y_stop = 0, int(array.shape[0])
    value_bounds = None
    for axis_range in request.selection.ranges:
        if axis_range.axis == "x":
            x_start = max(x_start, int(np.ceil((axis_range.low - ox) / dx)))
            x_stop = min(x_stop, int(np.floor((axis_range.high - ox) / dx)) + 1)
        elif axis_range.axis == "y":
            y_start = max(y_start, int(np.ceil((axis_range.low - oy) / dy)))
            y_stop = min(y_stop, int(np.floor((axis_range.high - oy) / dy)) + 1)
        elif axis_range.axis == "value":
            value_bounds = axis_range
        else:
            return _invalid(
                model, request,
                f"selection needs coordinate {axis_range.axis!r}; image coordinates are x, y, value")
    if x_start >= x_stop or y_start >= y_stop:
        return _invalid(model, request, "selection contains 0 finite points")

    # Rectangle selections on a regular raster are contiguous.  Keep this as a
    # view: ``np.ix_`` advanced indexing copied the entire selected frame before
    # the sample budget was applied (18.4 MiB for a 1920x1200 float64 image).
    selected_image = array[y_start:y_stop, x_start:x_stop]
    finite = np.isfinite(selected_image)
    if value_bounds is not None:
        # Compare in the source dtype.  AxisRange.mask intentionally coerces
        # arbitrary coordinate inputs to float, but doing so here would copy a
        # whole uint8/uint16 camera frame before the sample budget.
        finite &= (selected_image >= value_bounds.low) & (selected_image <= value_bounds.high)
    all_finite = bool(np.all(finite))
    row_counts = None if all_finite else np.count_nonzero(finite, axis=1)
    candidate_count = int(finite.size if all_finite else np.sum(row_counts, dtype=np.int64))
    if candidate_count == 0:
        return _invalid(model, request, "selection contains 0 finite points")

    # Select evenly spaced *valid ranks* without first materialising every valid
    # flat index.  ``flatnonzero`` used an additional int64 array as large as the
    # frame even though at most ``sample_budget`` rows reach the solver.
    take_count = min(candidate_count, request.sample_budget)
    if candidate_count > take_count:
        ranks = np.linspace(0, candidate_count - 1, take_count, dtype=np.int64)
    else:
        ranks = np.arange(candidate_count, dtype=np.int64)
    if all_finite:
        local_y, local_x = np.divmod(ranks, selected_image.shape[1])
    else:
        cumulative = np.cumsum(row_counts, dtype=np.int64)
        local_y = np.searchsorted(cumulative, ranks, side="right")
        previous = np.where(local_y == 0, 0, cumulative[local_y - 1])
        offsets = ranks - previous
        local_x = np.empty(take_count, dtype=np.int64)
        unique_rows, output_starts = np.unique(local_y, return_index=True)
        output_stops = np.r_[output_starts[1:], take_count]
        for row, output_start, output_stop in zip(unique_rows, output_starts, output_stops):
            valid_columns = np.flatnonzero(finite[row])
            output = slice(output_start, output_stop)
            local_x[output] = valid_columns[offsets[output]]

    rows = local_y + y_start
    cols = local_x + x_start
    selected = SelectedData(
        {"x": cols.astype(float) * dx + ox, "y": rows.astype(float) * dy + oy},
        array[rows, cols],
        rows * array.shape[1] + cols,
        request.selection,
        (1,),
    )
    return fit_selected(selected, request)


__all__ = [
    "FitModel", "FitRequest", "FitResult", "HistogramFitResult", "fit_data", "fit_histogram",
    "fit_image", "fit_model", "fit_models", "fit_selected",
]
