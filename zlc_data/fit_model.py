"""Closed model catalogue for generic named-axis fitting.

The catalogue contains only domain-neutral curve/image/distribution models.
Readout decisions and calibration PSFs remain in their bounded domains even
when their formulas look superficially similar.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from zlc_storage.canonical import canonical_text

from .axis import (
    HISTOGRAM_BIN,
    AxisRoleId,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
)


class ParameterUnitRelation(str, Enum):
    VALUE = "VALUE"
    AXIS_0 = "AXIS_0"
    AXIS_1 = "AXIS_1"
    INVERSE_AXIS_0 = "INVERSE_AXIS_0"
    RADIAN = "RADIAN"


class FitParameterDomain(str, Enum):
    REAL = "REAL"
    POSITIVE = "POSITIVE"
    NONNEGATIVE = "NONNEGATIVE"
    PHASE_RADIANS = "PHASE_RADIANS"


@dataclass(frozen=True)
class FitParameterDefinition:
    name: str
    unit_relation: ParameterUnitRelation
    domain: FitParameterDomain = FitParameterDomain.REAL

    def __post_init__(self) -> None:
        canonical_text(self.name, "fit parameter name")
        if not isinstance(self.unit_relation, ParameterUnitRelation):
            raise TypeError("unit_relation must be ParameterUnitRelation")
        if not isinstance(self.domain, FitParameterDomain):
            raise TypeError("domain must be FitParameterDomain")

    def accepts(self, value: float) -> bool:
        value = float(value)
        if not np.isfinite(value):
            return False
        if self.domain is FitParameterDomain.REAL:
            return True
        if self.domain is FitParameterDomain.POSITIVE:
            return value > 0.0
        if self.domain is FitParameterDomain.NONNEGATIVE:
            return value >= 0.0
        if self.domain is FitParameterDomain.PHASE_RADIANS:
            return -np.pi <= value < np.pi
        raise RuntimeError("unsupported fit parameter domain")

    def has_free_interval(self, lower: float | None, upper: float | None) -> bool:
        """Whether static/user bounds contain a representable free solver value."""

        domain_lower, domain_upper = self.solver_bounds
        effective_lower = max(domain_lower, -np.inf if lower is None else lower)
        effective_upper = min(domain_upper, np.inf if upper is None else upper)
        if not effective_lower < effective_upper:
            return False
        return bool(np.nextafter(effective_lower, effective_upper) < effective_upper)

    @property
    def solver_bounds(self) -> tuple[float, float]:
        """Static mathematical domain; data heuristics never become hard bounds."""

        if self.domain is FitParameterDomain.REAL:
            return (-np.inf, np.inf)
        if self.domain is FitParameterDomain.POSITIVE:
            return (float(np.nextafter(0.0, np.inf)), np.inf)
        if self.domain is FitParameterDomain.NONNEGATIVE:
            return (0.0, np.inf)
        if self.domain is FitParameterDomain.PHASE_RADIANS:
            return (-np.pi, float(np.nextafter(np.pi, -np.inf)))
        raise RuntimeError("unsupported fit parameter domain")


@dataclass(frozen=True)
class FitModelDefinition:
    model_id: str
    display_name: str
    axis_requirements: tuple[tuple[AxisRoleId, ...], ...]
    parameters: tuple[FitParameterDefinition, ...]
    require_common_axis_unit: bool = False
    require_common_coordinate_frame: bool = False

    def __post_init__(self) -> None:
        for field, value in (("model_id", self.model_id), ("display_name", self.display_name)):
            canonical_text(value, field)
        requirements = tuple(tuple(item) for item in self.axis_requirements)
        if len(requirements) not in (1, 2):
            raise ValueError("axis_requirements must describe one or two independent axes")
        normalized_requirements: list[tuple[AxisRoleId, ...]] = []
        for roles in requirements:
            if not roles or any(not isinstance(role, AxisRoleId) for role in roles):
                raise ValueError("fit axis requirement needs declared AxisRoleId values")
            if len(set(roles)) != len(roles):
                raise ValueError("fit axis requirement roles must be unique")
            normalized_requirements.append(
                tuple(sorted(roles, key=lambda item: item.value))
            )
        parameters = tuple(self.parameters)
        if not parameters or any(not isinstance(item, FitParameterDefinition) for item in parameters):
            raise ValueError("model requires FitParameterDefinition values")
        names = tuple(item.name for item in parameters)
        if len(set(names)) != len(names):
            raise ValueError("model parameter names must be unique")
        if not isinstance(self.require_common_axis_unit, bool):
            raise TypeError("require_common_axis_unit must be bool")
        if not isinstance(self.require_common_coordinate_frame, bool):
            raise TypeError("require_common_coordinate_frame must be bool")
        object.__setattr__(self, "axis_requirements", tuple(normalized_requirements))
        object.__setattr__(self, "parameters", parameters)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    @property
    def independent_arity(self) -> int:
        return len(self.axis_requirements)


VALUE = ParameterUnitRelation.VALUE
AXIS_0 = ParameterUnitRelation.AXIS_0
AXIS_1 = ParameterUnitRelation.AXIS_1
INVERSE_AXIS_0 = ParameterUnitRelation.INVERSE_AXIS_0
RADIAN = ParameterUnitRelation.RADIAN
POSITIVE = FitParameterDomain.POSITIVE
NONNEGATIVE = FitParameterDomain.NONNEGATIVE
PHASE_RADIANS = FitParameterDomain.PHASE_RADIANS
GENERIC_CURVE = (SCAN_POINT, SPATIAL_X, SPATIAL_Y, SPECTRAL)
HISTOGRAM_CURVE = (HISTOGRAM_BIN,)
SPACE_X = (SPATIAL_X,)
SPACE_Y = (SPATIAL_Y,)


_MODELS = (
    FitModelDefinition(
        "lorentzian",
        "Lorentzian",
        (GENERIC_CURVE,),
        (
            FitParameterDefinition("center", AXIS_0),
            FitParameterDefinition("fwhm", AXIS_0, POSITIVE),
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
        ),
    ),
    FitModelDefinition(
        "gaussian_offset",
        "Gaussian with offset",
        (GENERIC_CURVE,),
        (
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("sigma", AXIS_0, POSITIVE),
            FitParameterDefinition("center", AXIS_0),
        ),
    ),
    FitModelDefinition(
        "histogram_gaussian",
        "Single Gaussian",
        (HISTOGRAM_CURVE,),
        (
            FitParameterDefinition("amplitude", VALUE, NONNEGATIVE),
            FitParameterDefinition("center", AXIS_0),
            FitParameterDefinition("sigma", AXIS_0, POSITIVE),
        ),
    ),
    FitModelDefinition(
        "bimodal_gaussian",
        "Bimodal Gaussian",
        (HISTOGRAM_CURVE,),
        (
            FitParameterDefinition("center", AXIS_0),
            FitParameterDefinition("center_splitting", AXIS_0, NONNEGATIVE),
            FitParameterDefinition("left_amplitude", VALUE, NONNEGATIVE),
            FitParameterDefinition("left_sigma", AXIS_0, POSITIVE),
            FitParameterDefinition("right_amplitude", VALUE, NONNEGATIVE),
            FitParameterDefinition("right_sigma", AXIS_0, POSITIVE),
        ),
    ),
    FitModelDefinition(
        "symmetric_lorentzian_doublet",
        "Symmetric Lorentzian doublet",
        (GENERIC_CURVE,),
        (
            FitParameterDefinition("center", AXIS_0),
            FitParameterDefinition("common_fwhm", AXIS_0, POSITIVE),
            FitParameterDefinition("component_amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("center_splitting", AXIS_0, NONNEGATIVE),
        ),
    ),
    FitModelDefinition(
        "damped_sine",
        "Damped sine",
        (GENERIC_CURVE,),
        (
            FitParameterDefinition("amplitude", VALUE, NONNEGATIVE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("baseband_frequency", INVERSE_AXIS_0, POSITIVE),
            FitParameterDefinition("decay_time", AXIS_0, POSITIVE),
            FitParameterDefinition("phase", RADIAN, PHASE_RADIANS),
        ),
    ),
    FitModelDefinition(
        "exponential_decay",
        "Exponential decay",
        (GENERIC_CURVE,),
        (
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("decay_time", AXIS_0, POSITIVE),
        ),
    ),
    FitModelDefinition(
        "radial_gaussian_center",
        "Radial Gaussian center",
        (SPACE_X, SPACE_Y),
        (
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("one_over_e_radius", AXIS_0, POSITIVE),
            FitParameterDefinition("center_x", AXIS_0),
            FitParameterDefinition("center_y", AXIS_1),
        ),
        require_common_axis_unit=True,
        require_common_coordinate_frame=True,
    ),
)

_MODEL_BY_ID: Mapping[str, FitModelDefinition] = MappingProxyType(
    {model.model_id: model for model in _MODELS}
)


def fit_model_catalog() -> tuple[FitModelDefinition, ...]:
    return _MODELS


def fit_model_definition(model_id: str) -> FitModelDefinition:
    canonical_text(model_id, "model_id")
    try:
        return _MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown fit model {model_id!r}") from exc


def evaluate_fit_model(
    model: FitModelDefinition | str,
    coordinates: tuple[np.ndarray, ...],
    parameters: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Evaluate a closed-catalog model on the supplied absolute coordinates."""

    definition = fit_model_definition(model) if isinstance(model, str) else model
    if not isinstance(definition, FitModelDefinition):
        raise TypeError("model must be FitModelDefinition or canonical model id")
    if definition != fit_model_definition(definition.model_id):
        raise ValueError("model definition does not belong to the closed fit catalog")
    coords = tuple(np.asarray(value, dtype=np.float64) for value in coordinates)
    if len(coords) != definition.independent_arity:
        raise ValueError("coordinate arity does not match model")
    if not coords or any(item.shape != coords[0].shape for item in coords):
        raise ValueError("model coordinate arrays must share one shape")
    params = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if params.size != len(definition.parameters):
        raise ValueError("parameter count does not match model")
    for parameter, value in zip(definition.parameters, params):
        if not parameter.accepts(float(value)):
            raise ValueError(
                f"parameter {parameter.name!r} violates its {parameter.domain.value} domain"
            )
    evaluator, _ = _IMPLEMENTATION_BY_ID[definition.model_id]
    result = evaluator(*coords, *params)
    return np.asarray(result, dtype=np.float64)


def evaluate_fit_model_components(
    model: FitModelDefinition | str,
    coordinates: tuple[np.ndarray, ...],
    parameters: np.ndarray | tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Evaluate model-owned visual components without duplicating formulas."""

    definition = fit_model_definition(model) if isinstance(model, str) else model
    total = evaluate_fit_model(definition, coordinates, parameters)
    if definition.model_id != "bimodal_gaussian":
        return (total,)
    x = np.asarray(coordinates[0], dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    left, right = _bimodal_gaussian_components(x, *values)
    return (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        total,
    )


def initialize_fit_model(
    model: FitModelDefinition,
    coordinates: tuple[np.ndarray, ...],
    observations: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(model, FitModelDefinition):
        raise TypeError("model must be FitModelDefinition")
    if model != fit_model_definition(model.model_id):
        raise ValueError("model definition does not belong to the closed fit catalog")
    coords = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in coordinates)
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    if len(coords) != model.independent_arity or any(item.shape != y.shape for item in coords):
        raise ValueError("initializer coordinate/observation shapes disagree")
    if y.size < 2:
        raise ValueError("model initialization requires at least two observations")
    if not np.all(np.isfinite(y)) or any(not np.all(np.isfinite(item)) for item in coords):
        raise ValueError("fit initialization requires finite authoritative observations")
    _, initializer = _IMPLEMENTATION_BY_ID[model.model_id]
    seeds = initializer(*coords, y)
    canonical = tuple(tuple(float(value) for value in seed) for seed in seeds)
    if not canonical or any(len(seed) != len(model.parameters) for seed in canonical):
        raise ValueError("model initializer returned an invalid seed shape")
    if any(not np.isfinite(value) for seed in canonical for value in seed):
        raise ValueError("model initializer returned a non-finite seed")
    return canonical


def histogram_gaussian_display_diagnostic(
    bin_edges: np.ndarray,
    counts: np.ndarray,
) -> tuple[float, float, float] | None:
    """Return a fast, display-only Gaussian moment diagnostic.

    The rolling monitor's side distribution is a read-only visual aid, not a
    Fit command and never a source of published parameters.  It nevertheless
    reuses the closed model catalogue's initializer so its Gaussian formula
    and parameter ordering cannot drift into a second fitting authority.
    """

    edges = np.asarray(bin_edges, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if (
        edges.ndim != 1
        or weights.ndim != 1
        or edges.size != weights.size + 1
        or weights.size < 3
        or not np.all(np.isfinite(edges))
        or not np.all(np.isfinite(weights))
        or not np.all(np.diff(edges) > 0.0)
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
    ):
        return None
    centers = (edges[:-1] + edges[1:]) / 2.0
    try:
        parameters = initialize_fit_model(
            fit_model_definition("histogram_gaussian"),
            (centers,),
            weights,
        )[0]
    except ValueError:
        return None
    return tuple(float(value) for value in parameters)


def _span(values: np.ndarray) -> float:
    span = float(np.max(values) - np.min(values))
    return abs(span) if span != 0 else 1.0


def _positive_floor(scale: float) -> float:
    return max(abs(float(scale)) * 1e-12, np.finfo(np.float64).tiny)


def _safe_exp(value: float) -> float:
    return float(np.exp(np.clip(float(value), -700.0, 700.0)))


def _principal_phase(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _lorentzian(x, center, fwhm, amplitude, offset):
    half_squared = (fwhm / 2.0) ** 2
    return amplitude * half_squared / ((x - center) ** 2 + half_squared) + offset


def _gaussian_offset(x, amplitude, offset, sigma, center):
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset


def _bimodal_gaussian(
    x,
    center,
    center_splitting,
    left_amplitude,
    left_sigma,
    right_amplitude,
    right_sigma,
):
    left, right = _bimodal_gaussian_components(
        x,
        center,
        center_splitting,
        left_amplitude,
        left_sigma,
        right_amplitude,
        right_sigma,
    )
    return left + right


def _bimodal_gaussian_components(
    x,
    center,
    center_splitting,
    left_amplitude,
    left_sigma,
    right_amplitude,
    right_sigma,
):
    left_center = center - center_splitting / 2.0
    right_center = center + center_splitting / 2.0
    left = left_amplitude * np.exp(
        -((x - left_center) ** 2) / (2.0 * left_sigma**2)
    )
    right = right_amplitude * np.exp(
        -((x - right_center) ** 2) / (2.0 * right_sigma**2)
    )
    return left, right


def _histogram_gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))


def _symmetric_lorentzian_doublet(
    x,
    center,
    common_fwhm,
    component_amplitude,
    offset,
    center_splitting,
):
    return (
        _lorentzian(
            x - center_splitting / 2.0,
            center,
            common_fwhm,
            component_amplitude,
            0.0,
        )
        + _lorentzian(
            x + center_splitting / 2.0,
            center,
            common_fwhm,
            component_amplitude,
            0.0,
        )
        + offset
    )


def _damped_sine(x, amplitude, offset, frequency, decay_time, phase):
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase) * np.exp(
        -x / decay_time
    ) + offset


def _exponential_decay(x, amplitude, offset, decay_time):
    return amplitude * np.exp(-x / decay_time) + offset


def _radial_gaussian_center(
    x,
    y,
    amplitude,
    offset,
    one_over_e_radius,
    center_x,
    center_y,
):
    squared_radius = (x - center_x) ** 2 + (y - center_y) ** 2
    return amplitude * np.exp(-squared_radius / one_over_e_radius**2) + offset


def _radial_gaussian_center_jacobian(
    coordinates: tuple[np.ndarray, np.ndarray],
    parameters: np.ndarray,
) -> np.ndarray:
    """Exact residual Jacobian for the full-image radial polish."""

    x, y = np.broadcast_arrays(
        *(np.asarray(values, dtype=np.float64) for values in coordinates)
    )
    amplitude, _offset, radius, center_x, center_y = np.asarray(
        parameters,
        dtype=np.float64,
    )
    delta_x = x - center_x
    delta_y = y - center_y
    squared_radius = delta_x**2 + delta_y**2
    exponential = np.exp(-squared_radius / radius**2)
    scaled = amplitude * exponential
    jacobian = np.empty((x.size, 5), dtype=np.float64)
    jacobian[:, 0] = exponential.reshape(-1)
    jacobian[:, 1] = 1.0
    jacobian[:, 2] = (scaled * (2.0 * squared_radius / radius**3)).reshape(-1)
    jacobian[:, 3] = (scaled * (2.0 * delta_x / radius**2)).reshape(-1)
    jacobian[:, 4] = (scaled * (2.0 * delta_y / radius**2)).reshape(-1)
    return jacobian


def _radial_gaussian_center_axis_basis(
    x_values: np.ndarray,
    y_values: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return the exact separable model/derivative basis on regular axes."""

    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    _amplitude, _offset, radius, center_x, center_y = np.asarray(
        parameters, dtype=np.float64
    )
    delta_x = x - center_x
    delta_y = y - center_y
    radial_x = np.exp(-(delta_x**2) / radius**2)
    radial_y = np.exp(-(delta_y**2) / radius**2)
    return (
        radial_x,
        radial_y,
        radial_x * (2.0 * delta_x**2 / radius**3),
        radial_y * (2.0 * delta_y**2 / radius**3),
        radial_x * (2.0 * delta_x / radius**2),
        radial_y * (2.0 * delta_y / radius**2),
    )


def _init_lorentzian(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, ...], ...]:
    width = _span(x) / 4.0
    y_range = _span(y)
    low_y, high_y = float(np.min(y)), float(np.max(y))
    seeds = (
        (float(x[np.argmax(y)]), width, y_range, low_y),
        (float(x[np.argmin(y)]), width, -y_range, high_y),
    )
    return seeds


def _init_gaussian(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, ...], ...]:
    low_y, high_y = float(np.min(y)), float(np.max(y))
    amplitude = high_y - low_y or 1.0
    sigma = _span(x) / 6.0
    seeds = (
        (amplitude, low_y, sigma, float(x[np.argmax(y)])),
        (-amplitude, high_y, sigma, float(x[np.argmin(y)])),
    )
    return seeds


def _init_bimodal_gaussian(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    """Seed two positive peaks from the largest between-class separation."""

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = np.clip(y[order], 0.0, None)
    total = float(np.sum(y))
    if total <= 0.0:
        raise ValueError("bimodal Gaussian initialization requires positive counts")
    left_weight = np.cumsum(y)[:-1]
    left_moment = np.cumsum(y * x)[:-1]
    right_weight = total - left_weight
    total_moment = float(np.sum(y * x))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(
            (left_weight > 0.0) & (right_weight > 0.0),
            left_weight
            * right_weight
            * np.square(
                left_moment / left_weight
                - (total_moment - left_moment) / right_weight
            ),
            0.0,
        )
    split = int(np.argmax(score))
    if not np.isfinite(score[split]) or score[split] <= 0.0:
        split = max(0, min(len(x) - 2, len(x) // 2 - 1))

    def component(values_x, weights):
        weight = float(np.sum(weights))
        if weight <= 0.0:
            raise ValueError("bimodal Gaussian seed produced an empty component")
        center = float(np.sum(weights * values_x) / weight)
        variance = float(np.sum(weights * (values_x - center) ** 2) / weight)
        sigma = max(math.sqrt(max(variance, 0.0)), _span(x) / 100.0)
        nearest = int(np.argmin(np.abs(x - center)))
        amplitude = max(float(y[nearest]), 1.0)
        return amplitude, center, sigma

    left = component(x[: split + 1], y[: split + 1])
    right = component(x[split + 1 :], y[split + 1 :])
    if left[1] > right[1]:
        left, right = right, left
    center = (left[1] + right[1]) / 2.0
    splitting = right[1] - left[1]
    return ((center, splitting, left[0], left[2], right[0], right[2]),)


def _init_histogram_gaussian(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    weights = np.clip(y, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("histogram Gaussian initialization requires positive counts")
    center = float(np.sum(weights * x) / total)
    variance = float(np.sum(weights * (x - center) ** 2) / total)
    sigma = max(math.sqrt(max(variance, 0.0)), _span(x) / 100.0)
    amplitude = max(float(np.max(weights)), 1.0)
    return ((amplitude, center, sigma),)


def _init_symmetric_lorentzian_doublet(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    from scipy.signal import find_peaks

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    x_span = _span(x)
    y_range = _span(y)
    step = float(np.median(np.abs(np.diff(np.sort(np.unique(x)))))) if np.unique(x).size > 1 else 1.0
    step = step or 1.0
    seeds: list[tuple[float, ...]] = []
    for sign in (1.0, -1.0):
        signed = sign * y
        peaks, properties = find_peaks(signed, width=1, prominence=y_range / 8.0)
        if peaks.size == 0:
            continue
        order = peaks[np.argsort(signed[peaks])[::-1]][:4]
        widths = properties.get("widths", np.ones(peaks.size))
        width_by_peak = {int(peak): max(float(widths[i]) * step, step) for i, peak in enumerate(peaks)}
        first = int(order[0])
        for second_raw in order:
            second = int(second_raw)
            width = width_by_peak[first]
            seeds.append(
                (
                    float((x[first] + x[second]) / 2.0),
                    width,
                    sign * y_range,
                    float(np.min(y) if sign > 0 else np.max(y)),
                    float(abs(x[second] - x[first])),
                )
            )
    if not seeds:
        width = x_span / 8.0
        seeds = [
            (float(x[np.argmax(y)]), width, y_range, float(np.min(y)), width * 2.0),
            (float(x[np.argmin(y)]), width, -y_range, float(np.max(y)), width * 2.0),
        ]
    return tuple(seeds)


def _init_damped_sine(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, ...], ...]:
    local_amplitude = _span(y) / 2.0
    offset = float(np.mean(y))
    x_span = _span(x)
    ordered = np.argsort(x, kind="stable")
    sorted_x = x[ordered]
    sorted_y = y[ordered]
    differences = np.diff(sorted_x)
    uniform = (
        differences.size > 0
        and np.all(differences > 0)
        and np.allclose(differences, differences[0], rtol=1e-3, atol=_positive_floor(differences[0]))
    )
    if uniform:
        frequencies = np.fft.rfftfreq(sorted_y.size, d=float(differences[0]))
        spectrum = np.abs(np.fft.rfft(sorted_y - offset))
        frequency = (
            float(frequencies[1 + np.argmax(spectrum[1:])]) if frequencies.size > 1 else 1.0 / x_span
        )
    else:
        frequency = 1.0 / x_span
    frequency = max(abs(frequency), _positive_floor(1.0 / x_span))
    reference = float(sorted_x[0])
    amplitude = max(
        local_amplitude * _safe_exp(reference / x_span),
        _positive_floor(local_amplitude),
    )
    seeds = tuple(
        (
            amplitude,
            offset,
            frequency,
            x_span,
            _principal_phase(local_phase - 2.0 * np.pi * frequency * reference),
        )
        for local_phase in (-np.pi / 2.0, 0.0, np.pi / 2.0)
    )
    return seeds


def _init_exponential(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, ...], ...]:
    y_range = _span(y)
    offset = float(np.mean(y))
    decay_time = _span(x) / 2.0
    amplitude = y_range * _safe_exp(float(np.min(x)) / decay_time)
    return ((amplitude, offset, decay_time), (-amplitude, offset, decay_time))


def _init_center(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    spatial_seeds = _spatial_radial_center_seeds(x, y, z)
    if spatial_seeds:
        return spatial_seeds

    # Sparse/non-Cartesian point clouds have no neighbourhood topology on which
    # to distinguish a coherent image feature from one exceptional sample.  The
    # original full-observation moment remains the deterministic fallback for
    # those inputs (and for the deliberately tiny initializer examples).
    offset = float(np.median(z))
    seeds: list[tuple[float, ...]] = []
    for sign in (1.0, -1.0):
        weights = np.clip(sign * (z - offset), 0.0, None)
        total = float(np.sum(weights))
        if total <= 0:
            continue
        center_x = float(np.sum(weights * x) / total)
        center_y = float(np.sum(weights * y) / total)
        one_over_e_radius = float(
            np.sqrt(
                np.sum(weights * ((x - center_x) ** 2 + (y - center_y) ** 2))
                / total
            )
        )
        one_over_e_radius = one_over_e_radius or 1.0
        amplitude = (
            float(np.max(z) - offset)
            if sign > 0
            else float(np.min(z) - offset)
        )
        seeds.append((amplitude, offset, one_over_e_radius, center_x, center_y))
    if not seeds:
        raise ValueError("radial center fit requires non-flat contrast")
    return tuple(seeds)


def _spatial_radial_center_seeds(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    validity: np.ndarray | None = None,
    check_abort: Callable[[], None] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Find coherent extrema from packed observations or a regular image view."""

    image = np.asarray(z)
    if image.ndim == 2:
        x_values = np.asarray(x, dtype=np.float64).reshape(-1)
        y_values = np.asarray(y, dtype=np.float64).reshape(-1)
        if image.shape != (x_values.size, y_values.size):
            return ()
        valid = np.broadcast_to(
            True if validity is None else np.asarray(validity, dtype=bool),
            image.shape,
        )
    else:
        cartesian = _radial_cartesian_grid(x, y, z)
        if cartesian is None:
            return ()
        x_values, y_values, image, _observation_indices = cartesian
        valid = np.broadcast_to(True, image.shape)
    if x_values.size < 5 or y_values.size < 5:
        return ()
    sample_x = np.linspace(
        0, x_values.size - 1, min(x_values.size, 257), dtype=np.int64
    )
    sample_y = np.linspace(
        0, y_values.size - 1, min(y_values.size, 257), dtype=np.int64
    )
    sampled = image[np.ix_(sample_x, sample_y)]
    background = np.asarray(sampled[valid[np.ix_(sample_x, sample_y)]], dtype=np.float64)
    if background.size < 2:
        return ()
    offset = float(np.median(background))
    noise_floor = 3.0 * 1.4826 * float(np.median(np.abs(background - offset)))
    maximum = -math.inf
    minimum = math.inf
    maximum_index = (0, 0)
    minimum_index = (0, 0)
    for start, filtered in _median_filtered_image_stripes(
        image,
        valid,
        offset=offset,
        check_abort=check_abort,
    ):
        local_maximum_flat = int(np.argmax(filtered))
        local_minimum_flat = int(np.argmin(filtered))
        local_maximum = float(filtered.reshape(-1)[local_maximum_flat])
        local_minimum = float(filtered.reshape(-1)[local_minimum_flat])
        if local_maximum > maximum:
            local_row, local_column = np.unravel_index(
                local_maximum_flat,
                filtered.shape,
            )
            maximum = local_maximum
            maximum_index = (start + int(local_row), int(local_column))
        if local_minimum < minimum:
            local_row, local_column = np.unravel_index(
                local_minimum_flat,
                filtered.shape,
            )
            minimum = local_minimum
            minimum_index = (start + int(local_row), int(local_column))
    steps = tuple(
        float(np.min(difference[difference > 0.0]))
        for values in (x_values, y_values)
        if np.any((difference := np.abs(np.diff(values))) > 0.0)
    )
    radius_floor = min(steps, default=1.0)
    numeric_floor = np.finfo(np.float64).eps * max(
        abs(offset), abs(minimum), abs(maximum), 1.0
    )

    candidates = tuple(
        (sign, index, sign * (extreme - offset))
        for sign, index, extreme in (
            (1.0, maximum_index, maximum),
            (-1.0, minimum_index, minimum),
        )
        if sign * (extreme - offset) > max(noise_floor, numeric_floor)
    )
    if not candidates:
        return ()

    row_profiles = {
        row: np.empty(image.shape[1], dtype=np.float64)
        for row in {index[0] for _sign, index, _peak in candidates}
    }
    column_profiles = {
        column: np.empty(image.shape[0], dtype=np.float64)
        for column in {index[1] for _sign, index, _peak in candidates}
    }
    for start, filtered in _median_filtered_image_stripes(
        image,
        valid,
        offset=offset,
        check_abort=check_abort,
    ):
        stop = start + filtered.shape[0]
        for column, profile in column_profiles.items():
            profile[start:stop] = filtered[:, column]
        for row, profile in row_profiles.items():
            if start <= row < stop:
                profile[:] = filtered[row - start, :]

    seeds: list[tuple[float, ...]] = []
    for sign, (row, column), peak in candidates:
        level = peak / math.e
        center_x, center_y = float(x_values[row]), float(y_values[column])
        radius = radius_floor
        profiles = (
            sign * (column_profiles[column] - offset),
            sign * (row_profiles[row] - offset),
        )
        for profile, peak_index, values, center in zip(
            profiles,
            (row, column),
            (x_values, y_values),
            (center_x, center_y),
        ):
            below_left = np.flatnonzero(profile[:peak_index] < level)
            below_right = np.flatnonzero(profile[peak_index + 1 :] < level)
            left = int(below_left[-1] + 1) if below_left.size else 0
            right = (
                # ``below_right`` starts one sample after the peak; the
                # radius endpoint is the last sample still at/above the
                # one-over-e level, immediately before that crossing.
                int(peak_index + below_right[0])
                if below_right.size
                else profile.size - 1
            )
            radius = max(
                radius,
                abs(float(values[left]) - center),
                abs(float(values[right]) - center),
            )
        seeds.append((sign * peak, offset, radius, center_x, center_y))
    return tuple(seeds)


def _median_filtered_image_stripes(
    image: np.ndarray,
    validity: np.ndarray,
    *,
    offset: float,
    check_abort: Callable[[], None] | None,
):
    """Yield exact 3x3-median rows without materializing an image-sized copy."""

    from scipy.ndimage import median_filter

    stripe_rows = 64
    height = image.shape[0]
    for start in range(0, height, stripe_rows):
        if check_abort is not None:
            check_abort()
        stop = min(height, start + stripe_rows)
        halo_start = max(0, start - 1)
        halo_stop = min(height, stop + 1)
        source = np.asarray(image[halo_start:halo_stop])
        valid = np.asarray(validity[halo_start:halo_stop], dtype=bool)
        if bool(np.all(valid)):
            seed = source
        else:
            # The sampled background median can lie between integer camera
            # codes.  Invalid cells therefore need a floating tile; writing
            # the median into an integer scratch would bias the initializer.
            # This remains stripe-bounded rather than image-sized.
            seed = np.empty(source.shape, dtype=np.float64)
            seed.fill(offset)
            np.copyto(seed, source, where=valid)
        if seed.dtype.kind == "f" and seed.dtype.itemsize == 2:
            seed = seed.astype(np.float32)
        filtered = median_filter(seed, size=3, mode="nearest")
        core_start = start - halo_start
        core_stop = core_start + stop - start
        yield start, np.asarray(filtered[core_start:core_stop])


def _radial_cartesian_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Recover a complete Cartesian image without assuming storage order."""

    # FitProblem packs fit axes in canonical C order.  Recognize that ordinary
    # path in linear time; hashing millions of repeated camera coordinates with
    # ``np.unique`` cost more than the refinement itself.
    if z.size and x.shape == y.shape == z.shape:
        starts = np.concatenate(
            (
                np.array((0,), dtype=np.int64),
                np.flatnonzero(x[1:] != x[:-1]).astype(np.int64, copy=False) + 1,
                np.array((z.size,), dtype=np.int64),
            )
        )
        run_lengths = np.diff(starts)
        if run_lengths.size and np.all(run_lengths == run_lengths[0]):
            x_size = int(run_lengths.size)
            y_size = int(run_lengths[0])
            x_grid = x.reshape(x_size, y_size)
            y_grid = y.reshape(x_size, y_size)
            x_values = x_grid[:, 0]
            y_values = y_grid[0, :]
            if np.all(x_grid == x_values[:, None]) and np.all(
                y_grid == y_values[None, :]
            ):
                return (
                    np.asarray(x_values, dtype=np.float64),
                    np.asarray(y_values, dtype=np.float64),
                    np.asarray(z, dtype=np.float64).reshape(x_size, y_size),
                    np.arange(z.size, dtype=np.int64).reshape(x_size, y_size),
                )

    x_values, x_inverse = np.unique(x, return_inverse=True)
    y_values, y_inverse = np.unique(y, return_inverse=True)
    if x_values.size * y_values.size != z.size:
        return None
    flat_positions = x_inverse * y_values.size + y_inverse
    if np.unique(flat_positions).size != z.size:
        return None
    image = np.empty((x_values.size, y_values.size), dtype=np.float64)
    observation_indices = np.empty(image.shape, dtype=np.int64)
    image.reshape(-1)[flat_positions] = z
    observation_indices.reshape(-1)[flat_positions] = np.arange(z.size)
    return x_values, y_values, image, observation_indices


_IMPLEMENTATION_BY_ID = MappingProxyType(
    {
        "lorentzian": (_lorentzian, _init_lorentzian),
        "gaussian_offset": (_gaussian_offset, _init_gaussian),
        "histogram_gaussian": (
            _histogram_gaussian,
            _init_histogram_gaussian,
        ),
        "bimodal_gaussian": (_bimodal_gaussian, _init_bimodal_gaussian),
        "symmetric_lorentzian_doublet": (
            _symmetric_lorentzian_doublet,
            _init_symmetric_lorentzian_doublet,
        ),
        "damped_sine": (_damped_sine, _init_damped_sine),
        "exponential_decay": (_exponential_decay, _init_exponential),
        "radial_gaussian_center": (_radial_gaussian_center, _init_center),
    }
)
if _IMPLEMENTATION_BY_ID.keys() != _MODEL_BY_ID.keys():  # pragma: no cover
    raise RuntimeError("fit model metadata and implementations must share one closed key set")


__all__ = [
    "FitModelDefinition",
    "FitParameterDefinition",
    "FitParameterDomain",
    "ParameterUnitRelation",
    "fit_model_catalog",
    "fit_model_definition",
    "evaluate_fit_model_components",
    "histogram_gaussian_display_diagnostic",
]
