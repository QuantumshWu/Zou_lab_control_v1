"""Closed model catalogue for generic named-axis fitting.

The catalogue contains only domain-neutral curve/image models.  Readout
mixtures, calibration PSFs and histogram decisions deliberately remain in
their bounded domains even when their formulas look superficially similar.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

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
GENERIC_CURVE = (HISTOGRAM_BIN, SCAN_POINT, SPATIAL_X, SPATIAL_Y, SPECTRAL)
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


_IMPLEMENTATION_BY_ID = MappingProxyType(
    {
        "lorentzian": (_lorentzian, _init_lorentzian),
        "gaussian_offset": (_gaussian_offset, _init_gaussian),
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
]
