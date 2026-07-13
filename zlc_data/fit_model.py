"""Closed, versioned model catalogue for generic named-axis fitting.

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

from .axis import AxisRoleId, SCAN_POINT, SPATIAL_X, SPATIAL_Y, SPECTRAL


class FitFamily(str, Enum):
    CURVE_1D = "CURVE_1D"
    IMAGE_2D = "IMAGE_2D"


class ParameterUnitRelation(str, Enum):
    VALUE = "VALUE"
    AXIS_0 = "AXIS_0"
    AXIS_1 = "AXIS_1"
    INVERSE_AXIS_0 = "INVERSE_AXIS_0"
    RADIAN = "RADIAN"


@dataclass(frozen=True)
class FitParameterDefinition:
    name: str
    unit_relation: ParameterUnitRelation

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ValueError("fit parameter name must be non-empty canonical text")
        if not isinstance(self.unit_relation, ParameterUnitRelation):
            raise TypeError("unit_relation must be ParameterUnitRelation")


@dataclass(frozen=True)
class FitAxisRequirement:
    allowed_roles: tuple[AxisRoleId, ...]

    def __post_init__(self) -> None:
        roles = tuple(self.allowed_roles)
        if not roles or any(not isinstance(role, AxisRoleId) for role in roles):
            raise ValueError("fit axis requirement needs declared AxisRoleId values")
        if len(set(roles)) != len(roles):
            raise ValueError("fit axis requirement roles must be unique")
        object.__setattr__(self, "allowed_roles", tuple(sorted(roles, key=lambda item: item.value)))


@dataclass(frozen=True)
class FitModelDefinition:
    model_id: str
    version: int
    display_name: str
    family: FitFamily
    independent_arity: int
    axis_requirements: tuple[FitAxisRequirement, ...]
    parameters: tuple[FitParameterDefinition, ...]
    minimum_observations: int
    initializer_id: str
    require_common_axis_unit: bool = False
    require_common_coordinate_frame: bool = False
    solver_contract_id: str = "scipy_least_squares_trf_v1"

    def __post_init__(self) -> None:
        for field, value in (
            ("model_id", self.model_id),
            ("display_name", self.display_name),
            ("initializer_id", self.initializer_id),
            ("solver_contract_id", self.solver_contract_id),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field} must be non-empty canonical text")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version <= 0:
            raise ValueError("model version must be a positive integer")
        if not isinstance(self.family, FitFamily):
            raise TypeError("family must be FitFamily")
        if self.independent_arity not in (1, 2):
            raise ValueError("generic fit models support one or two independent axes")
        requirements = tuple(self.axis_requirements)
        if len(requirements) != self.independent_arity or any(
            not isinstance(item, FitAxisRequirement) for item in requirements
        ):
            raise ValueError("axis_requirements must describe every independent axis")
        parameters = tuple(self.parameters)
        if not parameters or any(not isinstance(item, FitParameterDefinition) for item in parameters):
            raise ValueError("model requires FitParameterDefinition values")
        names = tuple(item.name for item in parameters)
        if len(set(names)) != len(names):
            raise ValueError("model parameter names must be unique")
        if (
            not isinstance(self.minimum_observations, int)
            or isinstance(self.minimum_observations, bool)
            or self.minimum_observations <= len(parameters)
        ):
            raise ValueError("minimum_observations must exceed the parameter count")
        if not isinstance(self.require_common_axis_unit, bool):
            raise TypeError("require_common_axis_unit must be bool")
        if not isinstance(self.require_common_coordinate_frame, bool):
            raise TypeError("require_common_coordinate_frame must be bool")
        object.__setattr__(self, "axis_requirements", requirements)
        object.__setattr__(self, "parameters", parameters)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)


@dataclass(frozen=True)
class ModelInitialization:
    seeds: tuple[tuple[float, ...], ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        seeds = tuple(tuple(float(value) for value in seed) for seed in self.seeds)
        lower = tuple(float(value) for value in self.lower_bounds)
        upper = tuple(float(value) for value in self.upper_bounds)
        if not seeds:
            raise ValueError("model initialization requires at least one seed")
        width = len(lower)
        if width == 0 or len(upper) != width or any(len(seed) != width for seed in seeds):
            raise ValueError("model initialization dimensions disagree")
        if any(not np.isfinite(value) for seed in seeds for value in seed):
            raise ValueError("model seeds must be finite")
        if any(not np.isfinite(value) for value in lower + upper):
            raise ValueError("model bounds must be finite")
        if any(not low < high for low, high in zip(lower, upper)):
            raise ValueError("every model lower bound must be below its upper bound")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)


VALUE = ParameterUnitRelation.VALUE
AXIS_0 = ParameterUnitRelation.AXIS_0
AXIS_1 = ParameterUnitRelation.AXIS_1
INVERSE_AXIS_0 = ParameterUnitRelation.INVERSE_AXIS_0
RADIAN = ParameterUnitRelation.RADIAN
SPECTRAL_OR_SCAN = FitAxisRequirement((SCAN_POINT, SPECTRAL))
SCAN = FitAxisRequirement((SCAN_POINT,))
SPACE_X = FitAxisRequirement((SPATIAL_X,))
SPACE_Y = FitAxisRequirement((SPATIAL_Y,))


_MODELS = (
    FitModelDefinition(
        "lorentzian",
        1,
        "Lorentzian",
        FitFamily.CURVE_1D,
        1,
        (SPECTRAL_OR_SCAN,),
        (
            FitParameterDefinition("center", AXIS_0),
            FitParameterDefinition("full_width", AXIS_0),
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
        ),
        5,
        "lorentzian_extrema_v1",
    ),
    FitModelDefinition(
        "gaussian_offset",
        1,
        "Gaussian with offset",
        FitFamily.CURVE_1D,
        1,
        (SPECTRAL_OR_SCAN,),
        (
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("sigma", AXIS_0),
            FitParameterDefinition("center", AXIS_0),
        ),
        5,
        "gaussian_extrema_v2",
    ),
    FitModelDefinition(
        "zeeman_double_lorentzian",
        1,
        "Zeeman double Lorentzian",
        FitFamily.CURVE_1D,
        1,
        (SPECTRAL_OR_SCAN,),
        (
            FitParameterDefinition("center", AXIS_0),
            FitParameterDefinition("full_width", AXIS_0),
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("splitting", AXIS_0),
        ),
        6,
        "zeeman_peak_and_dip_v2",
    ),
    FitModelDefinition(
        "damped_sine",
        1,
        "Damped sine",
        FitFamily.CURVE_1D,
        1,
        (SCAN,),
        (
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("frequency", INVERSE_AXIS_0),
            FitParameterDefinition("decay_time", AXIS_0),
            FitParameterDefinition("phase", RADIAN),
        ),
        7,
        "damped_sine_fft_or_span_v1",
    ),
    FitModelDefinition(
        "exponential_decay",
        1,
        "Exponential decay",
        FitFamily.CURVE_1D,
        1,
        (SCAN,),
        (
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("decay_time", AXIS_0),
        ),
        4,
        "exponential_extrema_v1",
    ),
    FitModelDefinition(
        "radial_gaussian_center",
        1,
        "Radial Gaussian center",
        FitFamily.IMAGE_2D,
        2,
        (SPACE_X, SPACE_Y),
        (
            FitParameterDefinition("amplitude", VALUE),
            FitParameterDefinition("offset", VALUE),
            FitParameterDefinition("radius", AXIS_0),
            FitParameterDefinition("center_x", AXIS_0),
            FitParameterDefinition("center_y", AXIS_1),
        ),
        6,
        "radial_centroid_v1",
        True,
        True,
    ),
)

_MODEL_BY_KEY: Mapping[tuple[str, int], FitModelDefinition] = MappingProxyType(
    {(model.model_id, model.version): model for model in _MODELS}
)


def fit_model_catalog() -> tuple[FitModelDefinition, ...]:
    return _MODELS


def fit_model_definition(model_id: str, version: int = 1) -> FitModelDefinition:
    if not isinstance(model_id, str) or not model_id or model_id.strip() != model_id:
        raise ValueError("model_id must be canonical non-empty text")
    if not isinstance(version, int) or isinstance(version, bool):
        raise TypeError("model version must be an integer")
    try:
        return _MODEL_BY_KEY[(model_id, version)]
    except KeyError as exc:
        raise ValueError(f"unknown fit model version {(model_id, version)!r}") from exc


def evaluate_fit_model(
    model: FitModelDefinition | str,
    coordinates: tuple[np.ndarray, ...],
    parameters: np.ndarray | tuple[float, ...],
    *,
    version: int = 1,
) -> np.ndarray:
    definition = fit_model_definition(model, version) if isinstance(model, str) else model
    if not isinstance(definition, FitModelDefinition):
        raise TypeError("model must be FitModelDefinition or canonical model id")
    coords = tuple(np.asarray(value, dtype=np.float64) for value in coordinates)
    if len(coords) != definition.independent_arity:
        raise ValueError("coordinate arity does not match model")
    if not coords or any(item.shape != coords[0].shape for item in coords):
        raise ValueError("model coordinate arrays must share one shape")
    params = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if params.size != len(definition.parameters):
        raise ValueError("parameter count does not match model")
    key = definition.model_id
    if key == "lorentzian":
        result = _lorentzian(coords[0], *params)
    elif key == "gaussian_offset":
        result = _gaussian_offset(coords[0], *params)
    elif key == "zeeman_double_lorentzian":
        result = _zeeman_double_lorentzian(coords[0], *params)
    elif key == "damped_sine":
        result = _damped_sine(coords[0], *params)
    elif key == "exponential_decay":
        result = _exponential_decay(coords[0], *params)
    elif key == "radial_gaussian_center":
        result = _radial_gaussian_center(coords[0], coords[1], *params)
    else:  # pragma: no cover - the catalogue is a closed union
        raise RuntimeError(f"no evaluator for {key}")
    return np.asarray(result, dtype=np.float64)


def initialize_fit_model(
    model: FitModelDefinition,
    coordinates: tuple[np.ndarray, ...],
    observations: np.ndarray,
) -> ModelInitialization:
    if not isinstance(model, FitModelDefinition):
        raise TypeError("model must be FitModelDefinition")
    coords = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in coordinates)
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    if len(coords) != model.independent_arity or any(item.shape != y.shape for item in coords):
        raise ValueError("initializer coordinate/observation shapes disagree")
    if y.size < model.minimum_observations:
        raise ValueError(
            f"{model.model_id} requires at least {model.minimum_observations} observations"
        )
    if not np.all(np.isfinite(y)) or any(not np.all(np.isfinite(item)) for item in coords):
        raise ValueError("fit initialization requires finite authoritative observations")
    key = model.model_id
    if key == "lorentzian":
        return _init_lorentzian(coords[0], y)
    if key == "gaussian_offset":
        return _init_gaussian(coords[0], y)
    if key == "zeeman_double_lorentzian":
        return _init_zeeman(coords[0], y)
    if key == "damped_sine":
        return _init_damped_sine(coords[0], y)
    if key == "exponential_decay":
        return _init_exponential(coords[0], y)
    if key == "radial_gaussian_center":
        return _init_center(coords[0], coords[1], y)
    raise RuntimeError(f"no initializer for {key}")  # pragma: no cover


def _span(values: np.ndarray) -> float:
    span = float(np.max(values) - np.min(values))
    return abs(span) if span != 0 else 1.0


def _positive_floor(scale: float) -> float:
    return max(abs(float(scale)) * 1e-12, np.finfo(np.float64).tiny)


def _lorentzian(x, center, full_width, amplitude, offset):
    half_squared = (full_width / 2.0) ** 2
    return amplitude * half_squared / ((x - center) ** 2 + half_squared) + offset


def _gaussian_offset(x, amplitude, offset, sigma, center):
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset


def _zeeman_double_lorentzian(x, center, full_width, amplitude, offset, splitting):
    return (
        _lorentzian(x - splitting / 2.0, center, full_width, amplitude, 0.0)
        + _lorentzian(x + splitting / 2.0, center, full_width, amplitude, 0.0)
        + offset
    )


def _damped_sine(x, amplitude, offset, frequency, decay_time, phase):
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase) * np.exp(
        -x / decay_time
    ) + offset


def _exponential_decay(x, amplitude, offset, decay_time):
    return amplitude * np.exp(-x / decay_time) + offset


def _radial_gaussian_center(x, y, amplitude, offset, radius, center_x, center_y):
    return amplitude * np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / radius**2) + offset


def _init_lorentzian(x: np.ndarray, y: np.ndarray) -> ModelInitialization:
    width = _span(x) / 4.0
    y_range = _span(y)
    low_y, high_y = float(np.min(y)), float(np.max(y))
    seeds = (
        (float(x[np.argmax(y)]), width, y_range, low_y),
        (float(x[np.argmin(y)]), width, -y_range, high_y),
    )
    return ModelInitialization(
        seeds,
        (float(np.min(x)), _positive_floor(width), -10 * y_range, low_y - 10 * y_range),
        (float(np.max(x)), width * 10, 10 * y_range, high_y + 10 * y_range),
    )


def _init_gaussian(x: np.ndarray, y: np.ndarray) -> ModelInitialization:
    low_y, high_y = float(np.min(y)), float(np.max(y))
    amplitude = high_y - low_y or 1.0
    sigma = _span(x) / 6.0
    seeds = (
        (amplitude, low_y, sigma, float(x[np.argmax(y)])),
        (-amplitude, high_y, sigma, float(x[np.argmin(y)])),
    )
    y_range = _span(y)
    return ModelInitialization(
        seeds,
        (-10 * y_range, low_y - 10 * y_range, _positive_floor(sigma), float(np.min(x))),
        (10 * y_range, high_y + 10 * y_range, sigma * 20, float(np.max(x))),
    )


def _init_zeeman(x: np.ndarray, y: np.ndarray) -> ModelInitialization:
    from scipy.signal import find_peaks

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    x_span = _span(x)
    y_range = _span(y)
    step = float(np.median(np.abs(np.diff(np.sort(np.unique(x)))))) if np.unique(x).size > 1 else 1.0
    step = step or 1.0
    seeds: list[tuple[float, ...]] = []
    widths_seen: list[float] = []
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
            widths_seen.append(width)
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
        widths_seen.append(width)
    width = max(widths_seen) if widths_seen else x_span / 8.0
    low_y, high_y = float(np.min(y)), float(np.max(y))
    return ModelInitialization(
        tuple(seeds),
        (float(np.min(x)), _positive_floor(width), -10 * y_range, low_y - 10 * y_range, 0.0),
        (float(np.max(x)), width * 10, 10 * y_range, high_y + 10 * y_range, 2 * x_span),
    )


def _init_damped_sine(x: np.ndarray, y: np.ndarray) -> ModelInitialization:
    amplitude = _span(y) / 2.0
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
    seeds = (
        (amplitude, offset, frequency, x_span, np.pi / 2.0),
        (-amplitude, offset, frequency, x_span, np.pi / 2.0),
    )
    y_range = _span(y)
    return ModelInitialization(
        seeds,
        (
            -5 * abs(amplitude or 1.0),
            offset - 2 * y_range,
            _positive_floor(frequency),
            _positive_floor(x_span),
            -np.pi,
        ),
        (
            5 * abs(amplitude or 1.0),
            offset + 2 * y_range,
            frequency * 10,
            x_span * 20,
            2 * np.pi,
        ),
    )


def _init_exponential(x: np.ndarray, y: np.ndarray) -> ModelInitialization:
    y_range = _span(y)
    offset = float(np.mean(y))
    decay_time = _span(x) / 2.0
    return ModelInitialization(
        ((y_range, offset, decay_time), (-y_range, offset, decay_time)),
        (-4 * y_range, offset - y_range, _positive_floor(decay_time)),
        (4 * y_range, offset + y_range, decay_time * 10),
    )


def _init_center(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> ModelInitialization:
    offset = float(np.median(z))
    z_range = _span(z)
    seeds: list[tuple[float, ...]] = []
    radii: list[float] = []
    for sign in (1.0, -1.0):
        weights = np.clip(sign * (z - offset), 0.0, None)
        total = float(np.sum(weights))
        if total <= 0:
            continue
        center_x = float(np.sum(weights * x) / total)
        center_y = float(np.sum(weights * y) / total)
        radius = float(
            np.sqrt(
                np.sum(weights * ((x - center_x) ** 2 + (y - center_y) ** 2))
                / total
            )
        )
        radius = radius or 1.0
        amplitude = (
            float(np.max(z) - offset)
            if sign > 0
            else float(np.min(z) - offset)
        )
        seeds.append((amplitude, offset, radius, center_x, center_y))
        radii.append(radius)
    if not seeds:
        raise ValueError("radial center fit requires non-flat contrast")
    radius_scale = max(radii)
    return ModelInitialization(
        tuple(seeds),
        (
            -5 * z_range,
            float(np.min(z)) - z_range,
            _positive_floor(radius_scale),
            float(np.min(x)),
            float(np.min(y)),
        ),
        (
            5 * z_range,
            float(np.max(z)) + z_range,
            radius_scale * 20,
            float(np.max(x)),
            float(np.max(y)),
        ),
    )


__all__ = [
    "FitFamily",
    "FitAxisRequirement",
    "FitModelDefinition",
    "FitParameterDefinition",
    "ModelInitialization",
    "ParameterUnitRelation",
    "evaluate_fit_model",
    "fit_model_catalog",
    "fit_model_definition",
    "initialize_fit_model",
]
