"""Headless fit catalogue and solver used by every presentation backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks

from ._validation import finite_real as _finite_real
from ._validation import integer, text as _text
from .kinds import AxisRef


ArrayTuple = tuple[np.ndarray, ...]
Evaluator = Callable[..., np.ndarray]
Initializer = Callable[[ArrayTuple, np.ndarray], Sequence[float]]
CandidateInitializer = Callable[
    [ArrayTuple, np.ndarray], Sequence[Sequence[float]]
]
BoundsInitializer = Callable[
    [ArrayTuple, np.ndarray],
    Mapping[str, tuple[float | None, float | None]],
]


def _integer(value: object, name: str) -> int:
    result = integer(value, name)
    assert result is not None
    return result


class FitCancelled(RuntimeError):
    pass


class FitDeadlineExceeded(TimeoutError):
    pass


class ParameterDomain(str, Enum):
    REAL = "real"
    POSITIVE = "positive"
    NONNEGATIVE = "nonnegative"
    PHASE_RADIANS = "phase_radians"


class UnitRelation(str, Enum):
    VALUE = "value"
    AXIS_0 = "axis_0"
    AXIS_1 = "axis_1"
    INVERSE_AXIS_0 = "inverse_axis_0"
    RADIAN = "radian"


class FitTarget(str, Enum):
    """Semantic projection a model is authored to fit."""

    SERIES = "series"
    HISTOGRAM = "histogram"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class FitParameterDisplay:
    """One fitted parameter in the units currently painted on the plot."""

    name: str
    label: str
    value: float
    standard_error: float | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        for field_name in ("name", "label"):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), f"fit parameter {field_name}"),
            )
        value = _finite_real(self.value, "fit parameter value")
        object.__setattr__(self, "value", value)
        if self.standard_error is not None:
            error = _finite_real(
                self.standard_error,
                "fit parameter standard_error",
            )
            if error < 0.0:
                raise ValueError("fit parameter standard_error must be non-negative")
            object.__setattr__(self, "standard_error", error)
        object.__setattr__(
            self,
            "unit",
            _text(self.unit, "fit parameter unit", allow_empty=True),
        )


@dataclass(frozen=True, slots=True)
class FitParameterSpec:
    """One solver parameter and its renderer-facing display semantics.

    ``affine_point`` distinguishes absolute positions from spans or amplitudes.
    It applies only to parameter values; standard errors are always converted
    as differences and therefore never consume a unit offset.
    ``solver_unit_relation`` records the canonical unit used by the evaluator
    when it differs from the parameter's painted axis relation.
    """

    name: str
    unit_relation: UnitRelation
    domain: ParameterDomain = ParameterDomain.REAL
    display_label: str | None = None
    affine_point: bool = False
    solver_unit_relation: UnitRelation | None = None

    def __post_init__(self) -> None:
        name = _text(self.name, "fit parameter name")
        if not isinstance(self.unit_relation, UnitRelation):
            raise TypeError("unit_relation must be UnitRelation")
        if not isinstance(self.domain, ParameterDomain):
            raise TypeError("domain must be ParameterDomain")
        display_label = self.display_label
        if display_label is not None:
            display_label = _text(display_label, "fit parameter display_label")
        if not isinstance(self.affine_point, bool):
            raise TypeError("affine_point must be bool")
        solver_unit_relation = self.solver_unit_relation
        if solver_unit_relation is None:
            solver_unit_relation = self.unit_relation
        if not isinstance(solver_unit_relation, UnitRelation):
            raise TypeError("solver_unit_relation must be UnitRelation or None")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "display_label", display_label)
        object.__setattr__(self, "solver_unit_relation", solver_unit_relation)

    @property
    def bounds(self) -> tuple[float, float]:
        if self.domain is ParameterDomain.REAL:
            return -np.inf, np.inf
        if self.domain is ParameterDomain.POSITIVE:
            return float(np.nextafter(0.0, np.inf)), np.inf
        if self.domain is ParameterDomain.NONNEGATIVE:
            return 0.0, np.inf
        if self.domain is ParameterDomain.PHASE_RADIANS:
            return -np.pi, float(np.nextafter(np.pi, -np.inf))
        raise RuntimeError(self.domain)


@dataclass(frozen=True, slots=True)
class FitComponentSpec:
    """One named additive component of the presented model family."""

    component_id: str
    evaluator: Evaluator

    def __post_init__(self) -> None:
        component_id = _text(self.component_id, "fit component id")
        if not callable(self.evaluator):
            raise TypeError("fit component evaluator must be callable")
        object.__setattr__(self, "component_id", component_id)


@dataclass(frozen=True, slots=True)
class FitRadialGlyphSpec:
    """Parameter references needed to paint a radial center and radius."""

    center_parameters: tuple[str, str]
    radius_parameter: str

    def __post_init__(self) -> None:
        center_parameters = tuple(
            _text(name, "radial glyph center parameter")
            for name in self.center_parameters
        )
        if (
            len(center_parameters) != 2
            or center_parameters[0] == center_parameters[1]
        ):
            raise ValueError("radial glyph requires two distinct center parameters")
        radius_parameter = _text(
            self.radius_parameter,
            "radial glyph radius parameter",
        )
        object.__setattr__(self, "center_parameters", center_parameters)
        object.__setattr__(self, "radius_parameter", radius_parameter)


@dataclass(frozen=True, slots=True)
class FitPresentationSpec:
    """Backend-neutral metadata for additive curves or one radial glyph."""

    components: tuple[FitComponentSpec, ...] = ()
    crossover_components: tuple[str, str] | None = None
    radial_glyph: FitRadialGlyphSpec | None = None

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if any(not isinstance(item, FitComponentSpec) for item in components):
            raise TypeError("fit presentation components must be FitComponentSpec values")
        component_ids = tuple(item.component_id for item in components)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("fit presentation component ids must be unique")
        crossover = self.crossover_components
        if crossover is not None:
            crossover = tuple(
                _text(name, "fit crossover component id") for name in crossover
            )
            if (
                len(crossover) != 2
                or crossover[0] == crossover[1]
            ):
                raise ValueError("fit crossover requires two distinct component ids")
            unknown = set(crossover) - set(component_ids)
            if unknown:
                raise ValueError(
                    f"fit crossover names unknown components: {sorted(unknown)}"
                )
        if self.radial_glyph is not None and not isinstance(
            self.radial_glyph, FitRadialGlyphSpec
        ):
            raise TypeError("radial_glyph must be FitRadialGlyphSpec or None")
        if self.radial_glyph is not None and (
            components or crossover is not None
        ):
            raise ValueError("radial glyph presentation cannot also define components")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "crossover_components", crossover)


@dataclass(frozen=True, slots=True)
class FitModelSpec:
    """One fit model, including optional MathText presentation metadata."""

    model_id: str
    display_name: str
    independent_arity: int
    parameters: tuple[FitParameterSpec, ...]
    evaluator: Evaluator
    initializer: Initializer
    targets: tuple[FitTarget, ...]
    formula: str | None = None
    candidate_initializer: CandidateInitializer | None = None
    bounds_initializer: BoundsInitializer | None = None
    presentation: FitPresentationSpec = field(default_factory=FitPresentationSpec)
    coordinate_relations: tuple[UnitRelation, ...] | None = None
    default_for: tuple[FitTarget, ...] = ()
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        model_id = _text(self.model_id, "fit model id")
        display_name = _text(self.display_name, "fit model display name")
        independent_arity = _integer(
            self.independent_arity,
            "fit model independent_arity",
        )
        if independent_arity not in (1, 2):
            raise ValueError("fit models support one or two independent axes")
        parameters = tuple(self.parameters)
        if not parameters or any(not isinstance(item, FitParameterSpec) for item in parameters):
            raise ValueError("fit model requires parameter specifications")
        names = tuple(item.name for item in parameters)
        if len(names) != len(set(names)):
            raise ValueError("fit parameter names must be unique")
        if not callable(self.evaluator) or not callable(self.initializer):
            raise TypeError("evaluator and initializer must be callable")
        targets = tuple(self.targets)
        if (
            not targets
            or any(not isinstance(target, FitTarget) for target in targets)
            or len(targets) != len(set(targets))
        ):
            raise ValueError("fit model targets must be unique FitTarget values")
        defaults = tuple(self.default_for)
        if (
            any(not isinstance(target, FitTarget) for target in defaults)
            or len(defaults) != len(set(defaults))
            or not set(defaults).issubset(targets)
        ):
            raise ValueError("default_for must be unique members of targets")
        capabilities = frozenset(str(value).strip() for value in self.capabilities)
        if any(not value for value in capabilities):
            raise ValueError("fit capabilities must be non-empty text")
        if self.candidate_initializer is not None and not callable(
            self.candidate_initializer
        ):
            raise TypeError("candidate_initializer must be callable or None")
        if self.bounds_initializer is not None and not callable(
            self.bounds_initializer
        ):
            raise TypeError("bounds_initializer must be callable or None")
        if not isinstance(self.presentation, FitPresentationSpec):
            raise TypeError("presentation must be FitPresentationSpec")
        coordinate_relations = self.coordinate_relations
        if coordinate_relations is None:
            coordinate_relations = (
                (UnitRelation.AXIS_0,)
                if self.independent_arity == 1
                else (UnitRelation.AXIS_0, UnitRelation.AXIS_1)
            )
        else:
            coordinate_relations = tuple(coordinate_relations)
        if len(coordinate_relations) != self.independent_arity or any(
            not isinstance(relation, UnitRelation)
            or relation not in {UnitRelation.AXIS_0, UnitRelation.AXIS_1}
            for relation in coordinate_relations
        ):
            raise ValueError(
                "coordinate_relations must identify one plot axis per independent axis"
            )
        radial_glyph = self.presentation.radial_glyph
        if radial_glyph is not None:
            if self.independent_arity != 2:
                raise ValueError("radial glyph presentation requires a two-axis model")
            referenced = {
                *radial_glyph.center_parameters,
                radial_glyph.radius_parameter,
            }
            unknown = referenced - set(names)
            if unknown:
                raise ValueError(
                    f"radial glyph names unknown parameters: {sorted(unknown)}"
                )
        formula = self.formula
        if formula is not None:
            formula = _text(formula, "fit model formula")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "independent_arity", independent_arity)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "coordinate_relations", coordinate_relations)
        object.__setattr__(self, "default_for", defaults)
        object.__setattr__(self, "capabilities", capabilities)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    def parameter_index(self, name: str) -> int:
        parameter_name = _text(name, "fit parameter name")
        try:
            return self.parameter_names.index(parameter_name)
        except ValueError as error:
            raise ValueError(f"unknown fit parameter: {name!r}") from error

    def evaluate(self, coordinates: ArrayTuple, values: Sequence[float]) -> np.ndarray:
        coordinates = _coordinate_arrays(coordinates, self.independent_arity)
        parameters = np.asarray(values, dtype=np.float64).reshape(-1)
        if parameters.size != len(self.parameters):
            raise ValueError("fit parameter count does not match model")
        return np.asarray(self.evaluator(*coordinates, *parameters), dtype=np.float64)

    def evaluate_component(
        self,
        component_id: str,
        coordinates: ArrayTuple,
        values: Sequence[float],
    ) -> np.ndarray:
        requested = _text(component_id, "fit component id")
        component = next(
            (
                item
                for item in self.presentation.components
                if item.component_id == requested
            ),
            None,
        )
        if component is None:
            raise ValueError(f"unknown fit component: {component_id!r}")
        coordinates = _coordinate_arrays(coordinates, self.independent_arity)
        parameters = np.asarray(values, dtype=np.float64).reshape(-1)
        if parameters.size != len(self.parameters):
            raise ValueError("fit parameter count does not match model")
        return np.asarray(
            component.evaluator(*coordinates, *parameters),
            dtype=np.float64,
        )


class FitModelRegistry:
    def __init__(self, models: Sequence[FitModelSpec] = ()) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, FitModelSpec] = {}
        for model in models:
            self.register(model)

    def register(self, model: FitModelSpec, *, replace: bool = False) -> None:
        if not isinstance(model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        with self._lock:
            if model.model_id in self._models and not replace:
                raise ValueError(f"fit model already registered: {model.model_id}")
            for target in model.default_for:
                conflict = next(
                    (
                        registered.model_id
                        for registered in self._models.values()
                        if registered.model_id != model.model_id
                        and target in registered.default_for
                    ),
                    None,
                )
                if conflict is not None:
                    raise ValueError(
                        f"fit target {target.value!r} already has default model "
                        f"{conflict!r}"
                    )
            self._models[model.model_id] = model

    def get(self, model_id: str) -> FitModelSpec:
        selected = _text(model_id, "fit model id")
        with self._lock:
            try:
                return self._models[selected]
            except KeyError as error:
                raise ValueError(f"unknown fit model: {model_id!r}") from error

    def models(self) -> tuple[FitModelSpec, ...]:
        with self._lock:
            return tuple(self._models.values())

    def models_for(self, target: FitTarget) -> tuple[FitModelSpec, ...]:
        """Return compatible models with the target's authored default first."""

        if not isinstance(target, FitTarget):
            raise TypeError("target must be FitTarget")
        with self._lock:
            models = tuple(
                model for model in self._models.values() if target in model.targets
            )
        return tuple(
            sorted(models, key=lambda model: target not in model.default_for)
        )


@dataclass(frozen=True, slots=True)
class FitOptions:
    loss: str = "linear"
    max_nfev: int = 5000
    deadline_seconds: float | None = None

    def __post_init__(self) -> None:
        loss = _text(self.loss, "fit loss")
        if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("unsupported scipy least-squares loss")
        max_nfev = _integer(self.max_nfev, "max_nfev")
        if max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "max_nfev", max_nfev)
        if self.deadline_seconds is not None:
            deadline = _finite_real(self.deadline_seconds, "deadline_seconds")
            if deadline <= 0:
                raise ValueError("deadline_seconds must be positive")
            object.__setattr__(self, "deadline_seconds", deadline)


def _readonly(array: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(array)
    backing: object = source
    while isinstance(getattr(backing, "base", None), np.ndarray):
        backing = backing.base
    if not source.flags.writeable and isinstance(getattr(backing, "base", None), bytes):
        return source
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)


@dataclass(frozen=True, slots=True, eq=False)
class RegularImageFitInput:
    """A Cartesian image without materialized per-pixel coordinate grids.

    Rows are identified by ``y_coordinates`` and columns by ``x_coordinates``;
    consequently ``observations.shape`` must be ``(len(y), len(x))``.  The
    image storage is retained as a read-only view rather than copied.  Callers
    must therefore not mutate its backing storage while a fit is running.
    """

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    observations: np.ndarray
    valid_mask: np.ndarray | None = None
    selected_indices: np.ndarray | None = None
    minimum_observation: float | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.x_coordinates, dtype=np.float64).reshape(-1)
        y = np.asarray(self.y_coordinates, dtype=np.float64).reshape(-1)
        if x.size == 0 or y.size == 0:
            raise ValueError("regular image coordinates cannot be empty")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("regular image coordinates must be finite")
        for name, values in (("x", x), ("y", y)):
            differences = np.diff(values)
            if differences.size and not (
                np.all(differences > 0.0) or np.all(differences < 0.0)
            ):
                raise ValueError(
                    f"regular image {name} coordinates must be strictly monotonic"
                )

        image = np.asarray(self.observations)
        if image.ndim != 2 or image.shape != (y.size, x.size):
            raise ValueError(
                "regular image observations must have shape (len(y), len(x))"
            )
        if image.dtype.kind not in "fiu":
            raise TypeError("regular image observations must be real numeric values")
        image = image.view()
        image.setflags(write=False)

        valid = self.valid_mask
        if valid is not None:
            valid = np.asarray(valid, dtype=np.bool_)
            if valid.shape != image.shape:
                raise ValueError("regular image valid_mask must match observations")
            valid = valid.view()
            valid.setflags(write=False)

        indices = self.selected_indices
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.size != image.size:
                raise ValueError(
                    "regular image selected_indices must identify every image cell"
                )
            indices = indices.reshape(image.shape).view()
            indices.setflags(write=False)

        minimum = self.minimum_observation
        if minimum is not None:
            minimum = _finite_real(minimum, "regular image minimum_observation")

        object.__setattr__(self, "x_coordinates", _readonly(x))
        object.__setattr__(self, "y_coordinates", _readonly(y))
        object.__setattr__(self, "observations", image)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "minimum_observation", minimum)


@dataclass(frozen=True, slots=True)
class FitResult:
    """Immutable fit output with optional per-observation diagnostics.

    Ordinary vector fits retain their compact fitted/residual/index arrays.
    Regular-raster fits deliberately omit those arrays: the model parameters,
    covariance and observation count are sufficient for their overlay and
    diagnostics, while materializing three full image-sized arrays would
    defeat the regular-raster memory boundary.
    """

    model: FitModelSpec
    parameter_values: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    fitted_values: np.ndarray | None
    residuals: np.ndarray | None
    selected_indices: np.ndarray | None
    observation_count: int
    data_revision: int
    success: bool
    message: str
    reduced_chi_square: float
    covariance_valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        parameters = np.asarray(self.parameter_values, dtype=np.float64).reshape(-1)
        errors = np.asarray(self.standard_errors, dtype=np.float64).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        count = len(self.model.parameters)
        if parameters.shape != (count,) or errors.shape != (count,):
            raise ValueError("fit parameter/error shapes disagree with model")
        if covariance.shape != (count, count):
            raise ValueError("fit covariance shape disagrees with model")
        if not isinstance(self.covariance_valid, (bool, np.bool_)):
            raise TypeError("covariance_valid must be bool")
        covariance_valid = bool(self.covariance_valid)
        if covariance_valid:
            if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(errors)):
                raise ValueError("valid fit covariance and standard errors must be finite")
            if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15):
                raise ValueError("valid fit covariance must be symmetric")
            diagonal = np.diag(covariance)
            if np.any(diagonal < 0.0) or np.any(errors < 0.0):
                raise ValueError("valid fit variances and standard errors cannot be negative")
            if not np.allclose(errors**2, diagonal, rtol=1e-10, atol=1e-15):
                raise ValueError("standard errors must agree with fit covariance")
        else:
            covariance = np.full((count, count), np.nan, dtype=np.float64)
            errors = np.full(count, np.nan, dtype=np.float64)
        observation_count = _integer(self.observation_count, "observation_count")
        if observation_count <= 0:
            raise ValueError("observation_count must be positive")
        raw_diagnostics = (
            self.fitted_values,
            self.residuals,
            self.selected_indices,
        )
        if all(value is None for value in raw_diagnostics):
            fitted = residuals = indices = None
        elif any(value is None for value in raw_diagnostics):
            raise ValueError(
                "fitted values, residuals and selected indices must be all present or all absent"
            )
        else:
            fitted = np.asarray(self.fitted_values, dtype=np.float64).reshape(-1)
            residuals = np.asarray(self.residuals, dtype=np.float64).reshape(-1)
            if fitted.shape != residuals.shape:
                raise ValueError("fitted values and residuals must have equal shape")
            indices = np.asarray(self.selected_indices, dtype=np.int64).reshape(-1)
            if indices.shape != fitted.shape:
                raise ValueError(
                    "selected indices must identify every fitted observation"
                )
            if fitted.size != observation_count:
                raise ValueError(
                    "observation_count must match the compact diagnostic arrays"
                )
        data_revision = _integer(self.data_revision, "data_revision")
        if data_revision < 0:
            raise ValueError("data_revision must be non-negative")
        object.__setattr__(self, "parameter_values", _readonly(parameters))
        object.__setattr__(self, "standard_errors", _readonly(errors))
        object.__setattr__(self, "covariance", _readonly(covariance))
        object.__setattr__(
            self,
            "fitted_values",
            None if fitted is None else _readonly(fitted),
        )
        object.__setattr__(
            self,
            "residuals",
            None if residuals is None else _readonly(residuals),
        )
        object.__setattr__(
            self,
            "selected_indices",
            None if indices is None else _readonly(indices),
        )
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "data_revision", data_revision)
        object.__setattr__(
            self,
            "message",
            _text(self.message, "fit result message", allow_empty=True),
        )
        if not isinstance(self.success, (bool, np.bool_)):
            raise TypeError("fit result success must be bool")
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(
            self,
            "reduced_chi_square",
            _finite_real(self.reduced_chi_square, "reduced_chi_square"),
        )
        object.__setattr__(self, "covariance_valid", covariance_valid)

    @property
    def parameters(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.model.parameter_names, map(float, self.parameter_values), strict=True))
        )

    @property
    def errors(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.model.parameter_names, map(float, self.standard_errors), strict=True))
        )


@dataclass(frozen=True, slots=True, eq=False)
class FacetFitBatchResult:
    """Ordered results for one exact FacetGrid source revision.

    Cell order is the already-painted :class:`FacetData` order.  A cell owns
    either its ordinary ``FitResult`` or one diagnostic string; no second
    batch-fit result hierarchy or grid-specific solver model is introduced.
    Parameter arrays are derived from those ordinary results on demand.
    """

    facet: AxisRef
    facet_values: tuple[Any, ...]
    model: FitModelSpec
    results: tuple[FitResult | None, ...]
    errors: tuple[str | None, ...]
    source_revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.facet, AxisRef):
            raise TypeError("facet must be AxisRef")
        if not isinstance(self.model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        values = tuple(self.facet_values)
        results = tuple(self.results)
        errors = tuple(self.errors)
        if not values:
            raise ValueError("facet fit batch cannot be empty")
        if len(results) != len(values) or len(errors) != len(values):
            raise ValueError("facet values, results and errors must have equal length")
        revision = _integer(self.source_revision, "facet fit source_revision")
        if revision < 0:
            raise ValueError("facet fit source_revision must be non-negative")
        normalized_errors: list[str | None] = []
        for index, (result, error) in enumerate(zip(results, errors, strict=True)):
            if result is not None:
                if not isinstance(result, FitResult):
                    raise TypeError("facet fit results must contain FitResult or None")
                if result.model != self.model:
                    raise ValueError("facet fit cell model differs from batch model")
                if result.data_revision != revision:
                    raise ValueError("facet fit cell revision differs from source revision")
                if error is not None:
                    raise ValueError(
                        f"facet fit cell {index} cannot contain both a result and error"
                    )
                normalized_errors.append(None)
                continue
            if not isinstance(error, str) or not error.strip():
                raise ValueError(
                    f"facet fit cell {index} without a result requires an error"
                )
            normalized_errors.append(error.strip())
        object.__setattr__(self, "facet_values", values)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "errors", tuple(normalized_errors))
        object.__setattr__(self, "source_revision", revision)

    @property
    def success(self) -> np.ndarray:
        return _readonly(np.asarray(
            tuple(result is not None and result.success for result in self.results),
            dtype=np.bool_,
        ))

    @property
    def parameter_values(self) -> Mapping[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for parameter_index, name in enumerate(self.model.parameter_names):
            arrays[name] = _readonly(np.asarray(
                tuple(
                    np.nan
                    if result is None or not result.success
                    else float(result.parameter_values[parameter_index])
                    for result in self.results
                ),
                dtype=np.float64,
            ))
        return MappingProxyType(arrays)


class FitEngine:
    def __init__(self, registry: FitModelRegistry | None = None) -> None:
        self.registry = registry or default_fit_registry()

    def fit(
        self,
        model: str | FitModelSpec,
        coordinates: Sequence[np.ndarray] | RegularImageFitInput,
        observations: np.ndarray | None = None,
        *,
        selected_indices: np.ndarray | None = None,
        data_revision: int = 0,
        initial: Mapping[str, float] | Sequence[float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: FitOptions | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> FitResult:
        spec = self.registry.get(model) if isinstance(model, str) else model
        if not isinstance(spec, FitModelSpec):
            raise TypeError("model must be a registered id or FitModelSpec")
        opts = options or FitOptions()
        if isinstance(coordinates, RegularImageFitInput):
            if observations is not None:
                raise TypeError(
                    "observations belong inside RegularImageFitInput and must not "
                    "also be passed separately"
                )
            if selected_indices is not None:
                raise TypeError(
                    "regular-image selected_indices belong inside "
                    "RegularImageFitInput"
                )
            if "regular_image_radial" not in spec.capabilities:
                raise ValueError(
                    "this model does not declare regular-image radial capability"
                )
            from ._fit_radial import fit_regular_image_radial

            return fit_regular_image_radial(
                spec,
                coordinates,
                data_revision=data_revision,
                initial=initial,
                bounds=bounds,
                options=opts,
                cancelled=cancelled,
            )
        if observations is None:
            raise TypeError("observations are required for coordinate-array fitting")

        coords = _coordinate_arrays(tuple(coordinates), spec.independent_arity)
        values = np.asarray(observations, dtype=np.float64).reshape(-1)
        if any(item.shape != values.shape for item in coords):
            raise ValueError("coordinates and observations must have equal flattened shape")
        if selected_indices is None:
            indices = np.arange(values.size, dtype=np.int64)
        else:
            indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
            if indices.size != values.size:
                raise ValueError("selected_indices must match observations")
        finite = np.isfinite(values)
        for coordinate in coords:
            finite &= np.isfinite(coordinate)
        if not bool(np.all(finite)):
            coords = tuple(item[finite] for item in coords)
            values = values[finite]
            indices = indices[finite]
        if values.size <= len(spec.parameters):
            raise ValueError("fit requires more finite observations than parameters")
        default_bounds = (
            spec.bounds_initializer(coords, values)
            if spec.bounds_initializer is not None
            else None
        )
        lower, upper = _solver_bounds(spec, default_bounds, bounds)
        seeds = _initial_candidates(spec, coords, values, initial)
        low_inside = np.nextafter(lower, upper)
        high_inside = np.nextafter(upper, lower)
        seeds = tuple(np.minimum(np.maximum(seed, low_inside), high_inside) for seed in seeds)
        start = time.monotonic()
        invalid_residual = np.finfo(np.float64).max ** 0.25

        def check() -> None:
            if cancelled is not None and cancelled():
                raise FitCancelled("fit cancelled")
            if opts.deadline_seconds is not None and time.monotonic() - start > opts.deadline_seconds:
                raise FitDeadlineExceeded("fit deadline exceeded")

        def residual(parameters: np.ndarray) -> np.ndarray:
            check()
            predicted = spec.evaluate(coords, parameters).reshape(-1)
            if predicted.shape != values.shape or not np.all(np.isfinite(predicted)):
                return np.full(values.shape, invalid_residual)
            return predicted - values

        successful: list[tuple[float, Any, np.ndarray, np.ndarray]] = []
        unsuccessful: list[tuple[float, Any, np.ndarray, np.ndarray]] = []
        last_error: Exception | None = None
        for seed in seeds:
            check()
            try:
                candidate = least_squares(
                    residual,
                    seed,
                    bounds=(lower, upper),
                    loss=opts.loss,
                    max_nfev=opts.max_nfev,
                    x_scale="jac",
                )
                check()
                solver_residual = np.asarray(candidate.fun, dtype=np.float64).reshape(-1)
                if (
                    solver_residual.shape != values.shape
                    or not np.all(np.isfinite(solver_residual))
                    or bool(np.all(solver_residual == invalid_residual))
                ):
                    continue
                fitted_candidate = values + solver_residual
                if not np.all(np.isfinite(fitted_candidate)):
                    continue
                residual_candidate = -solver_residual
                rss = float(np.dot(residual_candidate, residual_candidate))
                if not math.isfinite(rss):
                    continue
                (successful if candidate.success else unsuccessful).append(
                    (rss, candidate, fitted_candidate, residual_candidate)
                )
            except (FitCancelled, FitDeadlineExceeded):
                raise
            except (ValueError, RuntimeError, FloatingPointError) as error:
                last_error = error
        candidates = successful or unsuccessful
        if not candidates:
            if last_error is not None:
                raise last_error
            raise RuntimeError("least-squares failed for every initializer")
        _rss, solved, fitted, residuals = min(candidates, key=lambda item: item[0])
        degrees = max(values.size - solved.x.size, 1)
        reduced = float(np.dot(residuals, residuals) / degrees)
        covariance, covariance_valid = _covariance(solved.jac, reduced)
        errors = (
            np.sqrt(np.maximum(np.diag(covariance), 0.0))
            if covariance_valid
            else np.full(solved.x.size, np.nan, dtype=np.float64)
        )
        return FitResult(
            spec,
            solved.x,
            errors,
            covariance,
            fitted,
            residuals,
            indices,
            values.size,
            data_revision,
            bool(solved.success),
            solved.message,
            reduced,
            covariance_valid=covariance_valid,
        )


@dataclass(frozen=True, slots=True)
class _RegularImageSummary:
    minimum: float
    maximum: float
    scale: float
    count: int
    all_valid: bool
    normalized_sum: float
    normalized_square_sum: float


_REGULAR_IMAGE_STRIPE_ROWS = 64
_REGULAR_IMAGE_SAMPLE_LIMIT = 257
_REGULAR_IMAGE_MIN_RELATIVE_CONTRAST = 0.05
_REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS = 50
_REGULAR_IMAGE_FTOL = 1e-10
_REGULAR_IMAGE_GTOL = 1e-8


def _fit_regular_radial_image(
    model: FitModelSpec,
    data: RegularImageFitInput,
    *,
    data_revision: int,
    initial: Mapping[str, float] | Sequence[float] | None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    options: FitOptions,
    cancelled: Callable[[], bool] | None,
) -> FitResult:
    """Fit the built-in radial Gaussian without expanding coordinate grids."""

    started = time.monotonic()

    def check() -> None:
        if cancelled is not None and cancelled():
            raise FitCancelled("fit cancelled")
        deadline = options.deadline_seconds
        if deadline is not None and time.monotonic() - started > deadline:
            raise FitDeadlineExceeded("fit deadline exceeded")

    summary = _regular_image_summary(data, check)
    if summary.count <= len(model.parameters):
        raise ValueError("fit requires more finite observations than parameters")
    optimization_data = _regular_image_optimization_input(data, check)
    optimization_summary = (
        summary
        if optimization_data is data
        else _regular_image_summary(optimization_data, check)
    )
    seed_coordinates, seed_values = _regular_image_sample(optimization_data, check)
    seeds = _initial_candidates(model, seed_coordinates, seed_values, initial)
    if initial is None:
        strongest = max(abs(float(seed[0])) for seed in seeds)
        seeds = tuple(
            seed
            for seed in seeds
            if abs(float(seed[0])) >= strongest * _REGULAR_IMAGE_MIN_RELATIVE_CONTRAST
        )
    default_bounds = (
        model.bounds_initializer(seed_coordinates, seed_values)
        if model.bounds_initializer is not None
        else None
    )
    lower, upper = _solver_bounds(model, default_bounds, bounds)
    lower_inside, upper_inside = np.nextafter(lower, upper), np.nextafter(upper, lower)
    seeds = tuple(np.clip(seed, lower_inside, upper_inside) for seed in seeds)

    value_range = _value_range(seed_values)
    x_span, y_span = _span(data.x_coordinates), _span(data.y_coordinates)
    natural_scale = np.asarray(
        (value_range, value_range, max(x_span, y_span) / 4.0, x_span / 4.0, y_span / 4.0)
    )
    optimization_scale = (
        optimization_summary.scale if options.loss == "linear" else 1.0
    )

    def optimization_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        check()
        if optimization_summary.all_valid and options.loss == "linear":
            return _regular_image_linear_objective(
                optimization_data, optimization_summary, parameters
            )
        cost, gradient, _rss, _information = _regular_image_striped_objective(
            optimization_data,
            parameters,
            optimization_scale,
            options.loss,
            False,
            check,
        )
        return cost, gradient

    full_scale = summary.scale if options.loss == "linear" else 1.0

    def full_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        check()
        if summary.all_valid and options.loss == "linear":
            return _regular_image_linear_objective(data, summary, parameters)
        cost, gradient, _rss, _information = _regular_image_striped_objective(
            data, parameters, full_scale, options.loss, False, check
        )
        return cost, gradient

    candidates: list[tuple[float, Any, np.ndarray]] = []
    last_error: Exception | None = None
    solver_options = {
        "ftol": _REGULAR_IMAGE_FTOL,
        "gtol": _REGULAR_IMAGE_GTOL,
        "maxfun": options.max_nfev,
        "maxiter": options.max_nfev,
        "maxls": _REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS,
    }
    for seed in seeds:
        check()
        parameter_scale = np.maximum(np.abs(seed), natural_scale)

        def scaled_objective(scaled: np.ndarray) -> tuple[float, np.ndarray]:
            cost, gradient = optimization_objective(scaled * parameter_scale)
            return cost, gradient * parameter_scale

        try:
            solved = minimize(
                scaled_objective,
                seed / parameter_scale,
                method="L-BFGS-B",
                jac=True,
                bounds=tuple(zip(lower / parameter_scale, upper / parameter_scale)),
                options=solver_options,
            )
            check()
            parameters = np.asarray(solved.x) * parameter_scale
            cost, _gradient = optimization_objective(parameters)
            if math.isfinite(cost) and np.all(np.isfinite(parameters)):
                candidates.append((cost, solved, parameters))
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except (ValueError, RuntimeError, FloatingPointError) as error:
            last_error = error
    if not candidates:
        if last_error is not None:
            raise last_error
        raise RuntimeError("regular-image optimizer failed for every initializer")

    _cost, solved, parameters = min(
        candidates, key=lambda item: (not bool(item[1].success), item[0])
    )
    if optimization_data is not data:
        parameter_scale = np.maximum(np.abs(parameters), natural_scale)
        initial_scaled = parameters / parameter_scale
        _full_cost, full_gradient = full_objective(parameters)
        if np.max(np.abs(full_gradient * parameter_scale)) > _REGULAR_IMAGE_GTOL:

            def scaled_full_objective(
                scaled: np.ndarray,
            ) -> tuple[float, np.ndarray]:
                cost, gradient = full_objective(scaled * parameter_scale)
                return cost, gradient * parameter_scale

            refined = minimize(
                scaled_full_objective,
                initial_scaled,
                method="L-BFGS-B",
                jac=True,
                bounds=tuple(
                    zip(lower / parameter_scale, upper / parameter_scale)
                ),
                options=solver_options,
            )
            check()
            refined_parameters = np.asarray(refined.x) * parameter_scale
            refined_cost, _gradient = full_objective(refined_parameters)
            if not math.isfinite(refined_cost) or not np.all(
                np.isfinite(refined_parameters)
            ):
                raise FloatingPointError(
                    "full-resolution regular-image refinement is non-finite"
                )
            solved, parameters = refined, refined_parameters
    if summary.all_valid and options.loss == "linear":
        information = _regular_image_linear_information(
            data, parameters, full_scale
        )
        final_cost, _final_gradient = full_objective(parameters)
        normalized_rss = 2.0 * final_cost
    else:
        _cost, _gradient, normalized_rss, information = _regular_image_striped_objective(
            data, parameters, full_scale, options.loss, True, check
        )
    degrees = max(summary.count - parameters.size, 1)
    normalized_reduced = normalized_rss / degrees
    covariance, covariance_valid = _covariance_from_information(
        information, normalized_reduced, summary.count
    )
    errors = (
        np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if covariance_valid
        else np.full(parameters.size, np.nan)
    )
    return FitResult(
        model,
        parameters,
        errors,
        covariance,
        None,
        None,
        None,
        summary.count,
        data_revision,
        bool(solved.success),
        solved.message,
        float(normalized_reduced * full_scale**2),
        covariance_valid=covariance_valid,
    )


def _regular_image_stripes(
    data: RegularImageFitInput,
    check: Callable[[], None],
):
    height = data.observations.shape[0]
    for start in range(0, height, _REGULAR_IMAGE_STRIPE_ROWS):
        check()
        stop = min(height, start + _REGULAR_IMAGE_STRIPE_ROWS)
        source = np.asarray(data.observations[start:stop])
        observed = source.astype(np.float64, copy=False)
        mask = None if data.valid_mask is None else data.valid_mask[start:stop]
        if source.dtype.kind == "f":
            finite = np.isfinite(source)
            mask = finite if mask is None else mask & finite
        if data.minimum_observation is not None:
            selected = observed >= data.minimum_observation
            mask = selected if mask is None else mask & selected
        if mask is not None and bool(np.all(mask)):
            mask = None
        yield start, observed, mask


def _regular_image_summary(
    data: RegularImageFitInput,
    check: Callable[[], None],
) -> _RegularImageSummary:
    minimum, maximum, scale, count = math.inf, -math.inf, 0.0, 0
    all_valid = True
    for _start, observed, mask in _regular_image_stripes(data, check):
        all_valid &= mask is None
        values = observed.reshape(-1) if mask is None else observed[mask]
        if values.size:
            low, high = float(np.min(values)), float(np.max(values))
            minimum, maximum = min(minimum, low), max(maximum, high)
            scale = max(scale, abs(low), abs(high))
            count += values.size
    if count == 0:
        raise ValueError("regular image has no finite valid observations")
    scale = scale or 1.0
    sums: list[float] = []
    square_sums: list[float] = []
    for _start, observed, mask in _regular_image_stripes(data, check):
        values = observed.reshape(-1) if mask is None else observed[mask]
        if values.size:
            normalized = values / scale
            sums.append(float(np.sum(normalized)))
            square_sums.append(float(np.dot(normalized, normalized)))
    return _RegularImageSummary(
        minimum, maximum, scale, count, all_valid, math.fsum(sums), math.fsum(square_sums)
    )


def _regular_image_sample(
    data: RegularImageFitInput,
    check: Callable[[], None],
) -> tuple[ArrayTuple, np.ndarray]:
    check()
    selection = data.valid_mask
    if data.observations.dtype.kind == "f":
        finite = np.isfinite(data.observations)
        selection = finite if selection is None else selection & finite
    if data.minimum_observation is not None:
        above = data.observations >= data.minimum_observation
        selection = above if selection is None else selection & above
    if selection is None:
        valid_y, valid_x = np.arange(data.y_coordinates.size), np.arange(data.x_coordinates.size)
    else:
        valid_y = np.flatnonzero(np.any(selection, axis=1))
        valid_x = np.flatnonzero(np.any(selection, axis=0))

    y_index, x_index = valid_y, valid_x
    sampled = np.asarray(data.observations[np.ix_(y_index, x_index)], dtype=np.float64)
    valid = np.isfinite(sampled)
    if selection is not None:
        valid &= selection[np.ix_(y_index, x_index)]
    if np.count_nonzero(valid) <= 5:
        raise ValueError("radial center fit needs a spatially coherent selection")
    offset = float(np.median(sampled[valid]))
    filtered = median_filter(np.where(valid, sampled, offset), size=3, mode="nearest")
    x_grid, y_grid = np.meshgrid(data.x_coordinates[x_index], data.y_coordinates[y_index])
    return (x_grid[valid], y_grid[valid]), filtered[valid]


def _regular_image_optimization_input(
    data: RegularImageFitInput,
    check: Callable[[], None],
) -> RegularImageFitInput:
    height, width = data.observations.shape
    if height <= _REGULAR_IMAGE_SAMPLE_LIMIT and width <= _REGULAR_IMAGE_SAMPLE_LIMIT:
        return data

    valid_y = np.zeros(height, dtype=np.bool_)
    valid_x = np.zeros(width, dtype=np.bool_)
    for start, observed, mask in _regular_image_stripes(data, check):
        stop = start + observed.shape[0]
        if mask is None:
            valid_y[start:stop] = True
            valid_x[:] = True
        else:
            valid_y[start:stop] = np.any(mask, axis=1)
            valid_x |= np.any(mask, axis=0)

    def bounded_indices(valid: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(valid)
        count = min(indices.size, _REGULAR_IMAGE_SAMPLE_LIMIT)
        positions = np.linspace(0, indices.size - 1, count, dtype=np.int64)
        return indices[positions]

    y_index, x_index = bounded_indices(valid_y), bounded_indices(valid_x)
    if y_index.size == 0 or x_index.size == 0:
        raise ValueError("regular image has no finite valid observations")
    observations = np.asarray(
        data.observations[np.ix_(y_index, x_index)], dtype=np.float64
    )
    valid = np.isfinite(observations)
    if data.valid_mask is not None:
        valid &= data.valid_mask[np.ix_(y_index, x_index)]
    if data.minimum_observation is not None:
        valid &= observations >= data.minimum_observation
    valid_mask = None if bool(np.all(valid)) else valid
    return RegularImageFitInput(
        data.x_coordinates[x_index],
        data.y_coordinates[y_index],
        observations,
        valid_mask=valid_mask,
    )


def _regular_image_linear_objective(
    data: RegularImageFitInput,
    summary: _RegularImageSummary,
    parameters: np.ndarray,
) -> tuple[float, np.ndarray]:
    amplitude, offset, radius, center_x, center_y = map(float, parameters)
    x, x_radius, x_center = _radial_axis_terms(data.x_coordinates, center_x, radius)
    y, y_radius, y_center = _radial_axis_terms(data.y_coordinates, center_y, radius)
    projected = np.einsum(
        "ij,jk->ik",
        data.observations,
        np.column_stack((x, x_radius, x_center)),
        optimize=False,
        dtype=np.float64,
        casting="unsafe",
    ) / summary.scale

    x_sum, y_sum = float(np.sum(x)), float(np.sum(y))
    x_square, y_square = float(np.dot(x, x)), float(np.dot(y, y))
    phi_sum, phi_square = x_sum * y_sum, x_square * y_square
    data_phi = float(np.dot(y, projected[:, 0]))
    data_derivatives = np.asarray(
        (
            np.dot(y, projected[:, 1]) + np.dot(y_radius, projected[:, 0]),
            np.dot(y, projected[:, 2]),
            np.dot(y_center, projected[:, 0]),
        )
    )
    derivative_sums = np.asarray(
        (
            y_sum * np.sum(x_radius) + np.sum(y_radius) * x_sum,
            y_sum * np.sum(x_center),
            np.sum(y_center) * x_sum,
        )
    )
    phi_derivatives = np.asarray(
        (
            y_square * np.dot(x, x_radius) + np.dot(y, y_radius) * x_square,
            y_square * np.dot(x, x_center),
            np.dot(y, y_center) * x_square,
        )
    )

    amplitude /= summary.scale
    offset /= summary.scale
    terms = (
        amplitude**2 * phi_square,
        2.0 * amplitude * offset * phi_sum,
        offset**2 * summary.count,
        -2.0 * amplitude * data_phi,
        -2.0 * offset * summary.normalized_sum,
        summary.normalized_square_sum,
    )
    rss = math.fsum(terms)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(math.fsum(map(abs, terms)), 1.0)
    if not math.isfinite(rss) or rss < -tolerance:
        raise FloatingPointError("regular-image objective is non-finite")
    rss = max(rss, 0.0)

    residual_phi = amplitude * phi_square + offset * phi_sum - data_phi
    residual_sum = amplitude * phi_sum + offset * summary.count - summary.normalized_sum
    geometry = amplitude * (
        amplitude * phi_derivatives + offset * derivative_sums - data_derivatives
    )
    gradient = np.r_[
        residual_phi / summary.scale,
        residual_sum / summary.scale,
        geometry,
    ]
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("regular-image gradient is non-finite")
    return 0.5 * rss, gradient


def _regular_image_linear_information(
    data: RegularImageFitInput,
    parameters: np.ndarray,
    scale: float,
) -> np.ndarray:
    amplitude, _offset, radius, center_x, center_y = map(float, parameters)
    x, x_radius, x_center = _radial_axis_terms(
        data.x_coordinates, center_x, radius
    )
    y, y_radius, y_center = _radial_axis_terms(
        data.y_coordinates, center_y, radius
    )
    terms: tuple[tuple[tuple[float, np.ndarray | None, np.ndarray | None], ...], ...] = (
        ((1.0, y, x),),
        ((1.0, None, None),),
        ((amplitude, y, x_radius), (amplitude, y_radius, x)),
        ((amplitude, y, x_center),),
        ((amplitude, y_center, x),),
    )

    def axis_inner(
        left: np.ndarray | None,
        right: np.ndarray | None,
        size: int,
    ) -> float:
        if left is None:
            return float(size) if right is None else float(np.sum(right))
        return float(np.sum(left)) if right is None else float(np.dot(left, right))

    information = np.empty((5, 5), dtype=np.float64)
    for row, row_terms in enumerate(terms):
        for column in range(row, len(terms)):
            value = math.fsum(
                row_scale
                * column_scale
                * axis_inner(row_y, column_y, data.y_coordinates.size)
                * axis_inner(row_x, column_x, data.x_coordinates.size)
                for row_scale, row_y, row_x in row_terms
                for column_scale, column_y, column_x in terms[column]
            ) / scale**2
            information[row, column] = information[column, row] = value
    if not np.all(np.isfinite(information)):
        raise FloatingPointError("regular-image information is non-finite")
    return information



def _radial_axis_terms(
    coordinates: np.ndarray,
    center: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = coordinates - center
    basis = np.exp(-(delta**2) / radius**2)
    return basis, basis * (2.0 * delta**2 / radius**3), basis * (2.0 * delta / radius**2)


def _regular_image_loss_terms(
    residual: np.ndarray,
    loss: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    squared = residual**2
    if loss == "linear":
        first = np.ones_like(residual)
        return squared, first, first
    if loss == "soft_l1":
        root = np.sqrt(1.0 + squared)
        rho = 2.0 * squared / (root + 1.0)
        first, second = 1.0 / root, -0.5 / root**3
    elif loss == "huber":
        outside = squared > 1.0
        root = np.sqrt(squared)
        inverse = np.divide(1.0, root, out=np.ones_like(root), where=root != 0.0)
        rho = np.where(outside, 2.0 * root - 1.0, squared)
        first, second = (
            np.where(outside, inverse, 1.0),
            np.where(outside, -0.5 * inverse**3, 0.0),
        )
    elif loss == "cauchy":
        first = 1.0 / (1.0 + squared)
        rho, second = np.log1p(squared), -(first**2)
    elif loss == "arctan":
        denominator = 1.0 + squared**2
        rho, first = np.arctan(squared), 1.0 / denominator
        second = -2.0 * squared / denominator**2
    else:
        raise RuntimeError(loss)
    information_weight = np.maximum(
        first + 2.0 * second * squared, np.finfo(np.float64).eps
    )
    return rho, first, information_weight
def _regular_image_striped_objective(
    data: RegularImageFitInput,
    parameters: np.ndarray,
    scale: float,
    loss: str,
    collect_information: bool,
    check: Callable[[], None],
) -> tuple[float, np.ndarray, float, np.ndarray]:
    amplitude, offset, radius, center_x, center_y = map(float, parameters)
    x_basis, x_radius, x_center = _radial_axis_terms(
        data.x_coordinates, center_x, radius
    )
    gradient = np.zeros(5, dtype=np.float64)
    information = np.zeros((5, 5), dtype=np.float64)
    costs: list[float] = []
    square_sums: list[float] = []
    for start, observed, mask in _regular_image_stripes(data, check):
        stop = start + observed.shape[0]
        y_basis, y_radius, y_center = _radial_axis_terms(
            data.y_coordinates[start:stop], center_y, radius
        )
        radial = y_basis[:, None] * x_basis[None, :]
        residual = (amplitude * radial + offset - observed) / scale
        derivatives = (
            radial,
            np.ones_like(radial),
            amplitude
            * (y_basis[:, None] * x_radius[None, :] + y_radius[:, None] * x_basis[None, :]),
            amplitude * y_basis[:, None] * x_center[None, :],
            amplitude * y_center[:, None] * x_basis[None, :],
        )
        residual = residual.reshape(-1)
        if mask is not None:
            selected = mask.reshape(-1)
            residual = residual[selected]
            derivatives = tuple(derivative.reshape(-1)[selected] for derivative in derivatives)
        else:
            derivatives = tuple(derivative.reshape(-1) for derivative in derivatives)
        rho, first, information_weight = _regular_image_loss_terms(residual, loss)
        costs.append(0.5 * float(np.sum(rho)))
        square_sums.append(float(np.dot(residual, residual)))
        weighted_residual = first * residual
        gradient += np.asarray(
            [np.dot(derivative, weighted_residual) / scale for derivative in derivatives]
        )
        if collect_information:
            jacobian = np.column_stack(derivatives) / scale
            information += jacobian.T @ (information_weight[:, None] * jacobian)
    cost, rss = math.fsum(costs), math.fsum(square_sums)
    finite = (
        math.isfinite(cost)
        and math.isfinite(rss)
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(information))
    )
    if not finite:
        raise FloatingPointError("regular-image objective is non-finite")
    return cost, gradient, rss, information


def _coordinate_arrays(coordinates: Sequence[np.ndarray], arity: int) -> ArrayTuple:
    arrays = tuple(np.asarray(item, dtype=np.float64).reshape(-1) for item in coordinates)
    if len(arrays) != arity:
        raise ValueError("coordinate arity does not match fit model")
    if arrays and any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("coordinate arrays must have equal shape")
    return arrays


def _solver_bounds(
    model: FitModelSpec,
    defaults: Mapping[str, tuple[float | None, float | None]] | None,
    requested: Mapping[str, tuple[float | None, float | None]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    names = set(model.parameter_names)
    for label, source in (("default bounds", defaults), ("bounds", requested)):
        unknown = set(source or ()) - names
        if unknown:
            raise ValueError(f"{label} name unknown parameters: {sorted(unknown)}")
    for parameter in model.parameters:
        low, high = parameter.bounds
        source = (
            requested
            if requested is not None and parameter.name in requested
            else defaults
        )
        if source is not None and parameter.name in source:
            source_low, source_high = source[parameter.name]
            if source_low is not None:
                low = max(low, float(source_low))
            if source_high is not None:
                high = min(high, float(source_high))
        if not low < high:
            raise ValueError(f"empty bounds for parameter {parameter.name!r}")
        lower.append(low)
        upper.append(high)
    return np.asarray(lower), np.asarray(upper)


def _initial_values(
    model: FitModelSpec,
    coordinates: ArrayTuple,
    values: np.ndarray,
    initial: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    if initial is None:
        seed = np.asarray(model.initializer(coordinates, values), dtype=np.float64).reshape(-1)
    elif isinstance(initial, Mapping):
        unknown = set(initial) - set(model.parameter_names)
        if unknown:
            raise ValueError(f"initial values name unknown parameters: {sorted(unknown)}")
        defaults = dict(zip(model.parameter_names, model.initializer(coordinates, values), strict=True))
        defaults.update({key: float(value) for key, value in initial.items()})
        seed = np.asarray([defaults[name] for name in model.parameter_names], dtype=np.float64)
    else:
        seed = np.asarray(initial, dtype=np.float64).reshape(-1)
    if seed.shape != (len(model.parameters),) or not np.all(np.isfinite(seed)):
        raise ValueError("fit initializer returned invalid parameter values")
    return seed


def _initial_candidates(
    model: FitModelSpec,
    coordinates: ArrayTuple,
    values: np.ndarray,
    initial: Mapping[str, float] | Sequence[float] | None,
) -> tuple[np.ndarray, ...]:
    if initial is not None or model.candidate_initializer is None:
        return (_initial_values(model, coordinates, values, initial),)
    candidates = tuple(model.candidate_initializer(coordinates, values))
    if not candidates:
        raise ValueError("fit initializer returned no candidates")
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        seed = _initial_values(model, coordinates, values, candidate)
        key = seed.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(seed)
    return tuple(unique)


def _covariance(
    jacobian: np.ndarray,
    reduced_chi_square: float,
) -> tuple[np.ndarray, bool]:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    parameter_count = jacobian.shape[1] if jacobian.ndim == 2 else 0
    invalid = np.full((parameter_count, parameter_count), np.nan, dtype=np.float64)
    reduced_chi_square = float(reduced_chi_square)
    if (
        jacobian.ndim != 2
        or parameter_count == 0
        or jacobian.shape[0] <= parameter_count
        or not np.all(np.isfinite(jacobian))
        or not math.isfinite(reduced_chi_square)
        or reduced_chi_square < 0.0
    ):
        return invalid, False
    try:
        _left, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return invalid, False
    if singular_values.size != parameter_count or not np.all(np.isfinite(singular_values)):
        return invalid, False
    tolerance = (
        np.finfo(np.float64).eps
        * max(jacobian.shape)
        * float(singular_values[0])
    )
    if singular_values[-1] <= tolerance:
        return invalid, False
    inverse_information = (right.T / singular_values**2) @ right
    covariance = inverse_information * reduced_chi_square
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(covariance)
    if not np.all(np.isfinite(covariance)) or np.any(diagonal < 0.0):
        return invalid, False
    return covariance, True


def _covariance_from_information(
    information: np.ndarray,
    reduced_chi_square: float,
    observation_count: int,
) -> tuple[np.ndarray, bool]:
    matrix = np.asarray(information, dtype=np.float64)
    count = matrix.shape[0] if matrix.ndim == 2 else 0
    invalid = np.full((count, count), np.nan, dtype=np.float64)
    if (
        matrix.shape != (count, count)
        or not count
        or observation_count <= count
        or not np.all(np.isfinite(matrix))
        or not math.isfinite(reduced_chi_square)
        or reduced_chi_square < 0.0
    ):
        return invalid, False
    norms = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    if np.any(norms <= np.finfo(np.float64).tiny):
        return invalid, False
    normalized = matrix / np.outer(norms, norms)
    try:
        values, vectors = np.linalg.eigh((normalized + normalized.T) / 2.0)
    except np.linalg.LinAlgError:
        return invalid, False
    tolerance = np.finfo(np.float64).eps * count * float(np.max(values))
    if values[0] <= tolerance:
        return invalid, False
    covariance = (vectors / values) @ vectors.T
    covariance *= float(reduced_chi_square) / np.outer(norms, norms)
    covariance = (covariance + covariance.T) / 2.0
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0.0):
        return invalid, False
    return covariance, True


VALUE = UnitRelation.VALUE
AXIS_0 = UnitRelation.AXIS_0
AXIS_1 = UnitRelation.AXIS_1
INVERSE_AXIS_0 = UnitRelation.INVERSE_AXIS_0
RADIAN = UnitRelation.RADIAN
REAL = ParameterDomain.REAL
POSITIVE = ParameterDomain.POSITIVE
NONNEGATIVE = ParameterDomain.NONNEGATIVE
PHASE = ParameterDomain.PHASE_RADIANS


def _span(x: np.ndarray) -> float:
    return max(float(np.ptp(x)), np.finfo(np.float64).eps)


def _edge_offset(y: np.ndarray) -> float:
    count = max(1, min(y.size // 10, 20))
    return float(np.median(np.concatenate((y[:count], y[-count:]))))


def _peak_seed(coords: ArrayTuple, y: np.ndarray) -> tuple[float, float, float, float]:
    x = coords[0]
    offset = _edge_offset(y)
    delta = y - offset
    index = int(np.argmax(np.abs(delta)))
    amplitude = float(delta[index])
    if amplitude == 0:
        amplitude = float(np.ptp(y) or 1.0)
    center = float(x[index])
    return amplitude, offset, center, _span(x)


def _data_interval(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if low < high:
        return low, high
    padding = math.sqrt(np.finfo(np.float64).eps) * max(abs(low), 1.0)
    return low - padding, high + padding


def _value_range(values: np.ndarray) -> float:
    low, high = _data_interval(values)
    return high - low


def _lorentzian(x, center, fwhm, amplitude, offset):
    half_squared = (fwhm / 2.0) ** 2
    return amplitude * half_squared / ((x - center) ** 2 + half_squared) + offset


def _gaussian_offset(x, amplitude, offset, sigma, center):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset


def _histogram_gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _bimodal_left(
    x,
    center,
    center_splitting,
    left_amplitude,
    left_sigma,
    _right_amplitude,
    _right_sigma,
):
    return _histogram_gaussian(
        x,
        left_amplitude,
        center - center_splitting / 2.0,
        left_sigma,
    )


def _bimodal_right(
    x,
    center,
    center_splitting,
    _left_amplitude,
    _left_sigma,
    right_amplitude,
    right_sigma,
):
    return _histogram_gaussian(
        x,
        right_amplitude,
        center + center_splitting / 2.0,
        right_sigma,
    )


def _bimodal_gaussian(
    x,
    center,
    center_splitting,
    left_amplitude,
    left_sigma,
    right_amplitude,
    right_sigma,
):
    parameters = (
        center,
        center_splitting,
        left_amplitude,
        left_sigma,
        right_amplitude,
        right_sigma,
    )
    return _bimodal_left(x, *parameters) + _bimodal_right(x, *parameters)


def _symmetric_lorentzian_doublet(x, center, common_fwhm, component_amplitude, offset, center_splitting):
    return _lorentzian(x, center - center_splitting / 2, common_fwhm, component_amplitude, offset) + _lorentzian(
        x, center + center_splitting / 2, common_fwhm, component_amplitude, 0.0
    )


def _damped_sine(x, amplitude, offset, baseband_frequency, decay_time, phase):
    return offset + amplitude * np.exp(-x / decay_time) * np.sin(
        2 * np.pi * baseband_frequency * x + phase
    )


def _exponential_decay(x, amplitude, offset, decay_time):
    return offset + amplitude * np.exp(-x / decay_time)


def _radial_gaussian_center(x, y, amplitude, offset, one_over_e_radius, center_x, center_y):
    radius_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    return offset + amplitude * np.exp(-radius_squared / one_over_e_radius**2)


def _init_lorentzian(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    return _lorentzian_candidates(coords, y)[0]


def _lorentzian_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    x = coords[0]
    width = _span(x) / 4.0
    y_range = _value_range(y)
    low_y, high_y = float(np.min(y)), float(np.max(y))
    return (
        (float(x[np.argmax(y)]), width, y_range, low_y),
        (float(x[np.argmin(y)]), width, -y_range, high_y),
    )


def _lorentzian_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    x_low, x_high = _data_interval(coords[0])
    y_low, y_high = _data_interval(y)
    x_span = _span(coords[0])
    y_range = _value_range(y)
    width = x_span / 4.0
    return {
        "center": (x_low, x_high),
        "fwhm": (width / 10.0, width * 10.0),
        "amplitude": (-10.0 * y_range, 10.0 * y_range),
        "offset": (y_low - 10.0 * y_range, y_high + 10.0 * y_range),
    }


def _init_gaussian(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    amplitude, offset, center, span = _peak_seed(coords, y)
    return amplitude, offset, span / 6.0, center


def _init_histogram(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    weights = np.maximum(y, 0)
    total = float(np.sum(weights))
    if total <= 0:
        center = float(np.mean(x))
        sigma = _span(x) / 6
    else:
        center = float(np.sum(x * weights) / total)
        sigma = max(float(np.sqrt(np.sum(weights * (x - center) ** 2) / total)), _span(x) / 1000)
    return max(float(np.max(y)), 0.0), center, sigma


def _init_bimodal(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    midpoint = float((np.min(x) + np.max(x)) / 2)
    left = y[x <= midpoint]
    right = y[x > midpoint]
    left_x = x[x <= midpoint]
    right_x = x[x > midpoint]
    span = _span(x)
    lc = float(left_x[np.argmax(left)]) if left.size else midpoint - span / 4
    rc = float(right_x[np.argmax(right)]) if right.size else midpoint + span / 4
    la = max(float(np.max(left)) if left.size else float(np.max(y)), 0.0)
    ra = max(float(np.max(right)) if right.size else float(np.max(y)), 0.0)
    return (lc + rc) / 2, abs(rc - lc), la, span / 10, ra, span / 10


def _init_doublet(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    return _doublet_candidates(coords, y)[0]


def _doublet_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    order = np.argsort(coords[0], kind="stable")
    x = coords[0][order]
    ordered_y = y[order]
    x_span = _span(x)
    y_range = _value_range(ordered_y)
    unique_x = np.unique(x)
    step = (
        float(np.median(np.abs(np.diff(unique_x))))
        if unique_x.size > 1
        else x_span
    )
    step = max(step, np.finfo(np.float64).eps)
    seeds: list[tuple[float, ...]] = []
    for sign in (1.0, -1.0):
        signed = sign * ordered_y
        peaks, properties = find_peaks(
            signed,
            width=1,
            prominence=y_range / 8.0,
        )
        if peaks.size == 0:
            continue
        strongest = peaks[np.argsort(signed[peaks])[::-1]][:4]
        widths = properties.get("widths", np.ones(peaks.size))
        width_by_peak = {
            int(peak): max(float(widths[index]) * step, step)
            for index, peak in enumerate(peaks)
        }
        first = int(strongest[0])
        for second_raw in strongest:
            second = int(second_raw)
            width = width_by_peak[first]
            seeds.append(
                (
                    float((x[first] + x[second]) / 2.0),
                    width,
                    sign * y_range,
                    float(np.min(ordered_y) if sign > 0.0 else np.max(ordered_y)),
                    float(abs(x[second] - x[first])),
                )
            )
    if seeds:
        return tuple(seeds)
    width = x_span / 8.0
    return (
        (
            float(x[np.argmax(ordered_y)]),
            width,
            y_range,
            float(np.min(ordered_y)),
            width * 2.0,
        ),
        (
            float(x[np.argmin(ordered_y)]),
            width,
            -y_range,
            float(np.max(ordered_y)),
            width * 2.0,
        ),
    )


def _doublet_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    x_low, x_high = _data_interval(coords[0])
    y_low, y_high = _data_interval(y)
    x_span = _span(coords[0])
    y_range = _value_range(y)
    width = float(_doublet_candidates(coords, y)[0][1])
    return {
        "center": (x_low, x_high),
        "common_fwhm": (width / 10.0, width * 10.0),
        "component_amplitude": (-10.0 * y_range, 10.0 * y_range),
        "offset": (y_low - 10.0 * y_range, y_high + 10.0 * y_range),
        "center_splitting": (0.0, 2.0 * x_span),
    }


def _init_damped_sine(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    order = np.argsort(x)
    sx = x[order]
    sy = y[order]
    offset = float(np.mean(sy))
    amplitude = max(float(np.ptp(sy) / 2), np.finfo(np.float64).eps)
    spacing = float(np.median(np.diff(sx))) if sx.size > 1 else 1.0
    spectrum = np.abs(np.fft.rfft(sy - offset))
    frequencies = np.fft.rfftfreq(sy.size, d=max(abs(spacing), np.finfo(float).eps))
    frequency = float(frequencies[1 + np.argmax(spectrum[1:])]) if spectrum.size > 1 else 1 / _span(x)
    frequency = max(frequency, np.finfo(float).eps)
    decay_time = _span(x)
    origin = float(np.min(x))
    exponent = origin / decay_time
    if abs(exponent) < 700.0:
        amplitude *= math.exp(exponent)
    phase = float(
        (-2.0 * np.pi * frequency * origin + np.pi) % (2.0 * np.pi) - np.pi
    )
    return amplitude, offset, frequency, decay_time, phase


def _damped_sine_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    amplitude, offset, frequency, decay_time, phase = _init_damped_sine(coords, y)
    return tuple(
        (
            amplitude,
            offset,
            frequency,
            decay_time,
            float((phase + shift + np.pi) % (2.0 * np.pi) - np.pi),
        )
        for shift in (-np.pi / 2.0, 0.0, np.pi / 2.0)
    )


def _damped_sine_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    amplitude, _offset, frequency, decay_time, _phase = _init_damped_sine(coords, y)
    y_low, y_high = _data_interval(y)
    return {
        "amplitude": (amplitude / 5.0, amplitude * 5.0),
        "offset": (y_low, y_high),
        "baseband_frequency": (frequency / 5.0, frequency * 5.0),
        "decay_time": (decay_time / 5.0, decay_time * 5.0),
    }


def _init_exponential(coords: ArrayTuple, y: np.ndarray) -> Sequence[float]:
    x = coords[0]
    offset = float(np.median(y[np.argsort(x)[-max(1, y.size // 10) :]]))
    decay_time = _span(x) / 3
    observed_amplitude = float(y[np.argmin(x)] - offset)
    exponent = float(np.min(x)) / decay_time
    amplitude = (
        observed_amplitude * math.exp(exponent)
        if abs(exponent) < 700.0
        else observed_amplitude
    )
    return amplitude or float(np.ptp(y) or 1.0), offset, decay_time


def _exponential_candidates(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Sequence[Sequence[float]]:
    amplitude, offset, decay_time = _init_exponential(coords, y)
    return (
        (amplitude, offset, decay_time),
        (-amplitude, offset, decay_time),
    )


def _exponential_bounds(
    coords: ArrayTuple,
    y: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    amplitude, _offset, decay_time = _init_exponential(coords, y)
    y_low, y_high = _data_interval(y)
    y_range = _value_range(y)
    amplitude_limit = max(4.0 * y_range, 10.0 * abs(amplitude))
    return {
        "amplitude": (-amplitude_limit, amplitude_limit),
        "offset": (y_low - 10.0 * y_range, y_high + 10.0 * y_range),
        "decay_time": (decay_time / 10.0, decay_time * 10.0),
    }


def _init_radial(coords: ArrayTuple, values: np.ndarray) -> Sequence[float]:
    return _radial_seed(coords, values, 1.0)


def _radial_seed(
    coords: ArrayTuple,
    values: np.ndarray,
    sign: float,
) -> tuple[float, ...]:
    x, y = coords
    offset = float(np.median(values))
    weights = np.maximum(sign * (values - offset), 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        center_x, center_y = float(np.mean(x)), float(np.mean(y))
        radius = max(_span(x), _span(y)) / 4
    else:
        center_x = float(np.sum(x * weights) / total)
        center_y = float(np.sum(y * weights) / total)
        radius = max(
            float(np.sqrt(np.sum(weights * ((x - center_x) ** 2 + (y - center_y) ** 2)) / total)),
            np.finfo(float).eps,
        )
    amplitude = (
        float(np.max(values) - offset)
        if sign > 0.0
        else float(np.min(values) - offset)
    )
    if amplitude == 0.0:
        amplitude = sign * _value_range(values)
    return amplitude, offset, radius, center_x, center_y


def _radial_candidates(
    coords: ArrayTuple,
    values: np.ndarray,
) -> Sequence[Sequence[float]]:
    return (
        _radial_seed(coords, values, 1.0),
        _radial_seed(coords, values, -1.0),
    )


def _radial_bounds(
    coords: ArrayTuple,
    values: np.ndarray,
) -> Mapping[str, tuple[float | None, float | None]]:
    x_low, x_high = _data_interval(coords[0])
    y_low, y_high = _data_interval(coords[1])
    value_low, value_high = _data_interval(values)
    value_range = _value_range(values)
    radii = [float(seed[2]) for seed in _radial_candidates(coords, values)]
    radius_low = max(min(radii) / 10.0, np.finfo(np.float64).eps)
    radius_high = max(radii) * 10.0
    return {
        "amplitude": (-4.0 * value_range, 4.0 * value_range),
        "offset": (value_low - value_range, value_high + value_range),
        "one_over_e_radius": (radius_low, radius_high),
        "center_x": (x_low, x_high),
        "center_y": (y_low, y_high),
    }


def builtin_fit_models() -> tuple[FitModelSpec, ...]:
    return (
        FitModelSpec(
            "lorentzian",
            "Lorentzian",
            1,
            (
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "fwhm", AXIS_0, POSITIVE, display_label=r"$\mathrm{FWHM}$"
                ),
                FitParameterSpec("amplitude", VALUE, display_label=r"$H$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
            ),
            _lorentzian,
            _init_lorentzian,
            (FitTarget.SERIES,),
            formula=(
                r"$f(x)=H\frac{(\mathrm{FWHM}/2)^2}"
                r"{(x-x_0)^2+(\mathrm{FWHM}/2)^2}+B$"
            ),
            candidate_initializer=_lorentzian_candidates,
            bounds_initializer=_lorentzian_bounds,
            default_for=(FitTarget.SERIES,),
        ),
        FitModelSpec(
            "gaussian_offset",
            "Gaussian with offset",
            1,
            (
                FitParameterSpec("amplitude", VALUE, display_label=r"$A$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "sigma", AXIS_0, POSITIVE, display_label=r"$\sigma$"
                ),
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
            ),
            _gaussian_offset,
            _init_gaussian,
            (FitTarget.SERIES,),
            formula=r"$f(x)=A e^{-\frac{1}{2}((x-x_0)/\sigma)^2}+B$",
        ),
        FitModelSpec(
            "histogram_gaussian",
            "Single Gaussian",
            1,
            (
                FitParameterSpec(
                    "amplitude", VALUE, NONNEGATIVE, display_label=r"$A$"
                ),
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "sigma", AXIS_0, POSITIVE, display_label=r"$\sigma$"
                ),
            ),
            _histogram_gaussian,
            _init_histogram,
            (FitTarget.HISTOGRAM,),
            formula=r"$f(x)=A e^{-\frac{1}{2}((x-x_0)/\sigma)^2}$",
        ),
        FitModelSpec(
            "bimodal_gaussian",
            "Bimodal Gaussian",
            1,
            (
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "center_splitting",
                    AXIS_0,
                    NONNEGATIVE,
                    display_label=r"$\delta$",
                ),
                FitParameterSpec(
                    "left_amplitude", VALUE, NONNEGATIVE, display_label=r"$A_L$"
                ),
                FitParameterSpec(
                    "left_sigma", AXIS_0, POSITIVE, display_label=r"$\sigma_L$"
                ),
                FitParameterSpec(
                    "right_amplitude", VALUE, NONNEGATIVE, display_label=r"$A_R$"
                ),
                FitParameterSpec(
                    "right_sigma", AXIS_0, POSITIVE, display_label=r"$\sigma_R$"
                ),
            ),
            _bimodal_gaussian,
            _init_bimodal,
            (FitTarget.HISTOGRAM,),
            formula=(
                r"$f(x)=A_L e^{-\frac{1}{2}((x-x_0+\delta/2)/\sigma_L)^2}"
                r"+A_R e^{-\frac{1}{2}((x-x_0-\delta/2)/\sigma_R)^2}$"
            ),
            presentation=FitPresentationSpec(
                components=(
                    FitComponentSpec("left", _bimodal_left),
                    FitComponentSpec("right", _bimodal_right),
                ),
                crossover_components=("left", "right"),
            ),
            default_for=(FitTarget.HISTOGRAM,),
        ),
        FitModelSpec(
            "symmetric_lorentzian_doublet",
            "Symmetric Lorentzian doublet",
            1,
            (
                FitParameterSpec(
                    "center", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "common_fwhm",
                    AXIS_0,
                    POSITIVE,
                    display_label=r"$\mathrm{FWHM}$",
                ),
                FitParameterSpec(
                    "component_amplitude", VALUE, display_label=r"$H$"
                ),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "center_splitting",
                    AXIS_0,
                    NONNEGATIVE,
                    display_label=r"$\delta$",
                ),
            ),
            _symmetric_lorentzian_doublet,
            _init_doublet,
            (FitTarget.SERIES,),
            formula=(
                r"$f(x)=H[L(x;x_0-\delta/2)+L(x;x_0+\delta/2)]+B$"
            ),
            candidate_initializer=_doublet_candidates,
            bounds_initializer=_doublet_bounds,
        ),
        FitModelSpec(
            "damped_sine",
            "Damped sine",
            1,
            (
                FitParameterSpec(
                    "amplitude", VALUE, NONNEGATIVE, display_label=r"$A$"
                ),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "baseband_frequency",
                    INVERSE_AXIS_0,
                    POSITIVE,
                    display_label=r"$f$",
                ),
                FitParameterSpec(
                    "decay_time", AXIS_0, POSITIVE, display_label=r"$\tau$"
                ),
                FitParameterSpec("phase", RADIAN, PHASE, display_label=r"$\varphi$"),
            ),
            _damped_sine,
            _init_damped_sine,
            (FitTarget.SERIES,),
            formula=r"$f(t)=A\sin(2\pi f t+\varphi)e^{-t/\tau}+B$",
            candidate_initializer=_damped_sine_candidates,
            bounds_initializer=_damped_sine_bounds,
        ),
        FitModelSpec(
            "exponential_decay",
            "Exponential decay",
            1,
            (
                FitParameterSpec("amplitude", VALUE, display_label=r"$A$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "decay_time", AXIS_0, POSITIVE, display_label=r"$\tau$"
                ),
            ),
            _exponential_decay,
            _init_exponential,
            (FitTarget.SERIES,),
            formula=r"$f(t)=Ae^{-t/\tau}+B$",
            candidate_initializer=_exponential_candidates,
            bounds_initializer=_exponential_bounds,
        ),
        FitModelSpec(
            "radial_gaussian_center",
            "Radial Gaussian center",
            2,
            (
                FitParameterSpec("amplitude", VALUE, display_label=r"$A$"),
                FitParameterSpec(
                    "offset", VALUE, display_label=r"$B$", affine_point=True
                ),
                FitParameterSpec(
                    "one_over_e_radius", AXIS_0, POSITIVE, display_label=r"$R$"
                ),
                FitParameterSpec(
                    "center_x", AXIS_0, display_label=r"$x_0$", affine_point=True
                ),
                FitParameterSpec(
                    "center_y",
                    AXIS_1,
                    display_label=r"$y_0$",
                    affine_point=True,
                    solver_unit_relation=AXIS_0,
                ),
            ),
            _radial_gaussian_center,
            _init_radial,
            (FitTarget.IMAGE,),
            formula=r"$f(x,y)=Ae^{-((x-x_0)^2+(y-y_0)^2)/R^2}+B$",
            candidate_initializer=_radial_candidates,
            bounds_initializer=_radial_bounds,
            presentation=FitPresentationSpec(
                radial_glyph=FitRadialGlyphSpec(
                    ("center_x", "center_y"),
                    "one_over_e_radius",
                ),
            ),
            coordinate_relations=(AXIS_0, AXIS_0),
            default_for=(FitTarget.IMAGE,),
            capabilities=frozenset({"regular_image_radial"}),
        ),
    )


def default_fit_registry() -> FitModelRegistry:
    return FitModelRegistry(builtin_fit_models())


_FIT_ASSIGNMENT = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_fit_initials(model: FitModelSpec, command: str) -> dict[str, float]:
    """Parse ``name=value`` fit initials from one compact command field."""

    if not isinstance(model, FitModelSpec):
        raise TypeError("model must be FitModelSpec")
    if not isinstance(command, str):
        raise TypeError("command must be text")
    text = command.strip()
    if not text:
        return {}
    allowed = set(model.parameter_names)
    values: dict[str, float] = {}
    position = 0
    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position == len(text):
            break
        match = _FIT_ASSIGNMENT.match(text, position)
        if match is None:
            raise ValueError(f"invalid fit assignment near {text[position:]!r}")
        name = match.group("name")
        if name not in allowed:
            raise KeyError(f"fit model {model.model_id!r} has no parameter {name!r}")
        if name in values:
            raise ValueError(f"fit parameter {name!r} is assigned more than once")
        value = float(match.group("value"))
        if not math.isfinite(value):
            raise ValueError(f"fit parameter {name!r} must be finite")
        values[name] = value
        position = match.end()
        if position < len(text) and not (
            text[position].isspace() or text[position] == ","
        ):
            raise ValueError(f"invalid fit assignment separator near {text[position:]!r}")
    return values


def format_fit_initials(
    model: FitModelSpec,
    values: Mapping[str, float],
) -> str:
    """Format initials in model parameter order for the same command field."""

    if not isinstance(model, FitModelSpec):
        raise TypeError("model must be FitModelSpec")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    unknown = tuple(name for name in values if name not in model.parameter_names)
    if unknown:
        raise KeyError(f"fit model {model.model_id!r} has no parameter {unknown[0]!r}")
    parts = []
    for name in model.parameter_names:
        if name not in values:
            continue
        value = float(values[name])
        if not math.isfinite(value):
            raise ValueError(f"fit parameter {name!r} must be finite")
        parts.append(f"{name}={value:.17g}")
    return ", ".join(parts)


__all__ = [
    "FacetFitBatchResult",
    "FitCancelled",
    "FitComponentSpec",
    "FitDeadlineExceeded",
    "FitEngine",
    "FitModelRegistry",
    "FitModelSpec",
    "FitOptions",
    "FitParameterDisplay",
    "FitParameterSpec",
    "FitPresentationSpec",
    "FitRadialGlyphSpec",
    "FitResult",
    "FitTarget",
    "format_fit_initials",
    "parse_fit_initials",
    "ParameterDomain",
    "RegularImageFitInput",
    "UnitRelation",
    "builtin_fit_models",
    "default_fit_registry",
]
