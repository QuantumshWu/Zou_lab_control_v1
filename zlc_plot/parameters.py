"""Typed, plot-kind-neutral display parameter declarations.

Concrete plot kinds register their own :class:`ParameterSchema`.  This module
deliberately contains no names such as ``bin_count`` or ``relim_mode``; it only
defines how a parameter is validated and which part of a rendered view it can
invalidate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import IntFlag, auto
from types import MappingProxyType
from numbers import Real
from typing import Any, Generic, TypeVar, cast

from ._validation import finite_real

T = TypeVar("T")


class RenderEffect(IntFlag):
    """Independent work caused by an accepted display-parameter change.

    Effects are flags rather than an ordered severity.  A unit change, for
    example, simultaneously reprojects data, axis text, selectors and an
    existing fit presentation; no single ``strongest`` category can describe
    that transaction without losing information.
    """

    NONE = 0
    VIEW_PROJECTION = auto()
    PAYLOAD_PROJECTION = auto()
    BASE_GEOMETRY = auto()
    BASE_STYLE = auto()
    AXIS_TRANSFORM = auto()
    TEXT = auto()
    CHROME = auto()
    OVERLAY = auto()
    LAYOUT = auto()
    FIT_SELECTION = auto()
    INTERACTION_REPROJECT = auto()

RuntimeType = type[Any] | tuple[type[Any], ...]
Normalizer = Callable[[object], T]
_TransitionNormalizer = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, object],
]
_FullStateValidator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ParameterSpec(Generic[T]):
    """Declaration for one typed display parameter.

    ``normalizer`` is useful at a UI boundary (for example converting a spin
    box value to an enum).  Its result is type-checked before storage.
    """

    name: str
    value_type: RuntimeType
    effects: RenderEffect
    default: T
    normalizer: Normalizer[T] | None = None
    allow_none: bool = False
    label: str | None = None
    choices: tuple[T, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("parameter name must not have surrounding whitespace")
        _validate_runtime_type(self.value_type)
        if not isinstance(self.effects, RenderEffect):
            raise TypeError("effects must be RenderEffect")
        if self.effects == RenderEffect.NONE:
            raise ValueError("parameter effects must not be RenderEffect.NONE")
        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("normalizer must be callable or None")
        label = self.name.replace("_", " ").strip().title() if self.label is None else self.label
        if not isinstance(label, str) or not label.strip():
            raise ValueError("parameter label must be non-empty text")
        object.__setattr__(self, "label", label.strip())
        minimum = _finite_bound(self.minimum, "minimum")
        maximum = _finite_bound(self.maximum, "maximum")
        step = _finite_bound(self.step, "step")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("parameter minimum must not exceed maximum")
        if step is not None and step <= 0.0:
            raise ValueError("parameter step must be positive")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "step", step)
        prepared_choices = tuple(self._prepare_value(value) for value in self.choices)
        if any(value is None for value in prepared_choices):
            raise ValueError("parameter choices must not contain None")
        if len(set(prepared_choices)) != len(prepared_choices):
            raise ValueError("parameter choices must be unique")
        object.__setattr__(self, "choices", prepared_choices)
        normalized = self.prepare(self.default)
        object.__setattr__(self, "default", normalized)

    def prepare(self, value: object) -> T | None:
        """Normalize and validate a value for storage in a display state."""

        prepared = self._prepare_value(value)
        if prepared is None:
            return None
        if self.choices and prepared not in self.choices:
            choices = ", ".join(repr(choice) for choice in self.choices)
            raise ValueError(f"parameter {self.name!r} must be one of {choices}")
        if isinstance(prepared, Real) and not isinstance(prepared, bool):
            numeric = float(prepared)
            if self.minimum is not None and numeric < self.minimum:
                raise ValueError(
                    f"parameter {self.name!r} must be at least {self.minimum:g}"
                )
            if self.maximum is not None and numeric > self.maximum:
                raise ValueError(
                    f"parameter {self.name!r} must be at most {self.maximum:g}"
                )
        return cast(T, prepared)

    def _prepare_value(self, value: object) -> T | None:
        prepared = self.normalizer(value) if self.normalizer is not None else value
        if prepared is None:
            if self.allow_none:
                return None
            raise TypeError(f"parameter {self.name!r} does not accept None")
        if not _matches_runtime_type(prepared, self.value_type):
            expected = _runtime_type_name(self.value_type)
            raise TypeError(
                f"parameter {self.name!r} requires {expected}; "
                f"got {type(prepared).__name__}"
            )
        return cast(T, prepared)


class FrozenParameters(Mapping[str, Any]):
    """A small immutable mapping of scalar display parameters."""

    __slots__ = ("__data",)

    def __init__(self, values: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> None:
        self.__data = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> Any:
        return self.__data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"FrozenParameters({dict(self.__data)!r})"


class ParameterSchema(Mapping[str, ParameterSpec[Any]]):
    """An immutable registry of parameters for one view or plot kind."""

    __slots__ = ("__full_state_validator", "__specs", "__transition_normalizer")

    def __init__(
        self,
        specs: Iterable[ParameterSpec[Any]],
        *,
        transition_normalizer: _TransitionNormalizer | None = None,
        full_state_validator: _FullStateValidator | None = None,
    ) -> None:
        registered: dict[str, ParameterSpec[Any]] = {}
        for spec in specs:
            if not isinstance(spec, ParameterSpec):
                raise TypeError("schema entries must be ParameterSpec instances")
            if spec.name in registered:
                raise ValueError(f"duplicate parameter name: {spec.name!r}")
            registered[spec.name] = spec
        if transition_normalizer is not None and not callable(transition_normalizer):
            raise TypeError("transition_normalizer must be callable or None")
        if full_state_validator is not None and not callable(full_state_validator):
            raise TypeError("full_state_validator must be callable or None")
        self.__specs = registered
        self.__transition_normalizer = transition_normalizer
        self.__full_state_validator = full_state_validator

    def __getitem__(self, key: str) -> ParameterSpec[Any]:
        return self.__specs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__specs)

    def __len__(self) -> int:
        return len(self.__specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.__specs)

    def prepare_updates(self, updates: Mapping[str, object]) -> dict[str, Any]:
        if not isinstance(updates, Mapping):
            raise TypeError("display parameter updates must be a mapping")
        unknown = tuple(name for name in updates if name not in self.__specs)
        if unknown:
            joined = ", ".join(repr(name) for name in unknown)
            raise KeyError(f"unknown display parameter(s): {joined}")
        return {name: self.__specs[name].prepare(value) for name, value in updates.items()}

    def initial_values(self, overrides: Mapping[str, object] | None = None) -> FrozenParameters:
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError("initial parameter overrides must be a mapping or None")
        defaults = FrozenParameters(
            (name, spec.default) for name, spec in self.__specs.items()
        )
        return self.prepare_transition(defaults, {} if overrides is None else overrides)

    def prepare_transition(
        self,
        current: Mapping[str, Any],
        updates: Mapping[str, object],
    ) -> FrozenParameters:
        """Normalize updates and validate the resulting complete state."""

        return self._transition_prepared(current, self.prepare_updates(updates))

    def _transition_prepared(
        self,
        current: Mapping[str, Any],
        prepared: Mapping[str, Any],
    ) -> FrozenParameters:
        if self.__transition_normalizer is not None:
            transitioned = self.__transition_normalizer(current, prepared)
            if not isinstance(transitioned, Mapping):
                raise TypeError("transition_normalizer must return a mapping")
            retained = {
                name: value
                for name, value in transitioned.items()
                if name in prepared and value is prepared[name]
            }
            generated = {
                name: value
                for name, value in transitioned.items()
                if name not in retained
            }
            retained.update(self.prepare_updates(generated))
            prepared = retained
        values = dict(current)
        values.update(prepared)
        self._validate_full_state(values)
        return FrozenParameters(values)

    def _validate_full_state(self, values: Mapping[str, Any]) -> None:
        if self.__full_state_validator is not None:
            self.__full_state_validator(values)

    def effects_for(self, names: Iterable[str]) -> RenderEffect:
        names_tuple = tuple(names)
        unknown = tuple(name for name in names_tuple if name not in self.__specs)
        if unknown:
            joined = ", ".join(repr(name) for name in unknown)
            raise KeyError(f"unknown display parameter(s): {joined}")
        combined = RenderEffect.NONE
        for name in names_tuple:
            combined |= self.__specs[name].effects
        return combined


def _validate_runtime_type(value_type: RuntimeType) -> None:
    if isinstance(value_type, type):
        return
    if (
        not isinstance(value_type, tuple)
        or not value_type
        or not all(isinstance(item, type) for item in value_type)
    ):
        raise TypeError("value_type must be a type or a non-empty tuple of types")


def _matches_runtime_type(value: object, value_type: RuntimeType) -> bool:
    types = (value_type,) if isinstance(value_type, type) else value_type
    # ``bool`` is an ``int`` subclass, but accepting a checkbox value for a
    # numeric parameter is almost always a UI integration error.
    if isinstance(value, bool) and bool not in types:
        return False
    return isinstance(value, types)


def _runtime_type_name(value_type: RuntimeType) -> str:
    types = (value_type,) if isinstance(value_type, type) else value_type
    return " or ".join(item.__name__ for item in types)


def _finite_bound(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return finite_real(value, f"parameter {name}")


__all__ = [
    "FrozenParameters",
    "ParameterSchema",
    "ParameterSpec",
    "RenderEffect",
]
