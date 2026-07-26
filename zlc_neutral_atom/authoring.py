"""Typed ordinary-field declarations beside neutral-atom Requests/Intents.

These immutable values own field semantics and stable human labels.  They are
not GUI schemas: they contain no widget, dialog, callback, registry, schema-id
dispatch, persistence codec, or request constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from zlc_storage import canonical_text


_KINDS = frozenset(
    {"text", "int", "float", "number", "choice", "bool", "axis_range", "path"}
)
MINIMUM_POSITIVE_FLOAT = math.nextafter(0.0, math.inf)


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and bool(left == right)


def _scalar(value: object, field: str) -> None:
    if value is None:
        return
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{field} must be an immutable scalar") from error
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


@dataclass(frozen=True, slots=True)
class AuthoringChoice:
    value: object
    label: str

    def __post_init__(self) -> None:
        _scalar(self.value, "authoring choice")
        canonical_text(self.label, "authoring choice label")


@dataclass(frozen=True, slots=True)
class AuthoringField:
    key: str
    kind: str
    label: str
    default: object = None
    required: bool = False
    unit: str = ""
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[AuthoringChoice, ...] = ()
    dynamic_choices: bool = False
    description: str = ""
    allow_blank: bool | None = None

    def __post_init__(self) -> None:
        canonical_text(self.key, "authoring field key")
        canonical_text(self.label, "authoring field label")
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported authoring field kind {self.kind!r}")
        if type(self.required) is not bool or type(self.dynamic_choices) is not bool:
            raise TypeError("authoring required/dynamic_choices must be bool")
        if self.allow_blank is not None and type(self.allow_blank) is not bool:
            raise TypeError("authoring allow_blank must be bool or None")
        if not isinstance(self.unit, str) or not isinstance(self.description, str):
            raise TypeError("authoring unit/description must be strings")
        choices = tuple(self.choices)
        if any(not isinstance(choice, AuthoringChoice) for choice in choices):
            raise TypeError("authoring choices must contain AuthoringChoice values")
        values = tuple(choice.value for choice in choices)
        if any(
            any(_typed_equal(value, prior) for prior in values[:index])
            for index, value in enumerate(values)
        ):
            raise ValueError("authoring choice values must be unique")
        object.__setattr__(self, "choices", choices)
        if self.kind == "choice":
            if self.dynamic_choices:
                if choices or self.default is not None:
                    raise ValueError("dynamic choices are injected only by composition")
            elif not choices or not any(
                _typed_equal(self.default, value) for value in values
            ):
                raise ValueError("static choice default must be a declared choice")
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("choice fields cannot declare numeric bounds")
        elif choices or self.dynamic_choices:
            raise ValueError("only choice fields can declare choices")
        if self.kind not in {"int", "float", "number", "axis_range"} and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("only numeric fields can declare bounds")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("authoring minimum exceeds maximum")
        if self.kind == "int":
            for value in (self.default, self.minimum, self.maximum):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int)
                ):
                    raise TypeError("integer authoring values must be int")
        elif self.kind in {"float", "number"}:
            for value in (self.default, self.minimum, self.maximum):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                ):
                    raise TypeError("numeric authoring values must be int or float")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("numeric authoring values must be finite")
        if self.kind == "bool" and type(self.default) is not bool:
            raise TypeError("bool authoring default must be bool")
        if self.kind in {"text", "path"} and self.default is not None:
            if not isinstance(self.default, str):
                raise TypeError(f"{self.kind} authoring default must be str or None")
        if self.kind == "axis_range":
            default = self.default
            if (
                not isinstance(default, tuple)
                or len(default) != 3
                or isinstance(default[0], bool)
                or isinstance(default[1], bool)
                or not isinstance(default[0], (int, float))
                or not isinstance(default[1], (int, float))
                or not isinstance(default[2], int)
                or isinstance(default[2], bool)
            ):
                raise TypeError(
                    "axis_range authoring default must be (number, number, int)"
                )
            lo, hi, points = default
            if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
                raise ValueError("axis_range authoring endpoints must be finite")
            if points < 2:
                raise ValueError("axis_range authoring points must be at least 2")
            if self.minimum is not None and min(float(lo), float(hi)) < self.minimum:
                raise ValueError("axis_range authoring default is below minimum")
            if self.maximum is not None and max(float(lo), float(hi)) > self.maximum:
                raise ValueError("axis_range authoring default is above maximum")
        if self.default is not None and self.kind in {"int", "float", "number"}:
            if self.minimum is not None and self.default < self.minimum:
                raise ValueError("authoring default is below minimum")
            if self.maximum is not None and self.default > self.maximum:
                raise ValueError("authoring default is above maximum")
        if self.kind not in {"int", "float", "number"} and self.allow_blank is not None:
            raise ValueError("only scalar numeric fields can declare allow_blank")
        _scalar(self.default, f"authoring field {self.key!r} default")

    def freeze(self, value: object) -> object:
        """Validate one authored leaf without adding presentation coercions."""

        if value is None:
            if self.required:
                raise ValueError(f"{self.label} is required")
            if self.kind in {"int", "float", "number"} and not self.allow_blank:
                raise ValueError(f"{self.label} cannot be blank")
            return None
        if isinstance(value, str) and not value.strip():
            if self.kind in {"int", "float", "number"}:
                raise TypeError(
                    f"{self.label} optional numeric value must be None, not text"
                )
            if self.required:
                raise ValueError(f"{self.label} is required")
            return value

        if self.kind in {"text", "path"}:
            if not isinstance(value, str):
                raise TypeError(f"{self.label} must be str")
            return value
        if self.kind == "bool":
            if type(value) is not bool:
                raise TypeError(f"{self.label} must be bool")
            return value
        if self.kind == "choice":
            _scalar(value, self.label)
            if not self.dynamic_choices and not any(
                _typed_equal(value, choice.value) for choice in self.choices
            ):
                raise ValueError(f"{self.label} is not a declared choice")
            return value
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{self.label} must be int")
            scalar = value
        elif self.kind in {"float", "number"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{self.label} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{self.label} must be finite")
            scalar = value
        elif self.kind == "axis_range":
            if not isinstance(value, tuple) or len(value) != 3:
                raise TypeError(
                    f"{self.label} must be a (start, stop, points) tuple"
                )
            start, stop, points = value
            if (
                isinstance(start, bool)
                or isinstance(stop, bool)
                or not isinstance(start, (int, float))
                or not isinstance(stop, (int, float))
                or isinstance(points, bool)
                or not isinstance(points, int)
            ):
                raise TypeError(
                    f"{self.label} must contain two numbers and an integer"
                )
            if not math.isfinite(float(start)) or not math.isfinite(float(stop)):
                raise ValueError(f"{self.label} endpoints must be finite")
            if points < 2:
                raise ValueError(f"{self.label} points must be at least 2")
            if self.minimum is not None and min(start, stop) < self.minimum:
                raise ValueError(f"{self.label} is below its minimum")
            if self.maximum is not None and max(start, stop) > self.maximum:
                raise ValueError(f"{self.label} exceeds its maximum")
            return value
        else:  # pragma: no cover - constructor closes the vocabulary
            raise RuntimeError(f"unsupported authoring kind {self.kind!r}")

        if self.minimum is not None and scalar < self.minimum:
            raise ValueError(f"{self.label} is below its minimum")
        if self.maximum is not None and scalar > self.maximum:
            raise ValueError(f"{self.label} exceeds its maximum")
        return value


@dataclass(frozen=True, slots=True)
class AuthoringSchema:
    fields: tuple[AuthoringField, ...]

    def __post_init__(self) -> None:
        values = tuple(self.fields)
        if not values or any(not isinstance(field, AuthoringField) for field in values):
            raise TypeError("authoring schema requires AuthoringField values")
        keys = tuple(field.key for field in values)
        if len(keys) != len(set(keys)):
            raise ValueError("authoring schema keys must be unique")
        object.__setattr__(self, "fields", values)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(field.key for field in self.fields)

    def freeze(self, values: Mapping[str, object]) -> dict[str, object]:
        """Freeze exact authored leaves, filling only owner-declared defaults."""

        if not isinstance(values, Mapping):
            raise TypeError("authoring values must be a mapping")
        unknown = set(values) - set(self.keys)
        if unknown:
            raise ValueError(
                "authoring values contain unknown fields: "
                f"{tuple(sorted(map(str, unknown)))}"
            )
        return {
            field.key: field.freeze(
                values[field.key] if field.key in values else field.default
            )
            for field in self.fields
        }


__all__ = [
    "AuthoringChoice",
    "AuthoringField",
    "AuthoringSchema",
    "MINIMUM_POSITIVE_FLOAT",
]
