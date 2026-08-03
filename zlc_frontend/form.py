"""Headless, immutable presentation contracts for simple parameter forms.

The domain owner remains the sole source of value semantics and validation.  A
``FormSpec`` is only its ordered presentation projection; it is neither a saved
schema nor a generic request builder.  Qt consumes this module, while importing
it never loads Qt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Literal, TypeAlias


FormFieldKind: TypeAlias = Literal[
    "text",
    "int",
    "float",
    "number",
    "choice",
    "bool",
    "axis_range",
    "path",
    "signal",
]
_FORM_FIELD_KINDS = frozenset(
    {
        "text",
        "int",
        "float",
        "number",
        "choice",
        "bool",
        "axis_range",
        "path",
        "signal",
    }
)
_INT_TEXT = re.compile(r"[+-]?\d+")
_NUMBER_TEXT = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)


def parse_number_text(text: str, field: str = "number") -> int | float:
    """Parse one finite decimal leaf, preserving authored int-vs-float type."""

    if not isinstance(text, str):
        raise TypeError(f"{field} text must be str")
    value = text.strip()
    if not value:
        raise ValueError(f"{field} is required")
    if _INT_TEXT.fullmatch(value) is not None:
        return int(value, 10)
    if _NUMBER_TEXT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a finite decimal number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _typed_equal(left: object, right: object) -> bool:
    """Compare choice values without conflating ``1``, ``1.0`` and ``True``."""

    return type(left) is type(right) and bool(left == right)


def choice_value_to_tree(field: "FormFieldProps", value: object) -> object:
    """Encode one exact typed choice as its current JSON scalar."""

    if not isinstance(field, FormFieldProps) or field.kind != "choice":
        raise TypeError("choice serialization requires a choice FormFieldProps")
    choice = field.choice_for(value)
    if choice is None:
        raise ValueError(f"field {field.key!r} received an undeclared choice")
    encoded = choice.value.value if isinstance(choice.value, Enum) else choice.value
    if encoded is None or isinstance(encoded, (dict, list, tuple, set)):
        raise TypeError(
            f"choice field {field.key!r} needs an owner scalar serializer"
        )
    if isinstance(encoded, float) and not math.isfinite(encoded):
        raise ValueError(f"choice field {field.key!r} encoded a non-finite float")
    return encoded


def choice_value_from_tree(field: "FormFieldProps", payload: object) -> object:
    """Decode one current JSON scalar back to the declared typed choice."""

    if not isinstance(field, FormFieldProps) or field.kind != "choice":
        raise TypeError("choice decoding requires a choice FormFieldProps")
    matches = tuple(
        choice
        for choice in field.choices
        if _typed_equal(choice_value_to_tree(field, choice.value), payload)
    )
    if len(matches) != 1:
        raise ValueError(f"field {field.key!r} has no unique encoded choice {payload!r}")
    return matches[0].value


def _require_immutable_scalar(value: object, *, where: str) -> None:
    if value is None:
        return
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{where} must be an immutable scalar value") from exc


@dataclass(frozen=True, slots=True)
class FormChoice:
    """One display label paired with its unchanged typed value."""

    label: str
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("choice label must be a non-empty string")
        if self.value is None:
            raise ValueError("None represents no selection and cannot be a choice value")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("choice values cannot contain non-finite floats")
        _require_immutable_scalar(self.value, where=f"choice {self.label!r}")


@dataclass(frozen=True, slots=True)
class FormFieldProps:
    """Presentation-only properties for one declared form field.

    ``minimum`` and ``maximum`` are real owner-declared bounds.  ``None`` means
    unbounded; the Qt layer must not invent a widget limit in its place.
    """

    key: str
    kind: FormFieldKind
    label: str
    default: object = None
    required: bool = False
    unit: str = ""
    description: str = ""
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[FormChoice, ...] = ()
    allow_blank: bool | None = None
    path_mode: Literal["file", "dir"] = "file"
    file_filter: str = "All files (*)"
    base_dir: str = ""
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("form field key must be a non-empty string")
        if self.key != self.key.strip():
            raise ValueError("form field key cannot have surrounding whitespace")
        if self.kind not in _FORM_FIELD_KINDS:
            raise ValueError(f"unsupported form field kind: {self.kind!r}")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"field {self.key!r} label must be non-empty")
        if not isinstance(self.required, bool):
            raise TypeError(f"field {self.key!r} required must be bool")
        if not isinstance(self.unit, str) or not isinstance(self.description, str):
            raise TypeError(f"field {self.key!r} unit/description must be strings")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError(
                f"field {self.key!r} unavailable_reason must be a string"
            )
        if self.allow_blank is not None and not isinstance(self.allow_blank, bool):
            raise TypeError(f"field {self.key!r} allow_blank must be bool or None")
        if self.path_mode not in {"file", "dir"}:
            raise ValueError(f"field {self.key!r} path_mode must be file or dir")
        if not all(
            isinstance(value, str)
            for value in (self.file_filter, self.base_dir)
        ):
            raise TypeError(
                f"field {self.key!r} path metadata must contain strings"
            )

        choices = tuple(self.choices)
        object.__setattr__(self, "choices", choices)
        for choice in choices:
            if not isinstance(choice, FormChoice):
                raise TypeError(f"field {self.key!r} choices must contain FormChoice")
        for index, choice in enumerate(choices):
            if any(_typed_equal(choice.value, prior.value) for prior in choices[:index]):
                raise ValueError(f"field {self.key!r} has duplicate typed choice values")

        if self.kind == "choice":
            if not choices:
                if not self.required:
                    raise ValueError(
                        f"empty choice field {self.key!r} must be required"
                    )
                if self.default is not None:
                    raise ValueError(
                        f"empty choice field {self.key!r} cannot have a default"
                    )
                if not self.unavailable_reason.strip():
                    raise ValueError(
                        f"empty choice field {self.key!r} needs an "
                        "unavailable_reason"
                    )
            elif self.unavailable_reason:
                raise ValueError(
                    f"available choice field {self.key!r} cannot declare an "
                    "unavailable_reason"
                )
            if self.minimum is not None or self.maximum is not None:
                raise ValueError(f"choice field {self.key!r} cannot declare numeric bounds")
            if self.default is not None and not any(
                _typed_equal(self.default, choice.value) for choice in choices
            ):
                raise ValueError(f"field {self.key!r} default is not a typed choice value")
        elif choices:
            raise ValueError(f"non-choice field {self.key!r} cannot declare choices")
        elif self.unavailable_reason:
            raise ValueError(
                f"non-choice field {self.key!r} cannot declare an "
                "unavailable_reason"
            )

        if self.kind not in {"int", "float", "number", "axis_range"} and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError(f"non-numeric field {self.key!r} cannot declare bounds")

        if self.kind == "int":
            for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
                if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                    raise TypeError(f"int field {self.key!r} {name} must be int")
            if self.default is not None and (
                not isinstance(self.default, int) or isinstance(self.default, bool)
            ):
                raise TypeError(f"int field {self.key!r} default must be int or None")
        elif self.kind == "float":
            for name, value in (
                ("minimum", self.minimum),
                ("maximum", self.maximum),
                ("default", self.default),
            ):
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"float field {self.key!r} {name} must be numeric")
                if not math.isfinite(float(value)):
                    raise ValueError(f"float field {self.key!r} {name} must be finite")
            if self.default is not None:
                object.__setattr__(self, "default", float(self.default))
        elif self.kind == "number":
            for name, value in (
                ("minimum", self.minimum),
                ("maximum", self.maximum),
                ("default", self.default),
            ):
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"number field {self.key!r} {name} must be numeric")
                if not math.isfinite(float(value)):
                    raise ValueError(f"number field {self.key!r} {name} must be finite")
        elif self.kind == "text":
            if self.default is not None and not isinstance(self.default, str):
                raise TypeError(f"text field {self.key!r} default must be str or None")
        elif self.kind == "bool":
            if not isinstance(self.default, bool):
                raise TypeError(f"bool field {self.key!r} default must be bool")
        elif self.kind == "axis_range":
            for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(
                        f"axis_range field {self.key!r} {name} must be numeric"
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"axis_range field {self.key!r} {name} must be finite"
                    )
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
                    f"axis_range field {self.key!r} default must be "
                    "(number, number, int)"
                )
            lo, hi, points = default
            if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
                raise ValueError(
                    f"axis_range field {self.key!r} endpoints must be finite"
                )
            if points < 2:
                raise ValueError(
                    f"axis_range field {self.key!r} points must be at least 2"
                )
            if self.minimum is not None and min(float(lo), float(hi)) < self.minimum:
                raise ValueError(
                    f"axis_range field {self.key!r} default is below minimum"
                )
            if self.maximum is not None and max(float(lo), float(hi)) > self.maximum:
                raise ValueError(
                    f"axis_range field {self.key!r} default is above maximum"
                )
        elif self.kind in {"path", "signal"}:
            if self.default is not None and not isinstance(self.default, str):
                raise TypeError(
                    f"{self.kind} field {self.key!r} default must be str or None"
                )
        if self.kind not in {"int", "float", "number"} and self.allow_blank is not None:
            raise ValueError(
                f"non-scalar field {self.key!r} cannot declare allow_blank"
            )
        if self.kind != "path" and (
            self.path_mode != "file"
            or self.file_filter != "All files (*)"
            or self.base_dir
        ):
            raise ValueError(
                f"non-path field {self.key!r} cannot declare path metadata"
            )
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError(f"field {self.key!r} minimum exceeds maximum")
        if self.default is not None and self.kind in {"int", "float", "number"}:
            numeric = float(self.default) if self.kind == "float" else self.default
            if self.minimum is not None and numeric < self.minimum:
                raise ValueError(f"field {self.key!r} default is below minimum")
            if self.maximum is not None and numeric > self.maximum:
                raise ValueError(f"field {self.key!r} default is above maximum")

        _require_immutable_scalar(self.default, where=f"field {self.key!r} default")

    @property
    def blank_allowed(self) -> bool:
        """Whether a scalar editor may hold an explicit blank value."""

        if self.kind not in {"int", "float", "number"}:
            return False
        if self.allow_blank is not None:
            return self.allow_blank
        return self.default is None and not self.required

    @property
    def required_choice_unavailable(self) -> bool:
        """Whether installation discovery found no value for a required choice.

        This is distinct from an operator leaving an available choice blank.  It
        keeps product vocabulary visible without manufacturing a placeholder
        value that could be mistaken for a real device role.
        """

        return self.kind == "choice" and self.required and not self.choices

    @property
    def row_label(self) -> str:
        label = self.label.strip()
        if self.unit:
            label = f"{label} ({self.unit})"
        if self.required:
            label = f"{label} *"
        return label

    def choice_for(self, value: object) -> FormChoice | None:
        """Return the canonical typed choice matching ``value``, if any."""

        if self.kind != "choice":
            return None
        return next(
            (choice for choice in self.choices if _typed_equal(value, choice.value)),
            None,
        )


@dataclass(frozen=True, slots=True)
class FormSpec:
    """An ordered, exact-key collection of simple form fields."""

    fields: tuple[FormFieldProps, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        object.__setattr__(self, "fields", fields)
        if any(not isinstance(field, FormFieldProps) for field in fields):
            raise TypeError("form spec fields must contain FormFieldProps")
        keys = tuple(field.key for field in fields)
        if len(keys) != len(set(keys)):
            raise ValueError("form spec field keys must be unique")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(field.key for field in self.fields)

    def default_values(self) -> dict[str, object]:
        return {field.key: field.default for field in self.fields}


def project_authoring_form(
    schema: object,
    *,
    dynamic_choices: Mapping[str, object] | None = None,
    path_presentations: Mapping[str, object] | None = None,
) -> FormSpec:
    """Project an ordered owner declaration into the canonical form contract.

    The source declaration remains domain-owned.  This projector deliberately
    consumes its small structural field vocabulary instead of importing a
    product/domain schema type back into the frontend.
    """

    source_fields = getattr(schema, "fields", None)
    if source_fields is None:
        raise TypeError("schema must expose ordered authoring fields")
    fields = tuple(source_fields)
    required_attributes = (
        "key",
        "kind",
        "label",
        "default",
        "required",
        "unit",
        "description",
        "minimum",
        "maximum",
        "choices",
        "dynamic_choices",
        "allow_blank",
    )
    if any(
        any(not hasattr(field, attribute) for attribute in required_attributes)
        for field in fields
    ):
        raise TypeError("authoring fields do not expose the canonical vocabulary")

    injected = dict(dynamic_choices or {})
    dynamic_keys = {field.key for field in fields if field.dynamic_choices}
    if set(injected) != dynamic_keys:
        raise ValueError(
            "dynamic choice keys must exactly cover the owner declaration"
        )
    if any(
        key != getattr(value, "field_key", None)
        or not hasattr(value, "choices")
        or not hasattr(value, "default")
        or not hasattr(value, "unavailable_reason")
        for key, value in injected.items()
    ):
        raise TypeError(
            "dynamic choices must map field keys to their exact owner presentation"
        )

    paths = dict(path_presentations or {})
    path_keys = {field.key for field in fields if field.kind == "path"}
    if not set(paths) <= path_keys:
        raise ValueError("path presentation keys must name owner-declared path fields")
    if any(
        key != getattr(value, "field_key", None)
        or not hasattr(value, "mode")
        or not hasattr(value, "file_filter")
        or not hasattr(value, "base_dir")
        for key, value in paths.items()
    ):
        raise TypeError(
            "path presentations must map field keys to their exact owner hint"
        )

    projected = []
    for field in fields:
        if field.dynamic_choices:
            projection = injected[field.key]
            choices = tuple(
                FormChoice(choice.label, choice.value)
                for choice in projection.choices
            )
            values = tuple(choice.value for choice in projection.choices)
            if values and not any(
                _typed_equal(projection.default, value) for value in values
            ):
                raise ValueError("dynamic choice default is not one available value")
            if not values and projection.default is not None:
                raise ValueError("unavailable dynamic choice cannot have a default")
            default = projection.default
            unavailable = projection.unavailable_reason
        else:
            choices = tuple(
                FormChoice(choice.label, choice.value)
                for choice in field.choices
            )
            default = field.default
            unavailable = ""
        path = paths.get(field.key)
        projected.append(
            FormFieldProps(
                field.key,
                field.kind,
                field.label,
                default=default,
                required=field.required,
                unit=field.unit,
                description=field.description,
                minimum=field.minimum,
                maximum=field.maximum,
                choices=choices,
                allow_blank=field.allow_blank,
                path_mode="file" if path is None else path.mode,
                file_filter=(
                    "All files (*)" if path is None else path.file_filter
                ),
                base_dir="" if path is None else path.base_dir,
                unavailable_reason=unavailable,
            )
        )
    return FormSpec(tuple(projected))


__all__ = [
    "choice_value_from_tree",
    "choice_value_to_tree",
    "FormChoice",
    "FormFieldKind",
    "FormFieldProps",
    "FormSpec",
    "parse_number_text",
    "project_authoring_form",
]
