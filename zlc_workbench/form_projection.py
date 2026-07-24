"""Mechanical Request/Intent authoring-schema projection into ``FormSpec``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.authoring import AuthoringSchema


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and bool(left == right)


@dataclass(frozen=True, slots=True)
class PresentedChoice:
    value: object
    label: str


@dataclass(frozen=True, slots=True)
class DynamicChoiceProjection:
    choices: tuple[PresentedChoice, ...]
    default: object = None
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        values = tuple(self.choices)
        if any(not isinstance(choice, PresentedChoice) for choice in values):
            raise TypeError("dynamic choices must contain PresentedChoice values")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("dynamic choice unavailable_reason must be str")
        object.__setattr__(self, "choices", values)


@dataclass(frozen=True, slots=True)
class PathPresentation:
    """Workbench-only file-dialog placement for an owner-declared path."""

    mode: Literal["file", "dir"] = "file"
    file_filter: str = "All files (*)"
    base_dir: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"file", "dir"}:
            raise ValueError("path presentation mode must be file or dir")
        if not isinstance(self.file_filter, str) or not isinstance(self.base_dir, str):
            raise TypeError("path presentation filter/base_dir must be strings")


def freeze_authoring_values(
    schema: AuthoringSchema,
    values: Mapping[str, object],
    *,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, object]:
    """Freeze exactly the owner fields, filling only owner-declared defaults."""

    if not isinstance(schema, AuthoringSchema):
        raise TypeError("schema must be AuthoringSchema")
    if not isinstance(values, Mapping):
        raise TypeError("authoring values must be a mapping")
    extras = tuple(extra_keys)
    if any(not isinstance(key, str) or not key for key in extras):
        raise ValueError("extra authoring keys must be non-empty strings")
    if len(set(extras)) != len(extras) or set(extras) & set(schema.keys):
        raise ValueError("extra authoring keys must be unique and non-owner")
    unknown = set(values) - set(schema.keys) - set(extras)
    if unknown:
        raise ValueError(
            "authoring values contain unknown fields: "
            f"{tuple(sorted(map(str, unknown)))}"
        )
    return schema.freeze(
        {key: value for key, value in values.items() if key in schema.keys}
    )


def project_authoring_form(
    schema: AuthoringSchema,
    *,
    dynamic_choices: Mapping[str, DynamicChoiceProjection] | None = None,
    path_presentations: Mapping[str, PathPresentation] | None = None,
) -> FormSpec:
    """Auto-project ordinary fields with only runtime/presentation injections."""

    if not isinstance(schema, AuthoringSchema):
        raise TypeError("schema must be AuthoringSchema")
    injected = dict(dynamic_choices or {})
    dynamic_keys = {
        field.key for field in schema.fields if field.dynamic_choices
    }
    if set(injected) != dynamic_keys:
        raise ValueError(
            "dynamic choice keys must exactly cover the owner declaration"
        )
    paths = dict(path_presentations or {})
    path_keys = {field.key for field in schema.fields if field.kind == "path"}
    if not set(paths) <= path_keys:
        raise ValueError(
            "path presentation keys must name owner-declared path fields"
        )
    if any(not isinstance(value, PathPresentation) for value in paths.values()):
        raise TypeError("path presentations must contain PathPresentation values")
    fields = []
    for field in schema.fields:
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
        path = paths.get(field.key, PathPresentation())
        fields.append(
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
                path_mode=path.mode,
                file_filter=path.file_filter,
                base_dir=path.base_dir,
                unavailable_reason=unavailable,
            )
        )
    return FormSpec(tuple(fields))


__all__ = [
    "DynamicChoiceProjection",
    "PathPresentation",
    "PresentedChoice",
    "freeze_authoring_values",
    "project_authoring_form",
]
