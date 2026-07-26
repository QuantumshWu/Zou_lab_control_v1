"""Mechanical Request/Intent authoring-schema projection into ``FormSpec``."""

from __future__ import annotations

from typing import Mapping

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_neutral_atom.logic_node_declaration import (
    DynamicChoicePresentation,
    PathPresentationHint,
)


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and bool(left == right)


def project_authoring_form(
    schema: AuthoringSchema,
    *,
    dynamic_choices: Mapping[str, DynamicChoicePresentation] | None = None,
    path_presentations: Mapping[str, PathPresentationHint] | None = None,
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
    if any(
        not isinstance(value, DynamicChoicePresentation)
        or key != value.field_key
        for key, value in injected.items()
    ):
        raise TypeError(
            "dynamic choices must map field keys to their exact owner presentation"
        )
    paths = dict(path_presentations or {})
    path_keys = {field.key for field in schema.fields if field.kind == "path"}
    if not set(paths) <= path_keys:
        raise ValueError(
            "path presentation keys must name owner-declared path fields"
        )
    if any(
        not isinstance(value, PathPresentationHint) or key != value.field_key
        for key, value in paths.items()
    ):
        raise TypeError(
            "path presentations must map field keys to their exact owner hint"
        )
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
        path = paths.get(field.key)
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
                path_mode="file" if path is None else path.mode,
                file_filter="All files (*)" if path is None else path.file_filter,
                base_dir="" if path is None else path.base_dir,
                unavailable_reason=unavailable,
            )
        )
    return FormSpec(tuple(fields))


__all__ = [
    "project_authoring_form",
]
