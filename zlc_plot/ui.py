"""Frontend-neutral mapping from plot parameter schemas to editor controls.

The plot-kind schema remains authoritative for names, validation and render
effects.  Frontends consume this module to choose a widget without duplicating
those rules; no Qt, Jupyter or Matplotlib object is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from .parameters import ParameterSchema, RenderEffect
from .kinds import PlotKind
from .resolver import (
    _plot_spec_authoring_control_names,
    _plot_spec_authoring_schema,
    _plot_spec_authoring_values,
)
from .specs import PlotSpec
from zlc_data import DatasetSchema


class ControlKind(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    TEXT = "text"
    CHOICE = "choice"


@dataclass(frozen=True, slots=True)
class ParameterControl:
    """One immutable, toolkit-independent parameter editor description."""

    name: str
    label: str
    kind: ControlKind
    value: object
    allow_none: bool
    choices: tuple[object, ...]
    minimum: float | None
    maximum: float | None
    step: float | None
    effects: RenderEffect


def parameter_controls(
    schema: ParameterSchema,
    values: Mapping[str, object],
    *,
    choice_overrides: Mapping[str, tuple[object, ...]] | None = None,
) -> tuple[ParameterControl, ...]:
    """Project one canonical schema/state pair into ordered UI controls.

    ``choice_overrides`` supplies data- or environment-dependent editor
    domains, such as compatible units and the package-defined colormap
    catalogue. It changes only the editor choices; the core schema still
    validates every submitted value.
    """

    if not isinstance(schema, ParameterSchema):
        raise TypeError("schema must be ParameterSchema")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    if choice_overrides is not None and not isinstance(choice_overrides, Mapping):
        raise TypeError("choice_overrides must be a mapping or None")
    overrides = {} if choice_overrides is None else dict(choice_overrides)
    unknown = tuple(name for name in overrides if name not in schema)
    if unknown:
        joined = ", ".join(repr(name) for name in unknown)
        raise KeyError(f"choice override refers to unknown parameter(s): {joined}")
    result = []
    for name, spec in schema.items():
        if name not in values:
            raise KeyError(f"display state is missing parameter {name!r}")
        choices = tuple(overrides.get(name, spec.choices))
        kind = _control_kind(spec.value_type, choices)
        result.append(
            ParameterControl(
                name=name,
                label=str(spec.label),
                kind=kind,
                value=values[name],
                allow_none=spec.allow_none,
                choices=choices,
                minimum=spec.minimum,
                maximum=spec.maximum,
                step=spec.step,
                effects=spec.effects,
            )
        )
    return tuple(result)


def plot_spec_controls(
    schema: DatasetSchema,
    kind: PlotKind,
    *,
    spec: PlotSpec | None = None,
    values: Mapping[str, object] | None = None,
) -> tuple[ParameterControl, ...]:
    """Describe an unresolved or existing PlotSpec with the same UI vocabulary."""

    parameter_schema, _sample_refs = _plot_spec_authoring_schema(schema, kind)
    prepared = _plot_spec_authoring_values(schema, kind, spec)
    if values is not None:
        prepared = dict(parameter_schema.prepare_transition(prepared, values))
    names = _plot_spec_authoring_control_names(schema, kind, prepared)
    by_name = {
        control.name: control
        for control in parameter_controls(parameter_schema, prepared)
    }
    return tuple(by_name[name] for name in names)


def _control_kind(value_type: object, choices: tuple[object, ...]) -> ControlKind:
    if choices:
        return ControlKind.CHOICE
    types = (value_type,) if isinstance(value_type, type) else tuple(value_type)
    if bool in types:
        return ControlKind.BOOLEAN
    if types == (int,):
        return ControlKind.INTEGER
    if any(item in (int, float) for item in types):
        return ControlKind.NUMBER
    if str in types:
        return ControlKind.TEXT
    raise TypeError("parameter value type has no standard UI control mapping")


__all__ = [
    "ControlKind",
    "ParameterControl",
    "parameter_controls",
    "plot_spec_controls",
]
