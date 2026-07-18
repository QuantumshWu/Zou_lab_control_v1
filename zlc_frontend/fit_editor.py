"""Headless presentation projection for typed fit constraints.

``zlc_data`` remains the sole owner of model, axis, and constraint semantics.
This module only projects one already-bound fit into the shared scalar-form
contract and rebuilds a validated ``FitSpec`` from an exact form state.  It
contains no Qt, repository, execution, display selection, or persistence
authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from zlc_data import (
    BoundFit,
    FitParameterConstraint,
    FitSpec,
    bind_fit,
)

from .authority import describe_authoritative_transform
from .form import FormFieldProps, FormSpec


_CONSTRAINT_FIELDS = ("initial", "lower", "upper", "fixed")
_CONSTRAINT_DESCRIPTIONS = {
    "initial": "Optional solver seed; blank uses the model initializer.",
    "lower": "Optional inclusive lower bound.",
    "upper": "Optional inclusive upper bound.",
    "fixed": "Optional fixed value; if initial is also set it must agree.",
}


def fit_axis_summary(bound: BoundFit) -> str:
    """Describe the exact named fit/batch-axis split without reducing an axis."""

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")

    def describe(axis_id) -> str:
        axis = bound.effective_schema.axis(axis_id)
        unit = f" {axis.unit}" if axis.unit else ""
        return (
            f"{axis.name} ({axis.axis_id}) "
            f"[{axis.role.value}; size={axis.size}]{unit}"
        )

    fit_axes = ", ".join(describe(axis_id) for axis_id in bound.spec.fit_axis_ids)
    batch_axes = ", ".join(
        describe(axis_id) for axis_id in bound.spec.batch_axis_ids
    )
    return f"fit axes: {fit_axes} · batch axes: {batch_axes or 'none'}"


def fit_authority_summary(bound: BoundFit) -> str:
    """Describe the immutable transform authority carried by one bound Fit."""

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    transform = bound.spec.committed_transform
    return describe_authoritative_transform(
        None if transform is None else transform.spec
    )


def fit_constraint_form(bound: BoundFit) -> FormSpec:
    """Project the catalog-owned parameter metadata into the shared form DSL."""

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    fields = []
    constraints = {
        constraint.parameter_name: constraint
        for constraint in bound.spec.constraints
    }
    for parameter, unit in zip(
        bound.parameter_definitions,
        bound.parameter_units,
        strict=True,
    ):
        constraint = constraints.get(parameter.name)
        for field in _CONSTRAINT_FIELDS:
            fields.append(
                FormFieldProps(
                    f"{parameter.name}.{field}",
                    "float",
                    f"{parameter.name} {field}",
                    default=(
                        None if constraint is None else getattr(constraint, field)
                    ),
                    unit=unit,
                    description=(
                        f"{_CONSTRAINT_DESCRIPTIONS[field]} "
                        f"Domain: {parameter.domain.value}."
                    ),
                )
            )
    return FormSpec(tuple(fields))


def fit_spec_from_form(
    bound: BoundFit,
    values: Mapping[str, object],
) -> FitSpec:
    """Rebuild and domain-validate one authority-bearing fit request.

    The mapping must contain the exact projected keys.  In particular, no
    current viewport, display reduction, or selector value is consulted here.
    """

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    form = fit_constraint_form(bound)
    supplied = set(values)
    expected = set(form.keys)
    if supplied != expected:
        missing = tuple(sorted(expected - supplied))
        extra = tuple(sorted(supplied - expected))
        raise ValueError(
            f"fit constraint values require exact keys; missing={missing!r}, "
            f"extra={extra!r}"
        )

    constraints = []
    for parameter in bound.parameter_definitions:
        fields = {
            name: values[f"{parameter.name}.{name}"]
            for name in _CONSTRAINT_FIELDS
        }
        if any(value is not None for value in fields.values()):
            constraints.append(FitParameterConstraint(parameter.name, **fields))
    candidate = replace(bound.spec, constraints=tuple(constraints))
    return bind_fit(candidate, bound.expected_schema).spec


__all__ = [
    "fit_authority_summary",
    "fit_axis_summary",
    "fit_constraint_form",
    "fit_spec_from_form",
]
