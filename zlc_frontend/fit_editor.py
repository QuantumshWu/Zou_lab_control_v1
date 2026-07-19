"""Headless presentation projection for typed fit constraints.

``zlc_data`` remains the sole owner of model, axis, and constraint semantics.
This module only projects one already-bound fit into the shared scalar-form
contract and rebuilds a validated ``FitSpec`` from an exact form state.  It
contains no Qt, repository, execution, display selection, or persistence
authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from zlc_data import (
    BoundFit,
    FitParameterConstraint,
    FitSpec,
    Selection,
    fit_binding_retained_upper_bound_nbytes,
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


@dataclass(frozen=True, slots=True)
class FitAuthoringOption:
    """Small presentation projection of a worker-bound Fit request."""

    spec: FitSpec
    display_name: str
    constraint_form: FormSpec
    parameter_names: tuple[str, ...]
    fit_axis_roles: tuple[object, ...]
    batch_axis_sizes: tuple[tuple[object, int], ...]
    axis_summary: str
    authority_summary: str
    retained_upper_bound_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FitSpec):
            raise TypeError("Fit authoring option requires FitSpec")
        if not self.display_name:
            raise ValueError("Fit authoring display_name must be non-empty")
        if not isinstance(self.constraint_form, FormSpec):
            raise TypeError("Fit authoring constraint_form must be FormSpec")
        if len(self.parameter_names) * len(_CONSTRAINT_FIELDS) != len(
            self.constraint_form.fields
        ):
            raise ValueError("Fit authoring form differs from its parameter inventory")
        if len(self.fit_axis_roles) != len(self.spec.fit_axis_ids):
            raise ValueError("Fit authoring roles differ from its fit axes")
        if tuple(axis_id for axis_id, _size in self.batch_axis_sizes) != (
            self.spec.batch_axis_ids
        ) or any(size <= 0 for _axis_id, size in self.batch_axis_sizes):
            raise ValueError("Fit authoring batch sizes differ from its batch axes")
        if not self.axis_summary or not self.authority_summary:
            raise ValueError("Fit authoring summaries must be non-empty")
        if self.retained_upper_bound_bytes <= 0:
            raise ValueError("Fit authoring retained bound must be positive")


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


def fit_authoring_option_additional_peak_upper_bound_nbytes(
    bound: BoundFit,
) -> int:
    """Gate GUI DTO/form text construction before any summary is formatted."""

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    # Binding retention includes two complete schema metadata envelopes.  Four
    # times that owner bound dominates the six-copy selected Qt text envelope,
    # every catalog parameter field, and the bounded authority formatter while
    # the BoundFit itself is still live.  fit_authoring_option enforces this
    # domination as a postcondition so future formatter growth fails closed.
    return 4 * fit_binding_retained_upper_bound_nbytes(
        bound.spec,
        bound.expected_schema,
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


def fit_authoring_option(bound: BoundFit) -> FitAuthoringOption:
    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    preflight_bound = fit_authoring_option_additional_peak_upper_bound_nbytes(bound)
    form = fit_constraint_form(bound)
    axis_summary = fit_axis_summary(bound)
    authority_summary = fit_authority_summary(bound)
    parameter_names = tuple(
        parameter.name for parameter in bound.parameter_definitions
    )
    spec_text_characters = (
        len(bound.spec.input_schema_fingerprint)
        + len(bound.spec.model_id)
        + sum(len(axis_id.value) for axis_id in bound.spec.fit_axis_ids)
        + sum(len(axis_id.value) for axis_id in bound.spec.batch_axis_ids)
        + sum(
            len(constraint.parameter_name)
            for constraint in bound.spec.constraints
        )
    )
    transform_item_count = 0
    transform = bound.spec.committed_transform
    if transform is not None:
        spec_text_characters += len(transform.input_schema_fingerprint) + len(
            transform.output_schema_fingerprint
        )
        for operation in transform.spec.operations:
            if isinstance(operation, Selection):
                transform_item_count += len(operation.terms)
                for term in operation.terms:
                    spec_text_characters += len(term.axis_id.value)
                    frame = getattr(term, "coordinate_frame", None)
                    if frame is not None:
                        spec_text_characters += len(frame.value)
            else:
                transform_item_count += len(operation.axis_ids)
                spec_text_characters += sum(
                    len(axis_id.value) for axis_id in operation.axis_ids
                )
    retained = (
        64 * 1024
        + 4096 * len(form.fields)
        # A selected option's Python strings coexist with Qt QString copies in
        # the combo/form/summary labels and with short-lived ``.text()`` /
        # ``setText`` handoff values.  During replacement the old main summary
        # can coexist with all four new representations, so six Unicode-width
        # copies form the hard frontend envelope.  Counting only the headless
        # DTO would make a long
        # non-BMP axis label escape the aggregate GUI budget.
        + 24 * (
            len(bound.model.display_name)
            + len(axis_summary)
            + len(authority_summary)
            + sum(len(name) for name in parameter_names)
            + sum(
                len(field.key)
                + len(field.label)
                + len(field.unit)
                + len(field.description)
                + sum(
                    len(choice.label)
                    + (len(choice.value) if isinstance(choice.value, str) else 0)
                    for choice in field.choices
                )
                for field in form.fields
            )
            + spec_text_characters
        )
        + 1024 * (
            len(bound.spec.fit_axis_ids)
            + len(bound.spec.batch_axis_ids)
            + len(bound.spec.constraints)
            + transform_item_count
        )
    )
    if retained > preflight_bound:
        raise RuntimeError(
            "Fit authoring option exceeded its data-free construction bound"
        )
    return FitAuthoringOption(
        bound.spec,
        bound.model.display_name,
        form,
        parameter_names,
        tuple(
            bound.effective_schema.axis(axis_id).role
            for axis_id in bound.spec.fit_axis_ids
        ),
        tuple(
            (axis_id, bound.effective_schema.axis(axis_id).size)
            for axis_id in bound.spec.batch_axis_ids
        ),
        axis_summary,
        authority_summary,
        preflight_bound,
    )


def fit_spec_from_form(
    option: FitAuthoringOption,
    values: Mapping[str, object],
) -> FitSpec:
    """Rebuild and domain-validate one authority-bearing fit request.

    The mapping must contain the exact projected keys.  In particular, no
    current viewport, display reduction, or selector value is consulted here.
    """

    if not isinstance(option, FitAuthoringOption):
        raise TypeError("option must be FitAuthoringOption")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    form = option.constraint_form
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
    for parameter_name in option.parameter_names:
        fields = {
            name: values[f"{parameter_name}.{name}"]
            for name in _CONSTRAINT_FIELDS
        }
        if any(value is not None for value in fields.values()):
            constraints.append(FitParameterConstraint(parameter_name, **fields))
    return replace(option.spec, constraints=tuple(constraints))


__all__ = [
    "FitAuthoringOption",
    "fit_authoring_option_additional_peak_upper_bound_nbytes",
    "fit_authoring_option",
    "fit_authority_summary",
    "fit_axis_summary",
    "fit_constraint_form",
    "fit_spec_from_form",
]
