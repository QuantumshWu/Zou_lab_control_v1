"""Headless projection between bound Fit requests and their text editor.

``zlc_data`` remains the sole owner of model, axis, and constraint semantics.
This module exposes a small, reversible authoring value for the Figure UI and
rebuilds a validated ``FitSpec`` from the one visible arguments line.  It
contains no Qt, repository, execution, display selection, or persistence
authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from zlc_data import (
    AxisId,
    BoundFit,
    FitSpec,
    Selection,
)

from ._fit_arguments import format_fit_arguments, parse_fit_arguments
from .authority import describe_authoritative_transform
from .data_figure import DataFigure
from .figure import AxisViewRole, ViewIntent


@dataclass(frozen=True, slots=True)
class FitAuthoringOption:
    """Small presentation projection of a worker-bound Fit request."""

    spec: FitSpec
    display_name: str
    parameter_names: tuple[str, ...]
    argument_text: str
    fit_axis_roles: tuple[object, ...]
    batch_axis_sizes: tuple[tuple[object, int], ...]
    axis_summary: str
    authority_summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FitSpec):
            raise TypeError("Fit authoring option requires FitSpec")
        if not self.display_name:
            raise ValueError("Fit authoring display_name must be non-empty")
        if (
            not self.parameter_names
            or len(set(self.parameter_names)) != len(self.parameter_names)
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in self.parameter_names
            )
        ):
            raise ValueError(
                "Fit authoring parameter names must be unique identifiers"
            )
        if not isinstance(self.argument_text, str):
            raise TypeError("Fit authoring argument_text must be text")
        # The prefilled text is part of the reversible presentation contract.
        # Validate it here so an invalid option never reaches either Qt host.
        parse_fit_arguments(self.argument_text, self.parameter_names)
        if len(self.fit_axis_roles) != len(self.spec.fit_axis_ids):
            raise ValueError("Fit authoring roles differ from its fit axes")
        if tuple(axis_id for axis_id, _size in self.batch_axis_sizes) != (
            self.spec.batch_axis_ids
        ) or any(size <= 0 for _axis_id, size in self.batch_axis_sizes):
            raise ValueError("Fit authoring batch sizes differ from its batch axes")
        if not self.axis_summary or not self.authority_summary:
            raise ValueError("Fit authoring summaries must be non-empty")


def fit_projection_metadata(
    figure: DataFigure,
    intent: ViewIntent,
) -> tuple[tuple[AxisId, ...], tuple[tuple[AxisId, AxisViewRole], ...]]:
    """Project one Figure's declared view roles into exact Fit axes.

    Both DataFigure windows and embedded TaskConsole panels call this owner;
    neither GUI shell is allowed to reinterpret rank, shape, X/Y, or batch
    roles for itself.
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("fit projection requires DataFigure")
    if not isinstance(intent, ViewIntent):
        raise TypeError("fit projection requires ViewIntent")
    if len(figure.document.layers) != 1:
        raise ValueError("Fit projection requires exactly one Figure layer")
    layer = figure.document.layers[0]
    roles = tuple(
        sorted(
            (
                (binding.axis_id, binding.role)
                for binding in layer.view.axis_bindings
            ),
            key=lambda item: item[0].value,
        )
    )
    if intent is ViewIntent.CURVE:
        fit_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.X
        )
    elif intent is ViewIntent.IMAGE:
        x_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.IMAGE_X
        )
        y_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.IMAGE_Y
        )
        fit_axes = (*x_axes, *y_axes)
    else:
        fit_axes = ()
    expected = (
        1
        if intent is ViewIntent.CURVE
        else 2
        if intent is ViewIntent.IMAGE
        else 0
    )
    if len(fit_axes) != expected:
        raise ValueError("typed figure has ambiguous fitted display axes")
    return fit_axes, roles


def validate_fit_authoring_options(
    options: tuple[FitAuthoringOption, ...],
    *,
    fit_axis_ids: tuple[AxisId, ...],
    axis_roles: tuple[tuple[AxisId, AxisViewRole], ...],
    selection: Selection | None,
    allow_prepared_transform: bool = False,
) -> tuple[FitAuthoringOption, ...]:
    """Keep only results that can map back onto the exact visible Figure."""

    prepared_options = tuple(options)
    if not prepared_options or any(
        not isinstance(option, FitAuthoringOption)
        for option in prepared_options
    ):
        raise ValueError("Fit preparation produced no FitAuthoringOption")
    if any(not isinstance(axis_id, AxisId) for axis_id in fit_axis_ids):
        raise TypeError("fit_axis_ids must contain AxisId values")
    if any(
        not isinstance(axis_id, AxisId)
        or not isinstance(role, AxisViewRole)
        for axis_id, role in axis_roles
    ):
        raise TypeError("axis_roles must contain AxisId/AxisViewRole pairs")
    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("selection must be Selection or None")

    role_by_axis = dict(axis_roles)
    accepted_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
        AxisViewRole.SLIDER,
    }
    prepared = []
    for option in prepared_options:
        if option.spec.fit_axis_ids != fit_axis_ids:
            continue
        batch_sizes = dict(option.batch_axis_sizes)

        def batch_axis_is_replayable(axis_id: AxisId) -> bool:
            role = role_by_axis.get(axis_id)
            if role in accepted_batch_roles:
                return True
            return bool(
                role is AxisViewRole.REDUCED
                and batch_sizes[axis_id] == 1
            )

        if any(
            not batch_axis_is_replayable(axis_id)
            for axis_id in option.spec.batch_axis_ids
        ):
            continue
        transform = option.spec.committed_transform
        if selection is None:
            if transform is not None and not allow_prepared_transform:
                continue
        else:
            if transform is None:
                continue
            if tuple(transform.spec.operations) != (selection,):
                continue
        prepared.append(option)
    if not prepared:
        raise ValueError(
            "the visible panel cannot map an authoritative Fit result without "
            "reducing or guessing a named batch axis"
        )
    return tuple(prepared)


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


def fit_authoring_option(bound: BoundFit) -> FitAuthoringOption:
    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    axis_summary = fit_axis_summary(bound)
    authority_summary = fit_authority_summary(bound)
    parameter_names = tuple(
        parameter.name for parameter in bound.parameter_definitions
    )
    argument_text = format_fit_arguments(
        bound.spec.constraints,
        parameter_names,
    )
    return FitAuthoringOption(
        bound.spec,
        bound.model.display_name,
        parameter_names,
        argument_text,
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
    )


def fit_spec_from_arguments(
    option: FitAuthoringOption,
    arguments: str,
) -> FitSpec:
    """Parse and domain-validate one authority-bearing Fit request.

    No current viewport, display reduction, or selector value is consulted.
    Empty text means automatic model initialization and domains.
    """

    if not isinstance(option, FitAuthoringOption):
        raise TypeError("option must be FitAuthoringOption")
    constraints = parse_fit_arguments(arguments, option.parameter_names)
    return replace(option.spec, constraints=constraints)


__all__ = [
    "FitAuthoringOption",
    "fit_authoring_option",
    "fit_authority_summary",
    "fit_axis_summary",
    "fit_projection_metadata",
    "fit_spec_from_arguments",
    "validate_fit_authoring_options",
]
