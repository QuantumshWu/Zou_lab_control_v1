"""Current TaskConsole panel parameters that directly affect rendering.

This catalog is deliberately small.  A declaration belongs here only when the
current panel renderer consumes the same key.  Fit authoring is the typed
``FitSpec`` flow exposed by DataFigure, not a histogram display toggle; fields
without a current renderer consumer are not saved-layout vocabulary.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .form import FormFieldProps, FormSpec, choice_value_from_tree
from .histogram_display import histogram_display_form_spec
from .image_display import image_display_form_spec

__all__ = ["PANEL_PARAMS", "panel_param_decls", "resolved_panel_param"]


def _canonical_field(spec: FormSpec, key: str) -> FormFieldProps:
    matches = tuple(field for field in spec.fields if field.key == key)
    if len(matches) != 1:
        raise RuntimeError(f"canonical form has no unique field {key!r}")
    return matches[0]


_IMAGE_COLORMAP = _canonical_field(image_display_form_spec(), "colormap")
_HISTOGRAM_BIN_COUNT = _canonical_field(
    histogram_display_form_spec(),
    "bin_count",
)
_HISTOGRAM_COUNT_SCALE = _canonical_field(
    histogram_display_form_spec(),
    "count_scale",
)


PANEL_PARAMS: Mapping[str, tuple[FormFieldProps, ...]] = MappingProxyType({
    "2d": (
        _IMAGE_COLORMAP,
    ),
    "sites": (
        _IMAGE_COLORMAP,
    ),
    "monitor": (
        FormFieldProps(
            key="show_dist",
            kind="bool",
            label="side distribution",
            default=True,
            description="Show the side distribution histogram beside the rolling trace",
        ),
    ),
    "hist": (
        _HISTOGRAM_BIN_COUNT,
        _HISTOGRAM_COUNT_SCALE,
    ),
})


def panel_param_decls(param_kind: str) -> tuple[FormFieldProps, ...]:
    """Return the exact render-consumed parameter declarations for one kind."""

    return PANEL_PARAMS.get(str(param_kind), ())


def resolved_panel_param(
    param_kind: str,
    params: Mapping[str, object],
    key: str,
) -> object:
    """Resolve one declared key from persisted values or its declaration."""

    declarations = panel_param_decls(param_kind)
    try:
        declaration = next(item for item in declarations if item.key == key)
    except StopIteration as error:
        raise KeyError(f"panel kind {param_kind!r} has no parameter {key!r}") from error
    if key not in params:
        return declaration.default
    value = params[key]
    return (
        choice_value_from_tree(declaration, value)
        if declaration.kind == "choice"
        else value
    )
