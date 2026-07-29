"""Canonical Plot Panel display forms and their authored-state mapping.

Curve, image, and histogram modules own the physical display values and their
validators.  This module owns only the typed PlotKind dispatch shared by every
Figure host.  It deliberately contains no Qt, Workbench layout, or second
display parameter vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from zlc_storage import exact_mapping

from .curve_display import (
    CurveDisplayState,
    curve_display_form_spec,
    curve_display_form_values,
    curve_display_from_form,
)
from .display_range import RelimMode, display_range_from_form
from .figure import GRID_INTENTS, ViewIntent
from .form import (
    FormFieldProps,
    FormSpec,
    choice_value_from_tree,
    choice_value_to_tree,
)
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramDisplayState,
    histogram_display_form_spec,
    histogram_display_form_values,
    histogram_display_from_form,
)
from .image_display import (
    ImageDisplayState,
    image_display_form_spec,
    image_display_form_values,
    image_display_from_form,
)
from .meter_display import MeterDisplayState
from .plot_kind import PlotKind


_ROLLING_DISTRIBUTION = FormFieldProps(
    key="show_dist",
    kind="bool",
    label="Side distribution",
    default=True,
    description="Show the side histogram beside the rolling trace",
)
_ROLLING_DISPLAY_FORM = FormSpec(
    curve_display_form_spec().fields + (_ROLLING_DISTRIBUTION,)
)
def _display_family_for_kind(
    kind: PlotKind,
    cell_intent: ViewIntent | None,
) -> ViewIntent | None:
    if not isinstance(kind, PlotKind):
        raise TypeError("panel display kind must be PlotKind")
    if kind is PlotKind.GRID:
        if cell_intent not in GRID_INTENTS:
            raise ValueError(
                "Grid display form requires CURVE, HISTOGRAM, or IMAGE cell intent"
            )
        return cell_intent
    if cell_intent is not None:
        raise ValueError("only Grid display forms accept a cell intent")
    return {
        PlotKind.CURVE: ViewIntent.CURVE,
        PlotKind.ROLLING: ViewIntent.CURVE,
        PlotKind.HISTOGRAM: ViewIntent.HISTOGRAM,
        PlotKind.IMAGE: ViewIntent.IMAGE,
        PlotKind.SITE_MAP: ViewIntent.IMAGE,
        PlotKind.METER: ViewIntent.METER,
    }.get(kind)


def panel_display_form_spec(
    kind: PlotKind,
    *,
    cell_intent: ViewIntent | None = None,
) -> FormSpec | None:
    """Return the sole canonical display form for one PlotKind surface."""

    intent = _display_family_for_kind(kind, cell_intent)
    if kind is PlotKind.ROLLING:
        return _ROLLING_DISPLAY_FORM
    if intent is ViewIntent.CURVE:
        return curve_display_form_spec()
    if intent is ViewIntent.HISTOGRAM:
        return histogram_display_form_spec()
    if intent is ViewIntent.IMAGE:
        return image_display_form_spec()
    if intent is ViewIntent.METER:
        return None
    raise ValueError(f"plot kind {kind.value!r} has no dataset display form")


def panel_display_form_labels(kind: PlotKind) -> tuple[str, ...]:
    """Return a stable label inventory without making Workbench know Grid fields."""

    return tuple(
        dict.fromkeys(
            field.label for field in _panel_display_fields(kind)
        )
    )


def panel_display_param_keys(kind: PlotKind) -> frozenset[str]:
    """Return every canonical display key that this PlotKind may persist."""

    return frozenset(field.key for field in _panel_display_fields(kind))


def _panel_display_fields(kind: PlotKind) -> tuple[FormFieldProps, ...]:
    """Return the canonical field union for one panel kind."""

    if kind is PlotKind.GRID:
        specs = tuple(
            panel_display_form_spec(kind, cell_intent=intent)
            for intent in GRID_INTENTS
        )
    else:
        specs = (panel_display_form_spec(kind),)
    return tuple(
        field
        for spec in specs
        if spec is not None
        for field in spec.fields
    )


def panel_display_form_values_from_tree(
    kind: PlotKind,
    params: Mapping[str, object],
    *,
    cell_intent: ViewIntent | None = None,
) -> dict[str, object]:
    """Decode only the selected canonical FormSpec from a saved panel record."""

    if not isinstance(params, Mapping):
        raise TypeError("panel display params must be a mapping")
    spec = panel_display_form_spec(kind, cell_intent=cell_intent)
    if spec is None:
        return {}
    values = spec.default_values()
    for field in spec.fields:
        if field.key not in params:
            continue
        raw = params[field.key]
        values[field.key] = (
            choice_value_from_tree(field, raw)
            if field.kind == "choice"
            else raw
        )
    return values


def panel_display_form_values_to_tree(
    kind: PlotKind,
    values: Mapping[str, object],
    *,
    cell_intent: ViewIntent | None = None,
) -> dict[str, object]:
    """Encode one complete owner-validated display form into current scalars."""

    spec = panel_display_form_spec(kind, cell_intent=cell_intent)
    if spec is None:
        if values:
            raise ValueError("METER has no display form values")
        return {}
    exact = exact_mapping(
        values,
        frozenset(spec.keys),
        "panel display form",
        discriminator=None,
    )
    fields = {field.key: field for field in spec.fields}
    return {
        key: (
            choice_value_to_tree(fields[key], value)
            if fields[key].kind == "choice"
            else value
        )
        for key, value in exact.items()
    }


def panel_display_state_intent(state: object) -> ViewIntent:
    """Return the canonical display family for one typed display state."""

    if isinstance(state, ImageDisplayState):
        return ViewIntent.IMAGE
    if isinstance(state, CurveDisplayState):
        return ViewIntent.CURVE
    if isinstance(state, (HistogramDisplayState, FacetedHistogramDisplayState)):
        return ViewIntent.HISTOGRAM
    if isinstance(state, MeterDisplayState):
        return ViewIntent.METER
    raise TypeError("unknown Plot Panel display state")


def panel_display_form_values(
    kind: PlotKind,
    state: object,
    *,
    rolling_distribution: bool = False,
) -> dict[str, object]:
    """Project one authored display state through its canonical FormSpec."""

    intent = panel_display_state_intent(state)
    expected = _display_family_for_kind(
        kind,
        intent if kind is PlotKind.GRID else None,
    )
    if expected is not intent:
        raise ValueError("PlotKind and display state intent disagree")
    if isinstance(state, FacetedHistogramDisplayState):
        state = state.display
    if isinstance(state, ImageDisplayState):
        values = image_display_form_values(state)
    elif isinstance(state, CurveDisplayState):
        values = curve_display_form_values(state)
    elif isinstance(state, HistogramDisplayState):
        values = histogram_display_form_values(state)
    elif isinstance(state, MeterDisplayState):
        return {}
    else:
        raise TypeError("unknown Plot Panel display state")
    if kind is PlotKind.ROLLING:
        if not isinstance(rolling_distribution, bool):
            raise TypeError("rolling_distribution must be bool")
        values["show_dist"] = rolling_distribution
    return values


def panel_display_state_from_form(
    kind: PlotKind,
    base: object,
    values: Mapping[str, object],
    *,
    current_value_limits: tuple[float, float] | None,
    rolling_distribution: bool = False,
) -> tuple[object, bool]:
    """Apply the selected owner handler; Grid preserves per-cell thresholds."""

    intent = panel_display_state_intent(base)
    spec = panel_display_form_spec(
        kind,
        cell_intent=intent if kind is PlotKind.GRID else None,
    )
    if spec is None:
        raise ValueError("METER has no authored display form")
    exact = exact_mapping(
        values,
        frozenset(spec.keys),
        "panel display form",
        discriminator=None,
    )
    next_rolling = rolling_distribution
    if kind is PlotKind.ROLLING:
        next_rolling = exact.pop("show_dist")
        if not isinstance(next_rolling, bool):
            raise TypeError("show_dist form value must be bool")
    if isinstance(base, FacetedHistogramDisplayState):
        display = histogram_display_from_form(
            base.display,
            exact,
            current_count_limits=current_value_limits,
        )
        return FacetedHistogramDisplayState(display, base.cell_thresholds), next_rolling
    if isinstance(base, ImageDisplayState):
        return (
            image_display_from_form(
                base,
                exact,
                current_color_limits=current_value_limits,
            ),
            next_rolling,
        )
    if isinstance(base, CurveDisplayState):
        return (
            curve_display_from_form(
                base,
                exact,
                current_y_limits=current_value_limits,
            ),
            next_rolling,
        )
    if isinstance(base, HistogramDisplayState):
        return (
            histogram_display_from_form(
                base,
                exact,
                current_count_limits=current_value_limits,
            ),
            next_rolling,
        )
    raise TypeError("unknown authored Plot Panel display state")


def _submitted_fixed_range(
    intent: ViewIntent,
    values: Mapping[str, object],
) -> tuple[float, float] | None:
    keys = {
        ViewIntent.CURVE: ("y_min", "y_max"),
        ViewIntent.HISTOGRAM: ("count_min", "count_max"),
        ViewIntent.IMAGE: ("color_min", "color_max"),
    }.get(intent)
    if keys is None:
        return None
    return display_range_from_form(values, keys[0], keys[1], "fixed display range")


def panel_display_state_from_params(
    kind: PlotKind,
    params: Mapping[str, object],
    *,
    revision: int,
    cell_intent: ViewIntent | None = None,
    focus=None,
    thresholds: tuple[float, ...] = (),
    cell_thresholds=(),
    home_view: bool = False,
) -> object:
    """Resolve saved current-form scalars into the sole authored state type."""

    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("panel display revision must be a nonnegative integer")
    if not isinstance(home_view, bool):
        raise TypeError("home_view must be bool")
    intent = _display_family_for_kind(kind, cell_intent)
    if intent is ViewIntent.METER:
        return MeterDisplayState(
            0 if focus is None else int(focus.panel_index),
            None if focus is None else focus.address,
            revision,
        )
    if intent not in GRID_INTENTS:
        raise ValueError("plot kind has no authored Dataset display state")
    values = panel_display_form_values_from_tree(
        kind,
        params,
        cell_intent=cell_intent,
    )
    rolling = bool(values.get("show_dist", False))
    if intent is ViewIntent.CURVE:
        base: object = CurveDisplayState()
    elif intent is ViewIntent.HISTOGRAM:
        base = HistogramDisplayState()
    else:
        base = ImageDisplayState()
    fixed = (
        _submitted_fixed_range(intent, values)
        if values["relim_mode"] is RelimMode.FIXED
        else None
    )
    state, _rolling = panel_display_state_from_form(
        kind,
        base,
        values,
        current_value_limits=fixed,
        rolling_distribution=rolling,
    )
    if isinstance(state, CurveDisplayState):
        state = replace(
            state,
            revision=revision,
            x_view=None if home_view else state.x_view,
        )
    elif isinstance(state, HistogramDisplayState):
        state = replace(
            state,
            revision=revision,
            x_view=None if home_view else state.x_view,
            thresholds=tuple(thresholds),
        )
        if kind is PlotKind.GRID:
            state = FacetedHistogramDisplayState(state, tuple(cell_thresholds))
    elif isinstance(state, ImageDisplayState):
        state = replace(
            state,
            revision=revision,
            x_view=None if home_view else state.x_view,
            y_view=None if home_view else state.y_view,
        )
    return state


def panel_display_value_range_keys(state: object) -> tuple[str, str] | None:
    """Return the canonical authored value-limit fields for one state."""

    if isinstance(state, ImageDisplayState):
        return ("color_min", "color_max")
    if isinstance(state, CurveDisplayState):
        return ("y_min", "y_max")
    if isinstance(state, (HistogramDisplayState, FacetedHistogramDisplayState)):
        return ("count_min", "count_max")
    if isinstance(state, MeterDisplayState):
        return None
    raise TypeError("unknown Plot Panel display state")


__all__ = [
    "panel_display_form_labels",
    "panel_display_form_spec",
    "panel_display_form_values",
    "panel_display_form_values_from_tree",
    "panel_display_form_values_to_tree",
    "panel_display_param_keys",
    "panel_display_state_from_form",
    "panel_display_state_from_params",
    "panel_display_state_intent",
    "panel_display_value_range_keys",
]
