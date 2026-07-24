"""Presentation policy shared by live Figure panels.

The values in this module describe frontend controls and initial views.  They
do not own a desktop window, Qt object, runtime signal, or authoritative data
reduction, so every Workbench shell consumes them through :mod:`zlc_frontend`
instead of restating them locally.
"""

from __future__ import annotations

from .form import FormChoice, FormFieldProps


VIEW_SPEC_PARAM = "view_spec"
HISTOGRAM_THRESHOLDS_PARAM = "histogram_thresholds"
HISTOGRAM_CELL_THRESHOLDS_PARAM = "histogram_cell_thresholds"
RELIM_MODES = ("tight", "normal", "fixed")
RELIM_PARAM = FormFieldProps(
    key="relim",
    kind="choice",
    label="relim",
    default="tight",
    choices=tuple(FormChoice(value, value) for value in RELIM_MODES),
    description=(
        "Relim mode:\n"
        "  tight  = autoscale hugs the data\n"
        "  normal = autoscale with the matplotlib default margin\n"
        "  fixed  = pin the value range to the lo/hi controls"
    ),
)


def panel_view_intents():
    """Return the ordinary panel-key to typed Figure-intent mapping."""

    from zlc_frontend.figure import ViewIntent

    return {
        "2d": ViewIntent.IMAGE,
        "1d": ViewIntent.CURVE,
        "monitor": ViewIntent.CURVE,
        "hist": ViewIntent.HISTOGRAM,
    }


def grid_view_intents():
    """Return the choices offered by a grid panel's view selector."""

    from zlc_frontend.figure import GRID_INTENTS, ViewIntent

    # Keep Main's terse product vocabulary while the typed Figure contract is
    # the sole source of which intents are legal.  These are labels, not a
    # second plot-kind registry.
    labels = {
        ViewIntent.CURVE: "1d",
        ViewIntent.HISTOGRAM: "hist",
        ViewIntent.IMAGE: "2d",
    }
    return tuple((labels[intent], intent) for intent in GRID_INTENTS)


def automatic_panel_kind(schema) -> str | None:
    """Choose an initial ordinary panel from declared axis roles only.

    This is presentation policy, not an authoritative reduction.  It delegates
    axis binding to :func:`zlc_frontend.figure.suggest_view` and never inspects
    ndarray rank, singleton lengths, or values.  A spatial plane or genuine
    two-axis scan starts as an image; a one-axis scan/spectrum starts as a
    curve.  If neither public Figure contract binds safely, no panel is
    manufactured and the missing view decision remains explicit.
    """

    from zlc_data import SCAN_POINT, SPATIAL_X, SPATIAL_Y, SPECTRAL
    from zlc_frontend.figure import ViewIntent, suggest_view

    axes = tuple(schema.point_axes) + tuple(schema.cell_schema.data_axes)
    roles = tuple(axis.role for axis in axes)
    spatial_plane = roles.count(SPATIAL_X) == 1 and roles.count(SPATIAL_Y) == 1
    scan_axes = tuple(
        axis for axis in schema.point_axes if axis.role in (SCAN_POINT, SPECTRAL)
    )
    preferred = (
        (("2d", ViewIntent.IMAGE), ("1d", ViewIntent.CURVE))
        if spatial_plane or len(scan_axes) >= 2
        else (("1d", ViewIntent.CURVE), ("2d", ViewIntent.IMAGE))
    )
    for kind, intent in preferred:
        if suggest_view(schema, intent).spec is not None:
            return kind
    return None


def repeat_mode_label(mode) -> str:
    """Return the operator-facing label for a typed repeat view mode."""

    from zlc_frontend.figure import RepeatViewMode

    return {
        RepeatViewMode.MEAN: "Mean",
        RepeatViewMode.SUM: "Sum",
        RepeatViewMode.LATEST: "Latest repeat",
        RepeatViewMode.BATCH: "Overlay repeats",
        RepeatViewMode.SAMPLE: "Pool as samples",
        RepeatViewMode.FACET: "Facet repeats",
    }[mode]


__all__ = [
    "HISTOGRAM_CELL_THRESHOLDS_PARAM",
    "HISTOGRAM_THRESHOLDS_PARAM",
    "RELIM_MODES",
    "RELIM_PARAM",
    "VIEW_SPEC_PARAM",
    "automatic_panel_kind",
    "grid_view_intents",
    "panel_view_intents",
    "repeat_mode_label",
]
