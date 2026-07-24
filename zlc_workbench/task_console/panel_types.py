"""Closed TaskConsole panel presentation vocabulary.

These values describe what the current renderer can actually consume.  They
contain no Qt objects, catalog lookup, runtime state, or shape guessing.
"""

from __future__ import annotations

from zlc_data.param_decl import ParamDecl


VIEW_SPEC_PARAM = "view_spec"
DEFAULT_GRID_INTENT_PARAM = "default_grid_intent"
DEFAULT_GRID_FACET_AXIS_PARAM = "default_grid_facet_axis"
HISTOGRAM_THRESHOLDS_PARAM = "histogram_thresholds"
HISTOGRAM_CELL_THRESHOLDS_PARAM = "histogram_cell_thresholds"
RELIM_MODES = ("tight", "normal", "fixed")
RELIM_PARAM = ParamDecl(
    key="relim",
    label="relim",
    kind="choice",
    default="tight",
    choices=RELIM_MODES,
    display=True,
    tooltip=(
        "Relim mode:\n"
        "  tight  = autoscale hugs the data\n"
        "  normal = autoscale with the matplotlib default margin\n"
        "  fixed  = pin the value range to the lo/hi controls"
    ),
)


def panel_view_intents():
    from zlc_frontend.figure import ViewIntent

    return {
        "2d": ViewIntent.IMAGE,
        "1d": ViewIntent.CURVE,
        "monitor": ViewIntent.CURVE,
        "hist": ViewIntent.HISTOGRAM,
    }


def grid_view_intents():
    from zlc_frontend.figure import ViewIntent

    return (
        ("Curves", ViewIntent.CURVE),
        ("Distribution", ViewIntent.HISTOGRAM),
        ("Images", ViewIntent.IMAGE),
    )


def automatic_panel_kind(schema) -> str | None:
    """Choose an initial ordinary panel from declared axis roles only.

    This is presentation policy, not an authoritative reduction.  It delegates
    the actual axis binding to ``suggest_view`` and never inspects ndarray rank,
    singleton lengths, or values.  A camera-like spatial plane and a genuine
    two-axis scan start as an image; a one-axis scan/spectrum starts as a curve.
    If neither public Figure contract can safely bind the schema, no panel is
    manufactured and the operator can make the missing view decision explicitly.
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
        ("2d", ViewIntent.IMAGE),
        ("1d", ViewIntent.CURVE),
    ) if spatial_plane or len(scan_axes) >= 2 else (
        ("1d", ViewIntent.CURVE),
        ("2d", ViewIntent.IMAGE),
    )
    for kind, intent in preferred:
        if suggest_view(schema, intent).spec is not None:
            return kind
    return None


def repeat_mode_label(mode) -> str:
    from zlc_frontend.figure import RepeatViewMode

    return {
        RepeatViewMode.MEAN: "Mean",
        RepeatViewMode.SUM: "Sum",
        RepeatViewMode.LATEST: "Latest repeat",
        RepeatViewMode.BATCH: "Overlay repeats",
        RepeatViewMode.SAMPLE: "Pool as samples",
        RepeatViewMode.FACET: "Facet repeats",
    }[mode]
