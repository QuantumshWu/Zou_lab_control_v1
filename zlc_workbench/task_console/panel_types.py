"""Closed TaskConsole panel presentation vocabulary.

These values describe what the current renderer can actually consume.  They
contain no Qt objects, catalog lookup, runtime state, or shape guessing.
"""

from __future__ import annotations

from zlc_data.param_decl import ParamDecl


VIEW_SPEC_PARAM = "view_spec"
HISTOGRAM_THRESHOLDS_PARAM = "histogram_thresholds"
HISTOGRAM_CELL_THRESHOLDS_PARAM = "histogram_cell_thresholds"
RELIM_MODES = ("tight", "normal", "fixed")
RELIM_PARAM = ParamDecl(
    key="relim",
    label="relim",
    kind="choice",
    default="normal",
    choices=RELIM_MODES,
    display=True,
    tooltip=(
        "Relim mode:\n"
        "  tight  = autoscale hugs the data\n"
        "  normal = autoscale, holding the window until data leaves it\n"
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
