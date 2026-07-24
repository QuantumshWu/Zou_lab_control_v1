"""The frontend plot-kind vocabulary: which views exist and what each accepts.

A plot kind's headless vocabulary is its key, human label, fitting family, whether it is
offered in Add Panel, and accepted value shape.  Rendering classes and display controls
belong to the frontend renderer that actually consumes them.

The console builds its menu and input-format text from these words; saved-figure readers
validate stored kinds against them.  Neither needs to import a renderer to read a string.

This table deliberately contains no display controls.  A control belongs to the typed
renderer that consumes it; putting an unconsumed ``repeat_mode`` or expression knob in
the vocabulary creates a convincing UI that does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PLOT_KIND_SPECS", "PLOT_KIND_SPEC_BY_KEY", "PlotKindSpec"]


@dataclass(frozen=True)
class PlotKindSpec:
    """ONE declarative record per plot kind -- everything about a kind EXCEPT its renderer.

    Fields
    ------
    key            the canonical kind string (``"1d"``, ``"2d"``, ...).
    label          the human Add-Panel / panel-title label.
    panel          True if this kind is offered in the live console ADD-PANEL
                   dropdown (you add a blank panel of it and wire it to a signal).
                   False retains the kind in the renderer/saved-figure vocabulary
                   without advertising an unfinished live product.
    input_format   one-line description of the accepted ``value`` shape.
    A live panel binds exactly one typed dataset.  Multi-input transforms are
    explicit Processor outputs or joined datasets, not arbitrary GUI expressions.
    """

    key: str
    label: str
    panel: bool = True
    input_format: str = ""


#: The ONE plot-kind table, in Add-Panel MENU order.  ``monitor`` names the DEFAULT rolling
#: variant (the side distribution is a toggle on the same kind, not a second kind).
#:
#: This literal is COMPLETE: every kind is here, ``grid`` included.  The render layer pairs
#: each spec with a class, and does so in two steps only because ``GridPlot`` is defined
#: later in that module than the table -- a fact about Python's execution order in ONE file,
#: which is why the deferral lives there and the vocabulary here stays whole.
PLOT_KIND_SPECS: tuple[PlotKindSpec, ...] = (
    PlotKindSpec(
        key="2d", label="2D image",
        input_format="value must be a 2D array / camera frame (H×W)",
    ),
    PlotKindSpec(
        key="sites", label="Site map", panel=True,
        input_format=(
            "value must carry a typed calibration site map, or an exact "
            "single-cell occupancy view with its same-shot frame and "
            "admitted calibration geometry"),
    ),
    PlotKindSpec(
        key="1d", label="1D vector",
        input_format="value must be a 1D vector (N,) or per-site array",
    ),
    PlotKindSpec(
        key="monitor", label="Rolling trace",
        input_format="value must be a scalar per shot (rolling trace)",
    ),
    PlotKindSpec(
        key="hist", label="Distribution",
        input_format="value must be a 1D sample vector",
    ),
    # Static timing diagram -- not a blank live-console panel.
    PlotKindSpec(key="pulse", label="Pulse sequence", panel=False),
    # GRID is a TaskConsole layout over one typed dataset ViewSpec.  Its exact
    # named FACET bindings are evaluated by the ordinary FigureEvaluator; it
    # does not own a second shape-driven slicer or renderer.
    PlotKindSpec(
        key="grid",
        label="Site grid",
        panel=True,
        input_format=(
            "value must admit an explicit named-axis CURVE, HISTOGRAM, or "
            "IMAGE facet view"
        ),
    ),
)

#: ``key -> PlotKindSpec`` for O(1) lookup.  Insertion order = menu order.
PLOT_KIND_SPEC_BY_KEY: dict[str, PlotKindSpec] = {spec.key: spec for spec in PLOT_KIND_SPECS}
