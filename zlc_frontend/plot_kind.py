"""The sole typed vocabulary for every frontend Figure surface.

The console builds its menu and input-format text from this table.  Persistence
encodes :class:`PlotKind.value` only at the wire boundary; production contracts
never keep a parallel string vocabulary.

This table deliberately contains no display controls.  A control belongs to the typed
renderer that consumes it; putting unconsumed view or expression knobs in the vocabulary
creates a convincing UI that does nothing.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping

__all__ = ["PlotKind"]


class PlotKind(str, Enum):
    """The sole typed vocabulary for every Figure surface."""

    IMAGE = "2d"
    SITE_MAP = "sites"
    CURVE = "1d"
    METER = "meter"
    ROLLING = "monitor"
    HISTOGRAM = "hist"
    PULSE = "pulse"
    GRID = "grid"

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        return _PLOT_KIND_METADATA[self][0]

    @property
    def panel(self) -> bool:
        return _PLOT_KIND_METADATA[self][1]

    @property
    def input_format(self) -> str:
        return _PLOT_KIND_METADATA[self][2]


#: The ONE plot-kind table, in Add-Panel MENU order.  ``monitor`` names the DEFAULT rolling
#: variant (the side distribution is a toggle on the same kind, not a second kind).
#:
#: This literal is COMPLETE: every kind is here, ``grid`` included.  The render layer pairs
#: each spec with a class, and does so in two steps only because ``GridPlot`` is defined
#: later in that module than the table -- a fact about Python's execution order in ONE file,
#: which is why the deferral lives there and the vocabulary here stays whole.
_PLOT_KIND_METADATA: Mapping[PlotKind, tuple[str, bool, str]] = MappingProxyType({
    PlotKind.IMAGE: (
        "2D image", True, "value must be a 2D array / camera frame (H×W)"
    ),
    PlotKind.SITE_MAP: (
        "Site map",
        True,
        (
            "value must carry a typed SiteMapPresentation whose site state, "
            "background, geometry and revision are already joined"),
    ),
    PlotKind.CURVE: (
        "1D vector", True, "value must be a 1D vector (N,) or per-site array"
    ),
    PlotKind.METER: (
        "Meter", False, "value must resolve to one explicit scalar"
    ),
    PlotKind.ROLLING: (
        "Rolling trace",
        True,
        (
            "value must carry one explicit MONITOR_HISTORY point axis; "
            "the panel does not manufacture history"
        ),
    ),
    PlotKind.HISTOGRAM: (
        "Distribution", True, "value must be a 1D sample vector"
    ),
    # Static timing diagram -- not a blank live-console panel.
    PlotKind.PULSE: ("Pulse sequence", False, ""),
    # GRID is a TaskConsole layout over one typed dataset ViewSpec.  Its exact
    # named FACET bindings are evaluated by the ordinary FigureEvaluator; it
    # does not own a second shape-driven slicer or renderer.
    PlotKind.GRID: (
        "Site grid",
        True,
        (
            "value must admit an explicit named-axis CURVE, HISTOGRAM, or "
            "IMAGE facet view"
        ),
    ),
})

if set(_PLOT_KIND_METADATA) != set(PlotKind):
    raise RuntimeError("plot kind metadata must be complete")
