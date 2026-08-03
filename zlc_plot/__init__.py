"""Lazy public facade for Zou Lab Control plotting, interaction and Fit.

Importing a headless value module such as :mod:`zlc_plot.kinds` or
:mod:`zlc_plot.specs` must not initialize Matplotlib or Qt.  Public facade
names therefore resolve from their sole declaring module on first access.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS_BY_MODULE = {
    ".api": (
        "curve",
        "facet_grid",
        "histogram",
        "image",
        "pulse_timeline",
        "rolling",
        "show",
    ),
    ".backends": (
        "BackendUnavailableError",
        "Qt5PlotWidget",
        "ensure_qt5_application",
        "notebook_available",
        "qt5_available",
    ),
    ".config": ("DEFAULTS", "PlotLibraryDefaults"),
    ".codec": ("plot_spec_from_tree", "plot_spec_to_tree"),
    ".fit": (
        "FitEngine",
        "FacetFitBatchResult",
        "FitCancelled",
        "FitDeadlineExceeded",
        "FitModelRegistry",
        "FitModelSpec",
        "FitOptions",
        "FitParameterDisplay",
        "FitResult",
        "FitTarget",
        "RegularImageFitInput",
        "builtin_fit_models",
        "format_fit_initials",
        "parse_fit_initials",
    ),
    ".kinds": ("AxisRef", "PlotKind"),
    ".notebook": ("NotebookView",),
    ".live": (
        "LiveDataRevision",
        "LivePlotController",
        "LivePlotMetrics",
        "LiveUpdateError",
    ),
    ".primitives": (
        "ImageFrame",
        "ImagePointOverlay",
        "PointMarker",
        "PointStatus",
        "PulseAnalogTrace",
        "PulseBlock",
        "PulseChannel",
        "PulseDacScanSegment",
        "PulseRepeatMarker",
        "PulseScanRegion",
        "PulseTimelineData",
    ),
    ".raster": (
        "RasterBuffer",
        "RasterFront",
        "RasterIdentity",
        "RasterOperation",
        "RasterPlotHost",
    ),
    ".resolver": ("axis_choices", "default_plot_spec", "resolve_plot_spec"),
    ".selectors": (
        "CrosshairPoint",
        "NumericRange",
        "RectangleRange",
        "SelectorKind",
        "SelectorState",
    ),
    ".session": (
        "DisplayDescription",
        "FitEvent",
        "FitSelection",
        "FitScope",
        "PlotSession",
        "PlotSessionConfig",
        "PulseTimelineSelectionData",
        "SelectionChange",
        "SelectionData",
        "SelectionEvent",
        "SelectorData",
        "SessionRevisions",
    ),
    ".specs": (
        "CurvePlot",
        "FacetGridPlot",
        "HistogramPlot",
        "ImagePlot",
        "PlotLabels",
        "PlotSpec",
        "PulseTimelinePlot",
        "Reduction",
        "RelimMode",
        "RollingPlot",
    ),
    ".ui": (
        "ControlKind",
        "ParameterControl",
        "parameter_controls",
        "plot_spec_controls",
    ),
}

_EXPORT_MODULE_BY_NAME = {
    name: module_name
    for module_name, names in _EXPORTS_BY_MODULE.items()
    for name in names
}
if sum(map(len, _EXPORTS_BY_MODULE.values())) != len(_EXPORT_MODULE_BY_NAME):
    raise RuntimeError("zlc_plot exports one name from two modules")

__all__ = tuple(_EXPORT_MODULE_BY_NAME)


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULE_BY_NAME[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
