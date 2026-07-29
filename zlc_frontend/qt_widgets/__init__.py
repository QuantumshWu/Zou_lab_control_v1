"""Lazy public Qt surface for :mod:`zlc_frontend`.

Importing one lightweight owner such as :func:`ensure_qt_app` must not also
construct the board/render stack or import Matplotlib.  Public names therefore
resolve to their declaring module on first access.  This module owns only the
explicit package API; individual widgets keep their implementation ownership.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS_BY_MODULE = {
    "..image_view": ("ImageViewportTransform",),
    "..selector": (
        "CrossGesture",
        "CurveInteractionIntent",
        "CurveRangeGesture",
        "CurveViewportCommit",
        "HistogramInteractionIntent",
        "HistogramRangeGesture",
        "HistogramViewportCommit",
        "ImageColorLimitsCommit",
        "ImageInteractionCommit",
        "ImageViewportCommit",
        "PanelInteractionOrigin",
        "RectangleGesture",
    ),
    ".board": ("QtRasterBoard",),
    ".frozen_raster": ("FrozenRasterView", "FrozenRasterWindow"),
    "._raster_front": ("owned_qimage_for_raster",),
    ".faceted_panel_host": ("FacetedPanelHost",),
    ".owner_wake": (
        "error_summary",
        "QtOwnerWake",
        "RASTER_WORK_EXECUTOR",
        "SerialWorkerWindow",
        "wait_for_owner_retirement",
    ),
    ".raster_surface": ("RasterPixelRatioObserver",),
    ".panel_host": ("SinglePanelHost",),
    ".figure_surface_host": (
        "FigureOutputAuthority",
        "FigureSurfaceContext",
        "FigureSurfaceHost",
    ),
    ".published_signals": ("PublishedSignalsLegend",),
    ".view_spec_editor": ("ViewSpecEditor",),
    ".form": (
        "FORM_WIDGET_HANDLERS",
        "FluentParameterForm",
        "FormRuntimeContext",
        "FormWidgetHandler",
    ),
    ".fit_authoring": ("FitAuthoringPane",),
    ".figure_surface_lane": (
        "FigureSurfaceCompletion",
        "FigureSurfaceLane",
        "FigureSurfaceRenderRequest",
    ),
    ".figure_info_pane": ("FigureInfoPane",),
    ".flow_graph_view": ("FlowGraphView",),
    ".signal_picker": (
        "coerce_short_labels",
        "fill_grouped_signal_combo",
        "grouped_signal_items",
        "read_editable_combo",
        "signal_state",
        "signal_tree_groups",
    ),
    ".display_editor": (
        "FluentRevisionedFormEditor",
        "runtime_range_placeholders",
        "sync_revisioned_form_editors",
    ),
    ".fluent": (
        "ElidedLabel",
        "FluentButton",
        "FluentCheckBox",
        "FluentCodeEdit",
        "FluentComboBox",
        "FluentDoubleSpinBox",
        "FluentFormGrid",
        "FluentFrame",
        "FluentGroupBox",
        "FluentLabel",
        "FluentLineEdit",
        "FluentPathEdit",
        "FluentPopup",
        "FluentReadoutEdit",
        "FluentReadoutMultiline",
        "FluentSettingsPopupAnchor",
        "FluentScrollArea",
        "FluentSectionLabel",
        "FluentSettingRow",
        "FluentSpinBox",
        "FluentStatusDot",
        "FluentStatusStrip",
        "FluentSwitch",
        "FluentTabWidget",
        "FluentTreeComboBox",
        "Metrics",
        "ensure_qt_app",
        "fluent_confirm",
        "fluent_font_size",
        "fluent_message",
        "fluent_scrollbar_thickness",
        "fluent_widget_stylesheet",
        "format_compact_number",
        "launch_fluent_window",
        "launch_qt_window",
        "measure_text_width",
        "muted_note_label",
        "popup_gap",
        "release_window",
        "scaled_px",
        "screen_fit_window_size",
        "set_fluent_scale",
        "setting_label_width",
        "signals_blocked",
        "window_pad",
    ),
    ".style": (
        "ACCENT",
        "API_VIOLET",
        "API_VIOLET_DARK",
        "BG",
        "CARD_PAD",
        "CARD_TITLE_PX",
        "COMBO_TRI_SIZE",
        "COMBO_WIDTH",
        "DIVIDER",
        "EDIT_PADDING_H",
        "FLUENT_SCALE_MAX",
        "FLUENT_SCALE_MIN",
        "FONT",
        "FONT_SIZE",
        "GREEN",
        "GRAPHITE",
        "GREY",
        "HINT",
        "HOVER",
        "MUTED_LABEL_STYLE",
        "ORANGE",
        "ORANGE_DARK",
        "ORANGE_TINT",
        "PADDING_H",
        "PADDING_V",
        "PLACEHOLDER",
        "RADIUS",
        "RED",
        "STEP_WIDTH",
        "SURFACE",
        "TEXT",
        "TITLE_LEFT_INSET",
        "WINDOW_PAD",
        "WINDOW_SCREEN_FRACTION",
        "YELLOW",
    ),
}

_EXPORT_MODULE_BY_NAME = {
    name: module_name
    for module_name, names in _EXPORTS_BY_MODULE.items()
    for name in names
}
if sum(map(len, _EXPORTS_BY_MODULE.values())) != len(_EXPORT_MODULE_BY_NAME):
    raise RuntimeError("zlc_frontend.qt_widgets exports one name from two modules")

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
