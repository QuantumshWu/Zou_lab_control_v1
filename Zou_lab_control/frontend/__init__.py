"""Jupyter plotting front-end for Zou lab experiment control.

The module keeps the Confocal_GUIv2 visual language while exposing a
hardware-decoupled API for notebook plotting, live updates, selectors, fitting,
unit conversion, and neutral-atom histogram/readout views.

============================================================================
SEALED-API DESIGN CONTRACT (authoritative; mirrored in frontend/AGENTS.md and
docs/MAINTAINER_NOTES.md).  The frontend exposes a SMALL public surface so an
external caller (notebook / lab) gets correct art, geometry, dpi and typography
WITHOUT being able to break the visual design.  Six rules:

1. Geometry is OWNED, never passed in.  No public callable accepts ``data_px``,
   ``margins_px``, ``spec`` or ``dpi``.  Sizes come only from ``FigureSpec``
   defaults, ``panel_plot_spec``, ``pulse_plot_spec`` and ``create_axes_fixed``.
   ``plot()``/``panel_plot()`` REJECT those keys (``_SEALED_PLOT_KWARGS``); the
   internal geometry channel is the private ``plot(_spec=...)`` argument.
2. Sizes/colormaps are curated, validated on entry.  Panel size is one of
   ``PANEL_SIZES`` (``panel_size_cells`` raises otherwise); an invalid cmap
   raises rather than silently falling back.
3. ONE typography system, one dpi.  ``style.DEFAULT_STYLE`` (300 dpi, fixed font
   sizes) is the only typography source and is exported READ-ONLY
   (MappingProxyType); there is no public font-size/colour/dpi override.
4. ONE display-scale rule.  On-screen scale comes only from
   ``qt_fluent.resolve_fluent_auto_scale`` (GUI controls) and
   ``qt_canvas.panel_canvas`` / ``PANEL_DISPLAY_SCALE`` (embedded figures).
   Public ``show_*`` entry points default to ``scale=None`` (auto).
5. Art-bearing fluent widgets stay INTERNAL.  ``qt_fluent.*`` / ``qt_canvas.*``
   (incl. ``FluentGroupBox(shadow=)``, ``add_fluent_shadow``,
   ``EmbeddedFigureCanvas``) are NOT re-exported here.  Shadows / borders /
   corner radii are construction details, never public knobs.
6. Adding a public parameter requires classifying it.  DATA (labels, title,
   bins, thresholds, relim_mode, cmap-from-a-list, channels) is allowed on the
   public surface; ART / GEOMETRY / TYPOGRAPHY (margins, dpi, colours, sizes,
   shadow, fonts) lives on INTERNAL classes only.
============================================================================
"""

from importlib import import_module

from .canvas import (
    FigureSpec,
    auto_data_size_px,
    close_all,
    configure_canvas,
    create_axes_fixed,
    design_dpi,
    display_figure,
    new_figure,
    save_figure_data,
    split_axes_horizontally,
)
from .data_figure import DataFigure, FitResult
from .jupyter import (
    BOOTSTRAP_CELL,
    NotebookBuildResult,
    NotebookExecutionResult,
    execute_notebook,
    notebook_setup,
    require_attrs,
    write_frontend_tutorial,
    write_neutral_atom_fpga_server_tutorial,
    write_neutral_atom_hardware_tutorial,
    write_neutral_atom_qcmos_live_tutorial,
    write_neutral_atom_tutorial,
    write_notebook,
)
from .live import (
    BaseLivePlot,
    HistogramFigure,
    Live1D,
    Live2DDis,
    LiveLive,
    LiveLiveDis,
    LiveSiteMap,
    PANEL_SIZES,
    PulseSequenceFigure,
    GridPlot,
    GridCell,
    HistogramCell,
    ImageCell,
    SiteHistogramGrid,
    load,
    panel_plot,
    panel_plot_spec,
    panel_size_cells,
    plot,
    site_histogram_grid,
    site_psf_grid,
    pulse_plot_channels,
    pulse_plot_spec,
    pulse_repeat_marker,
    pulse_repeat_markers,
    pulse_repeat_notation,
)
from .notes import (
    NotesBuildResult,
    build_frontend_manual,
    compile_notes_pdf,
    notes_template_dir,
    render_notes_pdf,
    render_tex_pdf,
    write_notes_tex,
)
from .session import RunSession, run
from .selectors import AreaSelector, CrossSelector, DragHLine, DragVLine, InteractionBundle, PlotState, ZoomPan, attach_interaction
from .style import DEFAULT_STYLE, FONT_PATH, apply_style, enable_long_output, style_context, use_widget_backend
from .ticks import SmartOffsetFormatter, SmartOffsetLocator, apply_smart_ticks


Live2D = Live2DDis
LiveHistogram = HistogramFigure


_PULSE_GUI_EXPORTS = {"PulseSequenceEditor", "show_pulse_gui"}
_TASK_CONSOLE_EXPORTS = {"TaskConsole", "TaskConsoleState", "PanelConfig", "LogicNodeConfig",
                         "default_console_state", "show_task_console"}


def __getattr__(name: str):
    if name in _PULSE_GUI_EXPORTS:
        pulse_gui = import_module(".pulse_gui", __name__)
        return getattr(pulse_gui, name)
    if name in _TASK_CONSOLE_EXPORTS:
        task_console = import_module(".task_console", __name__)
        return getattr(task_console, name)
    raise AttributeError(name)


def _register_neutral_atom_viewer() -> None:
    """Register this frontend as the experiment layer's viewer.

    Inversion of control: ``neutral_atom`` never imports the frontend; it routes
    optional plotting through ``Zou_lab_control._viewer_registry``, and the
    frontend opts in here on import.  The registry is a dependency-free leaf, so
    this does not pull ``neutral_atom`` into the frontend import.
    """

    try:
        from types import SimpleNamespace

        from Zou_lab_control._viewer_registry import register_plotter
        from .calibration_report import save_calibration_report

        register_plotter(SimpleNamespace(plot=plot, display_figure=display_figure, run=run,
                                         save_calibration_report=save_calibration_report))
    except Exception:  # pragma: no cover - plotting registration must never block import
        pass


_register_neutral_atom_viewer()


__all__ = [
    "AreaSelector",
    "BaseLivePlot",
    "BOOTSTRAP_CELL",
    "CrossSelector",
    "DEFAULT_STYLE",
    "DataFigure",
    "DragHLine",
    "DragVLine",
    "FONT_PATH",
    "FigureSpec",
    "FitResult",
    "HistogramFigure",
    "InteractionBundle",
    "Live1D",
    "Live2D",
    "Live2DDis",
    "LiveHistogram",
    "LiveLive",
    "LiveLiveDis",
    "LiveSiteMap",
    "GridPlot",
    "GridCell",
    "HistogramCell",
    "ImageCell",
    "SiteHistogramGrid",
    "NotebookBuildResult",
    "NotebookExecutionResult",
    "NotesBuildResult",
    "PlotState",
    "PulseSequenceEditor",
    "PulseSequenceFigure",
    "RunSession",
    "SmartOffsetFormatter",
    "SmartOffsetLocator",
    "ZoomPan",
    "apply_smart_ticks",
    "apply_style",
    "attach_interaction",
    "auto_data_size_px",
    "build_frontend_manual",
    "close_all",
    "compile_notes_pdf",
    "configure_canvas",
    "create_axes_fixed",
    "design_dpi",
    "display_figure",
    "enable_long_output",
    "execute_notebook",
    "load",
    "new_figure",
    "notebook_setup",
    "notes_template_dir",
    "PANEL_SIZES",
    "panel_plot",
    "panel_plot_spec",
    "panel_size_cells",
    "plot",
    "site_histogram_grid",
    "pulse_plot_channels",
    "pulse_plot_spec",
    "pulse_repeat_marker",
    "pulse_repeat_markers",
    "pulse_repeat_notation",
    "require_attrs",
    "render_notes_pdf",
    "render_tex_pdf",
    "run",
    "save_figure_data",
    "show_pulse_gui",
    "show_task_console",
    "split_axes_horizontally",
    "style_context",
    "use_widget_backend",
    "write_frontend_tutorial",
    "write_neutral_atom_fpga_server_tutorial",
    "write_neutral_atom_hardware_tutorial",
    "write_neutral_atom_qcmos_live_tutorial",
    "write_neutral_atom_tutorial",
    "write_notebook",
    "write_notes_tex",
]
