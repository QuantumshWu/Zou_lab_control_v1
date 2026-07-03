"""Jupyter plotting front-end for Zou lab experiment control.

The module keeps the Confocal_GUIv2 visual language while exposing a
hardware-decoupled API for notebook plotting, live updates, selectors, fitting,
unit conversion, and neutral-atom histogram/readout views.

============================================================================
SEALED-API DESIGN CONTRACT.  The frontend exposes a SMALL public surface so an
external caller (notebook / lab) gets correct art, geometry, dpi and typography
WITHOUT being able to break the visual design: geometry/dpi/typography/scale and
the art-bearing fluent widgets are OWNED here, never passed in or re-exported;
every public parameter is classified DATA (allowed) vs ART/geometry (internal).

The authoritative, numbered statement of those rules -- plus the failure history
that motivates each -- lives in ``frontend/AGENTS.md`` (single source; do not
re-list the rule text here, it drifts).  This module is their implementation.
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
from .data_figure import DataFigure, FitResult, SavedFigure, load_figure
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
    GRID_CELL_BY_KIND,
    HistogramCell,
    ImageCell,
    SiteHistogramGrid,
    grid,
    load,
    panel_plot,
    panel_plot_spec,
    panel_size_cells,
    plot,
    site_histogram_grid,
    site_psf_grid,
    pulse_plot_channels,
    pulse_plot_spec,
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
_FIGURE_VIEWER_EXPORTS = {"FigureViewer", "LoadedFigureNode", "show_figure_viewer"}
_DEVICE_MANAGER_EXPORTS = {"DeviceManagerPanel", "show_device_manager"}


def __getattr__(name: str):
    if name in _PULSE_GUI_EXPORTS:
        pulse_gui = import_module(".pulse_gui", __name__)
        return getattr(pulse_gui, name)
    if name in _TASK_CONSOLE_EXPORTS:
        task_console = import_module(".task_console", __name__)
        return getattr(task_console, name)
    if name in _FIGURE_VIEWER_EXPORTS:
        figure_viewer = import_module(".figure_viewer", __name__)
        return getattr(figure_viewer, name)
    if name in _DEVICE_MANAGER_EXPORTS:
        device_manager = import_module(".device_manager", __name__)
        return getattr(device_manager, name)
    raise AttributeError(name)


def site_map(centers, occupied, *, image=None, roi_radius=3.0,
             labels=("Camera x (px)", "Camera y (px)", "Counts"), display=True, **kwargs):
    """The sealed site-map view: a camera frame (``image=``) with one hollow ring per tweezer, FAINT
    when empty / BOLD when occupied -- the rings come from the single ``SITE_OCCUPANCY_STYLE`` source
    (a :class:`~.live.LiveSiteMap`, via the ``"sites"`` plot kind).  The experiment layer's
    ``views.plots`` routes its detection-image overlay through THIS so it never hard-codes ring art
    itself; ``roi_radius`` is the ring radius in camera pixels (#C2)."""
    import numpy as _np
    occ = _np.asarray(occupied, dtype=float).reshape(-1, 1)
    return plot(centers, occ, kind="sites", image=image, roi_radius=roi_radius,
                labels=labels, display=display, **kwargs)


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
        from .calibration_report import save_calibration_report, save_frame_image

        register_plotter(SimpleNamespace(plot=plot, grid=grid, display_figure=display_figure, run=run,
                                         site_map=site_map, save_calibration_report=save_calibration_report,
                                         save_frame_image=save_frame_image))
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional deps absent: never block import
        pass
    except Exception as exc:  # a REAL registration break must SURFACE (warn), not silently vanish
        import warnings
        warnings.warn(f"neutral-atom viewer registration failed: {exc!r}", RuntimeWarning, stacklevel=2)


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
    "FigureViewer",
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
    "LoadedFigureNode",
    "GridPlot",
    "GridCell",
    "GRID_CELL_BY_KIND",
    "HistogramCell",
    "ImageCell",
    "SiteHistogramGrid",
    "grid",
    "NotebookBuildResult",
    "NotebookExecutionResult",
    "NotesBuildResult",
    "PlotState",
    "PulseSequenceEditor",
    "PulseSequenceFigure",
    "RunSession",
    "SavedFigure",
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
    "load_figure",
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
    "pulse_repeat_markers",
    "pulse_repeat_notation",
    "require_attrs",
    "render_notes_pdf",
    "render_tex_pdf",
    "run",
    "save_figure_data",
    "show_device_manager",
    "show_figure_viewer",
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
