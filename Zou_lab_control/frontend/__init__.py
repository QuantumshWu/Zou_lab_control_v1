"""Jupyter plotting front-end for Zou lab experiment control.

The module keeps the Confocal_GUIv2 visual language while exposing a
hardware-decoupled API for notebook plotting, live updates, selectors, fitting,
unit conversion, and neutral-atom histogram/readout views.
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
    write_neutral_atom_tutorial,
    write_notebook,
)
from .live import (
    BaseLivePlot,
    HistogramFigure,
    Live1D,
    Live2DDis,
    LiveLiveDis,
    PulseSequenceFigure,
    plot,
    pulse_plot_channels,
    pulse_plot_spec,
    pulse_repeat_marker,
    pulse_repeat_markers,
    pulse_repeat_notation,
)
from .notes import NotesBuildResult, build_frontend_manual, compile_notes_pdf, notes_template_dir, render_notes_pdf, write_notes_tex
from .session import RunSession, run
from .selectors import AreaSelector, CrossSelector, DragHLine, DragVLine, InteractionBundle, PlotState, ZoomPan, attach_interaction
from .style import DEFAULT_STYLE, FONT_PATH, apply_style, enable_long_output, style_context, use_widget_backend
from .ticks import SmartOffsetFormatter, SmartOffsetLocator, apply_smart_ticks


Live2D = Live2DDis
LiveHistogram = HistogramFigure


_PULSE_GUI_EXPORTS = {"PulseSequenceEditor", "show_pulse_gui"}


def __getattr__(name: str):
    if name in _PULSE_GUI_EXPORTS:
        pulse_gui = import_module(".pulse_gui", __name__)
        return getattr(pulse_gui, name)
    raise AttributeError(name)


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
    "LiveLiveDis",
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
    "new_figure",
    "notebook_setup",
    "notes_template_dir",
    "plot",
    "pulse_plot_channels",
    "pulse_plot_spec",
    "pulse_repeat_marker",
    "pulse_repeat_markers",
    "pulse_repeat_notation",
    "require_attrs",
    "render_notes_pdf",
    "run",
    "save_figure_data",
    "show_pulse_gui",
    "split_axes_horizontally",
    "style_context",
    "use_widget_backend",
    "write_frontend_tutorial",
    "write_neutral_atom_fpga_server_tutorial",
    "write_neutral_atom_hardware_tutorial",
    "write_neutral_atom_tutorial",
    "write_notebook",
    "write_notes_tex",
]
