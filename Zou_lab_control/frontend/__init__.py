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

This module owns and implements that public surface.
============================================================================
"""

from importlib import import_module

from .canvas import (
    FigureSpec,
    close_all,
    configure_canvas,
    create_axes_fixed,
    design_dpi,
    display_figure,
    new_figure,
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
    notes_template_dir,
    render_notes_pdf,
    render_tex_pdf,
    write_notes_tex,
)
from .session import RunSession, run
from .selectors import AreaSelector, CrossSelector, DragHLine, DragVLine, InteractionBundle, PlotState, ZoomPan, attach_interaction
from zlc_frontend.render_style import (
    DEFAULT_STYLE,
    FONT_PATH,
    apply_style,
    style_context,
)
from zlc_frontend.notebook_integration import enable_long_output, use_widget_backend
from .ticks import SmartOffsetFormatter, SmartOffsetLocator, apply_smart_ticks


Live2D = Live2DDis
LiveHistogram = HistogramFigure


_PULSE_GUI_EXPORTS = {"PulseSequenceEditor", "show_pulse_gui"}
_TASK_CONSOLE_EXPORTS = {"TaskConsole", "TaskConsoleState", "PanelConfig", "LogicNodeConfig",
                         "default_console_state", "show_task_console"}
_FIGURE_VIEWER_EXPORTS = {"FigureViewer", "LoadedFigureNode", "show_figure_viewer"}
# NOTE: the device manager is DELIBERATELY NOT a public ``zf`` export.  Unlike the other three GUIs
# (task console / pulse editor / figure viewer -- each meaningfully runs WITHOUT a session: a pure viewer,
# or picks its own server connection), the device manager DISPLAYS the devices the neutral_atom device
# layer loaded, and ``frontend`` is sealed from ``neutral_atom`` (one-directional), so its factory can only
# ever take a ``DeviceSet`` that ``na`` hands it -- it cannot self-discover.  So its ONE public entry is the
# SESSION facade ``exp.device_manager()`` (``na`` owns the DeviceSet); the widget factory
# ``frontend.device_manager.show_device_manager`` stays INTERNAL (imported directly by ``na._gui``).


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

        register_plotter(SimpleNamespace(
            plot=plot,
            grid=grid,
            display_figure=display_figure,
            run=run,
            site_map=site_map,
        ))
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
    "build_frontend_manual",
    "close_all",
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
    "site_psf_grid",
    "pulse_plot_channels",
    "pulse_plot_spec",
    "pulse_repeat_markers",
    "pulse_repeat_notation",
    "require_attrs",
    "render_notes_pdf",
    "render_tex_pdf",
    "run",
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


# ---------------------------------------------------------------- domain ports
# This package is the legacy COMPOSITION ROOT for the render layer: it is the
# one module that already sees both the frontend and the pulse domain, so it is
# where the render layer's domain ports get wired.  Importing any submodule runs
# this file first, so a saved pulse figure always replays.  At Z0, when
# data_figure lives in zlc_frontend and this package is gone, zlc_workbench
# inherits exactly this call.
from zlc_frontend.domain_ports import register_pulse_state_factory as _register_pulse_state_factory


def _pulse_state_from_dict(data):
    """Kept lazy on purpose: importing the 3k-line pulse compiler is not the
    price of importing a plotting package."""

    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState

    return PulseTableState.from_dict(data)


_register_pulse_state_factory(_pulse_state_from_dict)


# The pulse-slots form's template reader.  Same composition-root role as the
# state factory above: the derivation is pulse-domain, the console gets rows.
from zlc_frontend.domain_ports import (
    PulseTemplateRows as _PulseTemplateRows,
    register_pulse_template_reader as _register_pulse_template_reader,
)


def _read_pulse_template(path):
    """Resolve a saved template and describe its slots as plain rows.

    Body lifted verbatim from the console call site it replaces, so the form
    draws exactly what it drew before; only the LOCATION changed.
    """

    import hashlib
    import json

    from Zou_lab_control.neutral_atom.operations.measurements.pulse_scan import (
        _resolve_probe_template,
        _semantic_api_names,
        _semantic_scan_names,
    )

    state = _resolve_probe_template(path)
    api_names = _semantic_api_names(state)
    api_rows = tuple(
        (slot.name, api_names[index], str(slot.kind), str(slot.target), str(slot.unit),
         float(state._read_api_field(slot)))
        for index, slot in enumerate(state.api_slots)
    )
    scan_names = _semantic_scan_names(state)
    scan_rows = tuple(
        (scan_names[i], str(s.kind), str(s.target), str(s.unit), s.label)
        for i, s in enumerate(state.scan_slots)
    )
    code = str(getattr(state, "scan_code", "") or "")
    if not code.strip() and getattr(state, "scan_table", None):
        code = ("scan_table = np.array("
                + repr([list(row) for row in state.scan_table]) + ", dtype=float)")
    payload = state.to_dict()
    program_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    # The per-slot sweep default needs pulse geometry (bus signed range, unit
    # table, clock tick), so it is derived HERE.  Same expressions the form used
    # to run itself, moved to the side of the seam that may import the compiler.
    from Zou_lab_control.neutral_atom.timing import scan_column_spec

    api_columns = tuple(
        scan_column_spec(coordinate, "dac" if kind == "dac" else "duration",
                         unit=(unit or "ns"))
        for _handle, coordinate, kind, _target, unit, _current in api_rows
    )
    scan_columns = tuple(
        scan_column_spec(coordinate, kind, unit=(unit or "ns"))
        for coordinate, kind, _target, unit, _label in scan_rows
    )
    return _PulseTemplateRows(api_rows=api_rows, scan_rows=scan_rows,
                              api_columns=api_columns, scan_columns=scan_columns,
                              program=code, program_id=program_id)


_register_pulse_template_reader(_read_pulse_template)
