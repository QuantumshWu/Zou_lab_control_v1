"""MOVED to :mod:`zlc_workbench.figure_viewer.plot_bridge_figure_viewer` (C20
transitional zone; this shell dies at Z0).

An explicit ALIAS shell: every name below IS the moved object, so the frontend facade's
re-exports (FigureViewer/LoadedFigureNode/show_figure_viewer) and every legacy import
keep working unchanged.
"""

from zlc_workbench.figure_viewer.plot_bridge_figure_viewer import (  # noqa: F401
    __all__,
    FIG_VALUE_KEY,
    FIG_X_KEY,
    FIG_CENTERS_KEY,
    FIG_FRAME_KEY,
    FIG_PREFIX,
    FIGURE_IMAGE_SUFFIXES,
    _resolve_npz,
    _stored_shape,
    _kind_label,
    LoadedFigureNode,
    _seed_state,
    FigureViewer,
    show_figure_viewer,
)
