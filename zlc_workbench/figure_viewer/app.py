"""The figure viewer's composition root -- the one place that opens the viewer window.

Every entry goes through :func:`open_figure_viewer`: the double-clickable
``figure_viewer.bat``, the root ``figure_viewer.py`` launcher, and a session's
``figure_viewer()`` sugar.  One composition root keeps "the thing the user opens" a
single object, exactly as the task console's and pulse editor's ``app.py`` do.

**Today this delegates to the legacy viewer stack** (``Zou_lab_control.neutral_atom._gui``
-> ``frontend.figure_viewer``).  That is the transitional state, registered by name in the
Z4 table, and it dies as the 1033-line shell is taken apart into ``zlc_frontend/qt_widgets``
and this app's plot_bridge zone (C20/C25).
"""

from __future__ import annotations

__all__ = ["open_figure_viewer"]


def open_figure_viewer(session=None, *, path=None, scale=None, **kwargs):
    """Open the saved-figure viewer and return the viewer widget.

    ``session`` may be ``None`` (the launcher path: a fresh window per call, reads files
    only, needs no experiment).  A live session gets the ONE-per-session reshow behaviour
    the legacy sugar already owns, so this root reuses proven logic instead of duplicating
    the singleton wiring (C22: the legacy window is the behaviour authority).
    """

    # The ONE named crossing into the legacy tree for this window (Z4).
    from Zou_lab_control.neutral_atom._gui import open_figure_viewer as _legacy_open

    if scale is not None:
        kwargs["scale"] = scale
    return _legacy_open(session, path=path, **kwargs)
