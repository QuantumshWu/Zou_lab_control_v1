"""The pulse editor's composition root -- the one place that opens the editor window.

Every entry goes through :func:`open_pulse_editor`: the double-clickable
``pulse_gui.bat``, the root ``pulse_gui.py`` launcher, and
``Experiment.pulse_gui()`` from a notebook.

The window is the ORIGINAL pulse editor UI -- the edit/preview/scan tabs, period
cards, channel rows, Fluent chrome -- hosted in :mod:`.plot_bridge_pulse_gui`
(the UI skeleton is kept BY DIRECTIVE 2026-07-21; it is never redesigned).  Its
DATA plane is the current domain stack: the authoring model and the
machine-verified runtime compiler live in ``zlc_neutral_atom.timing``, and
importing this window loads ZERO legacy-tree modules
(guard: ``tests/test_workbench_zero_legacy.py``).
"""

from __future__ import annotations

__all__ = ["open_pulse_editor"]


def open_pulse_editor(experiment=None, *, state=None, scale=None, **kwargs):
    """Open the pulse editor and return the editor widget.

    ``state`` may be a ``PulseTableState`` or a path to a saved program JSON.
    ``experiment=None`` opens the OFFLINE editor (each call its own window; no
    device is created or discovered).
    """

    from zlc_neutral_atom.timing.pulse_table import PulseTableState

    from .plot_bridge_pulse_gui import show_pulse_gui

    if state is not None and not isinstance(state, PulseTableState):
        state = PulseTableState.load(state)
    return show_pulse_gui(state=state, scale=scale, **kwargs)
