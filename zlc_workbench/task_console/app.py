"""The task console's composition root -- the one place that opens the console window.

Every entry goes through :func:`open_task_console`: the double-clickable
``task_console.bat``, the root ``task_console.py`` launcher, and
``Experiment.task_console()`` from a notebook.

The window is the ORIGINAL console UI -- the Monitor/Logic tabbed board, panel
cards, Fluent chrome -- hosted in :mod:`.plot_bridge_console` (the UI skeleton is
kept BY DIRECTIVE 2026-07-21; it is never redesigned).  Its DATA plane is being
rewired to the current zlc_* stack window by window; the legacy backend trees are
being deleted, and every remaining legacy import inside the skeleton is a listed
rewiring debt, not an accepted dependency.
"""

from __future__ import annotations

__all__ = ["open_task_console"]


def open_task_console(experiment, *, state=None, task=None, **kwargs):
    """Open the console UI for ``experiment`` and return the console body."""

    from .plot_bridge_console import show_task_console

    return show_task_console(experiment, state=state, task=task, **kwargs)
