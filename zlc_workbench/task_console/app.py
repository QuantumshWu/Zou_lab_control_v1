"""The task console's composition root -- the one place that builds the window.

Every entry into this window goes through :func:`open_task_console`: the double-clickable
``task_console.bat``, the root ``task_console.py`` launcher, and ``Experiment.task_console()``
from a notebook.  One composition root is what makes "the thing the user opens" a single
object that can be moved, measured and eventually rewritten, instead of three launchers that
each assemble a slightly different window.

**Today this delegates the whole window to the legacy shell**
(``Zou_lab_control.frontend.task_console``).  That is the transitional state, registered by
name in the Z2A table, and it dies as the shell is taken apart widget by widget: as classes
move into ``zlc_frontend.qt_widgets`` and ``plot_bridge``, this module grows the assembly and
the shell shrinks to aliases (C20).

Why the legacy shell and not the narrow ``Zou_lab_control.workbench`` window: behaviour
authority is main (C22).  Measured 2026-07-20, main's launcher opens a 49-widget console with
Monitor/Logic tabs and a panel board, while this branch's entry had been re-pointed at a
19-widget, tab-less window -- and the legacy shell had no importer left at all.  Restoring the
operator's window IS the migration; the narrow window stays as a component behind
``Zou_lab_control.workbench.open_task_console`` for the scan-intent editing it was built for.
"""

from __future__ import annotations

__all__ = ["console_catalogs", "open_task_console"]

#: The catalog accessors the console's Add-Panel dropdown is filled from, in the order the
#: legacy shell takes them.
_CATALOG_SPECS = (("measurements", "measurement_specs"),
                  ("processors", "processor_specs"),
                  ("tasks", "task_specs"))


def console_catalogs(experiment) -> dict:
    """The measurement / processor / task catalogs ``experiment`` can offer, if any.

    Asked for by name rather than assumed, because the two backends do not agree yet: the
    legacy session exposed ``readout.measurement_specs()`` and friends, and the current
    ``ReadoutFacade`` (measured 2026-07-20) exposes none of the three.  Until the operations
    catalogs land in ``zlc_neutral_atom``, an experiment that cannot answer yields empty
    catalogs -- which is the console's own ``--no-connect`` shape, a supported way to open it,
    not a broken one.  The moment the backend grows the accessors this fills itself, with no
    edit here: that is the point of asking instead of hardcoding.
    """

    readout = getattr(experiment, "readout", None)
    catalogs = {}
    for key, accessor in _CATALOG_SPECS:
        method = getattr(readout, accessor, None)
        if callable(method):
            catalogs[key] = tuple(method())
    return catalogs


def open_task_console(experiment=None, *, task=None, state=None, scale=None, **kwargs):
    """Open the task console bound to ``experiment`` and return the window.

    ``state`` loads a saved console layout from a path; ``task`` loads one by name from the
    workspace's ``tasks/``.  Passing neither opens the default layout.  ``experiment`` may be
    ``None`` for an offline console (no devices are created either way -- the console opens
    empty and STOPPED, and the operator starts nodes from the Logic tab).
    """

    # The legacy shell is imported HERE and nowhere else in the new packages: one named
    # crossing (Z2A) that disappears with the shell, rather than a habit spreading through
    # zlc_workbench.
    from Zou_lab_control.frontend.task_console import (
        TaskConsoleState,
        default_console_state,
        resolve_task_state,
        show_task_console,
    )
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    if state is not None and task is not None:
        raise TypeError("pass a console layout by path (state=) or by name (task=), not both")
    if state is not None:
        console_state = TaskConsoleState.load(state)
    elif task is not None:
        console_state = resolve_task_state(task)
    else:
        console_state = default_console_state()

    # The hub is created by the composition root, not by the window: the console is a VIEW
    # onto signals, and a notebook that wants to publish into the same hub needs the object
    # to exist independently of whether a window is open.
    return show_task_console(hub=SignalHub(), state=console_state, session=experiment,
                             scale=scale, **console_catalogs(experiment), **kwargs)
