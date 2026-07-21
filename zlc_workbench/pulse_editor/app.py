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

#: How many ports a freshly opened editor shows before the operator adds more.  The rest
#: are reachable from the Ports card; opening with all 22 would bury the ones in use.
INITIAL_VISIBLE_PORTS = 4


def open_pulse_editor(experiment=None, *, state=None, scale=None, **kwargs):
    """Open the pulse editor and return the editor widget.

    ``state`` may be a ``PulseTableState`` or a path to a saved program JSON.
    ``experiment=None`` opens the OFFLINE editor (each call its own window; no
    device is created or discovered).

    This is where lanes acquire their identity.  The board's constraints file is the
    only place that knows a lane is ``cooling`` on pin ``F15`` and that ten more form
    the ``da_bias_x`` DAC bus, so the catalog is built HERE, once, and handed to the
    window -- a catalog built without it shows raw ``ch00`` keys and cannot group a bus.
    A saved program carries its own catalog and keeps it; only the pin numbers, which
    are hardware and never travel inside a document, are supplied from the board.
    """

    from zlc_neutral_atom.timing.board_config import load_board_config
    from zlc_neutral_atom.timing.pulse_table import PulseTableState

    from .plot_bridge_pulse_gui import show_pulse_gui

    try:
        board = load_board_config()
    except (OSError, ValueError):
        board = None          # no board file: the editor still opens, on bare lanes

    if state is not None and not isinstance(state, PulseTableState):
        state = PulseTableState.load(state)
    if state is None and board is not None:
        catalog = board.port_catalog()
        visible = [p.key for p in catalog.ports if p.kind != "clock"][:INITIAL_VISIBLE_PORTS]
        state = PulseTableState(port_catalog=catalog, visible_ports=visible)
    kwargs.setdefault("channel_pins", dict(board.pins) if board is not None else None)
    return show_pulse_gui(state=state, scale=scale, **kwargs)
