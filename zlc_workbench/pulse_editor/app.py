"""The pulse editor's composition root -- the one place that opens the editor window.

Every entry goes through :func:`open_pulse_editor`: the double-clickable ``pulse_gui.bat``,
the root ``pulse_gui.py`` launcher, and ``Experiment.pulse_gui()`` from a notebook.  One
composition root keeps "the thing the user opens" a single object, exactly as the task
console's ``app.py`` does.

**Today this delegates to the legacy pulse-GUI stack** (``Zou_lab_control.neutral_atom._gui``
-> ``frontend.pulse_gui``).  That is the transitional state, registered by name in the Z4
table, and it dies as the 4475-line shell is taken apart batch by batch into
``zlc_frontend/qt_widgets`` and this app's plot_bridge zone.

Why the legacy editor and not the narrow pulse WORKBENCH: behaviour authority is main (C22).
Main's launcher opens the full ``PulseSequenceEditor`` (edit/preview/scan tabs); this branch's
entry had been re-pointed at the rebuilt workbench window.  The workbench stays a component
behind ``Zou_lab_control.workbench.open_pulse_workbench`` -- the tests that pin its behaviour
take it by that name.

One recorded drift (C22, not chased): main's ``show_pulse_gui`` takes a live ``sequencer=``;
this tree's takes ``target_descriptor=``/``command_port=`` -- the device interface was
re-architected before this round.  The legacy ``open_pulse_gui`` sugar owns that wiring, so
this root delegates to it rather than duplicating the port construction.
"""

from __future__ import annotations

__all__ = ["open_pulse_editor"]


def open_pulse_editor(experiment=None, *, state=None, scale=None, **kwargs):
    """Open the pulse-sequence editor and return the editor widget.

    ``state`` may be a ``PulseTableState`` or a path to a saved program.  ``experiment`` may
    be ``None`` for the offline editor (each call its own window; no device is created or
    discovered).  An experiment that exposes the legacy session surface gets the wired,
    ONE-per-session editor; the current notebook ``Experiment`` facade does not yet, so it
    opens the offline editor -- a supported mode, not a broken one -- and the wiring fills
    itself in when the backend grows the session surface, with no edit here.
    """

    # The ONE named crossing into the legacy tree (Z4): the legacy sugar already owns both
    # compositions -- offline (session=None) and session-wired (command port + singleton) --
    # so this root reuses proven logic instead of duplicating the port construction.
    from Zou_lab_control.neutral_atom._gui import open_pulse_gui
    from Zou_lab_control.neutral_atom.timing import PulseTableState

    if state is not None and not isinstance(state, PulseTableState):
        state = PulseTableState.load(state)
    session = experiment if hasattr(experiment, "_require_runtime_services") else None
    return open_pulse_gui(session, state=state, scale=scale, **kwargs)
