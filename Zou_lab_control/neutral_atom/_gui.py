"""GUI launchers for a notebook session -- a GUI-ACTION module (like :mod:`notes`).

Opening a window is the ONE place the experiment layer reaches the frontend, so the import
is LAZY and lives here, off the analysis / orchestration path: ``connect`` / ``sitemap`` /
``thresholds`` / ``detect`` never touch it, so virtual==real stays headless-clean and
``neutral_atom`` never pulls the frontend on its own import path.  This module is NOT one of
the sealed analysis directories (``core`` / ``operations`` / ``subsystems``) nor ``session.py``,
so it may import the frontend -- exactly as ``notes.py`` imports the frontend's PDF renderer.

The session's ``exp.task_console()`` / ``exp.pulse_gui()`` are thin sugar over these.  The
pulse editor also runs STANDALONE with no session (``open_pulse_gui()`` / the frontend's
``show_pulse_gui()``): the editor picks its OWN server connection and needs no experiment --
a session is only required when something must read the edited program back (e.g. a
measurement tuning a period duration).
"""

from __future__ import annotations

from typing import Any


def open_task_console(session: Any, *, task: str | None = None, **kwargs):
    """Open the Task console bound to ``session``.

    Fills the SignalHub + the auto-discovered measurement / processor / task catalogs from
    the session, so a notebook needs only ``exp.task_console()``.  ``task`` loads a saved
    layout (``tasks/<name>.json``)."""

    from Zou_lab_control.frontend.task_console import show_task_console

    from .core.signals import SignalHub

    readout = session.readout
    return show_task_console(
        hub=SignalHub(),
        session=session,
        task=task,
        measurements=readout.measurement_specs(),
        processors=readout.processor_specs(),
        tasks=readout.task_specs(),
        **kwargs,
    )


def open_pulse_gui(session: Any = None, *, state=None, **kwargs):
    """Open the pulse-sequence editor.

    With a ``session`` the editor binds to that experiment (channels / sequencer come from it,
    and a measurement can later read the edited program back).  With ``session=None`` it runs
    STANDALONE -- the editor picks its own server connection and needs no experiment."""

    from Zou_lab_control.frontend.pulse_gui import show_pulse_gui

    if session is None:
        return show_pulse_gui(state=state, **kwargs)
    return show_pulse_gui(
        experiment=session,
        sequencer=getattr(session, "sequencer", None),
        state=state,
        **kwargs,
    )


__all__ = ["open_task_console", "open_pulse_gui"]
