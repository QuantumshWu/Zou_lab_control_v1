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


def _alive(widget) -> bool:
    """True if a Qt widget's underlying C++ object still exists (not destroyed)."""

    try:
        widget.objectName()      # raises RuntimeError once the C++ object is gone
        return True
    except (RuntimeError, ReferenceError, AttributeError):
        return False


def _reshow(window) -> None:
    """Bring a hidden / minimised window back to the front."""

    if window is None:
        return
    window.showNormal()
    window.raise_()
    window.activateWindow()


def open_task_console(session: Any, *, task: str | None = None, **kwargs):
    """Open the Task console bound to ``session`` -- ONE per session (confocal-style singleton).

    The first call builds it (filling the SignalHub + the auto-discovered measurement /
    processor / task catalogs; ``task`` loads a saved ``tasks/<name>.json`` layout).  A later
    call RESHOWS the SAME console -- so a notebook never accumulates duplicate windows, and
    reopening restores the previous interface.  Closing its window (the X) STOPS every running
    node (releasing the camera / sequencer) but keeps the layout, since the window only hides."""

    from Zou_lab_control.frontend.task_console import show_task_console

    from .core.signals import SignalHub

    existing = getattr(session, "_zlc_task_console", None)
    if existing is not None and _alive(existing):
        _reshow(existing.window())          # singleton: reuse + restore, never a second window
        return existing

    readout = session.readout
    console = show_task_console(
        hub=SignalHub(),
        session=session,
        task=task,
        measurements=readout.measurement_specs(),
        processors=readout.processor_specs(),
        tasks=readout.task_specs(),
        hide_on_close=True,                 # close = stop nodes + hide (reopen restores)
        **kwargs,
    )
    session._zlc_task_console = console
    return console


def open_pulse_gui(session: Any = None, *, state=None, **kwargs):
    """Open the pulse-sequence editor.

    With a ``session`` the editor binds to that experiment (channels / sequencer come from it,
    and a measurement can later read the edited program back) and is a ONE-per-session singleton:
    a later ``exp.pulse_gui()`` reshows the SAME editor (its loaded program + edits) instead of a
    new window; closing it just hides it.  With ``session=None`` it runs STANDALONE -- the editor
    picks its own server connection, needs no experiment, and each call is its own window."""

    from Zou_lab_control.frontend.pulse_gui import show_pulse_gui

    if session is None:
        return show_pulse_gui(state=state, **kwargs)

    existing = getattr(session, "_zlc_pulse_gui", None)
    if existing is not None and _alive(existing):
        _reshow(getattr(existing, "_zlc_window", None) or existing.window())
        return existing

    editor = show_pulse_gui(
        experiment=session,
        sequencer=getattr(session, "sequencer", None),
        state=state,
        hide_on_close=True,
        **kwargs,
    )
    session._zlc_pulse_gui = editor
    return editor


def open_figure_viewer(session: Any = None, *, path=None, **kwargs):
    """Open the saved-figure viewer window (``exp.figure_viewer()``).

    A PURE VIEWER: it reopens a saved ``.npz`` (or a folder of them) with no hardware / acquisition,
    so it works with or WITHOUT a session.  With a ``session`` it is a ONE-per-session singleton (a
    later ``exp.figure_viewer()`` reshows the SAME window instead of a new one; closing it just hides
    it); with ``session=None`` each call is its own window.  ``path`` opens a file/folder on launch."""

    from Zou_lab_control.frontend.figure_viewer import show_figure_viewer

    if session is None:
        return show_figure_viewer(path=path, **kwargs)

    existing = getattr(session, "_zlc_figure_viewer", None)
    if existing is not None and _alive(existing):
        _reshow(getattr(existing, "_zlc_window", None) or existing.window())
        if path is not None:
            existing.open_path(path)
        return existing

    viewer = show_figure_viewer(path=path, hide_on_close=True, **kwargs)
    session._zlc_figure_viewer = viewer
    return viewer


def open_device_manager(session: Any, **kwargs):
    """Open the device manager bound to ``session`` -- ONE per session (confocal-style singleton).

    Shows every device the session's config loaded, grouped by device DOMAIN (Camera / Sequencer /
    Trap array / a future RF source -- the same registry the per-measurement device dropdowns read),
    plus a "Scan hardware" button that probes the buses.  It is the GUI face of ``na.load_devices`` /
    ``na.discover_devices``.  A later call RESHOWS the same window (rebuilt from the live DeviceSet if
    it was closed) -- so a notebook never accumulates duplicates."""

    from Zou_lab_control.frontend import show_device_manager

    existing = getattr(session, "_zlc_device_manager", None)
    if existing is not None and _alive(existing):
        _reshow(existing)
        return existing

    window = show_device_manager(session.devices, **kwargs)
    session._zlc_device_manager = window
    return window


def load_figure(path):
    """Reopen a ``.npz`` saved by a panel's / notebook figure's Save as a hardware-free
    ``SavedFigure`` -- ``na.load_figure('scan.npz').info_summary()`` tells what it holds and
    ``.plot(kind=...)`` re-renders it, all without a session.  Sugar over
    ``frontend.load_figure`` (the LAZY frontend reach lives here, off the analysis path)."""

    from Zou_lab_control.frontend import load_figure as _load_figure

    return _load_figure(path)


__all__ = ["open_task_console", "open_pulse_gui", "open_figure_viewer",
           "open_device_manager", "load_figure"]
