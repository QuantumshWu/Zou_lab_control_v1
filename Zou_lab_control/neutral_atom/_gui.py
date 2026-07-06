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
    # The console's running logic nodes are device CONSUMERS (worker threads blocking inside
    # camera.acquire): register their stop with BOTH session seams so the threads never keep old
    # camera / RPyC handles alive and drive closed hardware.
    #  * teardown (full): ``exp.close()`` stops EVERY node (the whole console is going away).
    #  * change (fine-grained): the device manager's "Load config" / a device reinit stops only
    #    the nodes riding the SWAPPED devices, so a camera swap leaves a scan on the untouched
    #    sequencer running (the user's requirement: don't stop unrelated work).
    session.add_device_teardown_hook(console.stop_all_nodes)
    session.add_device_change_hook(console.stop_nodes_using)
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


def _session_device_binding(session: Any) -> dict:
    """Bundle a session's device actions into the plain-callback dict the device-manager
    panel consumes (the panel stays session-agnostic; it never imports the session).  The
    ``devices`` GETTER matters: ``load_config`` REPLACES ``session.devices``, so the panel
    must re-read it after every apply rather than hold a stale set."""

    def _apply(cfg):
        session.load_config(cfg)     # builds the NEW set first: a bad config is a no-op
        return session.devices

    def _open():
        session.open_devices()
        return session.devices

    return {"devices": lambda: session.devices, "apply_config": _apply, "open_devices": _open}


def open_device_manager(session: Any, **kwargs):
    """Open the device manager bound to ``session`` -- ONE per session (confocal-style singleton).

    The session's DEVICE-INIT entry: a config EDITOR over the round-trippable ``{role:
    {"type", "params"}}`` dict (typed per-class forms, New/Load/Save/Apply) plus the
    hardware panel (bus scan with add-to-config, the loaded instances with snapshots).
    The GUI face of ``na.load_devices`` / ``na.discover_devices``; a later call RESHOWS
    the same window -- so a notebook never accumulates duplicates."""

    # Import the INTERNAL factory directly from the submodule: the widget factory is not a public
    # ``zf`` export (it is inherently coupled to na's device registry); ``exp.device_manager()``
    # and the module-level ``na.device_manager()`` are the two public entries.
    from Zou_lab_control.frontend.device_manager import show_device_manager

    from .devices.registry import device_config_dir

    existing = getattr(session, "_zlc_device_manager", None)
    if existing is not None and _alive(existing):
        _reshow(existing)
        return existing

    window = show_device_manager(
        session.devices, session_binding=_session_device_binding(session),
        config_dir=str(device_config_dir()), **kwargs)
    window.session = session          # the notebook can read back the bound session off the window
    session._zlc_device_manager = window
    return window


def device_manager(config: Any = None, **kwargs):
    """Open the device manager WITHOUT a session (``na.device_manager()``) -- the
    device-INIT entry point.

    Starts as a pure config editor: empty, or seeded from ``config`` (a template name such
    as ``"virtual"`` / ``"remote_template"``, a JSON path, or a config dict -- the same
    forms ``na.connect`` takes).  Its green button reads "Init devices": pressing it
    ``na.connect``\\ s the working config, and on success THIS window upgrades in place to
    the new session's manager (the Loaded card activates, the button becomes Apply, and a
    later ``exp.device_manager()`` reuses this same window).

    RETURNS the manager window, whose ``.session`` is ``None`` until Init and the live
    session afterwards -- so the notebook flow is ``mgr = na.device_manager("virtual")``,
    edit + press "Init devices", then ``exp = mgr.session``.  (A pure ``na.connect(config)``
    stays the one-liner when no editor is wanted; the manager is the visual init path.)"""

    from Zou_lab_control.frontend.device_manager import show_device_manager

    from .devices.registry import device_config_dir, read_config

    initial = dict(read_config(config)) if config is not None else {}
    state: dict = {}

    def _init(cfg_dict):
        from .session import connect

        session = connect(cfg_dict)
        window = state.get("window")
        if window is not None:
            # register the upgraded window as the new session's one-per-session manager AND
            # hand the fresh session back to the notebook via the window it already holds.
            session._zlc_device_manager = window
            window.session = session
        return _session_device_binding(session)

    window = show_device_manager(
        None, initial_config=initial, on_init_session=_init,
        config_dir=str(device_config_dir()), **kwargs)
    window.session = None            # populated by _init the moment "Init devices" succeeds
    state["window"] = window
    return window


def load_figure(path):
    """Reopen a ``.npz`` saved by a panel's / notebook figure's Save as a hardware-free
    ``SavedFigure`` -- ``na.load_figure('scan.npz').info_summary()`` tells what it holds and
    ``.plot(kind=...)`` re-renders it, all without a session.  Sugar over
    ``frontend.load_figure`` (the LAZY frontend reach lives here, off the analysis path)."""

    from Zou_lab_control.frontend import load_figure as _load_figure

    return _load_figure(path)


__all__ = ["open_task_console", "open_pulse_gui", "open_figure_viewer",
           "open_device_manager", "device_manager", "load_figure"]
