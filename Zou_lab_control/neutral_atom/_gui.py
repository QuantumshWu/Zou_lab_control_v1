"""GUI launchers for a notebook session -- a GUI-ACTION module (like :mod:`notes`).

Opening a window is the ONE place the experiment layer reaches the frontend, so the import
is LAZY and lives here, off the analysis / orchestration path: ``connect`` / ``sitemap`` /
``thresholds`` / ``detect`` never touch it, so virtual==real stays headless-clean and
``neutral_atom`` never pulls the frontend on its own import path.  This module is NOT one of
the sealed analysis directories (``core`` / ``operations`` / ``subsystems``) nor ``session.py``,
so it may import the frontend -- exactly as ``notes.py`` imports the frontend's PDF renderer.

The session's ``exp.task_console()`` / ``exp.pulse_gui()`` are thin composition entrypoints.
The pulse editor also runs without a session (``open_pulse_gui()`` / the frontend's
``show_pulse_gui()``), but that form is deliberately offline and never constructs a device.
Executable virtual or real use always enters through an installation-owned session.
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

    unavailable = getattr(session, "_hardware_unavailable_reason", None)
    if unavailable is not None:
        raise RuntimeError(
            f"{unavailable}; create a new NeutralAtomSession to re-establish hardware"
        )

    from Zou_lab_control.frontend.task_console import show_task_console

    from .core.signals import SignalHub

    runtime_services = session._require_runtime_services()
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
        runtime_fence=runtime_services,
        runtime_fence_provider=lambda: session._require_runtime_services(),
        hide_on_close=True,                 # close = stop nodes + hide (reopen restores)
        **kwargs,
    )
    session._zlc_task_console = console
    return console


def open_pulse_gui(session: Any = None, *, state=None, **kwargs):
    """Open the pulse-sequence editor.

    With a ``session`` the editor receives its immutable target descriptor and generation-bound
    command port from the installation authority and is a ONE-per-session singleton:
    a later ``exp.pulse_gui()`` reshows the SAME editor (its loaded program + edits) instead of a
    new window; closing it just hides it.  With ``session=None`` it is an offline editor and each
    call is its own window."""

    from Zou_lab_control.frontend.pulse_gui import show_pulse_gui

    if session is None:
        return show_pulse_gui(state=state, **kwargs)

    unavailable = getattr(session, "_hardware_unavailable_reason", None)
    if unavailable is not None:
        raise RuntimeError(
            f"{unavailable}; create a new NeutralAtomSession to re-establish hardware"
        )

    existing = getattr(session, "_zlc_pulse_gui", None)
    if existing is not None and _alive(existing):
        _reshow(getattr(existing, "_zlc_window", None) or existing.window())
        return existing

    runtime_services = session._require_runtime_services()
    sequencer = getattr(session._device_set, "sequencer", None)
    command_port = None
    target_descriptor = None
    if sequencer is not None:
        from zlc_workbench.pulse_control import managed_pulse_command_port

        command_port = managed_pulse_command_port(
            session, runtime_services, sequencer
        )
        target_descriptor = command_port.target
    editor = show_pulse_gui(
        target_descriptor=target_descriptor,
        command_port=command_port,
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
    ``devices`` GETTER matters: ``load_config`` replaces the private device set, so the panel
    must re-read it after every apply rather than hold a stale set."""

    def _apply(cfg):
        session.load_config(cfg)     # builds the NEW set first: a bad config is a no-op
        return session._device_set

    def _open():
        session.open_devices()
        return session._device_set

    return {"devices": lambda: session._device_set, "apply_config": _apply, "open_devices": _open}


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
        session._device_set, session_binding=_session_device_binding(session),
        config_dir=str(device_config_dir()), **kwargs)
    window.session = session          # the notebook can read back the bound session off the window
    session._zlc_device_manager = window
    return window


def open_device_viewer(session: Any, **kwargs):
    """Open the device viewer bound to ``session`` -- ONE per session (confocal-style singleton),
    the task console's "Devices" window.

    One tab per loaded device showing its snapshot + live runtime read-backs, and (``editable=True``
    by default) editing each device's basic runtime params LIVE like the API through the device's own
    validated setters.  It is NOT the config editor: NO add / remove / device-set swap -- so it cannot
    change WHICH hardware the session drives; the FULL config editor stays the separate
    ``exp.device_manager()`` / ``na.device_manager()`` entry.  Pass ``editable=False`` for a pure
    read-only peek.  A later call RESHOWS the same window (never a duplicate)."""

    from Zou_lab_control.frontend.device_manager import show_device_viewer

    existing = getattr(session, "_zlc_device_viewer", None)
    if existing is not None and _alive(existing):
        _reshow(existing)
        return existing

    # a PROVIDER (not a snapshot of the set) so a re-open re-reads the session's CURRENT devices --
    # load_config replaces the private set, and the viewer must never hold a stale set.
    if kwargs.get("editable", False):
        raise RuntimeError(
            "runtime device editing is unavailable until the typed control facade owns every write"
        )
    kwargs["editable"] = False
    window = show_device_viewer(devices_provider=lambda: session._device_set, **kwargs)
    window.session = session
    session._zlc_device_viewer = window
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
           "open_device_manager", "open_device_viewer", "device_manager", "load_figure"]
