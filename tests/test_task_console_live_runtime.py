"""Contracts for the VIEW / LOGIC model: a logic node is added STOPPED and
publishes nothing until Started; a plot is a blank pure view until a signal is set.

The dashboard composes stopped logic nodes with independently bound plot panels.
This pins:

  * adding a logic node puts NOTHING on the hub until Start (default stopped);
  * Start builds the node + publishes (display suppressed -- no auto plot);
  * adding a Plot makes a blank view that shows data only after its signal is set
    AND the producing node is Started;
  * a newly added site-map panel remains unbound until the user selects a source.

Offscreen Qt + virtual backend (same contract path as real hardware).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from conftest import fire_live_imaging   # explicit white-box hardware seam


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _virtual_session(grid=(3, 4)):
    import Zou_lab_control.neutral_atom as na

    return na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (40, 50)})


def _console(exp):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    runtime = exp._require_runtime_services()
    console = TaskConsole(
        hub=SignalHub(), state=default_console_state(), session=exp,
        measurements=exp.readout.measurement_specs(),
        processors=exp.readout.processor_specs(),
        tasks=exp.readout.task_specs(), window_px=(1200, 800),
        runtime_fence=runtime.fence)
    console._timer.stop()
    return console


def _pick(console, data):
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == data)
    kc.setCurrentIndex(i)
    console._add_panel()




def test_embedded_canvas_pins_design_dpi_against_backend_inflation():
    """An embedded panel figure must size from the DESIGN dpi captured when the canvas adopts it,
    NOT from a later-inflated ``figure._original_dpi``.  On a high-DPI screen under ``%matplotlib
    widget`` (ipympl, the notebook backend) the Qt backend inflates ``_original_dpi`` by the screen
    ratio (e.g. 300 -> 750), which made a task's mid-run Monitor figure render ~2.5x too big and
    clip (title cut, colorbar/dist off-screen) -- in the notebook only; the standalone .bat (Agg)
    kept _original_dpi=300.  Guard: even if _original_dpi is inflated after construction, the
    canvas's _zlc_sync uses the captured design dpi, so figure.dpi tracks the design, not 999."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from Zou_lab_control.frontend.qt_canvas import EmbeddedFigureCanvas
    from Zou_lab_control.frontend.style import DESIGN_DPI

    if EmbeddedFigureCanvas is None:
        pytest.skip("matplotlib Qt canvas unavailable")
    fig = Figure(dpi=DESIGN_DPI)
    fig.set_size_inches(2.0, 1.5)
    cv = EmbeddedFigureCanvas(fig, display_scale=1.0)
    assert cv._zlc_design_dpi == float(DESIGN_DPI)        # captured the design dpi
    fig._original_dpi = 999.0                              # simulate the backend inflating it later
    cv._zlc_sync()
    real = float(super(type(cv), cv).devicePixelRatioF()) or 1.0
    # figure.dpi follows the DESIGN dpi (x real x display_scale=1.0), NOT the inflated 999
    assert abs(fig.dpi - DESIGN_DPI * real) < 1e-6


def test_embedded_canvas_first_figure_boosted_dpi_unset_original_still_design_sized():
    """The user's "first task Monitor wrong size, second right" bug.  Under ``%matplotlib widget``
    on a hi-DPI screen the backend BOOSTS ``figure.dpi`` by the screen ratio; on the FIRST figure of
    a session ``figure._original_dpi`` is still UNSET (None) while ``figure.dpi`` is already boosted
    (e.g. 750).  The axes layout sizes the figure at DESIGN_DPI (``pixels / design_dpi(fig)``), so the
    canvas MUST also take DESIGN_DPI -- if it instead reads the boosted ``figure.dpi`` (the old
    ``_original_dpi or figure.dpi``) it captures 750 and the first panel renders ~2.5x too big; the
    SECOND task (``_original_dpi`` now 300) was correct.  Pin to DESIGN_DPI -> correct on the FIRST."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from Zou_lab_control.frontend.qt_canvas import EmbeddedFigureCanvas
    from Zou_lab_control.frontend.style import DESIGN_DPI

    if EmbeddedFigureCanvas is None:
        pytest.skip("matplotlib Qt canvas unavailable")
    fig = Figure(dpi=DESIGN_DPI)
    fig.set_size_inches(2.0, 1.5)                          # design inches (authored at DESIGN_DPI)
    # FIRST figure of a session: dpi already boosted by the backend, _original_dpi NOT yet set.
    if hasattr(fig, "_original_dpi"):
        del fig._original_dpi
    fig._set_dpi(float(DESIGN_DPI) * 2.5, forward=False)   # boosted (e.g. 750), _original_dpi unset
    cv = EmbeddedFigureCanvas(fig, display_scale=1.0)
    # must take the canonical DESIGN dpi, NOT the boosted 750 -> size-correct on the FIRST run
    assert cv._zlc_design_dpi == float(DESIGN_DPI)
    # the displayed widget is inches x DESIGN_DPI x display_scale (DPR cancels): NOT 2.5x bigger
    assert cv.width() == round(2.0 * DESIGN_DPI)


def test_first_and_second_task_panel_build_both_design_sized_under_ipympl_boost():
    """The user's literal report: "open task_console, open task -> Monitor size wrong; only the SECOND
    task is right."  That is a stale-cached OLD build (which inflated to the boosted dpi every time);
    the CURRENT code must show FIRST == SECOND == design-sized.  Build the task-panel canvas TWICE in
    ONE process (= one session) under the EXACT live-browser bug condition each time (figure.dpi boosted
    300->750 with _original_dpi UNSET -- ipympl on a 2.5x browser) and pin minimumSize the way PanelCard
    does; both builds must be design-sized and IDENTICAL (no first-wrong-second-right)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from Zou_lab_control.frontend.qt_canvas import EmbeddedFigureCanvas
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.style import DESIGN_DPI

    if EmbeddedFigureCanvas is None:
        pytest.skip("matplotlib Qt canvas unavailable")
    app = ensure_qt_app()

    def build_task_panel():
        fig = Figure(dpi=DESIGN_DPI)
        fig.set_size_inches(2.0, 1.5)
        if hasattr(fig, "_original_dpi"):
            del fig._original_dpi                              # _original_dpi UNSET (first-figure state)
        fig._set_dpi(float(DESIGN_DPI) * 2.5, forward=False)  # ipympl 2.5x browser boost: 300 -> 750
        cv = EmbeddedFigureCanvas(fig, display_scale=1.0)
        cv.setMinimumSize(cv.sizeHint())                      # the floor PanelCard pins at build
        app.processEvents()                                   # fire the deferred singleShot(0) resync
        return cv._zlc_design_dpi, cv.width(), cv.height(), cv.minimumWidth()

    first = build_task_panel()
    second = build_task_panel()
    assert first == second                                    # NO "first wrong, second right"
    # both builds are design-sized (NOT the boosted 2.5x), incl. the minimumSize floor
    assert first == (float(DESIGN_DPI), round(2.0 * DESIGN_DPI), round(1.5 * DESIGN_DPI), round(2.0 * DESIGN_DPI))


def test_embedded_canvas_deferred_resync_corrects_stale_layout_size():
    """The OTHER half of "first task wrong, second right": layout TIMING.  The construction-time sync
    runs before the canvas is inserted into its parent layout, so the sizeHint/minimumSize it first
    publishes can be the pre-layout values -- the FIRST task card keeps that stale size until some
    later relayout (which the SECOND task triggers).  EmbeddedFigureCanvas schedules a next-tick
    ``_zlc_resync`` (and re-syncs on ``showEvent``) that re-pins the geometry AND calls
    ``updateGeometry`` so the surrounding card is re-measured on the FIRST run.  Guard: the deferred
    resync fires without error and restores the design size after the widget was left a wrong size."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from Zou_lab_control.frontend.qt_canvas import EmbeddedFigureCanvas
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.style import DESIGN_DPI

    if EmbeddedFigureCanvas is None:
        pytest.skip("matplotlib Qt canvas unavailable")
    app = ensure_qt_app()
    fig = Figure(dpi=DESIGN_DPI)
    fig.set_size_inches(2.0, 1.5)
    cv = EmbeddedFigureCanvas(fig, display_scale=1.0)
    assert hasattr(cv, "_zlc_resync")                     # the deferred-resync hook exists
    app.processEvents()                                   # fire the queued singleShot(0, _zlc_resync)
    # Pin a STALE, too-big minimum floor (as a pre-layout build-time setMinimumSize would) AND leave
    # the widget a wrong size; the resync must clear the floor and restore the canonical design size --
    # a stale floor would otherwise pin the canvas too big forever (the "first wrong, second right"
    # trap, since resize cannot go below a stale minimum).
    cv.setMinimumSize(9999, 9999)
    cv.resize(9999, 9999)
    cv._zlc_resync()
    assert cv._zlc_design_dpi == float(DESIGN_DPI)
    assert cv.minimumWidth() == round(2.0 * DESIGN_DPI)   # stale floor cleared to the design floor
    assert cv.width() == round(2.0 * DESIGN_DPI)          # design-sized again after resync
    cv._zlc_resync()                                      # idempotent: no drift on a second call
    assert cv.width() == round(2.0 * DESIGN_DPI)
    assert cv.height() == round(1.5 * DESIGN_DPI)


def test_two_permanent_tabs_monitor_and_logic():
    exp = _virtual_session()
    console = _console(exp)
    try:
        titles = [console.tabs.tabText(i) for i in range(console.tabs.count())]
        assert titles[:2] == ["Monitor", "Logic"]
    finally:
        console.shutdown()
        exp.close()


def test_add_logic_node_is_stopped_and_publishes_nothing_until_start():
    """Adding a camera Measurement creates a STOPPED Logic node -- no node is
    built, nothing is on the hub.  Start builds + runs it (display suppressed: it
    only publishes ``frame_0``, it never opens a plot)."""
    exp = _virtual_session()
    console = _console(exp)
    try:
        from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

        _pick(console, ("camera", "live"))
        # a Logic-tab row, default STOPPED: no node built, nothing on the hub
        assert len(console.logic_nodes) == 1
        row = console.logic_nodes[0]
        assert console._logic_nodes[id(row)] is None
        assert not console.running_nodes          # no node running yet
        assert "frame_0" not in console.hub.names()
        assert row.node.kind == "camera"

        # Start -> builds a CameraMeasurement, registers it, runs it (publish-only)
        console._start_logic_node(row)
        fire_live_imaging(exp)                     # On Pulse: the trigger-driven camera streams
        node = console._logic_nodes[id(row)]
        assert isinstance(node, CameraMeasurement) and node in console.running_nodes
        deadline = time.monotonic() + 8.0
        while "frame_0" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert "frame_0" in console.hub.names()     # it publishes to the hub
        # display suppressed: starting a measurement created NO plot panel
        assert console.cards == []

        # Stop -> node stops, dropped from running_nodes
        console._stop_logic_node(row)
        assert not node.running
        assert node not in console.running_nodes
    finally:
        console.shutdown()
        exp.close()


def test_setting_popup_click_again_does_not_reopen_after_autoclose():
    """A Qt.Popup auto-closes on the press that lands on the Setting button, so the
    button's release must NOT re-open it (real toggle).  The guard: a click within
    the just-dismissed window is a no-op (does not re-show / re-refresh)."""
    exp = _virtual_session()
    console = _console(exp)
    try:
        _pick(console, "2d")
        card = console.cards[-1]
        opened = []
        card._refresh_signal_combo = lambda: opened.append(1)   # spy: only runs on a real open

        # The popup just auto-closed from THIS click -> the release must not re-open.
        card._note_settings_dismissed()
        card._open_settings()
        assert opened == [] and not card.settings_popup.isVisible()

        # A later click (outside the dismiss window) opens normally.
        card._settings_dismissed_at = time.monotonic() - 1.0
        card._open_settings()
        assert opened == [1]
    finally:
        console.shutdown()
        exp.close()


def test_remove_logic_node_stops_it_and_freezes_its_signal():
    """Remove = STOP and remove: after removing a running logic node, its thread is
    stopped, it is dropped from running_nodes, and its hub signal STOPS advancing
    (a plot reading it no longer gets new data) -- not merely the row disappearing."""
    exp = _virtual_session()
    console = _console(exp)
    try:
        _pick(console, ("camera", "live"))
        row = console.logic_nodes[0]
        console._start_logic_node(row)
        fire_live_imaging(exp)                     # On Pulse: the trigger-driven camera streams
        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 8.0
        while "frame_0" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert "frame_0" in console.hub.names()
        # it IS advancing while running
        v0 = console.hub.signal_versions().get("frame_0", 0)
        time.sleep(0.4)
        assert console.hub.signal_versions().get("frame_0", 0) > v0

        # Remove the node's row -> stop + drop everywhere
        console._remove_logic_node(row)
        assert not node.running                       # thread stopped (joined)
        assert node not in console.running_nodes      # dropped from the running set
        assert row not in console.logic_nodes         # row gone
        assert id(row) not in console._logic_nodes

        # and its signal no longer advances (nothing is publishing it anymore)
        v1 = console.hub.signal_versions().get("frame_0", 0)
        time.sleep(0.4)
        assert console.hub.signal_versions().get("frame_0", 0) == v1
    finally:
        console.shutdown()
        exp.close()


def test_add_plot_is_blank_until_signal_set_and_node_started():
    """A Plot panel is a blank pure view: nothing shows until its source signal is
    set AND the producing logic node is Started."""
    exp = _virtual_session()
    console = _console(exp)
    try:
        # add a blank 2D plot -- source is blank, no plotter built
        _pick(console, "2d")
        card = console.cards[-1]
        assert card.config.source == ""
        console.refresh_once()
        assert card.plotter is None               # blank: shows nothing

        # start a camera logic node so `frame_0` becomes available
        _pick(console, ("camera", "live"))
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        fire_live_imaging(exp)                     # On Pulse: the trigger-driven camera streams
        deadline = time.monotonic() + 8.0
        while "frame_0" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)

        # now wire the plot through the same two-part binding the signal picker applies: the REAL
        # input name lives in ``inputs`` and the source expression reads the ``signal`` slot.  Apply
        # through the card boundary so structure binding/render-version reset are exercised too;
        # assigning the compiled cache directly bypasses the production contract.
        from Zou_lab_control.neutral_atom.operations.signal_expr import DEFAULT_SOURCE
        card.config.inputs = ["frame_0"]
        card.source_edit.setText(DEFAULT_SOURCE)
        card._apply_source()
        console.refresh_once()
        assert card.plotter is not None
        assert np.asarray(card.plotter.data_y).size > 0
    finally:
        console.shutdown()
        exp.close()


def test_starting_a_second_camera_reports_driver_conflict_and_keeps_processors():
    """A conflicting camera row is rejected; both the admitted owner and a
    device-free reactive processor remain running."""
    exp = _virtual_session()
    console = _console(exp)
    meas_row = None
    try:
        _pick(console, ("camera", "live"))
        cam_row = console.logic_nodes[-1]
        console._start_logic_node(cam_row)
        cam_node = console._logic_nodes.get(id(cam_row))
        _pick(console, ("processor", "Analysis"))
        proc_row = console.logic_nodes[-1]
        console._start_logic_node(proc_row)
        proc_node = console._logic_nodes.get(id(proc_row))
        assert cam_node in console.running_nodes and proc_node in console.running_nodes

        # A second driver for the same camera is rejected; existing owners are never stopped.
        _pick(console, ("camera", "live"))
        meas_row = console.logic_nodes[-1]
        console._start_logic_node(meas_row)
        assert console._logic_nodes.get(id(cam_row)) is cam_node
        assert cam_node in console.running_nodes
        assert console._logic_nodes.get(id(proc_row)) is proc_node     # processor SURVIVES
        assert proc_node in console.running_nodes
        assert console._logic_nodes.get(id(meas_row)) is None
        assert "start failed" in meas_row.status_label.text().lower()
    finally:
        if meas_row is not None:
            console._stop_logic_node(meas_row)
        console.shutdown()
        exp.close()


def test_save_persists_edit_param_values_not_just_layout():
    """Saving captures current Edit-form values even before a node is started."""
    exp = _virtual_session()
    console = _console(exp)
    try:
        from Zou_lab_control.frontend.task_console import TaskConsoleState

        _pick(console, ("camera", "live"))
        row = console.logic_nodes[-1]
        editor = console._logic_editors[id(row)]
        editor.form._widgets["frames_per_cycle"].setValue(9)
        state = console.read_state()
        assert state.logic[-1].values["frames_per_cycle"] == 9
        restored = TaskConsoleState.from_dict(state.to_dict())
        assert restored.logic[-1].values["frames_per_cycle"] == 9
    finally:
        console.shutdown()
        exp.close()

def test_session_gui_is_a_singleton_that_reopens_the_same_window():
    """``exp.task_console()`` / ``exp.pulse_gui()`` are PER-SESSION singletons (confocal-style):
    a second call REUSES the same window rather than opening a duplicate, and closing the window
    (the X) hides it -- so reopening restores the SAME interface, not a blank new one.  The
    standalone ``show_pulse_gui()`` (no session) stays a fresh window per call."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.neutral_atom._gui import open_pulse_gui

    app = ensure_qt_app()
    exp = _virtual_session()
    try:
        before = len(getattr(app, "_zlc_retained_windows", []))
        c1 = exp.task_console()
        c2 = exp.task_console()
        assert c1 is c2                                          # singleton: the SAME console
        assert len(app._zlc_retained_windows) == before + 1     # exactly ONE window created
        c1.window().close()                                     # X -> hide (hide_on_close), not destroy
        c3 = exp.task_console()
        assert c3 is c1                                         # reopen reuses + restores it
        assert len(app._zlc_retained_windows) == before + 1     # still no duplicate window

        e1 = exp.pulse_gui()
        e2 = exp.pulse_gui()
        assert e1 is e2                                         # pulse editor singleton too
        assert e1.command_port is not None
        assert e1.target_descriptor == e1.command_port.target
        assert not hasattr(e1, "sequencer")
        assert not e1.conn_target_combo.isEnabled()
        assert not e1.conn_connect_button.isEnabled()
        assert open_pulse_gui() is not open_pulse_gui()         # standalone (no session): per-call window
    finally:
        exp.close()


def test_session_gui_launchers_delegate_and_pulse_runs_standalone(monkeypatch):
    """``exp.task_console()`` / ``exp.pulse_gui()`` are confocal-style session sugar that
    delegate to the frontend launchers via the GUI-action module (``neutral_atom`` reaches the
    frontend ONLY here, lazily -- never on its analysis-path import).  task_console fills the
    hub + the auto-discovered catalogs; pulse_gui receives a managed target/command port.  The
    pulse editor also runs STANDALONE offline with no session."""
    import Zou_lab_control.frontend.pulse_gui as pgmod
    import Zou_lab_control.frontend.task_console as tcmod
    import Zou_lab_control.neutral_atom._gui as gui

    from types import SimpleNamespace

    calls: dict = {}
    fake_console = SimpleNamespace()
    monkeypatch.setattr(tcmod, "show_task_console", lambda **kw: (calls.__setitem__("tc", kw), fake_console)[1])
    monkeypatch.setattr(pgmod, "show_pulse_gui", lambda **kw: (calls.setdefault("pg", []).append(kw), "EDITOR")[1])

    exp = _virtual_session()
    try:
        assert exp.task_console(task="foo") is fake_console
        tc = calls["tc"]
        assert tc["session"] is exp and tc["task"] == "foo"
        assert {"hub", "measurements", "processors", "tasks"} <= set(tc)   # catalogs filled from session
        # Runtime authority, rather than QWidget callbacks, owns close/config-swap safety.
        runtime = exp._require_runtime_services()
        assert tc["runtime_fence"] is runtime
        assert tc["runtime_fence_provider"]() is runtime
        assert not hasattr(exp, "_device_teardown_hooks")
        assert not hasattr(exp, "_device_change_hooks")

        assert exp.pulse_gui() == "EDITOR"
        managed = calls["pg"][-1]
        assert managed["command_port"] is not None
        assert managed["target_descriptor"] == managed["command_port"].target
        assert "experiment" not in managed
        assert "sequencer" not in managed

        gui.open_pulse_gui()                                               # STANDALONE
        offline = calls["pg"][-1]
        assert "experiment" not in offline
        assert "sequencer" not in offline
        assert "command_port" not in offline
    finally:
        exp.close()


def test_added_site_map_panel_opens_unbound_even_with_occupied_live():
    """A newly added site-map view never auto-binds merely because data exists."""
    exp = _virtual_session()
    console = _console(exp)
    try:
        console.hub.publish({"occupied": np.ones((1, 1, 12), dtype=bool)})
        before = list(console.cards)
        _pick(console, "sites")
        card = next(c for c in console.cards if c not in before)
        assert card.config.kind == "sites"
        assert card.config.inputs == [""]
        assert str(card.config.source).strip() == ""
    finally:
        console.shutdown()
        exp.close()
