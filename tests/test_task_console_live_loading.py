"""Contract: the corrected VIEW / LOGIC model -- a logic node is added STOPPED and
publishes nothing until Started; a plot is a blank pure view until a signal is set.

The old "Readout: Loading (camera+detect+calibrate)" one-click composite is GONE.
You build the dashboard from decoupled pieces: add a camera Measurement / a Task as a
STOPPED Logic node (Logic tab), Start it from its Edit, then add Plot panels (Monitor
board) pointed at the signals it publishes.  This pins:

  * adding a logic node puts NOTHING on the hub until Start (default stopped);
  * Start builds the node + publishes (display suppressed -- no auto plot);
  * adding a Plot makes a blank view that shows data only after its signal is set
    AND the producing node is Started;
  * the removed composite is no longer offered in Add Panel.

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


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _calibrated_virtual_session(grid=(3, 4)):
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (40, 50)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    return exp


def _console(exp):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    console = TaskConsole(
        hub=SignalHub(), state=default_console_state(), session=exp,
        measurements=exp.readout.measurement_specs(),
        processors=exp.readout.processor_specs(),
        tasks=exp.readout.task_specs(), window_px=(1200, 800))
    console._timer.stop()
    return console


def _pick(console, data):
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == data)
    kc.setCurrentIndex(i)
    console._add_panel()


def test_removed_loading_composite_is_not_offered():
    """The old ('live','loading') Readout composite is gone from Add Panel."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        kc = console.kind_combo
        data = [kc.itemData(i) for i in range(kc.count())]
        assert ("live", "loading") not in data
        # the camera Measurement + the calibrate Task ARE offered (as logic layers)
        assert ("camera", "live") in data
        assert ("task", "Calibrate readout") in data
    finally:
        console.shutdown()
        exp.close()


def test_two_permanent_tabs_monitor_and_logic():
    exp = _calibrated_virtual_session()
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
    only publishes ``frame``, it never opens a plot)."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

        _pick(console, ("camera", "live"))
        # a Logic-tab row, default STOPPED: no node built, nothing on the hub
        assert len(console.logic_nodes) == 1
        row = console.logic_nodes[0]
        assert console._logic_nodes[id(row)] is None
        assert not console.running_nodes          # no node running yet
        assert "frame" not in console.hub.names()
        assert row.node.kind == "camera"

        # Start -> builds a CameraMeasurement, registers it, runs it (publish-only)
        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        assert isinstance(node, CameraMeasurement) and node in console.running_nodes
        deadline = time.monotonic() + 8.0
        while "frame" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert "frame" in console.hub.names()     # it publishes to the hub
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
    exp = _calibrated_virtual_session()
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
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        _pick(console, ("camera", "live"))
        row = console.logic_nodes[0]
        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 8.0
        while "frame" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert "frame" in console.hub.names()
        # it IS advancing while running
        v0 = console.hub.signal_versions().get("frame", 0)
        time.sleep(0.4)
        assert console.hub.signal_versions().get("frame", 0) > v0

        # Remove the node's row -> stop + drop everywhere
        console._remove_logic_node(row)
        assert not node.running                       # thread stopped (joined)
        assert node not in console.running_nodes      # dropped from the running set
        assert row not in console.logic_nodes         # row gone
        assert id(row) not in console._logic_nodes

        # and its signal no longer advances (nothing is publishing it anymore)
        v1 = console.hub.signal_versions().get("frame", 0)
        time.sleep(0.4)
        assert console.hub.signal_versions().get("frame", 0) == v1
    finally:
        console.shutdown()
        exp.close()


def test_add_plot_is_blank_until_signal_set_and_node_started():
    """A Plot panel is a blank pure view: nothing shows until its source signal is
    set AND the producing logic node is Started."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        # add a blank 2D plot -- source is blank, no plotter built
        _pick(console, "2d")
        card = console.cards[-1]
        assert card.config.source == ""
        console.refresh_once()
        assert card.plotter is None               # blank: shows nothing

        # start a camera logic node so `frame` becomes available
        _pick(console, ("camera", "live"))
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        deadline = time.monotonic() + 8.0
        while "frame" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)

        # now wire the plot to that signal -> it shows data
        card.config.source = "value = frame"
        card._compiled_source = "value = frame"
        console.refresh_once()
        assert card.plotter is not None
        assert np.asarray(card.plotter.data_y).size > 0
    finally:
        console.shutdown()
        exp.close()


def test_task_logic_node_produces_calibration_off_the_hub_when_started():
    """A calibrate Task added as a Logic node is STOPPED; Start runs it.  Its result +
    mid-run output land on the NODE (node.calibration / node.result / node.output),
    NEVER the hub -- the hub carries only measurement + processor outputs."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        from Zou_lab_control.neutral_atom.operations.logic import CalibrateReadoutTask

        _pick(console, ("task", "Calibrate readout"))
        row = console.logic_nodes[-1]
        assert console._logic_nodes[id(row)] is None      # stopped

        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        assert isinstance(node, CalibrateReadoutTask)
        deadline = time.monotonic() + 12.0
        while not getattr(node, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert node.finished
        # result + calibration live on the INSTANCE; mid-run frame in the node's buffer
        assert node.calibration is not None
        assert np.asarray(node.result["centers"]).shape == (12, 2)
        assert "frame" in node.output.names()
        # the task put NOTHING on the hub (no cal_* signals leaked onto it)
        assert not any(n.startswith("cal_") for n in console.hub.names())
    finally:
        console.shutdown()
        exp.close()


def test_running_task_takes_a_fixed_panel_and_locks_the_console():
    """#5: while a task runs it OWNS the console (confocal-style) -- a dedicated
    Monitor panel shows its mid-run output (read off the task's OWN buffer, not the
    hub) and every other control is LOCKED (Add Panel / Edit no-op, header disabled);
    only Stop / waiting is allowed.  When it finishes a tick releases the lock and the
    transient panel is dropped."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        n_before = len(console.cards)
        _pick(console, ("task", "Calibrate readout"))
        row = console.logic_nodes[-1]
        console._start_logic_node(row)

        # LOCK engaged: dedicated task panel on the board + banner up + header disabled.
        # (isHidden(), not isVisible(): the window isn't shown in a headless test, so
        # isVisible() is always False -- isHidden() reflects the explicit shown flag.)
        assert console._task_locked is True
        assert console.task_banner.isHidden() is False
        assert console._task_card is not None and console._task_card in console.cards
        assert console.kind_combo.isEnabled() is False
        # locked: Add Panel no-ops while a task owns the console (only the task panel
        # was added; a second Add adds nothing).
        console._add_panel()
        assert len(console.cards) == n_before + 1

        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 12.0
        while not getattr(node, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert node.finished
        assert node.output.latest("frame") is not None       # mid-run frame buffered (off hub)

        # a tick detects completion -> lock released, banner hidden, transient panel gone.
        console._tick()
        assert console._task_locked is False
        assert console.task_banner.isHidden() is True
        assert console._task_card is None
        assert console.kind_combo.isEnabled() is True
        assert len(console.cards) == n_before
        assert not any(n.startswith("cal_") for n in console.hub.names())
    finally:
        console.shutdown()
        exp.close()


def test_self_finished_task_releases_lock_in_poll():
    """A one-shot task that finishes ON ITS OWN (not via the Stop button) must release
    the console lockout in the canonical node-lifecycle poll -- not only via the
    mid-run-panel refresh (which is skipped when a task has no mid-run panel).  So a
    completed calibration never leaves the dashboard locked forever."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        _pick(console, ("task", "Calibrate readout"))
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        assert console._task_locked is True
        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 12.0
        while not getattr(node, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert node.finished
        # the LIFECYCLE poll alone (no _refresh_task_panel) releases the lock
        console._poll_logic_nodes()
        assert console._task_locked is False
        assert console._running_task_row is None
        assert console.kind_combo.isEnabled() is True
    finally:
        console.shutdown()
        exp.close()


def test_save_persists_edit_param_values_not_just_layout():
    """#4: saving captures the CURRENT Edit-form parameter values (even for a node that
    was never Started), not just the panel geometry; a JSON round-trip restores them."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        from Zou_lab_control.frontend.task_console import TaskConsoleState

        _pick(console, ("task", "Calibrate readout"))
        row = console.logic_nodes[-1]
        editor = console._logic_editors[id(row)]
        # edit a param in the Edit form WITHOUT starting the node
        editor.form._widgets["calibration_frames"][1].setValue(9)
        # read_state flushes the open Edit form into the node config
        state = console.read_state()
        assert state.logic[-1].values["calibration_frames"] == 9
        # round-trip through JSON -> the edited value survives load
        restored = TaskConsoleState.from_dict(state.to_dict())
        assert restored.logic[-1].values["calibration_frames"] == 9
    finally:
        console.shutdown()
        exp.close()


def test_plot_edit_shows_producing_processor_param_form():
    """#2: a plot's signal comes from a measurement/processor; the plot's Edit shows
    THAT node's full parameter form (here the Judge-occupancy processor's
    calibration/source/ema), prefilled -- not an empty section -- since the processor
    exposes no live acquisition_parameters of its own."""
    from Zou_lab_control.frontend.task_console import PanelConfig, PanelCard
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        # start a camera (publishes frame) + a Judge-occupancy processor (publishes occupied)
        _pick(console, ("camera", "live"))
        console._start_logic_node(console.logic_nodes[-1])
        _pick(console, ("processor", "Judge occupancy"))
        console._start_logic_node(console.logic_nodes[-1])
        # a Plot reading the processor's `occupied` -> its Edit shows the processor's form
        card = PanelCard(PanelConfig(kind="sites", source="value = occupied"),
                         parent=console.board, names_provider=console.hub.names)
        console._attach_card(card)
        console._edit_card(card)
        editor = console._panel_editors[id(card)]
        assert editor.source_form is not None                          # not an empty source section
        vals = editor.source_form.collect_values()
        assert {"calibration", "source", "ema"} <= set(vals)           # the processor's full params
        assert vals["source"] == "frame" and vals["calibration"] == ""  # prefilled defaults (#3)
    finally:
        console.shutdown()
        exp.close()
