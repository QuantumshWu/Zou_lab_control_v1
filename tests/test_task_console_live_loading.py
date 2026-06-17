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


def test_task_logic_node_publishes_calibration_when_started():
    """A calibrate Task added as a Logic node is STOPPED; Start runs it (its mid-run
    + result land on the hub under cal_).  The whole loading readout is composed from
    these decoupled nodes -- no monolithic composite."""
    exp = _calibrated_virtual_session()
    console = _console(exp)
    try:
        from Zou_lab_control.neutral_atom.operations.logic import CalibrateReadoutTask

        _pick(console, ("task", "Calibrate readout"))
        row = console.logic_nodes[-1]
        assert console._logic_nodes[id(row)] is None      # stopped
        assert "cal_centers" not in console.hub.names()

        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        assert isinstance(node, CalibrateReadoutTask)
        deadline = time.monotonic() + 12.0
        while "cal_centers" not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.05)
        names = set(console.hub.names())
        assert {"cal_centers", "cal_thresholds", "cal_frame"} <= names
        assert np.asarray(console.hub.latest("cal_centers")).shape == (12, 2)
    finally:
        console.shutdown()
        exp.close()
