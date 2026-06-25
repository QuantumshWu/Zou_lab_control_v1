"""Contract: a finite scan can be RE-RUN ``repeat`` times, averaging each point.

This is the confocal "Repeat" model exposed end-to-end: the whole sweep runs ``repeat`` passes,
each x point's published value is the running MEAN over the passes done so far, and the node
self-stops only after the last point of the last pass.  Both scan node kinds
(:class:`ScannedMeasurementNode`, frame-reducing) and :class:`PulseScanNode` (device-driving,
decoupled y) support it, and the task console's measurement Edit exposes a "Repeat" spinbox
(but a camera / processor / task does NOT -- only a scanned measurement re-runs a sweep).
"""

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode


class _Reducer:
    n_series = 1
    labels = ("x", "y", "z")


class _Axis:
    label = "x"
    unit = ""
    values = (10.0, 20.0, 30.0)


class _CountingMeasurement:
    """measure() returns a monotonically increasing call index, so each pass yields DIFFERENT
    values for the same point -- letting the test verify the per-point mean over passes."""
    def __init__(self):
        self.axis = _Axis()
        self.reducer = _Reducer()
        self._calls = 0

    def measure(self, value, index):
        v = float(self._calls)
        self._calls += 1
        return v


def test_scan_repeat_averages_each_point_over_passes():
    hub = SignalHub()
    meas = _CountingMeasurement()
    node = ScannedMeasurementNode(hub, meas, x_key="x", y_key="y", prefix="m_", repeat=2)

    assert node.n_points == 3
    assert node.total_points == 6              # 3 points x 2 passes
    node.run_to_completion()
    assert node.finished
    assert node.points_done == 6               # all points of all passes

    # pass 1 measures calls 0,1,2 ; pass 2 measures 3,4,5 -> per-point mean = (0+3)/2, (1+4)/2, ...
    y = hub.latest("m_y")
    assert np.allclose(y, [1.5, 2.5, 3.5])
    # x axis is the swept values, stable
    assert np.allclose(hub.latest("m_x"), [10.0, 20.0, 30.0])
    # scan_done is 1.0 only after the final pass completes
    assert float(hub.latest("m_scan_done")) == 1.0


def test_occupancy_advertises_rate_grid_only_with_a_grid():
    """A processor must advertise ONLY what it actually publishes: rate_grid is emitted (by
    transform) only when a grid shape is known, so published_signals/output_specs must NOT list it
    otherwise (else a picker shows it 'waiting' forever)."""
    from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor
    hub = SignalHub()
    no_grid = OccupancyProcessor(hub, source_expr={"inputs": ["frame"], "source": "value = signal"},
                                 grid_shape=None, prefix="o_")
    assert "o_rate_grid" not in no_grid.published_signals()
    assert not any(s.name == "o_rate_grid" for s in no_grid.output_specs())
    gridded = OccupancyProcessor(hub, source_expr={"inputs": ["frame"], "source": "value = signal"},
                                 grid_shape=(2, 3), prefix="g_")
    assert "g_rate_grid" in gridded.published_signals()
    assert any(s.name == "g_rate_grid" for s in gridded.output_specs())


def test_scan_repeat_one_is_a_single_pass():
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=1)
    assert node.total_points == 3
    node.run_to_completion()
    # one pass -> the raw call indices 0,1,2 (no averaging)
    assert np.allclose(hub.latest("m_y"), [0.0, 1.0, 2.0])


def test_repeat_mode_combines_passes_and_does_not_collide_with_base_update_mode():
    """``repeat_mode`` (the confocal "Repeat mode" combine selector) must combine the passes per
    point WITHOUT clobbering the base per-shot ``update_mode`` (they are distinct fields).  add =
    sum over passes, replace = latest pass.  pass1 measures calls 0,1,2 ; pass2 measures 3,4,5."""
    for mode, expect in (("add", [3.0, 5.0, 7.0]), ("replace", [3.0, 4.0, 5.0])):
        hub = SignalHub()
        node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                      prefix="m_", repeat=2, repeat_mode=mode)
        assert node.update_mode == "roll"        # base per-shot mode left at its pass-through default
        node.run_to_completion()
        assert np.allclose(hub.latest("m_y"), expect), mode


def test_measurement_edit_exposes_repeat_camera_does_not(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.task_console import LogicNodeConfig

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        # a scanned measurement node Edit HAS a Repeat spin AND an Update-mode combo; collect_values
        # carries both (repeat + update_mode), the latter offering the scan combine modes.
        row = console._add_logic_node(LogicNodeConfig(kind="measurement", name="Pulse scan"))
        console._edit_logic_node(row)
        ed = console._logic_editors[id(row)]
        assert ed._repeat_spin is not None and ed._update_mode_combo is not None
        ed._repeat_spin.setValue(5)
        ed._update_mode_combo.setCurrentText("add")
        vals = ed.collect_values()
        assert int(vals["repeat"]) == 5 and vals["update_mode"] == "add"
        # a CAMERA is a measurement too -> it ALSO has Repeat + Update mode (repeat = average N
        # frames; modes are the continuous set roll/average/replace, NOT the scan's add).
        crow = console._add_logic_node(LogicNodeConfig(kind="camera", name="Camera"))
        console._edit_logic_node(crow)
        ced = console._logic_editors[id(crow)]
        assert ced._repeat_spin is not None and ced._update_mode_combo is not None
        cam_modes = [ced._update_mode_combo.itemText(i) for i in range(ced._update_mode_combo.count())]
        assert "roll" in cam_modes and "add" not in cam_modes
        # a PROCESSOR is NOT a measurement -> no repeat controls
        prow = console._add_logic_node(LogicNodeConfig(kind="processor", name="Judge occupancy"))
        console._edit_logic_node(prow)
        assert console._logic_editors[id(prow)]._repeat_spin is None
    finally:
        console.shutdown()


def test_repeat_cur_pushed_to_panels_plotting_the_node(monkeypatch):
    """Decoupled `xN` feed: the console pushes a running node's `_repeat_cur` onto the plotter of
    any panel whose source signal the node publishes (confocal's controller-push), and leaves other
    panels alone."""
    import types
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        node = types.SimpleNamespace(repeat=2, _repeat_cur=2,
                                     published_signals=lambda: frozenset({"m_y"}))
        cards = console.cards
        assert cards, "demo console should have at least one panel"
        saved = [(c, c.plotter) for c in cards[:2]]      # restore real plotters before teardown
        try:
            cards[0].plotter = types.SimpleNamespace(repeat_cur=1)
            cards[0].config.inputs = ["m_y"]             # this panel plots the node's signal
            if len(cards) > 1:
                cards[1].plotter = types.SimpleNamespace(repeat_cur=1)
                cards[1].config.inputs = ["other"]       # unrelated panel
            console._push_repeat_cur(node)
            assert cards[0].plotter.repeat_cur == 2, "panel plotting the node's signal must get repeat_cur"
            if len(cards) > 1:
                assert cards[1].plotter.repeat_cur == 1, "an unrelated panel must be left alone"
        finally:
            for card, plotter in saved:
                card.plotter = plotter
    finally:
        console.shutdown()


def test_api_sweep_table_has_same_form_as_scan_table(monkeypatch):
    """The API sweep table must be the SAME FORM as the scan-slot scan table: a per-column legend +
    column_stack/grid template buttons + a code editor (one shared renderer).  A bare editor (no
    template buttons) would NOT be the same form."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget
    from Zou_lab_control.frontend.qt_fluent import FluentButton

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = _PulseSlotsWidget()
    w.rebuild(api_rows=[("a1", "duration", "1", "us", 5.0)],
              scan_rows=[("s0", "duration", "2", "ns", "probe")])
    # both tables render their OWN column_stack + grid template buttons -> 2 of each (api + scan)
    btns = [b.text() for b in w.findChildren(FluentButton)]
    assert btns.count("column_stack") == 2 and btns.count("grid") == 2, btns
    # both tables have a code editor; the api one starts blank (blank = keep fixed)
    assert w._api_scan_code is not None and w._scan_code is not None
    assert not w._api_scan_code.toPlainText().strip()      # api sweep is OPTIONAL (blank by default)
