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


def test_scan_repeat_one_is_a_single_pass():
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=1)
    assert node.total_points == 3
    node.run_to_completion()
    # one pass -> the raw call indices 0,1,2 (no averaging)
    assert np.allclose(hub.latest("m_y"), [0.0, 1.0, 2.0])


def test_measurement_edit_exposes_repeat_camera_does_not(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.task_console import LogicNodeConfig

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        # a scanned measurement node Edit HAS a Repeat spin, and collect_values carries it
        row = console._add_logic_node(LogicNodeConfig(kind="measurement", name="Pulse scan"))
        console._edit_logic_node(row)
        ed = console._logic_editors[id(row)]
        assert ed._repeat_spin is not None
        ed._repeat_spin.setValue(5)
        assert int(ed.collect_values()["repeat"]) == 5
        # a camera node Edit does NOT (a continuous camera does not re-run a finite sweep)
        crow = console._add_logic_node(LogicNodeConfig(kind="camera", name="Camera"))
        console._edit_logic_node(crow)
        assert console._logic_editors[id(crow)]._repeat_spin is None
    finally:
        console.shutdown()
