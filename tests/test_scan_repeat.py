"""Contract: a finite scan can be RE-RUN ``repeat`` times, and the node only FILLS a raw
``(repeat, points, dim)`` block point-by-point -- it does NOT combine the repeats.  HOW the repeats
are displayed (average / add / replace / roll / new) is the PLOT's ``repeat_mode`` (#3), which
reduces the repeat axis via the single owned helper ``frontend.live.reduce_repeat``.

``repeat`` is a MEASUREMENT parameter: a positive int, or ``inf`` (free-run, keep only the most
recent ``REPEAT_RING`` passes).  The task console's measurement Edit exposes ``Repeat`` (with an
"infinity" special value) but NOT the combine selector -- that moved to each plot panel's Setting.
"""

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode, REPEAT_RING


class _Reducer:
    n_series = 1
    labels = ("x", "y", "z")


class _Axis:
    label = "x"
    unit = ""
    values = (10.0, 20.0, 30.0)


class _CountingMeasurement:
    """measure() returns a monotonically increasing call index, so each pass yields DIFFERENT
    values for the same point -- letting the test verify the raw per-pass block + the reductions."""
    def __init__(self):
        self.axis = _Axis()
        self.reducer = _Reducer()
        self._calls = 0

    def measure(self, value, index):
        v = float(self._calls)
        self._calls += 1
        return v


def test_scan_publishes_a_raw_repeat_block_not_a_combined_curve():
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=2)
    assert node.n_points == 3
    assert node.total_points == 6              # 3 points x 2 passes
    node.run_to_completion()
    assert node.finished and node.points_done == 6

    raw = np.asarray(hub.latest("m_y"))
    assert raw.shape == (2, 3, 1)              # RAW (repeat, points, dim) -- NOT a combined (3,) curve
    # pass 1 measured calls 0,1,2 (slot 0) ; pass 2 measured 3,4,5 (slot 1) -- node did NOT average.
    assert np.allclose(raw[0, :, 0], [0.0, 1.0, 2.0])
    assert np.allclose(raw[1, :, 0], [3.0, 4.0, 5.0])
    assert np.allclose(hub.latest("m_x"), [10.0, 20.0, 30.0])
    assert float(hub.latest("m_scan_done")) == 1.0


def test_repeat_one_is_a_single_pass_block():
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=1)
    assert node.total_points == 3
    node.run_to_completion()
    raw = np.asarray(hub.latest("m_y"))
    assert raw.shape == (1, 3, 1)
    assert np.allclose(raw[0, :, 0], [0.0, 1.0, 2.0])


def test_inf_repeat_is_a_free_running_ring():
    """``repeat=inf`` never finishes and keeps only the most recent REPEAT_RING passes; the raw
    block's repeat axis is the ring length, rolled so the newest pass is LAST."""
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=float("inf"))
    assert node.total_points == 0              # open-ended (free-run)
    for _ in range(node.n_points * 3):         # run 3 whole passes worth of points
        node.step()
        assert not node.finished               # inf -> never self-stops
    raw = np.asarray(hub.latest("m_y"))
    assert raw.shape == (REPEAT_RING, 3, 1)
    # the most recent pass measured calls 6,7,8 and is the LAST filled slice (rolled to the end).
    assert np.allclose(raw[-1, :, 0], [6.0, 7.0, 8.0])


def test_reduce_repeat_modes_collapse_the_repeat_axis():
    """The PLOT's reduction (the ONE owned helper) over a raw (repeat, points, dim) block."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data, REPEAT_MODES
    raw = np.array([[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]])    # (2, 3, 1)
    assert np.allclose(reduce_repeat(raw, "average")[:, 0], [1.5, 2.5, 3.5])
    assert np.allclose(reduce_repeat(raw, "add")[:, 0], [3.0, 5.0, 7.0])
    assert np.allclose(reduce_repeat(raw, "replace")[:, 0], [3.0, 4.0, 5.0])
    assert np.allclose(reduce_repeat(raw, "roll")[:, 0], [3.0, 4.0, 5.0])
    # 'new' keeps every repeat as its own column -> one line per repeat (1-D)
    new = reduce_repeat(raw, "new")
    assert new.shape == (3, 2) and np.allclose(new, [[0, 3], [1, 4], [2, 5]])
    assert repeats_with_data(raw) == 2
    assert set(REPEAT_MODES) == {"average", "add", "replace", "roll", "new"}


def test_reduce_repeat_average_ignores_not_yet_measured_repeats():
    """``average`` is a running mean over the repeats that HAVE data (NaN = not yet measured), so the
    magnitude is stable no matter how many passes finished -- this is what the user means by average."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data
    raw = np.array([[[0.0], [1.0], [2.0]], [[np.nan], [np.nan], [np.nan]]])    # only pass 0 has data
    assert np.allclose(reduce_repeat(raw, "average")[:, 0], [0.0, 1.0, 2.0])
    assert repeats_with_data(raw) == 1


def test_reduce_repeat_handles_a_2d_grid_block():
    """A 2-D scan publishes a raw (repeat, n0, n1) block; the reduction drops the repeat axis to the
    (n0, n1) map (so a 2-D panel shows one reduced image; 'new' is offered only for 1-D)."""
    from Zou_lab_control.frontend.live import reduce_repeat
    raw = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    assert reduce_repeat(raw, "average").shape == (2, 3)
    assert np.allclose(reduce_repeat(raw, "replace"), raw[1])


def test_reduce_repeat_passes_through_already_reduced_arrays():
    """A non-3-D array (an already-reduced curve, a plain image, a scalar) has NO repeat axis and
    must pass through untouched -- so a sitemap image is never mistaken for a repeat stack."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data
    img = np.arange(6.0).reshape(2, 3)
    assert np.allclose(reduce_repeat(img, "average"), img)     # 2-D image -> unchanged
    assert repeats_with_data(img) == 1


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


def test_measurement_edit_exposes_repeat_no_combine_camera_has_both(monkeypatch):
    """A SCAN measurement's Edit shows ``Repeat`` (with an ∞ free-run special value) but NOT a
    combine combo -- the combine moved to the plot's ``repeat mode`` Setting (#3).  A CAMERA (also a
    measurement) keeps Repeat + Update mode (it averages frames).  A processor has neither."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.task_console import LogicNodeConfig

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        row = console._add_logic_node(LogicNodeConfig(kind="measurement", name="Pulse scan"))
        console._edit_logic_node(row)
        ed = console._logic_editors[id(row)]
        assert ed._repeat_spin is not None
        assert ed._update_mode_combo is None             # the combine selector is the PLOT's, not here
        assert ed._repeat_spin.minimum() == 0            # 0 = the "∞" special value (free-run)
        ed._repeat_spin.setValue(5)
        assert int(ed.collect_values()["repeat"]) == 5
        ed._repeat_spin.setValue(0)
        assert ed.collect_values()["repeat"] == "inf"    # 0 serialises as the free-run sentinel

        crow = console._add_logic_node(LogicNodeConfig(kind="camera", name="Camera"))
        console._edit_logic_node(crow)
        ced = console._logic_editors[id(crow)]
        assert ced._repeat_spin is not None and ced._update_mode_combo is not None
        cam_modes = [ced._update_mode_combo.itemText(i) for i in range(ced._update_mode_combo.count())]
        assert "roll" in cam_modes and "add" not in cam_modes

        prow = console._add_logic_node(LogicNodeConfig(kind="processor", name="Judge occupancy"))
        console._edit_logic_node(prow)
        assert console._logic_editors[id(prow)]._repeat_spin is None
    finally:
        console.shutdown()


def test_plot_setting_has_a_repeat_mode_combo_that_reduces_the_block(monkeypatch):
    """A plot panel's Setting offers a ``repeat mode`` combo (the PLOT owns the reduction, decoupled
    from the node); switching it re-renders.  A 1-D panel offers all modes incl. ``new``; a 2-D panel
    offers all but ``new``."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        card = console.cards[0]
        card.config.role = "plot"
        card.config.kind = "1d"
        card._build_settings()
        assert getattr(card, "repeat_mode_combo", None) is not None
        modes = [card.repeat_mode_combo.itemText(i) for i in range(card.repeat_mode_combo.count())]
        assert "average" in modes and "new" in modes          # 1-D offers 'new'
        card._on_repeat_mode("add")
        assert card.config.params["repeat_mode"] == "add"     # persisted on the PLOT, not the node

        card.config.kind = "2d"
        card._build_settings()
        modes2 = [card.repeat_mode_combo.itemText(i) for i in range(card.repeat_mode_combo.count())]
        assert "average" in modes2 and "new" not in modes2    # 2-D: no per-repeat lines
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
    btns = [b.text() for b in w.findChildren(FluentButton)]
    assert btns.count("column_stack") == 2 and btns.count("grid") == 2, btns
    assert w._api_scan_code is not None and w._scan_code is not None
    assert not w._api_scan_code.toPlainText().strip()
