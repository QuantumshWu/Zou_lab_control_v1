"""Contract: a finite scan can be RE-RUN ``repeat`` times, and the node only FILLS a raw
``(R,P,*data_shape)`` block point-by-point -- it does NOT combine the repeats.  HOW the repeats
are displayed (average / add / replace / roll / new) is the PLOT's ``repeat_mode`` (#3), which
reduces the repeat axis via the single owned helper ``frontend.live.reduce_repeat``.

``repeat`` is the ONE MEASUREMENT parameter, 0 = ∞ (#H3u-2): ``K`` keeps a K-deep block (K passes
averaged) then STOPS; ``0`` rolls a 1-deep ring forever (the live monitor).  There is NO separate
free-run toggle -- 0 IS infinite, the same semantics everywhere.  The task console's measurement Edit
exposes that ONE ``Repeat (0 = ∞)`` integer, but NOT the combine selector -- that moved to each plot
panel's Setting.
"""

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode


class _Reducer:
    data_shape = (1,)
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
        return np.asarray([v])


def test_scan_publishes_a_raw_repeat_block_not_a_combined_curve():
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=2)
    assert node.n_points == 3
    assert node.total_points == 6              # 3 points x 2 passes
    node.run_to_completion()
    assert node.finished and node.points_done == 6

    raw = np.asarray(hub.latest("m_y"))
    assert raw.shape == (2, 3, 1)              # RAW (R,P,*data_shape) -- NOT a combined (3,) curve
    # pass 1 measured calls 0,1,2 (slot 0) ; pass 2 measured 3,4,5 (slot 1) -- node did NOT average.
    assert np.allclose(raw[0, :, 0], [0.0, 1.0, 2.0])
    assert np.allclose(raw[1, :, 0], [3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        hub.latest("m_x"), np.asarray([[[10.0], [20.0], [30.0]]]))
    assert node.finished                       # completion is control state, not a data signal


def test_repeat_one_is_a_single_pass_block():
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=1)
    assert node.total_points == 3
    node.run_to_completion()
    raw = np.asarray(hub.latest("m_y"))
    assert raw.shape == (1, 3, 1)
    assert np.allclose(raw[0, :, 0], [0.0, 1.0, 2.0])


def test_repeat_zero_is_infinite_rolling_one_deep():
    """``repeat=0`` = ∞ (the ONE infinite knob, no free-run toggle, #H3u-2): never finishes, a 1-deep
    rolling ring showing the LATEST pass (the live monitor)."""
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, _CountingMeasurement(), x_key="x", y_key="y",
                                  prefix="m_", repeat=0)
    assert node.total_points == 0              # open-ended (∞)
    assert node._ring == 1                     # ∞ rolls a 1-deep ring (not an old repeat-deep ring)
    for _ in range(node.n_points * 3):         # run 3 whole passes worth of points
        node.step()
        assert not node.finished               # ∞ -> never self-stops
    raw = np.asarray(hub.latest("m_y"))
    assert raw.shape == (1, 3, 1)              # 1-deep rolling ring = the latest pass
    # the most recent pass measured calls 6,7,8 and is the (only) filled slice.
    assert np.allclose(raw[-1, :, 0], [6.0, 7.0, 8.0])


def test_reduce_repeat_modes_collapse_the_repeat_axis():
    """The PLOT's reduction (the ONE owned helper) over a raw (R,P,*data_shape) block."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data, REPEAT_MODES
    raw = np.array([[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]])    # (2, 3, 1)
    assert np.allclose(reduce_repeat(raw, "average")[:, 0], [1.5, 2.5, 3.5])
    assert np.allclose(reduce_repeat(raw, "add")[:, 0], [3.0, 5.0, 7.0])
    assert np.allclose(reduce_repeat(raw, "replace")[:, 0], [3.0, 4.0, 5.0])
    assert np.allclose(reduce_repeat(raw, "roll")[:, 0], [3.0, 4.0, 5.0])
    # 'create' keeps every repeat as its own column -> one line per repeat (1-D)
    created = reduce_repeat(raw, "create")
    assert created.shape == (3, 2) and np.allclose(created, [[0, 3], [1, 4], [2, 5]])
    assert repeats_with_data(raw) == 2
    assert set(REPEAT_MODES) == {"average", "add", "replace", "roll", "create"}


def test_reduce_repeat_average_ignores_not_yet_measured_repeats():
    """``average`` is a running mean over the repeats that HAVE data (NaN = not yet measured), so the
    magnitude is stable no matter how many passes finished -- this is what the user means by average."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data
    raw = np.array([[[0.0], [1.0], [2.0]], [[np.nan], [np.nan], [np.nan]]])    # only pass 0 has data
    assert np.allclose(reduce_repeat(raw, "average")[:, 0], [0.0, 1.0, 2.0])
    assert repeats_with_data(raw) == 1


def test_reduce_repeat_handles_a_2d_grid_block():
    """A scalar 2-D scan flattens only its physical P axis; logical geometry stays in the schema."""
    from Zou_lab_control.frontend.live import reduce_repeat
    raw = np.arange(2 * 6, dtype=float).reshape(2, 6, 1)
    assert reduce_repeat(raw, "average").shape == (6, 1)
    assert np.allclose(reduce_repeat(raw, "replace"), raw[1])


def test_reduce_repeat_passes_through_already_reduced_arrays():
    """An external datum has no R axis only when the caller marks that boundary explicitly."""
    from Zou_lab_control.frontend.live import reduce_repeat, repeats_with_data
    img = np.arange(6.0).reshape(2, 3)
    assert np.allclose(reduce_repeat(img, "average", has_repeat=False), img)
    assert repeats_with_data(img, has_repeat=False) == 1


def test_repeat_is_a_measurement_param_auto_injected_as_one_knob(monkeypatch):
    """``repeat`` is a MEASUREMENT-layer param (a measurement decides how many times to acquire -- the
    plot never can), AUTO-INJECTED into the measurement / camera auto-form as the ONE integer knob with
    0 = ∞ (no separate free-run toggle, #H3u-2).  A processor has neither.  ``_repeat_value`` -> int."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.task_console import LogicNodeConfig

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        for kind, name in (("measurement", "Pulse scan"), ("camera", "Camera")):
            row = console._add_logic_node(LogicNodeConfig(kind=kind, name=name))
            console._edit_logic_node(row)
            form = console._logic_editors[id(row)].form
            w, d = form._widgets, form._decls
            assert "repeat" in w and d["repeat"].kind == "int"
            assert "free_run" not in d and "free_run" not in w   # 0 = ∞, NO separate toggle
            assert w["repeat"].minimum() == 0                    # 0 is the ∞ sentinel

        # the node build turns the form value into ONE int (0 = ∞); no tuple, no free_run.
        assert console._repeat_value({"repeat": 5}) == 5
        assert console._repeat_value({"repeat": 0}) == 0
        assert console._repeat_value({}) == 0                    # missing -> ∞ default

        prow = console._add_logic_node(LogicNodeConfig(kind="processor", name="Readout fidelity"))
        console._edit_logic_node(prow)
        pw = console._logic_editors[id(prow)].form._widgets
        assert "repeat" not in pw and "free_run" not in pw   # a processor does not acquire repeats
    finally:
        console.shutdown()


def test_plot_setting_has_only_repeat_mode_not_repeat(monkeypatch):
    """The PLOT Setting auto-renders ONLY ``repeat mode`` (how to DISPLAY the repeats) -- NOT a
    ``repeat`` count, because the plot can never tell a measurement how many times to run (#H3l).  A
    1-D panel offers ``create``; a 2-D panel does not.  Editing routes through _set_param."""
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
        assert "repeat_mode" in card.param_widgets
        assert "repeat" not in card.param_widgets            # the COUNT is the measurement's, not here
        combo = card.param_widgets["repeat_mode"]
        modes = [combo.itemText(i) for i in range(combo.count())]
        assert "average" in modes and "create" in modes        # 1-D offers 'create'
        card._set_param("repeat_mode", "add")
        assert card.config.params["repeat_mode"] == "add"      # persisted on the PLOT

        card.config.kind = "2d"
        card._build_settings()
        combo2 = card.param_widgets["repeat_mode"]
        modes2 = [combo2.itemText(i) for i in range(combo2.count())]
        assert "average" in modes2 and "create" not in modes2  # 2-D: no per-repeat lines

        # A DISTRIBUTION DEFAULTS to 'pool' (bin ALL repeats together) and now ALSO offers the GENERIC
        # base verbs (average/add/replace) -- ONE reduce_repeat collapses the repeat axis the same way
        # for any kind (#issue-1).  It does NOT offer the trace-only 'roll'.
        card.config.kind = "hist"
        card._build_settings()
        combo3 = card.param_widgets["repeat_mode"]
        modes3 = [combo3.itemText(i) for i in range(combo3.count())]
        assert modes3[0] == "pool"                                 # pool is the dist's canonical default (first)
        assert {"average", "add", "replace"} <= set(modes3)        # the base verbs are generic across kinds
        assert "create" in modes3                                  # a dist CAN create: one overlaid histogram per repeat
        assert "roll" not in modes3                                # 'roll' is a trace-only verb (no rolling histogram)
    finally:
        console.shutdown()


def test_pulse_slots_form_has_one_selected_sweep_program(monkeypatch):
    """The selected strategy owns one program while API resting values stay separate."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget
    from Zou_lab_control.frontend.qt_fluent import FluentButton

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = _PulseSlotsWidget()
    w.rebuild(api_rows=[("a1", "image_duration", "duration", "1", "us", 5.0)],
              scan_rows=[("probe_duration", "duration", "2", "ns", "probe")],
              hardware_program="scan_table = np.array([[10], [20]])", program_id="template-a")
    values = w.values_dict()
    assert set(values) == {"program_id", "api", "sweep_kind", "program"}
    assert values["program_id"] == "template-a"
    assert values["api"] == {"a1": 5.0}
    assert "[[10], [20]]" in values["program"]
    btns = [b.text() for b in w.findChildren(FluentButton)]
    assert btns.count("column_stack") == 1 and btns.count("grid") == 1, btns
    assert w._program_code is not None and w._sweep_combo is not None
