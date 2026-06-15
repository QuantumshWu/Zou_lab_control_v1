"""MECHANICAL guard for the P5 task-console Measurement section (FRONTEND).

P5 fills the Control tab's reserved Measurement placeholder with an
auto-generated, validated parameter form + one-click Start that streams a
scanned measurement into a Monitor result panel.  This pins the three contracts
the implementation must keep:

1. plumbing -- passing ``measurements`` enables the section, the combo lists the
   spec names, and selecting a spec generates the right widgets (the required
   ``capture_radius`` control exists), all read back BY KIND (no free-text eval);
2. one-click Start -- the Start slot builds a ``ScannedMeasurementFeed``,
   registers it into ``console.feeds`` (so Pause Meas. manages it) AND adds a
   result panel; ``run_to_completion()`` then leaves a DECAYING survival curve
   on the hub;
3. 1d x-y -- a 1-D panel fed an ``(N, 2)`` value plots y vs COL 0 (the scan x),
   not ``arange``; a plain vector still plots vs index.

These build the widgets directly on the offscreen Qt platform, so they do NOT
pull in the flaky demo GUI fixtures.
"""

from __future__ import annotations

from pathlib import Path
import sys

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


def _calibrated_virtual_session(grid=(2, 3)):
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    exp.readout.thresholds(frames=40, display=False)
    return exp


def _console(measurements):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    console = TaskConsole(hub=SignalHub(), state=default_console_state(),
                          feeds=[], measurements=measurements, window_px=(1200, 800))
    console._timer.stop()       # deterministic: drive refresh_once() / poll ourselves
    return console


def _open_measurement_editor(console, index=0):
    """Create a measurement panel via the header's Add Panel and return
    (spec, result_card, panel_editor, meas_form) -- the form lives in the
    panel's OWN Edit tab now, not a global Control launcher."""
    spec = console.measurements[index]
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == ("measurement", spec.name))
    kc.setCurrentIndex(i)
    console._add_panel()
    card = next(c for c in console.cards if c.config.params.get("measurement") == spec.name)
    editor = console._panel_editors[id(card)]
    return spec, card, editor, editor.meas_panel


# ----------------------------------------------------------------- plumbing
def test_no_measurements_no_measurement_entries_in_add_panel():
    """With no measurements wired, the Add Panel combo lists ONLY plot kinds (no
    measurement entries), and there is no global measurement form (the empty
    Control launcher is gone)."""
    console = _console(())
    try:
        kc = console.kind_combo
        data = [kc.itemData(i) for i in range(kc.count())]
        assert not any(isinstance(d, tuple) and d and d[0] == "measurement" for d in data)
        assert console.measurement_panel is None
        assert console.measurement_group is None
    finally:
        console.shutdown()


def test_measurements_listed_in_add_panel():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(specs)
    try:
        kc = console.kind_combo
        entries = [kc.itemData(i) for i in range(kc.count())]
        meas = [d[1] for d in entries if isinstance(d, tuple) and d and d[0] == "measurement"]
        assert meas == [s.name for s in specs]
    finally:
        console.shutdown()


def test_measurement_panel_edit_generates_typed_form_with_required_param():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(specs)
    try:
        spec, card, editor, form = _open_measurement_editor(console, 0)
        # the measurement's parameters are an auto-generated form IN the Edit tab
        assert form is not None
        assert set(form._widgets) == {d.key for d in spec.params}
        assert form._widgets["t_off"][0] == "axis_range"     # min/max/points triplet
        assert form._widgets["shots"][0] == "int"
        assert form._widgets["capture_radius"][0] == "float"
        assert form._widgets["per_site"][0] == "bool"
        vals = form.collect_values()
        assert set(vals) == {d.key for d in spec.params}
        assert isinstance(vals["t_off"], tuple) and len(vals["t_off"]) == 3
        assert vals["capture_radius"] == pytest.approx(spec.param("capture_radius").default)
        assert isinstance(vals["per_site"], bool)
    finally:
        console.shutdown()


# ----------------------------------------------------------------- Start RE-RUNS
def test_edit_start_reruns_into_same_result_panel():
    """Start in a measurement panel's Edit streams the scan into THAT panel
    (reusing the card Add Panel created -- no pile-up), and the result is the
    SAME decaying survival contract real hardware runs."""
    exp = _calibrated_virtual_session(grid=(5, 7))
    specs = exp.readout.measurement_specs()
    console = _console(specs)
    try:
        from Zou_lab_control.neutral_atom.operations.feeds import ScannedMeasurementFeed

        spec, card, editor, form = _open_measurement_editor(console, 0)
        _, lo, hi, pts = form._widgets["t_off"]
        lo.setValue(0.0); hi.setValue(120.0); pts.setValue(5)
        form._widgets["shots"][1].setValue(8)

        n_cards = len(console.cards)
        console._start_measurement(form)
        feed = console._meas_feed
        assert isinstance(feed, ScannedMeasurementFeed) and feed in console.feeds
        # NO new card: Start re-runs the panel Add Panel already created
        assert len(console.cards) == n_cards
        result = [c for c in console.cards if c.config.params.get("measurement") == spec.name]
        assert len(result) == 1 and result[0] is card

        feed.stop()
        feed.run_to_completion()
        console.refresh_once()

        x = console.hub.latest(spec.x_key)
        y = console.hub.latest(spec.y_key)
        assert x.shape == (5,) and y.shape == (5,)
        assert np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 1))
        assert y[0] > y[-1] + 0.3
        assert np.allclose(card.plotter.data_x[:, 0], x)
        assert "T" in form.status.text() and "µK" in form.status.text()
    finally:
        console.shutdown()


def test_edit_start_then_stop_releases_controls_and_stops_feed():
    exp = _calibrated_virtual_session(grid=(2, 3))
    specs = exp.readout.measurement_specs()
    console = _console(specs)
    try:
        spec, card, editor, form = _open_measurement_editor(console, 0)
        _, lo, hi, pts = form._widgets["t_off"]
        lo.setValue(0.0); hi.setValue(60.0); pts.setValue(3)
        form._widgets["shots"][1].setValue(2)

        console._start_measurement(form)
        feed = console._meas_feed
        console._stop_measurement()
        feed.stop()
        assert feed.running is False
        assert form.start_button.isEnabled()
        assert not form.stop_button.isEnabled()
    finally:
        console.shutdown()


# ----------------------------------------------------------------- 1d x-y
def test_1d_panel_xy_curve_uses_col0_as_x():
    """An (N, 2) value plots y vs col 0 -- but ONLY when the panel is explicitly
    flagged xy=True (the marker _ensure_result_panel sets on result cards)."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", source="value = xy",
                                 params={"xy": True, "xlabel": "T (s)", "ylabel": "Surv"}))
    try:
        xy = np.column_stack([np.array([0.0, 10.0, 20.0, 30.0]),
                              np.array([1.0, 0.8, 0.5, 0.2])])
        card.refresh({"xy": xy, "shot": 1})
        assert card.plotter is not None
        # data_x is COL 0 of the (N, 2) value, not arange(N)
        assert np.allclose(card.plotter.data_x[:, 0], [0.0, 10.0, 20.0, 30.0])
        assert np.allclose(card.plotter.data_y[:, 0], [1.0, 0.8, 0.5, 0.2])
        assert card.plotter.ax.get_xlabel() == "T (s)"
    finally:
        card.shutdown()


def test_1d_panel_plain_vector_still_uses_index():
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", source="value = vec"))
    try:
        card.refresh({"vec": np.array([3.0, 1.0, 4.0, 1.0, 5.0]), "shot": 1})
        assert card.plotter is not None
        assert np.allclose(card.plotter.data_x[:, 0], np.arange(5))
    finally:
        card.shutdown()


def test_1d_panel_without_xy_marker_flattens_n_by_2():
    """A plain 1d panel (no xy marker) that happens to produce an (N, 2) value
    flattens to a vector vs index -- the x-y meaning is opt-in, never inferred
    from shape alone (avoids the silent-reinterpretation trap)."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", source="value = arr"))
    try:
        arr = np.column_stack([np.array([0.0, 10.0, 20.0]),
                               np.array([1.0, 0.8, 0.5])])
        card.refresh({"arr": arr, "shot": 1})
        assert card.plotter is not None
        # flattened to 6 points vs index 0..5, NOT read as a 3-point x-y curve
        assert np.allclose(card.plotter.data_x[:, 0], np.arange(6))
        assert np.allclose(card.plotter.data_y[:, 0], arr.reshape(-1))
    finally:
        card.shutdown()
