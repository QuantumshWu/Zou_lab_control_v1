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


# ------------------------------------ coordinate axes = source param space (G1)
# confocal_gui's core coupling: a plot's axes ARE the producing source's
# parameter space, and a region selected on the plot writes that parameter back.
# For a camera-frame 2D panel the parameter is the qCMOS ROI: the image x/y are
# the REAL pixel coordinates (the ROI origin), NOT 0..N, and an area-select fills
# the ROI field.  This is the 2D analogue of the 1D scan-range writeback above.
def test_2d_panel_axes_use_source_roi_coordinates():
    """A 2D panel whose source reads a ROI-bearing camera signal builds its axes
    in the camera's real pixel space: x in [x0, x0+w-1], y in [y0, y0+h-1], with
    'Camera x/y (px)' labels -- so the displayed coordinates ARE the qCMOS ROI."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="2d", source="value = frame"))
    try:
        frame = np.random.rand(16, 24) * 100.0           # h=16, w=24
        # coord frame = the source's spatial region ENDPOINTS [x_min,x_max,y_min,y_max]
        region = [1648, 1672, 1144, 1160]                 # origin (1648,1144), 24x16
        card.refresh({"frame": frame, "__coord_frames__": {"frame": region}, "shot": 1})
        p = card.plotter
        assert np.isclose(p.x_array[0], 1648) and np.isclose(p.x_array[-1], 1648 + 24 - 1)
        assert np.isclose(p.y_array[0], 1144) and np.isclose(p.y_array[-1], 1144 + 16 - 1)
        assert p.ax.get_xlabel() == "Camera x (px)"
        assert p.ax.get_ylabel() == "Camera y (px)"
    finally:
        card.shutdown()


def test_2d_panel_axes_default_to_index_without_coord_frame():
    """No coordinate frame (the source isn't a ROI-bearing camera) -> the 2D axes
    fall back to plain 0..N indices, the generic behaviour."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="2d", source="value = frame"))
    try:
        frame = np.random.rand(16, 24) * 100.0
        card.refresh({"frame": frame, "shot": 1})
        p = card.plotter
        assert np.isclose(p.x_array[0], 0) and np.isclose(p.x_array[-1], 23)
        assert np.isclose(p.y_array[0], 0) and np.isclose(p.y_array[-1], 15)
    finally:
        card.shutdown()


def test_2d_panel_rebuilds_when_roi_shifts_same_shape():
    """A ROI that SHIFTS without resizing keeps the frame shape; the axes must
    still follow it.  The shape-only rebuild gate would skip this, so the panel
    also rebuilds when the source coordinate frame changes (else stale coords)."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="2d", source="value = frame"))
    try:
        frame = np.random.rand(16, 24) * 100.0
        card.refresh({"frame": frame, "__coord_frames__": {"frame": [1648, 1672, 1144, 1160]}, "shot": 1})
        assert np.isclose(card.plotter.x_array[0], 1648)
        # same-shape frame, SHIFTED region origin -> axes track the new origin
        card.refresh({"frame": frame, "__coord_frames__": {"frame": [100, 124, 200, 216]}, "shot": 2})
        assert np.isclose(card.plotter.x_array[0], 100)
        assert np.isclose(card.plotter.y_array[0], 200)
    finally:
        card.shutdown()


def _camera_console(roi=(1648, 64, 1144, 64)):
    """A console driving a 2D panel from a faked ROI-bearing camera, with the
    panel's Edit tab opened.  Only the lowest data source (the camera frame +
    its ROI) is faked; everything above is the real task-console path.  Returns
    (console, feed, cam, card, editor)."""
    import Zou_lab_control.frontend as zf
    from Zou_lab_control.frontend.task_console import TaskConsole
    from Zou_lab_control.neutral_atom.operations.feeds import CameraFrameFeed
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice

    class _RoiCamera(CameraDevice):
        def __init__(self, roi):
            self._exposure = 0.02
            self._roi = tuple(int(v) for v in roi)

        @property
        def exposure(self):
            return self._exposure

        @property
        def roi(self):
            return self._roi

        def configure(self, *, exposure=None, **kw):
            if exposure is not None:
                self._exposure = float(exposure)
            if kw.get("roi") is not None:
                self._roi = tuple(int(v) for v in kw["roi"])

        def acquire(self, frames=1, *, sequence=None, sequencer=None, stop=None, **kw):
            w, h = self._roi[1], self._roi[3]
            return [np.full((h, w), 50.0)]

    hub = SignalHub()
    cam = _RoiCamera(roi)
    feed = CameraFrameFeed(hub, cam)
    state = zf.TaskConsoleState(name="t", panels=[
        zf.PanelConfig(kind="2d", title="cam", size="2x2", source="value = frame")])
    console = TaskConsole(hub=hub, state=state, feeds=[feed])
    console._timer.stop()
    feed.step()
    console.refresh_once()
    card = console.cards[0]
    console._edit_card(card)
    editor = console._panel_editors[id(card)]
    editor.rebuild()
    return console, feed, cam, card, editor


def test_edit_area_select_writes_camera_region_and_apply_restarts_monitor():
    """The WRITE-back + RESTART half of G1: a region selected on a camera-frame 2D
    panel's Edit plot fills the acquisition ``region`` field as plot-coord ENDPOINTS
    [x_min,x_max,y_min,y_max] (NOT the device [x,w,y,h]); Apply restarts the feed so
    the camera re-arms AND the live Monitor panel re-acquires under the new window;
    the camera's device ROI is the INTERNAL conversion."""
    console, feed, cam, card, editor = _camera_console()
    try:
        # the selector callback is the GENERIC region writeback (not ROI-specific),
        # bound to BOTH the area selector and the zoom handler; the FEED keeps the
        # rectangle as endpoints in its {"region": [...]} param
        assert editor._plotter.area.callback.__name__ == "_read_region"
        assert editor._plotter.zoom.callback.__name__ == "_read_region"
        # a rectangle selected in pixel coords -> region ENDPOINTS, verbatim
        editor._plotter.area.range = [1670.0, 1690.0, 1166.0, 1186.0]
        editor._plotter.area.callback()
        assert editor._feed_widgets["region"].text() == "[1670, 1690, 1166, 1186]"
        # Apply on an IDLE feed converts region->device ROI internally, AUTO-STARTS
        # it (goes live), the Monitor re-acquires, and 'now:' tracks the applied
        # window (reported back as endpoints).
        assert not feed.running
        editor._restart_feed()
        assert feed.running                              # Apply made the source live
        assert cam.roi == (1670, 20, 1166, 20)           # device ROI = internal conversion
        assert np.isclose(card.plotter.x_array[0], 1670)
        assert np.isclose(card.plotter.x_array[-1], 1670 + 20 - 1)
        assert editor._feed_now_labels["region"].text() == "now: [1670, 1690, 1166, 1186]"
    finally:
        console.shutdown()


def test_2d_zoom_updates_region_from_view_area_overrides():
    """G-fix #1: ZOOM/PAN alone updates the region from the current view limits (the
    area selector is NOT required); when a rectangle IS drawn it OVERRIDES the
    view.  The region param is plot-coord ENDPOINTS (area else view precedence)."""
    console, feed, cam, card, editor = _camera_console()
    try:
        # ZOOM with no area selection -> region follows the view box (endpoints)
        editor._plotter.area.range = [None, None, None, None]
        editor._plotter.ax.set_xlim(1660, 1690)
        editor._plotter.ax.set_ylim(1190, 1160)        # image y runs high->low
        editor._plotter.zoom.callback()
        assert editor._feed_widgets["region"].text() == "[1660, 1690, 1160, 1190]"
        # now draw an area rectangle: it OVERRIDES the view even when the zoom
        # callback fires
        editor._plotter.area.range = [1672.0, 1682.0, 1170.0, 1180.0]
        editor._plotter.zoom.callback()
        assert editor._feed_widgets["region"].text() == "[1672, 1682, 1170, 1180]"
    finally:
        console.shutdown()


def test_camera_frames_per_cycle_is_adjustable_in_the_gui():
    """frames_per_cycle (one frame per emCCD trigger) is tunable IN task_console, not
    only via the constructor: it auto-surfaces in the 2D panel's Edit -> Acquisition
    form (it is in acquisition_parameters()), and Apply pushes it to the feed live so
    the second trigger (frame_1) starts publishing -- no notebook edit needed."""
    console, feed, cam, card, editor = _camera_console()
    try:
        assert "frames_per_cycle" in editor._feed_widgets        # auto-surfaced in the Edit form
        assert "frame_1" not in feed.published_signals()         # default: first trigger only
        editor._feed_widgets["frames_per_cycle"].setText("2")
        editor._restart_feed()                                   # the GUI Apply button
        assert feed.frames_per_cycle == 2                        # applied to the source live
        assert "frame_1" in feed.published_signals()             # second trigger now published
    finally:
        console.shutdown()


def test_feed_now_labels_track_applied_params_on_tick():
    """The Acquisition 'now:' references follow the source's CURRENT values via the
    console's per-tick refresh of the VISIBLE Edit tab (one general hook, not a
    per-field signal) -- so a queued edit on a running feed shows as applied once
    the loop picks it up."""
    import time
    console, feed, cam, card, editor = _camera_console()
    try:
        console.tabs.setCurrentWidget(editor)        # the per-tick hook refreshes the visible editor
        feed.start(rate_hz=50.0)
        # region endpoints [1700,1716,1180,1196] -> internal device ROI (1700,16,1180,16)
        console._restart_feed(feed, {"region": [1700, 1716, 1180, 1196]})   # running -> queued
        deadline = time.monotonic() + 2.0
        while cam.roi != (1700, 16, 1180, 16) and time.monotonic() < deadline:
            console.refresh_once()
            time.sleep(0.02)
        console.refresh_once()                        # a tick after the loop applied
        assert cam.roi == (1700, 16, 1180, 16)
        assert editor._feed_now_labels["region"].text() == "now: [1700, 1716, 1180, 1196]"
    finally:
        feed.stop()
        console.shutdown()


def test_show_task_console_auto_starts_passed_feeds():
    """show_task_console makes a passed-in producer feed LIVE on open, so the
    Monitor streams without the caller remembering feed.start(); shutdown stops it.
    start() is idempotent, so an already-running feed keeps its own rate."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import show_task_console
    from Zou_lab_control.neutral_atom.operations.feeds import CameraFrameFeed
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    hub = SignalHub()
    feed = CameraFrameFeed(hub, na.connect("virtual").devices.camera)
    assert not feed.running
    console = show_task_console(hub=hub, feeds=[feed], title="autostart-test")
    try:
        assert feed.running                          # launched on open, no manual start()
    finally:
        console.shutdown()
    assert not feed.running                           # shutdown stopped it


def test_apply_roi_to_running_feed_applies_in_loop_no_restart():
    """G-fix #2 (architecture): Apply on a RUNNING acquisition does NOT stop/start
    the feed from the GUI thread (which would stall the GUI and could run two
    acquire() calls on one camera).  The edit is queued and the loop's OWN thread
    applies it between shots, so the camera re-arms, the Monitor keeps streaming,
    and the SAME thread keeps running.  (See test_measurement_spec_feed for the
    no-concurrent-acquire invariant on a blocking camera.)"""
    import time
    console, feed, cam, card, editor = _camera_console()
    try:
        feed.start(rate_hz=50.0)
        time.sleep(0.08)
        thread_before = feed._thread
        console._restart_feed(feed, {"region": [100, 120, 200, 220]})   # endpoints -> device ROI (100,20,200,20)
        deadline = time.monotonic() + 2.0
        while cam.roi != (100, 20, 200, 20) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert cam.roi == (100, 20, 200, 20)        # applied by the loop, between shots
        assert feed.running                          # still streaming
        assert feed._thread is thread_before         # SAME thread -- no GUI-thread restart
    finally:
        feed.stop()
        console.shutdown()


def test_zoom_updates_measurement_scan_range():
    """G-fix #1 for a measurement panel: ZOOM on the 1D scan plot updates the
    measurement form's scan x-range (read by Start), the same way confocal binds
    zoom + area to ``_read_range``."""
    exp = _calibrated_virtual_session(grid=(2, 3))
    specs = exp.readout.measurement_specs()
    console = _console(specs)
    try:
        spec, card, editor, form = _open_measurement_editor(console, 0)
        _, lo, hi, pts = form._widgets["t_off"]
        lo.setValue(0.0); hi.setValue(120.0); pts.setValue(5)
        form._widgets["shots"][1].setValue(4)
        console._start_measurement(form)
        feed = console._meas_feed
        feed.stop(); feed.run_to_completion(); console.refresh_once()
        editor.rebuild()
        assert editor._plotter.zoom.callback.__name__ == "_read_scan_range"
        # ZOOM to a sub-range -> the form's Min/Max follow the view
        editor._plotter.ax.set_xlim(30.0, 70.0)
        editor._plotter.zoom.callback()
        assert form._widgets["t_off"][1].value() == pytest.approx(30.0, abs=1e-6)
        assert form._widgets["t_off"][2].value() == pytest.approx(70.0, abs=1e-6)
    finally:
        console.shutdown()
