"""#4 Edit-snapshot ROBUSTNESS guards (the per-panel Edit tab's interactive snapshot):

1. FIXED size = the figure's design size, so repeated rebuilds (e.g. scrolling the bins param, a
   structure knob that re-snapshots) keep it identical -- it can neither be stretched taller by the
   scroll-area layout (the reported "image balloons / cuts off after a few bin turns") nor squished.
2. the pin SURVIVES the canvas's own deferred ``_zlc_resync`` (what a showEvent fires): resync
   re-applies setFixedSize (min==max) when pinned, so it can never raise the floor past the ceiling
   and balloon the figure (symptom "改参数图变大小").
3. a Site map snapshot honours its lim mode (tight/normal) AND fixed lo/hi -- the ``sites`` branch
   used to OMIT relim/fixed in the snapshot builder, so a sitemap's lim never re-rendered (symptom 3).
4. a rebuild that RAISES mid-build keeps the LAST good snapshot, never a blank holder -- build-then-swap
   tears the old figure down only after the new one built cleanly (symptom "图有时消失")."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")

from Zou_lab_control.frontend.qt_fluent import ensure_qt_app  # noqa: E402


def test_edit_snapshot_canvas_is_fixed_size_and_idempotent_across_bin_rebuilds():
    ensure_qt_app()
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelEditor, PanelConfig, GAP)

    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
    console._timer.stop()
    try:
        from Zou_lab_control.frontend import panel_plot
        vals = np.concatenate([np.random.default_rng(0).normal(300.0, 20.0, 400),
                               np.random.default_rng(1).normal(460.0, 20.0, 300)])
        cfg = PanelConfig(kind="hist", title="dis", source="counts", row=GAP, col=GAP, size="2x4")
        card = console._new_panel_card(cfg)
        console._attach_card(card)
        # give the live card a real hist plotter (the render pipeline is not what we test here), and
        # stop the editor's pre-snapshot refresh from clearing it -- we test rebuild()'s SIZE behaviour.
        card.plotter = panel_plot(vals, kind="hist", size=cfg.size, bins=60, title="dis")
        console.refresh_once = lambda: None
        assert card.plotter is not None

        editor = PanelEditor(card, console)
        editor.rebuild()
        cv = editor._canvas
        assert cv is not None
        # FIXED: the snapshot canvas can neither grow nor squish (min == max == the design sizeHint)
        assert cv.minimumSize() == cv.maximumSize() == cv.sizeHint()
        size0 = (cv.width(), cv.height())

        # scrolling bins re-snapshots through rebuild(); the size must NOT drift across turns
        for b in (80, 120, 160, 200):
            card.config.params["bins"] = b
            editor.rebuild()
        cv = editor._canvas
        assert (cv.width(), cv.height()) == size0, "Edit snapshot canvas changed size across bin rebuilds"
        assert cv.minimumSize() == cv.maximumSize(), "snapshot canvas must stay FIXED, not growable"
    finally:
        console.shutdown()
        exp.close()


def _console():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
    console._timer.stop()
    console.refresh_once = lambda: None        # the editor's pre-snapshot refresh must not clear the plotter
    return exp, console


def _hist_card(console):
    from Zou_lab_control.frontend import panel_plot
    from Zou_lab_control.frontend.task_console import PanelConfig, GAP
    vals = np.concatenate([np.random.default_rng(0).normal(300.0, 20.0, 400),
                           np.random.default_rng(1).normal(460.0, 20.0, 300)])
    card = console._new_panel_card(PanelConfig(kind="hist", title="dis", source="counts",
                                               row=GAP, col=GAP, size="2x4"))
    console._attach_card(card)
    card.plotter = panel_plot(vals, kind="hist", size="2x4", bins=60, title="dis")
    return card


def test_edit_snapshot_canvas_stays_fixed_after_a_resync():
    """#4 symptom 1: the snapshot canvas's own deferred ``_zlc_resync`` (what a showEvent fires) must
    KEEP it pinned.  When pinned it re-applies setFixedSize (min==max); the old code re-published
    setMinimumSize(sizeHint()), which could raise the floor past the fixed ceiling and balloon the
    figure across param edits."""
    ensure_qt_app()
    from Zou_lab_control.frontend.task_console import PanelEditor
    exp, console = _console()
    try:
        editor = PanelEditor(_hist_card(console), console)
        editor.rebuild()
        cv = editor._canvas
        assert cv is not None and cv.minimumSize() == cv.maximumSize()      # pinned at build
        before = (cv.minimumSize().width(), cv.minimumSize().height())
        cv._zlc_resync()                                                    # the deferred resync a showEvent fires
        assert cv.minimumSize() == cv.maximumSize(), "resync must keep the pinned snapshot FIXED (no balloon)"
        assert (cv.minimumSize().width(), cv.minimumSize().height()) == before
    finally:
        console.shutdown()
        exp.close()


def test_edit_sites_snapshot_honours_relim_and_fixed_lim():
    """#4 symptom 3: the Edit snapshot of a Site map must re-render when its lim mode (tight/normal) OR
    fixed lo/hi changes.  The ``sites`` branch used to OMIT relim/fixed in the snapshot builder, so a
    sitemap's lim never updated (other kinds did).  ``_view_kwargs`` now carries them for EVERY kind."""
    ensure_qt_app()
    from Zou_lab_control.frontend import panel_plot
    from Zou_lab_control.frontend.task_console import PanelConfig, GAP, PanelEditor
    exp, console = _console()
    try:
        centers = np.column_stack([(np.arange(6) % 3) * 5.0, (np.arange(6) // 3) * 5.0])
        occ = np.random.default_rng(0).random(6)
        img = np.random.default_rng(1).normal(300.0, 20.0, (24, 30))
        card = console._new_panel_card(PanelConfig(kind="sites", title="map", source="occupied",
                                                   row=GAP, col=GAP, size="2x4"))
        console._attach_card(card)
        card.plotter = panel_plot(centers, occ, kind="sites", size="2x4", image=img,
                                  roi_radius=4.0, labels=("x", "y", "occ"), title="map")
        editor = PanelEditor(card, console)
        editor.rebuild()
        # relim mode reaches the SITES snapshot (was ignored): a real Edit-param edit re-snapshots.
        editor._edit_param("relim", "tight")
        assert editor._plotter.relim_mode == "tight"
        editor._edit_param("relim", "normal")
        assert editor._plotter.relim_mode == "normal"
        # fixed lo/hi reaches the sites snapshot too (the snapshot omitted fixed for ALL kinds).
        card.config.params.update({"relim": "fixed", "fixed_lo": 120.0, "fixed_hi": 480.0})
        editor.rebuild()
        assert editor._plotter.relim_mode == "fixed"
        assert float(editor._plotter.fixed_lo) == 120.0 and float(editor._plotter.fixed_hi) == 480.0
    finally:
        console.shutdown()
        exp.close()


def test_dis_fixed_lim_pins_the_count_axis_not_the_value_axis():
    """A Distribution's relim/fixed lo-hi pins the COUNT (y) axis -- the measured quantity, exactly
    as a 1D plot's relim pins its SIGNAL axis (the dependent axis), not the value/binning (x) axis.
    The value axis always frames the natural bin span; only the count headroom is operator-pinned."""
    ensure_qt_app()
    from Zou_lab_control.frontend.task_console import PanelEditor
    exp, console = _console()
    try:
        card = _hist_card(console)
        editor = PanelEditor(card, console)
        editor.rebuild()                                                    # auto: count fits, value frames bins
        value_lo, value_hi = editor._plotter.ax.get_xlim()                  # value (x) axis = bin span
        assert value_lo < 300.0 and value_hi > 460.0
        auto_top = float(editor._plotter.ax.get_ylim()[1])                  # count (y) auto headroom
        # fixed lo/hi pins the COUNT (y) axis; the value (x) axis must NOT move
        card.config.params.update({"relim": "fixed", "fixed_lo": 0.0, "fixed_hi": auto_top + 500.0})
        editor.rebuild()
        assert editor._plotter.relim_mode == "fixed"
        ylo, yhi = editor._plotter.ax.get_ylim()
        assert abs(ylo - 0.0) < 1e-6 and abs(yhi - (auto_top + 500.0)) < 1e-6, "fixed must pin the dis COUNT axis"
        assert editor._plotter.ax.get_xlim() == (value_lo, value_hi), "value (x) axis must not be pinned by fixed"
    finally:
        console.shutdown()
        exp.close()


def test_edit_rebuild_keeps_last_good_snapshot_on_build_error(monkeypatch):
    """#4 symptom 2: a rebuild that RAISES mid-build must leave the PREVIOUS snapshot installed, never a
    blank holder -- build-then-swap tears the old figure down only after the new one built cleanly."""
    ensure_qt_app()
    import Zou_lab_control.frontend.task_console as tc
    exp, console = _console()
    try:
        editor = tc.PanelEditor(_hist_card(console), console)
        editor.rebuild()
        good_canvas = editor._canvas
        assert good_canvas is not None
        monkeypatch.setattr(tc, "panel_plot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced build error")))
        editor.rebuild()                                                    # build raises mid-rebuild
        assert editor._canvas is good_canvas, "a failed rebuild must keep the last good snapshot, not blank it"
        assert editor._plotter is not None
        assert "could not snapshot" in editor.status.text().lower()
    finally:
        console.shutdown()
        exp.close()


def test_force_rebuild_keeps_card_plotter_on_failed_build(monkeypatch):
    """#4 root fix for the scroll-bins-vanishes race: a STRUCTURE knob (bins/colormap/fit) forces a
    rebuild via the ``_force_rebuild`` flag -- it does NOT pre-tear the plotter down.  So if the rebuild
    raises (a transient namespace), ``_build_plot``'s build-then-swap leaves the OLD plotter in place
    and the live card never blanks; the flag stays set so the next tick retries.  (Before, ``_set_param``
    called ``_reset_plot()`` which nulled ``card.plotter`` up-front, latching a permanent blank that the
    Edit ``rebuild`` then propagated via its ``card.plotter is None`` guard.)"""
    ensure_qt_app()
    import Zou_lab_control.frontend.task_console as tc
    exp, console = _console()
    try:
        card = _hist_card(console)
        good = card.plotter
        assert good is not None
        monkeypatch.setattr(tc, "panel_plot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        card._force_rebuild = True
        try:
            card._render(np.array([300.0, 320.0, 1400.0, 1420.0]))   # forced rebuild -> _build_plot raises
        except RuntimeError:
            pass
        assert card.plotter is good, "a failed forced rebuild must KEEP the old plotter (no blank)"
        assert card._force_rebuild is True, "flag stays set so the next tick retries the rebuild"
    finally:
        console.shutdown()
        exp.close()


def test_both_gui_windows_share_one_editable_screen_fraction():
    """#5: the task console and the pulse editor open at the SAME fraction of the screen, read from ONE
    user-editable constant ``qt_fluent.WINDOW_SCREEN_FRACTION`` -- not two divergent literals (the pulse
    editor's 0.90 vs the console's stray 0.84)."""
    import inspect

    from Zou_lab_control.frontend import pulse_gui, qt_fluent
    from Zou_lab_control.frontend.task_console import TaskConsole, show_task_console

    frac = qt_fluent.WINDOW_SCREEN_FRACTION
    assert pulse_gui.DEFAULT_WINDOW_RATIO == frac                       # pulse editor reads the constant
    assert inspect.signature(TaskConsole.__init__).parameters["window_ratio"].default == frac
    assert inspect.signature(show_task_console).parameters["window_ratio"].default == frac


def test_edit_status_line_never_widens_the_page():
    """A long save log (the saved absolute path) must NOT balloon the Edit page into a horizontal
    scroll: the status label's horizontal policy is Ignored, so it takes the available width and clips
    -- the page width is owned by the canvas/rows, never by a log string (#layout-fixed)."""
    ensure_qt_app()
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.task_console import PanelEditor
    exp, console = _console()
    try:
        editor = PanelEditor(_hist_card(console), console)
        editor.rebuild()
        assert editor.status.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Ignored
        w0 = editor.sizeHint().width()
        editor.status.setText("saved fig.png + fig.npy -> " + "C:/" + "averylongfoldername/" * 40)
        editor.status.adjustSize()
        assert editor.sizeHint().width() <= w0 + 4, \
            "a long status string must not grow the Edit page width (it must clip, not balloon)"
    finally:
        console.shutdown()
        exp.close()


def test_setting_popup_relim_fixed_grows_down_from_a_fixed_top_left():
    """Toggling relim -> ``fixed`` reveals the lo/hi row, which makes the Setting popup TALLER.  The
    popup is a free-floating top-level Qt.Popup whose window size is fixed at open time, so a bare
    ``setVisible`` reflows the rows inside that fixed window -> the layout "jumps".  The fix re-runs the
    one sizing rule (``_size_settings_popup``: top-left anchored, grow-down, high-water) so the window
    GROWS DOWNWARD: its top-left ``pos`` must NOT move (no re-center/teleport) and its height must only
    grow -- a smooth single-direction expand.  Uses a LARGE window so the popup is not already capped at
    the panel bottom (which would scroll instead of grow), isolating the row-toggle growth."""
    ensure_qt_app()
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelConfig, GAP)

    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(1600, 1100))
    console._timer.stop()
    try:
        card = console._new_panel_card(PanelConfig(kind="1d", title="t", source="value",
                                                   row=GAP, col=GAP, size="4x4"))
        console._attach_card(card)
        card._open_settings()
        popup = card.settings_popup
        assert popup.isVisible()
        assert not card.fixed_lim_row.isVisible()                       # auto: lo/hi row hidden
        pos0 = (popup.x(), popup.y())
        h0 = popup.height()

        card._set_param("relim", "fixed")                              # reveal the lo/hi row
        assert card.fixed_lim_row.isVisible()
        assert (popup.x(), popup.y()) == pos0, \
            "the Setting popup must stay top-left ANCHORED when a row appears (no re-center/jump)"
        assert popup.height() > h0, "the popup must GROW DOWNWARD to fit the revealed lo/hi row"
        grown_h = popup.height()

        # Toggling back never moves the top-left, and the high-water mark keeps it tall (no shrink-jump).
        card._set_param("relim", "normal")
        assert (popup.x(), popup.y()) == pos0, "toggling back must not move the popup either"
        assert popup.height() == grown_h, "high-water: the popup never snaps shorter within a session"
    finally:
        console.shutdown()
        exp.close()


def test_edit_tab_fixed_lim_toggle_never_moves_the_snapshot_canvas():
    """The Edit-tab '跳'/jump: the fixed lo/hi row lives ABOVE the snapshot canvas in one shared
    scroll page, so the old ``setVisible`` reveal reflowed the unit row + the whole canvas DOWN by the
    row's height on every relim change.  Fixed: the row stays PERMANENTLY in the layout and only its
    inputs enable when relim == 'fixed', so the canvas y-position is invariant across every relim mode
    (unlike the Setting popup, which has nothing below the row and grows downward)."""
    ensure_qt_app()
    from PyQt5 import QtCore, QtWidgets
    from Zou_lab_control.frontend.task_console import PanelEditor
    exp, console = _console()
    try:
        editor = PanelEditor(_hist_card(console), console)
        editor.rebuild()
        holder = QtWidgets.QWidget(); QtWidgets.QVBoxLayout(holder).addWidget(editor)
        holder.resize(360, 720); holder.show()
        QtWidgets.QApplication.instance().processEvents()
        page = editor.findChild(QtWidgets.QScrollArea).widget()
        canvas_y = lambda: editor._canvas.mapTo(page, QtCore.QPoint(0, 0)).y()
        ys, enabled = [], []
        for mode in ("tight", "fixed", "tight", "normal", "fixed"):
            editor._edit_param("relim", mode)
            QtWidgets.QApplication.instance().processEvents()
            assert editor.ed_fixed_row.isVisible(), "the Edit fixed lo/hi row must stay in the layout (never hidden)"
            ys.append(canvas_y())
            enabled.append(bool(editor.ed_fixed_lo.isEnabled()))
        assert len(set(ys)) == 1, f"the snapshot canvas must NOT move when relim toggles (the jump), got y={ys}"
        assert enabled == [False, True, False, False, True], f"lo/hi must enable ONLY in fixed, got {enabled}"
    finally:
        console.shutdown()
        exp.close()


def test_edit_tab_exposes_repeat_mode_like_the_setting_popup():
    """repeat mode (the plot's repeat-DISPLAY collapse) must be tunable in the Edit tab too, not only
    the Setting popup -- both render the SAME ``_repeat_param_specs`` and write ``config.params`` via the
    one ``_set_param``, so the two surfaces never drift."""
    ensure_qt_app()
    from Zou_lab_control.frontend.task_console import PanelEditor
    exp, console = _console()
    try:
        card = _hist_card(console)
        editor = PanelEditor(card, console)
        assert "repeat_mode" in editor.ed_params, "the Edit tab must render the repeat_mode row (like the Setting popup)"
        modes = card._kind_repeat_modes()
        editor._edit_param("repeat_mode", modes[-1])                    # editing it routes through the ONE writer
        assert card.config.params.get("repeat_mode") == modes[-1]
    finally:
        console.shutdown()
        exp.close()
