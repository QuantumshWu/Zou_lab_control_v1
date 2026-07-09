"""Two GUI-layout invariants from the art pass:

1. The Figure Viewer's Plot info tab shows axis labels PER-AXIS (the real label each axis draws),
   never the whole ``(xlabel, ylabel, zlabel)`` tuple rendered as one row's Python list repr.
2. The device manager's "Loaded (session)" rows are COMPACT (an elided role name + two fixed action
   buttons, the SAME plain-row shape the Discovered list uses -- NOT a wide fixed-label-column
   ``FluentSettingRow``), so the right pane can never overflow into a horizontal scrollbar that clips
   the "Snapshot" button and leaves a stray corner square.
"""

import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def qt():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def test_figure_viewer_shows_labels_per_axis_not_a_list_repr(qt, tmp_path):
    from Zou_lab_control.neutral_atom.views.plots import plot_image
    from Zou_lab_control.frontend.figure_viewer import FigureViewer, FluentReadoutMultiline
    fig = plot_image(np.random.default_rng(0).poisson(300, size=(24, 24)).astype(float), display=False)
    out = fig.save(str(tmp_path / "shot"))
    viewer = FigureViewer()
    viewer.open_path(str(out["data"]))
    values = []
    for i in range(viewer.plot_layout.count()):
        row = viewer.plot_layout.itemAt(i).widget()
        values += [c.toPlainText() for c in row.findChildren(FluentReadoutMultiline)]
    # Each axis label is its OWN row value (the real label the axis draws) --
    assert "Camera x (px)" in values and "Camera y (px)" in values and "Counts" in values, values
    # -- and NO row holds the whole tuple's list/tuple repr (``['Camera x (px)', ...]``).
    assert not any(v.startswith(("[", "(")) and "'" in v for v in values), values


def test_figure_viewer_2d_reproduction_renders_the_colorbar_z_label(qt, tmp_path):
    """A saved 2D figure's THIRD label is the colour-bar (z) quantity ("Counts").  The reproduced 2D
    panel must LABEL its colour bar with it -- the figure node has to carry the z label onto the value
    signal (not drop it / mislabel it with the pixel y-axis), and the 2D panel builder has to read it
    onto the bar (it used to hard-code the z label to "").  A blank colour bar is the regression."""
    from Zou_lab_control.neutral_atom.views.plots import plot_image
    from Zou_lab_control.frontend.figure_viewer import FigureViewer
    fig = plot_image(np.random.default_rng(1).poisson(300, size=(24, 24)).astype(float),
                     display=False, labels=("Camera x (px)", "Camera y (px)", "Counts"))
    out = fig.save(str(tmp_path / "shot2d"))
    viewer = FigureViewer()
    viewer.open_path(str(out["data"]))
    zlabels = []
    for card in getattr(viewer.console, "cards", []):
        plotter = getattr(card, "_plotter", None) or getattr(card, "plotter", None)
        cax = getattr(plotter, "cax", None)
        if cax is not None:                       # a 2D image panel owns a colour-bar axis
            zlabels.append(getattr(plotter, "zlabel", "") or cax.get_ylabel())
    assert zlabels, "the reproduced figure had no 2D colour-bar panel"
    assert "Counts" in zlabels, zlabels


def test_saved_figure_axis_labels_are_kind_aware(qt, tmp_path):
    """``SavedFigure.axis_labels()`` is the ONE source the info tab + the reproduction read: a
    DISTRIBUTION (hist) draws only x and y -- its stored 3rd label ("Population") is the dormant
    ``labels[-1]`` and must NOT surface as a phantom "z"/"colour bar"; a 2D image / site map DOES draw a
    colour bar (labels[2]); a grid carries its axes in the recipe, not the top-level labels."""
    from Zou_lab_control.frontend.data_figure import load_figure

    def craft(name, info, data_x, data_y):
        p = tmp_path / name
        np.savez(p, data_x=np.asarray(data_x, float), data_y=np.asarray(data_y, float),
                 info=np.array(info, dtype=object))
        return load_figure(str(p))

    hist = craft("h.npz", {"kind": "hist", "labels": ["ROI counts", "Shots", "Population"], "name": "h"},
                 np.arange(50.0), np.random.default_rng(0).normal(300, 30, 50))
    assert [n for n, _ in hist.axis_labels()] == ["x label", "y label"]   # NO colour bar; 3rd label dropped
    assert all(lab != "Population" for _, lab in hist.axis_labels())

    sites = craft("s.npz", {"kind": "sites", "labels": ["Camera x (px)", "Camera y (px)", "Counts"], "name": "s"},
                  np.zeros((9, 2)), np.zeros(9))
    assert sites.axis_labels() == [("x label", "Camera x (px)"), ("y label", "Camera y (px)"),
                                   ("colour bar", "Counts")]

    grid = craft("g.npz", {"kind": "grid", "name": "g",
                           "figure_recipe": {"kind": "grid", "labels": ["Trap-off (s)", "Survival"],
                                             "per_cell": [{}]}},
                 np.zeros(1), np.zeros(1))
    assert grid.axis_labels() == [("x label", "Trap-off (s)"), ("y label", "Survival")]


def test_reproduced_figure_renders_the_saved_x_y_labels(qt, tmp_path):
    """A reopened figure draws the axis labels stored in its DATA -- not a kind's live default (a 2D
    panel's ROI-derived "X (px)", a hist's hard-coded "Value").  The saved labels are seeded into the
    panel params (``_seed_state``) and PanelCard._panel_labels renders them, so the rendered x / y /
    colour bar MATCH ``saved.labels``.  A rendered label != the saved one is the regression."""
    from Zou_lab_control.neutral_atom.views.plots import plot_image
    from Zou_lab_control.frontend.figure_viewer import FigureViewer
    fig = plot_image(np.random.default_rng(3).poisson(300, size=(20, 20)).astype(float),
                     display=False, labels=("Camera x (px)", "Camera y (px)", "Counts"))
    out = fig.save(str(tmp_path / "shot2d"))
    viewer = FigureViewer()
    viewer.open_path(str(out["data"]))
    card = viewer.console.cards[0]
    plotter = getattr(card, "plotter", None) or getattr(card, "_plotter", None)
    assert plotter.ax.get_xlabel() == "Camera x (px)", plotter.ax.get_xlabel()   # NOT "X (px)"
    assert plotter.ax.get_ylabel() == "Camera y (px)", plotter.ax.get_ylabel()
    assert plotter.cax.get_ylabel() == "Counts", plotter.cax.get_ylabel()


def test_fluent_message_body_wraps_anywhere_and_never_clips(qt):
    """A warning / error message body must WRAP an unbreakable long token (a Windows path, a dotted
    traceback frame, a long fingerprint) instead of running past the card edge / under the vertical
    scrollbar the way a word-BOUNDARY-wrapping QLabel did (that clipping was the reported bug).  So the
    body is a wrap-ANYWHERE ``FluentReadoutMultiline`` (WidgetWidth + WrapAtWordBoundaryOrAnywhere) and,
    with a long unbreakable token, has ZERO horizontal scroll range (nothing overflows sideways)."""
    from PyQt5 import QtWidgets, QtGui
    from Zou_lab_control.frontend.qt_fluent import _FluentMessageDialog, FluentReadoutMultiline, FluentLabel

    text = (
        "RuntimeError: geometry/layout mismatch: bitstream reports 0x5A87FD36 but host fingerprint is "
        "0x5AFC7CFB_A_VERY_LONG_UNBREAKABLE_TOKEN_WITH_NO_SPACES_AT_ALL_SO_IT_CANNOT_WORD_WRAP\n"
        'File "C:\\Users\\eadri\\Dropbox\\WorkCode\\Github\\Zou_lab_control_v1_claude\\x.py", line 279\n'
        + ("more text to force the vertical scrollbar. " * 40)
    )
    dlg = _FluentMessageDialog(None, "Connection failed", text, kind="warning")
    dlg.adjustSize()
    dlg.show()
    QtWidgets.QApplication.instance().processEvents()
    QtWidgets.QApplication.instance().processEvents()
    try:
        body = dlg.findChild(FluentReadoutMultiline)
        # ROOT of the fix: the body is a wrap-anywhere text block, NOT a word-boundary QLabel.
        assert body is not None, "the message body must be a FluentReadoutMultiline (wrap-anywhere)"
        assert not dlg.findChildren(FluentLabel) or all(
            not isinstance(w, FluentLabel) or w is not body for w in dlg.findChildren(FluentLabel))
        assert body.lineWrapMode() == QtWidgets.QPlainTextEdit.WidgetWidth
        assert body.wordWrapMode() == QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere
        # No horizontal overflow -> no glyphs clipped at the edge / hidden under the vertical bar.
        assert body.horizontalScrollBar().maximum() == 0, "long token overflowed sideways (clipped)"
        # Long body is CAPPED and scrolls vertically (does not grow the dialog off-screen).
        assert body.verticalScrollBar().maximum() > 0, "a long message must scroll, not grow unbounded"
    finally:
        dlg.close()


def test_fluent_message_short_body_hugs_and_is_not_capped(qt):
    """A SHORT message stays a plain hugging text block: it sizes to its natural height, well UNDER the
    scroll cap (the cap only kicks in for long content), so a one-line 'Saved …' is not forced into a
    scroll box.  (The cap is the single ``scaled_px(320, minimum=180)`` the dialog uses.)"""
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.qt_fluent import _FluentMessageDialog, FluentReadoutMultiline, scaled_px

    dlg = _FluentMessageDialog(None, "Saved", "Saved the pulse program to results/run_042.npz.", kind="info")
    dlg.adjustSize()
    dlg.show()
    QtWidgets.QApplication.instance().processEvents()
    try:
        body = dlg.findChild(FluentReadoutMultiline)
        assert body.height() < scaled_px(320, minimum=180), "a short message must NOT hit the scroll cap"
        assert body.horizontalScrollBar().maximum() == 0
    finally:
        dlg.close()


def test_fluent_scrollbar_thickness_is_the_single_source_of_the_bar_width(qt):
    """The scrollbar bar width lives in ONE place: ``fluent_scrollbar_thickness()``.  The painted CSS
    (``fluent_scrollbar_stylesheet``) and every layout that RESERVES the bar's gutter read that one
    value, so they can never silently drift (a hand-typed ``12`` was the smell the audit found)."""
    from Zou_lab_control.frontend.qt_fluent import fluent_scrollbar_thickness, fluent_scrollbar_stylesheet
    t = fluent_scrollbar_thickness()
    assert isinstance(t, int) and t > 0
    css = fluent_scrollbar_stylesheet("QScrollBar")
    assert f"width: {t}px" in css and f"height: {t}px" in css, "the CSS bar size must be the one thickness"


def test_device_manager_loaded_row_is_compact_and_cannot_overflow(qt):
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.device_manager import DeviceManagerPanel
    from Zou_lab_control.frontend.qt_fluent import ElidedLabel, FluentButton, FluentSettingRow
    from Zou_lab_control.neutral_atom._gui import _session_device_binding
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        panel = DeviceManagerPanel(exp.devices, session_binding=_session_device_binding(exp))
        loaded = panel._loaded_body
        # the Loaded body = the Open-devices button + one PLAIN row per device (the button is skipped).
        rows = [loaded.itemAt(i).widget() for i in range(loaded.count())
                if loaded.itemAt(i).widget() is not None
                and not isinstance(loaded.itemAt(i).widget(), FluentButton)]
        assert rows, "expected one Loaded row per device"
        for row in rows:
            # NOT the wide fixed-label-column form row (that column, plus the two buttons, was what
            # overflowed the pane into a horizontal scrollbar that clipped "Snapshot" to "Snaps" and
            # left a stray corner square) ...
            assert not isinstance(row, FluentSettingRow)
            # ... the role name is an ElidedLabel with a tiny minimum width, so it elides instead of
            # widening the pane -- the ONE flexible element that lets the row shrink to any width ...
            elided = row.findChildren(ElidedLabel)
            assert elided, "the Loaded row's role name must elide"
            assert min(e.minimumWidth() for e in elided) <= elided[0].fontMetrics().averageCharWidth() * 4
            # ... and it keeps its Snapshot button (the two fixed buttons then always fit).
            assert any(b.text() == "Snapshot" for b in row.findChildren(FluentButton))
    finally:
        exp.close()
