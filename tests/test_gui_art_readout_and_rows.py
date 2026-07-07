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
