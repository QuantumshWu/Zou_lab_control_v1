"""Contract: a ``FluentSettingRow`` places its control flush-LEFT under the label column, whether the
control FILLS the cell (an expanding combo / line-edit / switch) or is pinned to its sizeHint (a FIXED
capsule like the ``FluentTriStateToggle`` ``fit`` toggle, or a ``FluentButton``).

The bug this guards (regressed once when the tri-state toggle's horizontal policy was pinned Fixed):
adding a Fixed control with ``addWidget(control, 1)`` makes the stretch cell CENTRE it -- the toggle
floated in the middle of the row instead of left-aligning under its label like every other control.  The
fix puts a Fixed control in with ``Qt.AlignLeft`` + a trailing stretch (confocal's TriStateToggleSwitch
row idiom); an expanding control still gets the stretch so it fills.  Either way the control's LEFT EDGE
sits at the same x as the next control down -- the user's "the widget should be left-aligned like the
toggle switch" requirement, made mechanical so it cannot silently re-center.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _row_items(row):
    """The (control_item, has_trailing_stretch) of a FluentSettingRow: item 0 is the label, item 1 is
    the control, item 2 (if present) is the trailing stretch a left-aligned Fixed control adds."""
    layout = row.layout()
    items = [layout.itemAt(i) for i in range(layout.count())]
    return items


def test_fixed_control_is_left_aligned_not_stretched():
    """A FIXED-horizontal control (the FluentTriStateToggle capsule) is added left-aligned with a trailing
    stretch -- NOT with a stretch factor that would centre it -- so it sits flush-left under its label."""
    from PyQt5 import QtCore, QtWidgets
    from Zou_lab_control.frontend.qt_fluent import (
        ensure_qt_app, FluentSettingRow, FluentTriStateToggle)

    ensure_qt_app()
    toggle = FluentTriStateToggle(["none", "single", "double"])
    assert toggle.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Fixed
    row = FluentSettingRow("fit", toggle, label_width=60)
    items = _row_items(row)
    # label | control(AlignLeft, no stretch) | trailing stretch
    assert len(items) == 3, "a Fixed control row must be [label | control | trailing stretch]"
    ctrl_item = items[1]
    assert ctrl_item.widget() is toggle
    assert int(ctrl_item.alignment()) & int(QtCore.Qt.AlignLeft), \
        "a Fixed control must be added with Qt.AlignLeft (else the stretch cell centres it)"
    assert items[2].spacerItem() is not None, "a left-aligned Fixed control needs a trailing stretch"


def test_expanding_control_fills_the_cell():
    """An expanding control (a combo) is still added with a stretch factor so it FILLS the cell -- the
    left edge is identical to a Fixed control's, but it also extends to the right margin (no centring
    issue, and the trailing-stretch branch must NOT shrink it to its sizeHint)."""
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app, FluentSettingRow, FluentComboBox

    ensure_qt_app()
    combo = FluentComboBox()
    combo.addItems(["a", "b"])
    assert combo.sizePolicy().horizontalPolicy() != QtWidgets.QSizePolicy.Fixed
    row = FluentSettingRow("relim", combo, label_width=60)
    items = _row_items(row)
    # label | control(stretch=1) -- no trailing stretch (the control already fills)
    assert len(items) == 2, "an expanding control row is [label | control], control fills via stretch"
    assert items[1].widget() is combo


def test_tristate_and_switch_rows_share_a_left_edge():
    """End-to-end on a real hist Setting popup: the ``fit`` tri-state capsule and the ``ylog`` switch (and
    the relim combo) all start at the SAME x inside their rows -- the alignment the user asked for."""
    from PyQt5 import QtCore
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app, FluentTriStateToggle, FluentSwitch
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    app = ensure_qt_app()
    card = PanelCard(PanelConfig(kind="hist", inputs=["frame_0"], source="value = signal"))
    try:
        content = card._settings_scroll.widget()
        content.resize(560, content.sizeHint().height())
        content.show()
        app.processEvents()
        fit = card.param_widgets["fit"]
        ylog = card.param_widgets["ylog"]
        assert isinstance(fit, FluentTriStateToggle) and isinstance(ylog, FluentSwitch)
        # the control's left edge (mapped to its OWN row's coords) is the post-label column start; the
        # tri-state must land at the same row-local x as the switch (both flush-left under the label).
        fx = fit.mapTo(fit.parentWidget(), QtCore.QPoint(0, 0)).x()
        yx = ylog.mapTo(ylog.parentWidget(), QtCore.QPoint(0, 0)).x()
        assert fx == yx, f"fit toggle left x ({fx}) must match the switch left x ({yx}) -- same column"
        content.hide()
    finally:
        pass
