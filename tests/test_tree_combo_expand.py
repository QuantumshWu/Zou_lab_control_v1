"""Contract: FluentTreeComboBox expands a group on a NAME click (not only the triangle) and the
popup GROWS to fit a freshly-expanded parent's children (instead of scrolling/clipping them).

Both behaviours live INSIDE the reusable widget (`_ExpandableTreeView.mousePressEvent` +
`FluentTreeComboBox._resize_popup_to_contents`), so any caller gets them for free.  A design rule
that CAN be a test MUST be one.
"""

import pytest


def _combo():
    from Zou_lab_control.frontend.qt_fluent import FluentTreeComboBox
    c = FluentTreeComboBox()
    groups = [("occupancy", [("rate  (35,)", "rate", "occupancy · rate"),
                             ("occupied  (5,7)", "occupied", "occupancy · occupied")]),
              ("temperature", [("t_off  (20,)", "t_off", "temperature · t_off")])]
    c.set_signal_tree(groups, current="", none_label="(none)")
    return c


def test_parent_name_click_toggles_expansion(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets, QtCore, QtGui
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c = _combo()
    c.showPopup()
    tree = c.view()
    p1 = c._model.index(1, 0)                      # row 0 = (none); row 1 = first producer header
    assert not tree.isExpanded(p1)
    # click the ROW CENTRE (the name, NOT the triangle on the left edge) -> must toggle open
    rect = tree.visualRect(p1)
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPointF(rect.center()),
                           QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    tree.mousePressEvent(ev)
    assert tree.isExpanded(p1), "clicking a parent's name must expand it (not only the triangle)"
    c.hidePopup()


def test_popup_height_grows_when_a_parent_expands(monkeypatch):
    """The popup's DESIRED height (what `_resize_popup_to_contents` applies when the popup is
    visible) must increase when a parent expands -- i.e. the box grows to fit the children rather
    than keeping them scrolled.  We assert the pure computation `_desired_popup_height` (the visible
    `setFixedHeight` itself needs a real display the headless grab can't map)."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c = _combo()
    c.showPopup()
    tree = c.view()
    collapsed = c._desired_popup_height()
    tree.setExpanded(c._model.index(1, 0), True)
    expanded_one = c._desired_popup_height()
    tree.setExpanded(c._model.index(2, 0), True)
    expanded_two = c._desired_popup_height()
    assert collapsed > 0
    assert expanded_one > collapsed, "expanding a parent must make the popup taller"
    assert expanded_two > expanded_one, "expanding a second parent must make it taller still"
    c.hidePopup()
