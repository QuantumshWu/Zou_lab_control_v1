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


def test_popup_height_grows_to_fit_then_clamps_and_scrolls(monkeypatch):
    """The popup's DESIRED height grows to fit a freshly-expanded parent's children -- BUT is clamped
    to the space available at the anchor (never overruns the screen): once content exceeds that space the
    height stays at the boundary (and the tree then SCROLLS), exactly the Setting-popup rule (#issue-4).
    We assert the pure computation `_desired_popup_height` (the visible `setFixedHeight` needs a display)."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c = _combo()
    c.showPopup()
    tree = c.view()
    avail = c._anchor_available_height()
    collapsed = c._desired_popup_height()
    tree.setExpanded(c._model.index(1, 0), True)
    expanded_one = c._desired_popup_height()
    tree.setExpanded(c._model.index(2, 0), True)
    expanded_two = c._desired_popup_height()
    assert collapsed > 0
    # monotonic non-decreasing as more rows show...
    assert expanded_one >= collapsed and expanded_two >= expanded_one
    # ...and NEVER taller than the space available at the anchor (the clamp = the overrun fix).
    assert avail > 0 and expanded_two <= avail
    # when the fully-expanded content would exceed the available space, the height clamps to it (-> scroll)
    tree.setExpanded(c._model.index(1, 0), True)
    tree.setExpanded(c._model.index(2, 0), True)
    assert c._desired_popup_height() <= avail
    c.hidePopup()


def test_pick_updates_the_collapsed_display_immediately():
    """#1: clicking a leaf must update the box's COLLAPSED display text RIGHT AWAY (the producer-
    qualified label), not only after the Setting popup is closed and reopened.  Guards the widget
    contract behind the repaint() fix -- _on_tree_clicked sets _display, and _display_text() (what the
    collapsed box paints) returns it immediately."""
    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c = _combo()                                       # groups: producers with leaves (see _combo helper)
    before = c._display_text()
    # click the FIRST leaf of the SECOND producer group (index depends on the helper's groups)
    parent = c._model.item(c._model.rowCount() - 1)    # last producer group
    leaf = parent.child(0)
    c._on_tree_clicked(leaf.index())
    picked_full = str(leaf.data(QtCore.Qt.UserRole + 1))
    assert c._display_text() == picked_full            # collapsed box shows the pick NOW
    assert c._display_text() != before
    assert c.current_signal() == str(leaf.data(QtCore.Qt.UserRole))   # and the bound value matches
