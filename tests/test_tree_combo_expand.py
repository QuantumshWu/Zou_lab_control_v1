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


def test_first_open_resize_applies_despite_lagging_visible_flag(monkeypatch):
    """The popup must GROW to fit its content on the FIRST open/expand -- even though, on a real screen,
    super().showPopup() positions the container a beat BEFORE the view's ``isVisible()`` flag propagates
    (transiently False on the very first open).  ``_resize_popup_to_contents`` must apply the (already
    correct) ``_desired_popup_height`` regardless of that lagging flag; a ``not view.isVisible()`` guard
    used to DROP the first open's/expand's resize, so the picker opened CLIPPED until the SECOND open
    (offscreen sets the flag immediately, so this forces the real-window transient explicitly)."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c = _combo()
    c.showPopup()
    tree = c.view()
    tree.setExpanded(c._model.index(1, 0), True)        # expand both parents -> the full content height
    tree.setExpanded(c._model.index(2, 0), True)
    container = tree.window()
    desired = c._desired_popup_height()                 # computed while the flag is still good
    assert desired > c._popup_pad() + 30                # there IS content to grow into
    container.setFixedHeight(c._popup_pad() + 1)         # start in the clipped-on-first-open state
    monkeypatch.setattr(tree, "isVisible", lambda: False)   # the first-open transient: flag not yet set
    c._resize_popup_to_contents()                       # must STILL grow the container (the fix)
    assert container.height() == pytest.approx(max(40, desired), abs=6), \
        "first-open resize must apply despite the lagging visible flag (else the picker opens clipped)"
    c.hidePopup()


def test_pick_updates_the_collapsed_display_immediately_and_on_switch():
    """#1: picking a leaf updates the box's COLLAPSED text RIGHT AWAY to the producer-qualified label
    (not the raw verbose leaf text), AND switching to ANOTHER leaf updates it LIVE -- no close+reopen.
    Guards the live-derive design: _display_text() reads the CURRENT selection's full label from the
    populate-time map (no cached _display that goes stale)."""
    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c = _combo()                                       # producers: occupancy{rate,occupied}, temperature{t_off}
    occ = c._model.item(1)                             # row 0 = (none); row 1 = occupancy group
    rate, occupied = occ.child(0), occ.child(1)
    c._on_tree_clicked(rate.index())
    assert c._display_text() == "occupancy · rate"     # producer-qualified, NOT "rate  (35,)" raw text
    assert c.current_signal() == "rate"
    # SWITCH to a different leaf -> the collapsed text tracks it LIVE (the bug: stuck on the first pick)
    c._on_tree_clicked(occupied.index())
    assert c._display_text() == "occupancy · occupied"
    assert c.current_signal() == "occupied"
