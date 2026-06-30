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


def test_flat_combo_popup_always_opens_below_even_near_the_screen_bottom(monkeypatch):
    """The drop-down must ALWAYS sit BELOW the box, never flip ABOVE it -- even when the box is near the
    screen bottom (where stock Qt flips the popup upward).  ``_place_popup_below`` forces the container to
    the box's bottom edge + the outer gap; the height clamps to the below-the-box space so an overrun
    SCROLLS rather than flipping up.  Guards the rule on the BASE FluentComboBox so every combo inherits
    it."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets, QtCore
    from Zou_lab_control.frontend.qt_fluent import FluentComboBox
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scr = app.primaryScreen().availableGeometry()
    c = FluentComboBox()
    c.addItems([f"item {i}" for i in range(30)])           # many items so it would prefer to flip up if it could
    c.resize(160, 30)
    c.move(scr.left() + 40, scr.bottom() - 50)             # bottom of the box ~50 px from the screen bottom
    c.show()
    c.showPopup()
    container = c.view().window()
    combo_bottom = c.mapToGlobal(QtCore.QPoint(0, c.height())).y()
    container_top = container.mapToGlobal(QtCore.QPoint(0, 0)).y()
    assert container_top >= combo_bottom - 1, \
        "the drop-down must open BELOW the box (its top at/under the box bottom), never flipped above"
    c.hidePopup()


def test_tree_combo_popup_always_below_and_clamps_to_below_space_then_scrolls(monkeypatch):
    """The collapsible tree picker obeys the SAME always-below rule, and its grow cap is the below-the-box
    space: anchored near the screen bottom, the popup opens BELOW and ``_anchor_available_height`` /
    ``_desired_popup_height`` clamp to the (small) below space so the tree SCROLLS instead of flipping up
    or overrunning the screen.  ``_anchor_available_height`` measures from the box BOTTOM down only -- it
    has no 'opened above' branch anymore."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scr = app.primaryScreen().availableGeometry()
    c = _combo()
    c.resize(180, 30)
    c.move(scr.left() + 40, scr.bottom() - 50)             # near the bottom
    c.show()
    c.showPopup()
    tree = c.view()
    tree.setExpanded(c._model.index(1, 0), True)
    tree.setExpanded(c._model.index(2, 0), True)
    c._resize_popup_to_contents()
    container = tree.window()
    combo_bottom = c.mapToGlobal(QtCore.QPoint(0, c.height())).y()
    container_top = container.mapToGlobal(QtCore.QPoint(0, 0)).y()
    assert container_top >= combo_bottom - 1, "the tree popup must open BELOW the box, never flipped above"
    # the available height is measured from the box BOTTOM down (no 'above' side); near the bottom it is small
    avail = c._anchor_available_height()
    full = c._desired_popup_height()
    assert 0 < avail < 100, "near the screen bottom the below-the-box space is small (-> the popup scrolls)"
    assert full <= avail, "the desired height clamps to the below-the-box space (scroll, never flip up)"
    c.hidePopup()
