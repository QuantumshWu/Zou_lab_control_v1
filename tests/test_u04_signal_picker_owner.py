"""Current contracts for the one frontend-owned grouped signal picker."""

from __future__ import annotations

from pathlib import Path

from gui_user_flow import configure_offscreen_fast_path
from zlc_frontend.qt_widgets import (
    FluentTreeComboBox,
    coerce_short_labels,
    ensure_qt_app,
    fill_grouped_signal_combo,
    grouped_signal_items,
    read_editable_combo,
    signal_state,
    signal_tree_groups,
)
from zlc_frontend.qt_widgets import signal_picker


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (
    "coerce_short_labels",
    "fill_grouped_signal_combo",
    "grouped_signal_items",
    "read_editable_combo",
    "signal_state",
    "signal_tree_groups",
)


def test_the_helpers_are_owned_by_the_widget_package():
    for name in PUBLIC:
        assert getattr(signal_picker, name).__module__ == (
            "zlc_frontend.qt_widgets.signal_picker"
        ), name




def test_the_moved_module_stayed_dependency_closed():
    """It may import this package's widgets and nothing from the old tree."""

    import ast

    path = ROOT / "zlc_frontend" / "qt_widgets" / "signal_picker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("Zou_lab_control", "zlc_neutral_atom", "zlc_workbench")
    for module in imported:
        assert not any(module.startswith(bad) for bad in forbidden), module


def test_grouping_still_reads_the_same_two_signal_states():
    names = ["cam_frame", "roi_sum"]
    formats = {"cam_frame": "1920x1200 uint8"}

    assert signal_state("cam_frame", formats) == "ready"
    assert signal_state("roi_sum", formats) == "waiting"

    # `sources` maps a signal to the LIST of nodes that produce it.
    groups = signal_tree_groups(
        names,
        {"cam_frame": ["Camera"], "roi_sum": ["Camera"]},
        formats,
    )
    assert groups, "one producing node must yield one group"
    producers = [producer for producer, _leaves in groups]
    assert producers == ["Camera"]
    leaves = groups[0][1]
    assert {bare for _label, bare, _full in leaves} == set(names)


def test_the_flat_projection_shares_the_same_grouping():
    items = grouped_signal_items(
        ["a_x", "a_y"],
        {"a_x": ["Node"], "a_y": ["Node"]},
        {"a_x": "(4,)"},
    )
    # One disabled group header plus one row per signal.
    assert items[0] == ("Node", None)
    assert [name for _label, name in items[1:]] == ["a_x", "a_y"]


def test_coerce_short_labels_normalises_one_provider_callback():
    """It takes a CALLBACK, not a mapping, and swallows a failing provider."""

    assert coerce_short_labels(lambda: {"b": "B", "c": ""}) == {"b": "B"}
    assert coerce_short_labels(None) == {}
    assert coerce_short_labels({"a": "A"}) == {}, "a bare mapping is not callable"

    def broken():
        raise RuntimeError("provider blew up")

    assert coerce_short_labels(broken) == {}


def test_fill_grouped_signal_combo_is_re_exported_for_widget_callers():
    """The filler ships with the widget; constructing Qt here is left to the
    existing GUI suites, which already drive the real picker end to end."""

    assert callable(fill_grouped_signal_combo)
    assert fill_grouped_signal_combo.__module__ == (
        "zlc_frontend.qt_widgets.signal_picker"
    )
    assert callable(read_editable_combo)


def test_nested_signal_leaf_is_selected_by_the_real_popup_click():
    """The public tree must work through the same press/release path as a user."""

    from PyQt5 import QtCore, QtTest, QtWidgets

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    body = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(body)
    combo = FluentTreeComboBox(body)
    layout.addWidget(combo)
    fill_grouped_signal_combo(
        combo,
        names=("camera_frame",),
        sources={"camera_frame": ("Camera",)},
        formats={},
        labels={"camera_frame": "frame"},
        current="",
    )
    body.show()
    application.processEvents()

    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents()
    view = combo.view()
    parent = combo.model().index(0, 0)
    QtTest.QTest.mouseClick(
        view.viewport(),
        QtCore.Qt.LeftButton,
        pos=view.visualRect(parent).center(),
    )
    application.processEvents()
    assert view.isExpanded(parent)
    child = combo.model().index(0, 0, parent)
    view.scrollTo(child)
    application.processEvents()
    clicked = QtTest.QSignalSpy(view.clicked)
    picked = QtTest.QSignalSpy(combo.signalPicked)
    position = view.visualRect(child).center()
    assert view.indexAt(position) == child
    QtTest.QTest.mouseClick(
        view.viewport(),
        QtCore.Qt.LeftButton,
        pos=position,
    )
    application.processEvents()

    assert len(clicked) == 1
    assert len(picked) == 1
    assert combo.current_signal() == "camera_frame"
    body.close()
    body.deleteLater()
    application.processEvents()


def test_explicit_topology_delta_preserves_items_selection_and_expansion():
    """Adding a Figure output mutates the model, not the picker widget tree."""

    from PyQt5 import QtCore, QtWidgets

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    combo = FluentTreeComboBox()
    combo.set_signal_tree(
        (("Camera", (("frame", "camera.frame", "Camera · frame"),)),),
        current="camera.frame",
    )
    view = combo.view()
    assert isinstance(view, QtWidgets.QTreeView)
    camera_parent = combo.model().item(0)
    frame_item = camera_parent.child(0)
    view.setExpanded(camera_parent.index(), True)

    combo.set_signal_tree(
        (
            (
                "Camera",
                (
                    ("frame [1x1x(32,32)]", "camera.frame", "Camera · frame"),
                    ("center x", "fit.center_x", "Camera · center x"),
                ),
            ),
        ),
        current="camera.frame",
    )

    assert combo.model().item(0) is camera_parent
    assert camera_parent.child(0) is frame_item
    assert camera_parent.child(1).data(QtCore.Qt.UserRole) == "fit.center_x"
    assert combo.current_signal() == "camera.frame"
    assert view.isExpanded(camera_parent.index())
    combo.deleteLater()
    application.processEvents()
