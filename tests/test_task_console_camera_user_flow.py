"""Formal operator flow: DeviceManager -> Camera -> signal -> live 2-D panel.

The fixture chooses only the offscreen Qt platform.  Every product transition is
driven through the same visible controls as the desktop launcher; assertions may
inspect the resulting typed fronts, but never bypass a button to create them.
"""

from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import patch

import numpy as np


from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from gui_user_flow import (
    capture_offscreen_window,
    click_tab,
    configure_offscreen_fast_path,
    drag_mouse_move,
    require_offscreen_platform,
    until,
    widget_gone,
)
from zlc_workbench.task_console.console_records import console_signal_key
from zlc_frontend.qt_widgets import ensure_qt_app


def _choose_combo_data(combo, value, application) -> None:
    """Choose a normal combo entry through its visible popup and keyboard."""

    row = combo.findData(value)
    assert row >= 0, (value, [combo.itemData(i) for i in range(combo.count())])
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _ in range(row):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    assert combo.currentData() == value


def _choose_combo_text(combo, text, application) -> None:
    """Choose one visible menu label without depending on QVariant coercion."""

    row = combo.findText(text)
    assert row >= 0, (text, [combo.itemText(i) for i in range(combo.count())])
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _ in range(row):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    assert combo.currentText() == text


def _choose_signal_leaf(combo, signal, application) -> None:
    """Expand the visible producer row and click the requested signal leaf."""

    # Logic Edit forms are real scroll pages.  A user must bring a low field
    # into view before its below-anchored popup has usable screen height.
    ancestor = combo.parentWidget()
    while ancestor is not None and not isinstance(
        ancestor,
        QtWidgets.QAbstractScrollArea,
    ):
        ancestor = ancestor.parentWidget()
    if isinstance(ancestor, QtWidgets.QAbstractScrollArea):
        bar = ancestor.verticalScrollBar()
        if bar.maximum() > bar.minimum():
            QtTest.QTest.keyClick(bar, QtCore.Qt.Key_End)
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    model = combo.model()
    parent_index = child_index = None
    for parent_row in range(model.rowCount()):
        candidate_parent = model.index(parent_row, 0)
        for child_row in range(model.rowCount(candidate_parent)):
            candidate_child = model.index(child_row, 0, candidate_parent)
            if candidate_child.data(QtCore.Qt.UserRole) == signal:
                parent_index = candidate_parent
                child_index = candidate_child
                break
        if child_index is not None:
            break
    assert parent_index is not None and child_index is not None
    view.scrollTo(parent_index)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    if not view.isExpanded(parent_index):
        QtTest.QTest.mouseClick(
            view.viewport(),
            QtCore.Qt.LeftButton,
            pos=view.visualRect(parent_index).center(),
        )
        until(application, lambda: view.isExpanded(parent_index))
    # Expanding changes the popup's fitted height.  Let that formal widget
    # transition finish, then scroll the actual leaf into the visible viewport
    # before clicking it just as an operator would.
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    if not view.isVisible():
        QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
        until(application, view.isVisible)
    view.scrollTo(child_index)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    assert view.visualRect(child_index).isValid()
    click_position = view.visualRect(child_index).center()
    assert combo.isEnabled() and view.isEnabled() and view.isVisible()
    assert view.rect().contains(click_position), (
        view.visualRect(child_index),
        view.rect(),
    )
    hit = view.indexAt(click_position)
    assert hit == child_index, (
        view.visualRect(child_index),
        view.rect(),
        hit.data(QtCore.Qt.UserRole),
        child_index.data(QtCore.Qt.UserRole),
    )
    clicked = QtTest.QSignalSpy(view.clicked)
    picked = QtTest.QSignalSpy(combo.signalPicked)
    QtTest.QTest.mouseClick(
        view.viewport(),
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    assert len(clicked) == 1, "the visible tree leaf did not receive a Qt click"
    assert len(picked) == 1, "the tree combo did not publish the selected leaf"
    until(application, lambda: combo.currentData() == signal)


def _signal_leaf_keys(combo) -> set[str]:
    """Return the exact leaves currently visible through the product picker."""

    model = combo.model()
    found: set[str] = set()

    def visit(parent=QtCore.QModelIndex()) -> None:
        for row in range(model.rowCount(parent)):
            index = model.index(row, 0, parent)
            value = index.data(QtCore.Qt.UserRole)
            if value:
                found.add(str(value))
            visit(index)

    visit()
    return found


def _replace_spin_value(spin, text: str) -> None:
    """Edit the visible numeric control exactly as an operator would.

    Numeric owner declarations project to shared Fluent spin controls; an
    optional value uses the control's explicit ``Auto`` state for ``None``.
    """

    edit = spin.lineEdit() if hasattr(spin, "lineEdit") else spin
    QtTest.QTest.mouseClick(edit, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(edit, str(text))
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Return)


def _replace_path_value(path_widget, text: str) -> None:
    """Edit the text field of the shared visible path control."""

    edit = path_widget.edit
    QtTest.QTest.mouseClick(edit, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(edit, str(text))
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Return)


def _replace_axis_range(widget, minimum: str, maximum: str, points: str) -> None:
    """Edit the three visible controls of one swept axis as an operator would."""

    _replace_spin_value(widget.min_spin, minimum)
    _replace_spin_value(widget.max_spin, maximum)
    _replace_spin_value(widget.pts_spin, points)


def _visible_form_widgets(editor) -> dict[str, QtWidgets.QWidget]:
    """Resolve the current stable controls through the form owner's public API."""

    form = editor.form._parameter_form
    assert form is not None
    return {key: form.widget_for(key) for key in form.spec.keys}


def _resolved_artifact(console, output_key: str):
    """Read one retained FINAL Artifact through the typed producer boundary."""

    producer = console.resolve_console_producer(output_key)
    return producer.artifact if producer.artifact_resolved else None


def _wheel(widget, position, delta: int) -> None:
    """Deliver one wheel step through the real Qt widget event path."""

    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(widget.mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, int(delta)),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    widget.wheelEvent(event)
    assert event.isAccepted()


def _add_plot_and_bind(console, add_button, kind: str, signal: str, application):
    """Add one blank plot and wire its Setting popup through visible controls."""

    before = len(console.cards)
    _choose_combo_data(console.kind_combo, kind, application)
    QtTest.QTest.mouseClick(add_button, QtCore.Qt.LeftButton)
    assert len(console.cards) == before + 1
    card = console.cards[-1]
    click_tab(console, console.tabs.widget(0))
    QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
    until(application, lambda: card.settings_popup.isVisible())
    _choose_signal_leaf(card.signal_combo, signal, application)
    assert card.config.signal == signal
    return card


def test_grid_setting_edit_and_fit_remain_one_bounded_formal_window(
    tmp_path,
) -> None:
    """Grid authoring stays typed, shared, embedded, and width bounded."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    require_offscreen_platform(application)
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        SPATIAL_X,
        SPATIAL_Y,
        AxisId,
        AxisSpec,
        BlockId,
        CellValidity,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointLayout,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )
    from zlc_frontend.figure import (
        AxisViewRole,
        ViewIntent,
        grid_facet_axes,
        grid_facet_axis,
        view_spec_from_tree,
    )
    from zlc_frontend.qt_widgets import (
        FitAuthoringPane,
        FluentSectionLabel,
        FluentSettingRow,
    )
    from zlc_workbench.task_console.data_plane import ConsoleSignalValue

    repeat = AxisSpec(
        AxisId("formal.grid.repeat"),
        "repeat",
        REPEAT,
        2,
        (0, 1),
    )
    scan_x = AxisSpec(
        AxisId("formal.grid.bx"),
        "Bx",
        SCAN_POINT,
        3,
        (-1.0, 0.0, 1.0),
        "code",
    )
    scan_y = AxisSpec(
        AxisId("formal.grid.by"),
        "By",
        SCAN_POINT,
        2,
        (-2.0, 2.0),
        "code",
    )
    image_y = AxisSpec(
        AxisId("formal.grid.image-y"),
        "camera y",
        SPATIAL_Y,
        4,
        (0.0, 1.0, 2.0, 3.0),
        "px",
    )
    image_x = AxisSpec(
        AxisId("formal.grid.image-x"),
        "camera x",
        SPATIAL_X,
        5,
        (0.0, 1.0, 2.0, 3.0, 4.0),
        "px",
    )
    schema = DatasetSchema(
        repeat,
        (scan_x, scan_y),
        PointLayout.rect_c((scan_x.size, scan_y.size)),
        ValueSchema(
            (image_y, image_x),
            ValidityContract.value(),
            np.dtype("float64"),
            "counts",
        ),
    )
    values = np.arange(2 * 6 * 4 * 5, dtype=np.float64).reshape(2, 6, 4, 5)
    block = DataBlock(
        BlockId("formal-grid-block"),
        DatasetRevision(1),
        values,
        CellValidity(np.ones(values.shape[:2], dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("formal-grid-generation")),
        block,
    )
    grid_value = ConsoleSignalValue(
        name="formal-grid",
        source="formal immutable monitor boundary",
        snapshot=snapshot,
        coverage=None,
        run_id="formal-run",
        epoch_id="formal-epoch",
        join_digest="0" * 64,
    )

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "grid-setting-edit-fit",
            "--seed",
            "43",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        # Give Logic one formal row, then author the Grid itself through the
        # exact visible Add-Panel controls.
        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        logic_row = console.logic_nodes[-1]
        _choose_combo_data(console.kind_combo, "grid", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        card = console.cards[-1]
        assert card.config.kind == "grid"
        click_tab(console, console.tabs.widget(0))

        # A producer fixture may enter only at the immutable monitor boundary.
        # Two equally legal declared point axes intentionally leave the Grid
        # facet unresolved until the operator chooses one in Setting.
        card.config.signal = grid_value.name
        assert card._freeze_value_render_request(grid_value, 1, force=True) is None
        assert card._current_schema() == schema
        assert "choose a named facet axis" in card._status_text

        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        outer_geometry = QtCore.QRect(console_wrapper.geometry())
        outer_frame = QtCore.QRect(console_wrapper.frameGeometry())

        def visible_top_levels():
            return {
                widget
                for widget in application.topLevelWidgets()
                if widget.isVisible()
            }

        top_levels_before = visible_top_levels()
        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        until(application, lambda: card.settings_popup.isVisible())
        popup_width = card.settings_popup.width()
        assert popup_width == card.settings_popup.minimumWidth()
        assert popup_width == card.settings_popup.maximumWidth()
        assert (
            card._settings_scroll.horizontalScrollBarPolicy()
            == QtCore.Qt.ScrollBarAlwaysOff
        )

        # The typed resolver owns every legal named facet.  Qt only presents
        # those AxisId values and commits the one the operator clicks.
        _choose_combo_text(card.grid_intent_combo, "2d", application)
        expected_axes = grid_facet_axes(schema, ViewIntent.IMAGE)
        until(
            application,
            lambda: card.grid_facet_combo.count() == len(expected_axes),
        )
        offered_axis_ids = tuple(
            card.grid_facet_combo.itemData(index)
            for index in range(card.grid_facet_combo.count())
        )
        assert offered_axis_ids == tuple(axis.axis_id for axis in expected_axes)
        assert all(isinstance(axis_id, AxisId) for axis_id in offered_axis_ids)
        scan_x_index = offered_axis_ids.index(scan_x.axis_id)
        _choose_combo_text(
            card.grid_facet_combo,
            card.grid_facet_combo.itemText(scan_x_index),
            application,
        )
        until(
            application,
            lambda: "view_spec" in card.config.params,
        )
        authored = view_spec_from_tree(card.config.params["view_spec"])
        assert authored.intent is ViewIntent.IMAGE
        assert grid_facet_axis(authored) == scan_x.axis_id
        assert sum(
            binding.role is AxisViewRole.FACET
            for binding in authored.axis_bindings
        ) == 1
        assert not card.grid_colormap_row.isHidden()
        assert card.grid_bin_count_row.isHidden()
        assert card.grid_count_scale_row.isHidden()

        # The same typed facet survives a legal sub-plot change; the shared
        # inventory reveals bins/log-count and hides colormap without rebuilding
        # either Setting or Edit from a second schema.
        _choose_combo_text(card.grid_intent_combo, "hist", application)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if card._grid_cell_intent() is not ViewIntent.HISTOGRAM:
            offered_axis_ids = tuple(
                card.grid_facet_combo.itemData(index)
                for index in range(card.grid_facet_combo.count())
            )
            scan_x_index = offered_axis_ids.index(scan_x.axis_id)
            _choose_combo_text(
                card.grid_facet_combo,
                card.grid_facet_combo.itemText(scan_x_index),
                application,
            )
        until(
            application,
            lambda: card._grid_cell_intent() is ViewIntent.HISTOGRAM,
        )
        authored = view_spec_from_tree(card.config.params["view_spec"])
        assert grid_facet_axis(authored) == scan_x.axis_id
        assert not card.grid_bin_count_row.isHidden()
        assert not card.grid_count_scale_row.isHidden()
        assert card.grid_colormap_row.isHidden()

        inventory_names = (
            "grid_facet_row",
            "grid_intent_row",
            "grid_bin_count_row",
            "grid_count_scale_row",
            "grid_colormap_row",
        )
        setting_rows = tuple(getattr(card, name) for name in inventory_names)
        assert tuple(row._label.text() for row in setting_rows) == (
            "facet",
            "sub plot",
            "Bins",
            "Log count",
            "Colormap",
        )
        label_width = card.setting_label_width(card.fontMetrics())
        assert {row._label.width() for row in setting_rows} == {label_width}

        # Unbroken errors and user-authored fields are ordinary bounded content.
        long_text = "unbroken-field-or-error-" + "W" * 1200
        card.set_status(long_text, error=True)
        QtTest.QTest.keyClick(
            card._settings_scroll.verticalScrollBar(),
            QtCore.Qt.Key_End,
        )
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert not card.title_edit.visibleRegion().isEmpty()
        QtTest.QTest.mouseClick(card.title_edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(
            card.title_edit,
            QtCore.Qt.Key_A,
            QtCore.Qt.ControlModifier,
        )
        QtTest.QTest.keyClicks(card.title_edit, "Grid " + "X" * 400)
        QtTest.QTest.keyClick(card.title_edit, QtCore.Qt.Key_Return)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert card.settings_popup.width() == popup_width
        assert console_wrapper.geometry() == outer_geometry
        assert console_wrapper.frameGeometry() == outer_frame

        class _TopLevelShowProbe(QtCore.QObject):
            def __init__(self):
                super().__init__()
                self.shown = []

            def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
                try:
                    if (
                        event.type() == QtCore.QEvent.Show
                        and isinstance(watched, QtWidgets.QWidget)
                        and watched.isWindow()
                    ):
                        self.shown.append(type(watched).__name__)
                except RuntimeError:
                    # A widget can be deleted by the same event turn; deletion
                    # is not itself evidence that a top-level was presented.
                    pass
                return False

        assert not card.edit_button.visibleRegion().isEmpty()
        probe = _TopLevelShowProbe()
        application.installEventFilter(probe)
        try:
            QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
            until(
                application,
                lambda: id(card) in console._panel_editors,
            )
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        finally:
            application.removeEventFilter(probe)
        assert probe.shown == []
        assert visible_top_levels() == top_levels_before

        editor = console._panel_editors[id(card)]
        assert editor.window() is console_wrapper
        assert not editor.isWindow()
        assert editor._board is not None
        # The scroll page is the board's immediate stable layout owner; the
        # architectural invariant is that it remains inside this editor's
        # QObject tree and never becomes a transient top-level.
        assert editor.isAncestorOf(editor._board)
        assert not editor._board.isWindow()
        assert not editor._board.isWindow()
        assert editor._scroll.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
        edit_rows = tuple(getattr(editor, name) for name in inventory_names)
        assert tuple(row._label.text() for row in edit_rows) == tuple(
            row._label.text() for row in setting_rows
        )
        assert {row._label.width() for row in edit_rows} == {label_width}
        assert tuple(row.isHidden() for row in edit_rows) == tuple(
            row.isHidden() for row in setting_rows
        )

        # Fit is exactly one embedded model + one single-line args editor in
        # both surfaces.  Neither pane is a window or owns another figure host.
        assert any(
            section.text() == "Fit"
            for section in card.settings_popup.findChildren(FluentSectionLabel)
        )
        assert any(
            section.text() == "Fit"
            for section in editor.findChildren(FluentSectionLabel)
        )
        fit_panes = (card.fit_authoring_pane, editor._fit_pane)
        for pane in fit_panes:
            assert isinstance(pane, FitAuthoringPane)
            assert not pane.isWindow()
            rows = pane.findChildren(
                FluentSettingRow,
                "",
                QtCore.Qt.FindDirectChildrenOnly,
            )
            assert [row._label.text() for row in rows] == ["model", "args"]
            assert {row._label.width() for row in rows} == {label_width}
            assert isinstance(pane.arguments_edit, QtWidgets.QLineEdit)
            assert [pane.fit_button.text(), pane.clear_button.text()] == [
                "Fit",
                "Clear",
            ]
            assert not pane.findChildren(QtWidgets.QTextEdit)

        editor.status.setText(long_text)
        QtTest.QTest.mouseClick(editor.title_edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(
            editor.title_edit,
            QtCore.Qt.Key_A,
            QtCore.Qt.ControlModifier,
        )
        QtTest.QTest.keyClicks(editor.title_edit, "Edit " + "Y" * 400)
        QtTest.QTest.keyClick(editor.title_edit, QtCore.Qt.Key_Return)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert editor._scroll.horizontalScrollBar().maximum() == 0
        assert console_wrapper.geometry() == outer_geometry
        assert console_wrapper.frameGeometry() == outer_frame

        # Logic uses the same width-neutral error primitive; Monitor alone keeps
        # both scroll directions available for authored panel geometry.
        logic_row.set_state("error", status=long_text)
        click_tab(console, console.tabs.widget(1))
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert console.logic_scroll.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
        assert console.logic_scroll.horizontalScrollBar().maximum() == 0
        assert console.scroll.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
        assert console.scroll.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
        assert console_wrapper.geometry() == outer_geometry
        assert console_wrapper.frameGeometry() == outer_frame

        click_tab(console, editor)
        shot = capture_offscreen_window(
            application,
            console,
            tmp_path / "grid-setting-edit-fit.png",
            settle_ms=50,
        )
        assert shot["window_client"] == {
            "x": outer_geometry.x(),
            "y": outer_geometry.y(),
            "width": outer_geometry.width(),
            "height": outer_geometry.height(),
        }
        assert shot["window_frame"] == {
            "x": outer_frame.x(),
            "y": outer_frame.y(),
            "width": outer_frame.width(),
            "height": outer_frame.height(),
        }
        pixel_ratio = shot["device_pixel_ratio"]
        assert abs(
            shot["image_pixels"]["width"] / pixel_ratio
            - outer_geometry.width()
        ) <= 1
        assert abs(
            shot["image_pixels"]["height"] / pixel_ratio
            - outer_geometry.height()
        ) <= 1
        assert visible_top_levels() == top_levels_before
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_device_manager_camera_signal_drives_a_changing_2d_front(tmp_path) -> None:
    """The actual standalone entry's camera chain is live and dimensioned."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "camera-user-flow",
            "--seed",
            "31",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()

        assert sum(
            console.kind_combo.itemText(index) == "Measurement: Camera"
            for index in range(console.kind_combo.count())
        ) == 1
        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        assert len(console.logic_nodes) == 1
        row = console.logic_nodes[0]
        editor = console._logic_editors[id(row)]

        # Virtual MOT camera is a true free-running source, so this is the
        # deterministic live Camera role for an operator-path acceptance run.
        widgets = _visible_form_widgets(editor)
        role_combo = widgets["camera_role"]
        _choose_combo_data(role_combo, "mot_camera", application)
        _replace_spin_value(widgets["frames_per_cycle"], "3")
        _replace_spin_value(widgets["exposure"], "0.013")
        QtTest.QTest.mouseClick(editor.form.start_button, QtCore.Qt.LeftButton)
        frame_signals = tuple(
            console_signal_key(row.node.node_id, f"frame_{index}")
            for index in range(3)
        )
        signal = frame_signals[1]
        until(
            application,
            lambda: all(
                console._data.freeze().value(key) is not None
                for key in frame_signals
            ),
            timeout=15.0,
        )
        first_value = console._data.freeze().value(signal)
        until(
            application,
            lambda: (
                (value := console._data.freeze().value(signal)) is not None
                and value.snapshot.ref != first_value.snapshot.ref
            ),
            timeout=10.0,
        )

        # The visible Logic row must expose the data dimensions, not merely an
        # unbound signal name.
        until(
            application,
            lambda: (
                all(name in row.publishes_label.text() for name in (
                    "frame_0",
                    "frame_1",
                    "frame_2",
                ))
                and "—" not in row.publishes_label.text()
            ),
            timeout=3.0,
        )

        _choose_combo_data(console.kind_combo, "2d", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        assert len(console.cards) == 1
        card = console.cards[0]
        click_tab(console, console.tabs.widget(0))
        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        until(application, lambda: card.settings_popup.isVisible())
        _choose_signal_leaf(card.signal_combo, signal, application)
        assert card.config.signal == signal

        until(
            application,
            lambda: card.board is not None and card.board.front_frame is not None,
            timeout=15.0,
        )
        first_front = card.board.front_frame
        until(
            application,
            lambda: (
                card.board.front_frame is not None
                and card.board.front_frame.sequence > first_front.sequence
            ),
            timeout=10.0,
        )
        second_front = card.board.front_frame
        assert bytes(second_front.panels[0].raster.pixels) != bytes(
            first_front.panels[0].raster.pixels
        )

        # No-button movement is inert: the board does not even request tracking.
        board = card.board.board
        assert not board.hasMouseTracking()
        before = card.frozen_figure_output_state()[1:3]
        QtTest.QTest.mouseMove(board, board.rect().center())
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert card.frozen_figure_output_state()[1:3] == before

        # The product selector is one shared gesture owner, exercised here on
        # the actual live TaskConsole front rather than a synthetic board.  It
        # publishes only completed Figure-owned gestures; ordinary motion is
        # still inert and no pointer-motion data hover exists.
        QtTest.QTest.mouseClick(
            console.selectors_switch,
            QtCore.Qt.LeftButton,
        )
        until(application, lambda: board.selectors_enabled)
        binding = board._image_bindings[card.panel_id]
        target = board._selector_target(binding)
        assert target is not None
        plot = target[0]

        def plot_point(x_fraction: float, y_fraction: float) -> QtCore.QPoint:
            return QtCore.QPoint(
                int(round(plot.left() + x_fraction * plot.width())),
                int(round(plot.top() + y_fraction * plot.height())),
            )

        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=plot_point(0.68, 0.42),
        )
        until(
            application,
            lambda: card.frozen_figure_output_state()[2] is not None,
        )

        area_start = plot_point(0.18, 0.20)
        area_end = plot_point(0.48, 0.58)
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=area_start,
        )
        drag_mouse_move(board, area_end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.LeftButton,
            pos=area_end,
        )
        until(
            application,
            lambda: card.frozen_figure_output_state()[1] is not None,
        )

        # A fresh, non-dragging click in blank plot space clears Area.  It does
        # not restore the old rectangle and does not affect the locked Cross.
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.LeftButton,
            pos=plot_point(0.84, 0.84),
        )
        until(
            application,
            lambda: card.frozen_figure_output_state()[1] is None,
        )
        assert card.frozen_figure_output_state()[2] is not None

        # Keep one middle-button gesture alive across a worker answer.  The
        # second motion must rebase from the newly painted revision while
        # retaining the exact held input instead of crashing or splicing in a
        # newer live-camera value.
        zoom_revision = card._display_revision
        _wheel(board, plot_point(0.52, 0.52), -120)
        until(
            application,
            lambda: card._display_revision > zoom_revision,
        )
        zoom_revision = card._display_revision
        until(
            application,
            lambda: (
                (origin := board.visible_image_origin()) is not None
                and origin.presentation.panel_revision >= zoom_revision
            ),
            timeout=15.0,
        )
        pan_start = plot_point(0.52, 0.52)
        first_move = plot_point(0.55, 0.52)
        second_move = plot_point(0.45, 0.52)
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.MiddleButton,
            pos=pan_start,
        )
        initial_revision = card._display_revision
        drag_mouse_move(board, first_move, QtCore.Qt.MiddleButton)
        first_revision = card._display_revision
        assert first_revision > initial_revision
        until(
            application,
            lambda: (
                (origin := board.visible_image_origin()) is not None
                and origin.presentation.panel_revision >= first_revision
            ),
            timeout=15.0,
        )
        drag_mouse_move(board, second_move, QtCore.Qt.MiddleButton)
        assert card._display_revision > first_revision
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=second_move,
        )
        assert board.image_selector_fault(card.panel_id) is None

        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-live-2d.png",
            settle_ms=100,
        )

        # Open the real per-panel Edit surface from the Setting popup.  It must
        # reuse the accepted front inside the tab, not launch a second window.
        if not card.settings_popup.isVisible():
            QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        until(application, lambda: card.settings_popup.isVisible())
        QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: (
                id(card) in console._panel_editors
                and console._panel_editors[id(card)]._board is not None
            ),
        )
        edit = console._panel_editors[id(card)]
        assert edit.window() is console.window()
        assert edit._board.isVisible(), (
            console.tabs.currentWidget() is edit,
            edit.isVisible(),
            edit._board.isHidden(),
            console.tabs.currentIndex(),
            console.tabs.indexOf(edit),
        )
        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-live-edit.png",
            settle_ms=100,
        )
        edit_scroll = next(
            scroll
            for scroll in edit.findChildren(QtWidgets.QScrollArea)
            if scroll.isVisible()
        )
        QtTest.QTest.keyClick(
            edit_scroll.verticalScrollBar(),
            QtCore.Qt.Key_End,
        )
        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-live-edit-scrolled.png",
            settle_ms=100,
        )
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_calibration_and_mot_tasks_open_their_declared_live_panels(tmp_path) -> None:
    """The two flagship tasks run from their real forms and open typed panels."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "task-user-flow",
            "--seed",
            "37",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        _choose_combo_text(console.kind_combo, "Task: Calibrate readout", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        calibration_row = console.logic_nodes[-1]
        calibration_editor = console._logic_editors[id(calibration_row)]
        calibration_widgets = _visible_form_widgets(calibration_editor)
        assert set(calibration_widgets) == {
            "source_mode",
            "folder",
            "save_frames",
            "pulse",
            "threshold_method",
            "reference_exposure_s",
            "readout_exposure_s",
            "threshold_frames",
            "roi_radius",
            "camera_role",
        }
        _replace_path_value(
            calibration_widgets["folder"],
            str(tmp_path / "calibration-output"),
        )
        _replace_spin_value(calibration_widgets["threshold_frames"], "10")
        QtTest.QTest.mouseClick(
            calibration_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )

        calibration_frame = console_signal_key(calibration_row.node.node_id, "frame")
        saw_calibration_panel = False
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            saw_calibration_panel = saw_calibration_panel or any(
                card.config.signal == calibration_frame
                and card.config.kind == "2d"
                for card in console.cards
            )
            if not console._task_locked:
                break
            time.sleep(0.005)
        assert saw_calibration_panel
        assert not console._task_locked
        calibration_final = console_signal_key(
            calibration_row.node.node_id,
            "calibration",
        )
        assert _resolved_artifact(console, calibration_final) is not None, (
            calibration_row.status_label.text()
        )
        fidelity_site = console_signal_key(
            calibration_row.node.node_id,
            "fidelity_site",
        )
        threshold_site = console_signal_key(
            calibration_row.node.node_id,
            "fidelity_threshold",
        )
        centers_site = console_signal_key(
            calibration_row.node.node_id,
            "fidelity_centers",
        )
        aggregate_fidelity = console_signal_key(
            calibration_row.node.node_id,
            "aggregate_fidelity",
        )
        global_fidelity = console_signal_key(
            calibration_row.node.node_id,
            "global_fidelity",
        )
        calibration_front = console._data.freeze()
        fidelity_value = calibration_front.value(fidelity_site)
        threshold_value = calibration_front.value(threshold_site)
        centers_value = calibration_front.value(centers_site)
        assert fidelity_value is not None
        assert threshold_value is not None
        assert centers_value is not None
        assert calibration_front.value(aggregate_fidelity) is not None
        assert calibration_front.value(global_fidelity) is not None
        site_axis = fidelity_value.schema.cell_schema.data_axes[0]
        assert site_axis.role.value == "site"
        assert fidelity_value.schema.physical_shape == (1, 1, site_axis.size)
        assert threshold_value.schema.cell_schema.data_axes == (site_axis,)
        assert centers_value.schema.physical_shape == (1, 1, site_axis.size, 2)
        assert "site fidelity" in calibration_row.publishes_label.text()
        assert (
            f"1 × 1 × ({site_axis.size})"
            in calibration_row.publishes_label.text()
        )

        fidelity_card = _add_plot_and_bind(
            console,
            add,
            "1d",
            fidelity_site,
            application,
        )
        until(
            application,
            lambda: (
                fidelity_card.board is not None
                and fidelity_card.board.front_frame is not None
            ),
            timeout=15.0,
        )
        assert (
            fidelity_card.frozen_plot_panel_contract().value_label
            == "Readout fidelity"
        )

        _choose_combo_text(console.kind_combo, "Task: Optimize MOT field", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        mot_row = console.logic_nodes[-1]
        mot_editor = console._logic_editors[id(mot_row)]
        mot_widgets = _visible_form_widgets(mot_editor)
        assert set(mot_widgets) == {
            "pulse",
            "center_x",
            "center_y",
            "center_z",
            "span",
            "points",
            "roi_cx",
            "roi_cy",
            "roi_radius",
            "folder",
            "camera_role",
        }
        _replace_spin_value(mot_widgets["points"], "2")
        _replace_path_value(
            mot_widgets["folder"],
            str(tmp_path / "mot-output"),
        )
        QtTest.QTest.mouseClick(
            mot_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        mot_grid = console_signal_key(mot_row.node.node_id, "grid")
        saw_mot_panel = False
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            saw_mot_panel = saw_mot_panel or any(
                card.config.signal == mot_grid
                and card.config.kind == "grid"
                for card in console.cards
            )
            if not console._task_locked:
                break
            time.sleep(0.005)
        assert saw_mot_panel
        assert not console._task_locked
        mot_final = console_signal_key(mot_row.node.node_id, "mot_field")
        mot_value = console._data.freeze().value(mot_final)
        assert mot_value is not None
        mot_schema = mot_value.schema
        assert tuple(axis.role.value for axis in mot_schema.point_axes) == (
            "scan-point",
            "scan-point",
            "scan-point",
        )
        assert mot_schema.cell_schema.is_scalar
        assert tuple(
            axis.size for axis in mot_schema.cell_schema.data_axes
        ) == (1,)
        assert mot_schema.physical_shape == (
            1,
            mot_schema.point_layout.storage_size,
            1,
        )
        mot_shape = f"1 × {mot_schema.point_layout.storage_size} × (1)"
        assert mot_shape in mot_row.publishes_label.text()
        assert "points:" not in mot_row.publishes_label.text()
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_calibration_retires_the_exact_conflicting_camera_row_and_retries(
    tmp_path,
) -> None:
    """A Task uses typed RunId admission to hand off its occupied camera."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "calibration-conflict-handoff",
            "--seed",
            "47",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        camera_row = console.logic_nodes[-1]
        camera_editor = console._logic_editors[id(camera_row)]
        camera_widgets = _visible_form_widgets(camera_editor)
        _choose_combo_data(camera_widgets["camera_role"], "camera", application)
        _replace_spin_value(camera_widgets["repeat"], "0")
        QtTest.QTest.mouseClick(
            camera_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        until(
            application,
            lambda: (
                (node := console._logic_nodes.get(id(camera_row))) is not None
                and node.handle is not None
                and node.running
            ),
            timeout=15.0,
        )
        camera_run_id = console._logic_nodes[id(camera_row)].handle.run_id

        _choose_combo_text(console.kind_combo, "Task: Calibrate readout", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        calibration_row = console.logic_nodes[-1]
        calibration_editor = console._logic_editors[id(calibration_row)]
        calibration_widgets = _visible_form_widgets(calibration_editor)
        _replace_path_value(
            calibration_widgets["folder"],
            str(tmp_path / "calibration-output"),
        )
        _replace_spin_value(calibration_widgets["threshold_frames"], "10")
        _choose_combo_data(
            calibration_widgets["camera_role"],
            "camera",
            application,
        )
        QtTest.QTest.mouseClick(
            calibration_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )

        calibration_signal = console_signal_key(
            calibration_row.node.node_id,
            "calibration",
        )
        until(
            application,
            lambda: (
                _resolved_artifact(console, calibration_signal) is not None
                and not console._task_locked
            ),
            timeout=25.0,
        )
        assert console._logic_nodes.get(id(camera_row)) is None
        assert not camera_row.stop_button.isEnabled()
        assert camera_row.status_label.text() == "stopped"
        assert _resolved_artifact(console, calibration_signal) is not None
        assert camera_run_id != getattr(
            getattr(console._last_node[id(calibration_row)], "handle", None),
            "run_id",
            None,
        )
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_calibration_coupled_measurements_and_live_occupancy_share_one_console(
    tmp_path,
) -> None:
    """The remaining Main readout chain runs only through formal Qt controls."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser
    from zlc_neutral_atom.runtime.run import RunState

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "readout-user-flow",
            "--seed",
            "43",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    pulse_body = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        # First create the exact CalibrationArtifactRef all authoritative
        # classifiers below must select.  This is the only logic layer in this
        # flow that opens its declared run-scoped diagnostic panel.
        _choose_combo_text(console.kind_combo, "Task: Calibrate readout", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        calibration_row = console.logic_nodes[-1]
        calibration_editor = console._logic_editors[id(calibration_row)]
        calibration_widgets = _visible_form_widgets(calibration_editor)
        _replace_path_value(
            calibration_widgets["folder"],
            str(tmp_path / "calibration-output"),
        )
        _replace_spin_value(calibration_widgets["threshold_frames"], "10")
        QtTest.QTest.mouseClick(
            calibration_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        calibration_signal = console_signal_key(
            calibration_row.node.node_id,
            "calibration",
        )
        until(
            application,
            lambda: (
                _resolved_artifact(console, calibration_signal) is not None
                and not console._task_locked
            ),
            timeout=25.0,
        )

        # Temperature takes only the explicit calibration Artifact in addition
        # to Main's visible physics parameters.  A Measurement never auto-opens
        # a panel; the operator creates and wires the 1-D view afterwards.
        _choose_combo_text(console.kind_combo, "Measurement: Temperature", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        temperature_row = console.logic_nodes[-1]
        temperature_editor = console._logic_editors[id(temperature_row)]
        temperature_widgets = _visible_form_widgets(temperature_editor)
        assert set(temperature_widgets) == {
            "pulse",
            "t_off",
            "shots",
            "per_site",
            "calibration",
        }
        assert _signal_leaf_keys(temperature_widgets["calibration"]) == {
            calibration_signal
        }
        _replace_axis_range(temperature_widgets["t_off"], "20", "40", "2")
        _replace_spin_value(temperature_widgets["shots"], "1")
        _choose_signal_leaf(
            temperature_widgets["calibration"],
            calibration_signal,
            application,
        )
        cards_before = tuple(console.cards)
        QtTest.QTest.mouseClick(
            temperature_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        temperature_signal = console_signal_key(
            temperature_row.node.node_id,
            "survival",
        )
        until(
            application,
            lambda: console._data.freeze().value(temperature_signal) is not None,
            timeout=25.0,
        )
        assert tuple(console.cards) == cards_before
        temperature_value = console._data.freeze().value(temperature_signal)
        temperature_axis = temperature_value.snapshot.block.schema.point_axes[0]
        assert (temperature_axis.name, temperature_axis.unit) == (
            "Trap-off time",
            "s",
        )
        temperature_card = _add_plot_and_bind(
            console,
            add,
            "1d",
            temperature_signal,
            application,
        )
        until(
            application,
            lambda: temperature_card.board is not None
            and temperature_card.board.front_frame is not None,
            timeout=15.0,
        )

        # Readout-duration fidelity uses the same exact Calibration output.  It
        # performs its supported camera API update only between duration points;
        # every point's shots remain one hardware-timed FPGA run.
        _choose_combo_text(
            console.kind_combo,
            "Measurement: Fidelity vs duration",
            application,
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        fidelity_row = console.logic_nodes[-1]
        fidelity_editor = console._logic_editors[id(fidelity_row)]
        fidelity_widgets = _visible_form_widgets(fidelity_editor)
        assert set(fidelity_widgets) == {
            "pulse",
            "duration",
            "shots",
            "site",
            "calibration",
        }
        assert _signal_leaf_keys(fidelity_widgets["calibration"]) == {
            calibration_signal
        }
        _replace_axis_range(fidelity_widgets["duration"], "2", "4", "2")
        _replace_spin_value(fidelity_widgets["shots"], "2")
        _choose_signal_leaf(
            fidelity_widgets["calibration"],
            calibration_signal,
            application,
        )
        cards_before = tuple(console.cards)
        QtTest.QTest.mouseClick(
            fidelity_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        fidelity_signal = console_signal_key(fidelity_row.node.node_id, "fidelity")
        until(
            application,
            lambda: console._data.freeze().value(fidelity_signal) is not None,
            timeout=25.0,
        )
        assert tuple(console.cards) == cards_before
        fidelity_card = _add_plot_and_bind(
            console,
            add,
            "1d",
            fidelity_signal,
            application,
        )
        until(
            application,
            lambda: fidelity_card.board is not None
            and fidelity_card.board.front_frame is not None,
            timeout=15.0,
        )

        # The science Camera is externally triggered.  Use the real Pulse GUI
        # On-Pulse control to provide its continuous hardware schedule; neither
        # the Camera Measurement nor Occupancy owns or fakes that producer.
        pulse_body = flow.pulse
        pulse_path = (Path("pulses/probe_template.json")).resolve()
        with patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(pulse_path), "ZLC pulse (*.json)"),
        ):
            QtTest.QTest.mouseClick(
                pulse_body.schedule_view.load_button,
                QtCore.Qt.LeftButton,
            )
        until(
            application,
            lambda: pulse_body._controller.current_path == pulse_path,
            timeout=15.0,
        )
        QtTest.QTest.mouseClick(
            pulse_body.schedule_view.fire_button,
            QtCore.Qt.LeftButton,
        )
        until(
            application,
            lambda: pulse_body.active_snapshot is not None
            and pulse_body.active_snapshot.state is RunState.RUNNING,
            timeout=15.0,
        )

        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        camera_row = console.logic_nodes[-1]
        camera_editor = console._logic_editors[id(camera_row)]
        camera_widgets = _visible_form_widgets(camera_editor)
        _choose_combo_data(camera_widgets["camera_role"], "camera", application)
        _replace_spin_value(camera_widgets["repeat"], "0")
        cards_before = tuple(console.cards)
        QtTest.QTest.mouseClick(camera_editor.form.start_button, QtCore.Qt.LeftButton)
        camera_signal = console_signal_key(camera_row.node.node_id, "frame_0")
        until(
            application,
            lambda: console._data.freeze().value(camera_signal) is not None,
            timeout=20.0,
        )
        first_camera = console._data.freeze().value(camera_signal)
        until(
            application,
            lambda: (
                (value := console._data.freeze().value(camera_signal)) is not None
                and value.snapshot.ref != first_camera.snapshot.ref
            ),
            timeout=10.0,
        )
        assert tuple(console.cards) == cards_before

        _choose_combo_text(
            console.kind_combo,
            "Processor: Judge occupancy",
            application,
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        occupancy_row = console.logic_nodes[-1]
        occupancy_editor = console._logic_editors[id(occupancy_row)]
        occupancy_widgets = _visible_form_widgets(occupancy_editor)
        assert set(occupancy_widgets) == {
            "model_kind",
            "camera_frame",
            "calibration_source",
            "calibration_output",
            "calibration_path",
        }
        assert _signal_leaf_keys(occupancy_widgets["camera_frame"]) == {
            camera_signal
        }
        assert _signal_leaf_keys(occupancy_widgets["calibration_output"]) == {
            calibration_signal
        }
        _choose_signal_leaf(
            occupancy_widgets["camera_frame"],
            camera_signal,
            application,
        )
        # Coupled Measurements above already exercise the Task-output branch.
        # Occupancy takes the other authoritative branch: one exact saved
        # calibration pointer, never a latest-directory lookup.
        calibration_pointer = (
            tmp_path / "calibration-output" / "calibration_ref.json"
        )
        assert calibration_pointer.is_file()
        _choose_combo_data(
            occupancy_widgets["calibration_source"],
            "saved",
            application,
        )
        _replace_path_value(
            occupancy_widgets["calibration_path"],
            str(calibration_pointer),
        )
        _choose_combo_text(
            occupancy_widgets["model_kind"],
            "box",
            application,
        )
        cards_before = tuple(console.cards)
        QtTest.QTest.mouseClick(
            occupancy_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        occupied_signal = console_signal_key(occupancy_row.node.node_id, "occupied")
        rate_signal = console_signal_key(occupancy_row.node.node_id, "rate")
        until(
            application,
            lambda: (
                console._data.freeze().value(occupied_signal) is not None
                and console._data.freeze().value(rate_signal) is not None
            ),
            timeout=20.0,
        )
        assert tuple(console.cards) == cards_before
        first_occupied = console._data.freeze().value(occupied_signal)
        first_rate = console._data.freeze().value(rate_signal)
        until(
            application,
            lambda: (
                (occupied := console._data.freeze().value(occupied_signal))
                is not None
                and occupied.snapshot.ref != first_occupied.snapshot.ref
                and (rate := console._data.freeze().value(rate_signal)) is not None
                and rate.snapshot.ref != first_rate.snapshot.ref
            ),
            timeout=10.0,
        )

        sites_card = _add_plot_and_bind(
            console,
            add,
            "sites",
            occupied_signal,
            application,
        )
        rate_card = _add_plot_and_bind(
            console,
            add,
            "monitor",
            rate_signal,
            application,
        )
        until(
            application,
            lambda: (
                sites_card.board is not None
                and sites_card.board.front_frame is not None
                and rate_card.board is not None
                and rate_card.board.front_frame is not None
            ),
            timeout=20.0,
        )
        first_sites_front = sites_card.board.front_frame
        first_rate_front = rate_card.board.front_frame
        until(
            application,
            lambda: (
                sites_card.board.front_frame is not None
                and sites_card.board.front_frame.sequence
                > first_sites_front.sequence
                and rate_card.board.front_frame is not None
                and rate_card.board.front_frame.sequence
                > first_rate_front.sequence
            ),
            timeout=10.0,
        )
        assert not sites_card.board.board.hasMouseTracking()
        assert not rate_card.board.board.hasMouseTracking()
    finally:
        if (
            pulse_body is not None
            and pulse_body.active_snapshot is not None
            and not pulse_body.active_snapshot.state.terminal
        ):
            QtTest.QTest.mouseClick(
                pulse_body.schedule_view.safe_button,
                QtCore.Qt.LeftButton,
            )
            until(
                application,
                lambda: pulse_body.active_snapshot is not None
                and pulse_body.active_snapshot.state.terminal,
                timeout=15.0,
            )
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.finish_close(application, timeout_seconds=15.0)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_grey_molasses_uses_virtual_rf_and_requires_manual_plot_binding(
    tmp_path,
) -> None:
    """The last current-visible Main definition runs through formal Qt controls."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser
    from zlc_neutral_atom.logic_nodes.release_recapture.timing import (
        TriggeredReleaseRecaptureResult,
    )

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "grey-molasses-user-flow",
            "--seed",
            "47",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        # Grey uses the exact artifact emitted by the ordinary Calibration Task;
        # no hidden session calibration is allowed.
        _choose_combo_text(console.kind_combo, "Task: Calibrate readout", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        calibration_row = console.logic_nodes[-1]
        calibration_editor = console._logic_editors[id(calibration_row)]
        calibration_widgets = _visible_form_widgets(calibration_editor)
        _replace_path_value(
            calibration_widgets["folder"],
            str(tmp_path / "calibration-output"),
        )
        _replace_spin_value(calibration_widgets["threshold_frames"], "10")
        QtTest.QTest.mouseClick(
            calibration_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        calibration_signal = console_signal_key(
            calibration_row.node.node_id,
            "calibration",
        )
        until(
            application,
            lambda: (
                _resolved_artifact(console, calibration_signal) is not None
                and not console._task_locked
            ),
            timeout=25.0,
        )

        _choose_combo_text(
            console.kind_combo,
            "Measurement: Grey molasses detuning",
            application,
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        grey_row = console.logic_nodes[-1]
        grey_editor = console._logic_editors[id(grey_row)]
        grey_widgets = _visible_form_widgets(grey_editor)
        assert set(grey_widgets) == {
            "pulse",
            "detuning",
            "t_off",
            "shots",
            "per_site",
            "rf_role",
            "calibration",
        }
        grey_spec = console._spec_for_logic(grey_row.node)
        assert grey_spec is not None
        detuning_decl = next(
            field for field in grey_spec.form.fields if field.key == "detuning"
        )
        assert (detuning_decl.label, detuning_decl.unit) == (
            "Two-photon detuning",
            "Γ",
        )
        assert _signal_leaf_keys(grey_widgets["calibration"]) == {
            calibration_signal
        }
        _replace_axis_range(grey_widgets["detuning"], "-0.2", "0.2", "3")
        _replace_spin_value(grey_widgets["shots"], "1")
        _choose_combo_text(grey_widgets["rf_role"], "rf", application)
        _choose_signal_leaf(
            grey_widgets["calibration"],
            calibration_signal,
            application,
        )

        cards_before = tuple(console.cards)
        QtTest.QTest.mouseClick(
            grey_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        grey_node = console._logic_nodes[id(grey_row)]
        recapture_signal = console_signal_key(grey_row.node.node_id, "recapture")
        until(
            application,
            lambda: (
                grey_node.final_result_resolved
                and console._data.freeze().value(recapture_signal) is not None
            ),
            timeout=25.0,
        )
        assert tuple(console.cards) == cards_before
        result = grey_node.final_result
        assert isinstance(result, TriggeredReleaseRecaptureResult)
        assert result.rf_terminal is not None
        assert result.rf_terminal.advanced_points == 3
        recapture = console._data.freeze().value(recapture_signal)
        axis = recapture.snapshot.block.schema.point_axes[0]
        assert (axis.name, axis.unit, axis.coordinates) == (
            "Two-photon detuning",
            "Γ",
            (-0.2, 0.0, 0.2),
        )

        card = _add_plot_and_bind(
            console,
            add,
            "1d",
            recapture_signal,
            application,
        )
        until(
            application,
            lambda: card.board is not None and card.board.front_frame is not None,
            timeout=15.0,
        )
        assert card.frozen_plot_panel_contract().value_label == "Recapture rate"
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.finish_close(application, timeout_seconds=15.0)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_running_camera_signal_drives_pulse_scan_without_stopping_camera(
    tmp_path,
) -> None:
    """PulseScan consumes a running Camera signal without owning its Run."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "pulse-scan-user-flow",
            "--seed",
            "41",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        # There is one Camera definition.  It is started first and remains the
        # sole owner of its live stream; PulseScan consumes frame_0 through the
        # producer-owned association capability and owns only the sequencer.
        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        camera_row = console.logic_nodes[-1]
        camera_editor = console._logic_editors[id(camera_row)]
        camera_widgets = _visible_form_widgets(camera_editor)
        _choose_combo_data(camera_widgets["camera_role"], "camera", application)
        _replace_spin_value(camera_widgets["repeat"], "0")
        QtTest.QTest.mouseClick(
            camera_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        until(
            application,
            lambda: (
                (node := console._logic_nodes.get(id(camera_row))) is not None
                and node.handle is not None
                and node.running
            ),
            timeout=15.0,
        )
        camera_node = console._logic_nodes[id(camera_row)]
        camera_handle = camera_node.handle
        camera_run_id = camera_handle.run_id
        camera_signal = console_signal_key(camera_row.node.node_id, "frame_0")

        _choose_combo_text(console.kind_combo, "Measurement: Pulse scan", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        scan_row = console.logic_nodes[-1]
        scan_editor = console._logic_editors[id(scan_row)]
        widgets = _visible_form_widgets(scan_editor)
        assert set(widgets) == {"pulse", "pulse_slots", "y_signal"}

        slots = widgets["pulse_slots"]
        assert slots.isVisible()
        assert slots._program_code.isVisible()
        assert "scan_table" in slots._program_code.toPlainText()
        assert slots._sweep_combo.currentText() in {
            "Scan slots (hardware table)",
            "API slots (one pulse per point)",
        }
        scan_spec = console._spec_for_logic(scan_row.node)
        assert scan_spec is not None
        y_parameter = next(
            parameter for parameter in scan_spec.input_fields
            if parameter.key == "y_signal"
        )
        assert y_parameter.label == "Signal (y)"
        assert _signal_leaf_keys(widgets["y_signal"]) == {camera_signal}
        _choose_signal_leaf(widgets["y_signal"], camera_signal, application)

        QtTest.QTest.mouseClick(scan_editor.form.start_button, QtCore.Qt.LeftButton)
        scan_signal = console_signal_key(scan_row.node.node_id, "scan")
        # A Measurement never manufactures a viewer, either at Start or when
        # its FINAL result arrives.  Plot ownership remains an explicit Monitor
        # action; only Tasks open their declared run-scoped panels.
        assert not any(card.config.signal == scan_signal for card in console.cards)
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            if console._data.freeze().value(scan_signal) is not None:
                break
            time.sleep(0.005)
        value = console._data.freeze().value(scan_signal)
        if value is None:
            raise AssertionError(scan_row.status_label.text())
        current_camera = console._logic_nodes.get(id(camera_row))
        assert current_camera is camera_node
        assert current_camera.handle is camera_handle
        assert current_camera.handle.run_id == camera_run_id
        assert current_camera.running
        assert camera_row.status_label.text() == "running"
        assert camera_row.stop_button.isEnabled()
        camera_value = console._data.freeze().value(camera_signal)
        assert camera_value is not None
        assert camera_value.run_id == camera_run_id.value
        assert not any(card.config.signal == scan_signal for card in console.cards)
        data_roles = {
            axis.role for axis in value.snapshot.block.schema.cell_schema.data_axes
        }
        from zlc_data import SPATIAL_X, SPATIAL_Y

        plot_kind = (
            "2d" if {SPATIAL_X, SPATIAL_Y}.issubset(data_roles) else "1d"
        )
        scan_card = _add_plot_and_bind(
            console,
            add,
            plot_kind,
            scan_signal,
            application,
        )
        until(
            application,
            lambda: scan_card.board is not None
            and scan_card.board.front_frame is not None,
            timeout=15.0,
        )
        assert value.snapshot.block.schema.repeat_axis.size >= 1
        assert value.snapshot.block.schema.point_layout.storage_size >= 1
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
