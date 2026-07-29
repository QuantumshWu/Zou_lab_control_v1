"""Formal operator flow: DeviceManager -> Camera -> signal -> live 2-D panel.

The fixture chooses only the offscreen Qt platform.  Every product transition is
driven through the same visible controls as the desktop launcher; assertions may
inspect the resulting typed fronts, but never bypass a button to create them.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from zlc_data import (
    CellValidity,
    MONITOR_HISTORY,
    READOUT_EVENT,
    REPEAT,
    SPATIAL_X,
    SPATIAL_Y,
)
from gui_user_flow import (
    capture_offscreen_window,
    choose_combo_data as _choose_combo_data,
    choose_combo_text as _choose_combo_text,
    click_tab,
    configure_offscreen_fast_path,
    current_logic_editor as _current_logic_editor,
    drag_mouse_move,
    replace_path_value as _replace_path_value,
    replace_spin_value as _replace_spin_value,
    require_offscreen_platform,
    until,
    visible_form_widgets as _visible_form_widgets,
    widget_gone,
)
from zlc_workbench.task_console.console_records import (
    console_signal_key,
    panel_signal_key,
)
from zlc_frontend import PlotKind
from zlc_frontend.qt_widgets import ensure_qt_app


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


def _replace_axis_range(widget, minimum: str, maximum: str, points: str) -> None:
    """Edit the three visible controls of one swept axis as an operator would."""

    _replace_spin_value(widget.min_spin, minimum)
    _replace_spin_value(widget.max_spin, maximum)
    _replace_spin_value(widget.pts_spin, points)


def _resolved_artifact(console, output_key: str):
    """Read one retained FINAL Artifact through the typed producer boundary."""

    producer = console.resolve_console_producer(output_key)
    return producer.artifact if producer.artifact_resolved else None


def _current_panel_editor(console, application):
    """Return the Panel Edit page reached through the visible Edit button."""

    from zlc_workbench.task_console.panel_editor import PanelEditor

    until(
        application,
        lambda: isinstance(console.tabs.currentWidget(), PanelEditor),
    )
    editor = console.tabs.currentWidget()
    assert editor.isVisible() and not editor.isWindow()
    return editor


def _render_value_or_none(card):
    """Read the public immutable value currently presented by one panel."""

    try:
        return card.frozen_render_value()
    except RuntimeError:
        return None


def _dataset_producer_or_none(console, signal_key: str):
    """Resolve a running typed producer once its public binding is ready."""

    try:
        producer = console.resolve_console_producer(signal_key)
    except RuntimeError:
        return None
    return producer if producer.running and producer.output_binding is not None else None


def _add_plot_and_bind(console, add_button, kind: str, signal: str, application):
    """Add one blank plot and wire its Setting popup through visible controls."""

    before = len(console.cards)
    _choose_combo_data(console.kind_combo, PlotKind(kind), application)
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
        AxisSourceRef,
        AxisSpec,
        BlockId,
        CellValidity,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        GridTopology,
        OwnedSnapshot,
        PointColumn,
        PointTable,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )
    from zlc_frontend.figure import (
        AxisViewRole,
        FixedIndex,
        ViewIntent,
        grid_facet_source,
        view_spec_from_tree,
    )
    from zlc_frontend.qt_widgets import (
        FitAuthoringPane,
        FluentParameterForm,
        FluentSectionLabel,
        FluentSettingRow,
    )
    from zlc_neutral_atom.catalog import DefinitionKey
    from zlc_neutral_atom.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )
    from zlc_neutral_atom.runtime.dataset import MonitorCoverage

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
        PointTable(
            scan_x.size * scan_y.size,
            (
                PointColumn(
                    scan_x.axis_id,
                    scan_x.name,
                    scan_x.role,
                    PointColumn.NUMERIC,
                    tuple(
                        value
                        for value in scan_x.coordinates
                        for _ in scan_y.coordinates
                    ),
                    scan_x.unit,
                    scan_x.coordinate_frame,
                ),
                PointColumn(
                    scan_y.axis_id,
                    scan_y.name,
                    scan_y.role,
                    PointColumn.NUMERIC,
                    tuple(scan_y.coordinates) * scan_x.size,
                    scan_y.unit,
                    scan_y.coordinate_frame,
                ),
            ),
        ),
        GridTopology(
            (scan_x.axis_id, scan_y.axis_id),
            (scan_x.coordinates, scan_y.coordinates),
            tuple(
                (x_index, y_index)
                for x_index in range(scan_x.size)
                for y_index in range(scan_y.size)
            ),
        ),
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
    console = None
    grid_node = None
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
        declaration = DatasetOutputDeclaration("grid", "test.formal-grid")
        grid_node = SimpleNamespace(
            instance_id="formal-immutable-monitor-boundary",
            definition_key=DefinitionKey("tests", "formal-grid"),
            dataset_output_declarations=(declaration,),
            signal_key=lambda _name: "formal-grid",
        )
        live_output = LiveDatasetOutput(
            declaration,
            snapshot,
            MonitorCoverage(12, 12, 0, False),
            "0" * 64,
        )
        slot = SimpleNamespace(
            freeze_live_outputs=lambda: (
                "formal-run",
                "formal-epoch",
                {"grid": live_output},
            ),
            close=lambda: None,
            notification_failure=None,
        )
        plane = console._data
        plane.reserve(grid_node)
        lifecycle = plane.begin_run_lifecycle(grid_node)
        plane.bind_run_lifecycle(lifecycle, "formal-run", preemptible=True)
        plane.attach(grid_node, slot)
        plane.mark_changed(grid_node, slot)
        plane.freeze()
        grid_publication = plane.latest_publication("formal-grid")
        assert grid_publication is not None
        grid_value = grid_publication.value("formal-grid")
        assert grid_value is not None
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
        _choose_combo_data(console.kind_combo, PlotKind.GRID, application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        card = console.cards[-1]
        assert card.config.kind == PlotKind.GRID
        click_tab(console, console.tabs.widget(0))

        # A producer fixture may enter only at the immutable monitor boundary.
        # This schema deliberately has multiple legal Grid views, so it must
        # remain unresolved until the operator chooses a cell intent and facet.
        card.config.signal = grid_value.name
        request = card._freeze_value_render_request(
            grid_value,
            1,
            force=True,
            publication=grid_publication,
        )
        assert request is None
        assert card._current_schema() == schema, card.status.text()

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
        assert card.view_spec_editor.isVisible()
        popup_width = card.settings_popup.width()
        assert popup_width == card.settings_popup.minimumWidth()
        assert popup_width == card.settings_popup.maximumWidth()
        assert (
            card._settings_scroll.horizontalScrollBarPolicy()
            == QtCore.Qt.ScrollBarAlwaysOff
        )

        # The frontend form is the one keyed authoring owner for both Setting
        # and Edit.  Qt presents typed SourceRefs and commits the exact view the
        # operator chooses; Workbench has no parallel Grid vocabulary.
        form = card.view_spec_editor.findChild(FluentParameterForm)
        assert form is not None
        assert form.keys == ("grid.intent",)
        _choose_combo_data(
            form.widget_for("grid.intent"),
            ViewIntent.IMAGE,
            application,
        )
        until(application, lambda: "grid.facet" in form.keys)
        facet_combo = form.widget_for("grid.facet")
        offered_sources = tuple(
            facet_combo.itemData(index)
            for index in range(facet_combo.count())
        )
        assert all(
            isinstance(source, AxisSourceRef) for source in offered_sources
        )
        point_rows = AxisSourceRef.point_rows()
        repeat_source = AxisSourceRef.tensor(repeat.axis_id)
        point_x = AxisSourceRef.point_coordinate(scan_x.axis_id)
        point_y = AxisSourceRef.point_coordinate(scan_y.axis_id)
        grid_x = AxisSourceRef.grid_dimension(scan_x.axis_id)
        grid_y = AxisSourceRef.grid_dimension(scan_y.axis_id)
        tensor_y = AxisSourceRef.tensor(image_y.axis_id)
        tensor_x = AxisSourceRef.tensor(image_x.axis_id)
        assert offered_sources == (
            repeat_source,
            point_rows,
            point_x,
            point_y,
            grid_x,
            grid_y,
            tensor_y,
            tensor_x,
        )
        _choose_combo_data(facet_combo, grid_x, application)
        until(
            application,
            lambda: "view_spec" in card.config.params,
        )
        authored = view_spec_from_tree(card.config.params["view_spec"])
        assert authored.intent is ViewIntent.IMAGE
        assert grid_facet_source(authored) == grid_x
        assert authored.binding(grid_y).role is AxisViewRole.SELECTED
        assert authored.binding(grid_y).selector == FixedIndex(0)
        assert sum(
            binding.role is AxisViewRole.FACET
            for binding in authored.source_bindings
        ) == 1
        request = card._freeze_value_render_request(
            grid_value,
            2,
            force=True,
            publication=grid_publication,
        )
        assert request is not None
        assert request.contract.figure.view == authored
        assert authored.binding(
            AxisSourceRef.tensor(image_x.axis_id)
        ).role is AxisViewRole.IMAGE_X
        assert authored.binding(
            AxisSourceRef.tensor(image_y.axis_id)
        ).role is AxisViewRole.IMAGE_Y

        # A legal cell-kind change retains the typed facet and reconciles only
        # the canonical frontend display form for the new intent.
        _choose_combo_data(
            form.widget_for("grid.intent"),
            ViewIntent.HISTOGRAM,
            application,
        )
        until(
            application,
            lambda: (
                "view_spec" in card.config.params
                and view_spec_from_tree(card.config.params["view_spec"]).intent
                is ViewIntent.HISTOGRAM
            ),
        )
        authored = view_spec_from_tree(card.config.params["view_spec"])
        assert grid_facet_source(authored) == grid_x
        label_width = card.setting_label_width(card.fontMetrics())
        setting_rows = card.settings_popup.findChildren(FluentSettingRow)
        assert any(row._label.text() == "Bins" for row in setting_rows)
        assert any(row._label.text() == "Log count" for row in setting_rows)

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
                lambda: console.tabs.currentWidget() is not None,
            )
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        finally:
            application.removeEventFilter(probe)
        assert probe.shown == []
        assert visible_top_levels() == top_levels_before

        editor = _current_panel_editor(console, application)
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
        edit_view_form = editor.view_spec_editor.findChild(FluentParameterForm)
        assert edit_view_form is not None
        assert edit_view_form.keys == form.keys
        assert (
            editor.display_form_surface._zlc_display_signature
            == card.display_form_surface._zlc_display_signature
        )
        assert set(editor.display_form_surface._zlc_display_widgets) == set(
            card.display_form_surface._zlc_display_widgets
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
        if console is not None and grid_node is not None:
            console._data.retire(grid_node)
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
        editor = _current_logic_editor(console, application)

        # Virtual MOT camera is a true free-running source, so this is the
        # deterministic live Camera role for an operator-path acceptance run.
        widgets = _visible_form_widgets(editor)
        role_combo = widgets["camera_role"]
        assert tuple(
            role_combo.itemData(index) for index in range(role_combo.count())
        ) == ("camera", "mot_camera")
        assert role_combo.currentData() == "camera"
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
            lambda: row.status_label.text() == "running"
            and row.stop_button.isEnabled(),
            timeout=15.0,
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

        _choose_combo_data(console.kind_combo, PlotKind.IMAGE, application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        assert len(console.cards) == 1
        card = console.cards[0]
        click_tab(console, console.tabs.widget(0))
        surface = None
        first_front = second_front = None
        seen_schemas = []
        # All declared frames use one stable PlotPanel surface.  Switching a
        # binding changes only the typed source; every selected output must then
        # continue advancing on its own live Camera revisions.
        for selected_signal in (
            frame_signals[0],
            frame_signals[2],
            frame_signals[1],
        ):
            if not card.settings_popup.isVisible():
                QtTest.QTest.mouseClick(
                    card.setting_button,
                    QtCore.Qt.LeftButton,
                )
                until(application, lambda: card.settings_popup.isVisible())
            _choose_signal_leaf(card.signal_combo, selected_signal, application)
            assert card.config.signal == selected_signal
            until(
                application,
                lambda: (
                    card.board is not None
                    and card.board.front_frame is not None
                    and (presented := _render_value_or_none(card)) is not None
                    and presented.name == selected_signal
                ),
                timeout=15.0,
            )
            if surface is None:
                surface = card.board.board
            else:
                assert card.board.board is surface
            selected_front = card.board.front_frame
            selected_value = card.frozen_render_value()
            selected_ref = selected_value.snapshot.ref
            frame_schema = selected_value.snapshot.block.schema
            seen_schemas.append(frame_schema)
            assert frame_schema.repeat_axis.role == REPEAT
            assert frame_schema.repeat_axis.size == 1
            assert frame_schema.point_table.row_count == 1
            assert frame_schema.point_table.columns == ()
            assert frame_schema.grid_topology is None
            assert tuple(
                axis.role for axis in frame_schema.cell_schema.data_axes
            ) == (SPATIAL_Y, SPATIAL_X)
            assert selected_value.shape == (
                1,
                1,
                *frame_schema.cell_schema.data_shape,
            )
            until(
                application,
                lambda: (
                    (presented := _render_value_or_none(card)) is not None
                    and presented.name == selected_signal
                    and presented.snapshot.ref != selected_ref
                    and card.board.front_frame is not None
                    and card.board.front_frame.sequence > selected_front.sequence
                ),
                timeout=10.0,
            )
            if selected_signal == signal:
                first_front = selected_front
                second_front = card.board.front_frame

        assert first_front is not None and second_front is not None
        assert len(set(seen_schemas)) == 1
        assert not any(
            axis.role in (MONITOR_HISTORY, READOUT_EVENT)
            for axis in (
                seen_schemas[0].repeat_axis,
                *seen_schemas[0].point_table.columns,
                *seen_schemas[0].cell_schema.data_axes,
            )
        )
        assert bytes(second_front.panels[0].raster.pixels) != bytes(
            first_front.panels[0].raster.pixels
        )

        # Exercise Fit through the actual DeviceManager -> Camera Measurement
        # -> live 2-D Plot Panel controls.  Fit is a live surface-local operation:
        # the Camera base keeps advancing, the Qt surface paints the canonical
        # overlay, and the data plane publishes only exact-parent parameters.
        from zlc_frontend.render_style import FIT_RADIAL_COLOR

        if not card.settings_popup.isVisible():
            QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        until(application, lambda: card.settings_popup.isVisible())
        fit_pane = card.fit_authoring_pane
        until(
            application,
            lambda: bool(fit_pane.fit_models)
            and fit_pane.fit_button.isEnabled(),
            timeout=15.0,
        )
        assert fit_pane.model_combo.currentData() == "radial_gaussian_center"
        fitted_ref = card.frozen_render_value().snapshot.ref

        def exact_fit_pixels() -> int:
            image = card.board.grab().toImage().convertToFormat(
                QtGui.QImage.Format_RGBA8888
            )
            bits = image.bits()
            bits.setsize(image.byteCount())
            rows = np.frombuffer(bits, dtype=np.uint8).reshape(
                image.height(),
                image.bytesPerLine(),
            )
            rgba = rows[:, : image.width() * 4].reshape(
                image.height(),
                image.width(),
                4,
            )
            color = QtGui.QColor(FIT_RADIAL_COLOR)
            expected = np.asarray(
                (color.red(), color.green(), color.blue()),
                dtype=np.uint8,
            )
            return int(np.all(rgba[..., :3] == expected, axis=-1).sum())

        before_fit_pixels = exact_fit_pixels()
        QtTest.QTest.mouseClick(fit_pane.fit_button, QtCore.Qt.LeftButton)
        center_x_signal = panel_signal_key(card.panel_id, "fit.center_x")
        until(
            application,
            lambda: (
                "converged" in card.status.text()
                and center_x_signal in _signal_leaf_keys(card.signal_combo)
                and exact_fit_pixels() > before_fit_pixels
            ),
            timeout=20.0,
        )
        until(
            application,
            lambda: card.frozen_render_value().snapshot.ref != fitted_ref,
            timeout=10.0,
        )
        assert "no retained producer transaction" not in card.status.text()
        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-live-fit.png",
            settle_ms=50,
        )
        QtTest.QTest.mouseClick(fit_pane.clear_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: (
                card.frozen_render_value().snapshot.ref != fitted_ref
                and center_x_signal not in _signal_leaf_keys(card.signal_combo)
                and card.status.text() == "Fit cleared"
            ),
            timeout=15.0,
        )
        if card.settings_popup.isVisible():
            QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
            until(application, lambda: not card.settings_popup.isVisible())

        # No-button movement is inert: the board does not even request tracking.
        board = card.board.board
        assert not board.hasMouseTracking()
        before = card.frozen_figure_output_state()[2:4]
        QtTest.QTest.mouseMove(board, board.rect().center())
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert card.frozen_figure_output_state()[2:4] == before

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
            lambda: card.frozen_figure_output_state()[3] is not None,
        )

        from zlc_frontend.figure_outputs import AREA_DATA_OUTPUT

        # Consume one real Area selection through a second 2-D panel.  The
        # derived publication must carry the exact painted publication as its
        # sole parent, so ROI pixels and source metadata cannot diverge.
        area_start = plot_point(0.18, 0.20)
        area_end = plot_point(0.48, 0.58)
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=area_start,
        )
        area_pressed_publication = card.frozen_render_publication()
        assert area_pressed_publication.value(signal) is card.frozen_render_value()
        drag_mouse_move(board, area_end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.LeftButton,
            pos=area_end,
        )
        until(
            application,
            lambda: card.frozen_figure_output_state()[2] is not None,
        )
        area_signal = panel_signal_key(card.panel_id, AREA_DATA_OUTPUT)
        assert "no retained producer transaction" not in card.status.text()
        # A hidden picker is deliberately parked.  Open the destination
        # panel's real Setting surface: its canonical provider must already
        # contain the Area key and let the operator bind it directly.
        area_card = _add_plot_and_bind(
            console,
            add,
            "2d",
            area_signal,
            application,
        )
        assert area_signal in _signal_leaf_keys(area_card.signal_combo)
        until(
            application,
            lambda: _render_value_or_none(area_card) is not None,
            timeout=15.0,
        )
        area_publication = area_card.frozen_render_publication()
        assert len(area_publication.parents) == 1
        area_parent = area_publication.parents[0]
        assert area_parent.owner_id == area_pressed_publication.owner_id
        assert area_parent.generation == area_pressed_publication.generation
        assert area_parent.sequence >= area_pressed_publication.sequence
        area_value = area_card.frozen_render_value()
        assert area_publication.value(area_signal) is area_value
        area_source = area_parent.value(signal)
        assert area_source is not None
        assert area_value.snapshot.ref.revision == area_source.snapshot.ref.revision
        if not area_card.settings_popup.isVisible():
            QtTest.QTest.mouseClick(
                area_card.setting_button,
                QtCore.Qt.LeftButton,
            )
        until(application, lambda: area_card.settings_popup.isVisible())
        assert area_card.view_spec_editor.isVisible()

        first_roi_schema = area_card.frozen_render_value().schema.fingerprint
        area_start = plot_point(0.20, 0.24)
        area_end = plot_point(0.62, 0.72)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=area_start)
        drag_mouse_move(board, area_end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=area_end)
        until(
            application,
            lambda: _render_value_or_none(area_card) is not None
            and area_card.frozen_render_value().schema.fingerprint
            != first_roi_schema,
            timeout=15.0,
        )
        assert not area_card._status_error
        assert area_card.view_spec_editor.isVisible()
        QtTest.QTest.mouseClick(
            area_card.setting_button,
            QtCore.Qt.LeftButton,
        )
        until(application, lambda: not area_card.settings_popup.isVisible())

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
        from zlc_frontend.qt_widgets import FitAuthoringPane
        from zlc_workbench.task_console.panel_editor import PanelEditor

        until(
            application,
            lambda: (
                isinstance(console.tabs.currentWidget(), PanelEditor)
                and console.tabs.currentWidget().findChild(FitAuthoringPane)
                is not None
            ),
        )
        edit = console.tabs.currentWidget()
        assert edit.window() is console.window()
        assert edit._board.isVisible(), (
            console.tabs.currentWidget() is edit,
            edit.isVisible(),
            edit._board.isHidden(),
            console.tabs.currentIndex(),
            console.tabs.indexOf(edit),
        )
        edit_request = edit.freeze_current_view_request()
        assert edit_request is not None
        edit_ref = edit_request.value.snapshot.ref
        edit_fit = edit.findChild(FitAuthoringPane)
        assert edit_fit is not None
        assert edit_fit is not card.fit_authoring_pane

        # Edit Fit belongs to this frozen surface only.  It may neither clear nor
        # publish the live panel's parameter route, and opening Edit must not stop
        # the live Camera front from advancing.
        fit_request_spy = QtTest.QSignalSpy(card.fit_requested)
        fit_clear_spy = QtTest.QSignalSpy(card.fit_output_clear_requested)
        live_ref = card.frozen_render_value().snapshot.ref
        until(application, lambda: edit_fit.fit_button.isEnabled(), timeout=15.0)
        QtTest.QTest.mouseClick(edit_fit.fit_button, QtCore.Qt.LeftButton)
        assert not edit_fit.fit_button.isEnabled()
        assert len(fit_request_spy) == 1
        assert fit_request_spy[0][0] is edit_fit
        assert fit_request_spy[0][0] is not fit_pane
        assert len(fit_clear_spy) == 0
        until(application, lambda: edit_fit.fit_button.isEnabled(), timeout=20.0)
        until(
            application,
            lambda: card.frozen_render_value().snapshot.ref != live_ref,
            timeout=10.0,
        )
        assert edit.freeze_current_view_request().value.snapshot.ref == edit_ref
        assert center_x_signal not in _signal_leaf_keys(card.signal_combo)
        assert len(fit_clear_spy) == 0
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


def test_finite_camera_progress_and_final_reach_one_formal_plot_panel(
    tmp_path,
) -> None:
    """Finite Camera remains atomic from visible Start through PlotPanel FINAL."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "finite-camera-user-flow",
            "--seed",
            "47",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console = None
    console_wrapper = None
    pulse_worker = None
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
        from zlc_pulse import PulseExecutionForm, load_pulse_document

        pulse_document = load_pulse_document(
            Path("pulses/imaging_template.json").resolve()
        )
        pulse_request = flow.experiment.pulse.request(
            pulse_document,
            PulseExecutionForm.STATIC_ONCE,
            api_values={
                parameter.parameter_id: pulse_document.field_value(
                    parameter.field
                )[0]
                for parameter in pulse_document.api_parameters
            },
        )

        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        row = console.logic_nodes[-1]
        editor = _current_logic_editor(console, application)
        widgets = _visible_form_widgets(editor)
        _choose_combo_data(widgets["camera_role"], "camera", application)
        _replace_spin_value(widgets["repeat"], "3")
        _replace_spin_value(widgets["frames_per_cycle"], "3")
        qualified = tuple(
            console_signal_key(row.node.node_id, f"frame_{index}")
            for index in range(3)
        )
        _choose_combo_data(console.kind_combo, PlotKind.IMAGE, application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        card = console.cards[-1]
        click_tab(console, console.tabs.widget(0))
        assert _render_value_or_none(card) is None
        QtTest.QTest.mouseClick(editor.form.start_button, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        until(application, lambda: card.settings_popup.isVisible())
        until(
            application,
            lambda: qualified[1] in _signal_leaf_keys(card.signal_combo),
            timeout=5.0,
        )
        _choose_signal_leaf(card.signal_combo, qualified[1], application)
        assert card.config.signal == qualified[1]

        pulse_errors = []

        def fire_cycles() -> None:
            try:
                for cycle in range(3):
                    flow.experiment.pulse.run(pulse_request)
                    if cycle < 2:
                        # These are three independent external experiments, not
                        # one software-timed physical pulse.  Leave the operator
                        # surface time to present each completed Camera cycle.
                        time.sleep(0.35)
            except BaseException as error:
                pulse_errors.append(error)

        pulse_worker = threading.Thread(target=fire_cycles)
        pulse_worker.start()

        observed = {}
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            try:
                publication = card.frozen_render_publication()
            except RuntimeError:
                publication = None
            if publication is not None:
                observed.setdefault(publication.sequence, publication)
            if (
                row.status_label.text().startswith("done")
                and publication is not None
                and not publication.value(qualified[1]).transient
            ):
                break
            time.sleep(0.005)

        assert row.status_label.text().startswith("done"), row.status_label.text()
        pulse_worker.join()
        assert not pulse_errors
        publications = tuple(observed[index] for index in sorted(observed))
        transient = tuple(
            publication
            for publication in publications
            if publication.value(qualified[1]).transient
        )
        terminal = tuple(
            publication
            for publication in publications
            if not publication.value(qualified[1]).transient
        )
        assert len(transient) >= 2
        assert terminal

        previous_written = 0
        previous_revision = 0
        expected_shape = None
        expected_dtype = None
        for publication in transient:
            assert tuple(publication.signals) == qualified
            values = tuple(publication.value(name) for name in qualified)
            assert all(value is not None for value in values)
            coverages = tuple(value.coverage for value in values)
            assert all(coverage == coverages[0] for coverage in coverages)
            coverage = coverages[0]
            assert coverage is not None
            assert coverage.total_cells == 3
            assert previous_written < coverage.written_cells <= 3
            previous_written = coverage.written_cells
            for value in values:
                shape = value.shape
                if expected_shape is None:
                    expected_shape = shape
                    expected_dtype = value.dtype
                assert shape == expected_shape
                assert shape[:2] == (3, 1)
                assert value.dtype == expected_dtype == np.dtype("uint16")
                assert value.schema.point_table.columns == ()
                assert value.schema.grid_topology is None
                assert value.snapshot.ref.revision.value > previous_revision
                validity = value.snapshot.block.validity
                assert isinstance(validity, CellValidity)
                assert validity.mask[:, 0].tolist() == (
                    [True] * coverage.written_cells
                    + [False] * (3 - coverage.written_cells)
                )
            previous_revision = values[0].snapshot.ref.revision.value

        final_publication = terminal[-1]
        assert tuple(final_publication.signals) == qualified
        assert final_publication.sequence > transient[-1].sequence
        for name in qualified:
            value = final_publication.value(name)
            assert value is not None and not value.transient
            assert value.coverage is None
            assert value.shape == expected_shape
            assert value.dtype == expected_dtype
            validity = value.snapshot.block.validity
            assert isinstance(validity, CellValidity)
            assert validity.mask[:, 0].tolist() == [True, True, True]
        assert card.frozen_render_publication() is final_publication
        assert not card._status_error
    finally:
        if pulse_worker is not None and pulse_worker.is_alive():
            pulse_worker.join(timeout=15.0)
        if console is not None:
            for row in reversed(console.logic_nodes):
                if not row.stop_button.isEnabled():
                    continue
                QtTest.QTest.mouseClick(row.stop_button, QtCore.Qt.LeftButton)
                until(
                    application,
                    lambda current=row: not current.stop_button.isEnabled(),
                    timeout=15.0,
                )
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.finish_close(application, timeout_seconds=15.0)
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
        calibration_editor = _current_logic_editor(console, application)
        calibration_widgets = _visible_form_widgets(calibration_editor)
        assert set(calibration_widgets) == {
            "folder",
            "pulse",
            "threshold_method",
            "reference_exposure_s",
            "readout_exposure_s",
            "threshold_frames",
            "roi_radius",
            "camera_role",
        }
        calibration_output = tmp_path / "calibration-output"
        _replace_path_value(
            calibration_widgets["folder"],
            str(calibration_output),
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
                and card.config.kind == PlotKind.IMAGE
                for card in console.cards
            )
            if calibration_row.status_label.text().startswith(
                "calibration artifact committed"
            ):
                break
            time.sleep(0.005)
        assert saw_calibration_panel
        assert calibration_row.status_label.text().startswith(
            "calibration artifact committed"
        )
        assert str(calibration_output) in calibration_row.status_label.text()
        site_map_signal = console_signal_key(
            calibration_row.node.node_id,
            "site_map",
        )
        until(
            application,
            lambda: any(
                card.config.signal == site_map_signal
                and card.config.kind == PlotKind.SITE_MAP
                and card.board is not None
                and card.board.front_frame is not None
                for card in console.cards
            ),
            timeout=15.0,
        )
        site_map_card = next(
            card
            for card in console.cards
            if card.config.signal == site_map_signal
            and card.config.kind == PlotKind.SITE_MAP
        )

        # The visible task panel itself is the public presentation boundary.
        from zlc_frontend.panel_size import DEFAULT_PANEL_SIZE
        from zlc_frontend.plot_layout import (
            PANEL_EXPORT_PIXEL_RATIO,
            panel_surface_geometry,
        )

        contract = site_map_card.frozen_plot_panel_contract()
        site_map_value = site_map_card.frozen_render_value()
        site_map_publication = site_map_card.frozen_render_publication()
        assert site_map_publication.value(site_map_signal) is site_map_value
        assert contract.figure.kind.value == "sites"
        assert contract.size_name == DEFAULT_PANEL_SIZE
        assert site_map_card.board.front_frame.panels[0].raster is not None

        # The saved report uses denser export pixels for the same ordinary
        # logical 2x2 surface.  It does not obtain resolution by authoring a
        # larger panel.
        overview_png = (calibration_output / "report" / "overview.png").read_bytes()
        assert overview_png.startswith(b"\x89PNG\r\n\x1a\n")
        import struct

        overview_pixels = struct.unpack(">II", overview_png[16:24])
        export_geometry = panel_surface_geometry(
            DEFAULT_PANEL_SIZE,
            pixel_ratio=PANEL_EXPORT_PIXEL_RATIO,
        )
        assert overview_pixels == export_geometry.raster_size
        assert export_geometry.logical_size == contract.logical_size

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
        assert "site fidelity" in calibration_row.publishes_label.text()
        assert all(
            output in calibration_row.publishes_label.text()
            for output in (
                "site threshold",
                "site centres",
                "aggregate fidelity",
                "global fidelity",
            )
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
            lambda: _render_value_or_none(fidelity_card) is not None,
            timeout=15.0,
        )
        fidelity_value = fidelity_card.frozen_render_value()
        site_axis = fidelity_value.schema.cell_schema.data_axes[0]
        assert site_axis.role.value == "site"
        assert fidelity_value.schema.physical_shape == (1, 1, site_axis.size)
        assert f"1 × 1 × ({site_axis.size})" in calibration_row.publishes_label.text()
        until(
            application,
            lambda: (
                fidelity_card.board is not None
                and fidelity_card.board.front_frame is not None
            ),
            timeout=15.0,
        )
        assert (
            fidelity_card.frozen_plot_panel_contract().figure.value_label
            == "Readout fidelity"
        )

        _choose_combo_text(console.kind_combo, "Task: Optimize MOT field", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        mot_row = console.logic_nodes[-1]
        mot_editor = _current_logic_editor(console, application)
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
        mot_final = console_signal_key(mot_row.node.node_id, "mot_field")
        saw_mot_panel = False
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            saw_mot_panel = saw_mot_panel or any(
                card.config.signal == mot_grid
                and card.config.kind == PlotKind.GRID
                for card in console.cards
            )
            if mot_row.status_label.text().startswith("done"):
                break
            time.sleep(0.005)
        assert saw_mot_panel
        assert mot_row.status_label.text().startswith("done")
        until(
            application,
            lambda: any(
                card.config.signal == mot_final
                and card.config.kind == PlotKind.GRID
                and card.board is not None
                and (
                    card.board.front_frame is not None
                    or card.board.showing_overview
                )
                for card in console.cards
            ),
            timeout=15.0,
        )
        mot_card = next(
            card for card in console.cards if card.config.signal == mot_final
        )
        mot_value = mot_card.frozen_render_value()
        mot_schema = mot_value.schema
        assert tuple(column.role.value for column in mot_schema.point_table.columns) == (
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
            mot_schema.point_table.row_count,
            1,
        )
        mot_shape = f"1 × {mot_schema.point_table.row_count} × (1)"
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
        camera_editor = _current_logic_editor(console, application)
        camera_widgets = _visible_form_widgets(camera_editor)
        _choose_combo_data(camera_widgets["camera_role"], "camera", application)
        _replace_spin_value(camera_widgets["repeat"], "0")
        QtTest.QTest.mouseClick(
            camera_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        until(
            application,
            lambda: camera_row.status_label.text() == "running",
            timeout=15.0,
        )

        _choose_combo_text(console.kind_combo, "Task: Calibrate readout", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        calibration_row = console.logic_nodes[-1]
        calibration_editor = _current_logic_editor(console, application)
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
                and calibration_row.status_label.text().startswith(
                    "calibration artifact committed"
                )
            ),
            timeout=25.0,
        )
        assert not camera_row.stop_button.isEnabled()
        assert camera_row.status_label.text() == "stopped"
        assert _resolved_artifact(console, calibration_signal) is not None
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
    console = None
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
        calibration_editor = _current_logic_editor(console, application)
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
                and calibration_row.status_label.text().startswith(
                    "calibration artifact committed"
                )
            ),
            timeout=25.0,
        )

        # Temperature takes only the explicit calibration Artifact in addition
        # to Main's visible physics parameters.  A Measurement never auto-opens
        # a panel; the operator creates and wires the 1-D view afterwards.
        _choose_combo_text(console.kind_combo, "Measurement: Temperature", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        temperature_row = console.logic_nodes[-1]
        temperature_editor = _current_logic_editor(console, application)
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
        # Dataset R is the authored shot sweep; the release pulse template has
        # no competing RepeatRegion authority.
        _replace_spin_value(temperature_widgets["shots"], "2")
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
            lambda: temperature_row.status_label.text().startswith("done"),
            timeout=25.0,
        )
        assert tuple(console.cards) == cards_before
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
        temperature_value = temperature_card.frozen_render_value()
        temperature_axis = temperature_value.snapshot.block.schema.point_table.columns[0]
        assert (temperature_axis.name, temperature_axis.unit) == (
            "Trap-off time",
            "s",
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
        fidelity_editor = _current_logic_editor(console, application)
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
            lambda: fidelity_row.status_label.text().startswith("done"),
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
        pulse_path = (Path("pulses/probe_template.json")).resolve()
        pulse_body = flow.pulse
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
        from zlc_neutral_atom.runtime.run import RunState

        until(
            application,
            lambda: pulse_body.active_snapshot is not None
            and pulse_body.active_snapshot.state is RunState.RUNNING,
            timeout=15.0,
        )

        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        camera_row = console.logic_nodes[-1]
        camera_editor = _current_logic_editor(console, application)
        camera_widgets = _visible_form_widgets(camera_editor)
        _choose_combo_data(camera_widgets["camera_role"], "camera", application)
        _replace_spin_value(camera_widgets["repeat"], "0")
        cards_before = tuple(console.cards)
        QtTest.QTest.mouseClick(camera_editor.form.start_button, QtCore.Qt.LeftButton)
        camera_signal = console_signal_key(camera_row.node.node_id, "frame_0")
        until(
            application,
            lambda: camera_row.status_label.text() == "running",
            timeout=20.0,
        )
        assert tuple(console.cards) == cards_before
        camera_card = _add_plot_and_bind(
            console,
            add,
            "2d",
            camera_signal,
            application,
        )
        until(
            application,
            lambda: camera_card.board is not None
            and camera_card.board.front_frame is not None,
            timeout=15.0,
        )
        first_camera = camera_card.frozen_render_value()
        until(
            application,
            lambda: (
                (value := _render_value_or_none(camera_card)) is not None
                and value.snapshot.ref != first_camera.snapshot.ref
            ),
            timeout=10.0,
        )

        _choose_combo_text(
            console.kind_combo,
            "Processor: Judge occupancy",
            application,
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        occupancy_row = console.logic_nodes[-1]
        occupancy_editor = _current_logic_editor(console, application)
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
            lambda: occupancy_row.status_label.text() == "running",
            timeout=20.0,
        )
        assert tuple(console.cards) == cards_before

        sites_card = _add_plot_and_bind(
            console,
            add,
            "sites",
            occupied_signal,
            application,
        )
        until(
            application,
            lambda: (
                sites_card.board is not None
                and sites_card.board.front_frame is not None
            ),
            timeout=20.0,
        )
        assert rate_signal in _signal_leaf_keys(sites_card.signal_combo)
        first_sites_front = sites_card.board.front_frame
        until(
            application,
            lambda: (
                sites_card.board.front_frame is not None
                and sites_card.board.front_frame.sequence
                > first_sites_front.sequence
            ),
            timeout=10.0,
        )
        assert not sites_card.board.board.hasMouseTracking()
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
        if console is not None:
            for row in reversed(console.logic_nodes):
                if not row.stop_button.isEnabled():
                    continue
                QtTest.QTest.mouseClick(
                    row.stop_button,
                    QtCore.Qt.LeftButton,
                )
                until(
                    application,
                    lambda current=row: not current.stop_button.isEnabled(),
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
        calibration_editor = _current_logic_editor(console, application)
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
                and calibration_row.status_label.text().startswith(
                    "calibration artifact committed"
                )
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
        grey_editor = _current_logic_editor(console, application)
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
        grey_spec = grey_editor.form.current_spec()
        detuning_decl = next(
            field for field in grey_spec.editor_fields if field.key == "detuning"
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
        recapture_signal = console_signal_key(grey_row.node.node_id, "recapture")
        until(
            application,
            lambda: grey_row.status_label.text().startswith("done"),
            timeout=25.0,
        )
        assert tuple(console.cards) == cards_before

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
        recapture = card.frozen_render_value()
        axis = recapture.snapshot.block.schema.point_table.columns[0]
        assert (axis.name, axis.unit, axis.values) == (
            "Two-photon detuning",
            "Γ",
            (-0.2, 0.0, 0.2),
        )
        assert (
            card.frozen_plot_panel_contract().figure.value_label
            == "Recapture rate"
        )
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
    console = None
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
        camera_editor = _current_logic_editor(console, application)
        camera_widgets = _visible_form_widgets(camera_editor)
        _choose_combo_data(camera_widgets["camera_role"], "camera", application)
        _replace_spin_value(camera_widgets["repeat"], "0")
        QtTest.QTest.mouseClick(
            camera_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        until(
            application,
            lambda: camera_row.status_label.text() == "running",
            timeout=15.0,
        )
        camera_signal = console_signal_key(camera_row.node.node_id, "frame_0")
        until(
            application,
            lambda: _dataset_producer_or_none(console, camera_signal) is not None,
            timeout=15.0,
        )
        camera_producer = _dataset_producer_or_none(console, camera_signal)
        assert camera_producer is not None
        assert camera_producer.running
        camera_binding_identity = camera_producer.output_binding.identity

        _choose_combo_text(console.kind_combo, "Measurement: Pulse scan", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        scan_row = console.logic_nodes[-1]
        scan_editor = _current_logic_editor(console, application)
        widgets = _visible_form_widgets(scan_editor)
        assert set(widgets) == {
            "pulse",
            "scan_sweep_count",
            "pulse_slots",
            "y_signal",
        }
        _replace_spin_value(widgets["scan_sweep_count"], "2")

        slots = widgets["pulse_slots"]
        assert slots.isVisible()
        assert slots._program_code.isVisible()
        program_source = slots._program_code.toPlainText()
        assert "scan_table" in program_source and "N = 21" in program_source
        slots._program_code.setPlainText(
            program_source.replace("N = 21", "N = 2")
        )
        assert slots._sweep_combo.currentText() in {
            "Scan slots (hardware table)",
            "API slots (one pulse per point)",
        }
        scan_spec = scan_editor.form.current_spec()
        y_parameter = next(
            parameter for parameter in scan_spec.editor_fields
            if parameter.key == "y_signal"
        )
        assert y_parameter.label == "Signal (y)"
        assert _signal_leaf_keys(widgets["y_signal"]) == {camera_signal}
        _choose_signal_leaf(widgets["y_signal"], camera_signal, application)
        assert "—" in camera_row.publishes_label.text()

        QtTest.QTest.mouseClick(scan_editor.form.start_button, QtCore.Qt.LeftButton)
        scan_signal = console_signal_key(scan_row.node.node_id, "scan")
        # A Measurement never manufactures a viewer, either at Start or when
        # its FINAL result arrives.  Plot ownership remains an explicit Monitor
        # action; only Tasks open their declared run-scoped panels.
        assert not any(card.config.signal == scan_signal for card in console.cards)
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            status = scan_row.status_label.text()
            if status.startswith("done") or status.startswith(("failed", "rejected")):
                break
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            time.sleep(0.005)
        assert scan_row.status_label.text().startswith("done"), (
            scan_row.status_label.text()
        )
        current_camera = console.resolve_console_producer(camera_signal)
        assert current_camera.running
        assert current_camera.output_binding.identity == camera_binding_identity
        assert camera_row.status_label.text() == "running"
        assert camera_row.stop_button.isEnabled()
        assert not any(card.config.signal == scan_signal for card in console.cards)
        scan_card = _add_plot_and_bind(
            console,
            add,
            "grid",
            scan_signal,
            application,
        )
        from zlc_data import AxisSourceRef
        from zlc_frontend.figure import ViewIntent
        from zlc_frontend.qt_widgets import FluentParameterForm

        if not scan_card.settings_popup.isVisible():
            QtTest.QTest.mouseClick(
                scan_card.setting_button,
                QtCore.Qt.LeftButton,
            )
        until(application, lambda: scan_card.settings_popup.isVisible())
        until(
            application,
            lambda: scan_card.view_spec_editor.isVisible()
            or not scan_card.status.text().startswith("waiting for "),
        )
        assert scan_card.view_spec_editor.isVisible(), scan_card.status.text()
        view_form = scan_card.view_spec_editor.findChild(FluentParameterForm)
        assert view_form is not None
        _choose_combo_data(
            view_form.widget_for("grid.intent"),
            ViewIntent.IMAGE,
            application,
        )
        until(application, lambda: "grid.facet" in view_form.keys)
        _choose_combo_data(
            view_form.widget_for("grid.facet"),
            AxisSourceRef.point_rows(),
            application,
        )
        deadline = time.monotonic() + 15.0
        while (
            scan_card.board is None
            or not (
                scan_card.board.front_frame is not None
                or scan_card.board.showing_overview
            )
        ) and time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            time.sleep(0.005)
        assert scan_card.board is not None, scan_card.status.text()
        assert (
            scan_card.board.front_frame is not None
            or scan_card.board.showing_overview
        ), scan_card.status.text()
        value = scan_card.frozen_render_value()
        assert value.snapshot.block.schema.repeat_axis.size == 2
        assert value.snapshot.block.schema.point_table.row_count >= 1
    finally:
        if console is not None:
            for row in reversed(console.logic_nodes):
                if not row.stop_button.isEnabled():
                    continue
                QtTest.QTest.mouseClick(row.stop_button, QtCore.Qt.LeftButton)
                until(
                    application,
                    lambda current=row: not current.stop_button.isEnabled(),
                    timeout=15.0,
                )
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
