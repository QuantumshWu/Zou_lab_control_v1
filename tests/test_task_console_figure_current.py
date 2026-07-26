"""Current Figure/TaskConsole contracts closed by the migration.

These tests enter at the immutable signal boundary and otherwise exercise the
real TaskConsole Qt surfaces.  They neither launch hardware nor write into the
repository tree.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np

from gui_user_flow import close_task_console


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _image_value(
    *,
    revision: int,
    center_x: float = 31.0,
    side: int = 64,
    dtype=None,
):
    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        SPATIAL_X,
        SPATIAL_Y,
        AxisId,
        AxisSpec,
        BlockId,
        DatasetComponentValidity,
        CoordinateFrameId,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointLayout,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )
    from zlc_neutral_atom.processing.signal_plane import SignalValue

    repeat = AxisSpec(AxisId("current.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId("current.point"), "point", SCAN_POINT, 1, (0,))
    frame = CoordinateFrameId("current.camera")
    side = int(side)
    if side < 2:
        raise ValueError("image side must be at least two")
    y_axis = AxisSpec(
        AxisId("current.y"),
        "camera y",
        SPATIAL_Y,
        side,
        tuple(float(index) for index in range(side)),
        "pixel",
        frame,
    )
    x_axis = AxisSpec(
        AxisId("current.x"),
        "camera x",
        SPATIAL_X,
        side,
        tuple(float(index) for index in range(side)),
        "pixel",
        frame,
    )
    yy, xx = np.meshgrid(
        np.arange(side, dtype=np.float64),
        np.arange(side, dtype=np.float64),
        indexing="ij",
    )
    image = 3.0 + 120.0 * np.exp(
        -((xx - float(center_x)) ** 2 + (yy - 29.0) ** 2) / 72.0
    )
    values = image[None, None, :, :]
    if dtype is not None:
        values = values.astype(dtype)
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            values.dtype,
            value_unit="count",
        ),
    )
    block = DataBlock(
        BlockId("current-image"),
        DatasetRevision(int(revision)),
        values,
        DatasetComponentValidity(
            (y_axis.axis_id, x_axis.axis_id),
            np.ones(values.shape, dtype=np.bool_),
        ),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("current-image-generation")),
        block,
    )
    return SignalValue(
        name="image",
        source_instance_id="current-immutable-boundary",
        snapshot=snapshot,
        coverage=None,
        run_id=f"current-run-{int(revision)}",
        epoch_id=f"current-epoch-{int(revision)}",
        join_digest="0" * 64,
    )


def _wait(application, predicate, *, timeout: float = 15.0) -> None:
    from PyQt5 import QtCore

    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _present_value(console, card, value, *, frame_key) -> None:
    from zlc_neutral_atom.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )
    from zlc_neutral_atom.runtime.dataset import MonitorCoverage

    state = getattr(console, "_current_test_source", None)
    if state is None:
        declaration = DatasetOutputDeclaration(
            "image",
            "tests.current-image",
        )
        state = SimpleNamespace(current=None)
        node = SimpleNamespace(
            instance_id="current-image",
            dataset_output_declarations=(declaration,),
            signal_key=lambda name: str(name),
        )
        slot = SimpleNamespace(
            freeze_live_outputs=lambda: state.current,
            close=lambda: None,
            notification_failure=None,
        )
        state.declaration = declaration
        state.node = node
        state.slot = slot
        console._data.attach(node, slot)
        console._current_test_source = state
    state.current = (
        value.run_id,
        value.epoch_id,
        {
            "image": LiveDatasetOutput(
                state.declaration,
                value.snapshot,
                MonitorCoverage(1, 1, 0, False),
                value.join_digest,
            )
        },
    )
    console._data.mark_changed(state.node)
    front = console._data.freeze()
    console._promote_data_front(front)
    value = front.value("image")
    assert value is not None
    request = card._freeze_value_render_request(
        value,
        frame_key,
        force=True,
    )
    assert request is not None
    console._render_lane.enqueue((request,))


def test_cross_publishes_selected_native_data_with_scalar_shape() -> None:
    from zlc_frontend import FigureSource, ImageDisplayState
    from zlc_frontend.figure_outputs import (
        CROSS_DATA_OUTPUT,
        bind_cross_data_commit,
        materialize_cross_outputs,
    )
    from zlc_frontend.figure import ViewIntent, suggest_view
    from zlc_frontend.panel_render import PanelComposer, PanelProvenance
    from zlc_frontend.shape_text import describe_dataset_shape

    value = _image_value(revision=1, dtype=np.uint8)
    suggestion = suggest_view(value.schema, ViewIntent.IMAGE)
    assert suggestion.spec is not None
    composer = PanelComposer(
        "cross-data",
        intent=ViewIntent.IMAGE,
        view=suggestion.spec,
    )
    try:
        frame, figure = composer.compose_with_figure(
            value.snapshot,
            display=ImageDisplayState(),
            provenance=PanelProvenance("run", "epoch", "0" * 64),
        )
        panel = frame.panels[0]
        commit = bind_cross_data_commit(
            panel.source_identity,
            (31.0, 29.0),
            figure,
            panel.display_payload,
        )
        outputs = materialize_cross_outputs(FigureSource(value.snapshot), commit)
    finally:
        composer.close()

    assert set(outputs) == {CROSS_DATA_OUTPUT}
    snapshot = outputs[CROSS_DATA_OUTPUT].snapshot
    assert snapshot.block.values.shape == (1, 1, 1)
    assert snapshot.block.values.dtype == np.dtype(np.uint8)
    assert snapshot.block.values[0, 0, 0] == value.values[0, 0, 29, 31]
    assert snapshot.block.schema.cell_schema.data_shape == (1,)
    assert describe_dataset_shape(snapshot.block.schema, snapshot.block.values) == (
        "1 × 1 × (1)"
    )


def test_auto_view_is_explicit_and_slider_edits_persist_the_typed_spec() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_frontend.figure import AxisViewRole, FixedIndex, dataset_axes
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="1d", title="Lineout", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        value = _image_value(revision=1)
        _present_value(console, card, value, frame_key=("image", 1))
        _wait(application, lambda: card.frozen_render_payload() is not None)

        schema = card.frozen_render_value().schema
        view = card._effective_view_spec(schema)
        assert view is not None
        assert set(card.view_spec_editor._rows) == {
            axis.axis_id for axis in dataset_axes(schema)
        }
        slider = next(
            binding
            for binding in view.axis_bindings
            if binding.role is AxisViewRole.SLIDER
        )
        _row, combo = card.view_spec_editor._rows[slider.axis_id]
        assert combo.isEnabled()
        combo.setCurrentIndex(5)
        _wait(
            application,
            lambda: isinstance(
                card._saved_view_spec(schema).binding(slider.axis_id).selector,
                FixedIndex,
            )
            and card._saved_view_spec(schema).binding(slider.axis_id).selector.index
            == 5,
        )

        QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
        _wait(application, lambda: id(card) in console._panel_editors)
        editor = console._panel_editors[id(card)]
        assert set(editor.view_spec_editor._rows) == set(card.view_spec_editor._rows)
        assert (
            editor.view_spec_editor._view.binding(slider.axis_id).selector.index
            == 5
        )
    finally:
        close_task_console(application, console)


def test_viewport_renders_reuse_one_prepared_plane_for_the_same_revision() -> None:
    from zlc_frontend import ImageDisplayState
    from zlc_frontend.figure import ViewIntent, suggest_view
    from zlc_frontend.panel_render import PanelComposer, PanelProvenance

    first = _image_value(revision=1, dtype=np.uint8)
    second = _image_value(revision=2, center_x=38.0, dtype=np.uint8)
    suggestion = suggest_view(first.schema, ViewIntent.IMAGE)
    assert suggestion.spec is not None
    composer = PanelComposer(
        "current-performance",
        intent=ViewIntent.IMAGE,
        view=suggestion.spec,
    )
    provenance = PanelProvenance("current-run", "current-epoch", "0" * 64)
    try:
        composer.compose_with_figure(
            first.snapshot,
            display=ImageDisplayState(),
            provenance=provenance,
        )
        renderer = composer._renderer
        prepared = renderer._prepared_image_value
        first_key = renderer._prepared_image_key
        assert first_key.evaluated_input.ref == first.snapshot.ref
        assert prepared is not None
        assert first.snapshot.block.values.dtype == np.dtype(np.uint8)
        assert prepared[0].dtype == np.dtype(np.uint8)
        assert np.shares_memory(prepared[0], first.snapshot.block.values)
        assert prepared[1].dtype == np.dtype(bool)
        assert np.shares_memory(
            prepared[1],
            first.snapshot.block.validity.mask,
        )

        composer.compose_with_figure(
            first.snapshot,
            display=ImageDisplayState(
                revision=1,
                x_view=(8.0, 54.0),
                y_view=(7.0, 53.0),
            ),
            provenance=provenance,
        )
        assert renderer._prepared_image_value is prepared
        assert renderer._prepared_image_key == first_key

        composer.compose_with_figure(
            second.snapshot,
            display=ImageDisplayState(),
            provenance=provenance,
        )
        assert renderer._prepared_image_key.evaluated_input.ref == second.snapshot.ref
        assert renderer._prepared_image_value is not prepared
        assert second.snapshot.block.values.dtype == np.dtype(np.uint8)
    finally:
        composer.close()


def test_task_console_dpr_change_retires_every_old_panel_front() -> None:
    from zlc_frontend.plot_layout import panel_surface_geometry
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="2d", title="Camera", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        _present_value(
            console,
            card,
            _image_value(revision=1),
            frame_key=("image", 1),
        )
        _wait(
            application,
            lambda: console._render_lane.idle
            and card.board is not None
            and card.board.front_frame is not None,
        )
        previous = card.board.front_frame
        ratio = card.raster_pixel_ratio + 0.5

        console._apply_raster_pixel_ratio(ratio)
        assert card.board.front_frame is None
        _wait(
            application,
            lambda: console._render_lane.idle
            and card.board.front_frame is not None,
        )

        current = card.board.front_frame
        assert current is not previous
        geometry = panel_surface_geometry(card.config.size, pixel_ratio=ratio)
        raster = current.panels[0].raster
        assert (raster.width, raster.height) == geometry.raster_size
        assert (card.board.width(), card.board.height()) == geometry.logical_size
    finally:
        console.close()
        application.processEvents()


def test_fit_button_presents_overlay_and_publishes_readable_figure_signals() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="2d", title="Camera", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        first = _image_value(revision=1)
        second = _image_value(revision=2, center_x=39.0)
        _present_value(console, card, first, frame_key=("image", 1))
        _wait(
            application,
            lambda: card.frozen_render_payload() is not None,
        )

        QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
        _wait(application, lambda: id(card) in console._panel_editors)
        editor = console._panel_editors[id(card)]
        fit_pane = editor._fit_pane
        assert fit_pane is not None
        _wait(
            application,
            lambda: bool(fit_pane.fit_models)
            and fit_pane.fit_button.isEnabled(),
        )
        assert (
            fit_pane.model_combo.currentData()
            == "radial_gaussian_center"
        )

        # The live monitor advances, but the visible Edit input does not.  Fit
        # must consume the surface that issued the command, not whatever the
        # monitor happens to show when the worker starts.
        _present_value(console, card, second, frame_key=("image", 2))
        _wait(
            application,
            lambda: card.frozen_render_value().snapshot.ref == second.snapshot.ref,
        )
        assert editor._snapshot_value.snapshot.ref == first.snapshot.ref
        QtTest.QTest.mouseClick(
            fit_pane.fit_button,
            QtCore.Qt.LeftButton,
        )
        _wait(
            application,
            lambda: card._fit_result is not None
            and card._fit_result.source_ref == first.snapshot.ref,
        )
        _wait(
            application,
            lambda: getattr(
                editor._board.front_frame.panels[0].display_payload,
                "fit_overlay",
                None,
            )
            is not None,
        )

        assert card.frozen_render_payload().fit_overlay is None
        edit_overlay = (
            editor._board.front_frame.panels[0].display_payload.fit_overlay
        )
        assert edit_overlay.source_ref == first.snapshot.ref
        assert edit_overlay.result_identity == card._fit_result_identity
        _wait(
            application,
            lambda: card.panel_id in console._passive_publisher_rows
            and "center_x"
            in console._passive_publisher_rows[
                card.panel_id
            ].publishes_label.text(),
        )
        publisher_row = console._passive_publisher_rows[card.panel_id]
        inventory = publisher_row.publishes_label.text()
        assert "center_y" in inventory
        assert "one_over_e_radius" in inventory
        assert "1 × 1 × (1)" in inventory
        assert "fit.center_x" in publisher_row.publishes_label.toolTip()
        fit_keys = tuple(
            key for key in console._tick_data.names() if "/fit." in key
        )
        assert fit_keys
        assert {
            console._tick_data.value(key).run_id for key in fit_keys
        } == {first.run_id}

        # A command from the live Setting surface atomically replaces the one
        # committed Fit.  Its result appears only on the matching live source,
        # and the old Edit snapshot no longer borrows the overlay.
        console.tabs.setCurrentIndex(0)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        _wait(
            application,
            lambda: bool(card.fit_authoring_pane.fit_models)
            and card.fit_authoring_pane.fit_button.isEnabled(),
        )
        QtTest.QTest.mouseClick(
            card.fit_authoring_pane.fit_button,
            QtCore.Qt.LeftButton,
        )
        _wait(
            application,
            lambda: card._fit_result is not None
            and card._fit_result.source_ref == second.snapshot.ref
            and card.frozen_render_payload().fit_overlay is not None,
        )
        _wait(
            application,
            lambda: (
                editor._board.front_frame.panels[0]
                .display_payload.fit_overlay
                is None
            ),
        )
        assert {
            console._tick_data.value(key).run_id
            for key in console._tick_data.names()
            if "/fit." in key
        } == {second.run_id}
    finally:
        close_task_console(application, console)


def test_edit_view_recomposes_the_frozen_input_until_explicit_refresh() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="2d", title="Camera", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        first = _image_value(revision=1, center_x=24.0)
        second = _image_value(revision=2, center_x=40.0)
        _present_value(console, card, first, frame_key=("image", 1))
        _wait(
            application,
            lambda: card.frozen_render_payload() is not None,
        )

        QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
        _wait(application, lambda: id(card) in console._panel_editors)
        editor = console._panel_editors[id(card)]
        assert editor._snapshot_value.snapshot.ref == first.snapshot.ref

        _present_value(console, card, second, frame_key=("image", 2))
        _wait(
            application,
            lambda: card.frozen_render_value().snapshot.ref == second.snapshot.ref,
        )
        assert editor._snapshot_value.snapshot.ref == first.snapshot.ref

        assert card._set_param("view_xlim", (12.0, 48.0))
        editor._request_snapshot_render()
        _wait(
            application,
            lambda: editor._snapshot_display is not None
            and editor._snapshot_display.revision
            == card._display_state(card.frozen_plot_panel_contract()).revision,
        )
        edit_payload = editor._board.front_frame.panels[0].display_payload
        live_payload = card.frozen_render_payload()
        assert edit_payload.evaluated_input.ref == first.snapshot.ref
        assert live_payload.evaluated_input.ref == second.snapshot.ref

        QtTest.QTest.mouseClick(editor.refresh_button, QtCore.Qt.LeftButton)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert editor._snapshot_value.snapshot.ref == second.snapshot.ref
        assert (
            editor._board.front_frame.panels[0].display_payload.evaluated_input.ref
            == second.snapshot.ref
        )
    finally:
        close_task_console(application, console)
