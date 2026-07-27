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


def _large_grid_value(*, revision: int):
    """Small pixels but enough facets to require the 4x4 Grid surface."""

    from zlc_data import (
        REPEAT,
        SPATIAL_X,
        SPATIAL_Y,
        AxisId,
        AxisSpec,
        BlockId,
        CoordinateFrameId,
        DataBlock,
        DatasetComponentValidity,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointLayout,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )
    from zlc_neutral_atom.processing.signal_plane import SignalValue

    repeat = AxisSpec(
        AxisId("current.grid.repeat"),
        "repeat",
        REPEAT,
        36,
        tuple(range(36)),
    )
    frame = CoordinateFrameId("current.grid.camera")
    y_axis = AxisSpec(
        AxisId("current.grid.y"),
        "camera y",
        SPATIAL_Y,
        3,
        (0.0, 1.0, 2.0),
        "pixel",
        frame,
    )
    x_axis = AxisSpec(
        AxisId("current.grid.x"),
        "camera x",
        SPATIAL_X,
        3,
        (0.0, 1.0, 2.0),
        "pixel",
        frame,
    )
    values = np.arange(36 * 3 * 3, dtype=np.float64).reshape(36, 1, 3, 3)
    schema = DatasetSchema(
        repeat,
        (),
        PointLayout.rect_c(()),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            values.dtype,
            value_unit="count",
        ),
    )
    block = DataBlock(
        BlockId("current-grid"),
        DatasetRevision(int(revision)),
        values,
        DatasetComponentValidity(
            (y_axis.axis_id, x_axis.axis_id),
            np.ones(values.shape, dtype=np.bool_),
        ),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("current-grid-generation")),
        block,
    )
    return SignalValue(
        name="image",
        source_instance_id="current-grid-boundary",
        snapshot=snapshot,
        coverage=None,
        run_id=f"current-grid-run-{int(revision)}",
        epoch_id="current-grid-epoch",
        join_digest="9" * 64,
    )


def _curve_value(*, revision: int, center: float):
    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        BlockId,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointLayout,
        StreamGenerationId,
        VALID,
        ValueSchema,
    )
    from zlc_neutral_atom.processing.signal_plane import SignalValue

    x = np.linspace(-5.0, 5.0, 161)
    repeat = AxisSpec(AxisId("current.curve.repeat"), "repeat", REPEAT, 1, (0,))
    scan = AxisSpec(
        AxisId("current.curve.scan"),
        "detuning",
        SCAN_POINT,
        len(x),
        tuple(float(value) for value in x),
        "MHz",
    )
    values = (
        2.0 + 18.0 * np.exp(-((x - float(center)) ** 2) / (2.0 * 0.8**2))
    )[None, :, None]
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((len(x),)),
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )
    block = DataBlock(
        BlockId("current-curve"),
        DatasetRevision(int(revision)),
        values,
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("current-curve-generation")),
        block,
    )
    return SignalValue(
        name="image",
        source_instance_id="current-curve-boundary",
        snapshot=snapshot,
        coverage=None,
        run_id=f"current-curve-run-{int(revision)}",
        epoch_id=f"current-curve-epoch-{int(revision)}",
        join_digest="7" * 64,
    )


def _histogram_value(*, revision: int):
    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        BlockId,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointLayout,
        StreamGenerationId,
        VALID,
        ValueSchema,
    )
    from zlc_neutral_atom.processing.signal_plane import SignalValue

    rng = np.random.default_rng(481516)
    samples = np.concatenate(
        (rng.normal(-2.0, 0.45, 360), rng.normal(2.1, 0.65, 440))
    )
    repeat = AxisSpec(
        AxisId("current.hist.repeat"),
        "repeat",
        REPEAT,
        samples.size,
    )
    point = AxisSpec(
        AxisId("current.hist.point"),
        "point",
        SCAN_POINT,
        1,
        (0,),
    )
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )
    block = DataBlock(
        BlockId("current-histogram"),
        DatasetRevision(int(revision)),
        samples[:, None, None],
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("current-histogram-generation")),
        block,
    )
    return SignalValue(
        name="image",
        source_instance_id="current-histogram-boundary",
        snapshot=snapshot,
        coverage=None,
        run_id=f"current-histogram-run-{int(revision)}",
        epoch_id=f"current-histogram-epoch-{int(revision)}",
        join_digest="6" * 64,
    )


def _wait(application, predicate, *, timeout: float = 15.0) -> None:
    from PyQt5 import QtCore

    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _present_value(console, card, value, *, frame_key) -> bool:
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
    schema = value.snapshot.block.schema
    total_cells = (
        schema.repeat_axis.size * schema.point_layout.storage_size
    )
    state.current = (
        value.run_id,
        value.epoch_id,
        {
            "image": LiveDatasetOutput(
                state.declaration,
                value.snapshot,
                MonitorCoverage(total_cells, total_cells, 0, False),
                value.join_digest,
            )
        },
    )
    console._data.mark_changed(state.node)
    front = console._data.freeze()
    console._promote_data_front(front)
    value = front.value("image")
    assert value is not None
    request = card.freeze_render_request(
        front,
        frame_key,
        force=True,
        source_component=console._data.capture_source_component(value),
    )
    if request is not None:
        console._render_lane.enqueue((request,))
    return request is not None


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
        axes_by_id = {axis.axis_id: axis for axis in dataset_axes(schema)}
        editable_axis_ids = {
            binding.axis_id
            for binding in view.axis_bindings
            if (
                binding.role in (AxisViewRole.SLIDER, AxisViewRole.SELECTED)
                and isinstance(binding.selector, FixedIndex)
                and axes_by_id[binding.axis_id].size > 1
            )
        }
        assert set(card.view_spec_editor._rows) == editable_axis_ids
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


def test_grid_initial_optimal_size_is_fresh_only_and_commits_with_its_raster() -> None:
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    value = _large_grid_value(revision=1)

    fresh = TaskConsole(state=TaskConsoleState(), window_px=(1000, 800))
    try:
        fresh.show()
        application.processEvents()
        grid_index = fresh.kind_combo.findData("grid")
        assert grid_index >= 0
        fresh.kind_combo.setCurrentIndex(grid_index)
        fresh._add_panel()
        card = fresh.cards[-1]
        assert card.config.size == "2x2"
        assert card._initial_grid_size_pending
        card.config.signal = value.name
        assert _present_value(fresh, card, value, frame_key=("grid", 1))
        # Worker submission alone cannot resize the card or mutate persisted
        # config.  The matching 4x4 raster is the commit receipt.
        assert card.config.size == "2x2"
        _wait(
            application,
            lambda: card._presented_contract is not None
            and card._presented_contract.size_name == "4x4",
        )
        assert card.config.size == "4x4"
        assert not card._initial_grid_size_pending
        assert card.size_combo.currentData() == "4x4"
        assert (
            card.board.width(),
            card.board.height(),
        ) == card._presented_contract.logical_size
    finally:
        close_task_console(application, fresh)

    loaded = TaskConsole(
        state=TaskConsoleState(
            panels=(
                PanelConfig(
                    kind="grid",
                    title="Loaded grid",
                    size="1x2",
                    signal="image",
                ),
            ),
        ),
        window_px=(1000, 800),
    )
    try:
        loaded.show()
        application.processEvents()
        card = loaded.cards[0]
        assert not card._initial_grid_size_pending
        assert _present_value(loaded, card, value, frame_key=("loaded-grid", 1))
        _wait(application, lambda: card._presented_contract is not None)
        assert card._presented_contract.size_name == "1x2"
        assert card.config.size == "1x2"
    finally:
        close_task_console(application, loaded)

    manual = TaskConsole(state=TaskConsoleState(), window_px=(1000, 800))
    try:
        manual.show()
        application.processEvents()
        grid_index = manual.kind_combo.findData("grid")
        assert grid_index >= 0
        manual.kind_combo.setCurrentIndex(grid_index)
        manual._add_panel()
        card = manual.cards[-1]
        assert card._initial_grid_size_pending
        card._on_size("2x4")
        assert not card._initial_grid_size_pending
        card.config.signal = value.name
        assert _present_value(manual, card, value, frame_key=("manual-grid", 1))
        _wait(application, lambda: card._presented_contract is not None)
        assert card._presented_contract.size_name == "2x4"
        assert card.config.size == "2x4"
    finally:
        close_task_console(application, manual)


def test_explicit_stock_grid_size_supersedes_in_flight_recommendation(
    monkeypatch,
) -> None:
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(state=TaskConsoleState(), window_px=(1000, 800))
    try:
        console.show()
        application.processEvents()
        grid_index = console.kind_combo.findData("grid")
        assert grid_index >= 0
        console.kind_combo.setCurrentIndex(grid_index)
        console._add_panel()
        card = console.cards[-1]
        card.config.signal = "image"

        captured = []
        lane_type = type(console._render_lane)

        def capture(_lane, requests) -> None:
            captured.extend(requests)

        with monkeypatch.context() as patch:
            patch.setattr(lane_type, "enqueue", capture)
            assert _present_value(
                console,
                card,
                _large_grid_value(revision=1),
                frame_key=("grid-size-race", 1),
            )
            recommendation = captured[-1]
            assert recommendation.contract.size_name == "4x4"

            captured.clear()
            card._on_size("2x2")
            assert not card._initial_grid_size_pending
            assert card.config.size == "2x2"
            assert captured
            explicit = captured[-1]
            assert explicit.contract.size_name == "2x2"
            assert explicit.source_key != recommendation.source_key
            assert not card.accept_render_result(
                recommendation,
                frame=object(),
                figure=object(),
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


def test_live_fit_keeps_its_exact_figure_and_paints_the_frontend_orange_ring() -> None:
    """A fast camera cannot outrun the Figure revision submitted to Fit."""

    from matplotlib.colors import to_rgba
    from PyQt5 import QtCore, QtTest

    from zlc_data import FitBatchStatus
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_frontend.render_style import FIT_RADIAL_COLOR
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
        fitted = _image_value(revision=1, center_x=27.0)
        newer = _image_value(revision=2, center_x=39.0)
        newest = _image_value(revision=3, center_x=45.0)
        assert _present_value(console, card, fitted, frame_key=("image", 1))
        _wait(
            application,
            lambda: card.frozen_render_payload() is not None,
        )
        before = card.board.front_frame.panels[0].raster

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        pane = card.fit_authoring_pane
        _wait(
            application,
            lambda: bool(pane.fit_models) and pane.fit_button.isEnabled(),
        )
        assert pane.model_combo.currentData() == "radial_gaussian_center"
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)

        # Both frames become real data-plane fronts while the solver/render
        # workers run.  Neither may replace the exact Figure being fitted.
        assert not _present_value(
            console,
            card,
            newer,
            frame_key=("image", 2),
        )
        assert not _present_value(
            console,
            card,
            newest,
            frame_key=("image", 3),
        )
        _wait(
            application,
            lambda: card._fit_result is not None
            and card._fit_result.source_ref == fitted.snapshot.ref
            and card.frozen_render_payload() is not None
            and card.frozen_render_payload().fit_overlay is not None,
        )

        assert card.frozen_render_value().snapshot.ref == fitted.snapshot.ref
        overlay = card.frozen_render_payload().fit_overlay
        assert overlay.status is FitBatchStatus.CONVERGED
        assert overlay.center_xy is not None
        assert overlay.one_over_e_radius is not None
        assert overlay.result_identity == card._fit_result_identity

        # Validate the actual immutable pixels shown by the formal Figure
        # surface, not merely the presence of an overlay DTO.
        after = card.board.front_frame.panels[0].raster
        before_rgba = np.frombuffer(before.pixels, dtype=np.uint8).reshape(
            before.height,
            before.width,
            4,
        )
        after_rgba = np.frombuffer(after.pixels, dtype=np.uint8).reshape(
            after.height,
            after.width,
            4,
        )
        expected_rgb = np.rint(
            np.asarray(to_rgba(FIT_RADIAL_COLOR)[:3]) * 255.0
        ).astype(np.uint8)
        before_orange = np.all(before_rgba[..., :3] == expected_rgb, axis=-1)
        after_orange = np.all(after_rgba[..., :3] == expected_rgb, axis=-1)
        assert int(after_orange.sum()) > int(before_orange.sum())

        parameter_names = {
            definition.name
            for definition in card._fit_result.parameter_definitions
        }
        _wait(
            application,
            lambda: {
                name.rsplit("/fit.", 1)[1]
                for name in console._tick_data.names()
                if "/fit." in name
            }
            == parameter_names,
        )
        # Predictions/overlays are presentation only; the public signal plane
        # contains model parameters and no fitted curve/image dataset.
        assert all(
            not name.endswith(("fit.curve", "fit.overlay", "fit.prediction"))
            for name in console._tick_data.names()
        )

        QtTest.QTest.mouseClick(pane.clear_button, QtCore.Qt.LeftButton)
        _wait(
            application,
            lambda: card.frozen_render_value().snapshot.ref == newest.snapshot.ref
            and card.frozen_render_payload().fit_overlay is None,
        )
        _wait(
            application,
            lambda: not any("/fit." in name for name in console._tick_data.names()),
        )
    finally:
        close_task_console(application, console)


def test_successful_fit_retires_with_its_producer_and_replacement_renders() -> None:
    """A producer generation owns its Fit pin, overlay and parameter routes."""

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
        old_value = _image_value(revision=1, center_x=27.0)
        replacement_value = _image_value(revision=2, center_x=42.0)
        assert _present_value(console, card, old_value, frame_key=("retire-fit", 1))
        _wait(application, lambda: card.frozen_render_payload() is not None)

        QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
        _wait(application, lambda: id(card) in console._panel_editors)
        editor = console._panel_editors[id(card)]
        _wait(
            application,
            lambda: editor._board is not None
            and editor._board.front_frame is not None,
        )
        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        pane = card.fit_authoring_pane
        _wait(application, lambda: bool(pane.fit_models) and pane.fit_button.isEnabled())
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _wait(
            application,
            lambda: card._fit_result is not None
            and card.frozen_render_payload().fit_overlay is not None
            and editor._board.front_frame.panels[0].display_payload.fit_overlay
            is not None
            and any("/fit." in name for name in console._tick_data.names()),
        )

        retired = console._current_test_source
        console._retire_logic_node_publications(retired.node)
        assert card._fit_active_spec is None
        assert card._fit_active_source is None
        assert card._fit_live_surface_source is None
        assert card._fit_result is None
        assert card._fit_result_identity is None
        assert pane._busy_kind == "prepare"
        assert card.board.front_frame is None
        assert not any("/fit." in name for name in console._tick_data.names())
        _wait(
            application,
            lambda: editor._board.front_frame.panels[0].display_payload.fit_overlay
            is None,
        )

        # Reuse the user-visible route with a new producer object.  No manual
        # Clear is needed and no old overlay/pin can intercept the new front.
        console._current_test_source = None
        assert _present_value(
            console,
            card,
            replacement_value,
            frame_key=("retire-fit", 2),
        )
        _wait(
            application,
            lambda: card._presented_value is not None
            and card._presented_value.run_id == replacement_value.run_id
            and card.frozen_render_payload().fit_overlay is None,
        )
        assert card._fit_live_surface_source is None
        assert not any("/fit." in name for name in console._tick_data.names())
    finally:
        close_task_console(application, console)


def test_inflight_fit_producer_retirement_clears_busy_and_rejects_late_result(
    monkeypatch,
) -> None:
    """Lane cancellation cannot be the only Fit retirement acknowledgement."""

    import threading

    from PyQt5 import QtCore, QtTest

    from zlc_frontend.qt_widgets import FigureFitLane, ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    original_execute = FigureFitLane._execute
    entered = threading.Event()
    release = threading.Event()

    def held_execute(request):
        entered.set()
        if not release.wait(10.0):
            raise RuntimeError("test Fit retirement gate timed out")
        return original_execute(request)

    monkeypatch.setattr(FigureFitLane, "_execute", staticmethod(held_execute))
    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="1d", title="Curve", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        old_value = _curve_value(revision=1, center=-0.8)
        replacement_value = _curve_value(revision=2, center=1.2)
        assert _present_value(console, card, old_value, frame_key=("retire-busy", 1))
        _wait(application, lambda: card.frozen_render_payload() is not None)

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        pane = card.fit_authoring_pane
        _wait(application, lambda: bool(pane.fit_models) and pane.fit_button.isEnabled())
        pane.model_combo.setCurrentIndex(
            pane.model_combo.findData("gaussian_offset")
        )
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        assert entered.wait(5.0)
        assert pane._busy_kind == "fit"

        retired = console._current_test_source
        console._retire_logic_node_publications(retired.node)
        assert card._fit_active_spec is None
        assert card._fit_pending_source_ref is None
        assert card._fit_live_surface_source is None
        assert pane._busy_kind == "prepare"
        assert not any("/fit." in name for name in console._tick_data.names())

        release.set()
        _wait(application, lambda: console._fit_lane._future is None)
        application.processEvents(QtCore.QEventLoop.AllEvents, 50)
        assert card._fit_result is None
        assert card._fit_live_surface_source is None
        assert not any("/fit." in name for name in console._tick_data.names())

        console._current_test_source = None
        assert _present_value(
            console,
            card,
            replacement_value,
            frame_key=("retire-busy", 2),
        )
        _wait(
            application,
            lambda: card._presented_value is not None
            and card._presented_value.run_id == replacement_value.run_id,
        )
        assert card.frozen_render_payload().fit_overlays == ()
    finally:
        release.set()
        close_task_console(application, console)


def test_curve_fit_pins_exact_live_front_and_every_failure_resumes_latest(
    monkeypatch,
) -> None:
    """CURVE shares the Fit command lifecycle; no failure can strand its pin."""

    import threading

    from PyQt5 import QtCore, QtTest

    from zlc_frontend.qt_widgets import FigureFitLane, ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    original_execute = FigureFitLane._execute
    entered = threading.Event()
    release = threading.Event()

    def held_execute(request):
        entered.set()
        if not release.wait(10.0):
            raise RuntimeError("test Fit gate timed out")
        return original_execute(request)

    monkeypatch.setattr(FigureFitLane, "_execute", staticmethod(held_execute))
    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="1d", title="Curve", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        fitted = _curve_value(revision=1, center=-0.8)
        newer = _curve_value(revision=2, center=0.6)
        newest = _curve_value(revision=3, center=1.4)
        assert _present_value(console, card, fitted, frame_key=("curve", 1))
        _wait(application, lambda: card.frozen_render_payload() is not None)
        before = card.board.front_frame.panels[0].raster.pixels

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        pane = card.fit_authoring_pane
        _wait(application, lambda: bool(pane.fit_models) and pane.fit_button.isEnabled())
        gaussian = pane.model_combo.findData("gaussian_offset")
        assert gaussian >= 0
        pane.model_combo.setCurrentIndex(gaussian)
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        assert entered.wait(5.0)
        assert not _present_value(console, card, newer, frame_key=("curve", 2))
        assert not _present_value(console, card, newest, frame_key=("curve", 3))
        release.set()
        _wait(
            application,
            lambda: card._fit_result is not None
            and card.frozen_render_value().snapshot.ref == fitted.snapshot.ref
            and bool(card.frozen_render_payload().fit_overlays),
        )
        assert card.board.front_frame.panels[0].raster.pixels != before
        parameter_names = {
            definition.name for definition in card._fit_result.parameter_definitions
        }
        _wait(
            application,
            lambda: {
                name.rsplit("/fit.", 1)[1]
                for name in console._tick_data.names()
                if "/fit." in name
            }
            == parameter_names,
        )
        assert all(
            not name.endswith(("fit.curve", "fit.overlay", "fit.prediction"))
            for name in console._tick_data.names()
        )

        QtTest.QTest.mouseClick(pane.clear_button, QtCore.Qt.LeftButton)
        _wait(
            application,
            lambda: card.frozen_render_value().snapshot.ref == newest.snapshot.ref
            and not card.frozen_render_payload().fit_overlays,
        )

        # A malformed worker completion is a terminal failure, not a permanent
        # command pin.  The newest admitted data front must resume immediately.
        monkeypatch.setattr(
            FigureFitLane,
            "_execute",
            staticmethod(lambda request: (request, object(), None)),
        )
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        failed_newest = _curve_value(revision=4, center=2.0)
        # The malformed completion may already have terminated the command by
        # the time this next front is offered.  Whether it is briefly pinned is
        # scheduler timing; the product contract is only that the failure
        # cannot strand it and this newest complete front becomes visible.
        _present_value(
            console,
            card,
            failed_newest,
            frame_key=("curve", 4),
        )
        _wait(
            application,
            lambda: card._fit_live_surface_source is None
            and card.frozen_render_value().snapshot.ref
            == failed_newest.snapshot.ref
            and pane.fit_button.isEnabled(),
        )
        assert not any("/fit." in name for name in console._tick_data.names())

        # Even an exception escaping the worker callable keeps the submitted
        # request identity, so the same panel-terminal cleanup releases the
        # exact-source pin and resumes the newest data-plane front.
        exception_entered = threading.Event()
        exception_release = threading.Event()
        terminal_failures = []
        original_finish_failure = card._finish_fit_failure

        def record_terminal_failure(request, diagnostic, *, cancelled):
            terminal_failures.append((request, diagnostic, cancelled))
            return original_finish_failure(
                request,
                diagnostic,
                cancelled=cancelled,
            )

        monkeypatch.setattr(
            card,
            "_finish_fit_failure",
            record_terminal_failure,
        )

        def exploding_execute(request):
            exception_entered.set()
            if not exception_release.wait(10.0):
                raise RuntimeError("test exception gate timed out")
            raise RuntimeError("unexpected executor failure")

        monkeypatch.setattr(
            FigureFitLane,
            "_execute",
            staticmethod(exploding_execute),
        )
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        assert exception_entered.wait(5.0)
        exception_newest = _curve_value(revision=5, center=2.8)
        assert not _present_value(
            console,
            card,
            exception_newest,
            frame_key=("curve", 5),
        )
        exception_release.set()
        _wait(
            application,
            lambda: card._fit_live_surface_source is None
            and card.frozen_render_value().snapshot.ref
            == exception_newest.snapshot.ref
            and pane.fit_button.isEnabled(),
        )
        assert len(terminal_failures) == 1
        assert "unexpected executor failure" in terminal_failures[0][1]
    finally:
        release.set()
        close_task_console(application, console)


def test_histogram_fit_draws_the_formal_components_and_publishes_only_parameters() -> None:
    """TaskConsole HIST consumes the same formal Fit path as DataFigure."""

    from PyQt5 import QtCore, QtTest

    from zlc_data import FitBatchStatus
    from zlc_frontend import HistogramPanelPayload
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="hist", title="Histogram", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        source = _histogram_value(revision=1)
        assert _present_value(console, card, source, frame_key=("histogram", 1))
        _wait(
            application,
            lambda: isinstance(card.frozen_render_payload(), HistogramPanelPayload),
        )
        before = card.board.front_frame.panels[0].raster.pixels

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        pane = card.fit_authoring_pane
        _wait(application, lambda: bool(pane.fit_models) and pane.fit_button.isEnabled())
        assert pane.fit_models[:2] == (
            "bimodal_gaussian",
            "histogram_gaussian",
        )
        bimodal = pane.model_combo.findData("bimodal_gaussian")
        assert bimodal >= 0
        pane.model_combo.setCurrentIndex(bimodal)
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _wait(
            application,
            lambda: card._fit_result is not None
            and card._fit_result.statuses == (FitBatchStatus.CONVERGED,)
            and isinstance(card.frozen_render_payload(), HistogramPanelPayload)
            and bool(card.frozen_render_payload().fit_overlays),
            timeout=25.0,
        )

        payload = card.frozen_render_payload()
        assert len(payload.fit_overlays) == 1
        assert len(payload.fit_overlays[0].component_predictions) == 3
        assert card.board.front_frame.panels[0].raster.pixels != before
        parameter_names = {
            definition.name for definition in card._fit_result.parameter_definitions
        }
        _wait(
            application,
            lambda: {
                name.rsplit("/fit.", 1)[1]
                for name in console._tick_data.names()
                if "/fit." in name
            }
            == parameter_names,
        )
        assert all(
            not name.endswith(("fit.curve", "fit.overlay", "fit.prediction"))
            for name in console._tick_data.names()
        )
    finally:
        close_task_console(application, console)


def test_fit_source_capture_failure_releases_the_frozen_command(
    monkeypatch,
) -> None:
    """A pre-worker ancestry failure is a terminal Fit completion."""

    from PyQt5 import QtCore, QtTest

    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.console_records import PanelConfig
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="1d", title="Curve", signal="image"),),
        ),
        window_px=(900, 700),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        fitted = _curve_value(revision=1, center=-0.5)
        assert _present_value(console, card, fitted, frame_key=("capture", 1))
        _wait(application, lambda: card.frozen_render_payload() is not None)

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        pane = card.fit_authoring_pane
        _wait(application, lambda: bool(pane.fit_models) and pane.fit_button.isEnabled())
        gaussian = pane.model_combo.findData("gaussian_offset")
        assert gaussian >= 0
        pane.model_combo.setCurrentIndex(gaussian)

        def fail_capture(_source):
            raise RuntimeError("causal ancestry unavailable")

        monkeypatch.setattr(
            console._data,
            "capture_source_component",
            fail_capture,
        )
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        _wait(
            application,
            lambda: card._fit_live_surface_source is None
            and card._fit_pending_source_ref is None
            and pane.fit_button.isEnabled(),
        )
        assert "causal ancestry unavailable" in card._status_text

        newest = _curve_value(revision=2, center=1.5)
        assert _present_value(console, card, newest, frame_key=("capture", 2))
        _wait(
            application,
            lambda: card.frozen_render_value().snapshot.ref == newest.snapshot.ref,
        )
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
