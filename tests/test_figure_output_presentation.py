"""Frontend ownership of Figure-derived signal presentation and binding facts."""

from __future__ import annotations

import os

import numpy as np

from gui_user_flow import close_task_console


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _image_snapshot():
    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        SPATIAL_X,
        SPATIAL_Y,
        AxisId,
        AxisSpec,
        BlockId,
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

    repeat = AxisSpec(AxisId("figure.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId("figure.point"), "point", SCAN_POINT, 1, (0,))
    y_axis = AxisSpec(
        AxisId("figure.y"), "camera y", SPATIAL_Y, 4, (0.0, 1.0, 2.0, 3.0), "px"
    )
    x_axis = AxisSpec(
        AxisId("figure.x"), "camera x", SPATIAL_X, 5, (0.0, 1.0, 2.0, 3.0, 4.0), "px"
    )
    values = np.arange(20, dtype=np.uint8).reshape(1, 1, 4, 5)
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
        BlockId("figure-output-presentation"),
        DatasetRevision(3),
        values,
        DatasetComponentValidity(
            (y_axis.axis_id, x_axis.axis_id),
            np.ones(values.shape, dtype=np.bool_),
        ),
        schema,
    )
    return (
        OwnedSnapshot(
            block.ref(StreamGenerationId("figure-output-presentation-generation")),
            block,
        ),
        y_axis,
        x_axis,
    )


def test_area_output_carries_complete_presentation_and_typed_source_transform():
    from zlc_data import (
        AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
        DataTransformSpec,
        IndexRangeSelection,
        Selection,
        projected_dataset_output_contract_id,
    )
    from zlc_frontend.figure_outputs import (
        AREA_DATA_OUTPUT,
        FigureOutputPresentation,
        materialize_area_outputs,
    )
    from zlc_frontend.figure_source import FigureSource
    from zlc_workbench.task_console.console_records import panel_signal_key
    from zlc_workbench.task_console.data_plane import (
        ConsoleDataPlane,
        ConsoleSignalValue,
    )

    snapshot, y_axis, x_axis = _image_snapshot()
    selection = Selection(
        (
            IndexRangeSelection(y_axis.axis_id, 1, 4),
            IndexRangeSelection(x_axis.axis_id, 1, 5),
        )
    )
    source_contract = "tests.camera-frame"
    outputs = materialize_area_outputs(
        FigureSource(snapshot, source_contract_id=source_contract),
        selection,
    )

    area = outputs[AREA_DATA_OUTPUT]
    assert area.presentation == FigureOutputPresentation(
        AREA_DATA_OUTPUT,
        projected_dataset_output_contract_id(
            source_contract,
            AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
        ),
        "Area data",
        "Area data",
        "Dataset inside the committed Figure Area selection.",
    )
    assert area.source_transform == DataTransformSpec((selection,))
    assert outputs.keys() == {
        value.presentation.name for value in outputs.values()
    }
    assert all(value.presentation.description for value in outputs.values())

    source_value = ConsoleSignalValue(
        name="image",
        source="camera",
        snapshot=snapshot,
        coverage=None,
        run_id="figure-area-run",
        epoch_id="figure-area-epoch",
        join_digest="0" * 64,
    )
    plane = ConsoleDataPlane()
    try:
        plane.publish_panel("figure-area-panel", source_value, outputs)
        published = plane.freeze().value(
            panel_signal_key("figure-area-panel", AREA_DATA_OUTPUT)
        )
        assert published is not None
        assert published.presentation == area.presentation
        assert published.source_transform == area.source_transform
    finally:
        plane.close()


def test_task_console_mechanically_adapts_frontend_figure_presentation():
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_data import AxisId
    from zlc_frontend.figure_outputs import (
        FIGURE_CROSS_COORDINATE_OUTPUT_CONTRACT_ID,
        SelectorAxisMetadata,
        materialize_cross_outputs,
    )
    from zlc_frontend.figure_source import FigureSource
    from zlc_neutral_atom.logic_node_declaration import OutputPresentation
    from zlc_workbench.task_console.console_records import PanelConfig, panel_signal_key
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.data_plane import ConsoleSignalValue
    from zlc_workbench.task_console.window import TaskConsole

    snapshot, _y_axis, _x_axis = _image_snapshot()
    source = ConsoleSignalValue(
        name="image",
        source="camera",
        snapshot=snapshot,
        coverage=None,
        run_id="figure-output-run",
        epoch_id="figure-output-epoch",
        join_digest="0" * 64,
    )
    outputs = materialize_cross_outputs(
        FigureSource(snapshot),
        (2.5, 1.5),
        (
            SelectorAxisMetadata(AxisId("figure.x"), "camera x", "px"),
            SelectorAxisMetadata(AxisId("figure.y"), "camera y", "px"),
        ),
    )
    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="2d", title="Camera", signal="image"),),
        ),
        window_px=(800, 600),
    )
    try:
        card = console.cards[0]
        console._data.publish_panel(card.panel_id, source, outputs)
        console._promote_data_front(console._data.freeze())
        topology = console._signal_topology()
        for output in outputs.values():
            presentation = output.presentation
            key = panel_signal_key(card.panel_id, presentation.name)
            projected = topology[key].declaration
            assert isinstance(projected, OutputPresentation)
            assert projected.name == presentation.name
            assert projected.contract_id == FIGURE_CROSS_COORDINATE_OUTPUT_CONTRACT_ID
            assert projected.short == presentation.short
            assert projected.axis_label == presentation.axis_label
            assert projected.description == presentation.description
    finally:
        close_task_console(application, console)
