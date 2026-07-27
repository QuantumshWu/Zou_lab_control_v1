"""Frontend ownership of Figure-derived signal presentation and binding facts."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from gui_user_flow import close_task_console


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _derived_routes(panel_id: str, outputs):
    from zlc_neutral_atom.processing.signal_plane import DerivedSignalOutput
    from zlc_workbench.task_console.console_records import panel_signal_key

    return {
        panel_signal_key(panel_id, value.presentation.name): DerivedSignalOutput(
            snapshot=value.snapshot,
            source_ref=value.source_ref,
            derivation_digest=value.derivation_digest,
            preserve_source_coverage=value.preserve_source_coverage,
            source_transform=value.source_transform,
        )
        for value in outputs.values()
    }


def _image_snapshot(*, revision: int = 3):
    from zlc_data import (
        REPEAT,
        SCAN_POINT,
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

    repeat = AxisSpec(AxisId("figure.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId("figure.point"), "point", SCAN_POINT, 1, (0,))
    coordinate_frame = CoordinateFrameId("figure.camera")
    y_axis = AxisSpec(
        AxisId("figure.y"),
        "camera y",
        SPATIAL_Y,
        4,
        (0.0, 1.0, 2.0, 3.0),
        "pixel",
        coordinate_frame,
    )
    x_axis = AxisSpec(
        AxisId("figure.x"),
        "camera x",
        SPATIAL_X,
        5,
        (0.0, 1.0, 2.0, 3.0, 4.0),
        "pixel",
        coordinate_frame,
    )
    values = (
        np.arange(20, dtype=np.uint8).reshape(1, 1, 4, 5)
        + np.uint8(revision)
    )
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
        DatasetRevision(revision),
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


def test_panel_derived_signal_advances_with_its_source_as_one_causal_front():
    from zlc_data import IndexRangeSelection, Selection
    from zlc_frontend.figure_outputs import AREA_DATA_OUTPUT, materialize_area_outputs
    from zlc_frontend.figure_source import FigureSource
    from zlc_neutral_atom.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )
    from zlc_neutral_atom.runtime.dataset import MonitorCoverage
    from zlc_workbench.task_console.console_records import panel_signal_key
    from zlc_workbench.task_console.presentation_index import (
        ConsolePresentationIndex,
    )
    from zlc_neutral_atom.processing.signal_plane import SignalDataPlane

    first_snapshot, y_axis, x_axis = _image_snapshot(revision=3)
    declaration = DatasetOutputDeclaration("frame", "tests.camera-frame")
    state = {
        "output": LiveDatasetOutput(
            declaration,
            first_snapshot,
            MonitorCoverage(1, 1, 0, False),
            "1" * 64,
        )
    }
    node = SimpleNamespace(
        instance_id="camera",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"camera/{name}",
    )
    slot = SimpleNamespace(
        freeze_live_outputs=lambda: (
            "camera-run",
            "camera-epoch",
            {"frame": state["output"]},
        ),
        close=lambda: None,
        notification_failure=None,
    )
    selection = Selection(
        (
            IndexRangeSelection(y_axis.axis_id, 1, 4),
            IndexRangeSelection(x_axis.axis_id, 1, 5),
        )
    )
    area_name = panel_signal_key("area-panel", AREA_DATA_OUTPUT)
    plane = SignalDataPlane()
    presentations = ConsolePresentationIndex()
    try:
        plane.attach(node, slot)
        plane.mark_changed(node)
        source_3 = plane.freeze().value("camera/frame")
        assert source_3 is not None
        capture_3 = plane.capture_source_component(source_3)
        outputs_3 = materialize_area_outputs(
            FigureSource(
                source_3.snapshot,
                source_contract_id="tests.camera-frame",
            ),
            selection,
        )
        published_3 = plane.publish_derived(
            "area-panel",
            source_3,
            _derived_routes("area-panel", outputs_3),
            source_component=capture_3,
        )
        presentations.publish(
            published_3,
            {area_name: outputs_3[AREA_DATA_OUTPUT].presentation},
        )
        coherent_3 = plane.freeze()
        presentations.reconcile_visible(coherent_3.signals)
        assert coherent_3.value("camera/frame").snapshot.ref.revision.value == 3
        area_3 = coherent_3.value(area_name)
        assert area_3.snapshot.ref.revision.value == 3
        assert presentations.presentation_for(area_3) is not None

        snapshot_4, _y_axis, _x_axis = _image_snapshot(revision=4)
        state["output"] = LiveDatasetOutput(
            declaration,
            snapshot_4,
            MonitorCoverage(1, 1, 0, False),
            "2" * 64,
        )
        plane.mark_changed(node)
        staged = plane.freeze()
        assert staged.value("camera/frame").snapshot.ref.revision.value == 3
        assert staged.value(area_name).snapshot.ref.revision.value == 3

        source_4 = plane.candidate_value("camera/frame")
        assert source_4 is not None
        assert source_4.snapshot.ref.revision.value == 4
        capture_4 = plane.capture_source_component(source_4)
        outputs_4 = materialize_area_outputs(
            FigureSource(
                source_4.snapshot,
                source_contract_id="tests.camera-frame",
            ),
            selection,
        )
        published_4 = plane.publish_derived(
            "area-panel",
            source_4,
            _derived_routes("area-panel", outputs_4),
            source_component=capture_4,
        )
        presentations.publish(
            published_4,
            {area_name: outputs_4[AREA_DATA_OUTPUT].presentation},
        )
        # The N+1 sidecar is staged, but the consumer-visible causal front is
        # still N.  Publishing N+1 must not invalidate the presentation beside
        # the exact N value that TaskConsole topology and Setting still read.
        presentations.reconcile_visible(staged.signals)
        assert presentations.presentation_for(staged.value(area_name)) is not None
        coherent_4 = plane.freeze()
        presentations.reconcile_visible(coherent_4.signals)
        assert coherent_4.value("camera/frame").snapshot.ref.revision.value == 4
        area_4 = coherent_4.value(area_name)
        assert area_4.snapshot.ref.revision.value == 4
        assert presentations.presentation_for(area_4) is not None
        assert presentations.presentation_for(area_3) is None
    finally:
        plane.close()


def test_presentation_replacement_is_atomic_and_retires_renamed_routes():
    """A conflict cannot half-admit siblings; a rename leaves no stale route."""

    import pytest

    from zlc_frontend.figure_outputs import FigureOutputPresentation
    from zlc_neutral_atom.processing.signal_plane import SignalValue
    from zlc_workbench.task_console.presentation_index import (
        ConsolePresentationIndex,
    )

    snapshot, _y_axis, _x_axis = _image_snapshot(revision=11)

    def value(name: str) -> SignalValue:
        return SignalValue(
            name=name,
            source_instance_id="presentation-owner",
            snapshot=snapshot,
            coverage=None,
            run_id="presentation-run",
            epoch_id="presentation-epoch",
            join_digest="a" * 64,
        )

    def presentation(name: str, short: str) -> FigureOutputPresentation:
        return FigureOutputPresentation(
            name,
            f"tests.presentation.{name}",
            short,
            short,
            f"{short} description",
        )

    old_name = "@panel/atomic/area.data"
    sibling_name = "@panel/atomic/cross.data"
    renamed_name = "@panel/atomic/roi.data"
    old_value = value(old_name)
    sibling_value = value(sibling_name)
    renamed_value = value(renamed_name)
    old_presentation = presentation("area.data", "Area")
    index = ConsolePresentationIndex()
    index.publish({old_name: old_value}, {old_name: old_presentation})
    index.reconcile_visible({old_name: old_value})

    # Put the new sibling first: the old loop-mutating implementation admitted
    # it before discovering the conflict in the second item.
    with pytest.raises(ValueError, match="conflicting presentations"):
        index.publish(
            {
                sibling_name: sibling_value,
                old_name: old_value,
            },
            {
                sibling_name: presentation("cross.data", "Cross"),
                old_name: presentation("area.data", "Conflicting area"),
            },
        )
    assert sibling_name not in index._routes
    assert index.presentation_for(old_value) == old_presentation

    renamed_presentation = presentation("roi.data", "ROI")
    prepared = index.prepare_publish(
        {renamed_name: renamed_value},
        {renamed_name: renamed_presentation},
        withdraw=(old_name,),
    )
    index.commit_prepared(prepared)
    index.reconcile_visible({renamed_name: renamed_value})
    assert old_name not in index._routes
    assert index.presentation_for(renamed_value) == renamed_presentation


def test_logic_generation_retirement_promotes_value_and_presentation_together():
    """Restart/remove retires the source and its full Figure descendant tree."""

    from zlc_data import IndexRangeSelection, Selection
    from zlc_frontend.figure_outputs import (
        AREA_DATA_OUTPUT,
        FigureOutputPresentation,
        materialize_area_outputs,
    )
    from zlc_frontend.figure_source import FigureSource
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_neutral_atom.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )
    from zlc_neutral_atom.runtime.dataset import MonitorCoverage
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    snapshot, _y_axis, _x_axis = _image_snapshot(revision=17)
    declaration = DatasetOutputDeclaration("figure", "tests.logic-figure")
    key = "logic/figure"
    node = SimpleNamespace(
        instance_id="logic-generation",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: key,
        published_signals=lambda: (key,),
    )
    closed = []
    slot = SimpleNamespace(
        freeze_live_outputs=lambda: (
            "logic-run",
            "logic-epoch",
            {
                "figure": LiveDatasetOutput(
                    declaration,
                    snapshot,
                    MonitorCoverage(1, 1, 0, False),
                    "f" * 64,
                )
            },
        ),
        close=lambda: closed.append(True),
        notification_failure=None,
    )
    presentation = FigureOutputPresentation(
        "figure",
        "tests.logic-figure",
        "Figure",
        "Figure",
        "Exact logic-owned Figure output.",
    )
    console = TaskConsole(state=TaskConsoleState())
    try:
        console._data.attach(node, slot)
        console._data.mark_changed(node)
        front = console._data.freeze()
        value = front.value(key)
        assert value is not None
        console._presentations.publish({key: value}, {key: presentation})
        console._presentations.register_final_projector(
            node.instance_id,
            lambda *_args: {},
        )
        console._promote_data_front(front)
        assert console._presentations.presentation_for(value) == presentation

        selection = Selection(
            (
                IndexRangeSelection(_y_axis.axis_id, 1, 4),
                IndexRangeSelection(_x_axis.axis_id, 1, 5),
            )
        )
        first_outputs = materialize_area_outputs(
            FigureSource(value.snapshot, source_contract_id=declaration.contract_id),
            selection,
        )
        first_name = "@panel/first-area/area.data"
        first_values = console._data.publish_derived(
            "first-area",
            value,
            _derived_routes("first-area", first_outputs),
            source_component=console._data.capture_source_component(value),
        )
        console._presentations.publish(
            first_values,
            {first_name: first_outputs[AREA_DATA_OUTPUT].presentation},
        )
        first_front = console._data.freeze()
        console._promote_data_front(first_front)
        first_value = first_front.value(first_name)
        assert first_value is not None

        downstream_selection = Selection(
            (
                IndexRangeSelection(_y_axis.axis_id, 0, 3),
                IndexRangeSelection(_x_axis.axis_id, 0, 4),
            )
        )
        second_outputs = materialize_area_outputs(
            FigureSource(
                first_value.snapshot,
                source_contract_id=first_outputs[
                    AREA_DATA_OUTPUT
                ].presentation.contract_id,
            ),
            downstream_selection,
        )
        second_name = "@panel/second-area/area.data"
        second_values = console._data.publish_derived(
            "second-area",
            first_value,
            _derived_routes("second-area", second_outputs),
            source_component=console._data.capture_source_component(first_value),
        )
        console._presentations.publish(
            second_values,
            {second_name: second_outputs[AREA_DATA_OUTPUT].presentation},
        )
        console._promote_data_front(console._data.freeze())
        assert console._tick_data.value(second_name) is not None

        console._retire_logic_node_publications(
            node,
            unregister_final_projector=True,
        )
        assert console._tick_data.value(key) is None
        assert console._tick_data.value(first_name) is None
        assert console._tick_data.value(second_name) is None
        assert key not in console._presentations._routes
        assert first_name not in console._presentations._routes
        assert second_name not in console._presentations._routes
        assert not console._data._derived
        assert node.instance_id not in console._presentations._final_projectors
        assert closed == [True]
    finally:
        close_task_console(application, console)


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

def test_task_console_mechanically_adapts_frontend_figure_presentation():
    from zlc_data import DataTransformSpec
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_frontend import ImageDisplayState
    from zlc_frontend.figure import ViewIntent, suggest_view
    from zlc_frontend.figure_outputs import (
        FIGURE_CROSS_DATA_OUTPUT_CONTRACT_ID,
        bind_cross_data_commit,
        materialize_cross_outputs,
    )
    from zlc_frontend.figure_source import FigureSource
    from zlc_frontend.panel_render import PanelComposer, PanelProvenance
    from zlc_neutral_atom.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )
    from zlc_neutral_atom.logic_node_declaration import OutputPresentation
    from zlc_neutral_atom.runtime.dataset import MonitorCoverage
    from zlc_workbench.task_console.console_records import PanelConfig, panel_signal_key
    from zlc_workbench.task_console.console_state import TaskConsoleState
    from zlc_workbench.task_console.window import TaskConsole

    snapshot, _y_axis, _x_axis = _image_snapshot()
    declaration = DatasetOutputDeclaration("image", "tests.camera-frame")
    live_output = LiveDatasetOutput(
        declaration,
        snapshot,
        MonitorCoverage(1, 1, 0, False),
        "1" * 64,
    )
    node = SimpleNamespace(
        instance_id="camera",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"camera/{name}",
    )
    slot = SimpleNamespace(
        freeze_live_outputs=lambda: (
            "figure-output-run",
            "figure-output-epoch",
            {"image": live_output},
        ),
        close=lambda: None,
        notification_failure=None,
    )
    suggestion = suggest_view(snapshot.block.schema, ViewIntent.IMAGE)
    assert suggestion.spec is not None
    composer = PanelComposer(
        "figure-presentation-cross",
        intent=ViewIntent.IMAGE,
        view=suggestion.spec,
    )
    try:
        frame, figure = composer.compose_with_figure(
            snapshot,
            display=ImageDisplayState(),
            provenance=PanelProvenance("run", "epoch", "0" * 64),
        )
        panel = frame.panels[0]
        commit = bind_cross_data_commit(
            panel.source_identity,
            (2.5, 1.5),
            figure,
            panel.display_payload,
        )
        outputs = materialize_cross_outputs(FigureSource(snapshot), commit)
    finally:
        composer.close()
    cross = next(iter(outputs.values()))
    assert isinstance(cross.source_transform, DataTransformSpec)
    assert cross.source_transform.operations
    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(
                PanelConfig(kind="2d", title="Camera", signal="camera/image"),
            ),
        ),
        window_px=(800, 600),
    )
    try:
        card = console.cards[0]
        console._data.attach(node, slot)
        console._data.mark_changed(node)
        source = console._data.freeze().value("camera/image")
        assert source is not None
        published = console._data.publish_derived(
            card.panel_id,
            source,
            _derived_routes(card.panel_id, outputs),
        )
        console._presentations.publish(
            published,
            {
                panel_signal_key(card.panel_id, value.presentation.name):
                value.presentation
                for value in outputs.values()
            },
        )
        console._card_output_presentations[card.panel_id] = frozenset(
            value.presentation for value in outputs.values()
        )
        console._signal_topology_changed()
        console._promote_data_front(console._data.freeze())
        topology = console._signal_topology()
        published_keys = []
        for output in outputs.values():
            presentation = output.presentation
            key = panel_signal_key(card.panel_id, presentation.name)
            published_keys.append(key)
            projected = topology[key].declaration
            assert isinstance(projected, OutputPresentation)
            assert projected.name == presentation.name
            assert projected.contract_id == FIGURE_CROSS_DATA_OUTPUT_CONTRACT_ID
            assert projected.short == presentation.short
            assert projected.axis_label == presentation.axis_label
            assert projected.description == presentation.description
        assert console._remove_panel(card)
        assert all(console._tick_data.value(key) is None for key in published_keys)
        assert all(key not in console._presentations._routes for key in published_keys)
    finally:
        close_task_console(application, console)
