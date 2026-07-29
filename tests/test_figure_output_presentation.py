"""Figure outputs use exact signal publications and static frontend metadata."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _image_snapshot(
    *,
    revision: int = 3,
    repeat_size: int = 1,
    point_count: int = 1,
):
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
        PointTable,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )

    repeat = AxisSpec(
        AxisId("figure.repeat"),
        "repeat",
        REPEAT,
        repeat_size,
        tuple(range(repeat_size)),
    )
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
    values = np.arange(
        repeat_size * point_count * 20,
        dtype=np.uint8,
    ).reshape(repeat_size, point_count, 4, 5) + np.uint8(revision)
    schema = DatasetSchema(
        repeat,
        PointTable(point_count),
        None,
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


def _area_commit(snapshot, y_axis, x_axis):
    from zlc_data import IndexRangeSelection, Selection
    from zlc_frontend import ImageDisplayState
    from zlc_frontend.figure import ViewIntent, suggest_view
    from zlc_frontend.figure_outputs import bind_area_data_commit
    from zlc_frontend.panel_render import PanelComposer, PanelProvenance

    suggestion = suggest_view(snapshot.block.schema, ViewIntent.IMAGE)
    assert suggestion.spec is not None
    composer = PanelComposer(
        "figure-area",
        intent=ViewIntent.IMAGE,
        view=suggestion.spec,
    )
    try:
        frame, figure = composer.compose_with_figure(
            snapshot,
            display=ImageDisplayState(),
            provenance=PanelProvenance("run", "epoch", "0" * 64),
        )
        selection = Selection(
            (
                IndexRangeSelection(y_axis.axis_id, 1, 4),
                IndexRangeSelection(x_axis.axis_id, 1, 5),
            )
        )
        return bind_area_data_commit(
            frame.panels[0].source_identity,
            selection,
            figure,
        )
    finally:
        composer.close()


def _derived(value):
    from zlc_neutral_atom.processing.signal_plane import DerivedSignalOutput

    return DerivedSignalOutput(
        snapshot=value.snapshot,
        source_ref=value.source_ref,
        derivation_digest=value.derivation_digest,
        preserve_source_coverage=value.preserve_source_coverage,
    )


def _camera_plane(*, revision: int = 3):
    from zlc_neutral_atom.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )
    from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
    from zlc_neutral_atom.runtime.dataset import MonitorCoverage

    snapshot, y_axis, x_axis = _image_snapshot(revision=revision)
    declaration = DatasetOutputDeclaration("frame", "tests.camera-frame")
    state = {
        "output": LiveDatasetOutput(
            declaration,
            snapshot,
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
    plane = SignalDataPlane()
    plane.reserve(node)
    plane.attach(node, slot)
    plane.mark_changed(node, slot)
    return plane, state, node, slot, declaration, snapshot, y_axis, x_axis


def test_panel_derived_signal_advances_with_its_exact_parent_publication() -> None:
    from zlc_frontend.figure_outputs import (
        AREA_DATA_OUTPUT,
        area_data_output_presentation,
        figure_selector_identity,
        materialize_area_outputs,
    )
    from zlc_frontend.figure_source import FigureSource
    from zlc_neutral_atom.dataset_output import LiveDatasetOutput
    from zlc_neutral_atom.runtime.dataset import MonitorCoverage
    from zlc_workbench.task_console.console_records import panel_signal_key

    (
        plane,
        state,
        node,
        slot,
        declaration,
        snapshot_3,
        y_axis,
        x_axis,
    ) = _camera_plane()
    area_name = panel_signal_key("area-panel", AREA_DATA_OUTPUT)
    plane.set_front_signals({"camera/frame", area_name})
    commit = _area_commit(snapshot_3, y_axis, x_axis)
    try:
        source_front_3 = plane.freeze()
        publication_3 = source_front_3.publication("camera/frame")
        assert publication_3 is not None
        source_3 = FigureSource(
            snapshot_3,
            source_contract_id=declaration.contract_id,
        )
        output_3 = materialize_area_outputs(source_3, commit)[AREA_DATA_OUTPUT]
        generation = plane.bind_continuous_derived(
            "figure/area-panel/area",
            source_name="camera/frame",
            output_names=(area_name,),
            route_identity=figure_selector_identity(commit),
        )
        assert plane.publish_continuous_derived(
            "figure/area-panel/area",
            generation,
            publication_3,
            {area_name: _derived(output_3)},
        )
        coherent_3 = plane.freeze()
        assert coherent_3.publication(area_name).parents == (publication_3,)
        assert coherent_3.value("camera/frame").snapshot.ref.revision.value == 3
        assert coherent_3.value(area_name).snapshot.ref.revision.value == 3

        snapshot_4, _y_axis, _x_axis = _image_snapshot(revision=4)
        state["output"] = LiveDatasetOutput(
            declaration,
            snapshot_4,
            MonitorCoverage(1, 1, 0, False),
            "2" * 64,
        )
        plane.mark_changed(node, slot)
        staged = plane.freeze()
        assert staged.value("camera/frame").snapshot.ref.revision.value == 3
        assert staged.value(area_name).snapshot.ref.revision.value == 3

        publication_4 = plane.latest_publication("camera/frame")
        assert publication_4 is not None and publication_4 is not publication_3
        source_4 = FigureSource(
            snapshot_4,
            source_contract_id=declaration.contract_id,
        )
        output_4 = materialize_area_outputs(source_4, commit)[AREA_DATA_OUTPUT]
        assert plane.publish_continuous_derived(
            "figure/area-panel/area",
            generation,
            publication_4,
            {area_name: _derived(output_4)},
        )
        coherent_4 = plane.freeze()
        assert coherent_4.publication(area_name).parents == (publication_4,)
        assert coherent_4.value("camera/frame").snapshot.ref.revision.value == 4
        assert coherent_4.value(area_name).snapshot.ref.revision.value == 4

        presentation = area_data_output_presentation(declaration.contract_id)
        assert presentation.name == AREA_DATA_OUTPUT
        assert not hasattr(coherent_4.value(area_name), "presentation")
    finally:
        plane.close()


def test_derived_sibling_bundle_is_atomic_and_route_replacement_is_explicit() -> None:
    from zlc_neutral_atom.processing.signal_plane import DerivedSignalOutput

    (
        plane,
        _state,
        _node,
        _slot,
        _declaration,
        snapshot,
        _y_axis,
        _x_axis,
    ) = _camera_plane(revision=11)
    try:
        front = plane.freeze()
        parent = front.publication("camera/frame")
        assert parent is not None
        source = front.value("camera/frame")
        assert source is not None
        generation = plane.bind_continuous_derived(
            "figure/atomic",
            source_name="camera/frame",
            output_names=("@panel/atomic/area.data", "@panel/atomic/cross.data"),
            route_identity="atomic-selector",
        )
        value = DerivedSignalOutput(snapshot, source.snapshot.ref, "a" * 64)
        with pytest.raises(ValueError, match="sibling vocabulary"):
            plane.publish_continuous_derived(
                "figure/atomic",
                generation,
                parent,
                {"@panel/atomic/area.data": value},
            )
        assert plane.freeze().value("@panel/atomic/area.data") is None
        assert plane.publish_continuous_derived(
            "figure/atomic",
            generation,
            parent,
            {
                "@panel/atomic/area.data": value,
                "@panel/atomic/cross.data": value,
            },
        )
        admitted = plane.freeze()
        publication = admitted.publication("@panel/atomic/area.data")
        assert publication is admitted.publication("@panel/atomic/cross.data")

        plane.withdraw_derived("figure/atomic")
        assert plane.freeze().value("@panel/atomic/area.data") is None
        renamed_generation = plane.bind_continuous_derived(
            "figure/atomic",
            source_name="camera/frame",
            output_names=("@panel/atomic/roi.data",),
            route_identity="renamed-selector",
        )
        assert plane.publish_continuous_derived(
            "figure/atomic",
            renamed_generation,
            parent,
            {"@panel/atomic/roi.data": value},
        )
        assert plane.freeze().value("@panel/atomic/roi.data") is not None
    finally:
        plane.close()


def test_source_retirement_removes_nested_figure_publications() -> None:
    from zlc_frontend.figure_outputs import AREA_DATA_OUTPUT, materialize_area_outputs
    from zlc_frontend.figure_source import FigureSource

    (
        plane,
        _state,
        node,
        _slot,
        declaration,
        snapshot,
        y_axis,
        x_axis,
    ) = _camera_plane(revision=17)
    first_name = "@panel/first/area.data"
    second_name = "@panel/second/area.data"
    try:
        source_front = plane.freeze()
        source_publication = source_front.publication("camera/frame")
        assert source_publication is not None
        commit = _area_commit(snapshot, y_axis, x_axis)
        first_output = materialize_area_outputs(
            FigureSource(snapshot, source_contract_id=declaration.contract_id),
            commit,
        )[AREA_DATA_OUTPUT]
        first_generation = plane.bind_continuous_derived(
            "figure/first",
            source_name="camera/frame",
            output_names=(first_name,),
            route_identity="first-area",
        )
        assert plane.publish_continuous_derived(
            "figure/first",
            first_generation,
            source_publication,
            {first_name: _derived(first_output)},
        )
        first_front = plane.freeze()
        first_publication = first_front.publication(first_name)
        first_value = first_front.value(first_name)
        assert first_publication is not None and first_value is not None

        second_generation = plane.bind_continuous_derived(
            "figure/second",
            source_name=first_name,
            output_names=(second_name,),
            route_identity="second-area",
        )
        assert plane.publish_continuous_derived(
            "figure/second",
            second_generation,
            first_publication,
            {
                second_name: _derived(
                    SimpleNamespace(
                        snapshot=first_value.snapshot,
                        source_ref=first_value.snapshot.ref,
                        derivation_digest="b" * 64,
                        preserve_source_coverage=True,
                    )
                )
            },
        )
        assert plane.freeze().value(second_name) is not None

        retired = plane.retire(node)
        assert retired.issuperset(
            {"camera/frame", first_name, second_name}
        )
        assert plane.freeze().names() == ()
    finally:
        plane.close()


def test_area_metadata_and_event_transform_remain_separate_from_signal_value() -> None:
    from zlc_data import CommittedTransform
    from zlc_data.output_contract import (
        AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
        projected_dataset_output_contract_id,
    )
    from zlc_frontend.figure_outputs import (
        AREA_DATA_OUTPUT,
        area_data_output_presentation,
        figure_event_transform,
        materialize_area_outputs,
    )
    from zlc_frontend.figure_source import FigureSource

    snapshot, y_axis, x_axis = _image_snapshot()
    commit = _area_commit(snapshot, y_axis, x_axis)
    source_contract = "tests.camera-frame"
    source = FigureSource(snapshot, source_contract_id=source_contract)
    area = materialize_area_outputs(source, commit)[AREA_DATA_OUTPUT]
    presentation = area_data_output_presentation(source_contract)

    assert presentation.contract_id == projected_dataset_output_contract_id(
        source_contract,
        AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
    )
    assert isinstance(commit.authority, CommittedTransform)
    assert figure_event_transform(source, commit) == commit.authority.spec
    assert not hasattr(area, "presentation")
    assert not hasattr(area, "source_transform")


@pytest.mark.parametrize(
    ("repeat_size", "point_count"),
    ((2, 1), (1, 2)),
)
def test_cross_event_transform_preserves_only_single_event_carriers(
    repeat_size: int,
    point_count: int,
) -> None:
    """Figure display carriers never leak into a signal-event projection."""

    from zlc_data import (
        AxisSourceRef,
        DataTransformSpec,
        IndexSelection,
        ReductionMethod,
        ReductionSpec,
        Selection,
    )
    from zlc_data.transform import commit_transform
    from zlc_frontend.figure import DatasetId, ViewIntent
    from zlc_frontend.figure_outputs import FigureCrossCommit, figure_event_transform
    from zlc_frontend.figure_source import FigureSource
    from zlc_frontend.render import SourceIdentity

    snapshot, y_axis, x_axis = _image_snapshot(
        repeat_size=repeat_size,
        point_count=point_count,
    )
    schema = snapshot.block.schema
    data_selection = Selection(
        (
            IndexSelection(y_axis.axis_id, 1),
            IndexSelection(x_axis.axis_id, 2),
        )
    )
    repeat_mean = ReductionSpec(
        (AxisSourceRef.tensor(schema.repeat_axis.axis_id),),
        ReductionMethod.MEAN,
    )
    committed = commit_transform(
        schema,
        DataTransformSpec((data_selection, repeat_mean)),
    )
    ref = snapshot.ref
    commit = FigureCrossCommit(
        SourceIdentity(
            DatasetId("figure-event-cross"),
            ref.block_id,
            ref.stream_generation,
            ref.schema_fingerprint,
        ),
        committed,
        ViewIntent.IMAGE,
        (2.0, 1.0),
    )
    source = FigureSource(snapshot)

    if repeat_size != 1 or point_count != 1:
        with pytest.raises(ValueError, match="local to one repeat/point event"):
            figure_event_transform(source, commit)
        return

    assert figure_event_transform(source, commit) == DataTransformSpec(
        (data_selection,)
    )


def test_singleton_cross_repeat_mean_is_removed_from_event_transform() -> None:
    """R=1/P=1 Cross keeps its data-axis selection and no Dataset carrier."""

    from zlc_data import (
        AxisSourceRef,
        DataTransformSpec,
        IndexSelection,
        ReductionMethod,
        ReductionSpec,
        Selection,
    )
    from zlc_data.transform import commit_transform
    from zlc_frontend.figure import DatasetId, ViewIntent
    from zlc_frontend.figure_outputs import FigureCrossCommit, figure_event_transform
    from zlc_frontend.figure_source import FigureSource
    from zlc_frontend.render import SourceIdentity

    snapshot, y_axis, x_axis = _image_snapshot()
    schema = snapshot.block.schema
    selection = Selection(
        (
            IndexSelection(y_axis.axis_id, 1),
            IndexSelection(x_axis.axis_id, 2),
        )
    )
    committed = commit_transform(
        schema,
        DataTransformSpec(
            (
                selection,
                ReductionSpec(
                    (AxisSourceRef.tensor(schema.repeat_axis.axis_id),),
                    ReductionMethod.MEAN,
                ),
            )
        ),
    )
    ref = snapshot.ref
    commit = FigureCrossCommit(
        SourceIdentity(
            DatasetId("figure-event-cross"),
            ref.block_id,
            ref.stream_generation,
            ref.schema_fingerprint,
        ),
        committed,
        ViewIntent.IMAGE,
        (2.0, 1.0),
    )

    assert figure_event_transform(FigureSource(snapshot), commit) == (
        DataTransformSpec((selection,))
    )
