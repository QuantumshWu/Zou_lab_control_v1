"""Headless contracts for zlc_frontend.figure."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib
from itertools import permutations, product
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pytest

from zlc_data import (
    COMPONENT,
    MONITOR_HISTORY,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    fit_spec_for,
)
from zlc_frontend.figure import (
    AxisViewRole,
    AxisViewBinding,
    DatasetDescriptor,
    DatasetId,
    DisplayReduction,
    DisplayReductionMethod,
    EvaluatedAxis,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
    FigureEvaluationCancelled,
    FigureEvaluationDeadlineExceeded,
    FigureEvaluationLimitExceeded,
    FigureEvaluationPolicy,
    FigureEvaluator,
    FigureLayer,
    FigureSelection,
    FixedIndex,
    LatestNonempty,
    RepeatViewMode,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
    decode_figure_document,
    decode_view_spec,
    encode_figure_document,
    encode_view_spec,
    figure_document_to_tree,
    suggest_view,
    validate_view_spec,
    view_spec_to_tree,
)
from zlc_storage.canonical import encode


def axis(name, role, size, *, coordinates=None, frame=None):
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)) if coordinates is None else tuple(coordinates),
        None,
        frame,
    )


def make_block(
    values,
    *,
    repeat_axis,
    point_axes,
    point_layout,
    data_axes=(),
    validity=VALID,
    component_axes=(),
    revision=4,
):
    contract = (
        ValidityContract.components(*component_axes)
        if component_axes
        else ValidityContract.value()
    )
    schema = DatasetSchema(
        repeat_axis,
        tuple(point_axes),
        point_layout,
        ValueSchema(tuple(data_axes), contract, np.asarray(values).dtype),
    )
    return DataBlock(
        BlockId("block-a"),
        DatasetRevision(revision),
        np.asarray(values),
        validity,
        schema,
    )


def resolved(block):
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("generation-a")), block)
    dataset_id = DatasetId("dataset-a")
    return dataset_id, ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),))


def document(block, view):
    dataset_id, datasets = resolved(block)
    doc = FigureDocument(
        "figure-a",
        3,
        (DatasetDescriptor(dataset_id, "signal", block.schema.fingerprint),),
        (FigureLayer("layer-a", dataset_id, view),),
    )
    return doc, datasets


def evaluated_data(block, view):
    doc, datasets = document(block, view)
    return FigureEvaluator().evaluate(doc, datasets)


def only_series(evaluated):
    return evaluated.layers[0].cells[0].series[0]


def test_suggest_curve_is_axis_total_role_driven_and_deterministic():
    repeat = axis("repeat", REPEAT, 3)
    detuning = axis("detuning", SCAN_POINT, 5)
    site = axis("site", SITE, 2)
    block = make_block(
        np.zeros((3, 5, 2)),
        repeat_axis=repeat,
        point_axes=(detuning,),
        point_layout=PointLayout.rect_c((5,)),
        data_axes=(site,),
    )
    first = suggest_view(block.schema, ViewIntent.CURVE)
    second = suggest_view(block.schema, ViewIntent.CURVE)
    assert first == second and first.status is SuggestionStatus.RESOLVED
    assert tuple(binding.axis_id.value for binding in first.spec.axis_bindings) == (
        "detuning",
        "repeat",
        "site",
    )
    assert {binding.axis_id: binding.role for binding in first.spec.axis_bindings} == {
        detuning.axis_id: AxisViewRole.X,
        repeat.axis_id: AxisViewRole.REDUCED,
        site.axis_id: AxisViewRole.BATCH,
    }
    validate_view_spec(block.schema, first.spec)


def test_monitor_history_is_display_only_curve_x_or_newest_image_slider():
    repeat = axis("history-repeat", REPEAT, 1)
    history = axis("monitor.history", MONITOR_HISTORY, 4)
    curve = make_block(
        np.arange(4, dtype=np.float64).reshape(1, 4),
        repeat_axis=repeat,
        point_axes=(history,),
        point_layout=PointLayout.rect_c((4,)),
    )
    curve_view = suggest_view(curve.schema, ViewIntent.CURVE)
    assert curve_view.status is SuggestionStatus.RESOLVED
    assert curve_view.spec.binding(history.axis_id).role is AxisViewRole.X

    y = axis("camera.y", SPATIAL_Y, 2)
    x = axis("camera.x", SPATIAL_X, 3)
    image = make_block(
        np.zeros((1, 4, 2, 3), dtype=np.uint16),
        repeat_axis=repeat,
        point_axes=(history,),
        point_layout=PointLayout.rect_c((4,)),
        data_axes=(y, x),
    )
    image_view = suggest_view(image.schema, ViewIntent.IMAGE)
    assert image_view.status is SuggestionStatus.RESOLVED
    history_binding = image_view.spec.binding(history.axis_id)
    assert history_binding.role is AxisViewRole.SLIDER
    assert history_binding.selector == FixedIndex(0)

    with pytest.raises(ValueError, match="declared-role axis"):
        fit_spec_for(curve.schema, "gaussian_offset")
    with pytest.raises(ValueError):
        fit_spec_for(
            curve.schema,
            "gaussian_offset",
            fit_axis_ids=(history.axis_id,),
        )


def test_ambiguous_axis_and_stale_selection_fail_without_tuple_order_guessing():
    repeat = axis("repeat", REPEAT, 1)
    first = axis("a", SCAN_POINT, 2)
    second = axis("b", SCAN_POINT, 3)
    schema = make_block(
        np.zeros((1, 6)),
        repeat_axis=repeat,
        point_axes=(first, second),
        point_layout=PointLayout.rect_c((2, 3)),
    ).schema
    suggestion = suggest_view(schema, ViewIntent.CURVE)
    assert suggestion.status is SuggestionStatus.NEEDS_INPUT
    assert {item.axis_id for item in suggestion.alternatives} == {
        first.axis_id,
        second.axis_id,
    }
    stale = suggest_view(
        schema,
        ViewIntent.CURVE,
        Selection.index(AxisId("removed-axis"), 0),
    )
    assert stale.status is SuggestionStatus.NEEDS_INPUT
    assert stale.reasons[0].code == "STALE_SELECTION_AXIS"


def test_scalar_selection_resolves_display_axis_ambiguity():
    repeat = axis("selected-repeat", REPEAT, 1)
    first = axis("selected-a", SCAN_POINT, 2)
    second = axis("selected-b", SCAN_POINT, 3)
    schema = make_block(
        np.zeros((1, 6)),
        repeat_axis=repeat,
        point_axes=(first, second),
        point_layout=PointLayout.rect_c((2, 3)),
    ).schema
    suggestion = suggest_view(
        schema,
        ViewIntent.CURVE,
        Selection.index(first.axis_id, 0),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    assert suggestion.spec.binding(first.axis_id).role is AxisViewRole.SELECTED
    assert suggestion.spec.binding(second.axis_id).role is AxisViewRole.X
    validate_view_spec(schema, suggestion.spec)


def test_automatic_layout_matches_independent_capacity_oracle_and_axis_permutations():
    repeat = axis("planner-repeat", REPEAT, 1)
    point = axis("planner-point", SCAN_POINT, 2)
    layout = PointLayout.rect_c((2,))
    cardinalities = (2, 3, 8, 17)
    role_defs = (
        ("planner-event", READOUT_EVENT),
        ("planner-site", SITE),
        ("planner-component", COMPONENT),
    )

    for sizes in product(cardinalities, repeat=3):
        feasible = False
        for assignment in product(("batch", "facet"), repeat=3):
            batch_product = 1
            facet_product = 1
            for size, bucket in zip(sizes, assignment):
                if bucket == "batch":
                    batch_product *= size
                else:
                    facet_product *= size
            if batch_product <= 32 and facet_product <= 36:
                feasible = True
                break

        expected_mapping = None
        defined_axes = tuple(
            axis(name, role, size)
            for (name, role), size in zip(role_defs, sizes)
        )
        for ordered_axes in permutations(defined_axes):
            schema = DatasetSchema(
                repeat,
                (point,),
                layout,
                ValueSchema(
                    tuple(ordered_axes),
                    ValidityContract.value(),
                    np.dtype("float64"),
                ),
            )
            suggestion = suggest_view(schema, ViewIntent.CURVE)
            assert (suggestion.spec is not None) is feasible
            if not feasible:
                continue
            mapping = tuple(
                sorted(
                    (
                        axis_spec.axis_id.value,
                        suggestion.spec.binding(axis_spec.axis_id).role.value,
                    )
                    for axis_spec in defined_axes
                )
            )
            if expected_mapping is None:
                expected_mapping = mapping
            else:
                assert mapping == expected_mapping

    site_a = axis("planner-site-a", SITE, 16)
    site_b = axis("planner-site-b", SITE, 3)
    same_role_mappings = set()
    for ordered_axes in permutations((site_a, site_b)):
        schema = DatasetSchema(
            repeat,
            (point,),
            layout,
            ValueSchema(
                tuple(ordered_axes),
                ValidityContract.value(),
                np.dtype("float64"),
            ),
        )
        suggestion = suggest_view(schema, ViewIntent.CURVE)
        assert suggestion.status is SuggestionStatus.RESOLVED
        same_role_mappings.add(
            tuple(
                (axis_id.value, suggestion.spec.binding(axis_id).role.value)
                for axis_id in (site_a.axis_id, site_b.axis_id)
            )
        )
    assert len(same_role_mappings) == 1

    repeat_batch = axis("planner-repeat-batch", REPEAT, 16)
    site = axis("planner-batch-site", SITE, 4)
    batch_schema = DatasetSchema(
        repeat_batch,
        (point,),
        layout,
        ValueSchema((site,), ValidityContract.value(), np.dtype("float64")),
    )
    batch_suggestion = suggest_view(
        batch_schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(repeat_mode=RepeatViewMode.BATCH),
    )
    assert batch_suggestion.status is SuggestionStatus.RESOLVED
    assert batch_suggestion.spec.binding(repeat_batch.axis_id).role is AxisViewRole.BATCH
    assert batch_suggestion.spec.binding(site.axis_id).role is AxisViewRole.FACET


def test_slider_bindings_preserve_contract_order_and_repeat_latest():
    repeat = axis("slider-repeat", REPEAT, 1)
    point = axis("slider-point", SCAN_POINT, 2)
    spectral = axis("slider-spectral", SPECTRAL, 3)
    site = axis("slider-site", SITE, 50)
    component = axis("slider-component", COMPONENT, 40)
    y = axis("slider-y", SPATIAL_Y, 2)
    x = axis("slider-x", SPATIAL_X, 3)
    image_schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((2,)),
        ValueSchema(
            (spectral, site, component, y, x),
            ValidityContract.value(),
            np.dtype("uint16"),
        ),
    )
    image_suggestion = suggest_view(
        image_schema,
        ViewIntent.IMAGE,
        Selection.index_range(site.axis_id, 5, 45),
    )
    assert image_suggestion.status is SuggestionStatus.RESOLVED
    for axis_id, expected_index in (
        (point.axis_id, 0),
        (spectral.axis_id, 0),
        (site.axis_id, 5),
        (component.axis_id, 0),
    ):
        binding = image_suggestion.spec.binding(axis_id)
        assert binding.role is AxisViewRole.SLIDER
        assert binding.selector == FixedIndex(expected_index)

    rolling_point = axis("rolling-meter-point", SCAN_POINT, 40)
    meter_schema = DatasetSchema(
        repeat,
        (rolling_point,),
        PointLayout.rect_c((40,)),
        ValueSchema((), ValidityContract.value(), np.dtype("float64")),
    )
    meter_suggestion = suggest_view(meter_schema, ViewIntent.METER)
    assert meter_suggestion.status is SuggestionStatus.RESOLVED
    assert meter_suggestion.spec.binding(repeat.axis_id).selector == FixedIndex(0)
    assert meter_suggestion.spec.binding(rolling_point.axis_id).selector == FixedIndex(0)

    repeated_point = axis(
        "repeated-meter-point",
        SCAN_POINT,
        2,
        coordinates=(10.0, 20.0),
    )
    repeated = axis("repeated-meter-repeat", REPEAT, 2)
    repeated_block = make_block(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        repeat_axis=repeated,
        point_axes=(repeated_point,),
        point_layout=PointLayout.rect_c((2,)),
    )
    repeated_suggestion = suggest_view(repeated_block.schema, ViewIntent.METER)
    assert repeated_suggestion.status is SuggestionStatus.RESOLVED
    assert isinstance(
        repeated_suggestion.spec.binding(repeated.axis_id).selector,
        LatestNonempty,
    )
    point_binding = repeated_suggestion.spec.binding(repeated_point.axis_id)
    assert point_binding.role is AxisViewRole.SLIDER
    assert point_binding.selector == FixedIndex(0)
    evaluated = evaluated_data(repeated_block, repeated_suggestion.spec)
    resolutions = {item.axis_id: item for item in evaluated.layers[0].resolutions}
    assert resolutions[repeated_point.axis_id].selector == "FIXED_INDEX"
    assert resolutions[repeated_point.axis_id].index == 0
    assert resolutions[repeated_point.axis_id].coordinate == 10.0
    assert resolutions[repeated.axis_id].selector == "LATEST_NONEMPTY"
    assert resolutions[repeated.axis_id].index == 1

    invalid_latest = replace(
        repeated_suggestion.spec,
        axis_bindings=tuple(
            replace(binding, selector=LatestNonempty())
            if binding.axis_id == repeated_point.axis_id
            else binding
            for binding in repeated_suggestion.spec.axis_bindings
        ),
    )
    with pytest.raises(ValueError, match="logical repeat axis"):
        validate_view_spec(repeated_block.schema, invalid_latest)


def test_sparse_point_sliders_use_one_physical_layout_tuple():
    repeat = axis("sparse-slider-repeat", REPEAT, 1)
    scan = axis("sparse-slider-scan", SCAN_POINT, 2)
    spectral = axis("sparse-slider-spectral", SPECTRAL, 2)
    y = axis("sparse-slider-y", SPATIAL_Y, 1)
    x = axis("sparse-slider-x", SPATIAL_X, 1)
    block = make_block(
        np.array([[[[7.0]]]]),
        repeat_axis=repeat,
        point_axes=(scan, spectral),
        point_layout=PointLayout.explicit((2, 2), ((0, 1),)),
        data_axes=(y, x),
    )
    suggestion = suggest_view(block.schema, ViewIntent.IMAGE)
    assert suggestion.status is SuggestionStatus.RESOLVED
    assert suggestion.spec.binding(scan.axis_id).selector == FixedIndex(0)
    assert suggestion.spec.binding(spectral.axis_id).selector == FixedIndex(1)
    evaluated = evaluated_data(block, suggestion.spec)
    np.testing.assert_array_equal(only_series(evaluated).data.values, [[7.0]])
    resolutions = {item.axis_id: item for item in evaluated.layers[0].resolutions}
    assert resolutions[scan.axis_id].index == 0
    assert resolutions[spectral.axis_id].index == 1

    hole = replace(
        suggestion.spec,
        axis_bindings=tuple(
            replace(binding, selector=FixedIndex(0))
            if binding.axis_id == spectral.axis_id
            else binding
            for binding in suggestion.spec.axis_bindings
        ),
    )
    with pytest.raises(ValueError, match="do not identify a physical point"):
        validate_view_spec(block.schema, hole)

    empty = suggest_view(
        block.schema,
        ViewIntent.IMAGE,
        Selection.index(spectral.axis_id, 0),
    )
    assert empty.status is SuggestionStatus.NEEDS_INPUT
    assert empty.reasons[0].code == "EMPTY_POINT_SELECTION"


def test_view_and_document_codec_are_strict_canonical_and_owner_delegating(
    monkeypatch,
):
    fingerprint = "0" * 64
    point_id = AxisId("point")
    repeat_id = AxisId("repeat")
    roi_id = AxisId("roi-x")
    shot_id = AxisId("shot")
    selection = Selection.index(roi_id, 2)
    view = ViewSpec(
        fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(
                roi_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            AxisViewBinding(point_id, AxisViewRole.X),
            AxisViewBinding(
                shot_id,
                AxisViewRole.SLIDER,
                selector=FixedIndex(1),
            ),
            AxisViewBinding(
                repeat_id,
                AxisViewRole.SELECTED,
                selector=LatestNonempty(),
            ),
        ),
        (selection,),
    )
    owner_selection_tree = {
        "schema": "zlc_data.Selection",
        "terms": [{"kind": "INDEX", "axis_id": "roi-x", "index": 2}],
    }
    expected_view_tree = {
        "schema": "zlc_frontend.ViewSpec",
        "schema_fingerprint": fingerprint,
        "intent": "CURVE",
        "axis_bindings": [
            {"axis_id": "point", "role": "X", "selector": None, "reduction": None},
            {
                "axis_id": "repeat",
                "role": "SELECTED",
                "selector": {"kind": "LATEST_NONEMPTY"},
                "reduction": None,
            },
            {
                "axis_id": "roi-x",
                "role": "REDUCED",
                "selector": None,
                "reduction": {"method": "MEAN"},
            },
            {
                "axis_id": "shot",
                "role": "SLIDER",
                "selector": {"kind": "FIXED_INDEX", "index": 1},
                "reduction": None,
            },
        ],
        "display_selections": [owner_selection_tree],
    }
    assert view_spec_to_tree(view) == expected_view_tree
    assert decode_view_spec(encode_view_spec(view)) == view

    dataset_a = DatasetId("dataset-a")
    dataset_z = DatasetId("dataset-z")
    doc = FigureDocument(
        "figure-golden",
        7,
        (
            DatasetDescriptor(dataset_z, "signal z", fingerprint),
            DatasetDescriptor(dataset_a, "signal a", fingerprint),
        ),
        (
            FigureLayer("layer-z", dataset_z, view),
            FigureLayer("layer-a", dataset_a, view),
        ),
        (FigureSelection("roi-choice", dataset_z, selection),),
    )
    expected_document_tree = {
        "schema": "zlc_frontend.FigureDocument",
        "document_id": "figure-golden",
        "revision": 7,
        "datasets": [
            {
                "dataset_id": "dataset-a",
                "label": "signal a",
                "schema_fingerprint": fingerprint,
            },
            {
                "dataset_id": "dataset-z",
                "label": "signal z",
                "schema_fingerprint": fingerprint,
            },
        ],
        "layers": [
            {"layer_id": "layer-z", "dataset_id": "dataset-z", "view": expected_view_tree},
            {"layer_id": "layer-a", "dataset_id": "dataset-a", "view": expected_view_tree},
        ],
        "selections": [
            {
                "selection_id": "roi-choice",
                "dataset_id": "dataset-z",
                "selection": owner_selection_tree,
            }
        ],
    }
    assert figure_document_to_tree(doc) == expected_document_tree
    assert decode_figure_document(encode_figure_document(doc)) == doc

    codec = importlib.import_module("zlc_frontend.figure.codec")
    real_to_tree = codec.selection_to_tree
    real_from_tree = codec.selection_from_tree
    owner_calls = {"to": 0, "from": 0}

    def observed_to_tree(value):
        owner_calls["to"] += 1
        return real_to_tree(value)

    def observed_from_tree(value):
        owner_calls["from"] += 1
        return real_from_tree(value)

    monkeypatch.setattr(codec, "selection_to_tree", observed_to_tree)
    monkeypatch.setattr(codec, "selection_from_tree", observed_from_tree)
    assert codec.view_spec_from_tree(codec.view_spec_to_tree(view)) == view
    assert codec.figure_document_from_tree(codec.figure_document_to_tree(doc)) == doc
    assert owner_calls == {"to": 4, "from": 4}

    extra = deepcopy(expected_view_tree)
    extra["authority_seed"] = "forbidden"
    with pytest.raises(ValueError, match="exactly"):
        decode_view_spec(encode(extra))
    unknown_view_schema = deepcopy(expected_view_tree)
    unknown_view_schema["schema"] = "zlc_frontend.UnknownView"
    with pytest.raises(ValueError, match="expected schema"):
        decode_view_spec(encode(unknown_view_schema))
    malformed_selector = deepcopy(expected_view_tree)
    malformed_selector["axis_bindings"][3]["selector"]["extra"] = True
    with pytest.raises(ValueError, match="display selector"):
        decode_view_spec(encode(malformed_selector))
    noncanonical_view = deepcopy(expected_view_tree)
    noncanonical_view["axis_bindings"].reverse()
    with pytest.raises(ValueError, match="non-canonical typed representation"):
        decode_view_spec(encode(noncanonical_view))
    noncanonical_document = deepcopy(expected_document_tree)
    noncanonical_document["datasets"].reverse()
    with pytest.raises(ValueError, match="non-canonical typed representation"):
        decode_figure_document(encode(noncanonical_document))
    unknown_document_schema = deepcopy(expected_document_tree)
    unknown_document_schema["schema"] = "zlc_frontend.UnknownFigure"
    with pytest.raises(ValueError, match="expected schema"):
        decode_figure_document(encode(unknown_document_schema))


def test_rect_f_product_layout_recovers_image_axes_before_repeat_mean():
    repeat = axis("repeat", REPEAT, 2)
    x = axis("x", SPATIAL_X, 2)
    y = axis("y", SPATIAL_Y, 3)
    layout = PointLayout.rect_f((2, 3))
    values = np.empty((2, layout.storage_size), dtype=float)
    for r in range(2):
        for p in range(layout.storage_size):
            ix, iy = layout.multi_index(p)
            values[r, p] = 100 * r + 10 * iy + ix
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(x, y),
        point_layout=layout,
    )
    view = suggest_view(block.schema, ViewIntent.IMAGE).spec
    image = only_series(evaluated_data(block, view)).data
    assert isinstance(image, EvaluatedImage)
    np.testing.assert_allclose(
        image.values,
        [[50, 51], [60, 61], [70, 71]],
    )
    np.testing.assert_array_equal(image.validity, np.ones((3, 2), dtype=bool))


def test_explicit_sparse_image_preserves_hole_as_invalid():
    repeat = axis("repeat", REPEAT, 1)
    x = axis("x", SPATIAL_X, 2)
    y = axis("y", SPATIAL_Y, 2)
    layout = PointLayout.explicit((2, 2), ((0, 0), (1, 0), (0, 1)))
    block = make_block(
        np.array([[1.0, 2.0, 3.0]]),
        repeat_axis=repeat,
        point_axes=(x, y),
        point_layout=layout,
    )
    image = only_series(
        evaluated_data(block, suggest_view(block.schema, ViewIntent.IMAGE).spec)
    ).data
    np.testing.assert_array_equal(image.values, [[1, 2], [3, 0]])
    np.testing.assert_array_equal(image.validity, [[True, True], [True, False]])


def test_roi_selection_does_not_invent_a_reducer_but_explicit_reduction_evaluates():
    frame = CoordinateFrameId("camera")
    repeat = axis("repeat", REPEAT, 2)
    point = axis("detuning", SCAN_POINT, 3)
    x = axis("x", SPATIAL_X, 2, frame=frame)
    y = axis("y", SPATIAL_Y, 2, frame=frame)
    values = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
    valid = np.ones_like(values, dtype=bool)
    valid[1, 1, 1, 1] = False
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((3,)),
        data_axes=(x, y),
        validity=ComponentValidity((x.axis_id, y.axis_id), valid),
        component_axes=(x.axis_id, y.axis_id),
    )
    roi = Selection.rectangle(
        x.axis_id,
        y.axis_id,
        0,
        1,
        0,
        1,
        coordinate_frame=frame,
    )
    suggestion = suggest_view(block.schema, ViewIntent.CURVE, roi)
    assert suggestion.status is SuggestionStatus.NEEDS_INPUT
    assert suggestion.reasons[0].code == "UNRESOLVED_INFORMATION_AXIS"
    explicit_view = ViewSpec(
        block.schema.fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(point.axis_id, AxisViewRole.X),
            AxisViewBinding(
                repeat.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            AxisViewBinding(
                x.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            AxisViewBinding(
                y.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
        ),
        (roi,),
    )
    validate_view_spec(block.schema, explicit_view)
    series = only_series(evaluated_data(block, explicit_view))
    assert isinstance(series.data, EvaluatedCurve)
    expected = []
    for p in range(3):
        selected = values[:, p].reshape(-1)
        selected_valid = valid[:, p].reshape(-1)
        expected.append(selected[selected_valid].mean())
    np.testing.assert_allclose(series.data.values, expected)
    assert set(series.reductions[0].axis_ids) == {
        repeat.axis_id,
        x.axis_id,
        y.axis_id,
    }
    assert series.reductions[0].minimum_contributors == 7
    assert series.reductions[0].maximum_contributors == 8


def test_latest_nonempty_is_one_global_repeat_not_per_point_gather():
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 2)
    block = make_block(
        np.array([[10.0, 11.0], [20.0, 21.0]]),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((2,)),
        validity=CellValidity(np.array([[True, True], [False, True]])),
    )
    view = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(repeat_mode=RepeatViewMode.LATEST),
    ).spec
    evaluated = evaluated_data(block, view)
    layer = evaluated.layers[0]
    assert layer.resolutions[0].selector == "LATEST_NONEMPTY"
    assert layer.resolutions[0].index == 1
    curve = layer.cells[0].series[0].data
    np.testing.assert_array_equal(curve.values, [20, 21])
    np.testing.assert_array_equal(curve.validity, [False, True])
    assert evaluated.inputs[0].ref.stream_generation == StreamGenerationId("generation-a")


def test_histogram_sample_axes_preserve_identity_and_invalid_drop_count():
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 1)
    site = axis("site", SITE, 2, coordinates=("A", "B"))
    values = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    valid = ComponentValidity(
        (site.axis_id,),
        np.array([[[True, False]], [[True, True]]]),
    )
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=(site,),
        validity=valid,
        component_axes=(site.axis_id,),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.HISTOGRAM,
        preferences=ViewPreferences(sample_axis_ids=(site.axis_id,)),
    )
    assert suggestion.status is SuggestionStatus.REVIEW_REQUIRED
    histogram = only_series(evaluated_data(block, suggestion.spec)).data
    assert isinstance(histogram, EvaluatedHistogram)
    np.testing.assert_array_equal(histogram.samples, [1, 3, 4])
    coords = {item.axis_id: item.coordinates for item in histogram.sample_coordinates}
    assert coords[repeat.axis_id] == (0, 1, 1)
    assert coords[site.axis_id] == ("A", "A", "B")
    assert histogram.dropped_count == 1


def test_curve_batch_produces_one_series_per_site_and_sum_is_display_only():
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 3)
    site = axis("site", SITE, 2)
    values = np.arange(12, dtype=float).reshape(2, 3, 2)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((3,)),
        data_axes=(site,),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(repeat_mode=RepeatViewMode.SUM),
    )
    evaluated = evaluated_data(block, suggestion.spec)
    cell = evaluated.layers[0].cells[0]
    assert len(cell.series) == 2
    for site_index, series in enumerate(cell.series):
        assert series.batch_address[0].axis_id == site.axis_id
        assert series.batch_address[0].index == site_index
        np.testing.assert_array_equal(
            series.data.values,
            values[:, :, site_index].sum(axis=0),
        )


def test_display_range_cardinality_allows_a_large_axis_to_batch_safely():
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 3)
    site = axis("site", SITE, 40)
    values = np.arange(2 * 3 * 40, dtype=float).reshape(2, 3, 40)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((3,)),
        data_axes=(site,),
    )
    selected_sites = Selection.index_range(site.axis_id, 10, 14)
    suggestion = suggest_view(block.schema, ViewIntent.CURVE, selected_sites)
    assert suggestion.status is SuggestionStatus.RESOLVED
    assert {
        binding.axis_id: binding.role for binding in suggestion.spec.axis_bindings
    }[site.axis_id] is AxisViewRole.BATCH
    validate_view_spec(block.schema, suggestion.spec)
    cell = evaluated_data(block, suggestion.spec).layers[0].cells[0]
    assert [series.batch_address[0].index for series in cell.series] == [10, 11, 12, 13]


def test_validate_rejects_mixed_joint_reducers_and_outside_fixed_selection():
    frame = CoordinateFrameId("camera")
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 3)
    x = axis("x", SPATIAL_X, 2, frame=frame)
    y = axis("y", SPATIAL_Y, 2, frame=frame)
    block = make_block(
        np.zeros((2, 3, 2, 2)),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((3,)),
        data_axes=(x, y),
    )
    roi = Selection.rectangle(
        x.axis_id, y.axis_id, 0, 1, 0, 1, coordinate_frame=frame
    )
    mixed_suggestion = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        roi,
        ViewPreferences(repeat_mode=RepeatViewMode.SUM),
    )
    assert mixed_suggestion.status is SuggestionStatus.NEEDS_INPUT
    assert mixed_suggestion.reasons[0].code == "UNRESOLVED_INFORMATION_AXIS"

    mean_spec = ViewSpec(
        block.schema.fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(point.axis_id, AxisViewRole.X),
            AxisViewBinding(
                repeat.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            AxisViewBinding(
                x.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            AxisViewBinding(
                y.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
        ),
        (roi,),
    )
    validate_view_spec(block.schema, mean_spec)
    mixed_spec = replace(
        mean_spec,
        axis_bindings=tuple(
            replace(
                binding,
                reduction=DisplayReduction(DisplayReductionMethod.SUM),
            )
            if binding.axis_id == repeat.axis_id
            else binding
            for binding in mean_spec.axis_bindings
        ),
    )
    with pytest.raises(ValueError, match="one common method"):
        validate_view_spec(block.schema, mixed_spec)

    selected_repeat = Selection.index_range(repeat.axis_id, 0, 1)
    selected_spec = suggest_view(
        make_block(
            np.zeros((2, 3)),
            repeat_axis=repeat,
            point_axes=(point,),
            point_layout=PointLayout.rect_c((3,)),
        ).schema,
        ViewIntent.CURVE,
        selected_repeat,
    ).spec
    outside = replace(
        selected_spec,
        axis_bindings=tuple(
            AxisViewBinding(
                repeat.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(1),
            )
            if binding.axis_id == repeat.axis_id
            else binding
            for binding in selected_spec.axis_bindings
        ),
    )
    with pytest.raises(IndexError, match="outside the display selection"):
        validate_view_spec(
            make_block(
                np.zeros((2, 3)),
                repeat_axis=repeat,
                point_axes=(point,),
                point_layout=PointLayout.rect_c((3,)),
            ).schema,
            outside,
        )


def test_display_sum_overflow_is_fail_closed():
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 1)
    values = np.array([[np.iinfo(np.int64).max], [1]], dtype=np.int64)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
    )
    view = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(repeat_mode=RepeatViewMode.SUM),
    ).spec
    with pytest.raises(OverflowError, match="integer SUM"):
        evaluated_data(block, view)


def test_evaluation_policy_cancels_and_rejects_before_materialization(monkeypatch):
    repeat = axis("repeat", REPEAT, 1)
    point = axis("point", SCAN_POINT, 3)
    block = make_block(
        np.arange(3.0).reshape(1, 3),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((3,)),
    )
    view = suggest_view(block.schema, ViewIntent.CURVE).spec
    doc, datasets = document(block, view)
    with pytest.raises(FigureEvaluationCancelled):
        FigureEvaluator().evaluate(doc, datasets, cancel_requested=lambda: True)
    with pytest.raises(FigureEvaluationDeadlineExceeded):
        FigureEvaluator().evaluate(
            doc, datasets, monotonic_deadline=time.monotonic() - 1.0
        )

    evaluator_module = importlib.import_module("zlc_frontend.figure.evaluate")

    def forbidden_extract(*args, **kwargs):
        raise AssertionError("policy must reject before materialization")

    monkeypatch.setattr(evaluator_module, "_extract", forbidden_extract)
    policy = FigureEvaluationPolicy(max_output_elements=2)
    with pytest.raises(FigureEvaluationLimitExceeded, match="output_elements"):
        FigureEvaluator(policy).evaluate(doc, datasets)

    large_repeat = axis("large-repeat", REPEAT, 1000)
    large_point = axis("large-point", SCAN_POINT, 1000)
    large_block = make_block(
        np.zeros((1000, 1000), dtype=np.float64),
        repeat_axis=large_repeat,
        point_axes=(large_point,),
        point_layout=PointLayout.rect_c((1000,)),
    )
    large_view = suggest_view(large_block.schema, ViewIntent.CURVE).spec
    large_doc, large_datasets = document(large_block, large_view)
    with pytest.raises(FigureEvaluationLimitExceeded, match="physical_rows"):
        FigureEvaluator().evaluate(large_doc, large_datasets)
    live_memory_policy = FigureEvaluationPolicy(
        max_physical_rows=2_000_000,
        max_reduction_contributions=2_000_000,
        max_live_nbytes=6_000_000,
    )
    with pytest.raises(FigureEvaluationLimitExceeded, match="live_nbytes"):
        FigureEvaluator(live_memory_policy).evaluate(large_doc, large_datasets)
    contribution_policy = FigureEvaluationPolicy(
        max_physical_rows=2_000_000,
        max_reduction_contributions=500_000,
        max_live_nbytes=512 * 1024 * 1024,
    )
    with pytest.raises(FigureEvaluationLimitExceeded, match="reduction_contributions"):
        FigureEvaluator(contribution_policy).evaluate(large_doc, large_datasets)


def test_real_36_by_32_grid_has_bounded_end_to_end_latency():
    repeat = axis("repeat", REPEAT, 4)
    point = axis("point", SCAN_POINT, 100)
    site = axis("site", SITE, 32)
    component = axis("component", COMPONENT, 36)
    values = np.arange(4 * 100 * 32 * 36, dtype=np.float32).reshape(4, 100, 32, 36)
    assert values.nbytes == 1_843_200
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((100,)),
        data_axes=(site, component),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(
            batch_axis_ids=(site.axis_id,), facet_axis_ids=(component.axis_id,)
        ),
    )
    started = time.perf_counter()
    evaluated = evaluated_data(block, suggestion.spec)
    elapsed = time.perf_counter() - started
    assert len(evaluated.layers[0].cells) == 36
    assert sum(len(cell.series) for cell in evaluated.layers[0].cells) == 36 * 32
    np.testing.assert_allclose(
        evaluated.layers[0].cells[0].series[0].data.values,
        values[:, :, 0, 0].mean(axis=0, dtype=np.float64),
    )
    np.testing.assert_allclose(
        evaluated.layers[0].cells[-1].series[-1].data.values,
        values[:, :, -1, -1].mean(axis=0, dtype=np.float64),
    )
    assert elapsed < 1.25


def test_public_evaluator_preserves_one_full_qcmos_frame_with_bounded_peak():
    repeat = AxisSpec(AxisId("camera-repeat"), "camera repeat", REPEAT, 1)
    point = AxisSpec(AxisId("camera-point"), "camera point", SCAN_POINT, 1)
    y = AxisSpec(AxisId("camera-y"), "camera y", SPATIAL_Y, 2304)
    x = AxisSpec(AxisId("camera-x"), "camera x", SPATIAL_X, 2304)
    values = np.zeros((1, 1, 2304, 2304), dtype=np.uint16)
    values[0, 0, 0, 0] = 11
    values[0, 0, -1, -1] = 22
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=(y, x),
    )
    view = suggest_view(block.schema, ViewIntent.IMAGE).spec
    doc, datasets = document(block, view)

    tracemalloc.start()
    with pytest.raises(FigureEvaluationLimitExceeded, match="live_nbytes"):
        FigureEvaluator(
            FigureEvaluationPolicy(max_live_nbytes=16 * 1024 * 1024)
        ).evaluate(doc, datasets)
    _, rejected_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert rejected_peak < 2 * 1024 * 1024

    tracemalloc.start()
    started = time.perf_counter()
    evaluated = FigureEvaluator().evaluate(doc, datasets)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    series = only_series(evaluated)
    image = series.data
    assert isinstance(image, EvaluatedImage)
    assert image.x_axis.axis_id == x.axis_id
    assert image.y_axis.axis_id == y.axis_id
    assert image.values.shape == (2304, 2304)
    assert image.values.dtype == np.dtype("<u2")
    assert (image.values[0, 0], image.values[-1, -1]) == (11, 22)
    assert image.validity.dtype == np.dtype(bool) and bool(np.all(image.validity))
    assert series.reductions[0].axis_ids == (repeat.axis_id,)
    assert series.reductions[0].minimum_contributors == 1
    assert series.reductions[0].maximum_contributors == 1
    assert not np.shares_memory(image.values, block.values)
    with pytest.raises(ValueError):
        image.values.setflags(write=True)
    with pytest.raises(ValueError):
        image.validity.setflags(write=True)
    retained = image.values.nbytes + image.validity.nbytes
    assert peak < 2.75 * retained
    assert elapsed < 1.0
    assert block.values.shape == (1, 1, 2304, 2304)
    assert block.values.dtype == np.dtype("<u2")
    assert (block.values[0, 0, 0, 0], block.values[0, 0, -1, -1]) == (11, 22)


@pytest.mark.parametrize("data_order", ("yx", "xy"))
def test_image_orientation_follows_axis_ids_not_trailing_shape(data_order):
    repeat = axis("orientation-repeat", REPEAT, 1)
    point = axis("orientation-point", SCAN_POINT, 1)
    y = axis("orientation-y", SPATIAL_Y, 2)
    x = axis("orientation-x", SPATIAL_X, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    data_axes = (y, x) if data_order == "yx" else (x, y)
    stored = expected if data_order == "yx" else expected.T
    block = make_block(
        stored.reshape((1, 1, *stored.shape)),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=data_axes,
    )
    image = only_series(
        evaluated_data(block, suggest_view(block.schema, ViewIntent.IMAGE).spec)
    ).data
    np.testing.assert_array_equal(image.values, expected)
    assert image.values.shape == (y.size, x.size)


def test_sparse_coordinate_image_selects_narrowest_axis_before_large_gather():
    frame = CoordinateFrameId("sparse-camera")
    repeat = axis("sparse-repeat", REPEAT, 1)
    point = axis("sparse-point", SCAN_POINT, 1)
    y_coordinates = tuple(
        index // 2 if index % 2 == 0 else 10_000 + index // 2
        for index in range(1024)
    )
    x_coordinates = (0, *(10_000 + index for index in range(1022)), 1)
    y = axis("sparse-y", SPATIAL_Y, 1024, coordinates=y_coordinates, frame=frame)
    x = axis("sparse-x", SPATIAL_X, 1024, coordinates=x_coordinates, frame=frame)
    values = np.arange(1024 * 1024, dtype=np.uint16).reshape(1, 1, 1024, 1024)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=(y, x),
    )
    selection = Selection.rectangle(
        x.axis_id,
        y.axis_id,
        0,
        1,
        0,
        511,
        coordinate_frame=frame,
    )
    view = suggest_view(block.schema, ViewIntent.IMAGE, selection).spec
    doc, datasets = document(block, view)
    tracemalloc.start()
    image = only_series(
        FigureEvaluator(
            FigureEvaluationPolicy(max_live_nbytes=512 * 1024)
        ).evaluate(doc, datasets)
    ).data
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    np.testing.assert_array_equal(
        image.values,
        block.values[0, 0, ::2][:, (0, -1)],
    )
    assert image.values.shape == (512, 2)
    assert peak < 512 * 1024


def test_single_repeat_image_keeps_component_axes_and_invalid_components():
    repeat = axis("component-repeat", REPEAT, 1)
    point = axis("component-point", SCAN_POINT, 1)
    component = axis("component", COMPONENT, 2)
    y = axis("component-y", SPATIAL_Y, 2)
    x = axis("component-x", SPATIAL_X, 3)
    values = np.arange(1, 13, dtype=np.uint16).reshape(1, 1, 2, 2, 3)
    valid = np.ones(values.shape, dtype=bool)
    valid[0, 0, 1, 1, 2] = False
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=(component, y, x),
        validity=ComponentValidity(
            (component.axis_id, y.axis_id, x.axis_id),
            valid,
        ),
        component_axes=(component.axis_id, y.axis_id, x.axis_id),
    )
    suggestion = suggest_view(block.schema, ViewIntent.IMAGE)
    assert suggestion.status is SuggestionStatus.RESOLVED
    evaluated = evaluated_data(block, suggestion.spec)
    assert len(evaluated.layers[0].cells) == 2
    for component_index, cell in enumerate(evaluated.layers[0].cells):
        assert cell.facet_address[0].axis_id == component.axis_id
        assert cell.facet_address[0].index == component_index
        image = cell.series[0].data
        expected = values[0, 0, component_index].copy()
        expected_valid = valid[0, 0, component_index]
        expected = np.where(expected_valid, expected, np.uint16(0))
        np.testing.assert_array_equal(image.values, expected)
        np.testing.assert_array_equal(image.validity, expected_valid)
        assert image.values.dtype == np.dtype("<u2")
    second_reduction = evaluated.layers[0].cells[1].series[0].reductions[0]
    assert second_reduction.minimum_contributors == 0
    assert second_reduction.maximum_contributors == 1


def test_multi_repeat_integer_mean_remains_canonical_and_fractional():
    repeat = axis("integer-repeat", REPEAT, 2)
    point = axis("integer-point", SCAN_POINT, 1)
    y = axis("integer-y", SPATIAL_Y, 1)
    x = axis("integer-x", SPATIAL_X, 2)
    values = np.array([[[[0, 2]]], [[[1, 3]]]], dtype=np.uint16)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=(y, x),
    )
    series = only_series(
        evaluated_data(block, suggest_view(block.schema, ViewIntent.IMAGE).spec)
    )
    assert series.data.values.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(series.data.values, [[0.5, 2.5]])
    assert series.reductions[0].minimum_contributors == 2
    assert series.reductions[0].maximum_contributors == 2


def test_explicit_current_point_admission_counts_visible_frame_not_rolling_history():
    repeat = axis("rolling-repeat", REPEAT, 1)
    point = axis("rolling-point", SCAN_POINT, 8)
    y = axis("rolling-y", SPATIAL_Y, 64)
    x = axis("rolling-x", SPATIAL_X, 64)
    values = np.empty((1, 8, 64, 64), dtype=np.uint16)
    for point_index in range(point.size):
        values[0, point_index] = point_index
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((point.size,)),
        data_axes=(y, x),
    )
    view = suggest_view(
        block.schema,
        ViewIntent.IMAGE,
        Selection.index(point.axis_id, 7),
    ).spec
    doc, datasets = document(block, view)
    evaluated = FigureEvaluator(
        FigureEvaluationPolicy(max_reduction_contributions=4_100)
    ).evaluate(doc, datasets)
    layer = evaluated.layers[0]
    assert layer.resolutions[0].axis_id == point.axis_id
    assert layer.resolutions[0].selector == "FIXED_INDEX"
    assert layer.resolutions[0].index == 7
    np.testing.assert_array_equal(only_series(evaluated).data.values, 7.0)


def test_evaluated_validity_is_exact_bool_and_complex_views_fail_closed():
    x_out = EvaluatedAxis(AxisId("dto-x"), "x", SPATIAL_X, None, (0,), (0,))
    y_out = EvaluatedAxis(AxisId("dto-y"), "y", SPATIAL_Y, None, (0,), (0,))
    with pytest.raises(TypeError, match="validity dtype must be bool"):
        EvaluatedImage(
            x_out,
            y_out,
            np.zeros((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.uint8),
        )
    with pytest.raises(TypeError, match="validity dtype must be bool"):
        EvaluatedCurve(
            x_out,
            None,
            np.zeros((1,), dtype=np.float64),
            np.array([np.nan]),
        )

    repeat = axis("complex-repeat", REPEAT, 1)
    point = axis("complex-point", SCAN_POINT, 1)
    y = axis("complex-y", SPATIAL_Y, 1)
    x = axis("complex-x", SPATIAL_X, 1)
    complex_block = make_block(
        np.ones((1, 1, 1, 1), dtype=np.complex128),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=(y, x),
    )
    suggestion = suggest_view(complex_block.schema, ViewIntent.IMAGE)
    assert suggestion.status is SuggestionStatus.NEEDS_INPUT
    assert suggestion.reasons[0].code == "CONTRACT_REJECTED"
    assert "complex-value projection" in suggestion.reasons[0].message

    real_block = make_block(
        np.ones((1, 1, 1, 1), dtype=np.float64),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
        data_axes=(y, x),
    )
    manual_complex_view = replace(
        suggest_view(real_block.schema, ViewIntent.IMAGE).spec,
        schema_fingerprint=complex_block.schema.fingerprint,
    )
    with pytest.raises(ValueError, match="complex-value projection"):
        validate_view_spec(complex_block.schema, manual_complex_view)


def test_500k_histogram_default_budget_rejects_before_materialization(monkeypatch):
    repeat = AxisSpec(AxisId("hist-repeat"), "hist repeat", REPEAT, 500_000)
    point = AxisSpec(AxisId("hist-point"), "hist point", SCAN_POINT, 1)
    block = make_block(
        np.zeros((500_000, 1), dtype=np.float64),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((1,)),
    )
    view = suggest_view(block.schema, ViewIntent.HISTOGRAM).spec
    doc, datasets = document(block, view)
    evaluator_module = importlib.import_module("zlc_frontend.figure.evaluate")

    def forbidden_extract(*args, **kwargs):
        raise AssertionError("histogram budget must reject before materialization")

    monkeypatch.setattr(evaluator_module, "_extract", forbidden_extract)
    started = time.perf_counter()
    with pytest.raises(FigureEvaluationLimitExceeded, match="histogram_samples"):
        FigureEvaluator().evaluate(doc, datasets)
    assert time.perf_counter() - started < 0.2


def test_ragged_explicit_reduction_never_pads_to_groups_times_max(monkeypatch):
    repeat = axis("repeat", REPEAT, 1)
    x = axis("x", SCAN_POINT, 257)
    reduced = axis("reduced", SPATIAL_X, 4096)
    mapping = tuple((0, index) for index in range(4096)) + tuple(
        (index, 0) for index in range(1, 257)
    )
    layout = PointLayout.explicit((257, 4096), mapping)
    values = np.concatenate(
        (
            np.arange(4096, dtype=np.float64),
            np.arange(1, 257, dtype=np.float64) + 10_000,
        )
    ).reshape(1, -1)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_axes=(x, reduced),
        point_layout=layout,
    )
    selection = Selection.index_range(reduced.axis_id, 0, reduced.size)
    suggestion = suggest_view(block.schema, ViewIntent.CURVE, selection)
    assert suggestion.status is SuggestionStatus.NEEDS_INPUT
    assert suggestion.reasons[0].code == "UNRESOLVED_INFORMATION_AXIS"
    explicit_view = ViewSpec(
        block.schema.fingerprint,
        ViewIntent.CURVE,
        (
            AxisViewBinding(x.axis_id, AxisViewRole.X),
            AxisViewBinding(
                repeat.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            AxisViewBinding(
                reduced.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
        ),
        (selection,),
    )
    validate_view_spec(block.schema, explicit_view)
    evaluator_module = importlib.import_module("zlc_frontend.figure.evaluate")
    original_zeros = evaluator_module.np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        normalized = tuple(shape) if isinstance(shape, tuple) else shape
        if normalized == (257, 4096):
            raise AssertionError("ragged groups must not be densified with padding")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(evaluator_module.np, "zeros", guarded_zeros)
    curve = only_series(evaluated_data(block, explicit_view)).data
    assert curve.values[0] == pytest.approx(np.arange(4096).mean())
    np.testing.assert_array_equal(curve.values[1:], np.arange(1, 257) + 10_000)
    np.testing.assert_array_equal(curve.validity, np.ones(257, dtype=bool))


def test_grouping_checks_deadline_every_4096_rows(monkeypatch):
    evaluator_module = importlib.import_module("zlc_frontend.figure.evaluate")
    reduced_axis = axis("reduced-rows", REPEAT, 8193)
    group_axis = axis("group", SCAN_POINT, 1)
    working = evaluator_module._WorkingData(
        np.ones(8193, dtype=np.float64),
        np.ones(8193, dtype=bool),
        (reduced_axis, group_axis),
        {
            reduced_axis.axis_id: np.arange(8193, dtype=np.int64),
            group_axis.axis_id: np.zeros(8193, dtype=np.int64),
        },
        (),
        {},
    )
    reduction = AxisViewBinding(
        reduced_axis.axis_id,
        AxisViewRole.REDUCED,
        reduction=DisplayReduction(DisplayReductionMethod.MEAN),
    )
    calls = 0

    def fake_monotonic():
        nonlocal calls
        calls += 1
        return 0.0 if calls < 3 else 2.0

    monkeypatch.setattr(evaluator_module.time, "monotonic", fake_monotonic)
    guard = evaluator_module._EvaluationGuard(
        FigureEvaluationPolicy(), None, 1.0
    )
    with pytest.raises(FigureEvaluationDeadlineExceeded):
        evaluator_module._reduce(working, (reduction,), guard)
    assert calls == 3


def test_evaluated_meter_rejects_non_numeric_public_values():
    assert EvaluatedMeter(np.float32(1.25), True).value == pytest.approx(1.25)
    assert EvaluatedMeter(np.bool_(True), True).value is True
    with pytest.raises(TypeError, match="numeric or boolean scalar"):
        EvaluatedMeter("1.25", True)
    with pytest.raises(TypeError, match="numeric or boolean scalar"):
        EvaluatedMeter([1.25], True)


def test_meter_latest_with_explicit_point_facets_yields_typed_scalars():
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 2)
    block = make_block(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        repeat_axis=repeat,
        point_axes=(point,),
        point_layout=PointLayout.rect_c((2,)),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.METER,
        preferences=ViewPreferences(facet_axis_ids=(point.axis_id,)),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    evaluated = evaluated_data(block, suggestion.spec)
    assert len(evaluated.layers[0].cells) == 2
    meters = [cell.series[0].data for cell in evaluated.layers[0].cells]
    assert all(isinstance(meter, EvaluatedMeter) for meter in meters)
    assert [meter.value for meter in meters] == [3.0, 4.0]


def test_figure_package_has_no_authority_transform_mutation_or_conversion_surface():
    root = Path(__file__).resolve().parents[1] / "zlc_frontend" / "figure"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    # Figure may narrowly consume a committed, selection-only transform so that
    # it displays the exact authority selection.  It must never mint, apply, or
    # convert one into a second authority surface.
    assert "commit_transform" not in source
    assert "apply_transform" not in source
    assert "DataTransformSpec" not in source
    assert "to_transform" not in source
    assert "expand_dataset_validity" not in source
