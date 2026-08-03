"""Narrow characterization of the single zlc_plot data and session owner."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_plot import (
    AxisRef,
    ControlKind,
    CurvePlot,
    HistogramPlot,
    ImagePlot,
    NumericRange,
    PlotKind,
    PlotSession,
    PulseAnalogTrace,
    PulseDacScanSegment,
    PulseTimelineData,
    PulseTimelinePlot,
    RasterPlotHost,
    RectangleRange,
    Reduction,
    RollingPlot,
    SelectorKind,
    plot_spec_controls,
    plot_spec_from_tree,
    plot_spec_to_tree,
    resolve_plot_spec,
)
from zlc_plot._dataset_bridge import bridge_snapshot
from zlc_plot.fit import FitEngine, FitOptions, RegularImageFitInput


def _scan_snapshot(
    revision: int,
    values: np.ndarray,
    *,
    schema: DatasetSchema | None = None,
) -> OwnedSnapshot:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if schema is None:
        repeat = AxisSpec(AxisId("scan.repeat"), "repeat", REPEAT, 1, (0,))
        scan = PointColumn(
            AxisId("scan.detuning"),
            "detuning",
            SCAN_POINT,
            PointColumn.NUMERIC,
            tuple(np.linspace(-2.0, 2.0, array.size)),
            "MHz",
        )
        schema = DatasetSchema(
            repeat,
            PointTable(array.size, (scan,)),
            None,
            ValueSchema.scalar(np.dtype("<f8"), "count"),
        )
    block = DataBlock(
        BlockId(f"scan-{revision}"),
        DatasetRevision(revision),
        array.reshape(schema.physical_shape),
        VALID,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId("scan-generation")),
        block,
    )


def _image_snapshot(
    revision: int = 5,
    *,
    height: int = 8,
    width: int = 10,
    dtype: np.dtype = np.dtype("|u1"),
) -> OwnedSnapshot:
    y = AxisSpec(
        AxisId("camera.y"),
        "camera y",
        SPATIAL_Y,
        height,
        tuple(range(height)),
        "pixel",
    )
    x = AxisSpec(
        AxisId("camera.x"),
        "camera x",
        SPATIAL_X,
        width,
        tuple(range(width)),
        "pixel",
    )
    repeat = AxisSpec(AxisId("camera.repeat"), "repeat", REPEAT, 1, (0,))
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema(
            (y, x),
            ValidityContract.components(y.axis_id, x.axis_id),
            dtype,
            "count",
        ),
    )
    values = np.arange(height * width, dtype=dtype).reshape(schema.physical_shape)
    block = DataBlock(
        BlockId(f"camera-{revision}"),
        DatasetRevision(revision),
        values,
        VALID,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId("camera-generation")),
        block,
    )


def _derived_ref(
    source: OwnedSnapshot,
    name: str,
    schema: DatasetSchema,
) -> DatasetRevisionRef:
    return DatasetRevisionRef(
        BlockId(name),
        StreamGenerationId(f"{name}-generation"),
        schema.fingerprint,
        source.ref.revision,
    )


def test_private_data_bridge_preserves_shape_dtype_and_readonly_storage() -> None:
    snapshot = _image_snapshot(height=32, width=48, dtype=np.dtype("<u2"))

    bridged = bridge_snapshot(snapshot)

    assert bridged.shape == (1, 1, 32, 48)
    assert bridged.values.dtype == np.dtype("<u2")
    assert np.shares_memory(bridged.values, snapshot.block.values)
    assert not bridged.values.flags.writeable
    assert not bridged.validity.flags.writeable
    assert bridged.schema.source is snapshot.block.schema


def test_live_session_accepts_equal_rebuilt_schema_but_not_new_generation() -> None:
    initial = _image_snapshot(5)
    rebuilt = _image_snapshot(6)
    assert rebuilt.block.schema == initial.block.schema
    assert rebuilt.block.schema is not initial.block.schema
    session = PlotSession(
        initial,
        ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y")),
    )
    try:
        session.update_data(rebuilt)
        different_generation = OwnedSnapshot(
            rebuilt.block.ref(StreamGenerationId("another-camera-generation")),
            rebuilt.block,
        )
        with pytest.raises(ValueError, match="authoritative DatasetSchema"):
            session.update_data(different_generation)
    finally:
        session.close()


def test_plot_spec_authoring_uses_physical_axes_and_owner_labels() -> None:
    repeat = AxisSpec(AxisId("author.repeat"), "repeat", REPEAT, 1, (0,))
    x = PointColumn(
        AxisId("author.x"),
        "x",
        SPATIAL_X,
        PointColumn.NUMERIC,
        (0, 0, 1, 1),
    )
    y = PointColumn(
        AxisId("author.y"),
        "y",
        SPATIAL_Y,
        PointColumn.NUMERIC,
        (0, 1, 0, 1),
    )
    plain = DatasetSchema(
        repeat,
        PointTable(4, (x, y)),
        None,
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )

    controls = plot_spec_controls(plain, PlotKind.IMAGE)
    by_name = {control.name: control for control in controls}
    assert by_name["x"].kind is ControlKind.CHOICE
    assert by_name["x"].value is None
    assert "Point · author.x" in tuple(map(str, by_name["x"].choices))
    assert str(Reduction.MEAN) == "Mean"
    with pytest.raises(ValueError, match="GridTopology"):
        resolve_plot_spec(
            plain,
            PlotKind.IMAGE,
            {
                "x": AxisRef.point("author.x"),
                "y": AxisRef.point("author.y"),
                "reduction": Reduction.MEAN,
            },
        )

    topology = GridTopology(
        (x.coordinate_id, y.coordinate_id),
        ((0, 1), (0, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    grid = DatasetSchema(repeat, plain.point_table, topology, plain.cell_schema)
    authored = resolve_plot_spec(
        grid,
        PlotKind.IMAGE,
        {
            "x": AxisRef.point_dimension("author.x"),
            "y": AxisRef.point_dimension("author.y"),
            "reduction": Reduction.MEAN,
        },
    )
    assert plot_spec_from_tree(plot_spec_to_tree(authored)) == authored


def test_replace_spec_resize_and_dpr_keep_one_figure_owner() -> None:
    snapshot = _scan_snapshot(1, np.asarray((1.0, 2.0, 3.0, 4.0)))
    session = PlotSession(
        snapshot,
        CurvePlot(AxisRef.point("scan.detuning")),
        size="2x2",
    )
    try:
        figure = session._renderer.figure
        session.replace_spec(HistogramPlot(samples=(AxisRef.point_rows(),)))
        session.set_size("4x2")
        session.set_device_pixel_ratio(1.5)

        assert session._renderer.figure is figure
        assert session.spec == HistogramPlot(samples=(AxisRef.point_rows(),))
        assert session.rgba().shape == (*session.surface_plan.raster_size[::-1], 4)
    finally:
        session.close()


def test_image_selector_publications_keep_dtype_and_exact_revision() -> None:
    snapshot = _image_snapshot()
    session = PlotSession(
        snapshot,
        ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y")),
    )
    events = []
    session.subscribe_selection(events.append)
    x_range = NumericRange(2.0, 5.0)
    y_range = NumericRange(1.0, 4.0)
    area = RectangleRange(x_range, y_range)
    try:
        session.set_area_selector(x_range, y_range, display=False)
        session.commit_selector(SelectorKind.AREA, area, display=False)
        selected = events[-1].data
        assert selected is not None
        assert selected.source_revisions == (snapshot.ref.revision.value,)
        roi = selected.materialize(
            reference_for=lambda schema: _derived_ref(snapshot, "area-roi", schema)
        )
        assert roi.block.values.shape == (1, 1, 4, 4)
        assert roi.block.values.dtype == snapshot.block.values.dtype

        session.set_crosshair_selector(3.0, 2.0, display=False)
        session.commit_selector(
            SelectorKind.CROSSHAIR,
            session.selector_state(SelectorKind.CROSSHAIR).value,
            display=False,
        )
        cross = events[-1].data
        assert cross is not None and cross.selected_value is not None
        np.testing.assert_array_equal(cross.selected_value.values, np.asarray((23,), dtype=np.uint8))
        assert cross.source_revisions == (snapshot.ref.revision.value,)
    finally:
        session.close()


def test_regular_image_fit_selection_is_zero_copy_and_has_no_coordinate_mesh() -> None:
    snapshot = _image_snapshot(height=64, width=96, dtype=np.dtype("<u2"))
    session = PlotSession(
        snapshot,
        ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y")),
    )
    try:
        selection = session.fit_selection("radial_gaussian_center")
        regular = selection.regular_image

        assert regular is not None
        assert regular.observations.shape == (64, 96)
        assert regular.observations.dtype == np.dtype("<u2")
        assert np.shares_memory(regular.observations, snapshot.block.values)
        assert regular.x_coordinates.shape == (96,)
        assert regular.y_coordinates.shape == (64,)
        assert regular.valid_mask is None
        assert regular.selected_indices is None
    finally:
        session.close()


def test_2304_square_regular_image_fit_keeps_compact_diagnostics() -> None:
    size = 2304
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    x_basis = np.exp(-((x - 0.13) ** 2) / 0.28**2)
    y_basis = np.exp(-((y + 0.09) ** 2) / 0.28**2)
    image = np.rint(
        300.0 + 3200.0 * np.multiply.outer(y_basis, x_basis)
    ).astype(np.uint16)
    regular = RegularImageFitInput(x, y, image)

    tracemalloc.start()
    try:
        result = FitEngine().fit(
            "radial_gaussian_center",
            regular,
            data_revision=7,
            initial={
                "amplitude": 3200.0,
                "offset": 300.0,
                "one_over_e_radius": 0.28,
                "center_x": 0.13,
                "center_y": -0.09,
            },
            options=FitOptions(max_nfev=20),
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.success
    assert result.observation_count == image.size
    assert result.fitted_values is None
    assert result.residuals is None
    assert result.selected_indices is None
    assert np.shares_memory(regular.observations, image)
    assert regular.observations.dtype == np.dtype("<u2")
    assert peak < 24 * 1024 * 1024


def test_rolling_history_is_plot_private_and_selection_keeps_all_revisions() -> None:
    schema = _scan_snapshot(1, np.asarray((10.0,))).block.schema
    first = _scan_snapshot(1, np.asarray((10.0,)), schema=schema)
    session = PlotSession(first, RollingPlot(), parameters={"window": 3})
    events = []
    session.subscribe_selection(events.append)
    try:
        for revision, value in ((2, 20.0), (3, 30.0), (4, 40.0)):
            session.update_data(_scan_snapshot(revision, np.asarray((value,)), schema=schema))

        assert session._payload.source_revisions == (2, 3, 4)
        assert schema.physical_shape == (1, 1, 1)
        selected_range = NumericRange(-2.0, 0.0)
        session.set_x_selector(-2.0, 0.0, display=False)
        session.commit_selector(
            SelectorKind.X_RANGE,
            selected_range,
            display=False,
        )
        selected = events[-1].data
        assert selected is not None
        assert selected.source_revisions == (2, 3, 4)
        materialized = selected.materialize(
            reference_for=lambda output_schema: _derived_ref(
                _scan_snapshot(4, np.asarray((40.0,)), schema=schema),
                "rolling-range",
                output_schema,
            )
        )
        assert materialized.block.values.shape == (1, 3, 1)
        np.testing.assert_array_equal(
            materialized.block.values.reshape(-1),
            np.asarray((20.0, 30.0, 40.0)),
        )
    finally:
        session.close()


def test_raster_front_freezes_exact_rolling_source_revisions() -> None:
    schema = _scan_snapshot(1, np.asarray((10.0,))).block.schema
    first = _scan_snapshot(1, np.asarray((10.0,)), schema=schema)
    host = RasterPlotHost.from_plot(
        first,
        RollingPlot(),
        parameters={"window": 3},
    )
    try:
        assert host.wait_for_front(timeout=5.0).source_revisions == (1,)
        fronts = []
        for revision, value in ((2, 20.0), (3, 30.0), (4, 40.0)):
            operation = host.update_data(
                _scan_snapshot(revision, np.asarray((value,)), schema=schema)
            ).result(timeout=5.0)
            fronts.append(operation.front)
        assert fronts[-1].source_revisions == (2, 3, 4)
        assert fronts[-1].identity.data_revision == 4
    finally:
        host.close(timeout=5.0)


def test_dac_only_pulse_timeline_is_a_valid_public_plot() -> None:
    timeline = PulseTimelineData(
        (),
        (),
        total_duration=2.0,
        analog_traces=(
            PulseAnalogTrace("dac0", "DAC 0", -1.0, 1.0, (0.0, 1.0, 2.0), (0.0, 0.5)),
        ),
        scan_dac_segments=(
            PulseDacScanSegment("dac0", 0.5, 1.5, 0.25, 1),
        ),
    )
    session = PlotSession(timeline, PulseTimelinePlot())
    try:
        assert session.rgba().shape == (*session.surface_plan.raster_size[::-1], 4)
    finally:
        session.close()
