"""Current explicit-axis coverage contracts for the vendored plot core."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
)
from zlc_data.axis import REPEAT, SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, HistogramPlot
from zlc_plot._dataset_bridge import bridge_snapshot
from zlc_plot.data_view import DataView, DataViewError


def _snapshot(*, data_axes=(), points=None, values=None, repeat=2) -> OwnedSnapshot:
    points = {"point": [0.0, 1.0]} if points is None else points
    columns = tuple(
        PointColumn(
            AxisId(name),
            name,
            SPATIAL_X if name != "y" else SPATIAL_Y,
            PointColumn.NUMERIC,
            tuple(values),
        )
        for name, values in points.items()
    )
    point_table = PointTable(len(next(iter(points.values()))), columns)
    repeat_axis = AxisSpec(AxisId("coverage.repeat"), "repeat", REPEAT, repeat, tuple(range(repeat)))
    cell_schema = (
        ValueSchema(
            tuple(data_axes),
            ValidityContract.value(),
            np.dtype("<f8"),
            "1",
        )
        if data_axes
        else ValueSchema.scalar(np.dtype("<f8"), "1")
    )
    schema = DatasetSchema(
        repeat_axis,
        point_table,
        None,
        cell_schema,
    )
    array = (
        np.arange(np.prod(schema.physical_shape), dtype=np.float64).reshape(schema.physical_shape)
        if values is None
        else np.asarray(values, dtype=np.float64).reshape(schema.physical_shape)
    )
    block = DataBlock(
        BlockId("coverage"),
        DatasetRevision(0),
        array,
        VALID,
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("coverage-generation")), block)


def _data_axis(name: str, values: tuple[float, ...]) -> AxisSpec:
    return AxisSpec(AxisId(name), name, AxisRoleId("other"), len(values), values)


def test_curve_does_not_flatten_unreferenced_data_axis() -> None:
    snapshot = _snapshot(data_axes=(_data_axis("scan", (0.0, 1.0)),))
    with pytest.raises(DataViewError, match="scan"):
        DataView(bridge_snapshot(snapshot)).curve(AxisRef.point("point"))


def test_histogram_samples_are_explicit() -> None:
    scan = _data_axis("scan", (0.0, 1.0))
    snapshot = _snapshot(
        data_axes=(scan,),
        points={"point": [0.0]},
        repeat=2,
    )
    view = DataView(bridge_snapshot(snapshot))
    with pytest.raises(DataViewError, match="scan"):
        view.histogram(
            bins=4,
            samples=(AxisRef.repeat(), AxisRef.point_rows()),
        )
    histogram = view.histogram(
        bins=4,
        samples=(AxisRef.repeat(), AxisRef.point_rows(), AxisRef.data("scan")),
    )
    assert int(histogram.counts.sum()) == 4
    assert HistogramPlot().samples == (
        AxisRef.repeat(),
        AxisRef.point_rows(),
    )


def test_image_does_not_flatten_unreferenced_data_axis() -> None:
    x = _data_axis("x", (0.0, 1.0))
    y = _data_axis("y", (0.0, 1.0))
    scan = _data_axis("scan", (0.0, 1.0))
    snapshot = _snapshot(
        data_axes=(x, y, scan),
        points={"point": [0.0]},
        values=np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float64),
    )
    with pytest.raises(DataViewError, match="scan"):
        DataView(bridge_snapshot(snapshot)).image(AxisRef.data("x"), AxisRef.data("y"))


def test_plain_point_coordinates_need_declared_grid_topology() -> None:
    snapshot = _snapshot(
        points={"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]},
        repeat=1,
    )
    with pytest.raises(DataViewError, match="GridTopology"):
        DataView(bridge_snapshot(snapshot)).image(AxisRef.point("x"), AxisRef.point("y"))
