"""Shared immutable panel frames for current Figure surface tests.

These builders own only evaluator/render fixtures.  Tests drive the resulting
fronts through the production host that they are exercising; no test module is
used as a fixture library and no interaction owner is recreated here.
"""

from __future__ import annotations

import numpy as np


def evaluated_input(dataset_name: str, sequence: int):
    from zlc_data import (
        BlockId,
        DatasetRevision,
        DatasetRevisionRef,
        StreamGenerationId,
    )
    from zlc_frontend.figure import DatasetId, EvaluatedInput

    schema = ("a" if dataset_name == "curve" else "b") * 64
    dataset_id = DatasetId(dataset_name)
    ref = DatasetRevisionRef(
        BlockId(f"{dataset_name}-block"),
        StreamGenerationId(f"{dataset_name}-generation"),
        schema,
        DatasetRevision(sequence + 1),
    )
    return EvaluatedInput(dataset_id, ref)


def curve_panel(
    sequence: int,
    *,
    display_revision: int = 0,
    offset: float = 0.0,
):
    from zlc_data import AxisId, AxisSourceRef, MONITOR_HISTORY
    from zlc_frontend.curve_display import CurveViewportTransform
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedCurve, EvaluatedSeries
    from zlc_frontend.render import (
        CoherenceStamp,
        CurvePanelPayload,
        PanelFrame,
        RasterBuffer,
        SourceIdentity,
    )

    input_identity = evaluated_input("curve", sequence)
    axis = EvaluatedAxis(
        AxisSourceRef.tensor(AxisId("monitor.history")),
        "Shots ago",
        MONITOR_HISTORY,
        "ms",
        (0, 1, 2, 3),
        (0.0, 1.0, 2.0, 3.0),
    )
    first = EvaluatedCurve(
        axis,
        "count",
        np.asarray((0.0, 1.0, 2.0, 3.0)) + offset,
        np.asarray((True, True, False, True)),
    )
    second = EvaluatedCurve(
        axis,
        "count",
        np.asarray((3.0, 2.0, 1.0, 0.0)) + offset,
        np.asarray((True, True, True, True)),
    )
    viewport = CurveViewportTransform(
        axis,
        display_revision,
        (0.20, 0.10, 0.80, 0.90),
        (-0.5, 3.5),
        (-1.0 + offset, 4.0 + offset),
        (-0.15, 3.15),
    )
    payload = CurvePanelPayload(
        input_identity,
        viewport,
        (EvaluatedSeries((), first), EvaluatedSeries((), second)),
        ("site 0", "site 1"),
    )
    stamp = CoherenceStamp((input_identity,))
    source = SourceIdentity(
        input_identity.dataset_id,
        input_identity.ref.block_id,
        input_identity.ref.stream_generation,
        input_identity.ref.schema_fingerprint,
    )
    raster = RasterBuffer(
        200,
        100,
        bytes((20, 30, 40, 255)) * (200 * 100),
    )
    return PanelFrame("curve", "curve", source, stamp, raster, payload)


def image_panel(sequence: int, *, viewport_revision: int = 0):
    from zlc_data import (
        AxisId,
        AxisSourceRef,
        AxisSpec,
        CoordinateFrameId,
        SPATIAL_X,
        SPATIAL_Y,
    )
    from zlc_frontend.figure import EvaluatedAxis, EvaluatedImage
    from zlc_frontend.image_display import ImageColormap
    from zlc_frontend.image_view import ImageViewportTransform
    from zlc_frontend.render import (
        CoherenceStamp,
        ImagePanelPayload,
        ImagePanelRasterGeometry,
        PanelFrame,
        RasterBuffer,
        SourceIdentity,
    )

    input_identity = evaluated_input("image", sequence)
    frame = CoordinateFrameId("camera")
    y_spec = AxisSpec(
        AxisId("camera.y"),
        "y",
        SPATIAL_Y,
        2,
        (0.0, 1.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    x_spec = AxisSpec(
        AxisId("camera.x"),
        "x",
        SPATIAL_X,
        2,
        (0.0, 1.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    viewport = ImageViewportTransform(
        (y_spec, x_spec), viewport_revision=viewport_revision
    )
    x_axis = EvaluatedAxis(
        AxisSourceRef.tensor(x_spec.axis_id),
        "x",
        SPATIAL_X,
        "pixel",
        (0, 1),
        (0.0, 1.0),
    )
    y_axis = EvaluatedAxis(
        AxisSourceRef.tensor(y_spec.axis_id),
        "y",
        SPATIAL_Y,
        "pixel",
        (0, 1),
        (0.0, 1.0),
    )
    image = EvaluatedImage(
        x_axis,
        y_axis,
        np.asarray(((1.0, 2.0), (3.0, 4.0 + sequence))),
        np.ones((2, 2), dtype=bool),
    )
    payload = ImagePanelPayload(
        image=image,
        evaluated_input=input_identity,
        viewport=viewport,
        data_range=(1.0, 4.0 + sequence),
        colormap=ImageColormap.GRAY,
        color_limits=(0.0, 10.0),
        raster_geometry=ImagePanelRasterGeometry(
            (0.10, 0.10, 0.65, 0.90),
            (0.70, 0.10, 0.82, 0.90),
            (0.87, 0.10, 0.92, 0.90),
        ),
    )
    stamp = CoherenceStamp((input_identity,))
    source = SourceIdentity(
        input_identity.dataset_id,
        input_identity.ref.block_id,
        input_identity.ref.stream_generation,
        input_identity.ref.schema_fingerprint,
    )
    raster = RasterBuffer(2, 2, bytes((1, 2, 3, 255)) * 4)
    return PanelFrame("image", "image", source, stamp, raster, payload)


__all__ = ["curve_panel", "evaluated_input", "image_panel"]
