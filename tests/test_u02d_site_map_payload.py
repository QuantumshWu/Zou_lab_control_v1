from __future__ import annotations

from dataclasses import replace
import math
from typing import get_args

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DatasetRevision,
    DatasetRevisionRef,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    StreamGenerationId,
)
from zlc_frontend.figure import (
    DatasetId,
    EvaluatedAxis,
    EvaluatedImage,
    EvaluatedInput,
)
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.render import (
    CoherenceStamp,
    CurvePanelPayload,
    DisplayPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
    RasterBuffer,
    SiteMapPanelPayload,
    SourceIdentity,
)


def _input(name: str, revision: int = 1) -> EvaluatedInput:
    schema = ("a" if name == "occupancy" else "b") * 64
    return EvaluatedInput(
        DatasetId(name),
        DatasetRevisionRef(
            BlockId(f"{name}-block"),
            StreamGenerationId(f"{name}-generation"),
            schema,
            DatasetRevision(revision),
        ),
    )


def _background(
    evaluated_input: EvaluatedInput,
    *,
    frame: CoordinateFrameId,
    revision: int = 7,
    x_coordinates: tuple[float, ...] = (20.0, 21.0, 22.0),
    y_coordinates: tuple[float, ...] = (10.0, 11.0),
) -> ImagePanelPayload:
    y_axis = AxisSpec(
        AxisId("camera.y"),
        "y",
        SPATIAL_Y,
        len(y_coordinates),
        y_coordinates,
        "pixel",
        frame,
    )
    x_axis = AxisSpec(
        AxisId("camera.x"),
        "x",
        SPATIAL_X,
        len(x_coordinates),
        x_coordinates,
        "pixel",
        frame,
    )
    shape = (len(y_coordinates), len(x_coordinates))
    image = EvaluatedImage(
        EvaluatedAxis(
            x_axis.axis_id,
            "x",
            SPATIAL_X,
            "pixel",
            tuple(range(len(x_coordinates))),
            x_axis.coordinates,
        ),
        EvaluatedAxis(
            y_axis.axis_id,
            "y",
            SPATIAL_Y,
            "pixel",
            tuple(range(len(y_coordinates))),
            y_axis.coordinates,
        ),
        np.arange(math.prod(shape), dtype=np.uint16).reshape(shape),
        np.ones(shape, dtype=bool),
    )
    histogram = (1,) * math.prod(shape) + (0,) * (255 - math.prod(shape))
    return ImagePanelPayload(
        image,
        evaluated_input,
        ImageViewportTransform((y_axis, x_axis), viewport_revision=revision),
        (0.0, float(math.prod(shape) - 1)),
        histogram,
        tuple(0xFF000000 | index for index in range(256)),
        (0.0, float(max(1, math.prod(shape) - 1))),
    )


def _payload(
    *,
    occupancy_input: EvaluatedInput | None = None,
    background_input: EvaluatedInput | None = None,
) -> SiteMapPanelPayload:
    occupancy_input = occupancy_input or _input("occupancy")
    background_input = background_input or _input("background")
    frame = CoordinateFrameId("camera-frame")
    centers = np.asarray(
        ((20.25, 10.25), (21.75, 10.75)),
        dtype=np.float32,
    )
    return SiteMapPanelPayload(
        _background(background_input, frame=frame),
        occupancy_input,
        AxisSpec(AxisId("readout.site"), "site", SITE, 2, ("left", "right")),
        frame,
        centers,
        np.asarray((True, False), dtype=bool),
        np.asarray((True, False), dtype=bool),
        "calibration:sha256:0123",
        "repeat=0;point=2",
    )


def _stamp(
    payload: SiteMapPanelPayload,
    *,
    inputs: tuple[EvaluatedInput, ...] | None = None,
    panel_revision: int | None = None,
) -> CoherenceStamp:
    return CoherenceStamp(
        "run-1",
        "epoch-1",
        "occupancy-camera-cell",
        "c" * 64,
        payload.join_key_digest,
        inputs
        or (payload.occupancy_input, payload.background.evaluated_input),
        (
            PanelPresentationIdentity(
                "sites",
                "sites-document",
                2,
                3,
                payload.background.viewport.viewport_revision
                if panel_revision is None
                else panel_revision,
            ),
        ),
    )


def _source(value: EvaluatedInput) -> SourceIdentity:
    return SourceIdentity(
        value.dataset_id,
        value.ref.block_id,
        value.ref.stream_generation,
        value.ref.schema_fingerprint,
    )


def _raster(*, width: int = 3, height: int = 2) -> RasterBuffer:
    return RasterBuffer(
        width,
        height,
        width,
        PixelFormat.INDEXED8,
        bytes(range(1, width * height + 1)),
    )


def test_site_map_remains_a_closed_payload_and_owns_exact_site_arrays() -> None:
    centers = np.asarray(((20.25, 10.25), (21.75, 10.75)), dtype=np.float32)
    occupied = np.asarray((True, False), dtype=bool)
    validity = np.asarray((True, False), dtype=bool)
    base = _payload()
    payload = replace(
        base,
        centers_xy=centers,
        occupied=occupied,
        site_validity=validity,
    )

    assert get_args(DisplayPayload) == (
        ImagePanelPayload,
        CurvePanelPayload,
        HistogramPanelPayload,
        SiteMapPanelPayload,
    )
    assert payload.centers_xy.dtype == np.dtype("<f8")
    assert payload.centers_xy.shape == (2, 2)
    assert payload.full_normalized_centers_xy.shape == (2, 2)
    assert not payload.full_normalized_centers_xy.flags.writeable
    assert payload.visible_ring_span[0] > 0.0
    assert payload.visible_ring_span[1] > 0.0
    assert payload.occupied.dtype == np.dtype(bool)
    assert payload.site_validity.dtype == np.dtype(bool)
    assert payload.calibration_identity == "calibration:sha256:0123"
    assert payload.cell_identity == "repeat=0;point=2"

    centers[:] = -1
    occupied[:] = False
    validity[:] = True
    assert payload.centers_xy.tolist() == [[20.25, 10.25], [21.75, 10.75]]
    assert payload.occupied.tolist() == [True, False]
    assert payload.site_validity.tolist() == [True, False]
    for array in (payload.centers_xy, payload.occupied, payload.site_validity):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_site_map_payload_fails_closed_on_axis_geometry_validity_and_identity() -> None:
    payload = _payload()
    with pytest.raises(ValueError, match="shape"):
        replace(payload, centers_xy=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        replace(payload, centers_xy=np.asarray(((0.0, 0.0), (np.nan, 1.0))))
    with pytest.raises(ValueError, match="cannot be painted"):
        replace(
            payload,
            centers_xy=np.asarray(((20.25, 10.25), (100.0, 10.75))),
        )
    with pytest.raises(ValueError, match="cannot be painted"):
        replace(
            payload,
            background=_background(
                payload.background.evaluated_input,
                frame=payload.coordinate_frame,
                x_coordinates=(20.0,),
            ),
            centers_xy=np.asarray(((20.0, 10.25), (20.0, 10.75))),
        )
    with pytest.raises(TypeError, match="bool dtype"):
        replace(payload, occupied=np.asarray((1, 0), dtype=np.uint8))
    with pytest.raises(TypeError, match="real numeric"):
        replace(payload, centers_xy=np.asarray((("1", "2"), ("3", "4"))))
    with pytest.raises(ValueError, match="invalid site"):
        replace(
            payload,
            occupied=np.asarray((True, True), dtype=bool),
            site_validity=np.asarray((True, False), dtype=bool),
        )
    with pytest.raises(ValueError, match="role SITE"):
        replace(
            payload,
            site_axis=AxisSpec(AxisId("point"), "point", SCAN_POINT, 2),
        )
    with pytest.raises(ValueError, match="coordinate_frame differs"):
        replace(payload, coordinate_frame=CoordinateFrameId("another-camera"))
    with pytest.raises(ValueError, match="canonical non-empty text"):
        replace(payload, calibration_identity=" guessed ")
    with pytest.raises(ValueError, match="canonical non-empty text"):
        replace(payload, cell_identity=" guessed ")
    with pytest.raises(ValueError, match="distinct dataset ids"):
        _payload(
            occupancy_input=_input("same"),
            background_input=_input("same"),
        )


def test_panel_frame_binds_site_source_and_both_exact_coherence_refs() -> None:
    payload = _payload()
    stamp = _stamp(payload)
    panel = PanelFrame(
        "sites",
        "readout-cell",
        _source(payload.occupancy_input),
        stamp,
        _raster(),
        payload,
    )
    assert panel.display_payload is payload

    changed_calibration = replace(payload, calibration_identity="calibration:other")
    with pytest.raises(ValueError, match="coherence digest"):
        PanelFrame(
            "sites",
            "readout-cell",
            _source(changed_calibration.occupancy_input),
            stamp,
            _raster(),
            changed_calibration,
        )
    changed_cell = replace(payload, cell_identity="repeat=1;point=2")
    with pytest.raises(ValueError, match="coherence digest"):
        PanelFrame(
            "sites",
            "readout-cell",
            _source(changed_cell.occupancy_input),
            stamp,
            _raster(),
            changed_cell,
        )

    with pytest.raises(ValueError, match="frozen coherence input"):
        PanelFrame(
            "sites",
            "readout-cell",
            _source(payload.background.evaluated_input),
            stamp,
            _raster(),
            payload,
        )

    wrong_block_source = SourceIdentity(
        payload.occupancy_input.dataset_id,
        BlockId("wrong-block"),
        payload.occupancy_input.ref.stream_generation,
        payload.occupancy_input.ref.schema_fingerprint,
    )
    with pytest.raises(ValueError, match="source identity"):
        PanelFrame(
            "sites",
            "readout-cell",
            wrong_block_source,
            stamp,
            _raster(),
            payload,
        )

    missing_background = _stamp(payload, inputs=(payload.occupancy_input,))
    with pytest.raises(ValueError, match="background is absent"):
        PanelFrame(
            "sites",
            "readout-cell",
            _source(payload.occupancy_input),
            missing_background,
            _raster(),
            payload,
        )

    stale_background = _input("background", revision=2)
    with pytest.raises(ValueError, match="background differs"):
        PanelFrame(
            "sites",
            "readout-cell",
            _source(payload.occupancy_input),
            _stamp(payload, inputs=(payload.occupancy_input, stale_background)),
            _raster(),
            payload,
        )

    stale_occupancy = _input("occupancy", revision=2)
    with pytest.raises(ValueError, match="frozen coherence input"):
        PanelFrame(
            "sites",
            "readout-cell",
            _source(payload.occupancy_input),
            _stamp(
                payload,
                inputs=(stale_occupancy, payload.background.evaluated_input),
            ),
            _raster(),
            payload,
        )


def test_panel_frame_uses_background_raster_geometry_and_viewport_revision() -> None:
    payload = _payload()
    source = _source(payload.occupancy_input)
    with pytest.raises(ValueError, match="INDEXED8"):
        PanelFrame(
            "sites",
            "readout-cell",
            source,
            _stamp(payload),
            RasterBuffer(3, 2, 12, PixelFormat.RGBA8888, bytes(24)),
            payload,
        )
    with pytest.raises(ValueError, match="geometry"):
        PanelFrame(
            "sites",
            "readout-cell",
            source,
            _stamp(payload),
            _raster(width=2, height=2),
            payload,
        )
    with pytest.raises(ValueError, match="payload revision"):
        PanelFrame(
            "sites",
            "readout-cell",
            source,
            _stamp(payload, panel_revision=6),
            _raster(),
            payload,
        )
