"""Narrow regressions for exact worker-raster hand-off semantics."""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _meter_figure():
    """Build the smallest real faceted METER figure under current data contracts."""

    from zlc_data import (
        REPEAT,
        SITE,
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
    from zlc_frontend import DataFigure
    from zlc_frontend.figure import (
        DatasetDescriptor,
        DatasetId,
        FigureDocument,
        FigureLayer,
        ResolvedDataset,
        ResolvedDatasetMap,
        SuggestionStatus,
        ViewIntent,
        ViewPreferences,
        suggest_view,
    )

    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
    site = AxisSpec(AxisId("site"), "site", SITE, 3, ("A", "B", "C"))
    values = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    valid = np.ones(values.shape, dtype=bool)
    schema = DatasetSchema(
        repeat,
        (),
        PointLayout.rect_c(()),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
            "count",
        ),
    )
    block = DataBlock(
        BlockId("meter-block"),
        DatasetRevision(7),
        values,
        DatasetComponentValidity((site.axis_id,), valid),
        schema,
    )
    dataset_id = DatasetId("meter-dataset")
    datasets = ResolvedDatasetMap(
        (
            ResolvedDataset(
                dataset_id,
                OwnedSnapshot(
                    block.ref(StreamGenerationId("meter-generation")),
                    block,
                ),
            ),
        )
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.METER,
        preferences=ViewPreferences(facet_axis_ids=(site.axis_id,)),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    document = FigureDocument(
        "meter-grid",
        2,
        (DatasetDescriptor(dataset_id, "Occupancy", schema.fingerprint),),
        (FigureLayer("meter-layer", dataset_id, suggestion.spec),),
    )
    return DataFigure(document, datasets)


def test_image_cache_is_bound_to_the_exact_evaluated_projection() -> None:
    from test_u02c_qt_curve_interaction import _image_panel
    from zlc_frontend.figure import EvaluatedImage, EvaluatedProjectionIdentity
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.matplotlib_render import ImagePanelAggRenderer

    payload = _image_panel(0).display_payload
    first = payload.image
    second = EvaluatedImage(
        first.x_axis,
        first.y_axis,
        np.asarray(first.values) + 17.0,
        first.validity,
        first.value_unit,
    )
    first_identity = EvaluatedProjectionIdentity(
        "document",
        0,
        payload.evaluated_input,
        "image",
        (),
        (),
        (),
        first,
    )
    second_identity = replace(first_identity, data=second)
    renderer = ImagePanelAggRenderer(width=320, height=240)
    try:
        renderer.render(
            first,
            payload.viewport,
            ImageDisplayState(),
            color_limits=(0.0, 100.0),
            data_range=(0.0, 100.0),
            title="image",
            projection_identity=first_identity,
        )
        renderer.render(
            second,
            payload.viewport,
            ImageDisplayState(),
            color_limits=(0.0, 100.0),
            data_range=(0.0, 100.0),
            title="image",
            projection_identity=second_identity,
        )
        assert renderer._prepared_image_key == second_identity
        assert np.array_equal(renderer._prepared_image_value[0], second.values)
    finally:
        renderer.close()


def test_invalid_full_resolution_uint_image_is_masked_without_dtype_promotion() -> None:
    from zlc_data import immutable_array
    from zlc_frontend._mpl_image import _decimate_image_view

    values = immutable_array(
        np.arange(48, dtype=np.uint8).reshape(6, 8),
        dtype=np.dtype(np.uint8),
        shape=(6, 8),
    )
    validity = np.ones(values.shape, dtype=bool)
    validity[2, 3] = False
    shown, _extent = _decimate_image_view(
        values,
        validity,
        (0.0, 8.0, 6.0, 0.0),
        (0.0, 8.0),
        (0.0, 6.0),
        (8, 6),
    )

    assert np.ma.isMaskedArray(shown)
    assert shown.dtype == np.dtype(np.uint8)
    assert np.shares_memory(np.ma.getdata(shown), values)
    assert bool(np.ma.getmaskarray(shown)[2, 3])


def test_evaluated_image_reuses_bytes_backed_values_and_compact_all_valid_plane() -> None:
    from test_u02c_qt_curve_interaction import _image_panel
    from zlc_data import immutable_array, immutable_bool_broadcast
    from zlc_frontend.figure import EvaluatedImage

    prototype = _image_panel(0).display_payload.image
    source = immutable_array(
        np.arange(prototype.values.size, dtype=np.uint16).reshape(
            prototype.values.shape
        ),
        dtype=np.dtype("<u2"),
        shape=prototype.values.shape,
    )
    transposed_source = source.T
    validity = immutable_bool_broadcast(True, transposed_source.shape)
    image = EvaluatedImage(
        prototype.y_axis,
        prototype.x_axis,
        transposed_source,
        validity,
        "count",
    )

    assert np.shares_memory(image.values, source)
    assert image.values.strides == transposed_source.strides
    assert np.shares_memory(image.validity, validity)
    assert image.validity.strides == (0, 0)


def test_live_image_renderer_keeps_invalid_uint_source_masked_and_uint() -> None:
    from test_u02c_qt_curve_interaction import _image_panel
    from zlc_frontend.figure import EvaluatedImage, EvaluatedProjectionIdentity
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.matplotlib_render import ImagePanelAggRenderer

    payload = _image_panel(0).display_payload
    source = payload.image
    values = np.arange(source.values.size, dtype=np.uint8).reshape(source.values.shape)
    validity = np.ones(source.validity.shape, dtype=bool)
    validity[0, 0] = False
    image = EvaluatedImage(
        source.x_axis,
        source.y_axis,
        values,
        validity,
        source.value_unit,
    )
    identity = EvaluatedProjectionIdentity(
        "invalid-uint-document",
        0,
        payload.evaluated_input,
        "image",
        (),
        (),
        (),
        image,
    )
    renderer = ImagePanelAggRenderer(width=320, height=240)
    try:
        renderer.render(
            image,
            payload.viewport,
            ImageDisplayState(),
            color_limits=(0.0, 255.0),
            data_range=(0.0, 255.0),
            title="invalid uint",
            projection_identity=identity,
        )
        artist_values = renderer._image_artist.get_array()
        assert np.ma.isMaskedArray(artist_values)
        assert artist_values.dtype == np.dtype(np.uint8)
        assert bool(np.ma.getmaskarray(artist_values)[0, 0])
    finally:
        renderer.close()


def test_invalid_image_decimation_promotes_only_the_display_sized_result() -> None:
    from zlc_data import immutable_array
    from zlc_frontend._mpl_image import _decimate_image_view

    values = immutable_array(
        np.arange(64, dtype=np.uint16).reshape(8, 8),
        dtype=np.dtype("<u2"),
        shape=(8, 8),
    )
    validity = np.ones(values.shape, dtype=bool)
    validity[:4, :4] = False
    shown, _extent = _decimate_image_view(
        values,
        validity,
        (0.0, 8.0, 8.0, 0.0),
        (0.0, 8.0),
        (0.0, 8.0),
        (2, 2),
    )

    assert np.ma.isMaskedArray(shown)
    assert shown.shape == (2, 2)
    assert shown.dtype == np.dtype(np.float64)
    assert shown.nbytes == 2 * 2 * np.dtype(np.float64).itemsize
    assert values.dtype == np.dtype(np.uint16)


def test_transposed_image_sampling_and_blocking_remain_views_until_reduction() -> None:
    from zlc_data import immutable_array, immutable_bool_broadcast
    from zlc_frontend._mpl_image import _block_view_2d, _image_distribution_values

    source = immutable_array(
        np.arange(48, dtype=np.uint16).reshape(8, 6),
        dtype=np.dtype("<u2"),
        shape=(8, 6),
    )
    transposed = source.T
    validity = immutable_bool_broadcast(True, transposed.shape)

    blocks = _block_view_2d(transposed, (2, 2))
    sampled = _image_distribution_values(transposed, validity)

    assert np.shares_memory(blocks, source)
    assert np.shares_memory(sampled, source)
    np.testing.assert_array_equal(sampled, np.ravel(transposed, order="F"))


def test_latest_resolution_value_is_not_persistent_artist_topology() -> None:
    from zlc_frontend.figure import EvaluatedFigureData, EvaluatedLayer
    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    figure = _meter_figure()
    evaluated = figure.evaluated
    layer = evaluated.layers[0]
    first = EvaluatedFigureData(
        evaluated.document_id,
        evaluated.document_revision,
        evaluated.inputs,
        (
            EvaluatedLayer(
                layer.layer_id,
                layer.dataset_id,
                (layer.cells[0],),
                layer.resolutions,
            ),
        ),
    )
    advanced_resolutions = tuple(
        replace(item, index=item.index + 1, coordinate=item.index + 1)
        for item in layer.resolutions
    )
    advanced = EvaluatedFigureData(
        evaluated.document_id,
        evaluated.document_revision,
        evaluated.inputs,
        (
            EvaluatedLayer(
                layer.layer_id,
                layer.dataset_id,
                (layer.cells[0],),
                advanced_resolutions,
            ),
        ),
    )
    renderer = SinglePanelAggRenderer(figure.document, width=320, height=240)
    try:
        renderer.render_meter(first, display_revision=0)
        renderer.render_meter(advanced, display_revision=1)
    finally:
        renderer.close()


def test_failed_family_change_keeps_the_old_front_and_binding() -> None:
    from test_u02c_qt_curve_interaction import _curve_panel, _image_panel
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app
    from zlc_frontend.render import BoardFrame

    application = ensure_qt_app()
    host = SinglePanelHost("curve")
    old = BoardFrame("host-board", 0, 0, (_curve_panel(0),))
    host.present_frame(old)
    image_panel = _image_panel(1)
    presentation = replace(
        image_panel.coherence_stamp.presentations[0],
        panel_id="curve",
    )
    image_panel = replace(
        image_panel,
        panel_id="curve",
        coherence_group="curve",
        coherence_stamp=replace(
            image_panel.coherence_stamp,
            presentations=(presentation,),
        ),
    )
    try:
        import pytest

        with pytest.raises(ValueError):
            host.present_frame(BoardFrame("wrong-board", 0, 1, (image_panel,)))
        assert host.front_frame is old
        assert host._bound_kind == "curve"
        assert "curve" in host.board._numeric_bindings

        host.present_frame(BoardFrame("host-board", 0, 1, (image_panel,)))
        assert host._bound_kind == "image"
        assert "curve" in host.board._image_bindings
        assert "curve" not in host.board._numeric_bindings
    finally:
        host.close()
        application.processEvents()


def test_faceted_host_accepts_one_indivisible_overview_artifact() -> None:
    from zlc_frontend import FacetedOverviewArtifact, RasterBuffer
    from zlc_frontend.qt_widgets import FacetedPanelHost, ensure_qt_app
    from zlc_frontend.render import (
        PanelPresentationIdentity,
    )

    figure = _meter_figure()
    _png, regions = figure.to_png_bytes_with_panel_regions()
    raster = RasterBuffer(480, 320, bytes(480 * 320 * 4))
    artifact = FacetedOverviewArtifact(
        figure,
        raster,
        regions,
        (480, 320),
        PanelPresentationIdentity(
            "meter",
            figure.document.document_id,
            figure.document.revision,
            0,
            0,
        ),
    )
    application = ensure_qt_app()
    host = FacetedPanelHost("meter")
    try:
        host.present_overview(artifact)
        assert host.overview_artifact is artifact
        assert host.size().width() == 480
        assert host.size().height() == 320
        assert host._overview._front[0] is raster.pixels
    finally:
        host.close()
        application.processEvents()


def test_newer_but_semantically_different_frame_does_not_ack_viewport() -> None:
    from test_u02c_qt_curve_interaction import (
        _accepted_curve_frame,
        _board,
        _curve_target,
        _frame,
        _point,
        _wheel,
    )

    application, board = _board(_frame(0), [], [])
    commands: list[object] = []
    board._numeric_bindings["curve"].callback = commands.append
    try:
        target = _curve_target(board)
        _wheel(board, _point(target.plot, 0.5, 0.5), 120)
        command = commands[-1]
        binding = board._numeric_bindings["curve"]
        assert binding.pending_viewport_answer is not None

        board.present(
            _frame(
                1,
                curve_revision=command.viewport.display_revision + 1,
            )
        )
        assert binding.pending_viewport_answer is not None
    finally:
        board.close()
        application.processEvents()

    application, board = _board(_frame(0), [], [])
    commands = []
    board._numeric_bindings["curve"].callback = commands.append
    try:
        target = _curve_target(board)
        _wheel(board, _point(target.plot, 0.5, 0.5), 120)
        command = commands[-1]
        board.present(_accepted_curve_frame(1, command))
        assert board._numeric_bindings["curve"].pending_viewport_answer is None
    finally:
        board.close()
        application.processEvents()


def test_image_viewport_coalesces_at_render_rate_without_losing_exact_ack() -> None:
    """Rapid wheel input retains one exact answer and one latest viewport."""

    from test_u02c_qt_curve_interaction import (
        _board,
        _curve_panel,
        _frame,
        _image_panel,
        _point,
        _wheel,
    )
    from zlc_frontend.render import BoardFrame
    from zlc_frontend.selector import ImageViewportCommit

    commands: list[ImageViewportCommit] = []
    application, board = _board(_frame(0), [], commands)
    board.set_selectors_enabled(True)
    try:
        binding = board._image_bindings["image"]
        target = board._selector_target(binding)
        assert target is not None
        centre = _point(target[0], 0.5, 0.5)

        _wheel(board, centre, -120)
        first = commands[-1]
        assert isinstance(first, ImageViewportCommit)
        assert binding.pending_viewport_answer is not None

        _wheel(board, centre, -120)
        assert commands == [first]
        queued = binding.queued_viewport_bounds
        assert queued is not None and queued != first.viewport.visible_bounds

        answered = _image_panel(
            1,
            viewport_revision=first.viewport.viewport_revision,
        )
        answered = replace(
            answered,
            source_identity=first.origin.source_identity,
            coherence_stamp=replace(
                answered.coherence_stamp,
                inputs=(first.origin.input_identity,),
            ),
            display_payload=replace(
                answered.display_payload,
                evaluated_input=first.origin.input_identity,
                viewport=first.viewport,
            ),
        )
        board.present(BoardFrame(
            "curve-board",
            0,
            1,
            (answered, _curve_panel(1)),
        ))

        assert len(commands) == 2
        latest = commands[-1]
        assert latest.viewport.visible_bounds == queued
        assert (
            latest.viewport.viewport_revision
            > first.viewport.viewport_revision
        )
        assert binding.queued_viewport_bounds is None
        assert binding.pending_viewport_answer is not None
        assert binding.pending_viewport_answer.viewport == latest.viewport
    finally:
        board.close()
        application.processEvents()


def test_one_numeric_panel_never_owns_two_inflight_answer_families() -> None:
    """A viewport answer serializes threshold authoring on the same panel."""

    from PyQt5 import QtCore, QtTest
    from test_u02e_qt_histogram_interaction import (
        _histogram_panel,
        _numeric_target,
        _point,
        _wheel,
    )
    from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app
    from zlc_frontend.render import BoardFrame

    application = ensure_qt_app()
    commands: list[object] = []
    board = QtRasterBoard(("histogram",), columns=1)
    board.resize(640, 420)
    board.show()
    board.present(BoardFrame(
        "numeric-board",
        0,
        0,
        (_histogram_panel(1, thresholds=(1.0,)),),
    ))
    board.bind_histogram_interaction("histogram", commands.append)
    board.set_selectors_enabled(True)
    application.processEvents()
    try:
        target = _numeric_target(board, "histogram")
        _wheel(board, _point(target.plot, 0.5, 0.5), -120)
        binding = board._numeric_bindings["histogram"]
        assert binding.pending_viewport_answer is not None

        x_low, x_high = target.payload.viewport.x_limits
        line_fraction = (1.0 - x_low) / (x_high - x_low)
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(target.plot, line_fraction, 0.5),
        )
        assert binding.threshold_drag is None
        assert binding.threshold_pending_answer is None
        assert binding.queued_thresholds is None
        assert len(commands) == 1
    finally:
        board.close()
        application.processEvents()


def test_numeric_pan_release_authors_a_final_unseen_pointer_position() -> None:
    """Release reuses motion geometry when Qt omitted the final move event."""

    from PyQt5 import QtCore, QtGui, QtTest
    from test_u02c_qt_curve_interaction import (
        _board,
        _curve_target,
        _frame,
        _point,
        _drag_move,
    )

    commands: list[object] = []
    application, board = _board(_frame(0), commands, [])
    board.set_selectors_enabled(True)
    try:
        target = _curve_target(board)
        press = _point(target.plot, 0.45, 0.5)
        motion = _point(target.plot, 0.50, 0.5)
        release = _point(target.plot, 0.70, 0.5)
        QtTest.QTest.mousePress(board, QtCore.Qt.MiddleButton, pos=press)
        binding = board._numeric_bindings["curve"]
        origin = binding.pan_origin
        anchor = binding.pan_anchor
        assert origin is not None and anchor is not None
        _drag_move(board, motion, QtCore.Qt.MiddleButton)
        first = commands[-1]

        release_x = (
            float(release.x()) - target.bounds.x()
        ) / max(1, target.bounds.width())
        expected = origin.panned_x_limits(
            anchor,
            release_x,
            start_x_limits=origin.x_limits,
        )
        board.mouseReleaseEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease,
            QtCore.QPointF(release),
            QtCore.Qt.MiddleButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        ))

        assert commands == [first]
        assert binding.queued_viewport_limits == pytest.approx(expected)
        assert board._selector_hold is None
    finally:
        board.close()
        application.processEvents()


def test_archive_payload_owns_complete_typed_presentation() -> None:
    from zlc_frontend import (
        FigurePresentationContract,
        MeterDisplayState,
        decode_figure_archive_payload,
        encode_figure_archive_payload,
    )
    from zlc_frontend.figure import ViewIntent

    figure = _meter_figure()
    presentation = FigurePresentationContract(
        intent=ViewIntent.METER,
        faceted=True,
        rolling_trace=False,
        rolling_distribution=False,
        title="occupancy",
        value_label="state",
        size_name="4x4",
        display=MeterDisplayState(0, None, revision=9),
    )
    payload = encode_figure_archive_payload(
        figure,
        presentation=presentation,
        metadata={"source": "virtual"},
    )
    reopened = decode_figure_archive_payload(payload)

    from zlc_storage import decode

    presentation_tree = decode(payload)["presentation"]
    assert "pixel_ratio" not in presentation_tree
    assert presentation_tree["size_name"] == "4x4"

    assert reopened.presentation == presentation
    assert reopened.metadata["source"] == "virtual"
    assert reopened.figure.document == figure.document


def test_frontend_contract_registry_and_rolling_histogram_diagnostic_are_closed() -> None:
    from types import MappingProxyType

    import pytest

    from zlc_data import histogram_gaussian_display_diagnostic
    from zlc_frontend.figure import VIEW_CONTRACTS, ViewIntent

    assert isinstance(VIEW_CONTRACTS, MappingProxyType)
    with pytest.raises(TypeError):
        VIEW_CONTRACTS[ViewIntent.CURVE] = object()

    assert histogram_gaussian_display_diagnostic(
        np.asarray((0.0, 1.0, 2.0)),
        np.asarray((1.0, 2.0, 3.0)),
    ) is None
    diagnostic = histogram_gaussian_display_diagnostic(
        np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0)),
        np.asarray((1.0, 4.0, 4.0, 1.0)),
    )
    assert diagnostic is not None
    amplitude, center, sigma = diagnostic
    assert amplitude == 4.0
    assert center == pytest.approx(0.0)
    assert sigma > 0.0
