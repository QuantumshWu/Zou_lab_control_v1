"""Narrow regressions for exact worker-raster hand-off semantics."""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _meter_figure(*, faceted: bool = True):
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
        PointTable,
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
    from zlc_data import AxisSourceRef

    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
    site = AxisSpec(AxisId("site"), "site", SITE, 3, ("A", "B", "C"))
    values = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    valid = np.ones(values.shape, dtype=bool)
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
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
        preferences=ViewPreferences(
            facet_sources=(
                (AxisSourceRef.tensor(site.axis_id),)
                if faceted
                else ()
            ),
        ),
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
    from figure_surface_fixtures import image_panel
    from zlc_frontend.figure import EvaluatedImage, EvaluatedProjectionIdentity
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.matplotlib_render import ImagePanelAggRenderer

    payload = image_panel(0).display_payload
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
    from zlc_data._arrays import immutable_array
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
    from figure_surface_fixtures import image_panel
    from zlc_data._arrays import immutable_array, immutable_bool_broadcast
    from zlc_frontend.figure import EvaluatedImage

    prototype = image_panel(0).display_payload.image
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
    from figure_surface_fixtures import image_panel
    from zlc_frontend.figure import EvaluatedImage, EvaluatedProjectionIdentity
    from zlc_frontend.image_display import ImageDisplayState
    from zlc_frontend.matplotlib_render import ImagePanelAggRenderer

    payload = image_panel(0).display_payload
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
    from zlc_data._arrays import immutable_array
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
    from zlc_data._arrays import immutable_array, immutable_bool_broadcast
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


def test_failed_family_change_keeps_the_old_front_and_interaction_family() -> None:
    from figure_surface_fixtures import curve_panel, image_panel
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app
    from zlc_frontend.render import BoardFrame

    application = ensure_qt_app()
    host = SinglePanelHost("curve")
    old = BoardFrame("host-board", 0, 0, (curve_panel(0),))
    host.present_frame(old)
    candidate = image_panel(1)
    presentation = replace(
        candidate.coherence_stamp.presentations[0],
        panel_id="curve",
    )
    candidate = replace(
        candidate,
        panel_id="curve",
        coherence_group="curve",
        coherence_stamp=replace(
            candidate.coherence_stamp,
            presentations=(presentation,),
        ),
    )
    try:
        import pytest

        old_origin = host.visible_interaction_origin()
        assert old_origin is not None
        with pytest.raises(ValueError):
            host.present_frame(BoardFrame("wrong-board", 0, 1, (candidate,)))
        assert host.front_frame is old
        assert host.visible_interaction_origin() == old_origin
        host.set_range_candidate(None)
        with pytest.raises(RuntimeError, match="image binding"):
            host.set_rectangle_candidate(None)

        current = BoardFrame("host-board", 0, 1, (candidate,))
        host.present_frame(current)
        assert host.front_frame is current
        assert host.visible_interaction_origin() != old_origin
        host.set_rectangle_candidate(None)
        with pytest.raises(RuntimeError, match="numeric binding"):
            host.set_range_candidate(None)
    finally:
        host.close()
        application.processEvents()


def test_faceted_host_accepts_one_indivisible_overview_artifact() -> None:
    from zlc_frontend.data_figure import FacetedOverviewArtifact
    from zlc_frontend.qt_widgets import FacetedPanelHost, ensure_qt_app
    from zlc_frontend.render import (
        PanelPresentationIdentity,
        RasterBuffer,
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
    from figure_surface_fixtures import curve_panel
    from gui_user_flow import normalized_subrect, point_in_rect, send_wheel
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app
    from zlc_frontend.render import BoardFrame

    application = ensure_qt_app()

    def frame(sequence, panel):
        return BoardFrame("curve-host", 0, sequence, (panel,))

    def accepted_panel(sequence, command):
        panel = curve_panel(
            sequence,
            display_revision=command.viewport.display_revision,
        )
        return replace(
            panel,
            source_identity=command.origin.source_identity,
            coherence_stamp=replace(
                panel.coherence_stamp,
                inputs=(command.origin.input_identity,),
            ),
            display_payload=replace(
                panel.display_payload,
                evaluated_input=command.origin.input_identity,
                viewport=command.viewport,
            ),
        )

    host = SinglePanelHost("curve")
    host.resize(640, 320)
    host.show()
    host.set_selectors_enabled(True)
    commands: list[object] = []
    host.viewCommitted.connect(commands.append)
    host.present_frame(frame(0, curve_panel(0)))
    board = host.board
    application.processEvents()
    try:
        payload = host.front_frame.panels[0].display_payload
        plot = normalized_subrect(board.rect(), payload.viewport.plot_bounds)
        send_wheel(board, point_in_rect(plot, 0.5, 0.5), 120)
        command = commands[-1]

        host.present_frame(
            frame(
                1,
                curve_panel(
                    1,
                    display_revision=command.viewport.display_revision + 1,
                ),
            )
        )
        payload = host.front_frame.panels[0].display_payload
        plot = normalized_subrect(board.rect(), payload.viewport.plot_bounds)
        send_wheel(board, point_in_rect(plot, 0.5, 0.5), 120)
        assert commands == [command]
    finally:
        host.close()
        application.processEvents()

    host = SinglePanelHost("curve")
    host.resize(640, 320)
    host.show()
    host.set_selectors_enabled(True)
    commands = []
    host.viewCommitted.connect(commands.append)
    host.present_frame(frame(0, curve_panel(0)))
    board = host.board
    application.processEvents()
    try:
        payload = host.front_frame.panels[0].display_payload
        plot = normalized_subrect(board.rect(), payload.viewport.plot_bounds)
        send_wheel(board, point_in_rect(plot, 0.5, 0.5), 120)
        command = commands[-1]
        host.present_frame(frame(1, accepted_panel(1, command)))
        payload = host.front_frame.panels[0].display_payload
        plot = normalized_subrect(board.rect(), payload.viewport.plot_bounds)
        send_wheel(board, point_in_rect(plot, 0.5, 0.5), 120)
        assert len(commands) == 2
        assert commands[-1].origin == host.visible_interaction_origin()
    finally:
        host.close()
        application.processEvents()


def test_image_viewport_coalesces_at_render_rate_without_losing_exact_ack() -> None:
    """Rapid wheel input retains one exact answer and one latest viewport."""

    from figure_surface_fixtures import image_panel
    from gui_user_flow import point_in_rect, raster_subrect, send_wheel
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app
    from zlc_frontend.render import BoardFrame
    from zlc_frontend.selector import ImageViewportCommit

    commands: list[ImageViewportCommit] = []
    application = ensure_qt_app()
    host = SinglePanelHost("image")
    host.resize(640, 420)
    host.show()
    host.set_selectors_enabled(True)
    host.viewCommitted.connect(commands.append)
    host.present_frame(BoardFrame(
        "image-host",
        0,
        0,
        (image_panel(0),),
    ))
    board = host.board
    application.processEvents()
    try:
        payload = host.front_frame.panels[0].display_payload
        target = raster_subrect(
            board.rect(), payload.raster_geometry.image_bounds
        )
        centre = point_in_rect(target, 0.5, 0.5)

        send_wheel(board, centre, -120)
        first = commands[-1]
        assert isinstance(first, ImageViewportCommit)

        send_wheel(board, centre, -120)
        assert commands == [first]
        expected = first.viewport.centered_zoom((0.5, 0.5), 1.0 / 1.1)

        answered = image_panel(
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
        host.present_frame(BoardFrame(
            "image-host",
            0,
            1,
            (answered,),
        ))

        assert len(commands) == 2
        latest = commands[-1]
        assert latest.viewport.visible_bounds == expected.visible_bounds
        assert (
            latest.viewport.viewport_revision
            > first.viewport.viewport_revision
        )
        assert latest.origin == host.visible_interaction_origin()
    finally:
        host.close()
        application.processEvents()


def test_numeric_pan_release_authors_a_final_unseen_pointer_position() -> None:
    """Release reuses motion geometry when Qt omitted the final move event."""

    from PyQt5 import QtCore, QtTest
    from figure_surface_fixtures import curve_panel
    from gui_user_flow import drag_mouse_move, normalized_subrect, point_in_rect
    from zlc_frontend.qt_widgets import SinglePanelHost, ensure_qt_app
    from zlc_frontend.render import BoardFrame

    commands: list[object] = []
    application = ensure_qt_app()
    host = SinglePanelHost("curve")
    host.resize(640, 320)
    host.show()
    host.set_selectors_enabled(True)
    host.viewCommitted.connect(commands.append)
    host.present_frame(BoardFrame(
        "curve-pan-host",
        0,
        0,
        (curve_panel(0),),
    ))
    board = host.board
    application.processEvents()
    try:
        payload = host.front_frame.panels[0].display_payload
        plot = normalized_subrect(board.rect(), payload.viewport.plot_bounds)
        press = point_in_rect(plot, 0.45, 0.5)
        motion = point_in_rect(plot, 0.50, 0.5)
        release = point_in_rect(plot, 0.70, 0.5)
        board_bounds = board.rect()
        origin = payload.viewport
        anchor = (
            float(press.x()) - board_bounds.x()
        ) / max(1, board_bounds.width())
        QtTest.QTest.mousePress(board, QtCore.Qt.MiddleButton, pos=press)
        drag_mouse_move(board, motion, QtCore.Qt.MiddleButton)
        first = commands[-1]

        release_x = (
            float(release.x()) - board_bounds.x()
        ) / max(1, board_bounds.width())
        expected = origin.panned_x_limits(
            anchor,
            release_x,
            start_x_limits=origin.x_limits,
        )
        QtTest.QTest.mouseRelease(
            board,
            QtCore.Qt.MiddleButton,
            pos=release,
        )

        assert commands == [first]
        answered = curve_panel(
            1,
            display_revision=first.viewport.display_revision,
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
        host.present_frame(BoardFrame(
            "curve-pan-host",
            0,
            1,
            (answered,),
        ))
        assert len(commands) == 2
        assert commands[-1].viewport.x_limits == pytest.approx(expected)
        assert commands[-1].origin == host.visible_interaction_origin()
    finally:
        host.close()
        application.processEvents()


def test_archive_payload_owns_complete_typed_presentation() -> None:
    from zlc_frontend import (
        FigureIntent,
        MeterDisplayState,
        PlotKind,
    )
    from zlc_frontend.figure_archive import (
        decode_figure_archive_payload,
        encode_figure_archive_payload,
    )

    figure = _meter_figure(faceted=False)
    figure_intent = FigureIntent(
        PlotKind.METER,
        "occupancy",
        "state",
        view=figure.document.layers[0].view,
    )
    display = MeterDisplayState(0, None, revision=9)
    payload = encode_figure_archive_payload(
        figure,
        figure_intent=figure_intent,
        size_name="4x4",
        display=display,
        metadata={"source": "virtual"},
    )
    reopened = decode_figure_archive_payload(payload)

    from zlc_storage import decode

    presentation_tree = decode(payload)["presentation"]
    assert "pixel_ratio" not in presentation_tree
    assert presentation_tree["size_name"] == "4x4"

    assert reopened.figure_intent == figure_intent
    assert reopened.size_name == "4x4"
    assert reopened.display == display
    assert reopened.metadata["source"] == "virtual"
    assert reopened.figure.document == figure.document


def test_frontend_contract_registry_and_rolling_histogram_diagnostic_are_closed() -> None:
    from types import MappingProxyType

    import pytest

    from zlc_data.fit import histogram_gaussian_display_diagnostic
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


def test_grid_view_editor_exposes_singleton_dataset_as_typed_unavailable() -> None:
    """A Dataset with no facet source remains a valid, non-fatal UI state."""

    from zlc_data import REPEAT, AxisId, AxisSpec, DatasetSchema, PointTable, ValueSchema
    from zlc_frontend.qt_widgets import (
        FluentParameterForm,
        ViewSpecEditor,
        ensure_qt_app,
    )

    schema = DatasetSchema(
        AxisSpec(AxisId("singleton.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(1),
        None,
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )
    application = ensure_qt_app()
    editor = ViewSpecEditor()
    editor.show()
    try:
        editor.reconcile(schema, None, faceted=True)
        application.processEvents()

        form = editor.findChild(FluentParameterForm)
        assert form is not None
        assert form.keys == ("grid.intent",)
        field = form.spec.fields[0]
        assert field.required_choice_unavailable
        control = form.widget_for("grid.intent")
        assert not control.isEnabled()
        assert "No declared sub plot" in control.toolTip()

        # Reconciliation of the same unavailable declaration is stable and
        # must never turn an empty candidate set into an invalid fake choice.
        editor.reconcile(schema, None, faceted=True)
        application.processEvents()
        assert form is editor.findChild(FluentParameterForm)
        assert not form.widget_for("grid.intent").isEnabled()
    finally:
        editor.close()
        application.processEvents()
