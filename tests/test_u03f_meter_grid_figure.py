"""U0.3f exact METER payload and single-layer grid explorer contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets

import Zou_lab_control.notebook as zlc
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPECTRAL,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    ReductionMethod,
    Selection,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend import DataFigure, MeterDisplayState, MeterPanelPayload
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    EvaluatedMeter,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    suggest_view,
)
from zlc_frontend.matplotlib_render import SinglePanelAggRenderer
from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: F401
from zlc_frontend.qt_widgets import QtRasterBoard
from Zou_lab_control.workbench import _figure as figure_workbench


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


def _axis(name: str, role, size: int, *, coordinates=None) -> AxisSpec:
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)) if coordinates is None else tuple(coordinates),
    )


def _meter_figure(*, layers: int = 1, revision: int = 7) -> DataFigure:
    repeat = _axis("repeat", REPEAT, 2)
    site = _axis("site", SITE, 3, coordinates=("A", "B", "C"))
    values = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    valid = np.ones(values.shape, dtype=bool)
    valid[1, 0, 1] = False
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
        DatasetRevision(revision),
        values,
        ComponentValidity((site.axis_id,), valid),
        schema,
    )
    dataset_id = DatasetId("meter-dataset")
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("meter-generation")),
        block,
    )
    datasets = ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),))
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
        tuple(
            FigureLayer(f"meter-layer-{index}", dataset_id, suggestion.spec)
            for index in range(layers)
        ),
    )
    return DataFigure(document, datasets)


def _until(application, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _blank_point(regions):
    for y in np.linspace(0.01, 0.99, 40):
        for x in np.linspace(0.01, 0.99, 40):
            if not any(region.contains(float(x), float(y)) for region in regions):
                return float(x), float(y)
    raise AssertionError("rendered overview unexpectedly has no blank margin")


def _camera_roi(experiment) -> Selection:
    schema = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=2)
    ).output_schema
    y_axis, x_axis = schema.cell_schema.data_axes
    return Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        x_axis.coordinates[0],
        x_axis.coordinates[min(15, x_axis.size - 1)],
        y_axis.coordinates[0],
        y_axis.coordinates[min(11, y_axis.size - 1)],
        coordinate_frame=x_axis.coordinate_frame,
    )


def test_meter_unit_validity_and_exact_focus_are_preserved():
    figure = _meter_figure()
    cells = figure.evaluated.layers[0].cells
    meters = tuple(cell.series[0].data for cell in cells)
    assert tuple(meter.value for meter in meters) == (4.0, 5.0, 6.0)
    assert tuple(meter.valid for meter in meters) == (True, False, True)
    assert tuple(meter.value_unit for meter in meters) == ("count",) * 3

    _png, regions = figure.to_png_bytes_with_panel_regions()
    selected_series = cells[1].series[0]
    focused = figure.focused_typed_panel(
        1,
        expected_selection=regions[1].selection,
        expected_intent=ViewIntent.METER,
    )
    assert focused.document.document_id != figure.document.document_id
    assert len(focused.document.layers) == len(focused.evaluated.layers) == 1
    assert len(focused.evaluated.layers[0].cells) == 1
    assert focused.evaluated.inputs == figure.evaluated.inputs
    assert focused.evaluated.layers[0].cells[0].series[0] is selected_series
    site_binding = focused.document.layers[0].view.binding(AxisId("site"))
    assert site_binding.role is AxisViewRole.SELECTED
    assert site_binding.selector.index == 1
    resolutions = {
        item.axis_id: item for item in focused.evaluated.layers[0].resolutions
    }
    assert resolutions[AxisId("site")].index == 1
    assert resolutions[AxisId("site")].coordinate == "B"
    with pytest.raises(ValueError, match="selection differs"):
        figure.focused_typed_panel(
            1,
            expected_selection=regions[0].selection,
            expected_intent=ViewIntent.METER,
        )
    updated = _meter_figure(revision=8)
    _updated_png, updated_regions = updated.to_png_bytes_with_panel_regions()
    updated_focus = updated.focused_typed_panel(
        1,
        expected_selection=updated_regions[1].selection,
        expected_intent=ViewIntent.METER,
    )
    assert updated_focus.document.document_id != focused.document.document_id
    assert len(figure.evaluated.layers[0].cells) == 3


def test_typed_meter_payload_is_exact_and_rejects_semantic_drift():
    figure = _meter_figure()
    _png, regions = figure.to_png_bytes_with_panel_regions()
    focused = figure.focused_typed_panel(
        2,
        expected_selection=regions[2].selection,
        expected_intent=ViewIntent.METER,
    )
    renderer = SinglePanelAggRenderer(focused.document, width=800, height=520)
    try:
        raster, payload = renderer.render_meter(
            focused.evaluated,
            display_revision=4,
        )
        assert raster.width == 800 and raster.height == 520
        assert payload.evaluated_input is focused.evaluated.inputs[0]
        assert payload.series[0] is focused.evaluated.layers[0].cells[0].series[0]
        assert payload.value_unit == "count"
        assert payload.display_revision == 4
        assert renderer._artists[0].get_text() == "6 count"
        with pytest.raises(ValueError, match="requires EvaluatedSeries"):
            replace(payload, series=())
        mixed = replace(
            payload.series[0],
            data=EvaluatedMeter(7.0, True, "other"),
        )
        with pytest.raises(ValueError, match="share value_unit"):
            replace(payload, series=(payload.series[0], mixed), series_labels=("a", "b"))
    finally:
        renderer.close()


def test_sparse_meter_layout_keeps_the_hole_in_its_logical_cell():
    repeat = _axis("repeat", REPEAT, 1)
    scan = _axis("scan", SCAN_POINT, 2)
    spectral = _axis("spectral", SPECTRAL, 2)
    schema = DatasetSchema(
        repeat,
        (scan, spectral),
        PointLayout.explicit((2, 2), ((0, 0), (1, 0), (0, 1))),
        ValueSchema((), ValidityContract.value(), np.dtype(float), "count"),
    )
    block = DataBlock(
        BlockId("sparse-meter-block"),
        DatasetRevision(1),
        np.asarray(((10.0, 20.0, 30.0),)),
        VALID,
        schema,
    )
    dataset_id = DatasetId("sparse-meter-dataset")
    datasets = ResolvedDatasetMap(
        (
            ResolvedDataset(
                dataset_id,
                OwnedSnapshot(
                    block.ref(StreamGenerationId("sparse-meter-generation")),
                    block,
                ),
            ),
        )
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.METER,
        preferences=ViewPreferences(
            facet_axis_ids=(scan.axis_id, spectral.axis_id),
        ),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    figure = DataFigure(
        FigureDocument(
            "sparse-meter-grid",
            0,
            (DatasetDescriptor(dataset_id, "Sparse", schema.fingerprint),),
            (FigureLayer("sparse-meter-layer", dataset_id, suggestion.spec),),
        ),
        datasets,
    )
    meters = tuple(
        cell.series[0].data for cell in figure.evaluated.layers[0].cells
    )
    assert tuple(meter.value for meter in meters) == (10.0, 30.0, 20.0, 0.0)
    assert tuple(meter.valid for meter in meters) == (True, True, True, False)
    _png, regions = figure.to_png_bytes_with_panel_regions()
    hole = figure.focused_typed_panel(
        3,
        expected_selection=regions[3].selection,
        expected_intent=ViewIntent.METER,
    )
    assert hole.evaluated.layers[0].cells[0].series[0].data.valid is False
    assert tuple(
        (term.axis_id, term.index) for term in regions[3].selection.terms
    ) == (
        (repeat.axis_id, 0),
        (scan.axis_id, 1),
        (spectral.axis_id, 1),
    )


def test_valid_nonfinite_meter_fails_before_mutating_agg_surface():
    figure = _meter_figure()
    _png, regions = figure.to_png_bytes_with_panel_regions()
    focused = figure.focused_typed_panel(
        0,
        expected_selection=regions[0].selection,
        expected_intent=ViewIntent.METER,
    )
    renderer = SinglePanelAggRenderer(focused.document, width=800, height=520)
    try:
        renderer.render_meter(focused.evaluated, display_revision=0)
        original_text = renderer._artists[0].get_text()
        cell = focused.evaluated.layers[0].cells[0]
        bad_series = replace(
            cell.series[0],
            data=EvaluatedMeter(float("nan"), True, "count"),
        )
        bad = replace(
            focused.evaluated,
            layers=(
                replace(
                    focused.evaluated.layers[0],
                    cells=(replace(cell, series=(bad_series,)),),
                ),
            ),
        )
        with pytest.raises(ValueError, match="must be finite"):
            renderer.render_meter(bad, display_revision=1)
        assert renderer._artists[0].get_text() == original_text
        renderer.render_meter(focused.evaluated, display_revision=2)
    finally:
        renderer.close()


def test_meter_grid_overview_focus_back_escape_and_atomic_exports(
    application,
    tmp_path,
):
    figure = _meter_figure()
    expected_series = figure.evaluated.layers[0].cells[1].series[0]
    window = figure_workbench.open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        assert window._view_family == "meter-overview"
        assert window._board_widget.front_frame is None
        overview = window._grid_overview
        assert overview is not None and len(overview.regions) == 3
        original_png = window._bundle.pages[0].png_bytes
        blank = _blank_point(overview.regions)
        window._focus_grid_region(*blank)
        assert window._view_family == "meter-overview"
        assert window._future is None

        region = overview.regions[1]
        window._focus_grid_region(
            (region.left + region.right) / 2.0,
            (region.top + region.bottom) / 2.0,
        )
        _until(
            application,
            lambda: window.worker_idle and window._view_family == "meter",
        )
        payload = window._board_widget.visible_meter_payload()
        assert isinstance(payload, MeterPanelPayload)
        assert payload.series[0] is expected_series
        assert payload.series[0].data.valid is False
        assert payload.value_unit == "count"
        assert window._mode.text() == "EXACT METER · DISPLAY ONLY"
        assert not window._interaction_switch.isVisible()
        assert not window._settings_button.isVisible()
        assert not window._analyze_button.isVisible()

        focused_frame = window._board_widget.front_frame
        focused_path = tmp_path / "focused.png"
        window._start_export(focused_path)
        _until(application, lambda: window.worker_idle)
        assert focused_path.exists()
        with Image.open(focused_path) as image:
            rgba = image.convert("RGBA")
            assert rgba.size == (
                focused_frame.panels[0].raster.width,
                focused_frame.panels[0].raster.height,
            )
            assert rgba.tobytes() == focused_frame.panels[0].raster.pixels

        window._show_grid_overview()
        assert window._view_family == "meter-overview"
        assert window._board_widget.front_frame is None
        assert window._bundle.pages[0].png_bytes is original_png
        overview_path = tmp_path / "overview.png"
        window._start_export(overview_path)
        _until(application, lambda: window.worker_idle)
        assert overview_path.read_bytes() == original_png

        region = overview.regions[2]
        window._focus_grid_region(
            (region.left + region.right) / 2.0,
            (region.top + region.bottom) / 2.0,
        )
        _until(application, lambda: window.worker_idle and window._view_family == "meter")
        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Escape)
        application.processEvents()
        assert window._view_family == "meter-overview"
        assert window._bundle.pages[0].png_bytes is original_png
    finally:
        window.shutdown()
        _until(application, lambda: window.closed)


def test_close_during_meter_focus_cannot_present_a_late_front(
    application,
    monkeypatch,
):
    figure = _meter_figure()
    entered = threading.Event()
    release = threading.Event()
    original = DataFigure.focused_typed_panel

    def blocked(self, *args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release focused panel derivation")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataFigure, "focused_typed_panel", blocked)
    window = figure_workbench.open_data_figure_workbench(figure)
    _until(application, lambda: window.raster_ready and window.worker_idle)
    overview = window._grid_overview
    assert overview is not None
    region = overview.regions[1]
    window._focus_grid_region(
        (region.left + region.right) / 2.0,
        (region.top + region.bottom) / 2.0,
    )
    _until(application, entered.is_set)
    window.shutdown()
    release.set()
    _until(application, lambda: window.closed)
    assert window._board_widget.front_frame is None


def test_multi_layer_meter_is_not_promoted_to_the_single_layer_explorer():
    figure = _meter_figure(layers=2)
    assert figure_workbench._classify_single_typed(figure)[0] is None
    _intent, count, reason = figure_workbench._classify_typed_grid(figure)
    assert count is None
    assert "one layer" in reason


def test_live_camera_meter_publishes_exact_typed_payload(
    application,
    tmp_path,
):
    experiment = zlc.connect("virtual", repository=tmp_path / "live-meter")
    window = None
    try:
        request = experiment.readout.camera_monitor_request(
            history_capacity=3,
            roi=_camera_roi(experiment),
            roi_reduction=ReductionMethod.MEAN,
            scalar_history_capacity=12,
        )
        window = experiment.readout.camera_monitor_gui(request)
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.front_frame is not None
                and isinstance(
                    board.front_frame.panels[3].display_payload,
                    MeterPanelPayload,
                )
            ),
        )
        frame = board.front_frame
        meter = frame.panels[3].display_payload
        histogram = frame.panels[2].display_payload
        assert meter.evaluated_input == histogram.evaluated_input
        assert meter.display_revision == frame.panels[0].coherence_stamp.presentations[3].panel_revision
        assert frame.panels[3].coherence_stamp is frame.panels[0].coherence_stamp
        assert board.visible_meter_payload("camera-monitor-roi-meter") is meter
    finally:
        if window is not None:
            window.close()
            _until(application, lambda: not window.isVisible(), timeout=8.0)
            window.deleteLater()
            application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            application.processEvents()
        experiment.close()
