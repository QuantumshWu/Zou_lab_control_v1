"""U0.3g exact SITE-faceted HISTOGRAM Grid product contracts."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app
import pytest

import Zou_lab_control.api as zlc
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    DatasetComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend import DataFigure, HistogramPanelPayload
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    EvaluatedHistogram,
    FigureDocument,
    FigureLayer,
    FixedIndex,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.histogram_display import (
    HistogramCountScale,
    HistogramDisplayState,
    histogram_display_form_values,
)
from zlc_frontend.selector import HistogramRangeGesture
import zlc_workbench.data_figure.app as figure_workbench
import zlc_frontend.data_figure_presentation as figure_presentation


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


def _axis(name: str, role, size: int, coordinates) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(coordinates))


def _histogram_grid(*, layers: int = 1, revision: int = 5) -> DataFigure:
    repeat = _axis("u03g.repeat", REPEAT, 4, range(4))
    point = _axis("u03g.point", SCAN_POINT, 1, ("readout",))
    site = _axis("u03g.site", SITE, 3, ("A", "B", "C"))
    values = np.asarray(
        (
            ((1.0, 10.0, 100.0),),
            ((1.0, 20.0, 200.0),),
            ((1.0, 30.0, 300.0),),
            ((1.0, 40.0, 400.0),),
        )
    )
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[2, 0, 1] = False
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
            "photoelectron",
        ),
    )
    block = DataBlock(
        BlockId("u03g-counts"),
        DatasetRevision(revision),
        values,
        DatasetComponentValidity((site.axis_id,), valid),
        schema,
    )
    dataset_id = DatasetId("u03g-counts")
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(repeat.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(
                point.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            AxisViewBinding(site.axis_id, AxisViewRole.FACET),
        ),
    )
    document = FigureDocument(
        "u03g-histogram-grid",
        2,
        (DatasetDescriptor(dataset_id, "Occupancy counts", schema.fingerprint),),
        tuple(
            FigureLayer(f"u03g-layer-{index}", dataset_id, view)
            for index in range(layers)
        ),
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03g-counts-generation")),
        block,
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )


def _until(application, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.shutdown()
    _until(application, lambda: window.closed)


def _center(region) -> tuple[float, float]:
    return (
        (region.left + region.right) / 2.0,
        (region.top + region.bottom) / 2.0,
    )


def _blank_point(regions):
    for y in np.linspace(0.01, 0.99, 40):
        for x in np.linspace(0.01, 0.99, 40):
            if not any(region.contains(float(x), float(y)) for region in regions):
                return float(x), float(y)
    raise AssertionError("histogram overview unexpectedly has no blank margin")


def _wheel_histogram(board, delta: int):
    binding = board._numeric_binding_for_kind(
        "histogram",
        panel_id="generic-typed",
    )
    assert binding is not None
    target = board._numeric_target(binding)
    assert target is not None
    position = QtCore.QPoint(
        int(round(target.plot.left() + 0.5 * target.plot.width())),
        int(round(target.plot.top() + 0.5 * target.plot.height())),
    )
    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(board.mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    board.wheelEvent(event)
    return event


def test_histogram_grid_overview_shares_bins_and_count_scale() -> None:
    from zlc_frontend.matplotlib_render import (
        release_agg_figure,
        render_evaluated_figure,
    )

    source = _histogram_grid()
    rendered = render_evaluated_figure(source.document, source.evaluated, {})
    try:
        axes = tuple(rendered.axes[:3])
        assert len(axes) == 3
        x_limits = tuple(axis.get_xlim() for axis in axes)
        y_limits = tuple(axis.get_ylim() for axis in axes)
        assert x_limits[0] == pytest.approx(x_limits[1])
        assert x_limits[0] == pytest.approx(x_limits[2])
        assert x_limits[0][0] <= 1.0
        assert x_limits[0][1] >= 400.0
        assert y_limits[0] == pytest.approx(y_limits[1])
        assert y_limits[0] == pytest.approx(y_limits[2])
        assert y_limits[0][1] >= 4.0
    finally:
        release_agg_figure(rendered)


def test_histogram_focus_reuses_exact_series_and_revision_identity() -> None:
    figure = _histogram_grid()
    _png, regions = figure.to_png_bytes_with_panel_regions()
    expected = figure.evaluated.layers[0].cells[1].series[0]
    focused = figure.focused_typed_panel(
        1,
        expected_selection=regions[1].focus_selection,
        expected_intent=ViewIntent.HISTOGRAM,
    )
    assert focused.evaluated.layers[0].cells[0].series[0] is expected
    assert focused.evaluated.inputs == figure.evaluated.inputs
    histogram = expected.data
    assert isinstance(histogram, EvaluatedHistogram)
    np.testing.assert_array_equal(histogram.samples, (10.0, 20.0, 40.0))
    assert histogram.dropped_count == 1
    assert histogram.value_unit == "photoelectron"
    assert focused.document.layers[0].view.binding(AxisId("u03g.site")).role is (
        AxisViewRole.SELECTED
    )
    with pytest.raises(ValueError, match="selection differs"):
        figure.focused_typed_panel(
            1,
            expected_selection=regions[0].focus_selection,
            expected_intent=ViewIntent.HISTOGRAM,
        )
    newer = _histogram_grid(revision=6)
    _new_png, newer_regions = newer.to_png_bytes_with_panel_regions()
    newer_focus = newer.focused_typed_panel(
        1,
        expected_selection=newer_regions[1].focus_selection,
        expected_intent=ViewIntent.HISTOGRAM,
    )
    assert newer_focus.document.document_id == focused.document.document_id
    assert newer_focus.evaluated.inputs[0].ref != focused.evaluated.inputs[0].ref


def test_sparse_histogram_layout_keeps_its_empty_logical_cell() -> None:
    repeat = _axis("u03g.sparse.repeat", REPEAT, 2, (0, 1))
    row = _axis("u03g.sparse.row", SCAN_POINT, 2, ("r0", "r1"))
    column = _axis("u03g.sparse.column", SCAN_POINT, 2, ("c0", "c1"))
    schema = DatasetSchema(
        repeat,
        (row, column),
        PointLayout.explicit((2, 2), ((0, 0), (1, 0), (0, 1))),
        ValueSchema.scalar(np.dtype(float), "count"),
    )
    block = DataBlock(
        BlockId("u03g-sparse-histogram"),
        DatasetRevision(1),
        np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))[..., None],
        VALID,
        schema,
    )
    dataset_id = DatasetId("u03g-sparse-histogram")
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(repeat.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(row.axis_id, AxisViewRole.FACET),
            AxisViewBinding(column.axis_id, AxisViewRole.FACET),
            AxisViewBinding(
                schema.cell_schema.data_axes[0].axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
        ),
    )
    figure = DataFigure(
        FigureDocument(
            "u03g-sparse-histogram-grid",
            0,
            (DatasetDescriptor(dataset_id, "Sparse", schema.fingerprint),),
            (FigureLayer("sparse", dataset_id, view),),
        ),
        ResolvedDatasetMap(
            (
                ResolvedDataset(
                    dataset_id,
                    OwnedSnapshot(
                        block.ref(StreamGenerationId("u03g-sparse-generation")),
                        block,
                    ),
                ),
            )
        ),
    )
    cells = figure.evaluated.layers[0].cells
    assert len(cells) == 4
    assert tuple(len(cell.series[0].data.samples) for cell in cells) == (2, 2, 2, 0)
    _png, regions = figure.to_png_bytes_with_panel_regions()
    hole = figure.focused_typed_panel(
        3,
        expected_selection=regions[3].focus_selection,
        expected_intent=ViewIntent.HISTOGRAM,
    )
    assert hole.evaluated.layers[0].cells[0].series[0] is cells[3].series[0]
    assert hole.evaluated.layers[0].cells[0].series[0].data.samples.size == 0
    assert tuple(
        (term.axis_id, term.index)
        for term in regions[3].focus_selection.terms
    ) == ((column.axis_id, 1), (row.axis_id, 1))


def test_histogram_grid_overview_focus_interaction_back_and_exports(
    application,
    tmp_path: Path,
) -> None:
    figure = _histogram_grid()
    expected = figure.evaluated.layers[0].cells[1].series[0]
    window = figure_workbench.create_data_figure_pane(figure)
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        overview = window._grid_overview
        assert overview is not None
        assert overview.intent is ViewIntent.HISTOGRAM
        assert window._view_family == "histogram-overview"
        assert window._board_widget.front_frame is None
        assert len(overview.regions) == 3
        original_png = window._bundle.pages[0].png_bytes

        window._focus_grid_region(*_blank_point(overview.regions))
        assert window._view_family == "histogram-overview"
        assert window._future is None

        window._focus_grid_region(*_center(overview.regions[1]))
        _until(
            application,
            lambda: window.worker_idle and window._view_family == "histogram",
        )
        payload = window._board_widget.visible_histogram_payload("generic-typed")
        assert isinstance(payload, HistogramPanelPayload)
        assert payload.series == (expected,)
        assert payload.value_unit == "photoelectron"
        assert payload.series[0].data.dropped_count == 1
        assert overview.histogram_home_x_limits is not None
        assert payload.viewport.home_x_limits == pytest.approx(
            overview.histogram_home_x_limits
        )
        assert payload.viewport.x_limits == pytest.approx(
            overview.histogram_home_x_limits
        )
        assert payload.viewport.x_limits_are_auto
        assert tuple(
            window._tabs.tabText(index) for index in range(window._tabs.count())
        ) == ("Histogram", "Edit")
        assert not window._overview_button.isHidden()
        assert window._overview_button.isEnabled()
        assert _wheel_histogram(window._board_widget, -120).isAccepted()
        _until(application, lambda: window.worker_idle and window.raster_ready)
        assert window._display.revision == 1
        origin = window._board_widget.visible_histogram_origin("generic-typed")
        assert origin is not None
        window._accept_numeric_interaction(
            HistogramRangeGesture(origin, (12.0, 42.0))
        )
        values = histogram_display_form_values(window._display)
        values["bin_count"] = 17
        values["count_scale"] = HistogramCountScale.LOG
        window._apply_display_form(
            window._edit_display,
            window._display.revision,
            values,
        )
        _until(application, lambda: window.worker_idle and window.raster_ready)
        updated = window._board_widget.visible_histogram_payload("generic-typed")
        assert updated.series == (expected,)
        assert updated.viewport.bin_count == 17
        assert updated.viewport.count_scale is HistogramCountScale.LOG
        assert window._board_widget._numeric_bindings[
            "generic-typed"
        ].applied_span == (12.0, 42.0)

        focused_frame = window._board_widget.front_frame
        focused_path = tmp_path / "histogram-focus.png"
        window._start_export(focused_path)
        _until(application, lambda: window.worker_idle and focused_path.exists())
        with Image.open(focused_path) as image:
            rgba = image.convert("RGBA")
            assert rgba.size == (
                focused_frame.panels[0].raster.width,
                focused_frame.panels[0].raster.height,
            )
            assert rgba.tobytes() == focused_frame.panels[0].raster.pixels

        window._overview_button.click()
        application.processEvents()
        assert window._view_family == "histogram-overview"
        assert window._tabs.currentWidget() is window._tab_host_for_board(
            window._boards[0]
        )
        assert not window._tabs.tabBar().isVisible()
        assert window._board_widget.front_frame is None
        assert window._bundle.pages[0].png_bytes is original_png
        overview_path = tmp_path / "histogram-overview.png"
        window._start_export(overview_path)
        _until(application, lambda: window.worker_idle and overview_path.exists())
        assert overview_path.read_bytes() == original_png

        window._focus_grid_region(*_center(overview.regions[2]))
        _until(application, lambda: window.worker_idle and window.raster_ready)
        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Escape)
        application.processEvents()
        assert window._view_family == "histogram-overview"
        assert window._bundle.pages[0].png_bytes is original_png
    finally:
        _close(application, window)


def test_escape_during_histogram_rerender_cannot_late_present(
    application,
    monkeypatch,
) -> None:
    figure = _histogram_grid()
    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    original = SinglePanelAggRenderer.render_interactive_histogram
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            if not release.wait(10.0):
                raise TimeoutError("test did not release histogram rerender")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        SinglePanelAggRenderer,
        "render_interactive_histogram",
        blocked,
    )
    window = figure_workbench.create_data_figure_pane(figure)
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        overview = window._grid_overview
        assert overview is not None
        original_png = window._bundle.pages[0].png_bytes
        window._focus_grid_region(*_center(overview.regions[0]))
        _until(application, lambda: window.worker_idle and window.raster_ready)
        values = histogram_display_form_values(window._display)
        values["bin_count"] = 19
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, entered.is_set)
        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Escape)
        release.set()
        _until(
            application,
            lambda: window.worker_idle
            and window._view_family == "histogram-overview",
        )
        assert window._board_widget.front_frame is None
        assert window._bundle.pages[0].png_bytes is original_png
    finally:
        release.set()
        _close(application, window)


def test_close_during_histogram_focus_cannot_present_a_late_front(
    application,
    monkeypatch,
) -> None:
    figure = _histogram_grid()
    entered = threading.Event()
    release = threading.Event()
    original = DataFigure.focused_typed_panel

    def blocked(self, *args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release histogram focus")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataFigure, "focused_typed_panel", blocked)
    window = figure_workbench.create_data_figure_pane(figure)
    _until(application, lambda: window.raster_ready and window.worker_idle)
    overview = window._grid_overview
    assert overview is not None
    window._focus_grid_region(*_center(overview.regions[1]))
    _until(application, entered.is_set)
    window.shutdown()
    release.set()
    _until(application, lambda: window.closed)
    assert window._board_widget.front_frame is None


def test_multi_layer_histogram_remains_whole_figure_fallback(application) -> None:
    figure = _histogram_grid(layers=2)
    intent, count, reason = figure_presentation.classify_faceted_data_figure(figure)
    assert intent is count is None
    assert "one layer" in reason
    window = figure_workbench.create_data_figure_pane(figure)
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        assert window._view_family == "encoded"
        assert window._grid_overview is None
    finally:
        _close(application, window)


def test_public_occupancy_counts_entry_opens_the_histogram_grid(
    application,
    monkeypatch,
    tmp_path: Path,
) -> None:
    from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import load_sitemap_pulse

    owner_thread = threading.get_ident()
    calls = []
    with zlc.connect("virtual", repository=tmp_path / "public-entry") as experiment:
        calibration_reference = experiment.nodes.calibration.sitemap(frames=12)
        document = load_sitemap_pulse()
        trigger_index = document.target.raw_lanes.index("ch11")
        trigger_run = 0
        previous = False
        periods = []
        for period in document.periods:
            states = list(period.states)
            high = bool(states[trigger_index])
            if high and not previous:
                trigger_run += 1
            states[trigger_index] = int(high and trigger_run == 2)
            periods.append(replace(period, states=tuple(states)))
            previous = high
        assert trigger_run == 3
        readout_document = replace(
            document,
            name="u03g-occupancy-readout",
            periods=tuple(periods),
            repeat=None,
        )
        capture_reference = experiment.readout.capture(
            readout_document,
            trigger_channel="ch11",
            readout_events_per_repeat=1,
        )
        reference = experiment.nodes.occupancy.detect(
            experiment.nodes.occupancy.detection_request(
                capture_reference,
                calibration_reference,
            )
        )
        expected_figure = experiment.figure(reference, output="counts")
        expected_cells = expected_figure.evaluated.layers[0].cells
        assert len(expected_cells) > 1
        original = type(experiment).figure

        def traced(self, source, *args, **options):
            calls.append((threading.get_ident(), source, args, options))
            return original(self, source, *args, **options)

        monkeypatch.setattr(type(experiment), "figure", traced)
        window = experiment.figure_gui(
            reference,
            output="counts",
        )
        try:
            _until(application, lambda: window.worker_idle and window.raster_ready)
            assert window._view_family == "histogram-overview"
            overview = window._grid_overview
            assert overview is not None
            assert len(overview.regions) == len(expected_cells)
            assert len(calls) == 1
            thread_id, source, args, options = calls[0]
            assert thread_id != owner_thread
            assert source == reference and args == ()
            assert options["output"] == "counts"
            assert options["selection"] is None
            assert options["intent"] is None
            window._focus_grid_region(*_center(overview.regions[1]))
            _until(
                application,
                lambda: window.worker_idle and window._view_family == "histogram",
            )
            payload = window._board_widget.visible_histogram_payload(
                "generic-typed"
            )
            expected = expected_cells[1].series[0]
            np.testing.assert_array_equal(
                payload.series[0].data.samples,
                expected.data.samples,
            )
            assert payload.series[0].data.dropped_count == expected.data.dropped_count
            assert payload.evaluated_input.ref == expected_figure.evaluated.inputs[0].ref
        finally:
            _close(application, window)
