"""W4 frozen DataFigure/calibration immutable-raster product-path oracles."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.workbench import open_data_figure_workbench
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    IndexSelection,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend import DataFigure
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    DisplayReductionMethod,
    EvaluatedHistogram,
    EvaluatedMeter,
    ViewIntent,
    ViewPreferences,
    suggest_view,
)
from zlc_frontend.qt_widgets import QtImageBoard, QtRasterBoard
from zlc_frontend.matplotlib_render import release_agg_figure
from zlc_neutral_atom.readout.calibration import (
    GridOrder,
    PerSitePsfFeature,
    site_grid_positions_yx,
)


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "pulses" / "probe_template.json"


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def capture_product(tmp_path_factory):
    with zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("w4-figure-workspace"),
    ) as experiment:
        yield experiment, experiment.readout.capture(PULSE)


@pytest.fixture(scope="module")
def calibration_product(capture_product):
    experiment, _capture_reference = capture_product
    reference = experiment.readout.sitemap(frames=12)
    computation = experiment.readout.load_calibration_computation(reference)
    return experiment, reference, computation


@pytest.fixture(scope="module")
def occupancy_product(capture_product, calibration_product):
    from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse

    experiment, _unrelated_capture = capture_product
    _same_experiment, calibration_reference, _computation = calibration_product
    document = load_packaged_sitemap_pulse()
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
        name="w4-occupancy-readout",
        periods=tuple(periods),
        repeat=None,
    )
    capture_reference = experiment.readout.capture(
        readout_document,
        trigger_channel="ch11",
        readout_events_per_repeat=1,
    )
    request = experiment.readout.detection_request(
        capture_reference,
        calibration_reference,
    )
    reference = experiment.readout.detect(request)
    return experiment, reference, experiment.readout.load_occupancy(reference)


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.close()
    _until(application, lambda: window.closed and not window.isVisible(), timeout=5.0)
    assert window not in getattr(application, "_zlc_retained_windows", ())


def _axis(name, role, size, coordinates) -> AxisSpec:
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(coordinates),
        None,
        None,
    )


def _faceted_curve_figure() -> DataFigure:
    repeat = _axis("repeat", REPEAT, 1, (0,))
    scan = _axis("detuning", SCAN_POINT, 4, (-2.0, -0.5, 1.0, 3.0))
    site = _axis("site", SITE, 2, ("left", "right"))
    values = np.asarray(
        (((1.0, 3.0), (2.0, 4.0), (3.0, 5.0), (4.0, 6.0)),)
    )
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
        ),
    )
    block = DataBlock(
        BlockId("w4-faceted-curve"),
        DatasetRevision(1),
        values,
        ComponentValidity((site.axis_id,), np.ones(values.shape, dtype=np.bool_)),
        schema,
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(facet_axis_ids=(site.axis_id,)),
    )
    assert suggestion.spec is not None
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "w4-faceted-document",
        0,
        (DatasetDescriptor(dataset_id, "site curves", schema.fingerprint),),
        (FigureLayer("data", dataset_id, suggestion.spec),),
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("w4-faceted-generation")),
        block,
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        render_memory_limit_bytes=64 << 20,
    )


def test_public_figure_gui_resolves_and_renders_off_the_qt_owner(
    application,
    capture_product,
    monkeypatch,
):
    experiment, capture_ref = capture_product
    owner_thread = threading.get_ident()
    resolution_started = threading.Event()
    release_resolution = threading.Event()
    render_started = threading.Event()
    release_render = threading.Event()
    resolution_calls = []
    render_threads: list[int] = []
    original_figure = type(experiment).figure
    original_render = DataFigure.to_png_bytes

    def gated_resolution(self, source, *args, **kwargs):
        resolution_calls.append((threading.get_ident(), source, args, kwargs))
        resolution_started.set()
        if not release_resolution.wait(2.0):
            raise TimeoutError("test did not release figure resolution")
        return original_figure(self, source, *args, **kwargs)

    def gated_render(self, *args, **kwargs):
        render_threads.append(threading.get_ident())
        render_started.set()
        if not release_render.wait(2.0):
            raise TimeoutError("test did not release figure render")
        return original_render(self, *args, **kwargs)

    monkeypatch.setattr(type(experiment), "figure", gated_resolution)
    monkeypatch.setattr(DataFigure, "to_png_bytes", gated_render)
    preferences = ViewPreferences()
    render_limit = 96 << 20
    window = None
    try:
        window = experiment.figure_gui(
            capture_ref,
            intent=ViewIntent.IMAGE,
            preferences=preferences,
            memory_limit_bytes=render_limit,
        )
        assert resolution_started.wait(2.0)
        assert not release_resolution.is_set()
        assert len(resolution_calls) == 1
        resolution_thread, source, args, options = resolution_calls[0]
        assert resolution_thread != owner_thread
        assert source == capture_ref and args == ()
        assert options == {
            "intent": ViewIntent.IMAGE,
            "selection": None,
            "preferences": preferences,
            "memory_limit_bytes": render_limit,
        }
        release_resolution.set()
        assert render_started.wait(2.0)
        assert len(render_threads) == 1
        assert render_threads[0] != owner_thread
        assert window.isVisible()
        release_render.set()
        _until(application, lambda: window.raster_ready)
        assert window.findChild(QtWidgets.QLabel, "figureViewerStatus").text() == "READY"
        assert window.findChild(QtWidgets.QLabel, "figureViewerMode").text() == (
            "FROZEN DATA FIGURE · DISPLAY ONLY"
        )
        assert "image" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerSummary",
        ).text()
        available = application.primaryScreen().availableGeometry()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert available.contains(window.frameGeometry())
        _close(application, window)
        assert experiment.figure_document(capture_ref).layers
    finally:
        release_resolution.set()
        release_render.set()
        if window is not None and window.isVisible():
            _close(application, window)


def test_shared_viewer_presents_one_coherent_multi_panel_front(
    application,
    monkeypatch,
):
    figure = _faceted_curve_figure()
    assert len(figure.evaluated.layers[0].cells) == 2
    calls = []
    original = QtImageBoard.present_encoded

    def traced_present(self, payload, *, image_format="PNG"):
        calls.append((payload, image_format))
        return original(self, payload, image_format=image_format)

    monkeypatch.setattr(QtImageBoard, "present_encoded", traced_present)
    window = open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready)
        summary = window.findChild(QtWidgets.QLabel, "figureViewerSummary").text()
        board = window.findChild(QtImageBoard, "figureViewerBoard")
        assert "curve" in summary
        assert "2 panel(s)" in summary
        assert board.has_front
        assert len(calls) == 1
        assert isinstance(calls[0][0], bytes)
        assert calls[0][0].startswith(b"\x89PNG\r\n\x1a\n")
        assert calls[0][1] == "PNG"
        _close(application, window)
    finally:
        if window.isVisible():
            _close(application, window)


def test_memory_rejection_never_reaches_qt_decode(application, monkeypatch):
    figure = _faceted_curve_figure()
    payload = figure.to_png_bytes(memory_limit_bytes=64 << 20)
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload[12:16] == b"IHDR"
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    decode_peak = len(payload) + 2 * width * height * 4
    assert width > 0 and height > 0 and decode_peak > 1
    calls = []

    def forbidden_decode(self, payload, *, image_format="PNG"):
        calls.append((self, payload, image_format))

    monkeypatch.setattr(QtImageBoard, "present_encoded", forbidden_decode)
    monkeypatch.setattr(DataFigure, "to_png_bytes", lambda self, **kwargs: payload)
    window = open_data_figure_workbench(
        figure,
        memory_limit_bytes=decode_peak - 1,
    )
    try:
        _until(application, lambda: window.worker_idle)
        status = window.findChild(QtWidgets.QLabel, "figureViewerStatus")
        diagnostic = window.findChild(QtWidgets.QLabel, "figureViewerDiagnostic")
        assert status.text() == "FIGURE FAILED"
        assert "MemoryError" in diagnostic.text()
        assert calls == []
        assert not window.raster_ready
        _close(application, window)
    finally:
        if window.isVisible():
            _close(application, window)


def test_physical_board_budget_is_checked_before_qt_decode(
    application,
    monkeypatch,
):
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    assert one_pixel_png[:8] == b"\x89PNG\r\n\x1a\n"
    calls = []
    monkeypatch.setattr(
        DataFigure,
        "to_png_bytes",
        lambda self, **kwargs: one_pixel_png,
    )
    monkeypatch.setattr(
        QtImageBoard,
        "present_encoded",
        lambda self, payload, *, image_format="PNG": calls.append(payload),
    )
    window = open_data_figure_workbench(
        _faceted_curve_figure(),
        memory_limit_bytes=1024,
    )
    try:
        _until(application, lambda: window.worker_idle)
        status = window.findChild(QtWidgets.QLabel, "figureViewerStatus")
        diagnostic = window.findChild(QtWidgets.QLabel, "figureViewerDiagnostic")
        assert status.text() == "DISPLAY FAILED"
        assert "Qt figure presentation requires" in diagnostic.text()
        assert calls == []
        assert not window.raster_ready
        _close(application, window)
    finally:
        if window.isVisible():
            _close(application, window)


def test_close_does_not_wait_for_an_inflight_render(
    application,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    present_calls = []
    render_calls = []
    original = DataFigure.to_png_bytes
    original_present = QtImageBoard.present_encoded

    def gated_render(self, *args, **kwargs):
        render_calls.append(self)
        started.set()
        if not release.wait(5.0):
            raise TimeoutError("test did not release closing render")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataFigure, "to_png_bytes", gated_render)

    def traced_present(self, payload, *, image_format="PNG"):
        present_calls.append((payload, image_format))
        return original_present(self, payload, image_format=image_format)

    monkeypatch.setattr(QtImageBoard, "present_encoded", traced_present)
    window = open_data_figure_workbench(_faceted_curve_figure())
    queued_window = None
    try:
        assert started.wait(5.0)
        queued_window = open_data_figure_workbench(_faceted_curve_figure())
        application.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.02)
        assert len(render_calls) == 1
        before = time.monotonic()
        queued_window.close()
        assert time.monotonic() - before < 0.1
        _until(
            application,
            lambda: queued_window.closed and not queued_window.isVisible(),
        )
        before = time.monotonic()
        window.close()
        assert time.monotonic() - before < 0.1
        assert not window.closed
        assert window.isVisible()
        release.set()
        _until(application, lambda: window.closed and not window.isVisible())
        assert window.worker_idle
        assert not window.raster_ready
        assert present_calls == []
        assert window not in getattr(application, "_zlc_retained_windows", ())
    finally:
        release.set()
        if queued_window is not None and queued_window.isVisible():
            _close(application, queued_window)
        if window.isVisible():
            _close(application, window)


def test_paired_calibration_load_projects_authoritative_arrays_without_reshape(
    calibration_product,
):
    _experiment, reference, computation = calibration_product
    from Zou_lab_control.workbench._calibration import _project_calibration

    assert computation.artifact.source_binding.source_capture_ref
    assert computation.report.request.grid_shape_yx == (5, 7)
    view = _project_calibration(computation)
    artifact = computation.artifact
    report = computation.report
    assert view.actual_centers_xy.shape == (35, 2)
    assert view.grid_shape_yx == (5, 7)
    assert view.site_grid_positions_yx == site_grid_positions_yx(
        (5, 7),
        computation.artifact.site_map.ordering,
    )
    assert view.occupied_labels.shape == report.labels.occupied.shape
    assert view.dark_labels.shape == report.labels.dark.shape
    assert view.label_validity.shape == report.labels.valid.shape
    np.testing.assert_array_equal(view.occupied_labels, report.labels.occupied)
    np.testing.assert_array_equal(view.dark_labels, report.labels.dark)
    np.testing.assert_array_equal(view.label_validity, report.labels.valid)
    np.testing.assert_allclose(view.actual_centers_xy, artifact.site_map.coordinates_xy)
    for projected, stored_model, stored_report in zip(
        view.models,
        artifact.models,
        report.models,
        strict=True,
    ):
        assert projected.signals.shape == report.labels.valid.shape
        np.testing.assert_array_equal(projected.signals, stored_report.short_signals)
        np.testing.assert_array_equal(
            projected.signal_validity,
            stored_report.short_validity,
        )
        np.testing.assert_array_equal(
            projected.runtime_thresholds,
            stored_model.thresholds,
        )
        np.testing.assert_array_equal(
            projected.runtime_usable,
            stored_model.usable_sites.mask,
        )
        assert len(projected.runtime_threshold_sources) == 35
    per_site = next(
        model for model in artifact.models
        if isinstance(model.feature, PerSitePsfFeature)
    )
    np.testing.assert_array_equal(view.psf_kernels, per_site.feature.kernels)
    assert reference.target_ref.startswith("calibration/")


def test_calibration_histogram_rejects_finite_invalid_filler(
    calibration_product,
    monkeypatch,
):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from zlc_frontend.calibration_render import _build_histogram_grid
    from Zou_lab_control.workbench._calibration import _project_calibration

    view = _project_calibration(calibration_product[2])
    model = view.models[0]
    sentinel = 9.87654321e99
    signals = np.array(model.signals, copy=True)
    signal_validity = np.array(model.signal_validity, copy=True)
    signals[0, 0] = sentinel
    signal_validity[0, 0] = False
    model = replace(
        model,
        signals=signals,
        signal_validity=signal_validity,
    )
    captured = []
    original_hist = Axes.hist

    def traced_hist(self, values, *args, **kwargs):
        captured.extend(np.asarray(values, dtype=float).tolist())
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", traced_hist)
    figure = Figure()
    _build_histogram_grid(view, model, figure)
    assert captured
    assert sentinel not in captured


def test_calibration_render_uses_one_source_plus_render_budget_before_agg(
    calibration_product,
    monkeypatch,
):
    import zlc_frontend.calibration_render as calibration_render
    from Zou_lab_control.workbench._calibration import _project_calibration

    view = _project_calibration(calibration_product[2])
    allocations = []
    monkeypatch.setattr(
        calibration_render,
        "_new_figure",
        lambda *_args, **_kwargs: allocations.append(True),
    )
    source_retained = 3 << 20
    render_without_source = (
        calibration_render._RENDER_FIXED_BYTES
        + calibration_render._RASTER_PEAK_MULTIPLIER * 1800 * 1100 * 4
        + calibration_render._ARRAY_PEAK_MULTIPLIER * view.array_nbytes
    )
    with pytest.raises(MemoryError, match="composition requires"):
        calibration_render.render_calibration_report(
            view,
            memory_limit_bytes=render_without_source + source_retained - 1,
            source_retained_upper_bound_bytes=source_retained,
        )
    assert allocations == []


def test_calibration_figure_release_remains_inside_matplotlib_lane(
    calibration_product,
    monkeypatch,
):
    import zlc_frontend.calibration_render as calibration_render
    from Zou_lab_control.workbench._calibration import _project_calibration

    events = []

    @contextmanager
    def traced_lane():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def traced_release(_figure):
        events.append("release")
        assert events[0] == "enter" and "exit" not in events

    monkeypatch.setattr(calibration_render, "render_style_context", traced_lane)
    monkeypatch.setattr(calibration_render, "release_agg_figure", traced_release)
    calibration_render._render_page(
        _project_calibration(calibration_product[2]),
        width=80,
        height=60,
        retained_png_bytes=0,
        source_retained_upper_bound_bytes=1,
        memory_limit_bytes=64 << 20,
        builder=lambda _figure: None,
        checkpoint=lambda: None,
    )
    assert events == ["enter", "release", "exit"]


def test_threshold_provenance_uses_the_domain_gate_not_numeric_equality(
    calibration_product,
):
    from zlc_neutral_atom.readout.analysis import (
        calibration_runtime_threshold_sources,
    )

    report = calibration_product[2].report
    first = replace(
        report.models[0],
        quick_thresholds=report.models[0].thresholds,
    )
    blocked = replace(
        report.models[-1],
        site_fidelity=tuple(
            replace(item, model_fidelity=float("nan"))
            for item in report.models[-1].site_fidelity
        ),
    )
    modified = replace(report, models=(first, *report.models[1:-1], blocked))
    sources = calibration_runtime_threshold_sources(modified)
    assert all(source == "quick-fallback" for model in sources for source in model)


def test_calibration_view_accepts_unusable_nan_and_omits_unusable_pool_samples(
    calibration_product,
    monkeypatch,
):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from zlc_frontend.calibration_render import _build_overview
    from zlc_frontend.render_style import render_style_context
    from Zou_lab_control.workbench._calibration import _project_calibration

    view = _project_calibration(calibration_product[2])
    model = next(item for item in view.models if item.is_default)
    signals = np.array(model.signals, copy=True)
    signal_validity = np.array(model.signal_validity, copy=True)
    thresholds = np.array(model.runtime_thresholds, copy=True)
    usable = np.array(model.runtime_usable, copy=True)
    nan_threshold_sentinel = 8.7654321e98
    finite_threshold_sentinel = 7.654321e97
    signals[:, 0] = nan_threshold_sentinel
    signals[:, 1] = finite_threshold_sentinel
    signal_validity[:, :2] = True
    usable[0] = False
    usable[1] = False
    thresholds[0] = np.nan
    modified_model = replace(
        model,
        signals=signals,
        signal_validity=signal_validity,
        runtime_thresholds=thresholds,
        runtime_usable=usable,
    )
    modified = replace(
        view,
        models=tuple(
            modified_model if item is model else item for item in view.models
        ),
    )
    captured = []
    original_hist = Axes.hist

    def traced_hist(self, values, *args, **kwargs):
        captured.extend(np.asarray(values, dtype=float).tolist())
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", traced_hist)
    with render_style_context():
        _build_overview(modified, Figure())
    assert captured
    assert nan_threshold_sentinel not in captured
    assert finite_threshold_sentinel not in captured


def test_calibration_grids_follow_every_declared_grid_order(calibration_product):
    from matplotlib.figure import Figure
    from zlc_frontend.calibration_render import _build_histogram_grid
    from Zou_lab_control.workbench._calibration import _project_calibration

    view = _project_calibration(calibration_product[2])
    model = view.models[0]
    for ordering in GridOrder:
        positions = site_grid_positions_yx(view.grid_shape_yx, ordering)
        figure = Figure()
        _build_histogram_grid(
            replace(view, site_grid_positions_yx=positions),
            model,
            figure,
        )
        columns = view.grid_shape_yx[1]
        for site, (row, column) in enumerate(positions):
            assert view.site_labels[site] in figure.axes[row * columns + column].get_title()


def test_public_calibration_report_gui_loads_and_renders_off_qt_owner(
    application,
    calibration_product,
    monkeypatch,
):
    experiment, reference, computation = calibration_product
    readout = experiment.readout
    owner_thread = threading.get_ident()
    loader_calls = []
    present_calls = []
    original_present = QtImageBoard.present_encoded

    def traced_load(self, candidate, *, memory_limit_bytes):
        loader_calls.append((threading.get_ident(), candidate, memory_limit_bytes))
        return computation, 8 << 20

    def traced_present(self, payload, *, image_format="PNG"):
        present_calls.append((threading.get_ident(), self.objectName(), payload))
        return original_present(self, payload, image_format=image_format)

    monkeypatch.setattr(
        type(readout),
        "_load_calibration_report_source",
        traced_load,
    )
    monkeypatch.setattr(QtImageBoard, "present_encoded", traced_present)
    window = readout.calibration_report_gui(reference)
    try:
        _until(application, lambda: window.raster_ready, timeout=45.0)
        assert len(loader_calls) == 1
        assert loader_calls[0][0] != owner_thread
        assert loader_calls[0][1] == reference
        assert loader_calls[0][2] == 512 << 20
        assert window.findChild(
            QtWidgets.QLabel,
            "calibrationReportMode",
        ).text() == "FROZEN CALIBRATION REPORT · DISPLAY ONLY"
        assert window.findChild(
            QtWidgets.QLabel,
            "calibrationReportStatus",
        ).text() == "READY"
        summary = window.findChild(
            QtWidgets.QLabel,
            "calibrationReportSummary",
        ).text()
        assert reference.target_ref in summary
        assert computation.artifact.source_binding.source_capture_ref.target_ref in summary
        assert "35 sites · 3 models" in summary
        tabs = window.findChild(QtWidgets.QTabWidget, "calibrationReportTabs")
        assert [tabs.tabText(index) for index in range(tabs.count())] == [
            "Overview",
            "box",
            "psf",
            "uniform_psf",
            "PSF kernels",
        ]
        assert len(present_calls) == tabs.count()
        assert all(thread_id == owner_thread for thread_id, _name, _payload in present_calls)
        assert all(payload.startswith(b"\x89PNG\r\n\x1a\n") for _thread, _name, payload in present_calls)
        assert all(
            board.has_front
            for board in window.findChildren(QtImageBoard)
            if board.objectName().startswith("calibrationReportBoard_")
        )
        _close(application, window)
        assert experiment.readout.load_calibration_computation(
            reference,
            memory_limit_bytes=512 << 20,
        ) is computation
    finally:
        if window.isVisible():
            _close(application, window)


def test_occupancy_document_uses_metadata_only_exact_output_schemas(
    occupancy_product,
    capture_product,
    monkeypatch,
):
    from zlc_neutral_atom.readout.occupancy_repository import OccupancyRepository

    experiment, reference, resolved = occupancy_product
    artifact = resolved.artifact

    def forbidden_admit(*_args, **_kwargs):
        raise AssertionError("figure_document must not materialize occupancy arrays")

    monkeypatch.setattr(OccupancyRepository, "admit", forbidden_admit)
    occupied = experiment.figure_document(reference)
    assert len(occupied.datasets) == len(occupied.layers) == 1
    assert occupied.datasets[0].schema_fingerprint == (
        artifact.occupied.schema.fingerprint
    )
    assert "occupancy occupied" in occupied.datasets[0].label
    occupied_view = occupied.layers[0].view
    assert occupied_view.intent is ViewIntent.METER
    repeat_binding = occupied_view.binding(
        artifact.occupied.schema.repeat_axis.axis_id
    )
    assert repeat_binding.role is AxisViewRole.REDUCED
    assert repeat_binding.reduction is not None
    assert repeat_binding.reduction.method is DisplayReductionMethod.MEAN
    site_axis = artifact.occupied.schema.cell_schema.data_axes[0]
    assert occupied_view.binding(site_axis.axis_id).role is AxisViewRole.FACET

    counts = experiment.figure_document(reference, occupancy_output="counts")
    assert counts.datasets[0].schema_fingerprint == artifact.counts.schema.fingerprint
    assert counts.layers[0].view.intent is ViewIntent.HISTOGRAM
    assert "occupancy counts" in counts.datasets[0].label

    def forbidden_inspect(*_args, **_kwargs):
        raise AssertionError("invalid output must fail before repository inspection")

    monkeypatch.setattr(OccupancyRepository, "inspect_final", forbidden_inspect)
    with pytest.raises(ValueError, match="occupancy_output"):
        experiment.figure_document(reference, occupancy_output="flattened")
    with pytest.raises(ValueError, match="only for OccupancyArtifactRef"):
        experiment.figure_document(
            capture_product[1],
            occupancy_output="occupied",
        )


def test_occupancy_figure_preserves_selected_block_lineage_validity_and_axes(
    occupancy_product,
):
    experiment, reference, resolved = occupancy_product
    artifact = resolved.artifact
    occupied = experiment.figure(reference)
    assert occupied.evaluated.inputs[0].ref == artifact.occupied_snapshot.ref
    assert artifact.occupied.values.shape == (
        artifact.occupied.schema.repeat_axis.size,
        artifact.occupied.schema.point_layout.storage_size,
        artifact.occupied.schema.cell_schema.data_axes[0].size,
    )
    layer = occupied.evaluated.layers[0]
    site_axis = artifact.occupied.schema.cell_schema.data_axes[0]
    raw_values = artifact.occupied.values
    raw_validity = artifact.occupied.validity.mask
    for cell in layer.cells:
        site = next(
            address.index
            for address in cell.facet_address
            if address.axis_id == site_axis.axis_id
        )
        meter = cell.series[0].data
        assert isinstance(meter, EvaluatedMeter)
        valid = raw_validity[:, 0, site]
        expected = raw_values[:, 0, site][valid]
        assert meter.valid is bool(expected.size)
        if expected.size:
            assert float(meter.value) == pytest.approx(float(np.mean(expected)))

    counts = experiment.figure(reference, occupancy_output="counts")
    assert counts.evaluated.inputs[0].ref == artifact.counts_snapshot.ref
    assert counts.document.datasets[0].schema_fingerprint == (
        artifact.counts.schema.fingerprint
    )
    first_cell = counts.evaluated.layers[0].cells[0]
    first_site = next(
        address.index
        for address in first_cell.facet_address
        if address.axis_id == site_axis.axis_id
    )
    histogram = first_cell.series[0].data
    assert isinstance(histogram, EvaluatedHistogram)
    valid = artifact.counts.validity.mask[:, 0, first_site]
    np.testing.assert_array_equal(
        histogram.samples,
        artifact.counts.values[:, 0, first_site][valid],
    )


def test_occupancy_preflight_rejects_known_peak_before_full_dependency_admit(
    occupancy_product,
    monkeypatch,
):
    import zlc_neutral_atom.readout.occupancy_repository as occupancy_repository
    from zlc_neutral_atom.artifacts.capture import CaptureRepository
    from zlc_neutral_atom.runtime.run import RunFailed

    experiment, _reference, resolved = occupancy_product
    artifact = resolved.artifact
    request = experiment.readout.detection_request(
        artifact.source_capture_ref,
        artifact.calibration_reference,
        model_kind=artifact.model_kind,
    )
    monkeypatch.setattr(
        occupancy_repository,
        "_estimate_committed_occupancy_peak_from_footprints",
        lambda **_kwargs: request.memory_limit_bytes + 1,
    )
    admits = []

    def forbidden_admit(*args, **kwargs):
        admits.append((args, kwargs))
        raise AssertionError("known-over-budget preflight must not admit dependencies")

    monkeypatch.setattr(CaptureRepository, "admit", forbidden_admit)
    with pytest.raises(RunFailed, match="occupancy analysis peak"):
        experiment.readout.detect(request)
    assert admits == []


def test_boolean_histogram_keeps_false_and_true_as_two_categories(
    occupancy_product,
    monkeypatch,
):
    import zlc_frontend.matplotlib_render as render_module

    experiment, reference, resolved = occupancy_product
    site_axis = resolved.artifact.occupied.schema.cell_schema.data_axes[0]
    figure = experiment.figure(
        reference,
        occupancy_output="occupied",
        intent=ViewIntent.HISTOGRAM,
        selection=Selection.index(site_axis.axis_id, 0),
    )
    assert isinstance(figure.evaluated.layers[0].cells[0].series[0].data, EvaluatedHistogram)
    observed_bins = []
    original_projection = render_module.HistogramBinProjection

    def traced_projection(values, bins=60):
        projection = original_projection(values, bins=bins)
        observed_bins.append(tuple(projection.bin_edges))
        return projection

    monkeypatch.setattr(
        render_module,
        "HistogramBinProjection",
        traced_projection,
    )
    rendered = figure.render()
    try:
        assert observed_bins == [(-0.5, 0.5, 1.5)]
        assert [tick.get_text() for tick in rendered.axes[0].get_xticklabels()] == [
            "false",
            "true",
        ]
    finally:
        release_agg_figure(rendered)


def test_public_occupancy_figure_gui_forwards_selected_output_off_qt_owner(
    application,
    occupancy_product,
    monkeypatch,
):
    experiment, reference, resolved = occupancy_product
    site_axis = resolved.artifact.counts.schema.cell_schema.data_axes[0]
    selection = Selection.index(site_axis.axis_id, 0)
    owner_thread = threading.get_ident()
    calls = []
    original = type(experiment).figure

    def traced(self, source, *args, **kwargs):
        calls.append((threading.get_ident(), source, args, kwargs))
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(type(experiment), "figure", traced)
    window = experiment.figure_gui(
        reference,
        occupancy_output="counts",
        intent=ViewIntent.HISTOGRAM,
        selection=selection,
    )
    try:
        _until(application, lambda: window.raster_ready, timeout=45.0)
        assert len(calls) == 1
        thread_id, source, args, options = calls[0]
        assert thread_id != owner_thread
        assert source == reference and args == ()
        assert options["occupancy_output"] == "counts"
        assert options["intent"] is ViewIntent.HISTOGRAM
        assert options["selection"] == selection
        assert options["preferences"] is None
        assert options["memory_limit_bytes"] == 512 << 20
        assert "histogram" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerSummary",
        ).text()
        _close(application, window)
        assert experiment.figure_document(reference).layers
    finally:
        if window.isVisible():
            _close(application, window)


def test_exact_occupancy_cell_selection_uses_named_axes_and_sparse_layout():
    from zlc_frontend.occupancy_render import OccupancyCellNavigation

    repeat = _axis("exact.repeat", REPEAT, 2, ("r0", "r1"))
    scan = _axis("exact.scan", SCAN_POINT, 3, (-2.0, 0.0, 4.0))
    event = _axis("exact.event", READOUT_EVENT, 1, ("readout",))
    site = _axis("exact.site", SITE, 2, ("left", "right"))
    schema = DatasetSchema(
        repeat,
        (scan, event),
        PointLayout.explicit((3, 1), ((2, 0), (0, 0))),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            np.dtype(bool),
        ),
    )
    selection = Selection(
        (
            IndexSelection(repeat.axis_id, 1),
            IndexSelection(scan.axis_id, 2),
        )
    )
    navigation = OccupancyCellNavigation(
        "occupancy/" + "0" * 64,
        schema.fingerprint,
        StreamGenerationId("exact-navigation"),
        schema.repeat_axis,
        schema.point_axes,
        schema.point_layout,
        schema.cell_layout,
        1 << 20,
        32 << 20,
    )
    repeat_index, point_storage_index, logical, label = navigation.resolve_selection(
        selection
    )
    assert (repeat_index, point_storage_index) == (1, 0)
    assert logical == (2, 0)
    assert "exact.repeat=r1" in label and "exact.scan=4" in label
    assert navigation.selection_at_linear(1) == navigation.selection_for_indices(
        0,
        (0, 0),
    )
    assert navigation.selection_at_linear(2) == navigation.selection_for_indices(
        1,
        (2, 0),
    )

    with pytest.raises(ValueError, match="explicit index"):
        navigation.resolve_selection(None)
    with pytest.raises(ValueError, match="absent from PointLayout"):
        navigation.resolve_selection(
            Selection(
                (
                    IndexSelection(repeat.axis_id, 0),
                    IndexSelection(scan.axis_id, 1),
                )
            ),
        )
    with pytest.raises(TypeError, match="only exact IndexSelection"):
        navigation.resolve_selection(
            Selection(
                (
                    IndexSelection(repeat.axis_id, 0),
                    Selection.index_range(scan.axis_id, 0, 1).terms[0],
                )
            ),
        )


def test_exact_occupancy_cell_loader_reads_one_same_address_without_full_capture(
    occupancy_product,
    monkeypatch,
):
    from zlc_neutral_atom.artifacts.capture import AdmittedCapture
    from zlc_neutral_atom.artifacts.capture_frames import CaptureFrameSource

    experiment, reference, resolved = occupancy_product
    calls = []
    original_read = CaptureFrameSource.read

    def traced_read(self, address):
        sample = original_read(self, address)
        calls.append((address, sample))
        return sample

    def forbidden_materialize(*_args, **_kwargs):
        raise AssertionError("exact-cell display must not materialize the full capture")

    monkeypatch.setattr(CaptureFrameSource, "read", traced_read)
    monkeypatch.setattr(CaptureFrameSource, "materialize", forbidden_materialize)
    monkeypatch.setattr(AdmittedCapture, "materialize_snapshot", forbidden_materialize)
    view, retained = experiment.readout._load_occupancy_cell_source(
        reference,
        None,
        memory_limit_bytes=512 << 20,
    )
    assert len(calls) == 1
    address, sample = calls[0]
    assert (address.repeat_index, address.point_storage_index) == (0, 0)
    axes = sample.image.schema.data_axes
    order_yx = (
        next(index for index, axis in enumerate(axes) if axis.role == SPATIAL_Y),
        next(index for index, axis in enumerate(axes) if axis.role == SPATIAL_X),
    )
    np.testing.assert_array_equal(
        view.background.values,
        np.transpose(sample.image.values, order_yx),
    )
    np.testing.assert_array_equal(
        view.occupied,
        resolved.artifact.occupied.values[0, 0, :],
    )
    np.testing.assert_array_equal(
        view.site_validity,
        resolved.artifact.occupied.validity.mask[0, 0, :],
    )
    assert retained >= view.array_nbytes
    assert view.background_input.ref.revision == view.occupancy_input.ref.revision
    assert view.background_input.dataset_id != view.occupancy_input.dataset_id
    assert view.home_viewport.y_axis.role == SPATIAL_Y
    assert view.home_viewport.x_axis.role == SPATIAL_X
    assert view.home_viewport.coordinate_frame == view.coordinate_frame
    assert view.calibration_identity == resolved.artifact.calibration_reference.target_ref
    assert "address=(0, 0)" in view.summary
    assert resolved.artifact.source_capture_ref.target_ref in view.summary
    assert resolved.artifact.calibration_reference.target_ref in view.summary


def test_occupancy_cell_budget_includes_all_retained_inspections(
    occupancy_product,
    monkeypatch,
):
    from zlc_frontend.occupancy_render import (
        estimate_occupancy_cell_view_retained_nbytes,
    )
    from zlc_neutral_atom.readout.occupancy_repository import OccupancyRepository

    experiment, reference, _resolved = occupancy_product
    from Zou_lab_control.notebook.facade import _service_guard, _occupancy_repository

    with _service_guard(experiment.readout._token) as app_services:
        inspected = _occupancy_repository(app_services).inspect_final(reference)
        source = app_services.capture_repository.inspect_final(
            inspected.source_capture_ref,
        )
        calibration = app_services.calibration_repository.inspect_final(
            inspected.calibration_reference,
        )
        admission_peak = _occupancy_repository(
            app_services
        ).admission_peak_upper_bound_bytes(
            reference,
            app_services.capture_repository,
            app_services.calibration_repository,
        )
    frame = source.dataset_schema.cell_schema
    sites = inspected.occupied_schema.cell_schema.data_axes[0].size
    view_bound = estimate_occupancy_cell_view_retained_nbytes(
        frame.data_shape,
        frame.dtype,
        sites,
    )
    inspection_headroom = (
        inspected.inspection_retained_upper_bound_bytes
        + source.inspection_retained_upper_bound_bytes
        + calibration.inspection_retained_upper_bound_bytes
    )
    limit = (
        admission_peak
        + view_bound
        + inspection_headroom
        - 1
    )
    assert admission_peak + view_bound <= limit
    calls = []

    def forbidden_admit(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("aggregate-over-budget display must reject before admission")

    monkeypatch.setattr(OccupancyRepository, "admit", forbidden_admit)
    with pytest.raises(MemoryError, match="dependency admission"):
        experiment.readout._load_occupancy_cell_source(
            reference,
            None,
            memory_limit_bytes=limit,
        )
    assert calls == []

def test_interactive_occupancy_required_minus_one_rejects_before_large_admission(
    occupancy_product,
    monkeypatch,
):
    from zlc_neutral_atom.artifacts.capture import CaptureRepository
    from zlc_neutral_atom.artifacts.capture_frames import CaptureFrameSource
    from zlc_neutral_atom.readout.calibration_repository import CalibrationRepository
    from zlc_neutral_atom.readout.occupancy_repository import OccupancyRepository

    experiment, reference, _resolved = occupancy_product
    navigation = experiment.readout._inspect_occupancy_cell_navigation(
        reference,
        memory_limit_bytes=512 << 20,
    )
    required = (
        navigation.retained_upper_bound_bytes
        + navigation.cell_peak_upper_bound_bytes
    )
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("insufficient SiteMap budget reached large-array admission")

    monkeypatch.setattr(OccupancyRepository, "admit", forbidden)
    monkeypatch.setattr(CalibrationRepository, "load", forbidden)
    monkeypatch.setattr(CaptureRepository, "admit", forbidden)
    monkeypatch.setattr(CaptureFrameSource, "read", forbidden)
    with pytest.raises(MemoryError, match="interactive occupancy cell requires"):
        experiment.readout._inspect_occupancy_cell_navigation(
            reference,
            memory_limit_bytes=required - 1,
        )
    assert calls == []
    exact = experiment.readout._inspect_occupancy_cell_navigation(
        reference,
        memory_limit_bytes=required,
    )
    assert exact.cell_peak_upper_bound_bytes == navigation.cell_peak_upper_bound_bytes


def test_public_exact_occupancy_cell_gui_stays_off_qt_owner(
    application,
    occupancy_product,
    monkeypatch,
):
    experiment, reference, _resolved = occupancy_product
    owner_thread = threading.get_ident()
    calls = []
    original = type(experiment.readout)._load_occupancy_cell_source

    def traced(self, *args, **kwargs):
        calls.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(experiment.readout), "_load_occupancy_cell_source", traced)
    navigation = experiment.readout._inspect_occupancy_cell_navigation(
        reference,
        memory_limit_bytes=512 << 20,
    )
    exact_limit = (
        navigation.retained_upper_bound_bytes
        + navigation.cell_peak_upper_bound_bytes
    )
    window = experiment.readout.occupancy_cell_gui(
        reference,
        memory_limit_bytes=exact_limit,
    )
    try:
        _until(application, lambda: window.raster_ready, timeout=45.0)
        assert calls and calls == [calls[0]] and calls[0] != owner_thread
        board = window.findChild(QtRasterBoard, "occupancyCellBoard")
        assert board is not None and board.visible_site_map_payload() is not None
        assert "address=(0, 0)" in window.findChild(
            QtWidgets.QLabel,
            "occupancyCellSummary",
        ).text()
        mode = window.findChild(QtWidgets.QLabel, "occupancyCellMode")
        assert "SAME-SHOT FRAME" in mode.text()
        _close(application, window)
        assert experiment.figure_document(reference).layers
    finally:
        if window.isVisible():
            _close(application, window)


def test_occupancy_navigation_inspection_reads_metadata_without_admission(
    occupancy_product,
    monkeypatch,
):
    from zlc_neutral_atom.readout.occupancy_repository import OccupancyRepository

    experiment, reference, resolved = occupancy_product
    calls = []

    def forbidden_admit(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("navigation metadata must not materialize occupancy arrays")

    monkeypatch.setattr(OccupancyRepository, "admit", forbidden_admit)
    navigation = experiment.readout._inspect_occupancy_cell_navigation(
        reference,
        memory_limit_bytes=512 << 20,
    )
    assert calls == []
    assert navigation.artifact_identity == reference.target_ref
    assert navigation.repeat_axis == resolved.artifact.occupied.schema.repeat_axis
    assert navigation.point_axes == resolved.artifact.occupied.schema.point_axes
    assert navigation.point_layout == resolved.artifact.occupied.schema.point_layout
    assert navigation.retained_upper_bound_bytes > 0
    with pytest.raises(ValueError, match="changed after navigation inspection"):
        experiment.readout._load_occupancy_cell_source(
            reference,
            None,
            memory_limit_bytes=512 << 20,
            expected_navigation=replace(
                navigation,
                generation=StreamGenerationId("stale-navigation"),
            ),
        )
    assert calls == []
