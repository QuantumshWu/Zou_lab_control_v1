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
    REPEAT,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
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
from zlc_frontend.qt_widgets import QtImageBoard
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
    from matplotlib.axes import Axes

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
    original_hist = Axes.hist

    def traced_hist(self, values, *args, **kwargs):
        observed_bins.append(tuple(kwargs["bins"]))
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", traced_hist)
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
