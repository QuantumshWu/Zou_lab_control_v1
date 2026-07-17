"""W4a frozen DataFigure Qt product-path oracles."""

from __future__ import annotations

import base64
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
    ViewIntent,
    ViewPreferences,
    suggest_view,
)
from zlc_frontend.qt_widgets import QtImageBoard


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
