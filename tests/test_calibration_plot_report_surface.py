"""Runtime-surface contracts for immutable calibration PlotReports."""

from __future__ import annotations

import os
import struct
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSourceRef,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend import (
    AxisViewRole,
    FigureSource,
    FigureIntent,
    FixedIndex,
    PlotKind,
    PlotReportDocument,
    SourceViewBinding,
    ViewIntent,
    ViewSpec,
    plot_report_page,
)
from zlc_frontend.encoded_raster import (
    EncodedRasterDocument,
    EncodedRasterPage,
    encode_raster_buffer_png,
)
from zlc_frontend.plot_layout import panel_surface_geometry
from zlc_frontend.plot_report import render_plot_report
from zlc_frontend.qt_widgets import RasterPixelRatioObserver, ensure_qt_app
from zlc_frontend.render import RasterBuffer
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)


def _document() -> PlotReportDocument:
    repeat = AxisSpec(AxisId("report.repeat"), "repeat", REPEAT, 1, (0,))
    frame = CoordinateFrameId("report.camera")
    y_axis = AxisSpec(
        AxisId("report.y"), "y", SPATIAL_Y, 2, (0, 1), "pixel", frame,
    )
    x_axis = AxisSpec(
        AxisId("report.x"), "x", SPATIAL_X, 2, (0, 1), "pixel", frame,
    )
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    values = np.asarray([[[[1.0, 2.0], [3.0, 4.0]]]], dtype="<f8")
    block = DataBlock(
        BlockId("calibration-report-surface"),
        DatasetRevision(0),
        values,
        DatasetComponentValidity(
            (y_axis.axis_id, x_axis.axis_id),
            np.ones(values.shape, dtype=np.bool_),
        ),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("calibration-report-surface")),
        block,
    )
    page = plot_report_page(
        "meter",
        figure=FigureIntent(PlotKind.IMAGE, "Image", "Counts"),
        source=FigureSource(snapshot),
    )
    return PlotReportDocument("immutable report", (page,))


def _png_size(payload: bytes) -> tuple[int, int]:
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", payload[16:24])


def _grid_report_page(cell_count: int):
    repeat = AxisSpec(AxisId("grid.repeat"), "repeat", REPEAT, 1, (0,))
    site_id = AxisId("grid.site")
    site_values = tuple(range(cell_count))
    point_table = PointTable(
        cell_count,
        (
            PointColumn(
                site_id,
                "site",
                SCAN_POINT,
                PointColumn.NUMERIC,
                site_values,
            ),
        ),
    )
    topology = GridTopology(
        (site_id,),
        (site_values,),
        tuple((index,) for index in range(cell_count)),
    )
    frame = CoordinateFrameId("grid.camera")
    y_axis = AxisSpec(
        AxisId("grid.y"), "y", SPATIAL_Y, 2, (0, 1), "pixel", frame,
    )
    x_axis = AxisSpec(
        AxisId("grid.x"), "x", SPATIAL_X, 2, (0, 1), "pixel", frame,
    )
    schema = DatasetSchema(
        repeat,
        point_table,
        topology,
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    values = np.arange(4 * cell_count, dtype="<f8").reshape(
        1,
        cell_count,
        2,
        2,
    )
    block = DataBlock(
        BlockId(f"grid-report-{cell_count}"),
        DatasetRevision(0),
        values,
        DatasetComponentValidity(
            (y_axis.axis_id, x_axis.axis_id),
            np.ones(values.shape, dtype=np.bool_),
        ),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId(f"grid-report-{cell_count}")),
        block,
    )
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.IMAGE,
        (
            SourceViewBinding(
                AxisSourceRef.tensor(repeat.axis_id),
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            SourceViewBinding(
                AxisSourceRef.grid_dimension(site_id),
                AxisViewRole.FACET,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(y_axis.axis_id),
                AxisViewRole.IMAGE_Y,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(x_axis.axis_id),
                AxisViewRole.IMAGE_X,
            ),
        ),
    )
    return plot_report_page(
        "grid",
        figure=FigureIntent(PlotKind.GRID, "Grid", "Counts", view=view),
        source=FigureSource(snapshot),
    )


def _until(application, predicate, *, timeout=5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.002)
    assert predicate()


class _TextState:
    def __init__(self, value: str):
        self._value = value

    def text(self) -> str:
        return self._value

    def setText(self, value: str) -> None:
        self._value = value


def test_plot_report_runtime_dpr_uses_frontend_surface_without_mutating_document():
    document = _document()
    authored_contract = document.pages[0].contract
    rendered = render_plot_report(document, surface_pixel_ratio=1.25)

    assert document.pages[0].contract is authored_contract
    assert authored_contract.pixel_ratio == 1.0
    assert _png_size(rendered.pages[0].png_bytes) == panel_surface_geometry(
        authored_contract.size_name,
        pixel_ratio=1.25,
    ).raster_size


def test_grid_report_uses_the_same_topology_optimal_size_as_live_grid():
    assert _grid_report_page(10).contract.size_name == "2x2"
    assert _grid_report_page(36).contract.size_name == "4x4"


def test_calibration_report_dpr_reuses_document_and_retires_old_front(monkeypatch):
    import zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_window as module
    from zlc_neutral_atom.logic_nodes.readout.calibration.ui.workbench_window import (
        CalibrationWorkbenchWindow,
    )

    application = ensure_qt_app()
    document = _document()
    load_calls = []
    render_calls = []
    first_started = threading.Event()
    release_first = threading.Event()
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    png = encode_raster_buffer_png(
        RasterBuffer(2, 2, b"\x00\x00\x00\xff" * 4)
    )

    def load_document(_loader, reference, cancelled):
        load_calls.append(reference)
        assert not cancelled.is_set()
        return document

    def render_document(candidate, ratio, cancelled):
        render_calls.append((candidate, ratio))
        if len(render_calls) == 1:
            first_started.set()
            assert release_first.wait(5.0)
        elif len(render_calls) == 3:
            refresh_started.set()
            assert release_refresh.wait(5.0)
        assert not cancelled.is_set()
        return EncodedRasterDocument(
            candidate.summary,
            (EncodedRasterPage("meter", "Meter", png),),
        )

    monkeypatch.setattr(module, "_load_calibration_report_document", load_document)
    monkeypatch.setattr(module, "_render_calibration_report", render_document)
    window = module.CalibrationReportWindow(
        lambda _reference: None,
        CalibrationArtifactRef("test/calibration.json"),
    )
    window.show()
    try:
        _until(application, first_started.is_set)
        first_target_ratio = window._report_surface_pixel_ratio + 0.25
        window._apply_report_surface_pixel_ratio(first_target_ratio)
        assert window._report_render_reason == "document"
        release_first.set()
        _until(application, lambda: window.worker_idle and window.raster_ready)
        assert len(load_calls) == 1
        assert len(render_calls) == 2
        assert render_calls[0][0] is document
        assert render_calls[1] == (document, first_target_ratio)
        assert window.report_document is document
        assert window._report_admitted_document_revision == (
            window._report_document_revision
        )
        assert isinstance(
            window._report_surface_observer,
            RasterPixelRatioObserver,
        )
        assert issubclass(
            CalibrationWorkbenchWindow,
            module.CalibrationReportSurfaceWindow,
        )

        old_boards = window._boards
        target_ratio = first_target_ratio + 0.5
        window._apply_report_surface_pixel_ratio(target_ratio)
        assert old_boards and all(not board.has_front for board in old_boards)
        assert not window.raster_ready
        assert window._report_render_active[3] == "surface"
        _until(application, refresh_started.is_set)
        assert len(load_calls) == 1
        assert window.report_document is document
        release_refresh.set()
        _until(application, lambda: window.worker_idle and window.raster_ready)
        assert len(render_calls) == 3
        assert render_calls[2] == (document, target_ratio)
        assert window.report_surface_revision == 2
    finally:
        release_first.set()
        release_refresh.set()
        window.close()
        _until(application, lambda: window.closed and not window.isVisible())


def test_workbench_surface_failure_warning_clears_after_current_retry():
    from zlc_neutral_atom.logic_nodes.readout.calibration.ui.workbench_window import (
        CalibrationWorkbenchWindow,
    )

    busy = []
    window = SimpleNamespace(
        _report_surface_restore_state=None,
        _status=_TextState("CALIBRATION READY"),
        _summary=_TextState("authoritative summary"),
        _diagnostic=_TextState("domain warning"),
        _set_busy=busy.append,
    )
    CalibrationWorkbenchWindow._report_render_started(
        window,
        1,
        "surface",
    )
    CalibrationWorkbenchWindow._report_render_failed(
        window,
        RuntimeError("first surface failed"),
        reason="surface",
    )
    assert window._status.text() == (
        "CALIBRATION READY · REPORT DISPLAY FAILED"
    )
    assert window._report_surface_restore_state is not None

    CalibrationWorkbenchWindow._report_render_started(
        window,
        2,
        "surface",
    )
    CalibrationWorkbenchWindow._report_render_succeeded(
        window,
        EncodedRasterDocument(
            "unused",
            (
                EncodedRasterPage(
                    "meter",
                    "Meter",
                    encode_raster_buffer_png(
                        RasterBuffer(1, 1, b"\x00\x00\x00\xff")
                    ),
                ),
            ),
        ),
        displayed=True,
        reason="surface",
    )
    assert window._status.text() == "CALIBRATION READY"
    assert window._summary.text() == "authoritative summary"
    assert window._diagnostic.text() == "domain warning"
    assert window._report_surface_restore_state is None
