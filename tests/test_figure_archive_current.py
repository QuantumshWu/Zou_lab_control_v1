"""Current DataFigure archive and formal FigureViewer product seam."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtCore, QtTest, QtWidgets

from zlc_data import (
    REPEAT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend.qt_widgets import (
    FluentPlotFitPanel,
    FluentPlotParameterPanel,
    ensure_qt_app,
)
from zlc_plot import AxisRef, ImagePlot, SelectorKind
from zlc_plot import NumericRange
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_window import (
    CalibrationReportWindow,
)
from zlc_workbench.data_figure.archive_io import (
    FigureArchive,
    LoadedFigureArchive,
    load_figure_archive,
    save_figure_archive,
)
from zlc_workbench.figure_viewer.app import open_figure_viewer


def _image_snapshot() -> OwnedSnapshot:
    repeat = AxisSpec(AxisId("capture.repeat"), "repeat", REPEAT, 1, (0,))
    y = AxisSpec(AxisId("camera.y"), "y", SPATIAL_Y, 8, tuple(range(8)), "pixel")
    x = AxisSpec(AxisId("camera.x"), "x", SPATIAL_X, 10, tuple(range(10)), "pixel")
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema((y, x), ValidityContract.value(), np.dtype("<u2"), "count"),
    )
    values = np.arange(np.prod(schema.physical_shape), dtype="<u2").reshape(
        schema.physical_shape
    )
    block = DataBlock(
        BlockId("viewer-image"),
        DatasetRevision(7),
        values,
        CellValidity(np.ones((1, 1), dtype=bool)),
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("viewer-generation")), block)


def _archive() -> FigureArchive:
    snapshot = _image_snapshot()
    spec = ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y"))
    return FigureArchive(
        snapshot,
        spec,
        "2x2",
        {"show_colorbar": True, "colormap": "gray"},
        {"operator": "test"},
    )


def _wait_until(application: QtWidgets.QApplication, predicate, timeout: float = 12.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        application.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Qt condition did not become true")
        time.sleep(0.01)
    application.processEvents()


def test_archive_roundtrip_delegates_schema_spec_and_validity_without_payload_hash(
    tmp_path: Path,
) -> None:
    archive = _archive()
    target = save_figure_archive(archive, tmp_path / "image.npz")
    loaded = load_figure_archive(target)

    assert isinstance(loaded, LoadedFigureArchive)
    assert loaded.path == target
    assert loaded.archive.spec == archive.spec
    assert loaded.archive.size == "2x2"
    assert dict(loaded.archive.parameters) == dict(archive.parameters)
    assert dict(loaded.archive.metadata) == {"operator": "test"}
    assert loaded.archive.snapshot.block.values.dtype == np.dtype("<u2")
    assert np.array_equal(
        loaded.archive.snapshot.block.values,
        archive.snapshot.block.values,
    )
    assert np.array_equal(
        loaded.archive.snapshot.block.validity.mask,
        archive.snapshot.block.validity.mask,
    )

    with np.load(target, allow_pickle=False) as envelope:
        assert set(envelope.files) == {"manifest", "values", "validity_mask"}
        manifest = json.loads(str(envelope["manifest"].item()))
    manifest_text = json.dumps(manifest).lower()
    assert "sha" not in manifest_text
    assert "digest" not in manifest_text
    assert "$zlc-bytes" not in manifest_text


def test_archive_has_no_historical_reader(tmp_path: Path) -> None:
    target = tmp_path / "old.npz"
    with target.open("wb") as stream:
        np.savez(stream, payload=np.arange(3), schema=np.asarray("old"))
    with pytest.raises(ValueError, match="unexpected field set"):
        load_figure_archive(target)


def test_formal_viewer_embeds_the_same_host_for_plot_settings_fit_and_save(
    tmp_path: Path,
) -> None:
    application = ensure_qt_app()
    path = save_figure_archive(_archive(), tmp_path / "source.npz")
    viewer = open_figure_viewer(path, output_root=tmp_path)
    try:
        _wait_until(application, lambda: viewer.figure_pane is not None)
        pane = viewer.figure_pane
        plot_widget = pane.findChild(QtWidgets.QWidget, "dataFigurePlot")
        parameter_panel = pane.findChild(FluentPlotParameterPanel)
        fit_panel = pane.findChild(FluentPlotFitPanel)
        assert plot_widget.host is pane.host
        assert parameter_panel.host is pane.host
        assert fit_panel.host is pane.host
        assert fit_panel.live is False
        assert pane.status.text() == "Figure ready"

        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        application.processEvents()
        assert fit_panel.isVisible()
        outer = getattr(viewer, "_zlc_window")
        captured = outer.grab()
        assert not captured.isNull()
        assert captured.width() > 0 and captured.height() > 0

        saved: list[LoadedFigureArchive] = []
        pane.archiveSaved.connect(saved.append)
        second = tmp_path / "saved-current.npz"
        pane.save_archive(second)
        _wait_until(application, lambda: bool(saved))
        assert saved[0].path == second.resolve()
        reopened = load_figure_archive(second)
        assert reopened.archive.spec == saved[0].archive.spec
        assert dict(reopened.archive.parameters) == dict(saved[0].archive.parameters)

        accepted_archive = viewer.archive
        viewer.open_path(tmp_path / "missing.npz")
        _wait_until(
            application,
            lambda: viewer.worker_idle and viewer.info_pane.status.severity == "error",
        )
        assert viewer.figure_pane is pane
        assert viewer.archive is accepted_archive
    finally:
        _wait_until(application, viewer.teardown)
        window = getattr(viewer, "_zlc_window", None)
        if window is not None:
            window.close()
        application.processEvents()


def test_calibration_report_pages_use_shared_interactive_plot_host(
    tmp_path: Path,
) -> None:
    """Every exported calibration page remains a real interactive DataFigure.

    The report must not regress to static PNG/QLabel rendering: one selector
    operation should advance each page's shared raster front and every page
    must expose its own ordinary DataFigure host.
    """

    application = ensure_qt_app()
    report_root = tmp_path / "report"
    report_root.mkdir()
    for name in ("site_map", "fidelity", "extra"):
        archive = _archive()
        archive = FigureArchive(
            archive.snapshot,
            archive.spec,
            archive.size,
            archive.parameters,
            {"calibration_page": name},
        )
        save_figure_archive(archive, report_root / f"{name}.npz")

    report = CalibrationReportWindow(
        report_root,
        CalibrationArtifactRef("tasks/calibration/test/calibration.json"),
    )
    report.show()
    try:
        _wait_until(
            application,
            lambda: all(
                pane.plot_widget is not None and pane._initial_outcome == "ready"
                for pane in report._panes
            ),
        )
        assert len(report._panes) == 3
        assert all(pane.host is not None for pane in report._panes)
        before = [pane.host.front.identity.sequence for pane in report._panes]
        for pane in report._panes:
            operation = pane.host.set_area_selector(
                NumericRange(1.0, 4.0),
                NumericRange(1.0, 4.0),
                display=False,
            )
            _wait_until(application, operation.done)
            assert operation.exception() is None
            assert any(
                selector.kind is SelectorKind.AREA
                for selector in operation.result().front.interaction.selectors
            )
        _wait_until(
            application,
            lambda: all(
                pane.host.front.identity.sequence > previous
                for pane, previous in zip(report._panes, before, strict=True)
            ),
        )
    finally:
        _wait_until(application, report.teardown)
        report.close()
        application.processEvents()
