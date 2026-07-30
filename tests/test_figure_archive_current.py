"""Current typed Figure archive and formal FigureViewer operator path."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets  # noqa: E402

from zlc_frontend import (  # noqa: E402
    CurveDisplayState,
)
from zlc_frontend.figure import ViewIntent  # noqa: E402
from zlc_frontend.display_range import RelimMode  # noqa: E402
from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app  # noqa: E402
from zlc_workbench.figure_viewer.app import open_figure_viewer  # noqa: E402
from zlc_workbench.data_figure.archive_io import (  # noqa: E402
    load_figure_archive,
    save_figure_archive,
)


@pytest.fixture
def application():
    return ensure_qt_app()


def _until(application, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt condition did not become true")


def _curve_figure():
    from zlc_data import (
        COMPONENT,
        REPEAT,
        SCAN_POINT,
        SITE,
        AxisId,
        AxisSourceRef,
        AxisSpec,
        BlockId,
        DataBlock,
        DatasetComponentValidity,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointColumn,
        PointTable,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )
    from zlc_frontend import DataFigure
    from zlc_frontend.figure import (
        AxisViewRole,
        DatasetDescriptor,
        DatasetId,
        DisplayReduction,
        DisplayReductionMethod,
        FigureDocument,
        FigureLayer,
        ResolvedDataset,
        ResolvedDatasetMap,
        SourceViewBinding,
        ViewIntent,
        ViewSpec,
    )

    coordinates = tuple(float(value) for value in np.linspace(-2.0, 2.0, 21))
    repeat = AxisSpec(AxisId("archive.repeat"), "Repeat", REPEAT, 2, (0, 1))
    scan_id = AxisId("archive.detuning")
    site = AxisSpec(
        AxisId("archive.site"),
        "Site",
        SITE,
        3,
        ("left", "middle", "right"),
    )
    component = AxisSpec(
        AxisId("archive.component"),
        "Component",
        COMPONENT,
        2,
        ("signal", "reference"),
    )
    x = np.arange(len(coordinates), dtype=np.float64)
    sites = np.stack(
        (
            1.0 + 4.0 * np.exp(-((x - 9.0) ** 2) / 18.0),
            2.0 + 3.0 * np.exp(-((x - 11.0) ** 2) / 14.0),
            0.5 + 2.0 * np.exp(-((x - 7.0) ** 2) / 12.0),
        ),
        axis=-1,
    )
    values = np.stack(
        (np.stack((sites, 0.5 * sites + 0.3), axis=-1),) * 2,
        axis=0,
    )
    values[1] += 0.2
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[:, 4, 1, 0] = False
    schema = DatasetSchema(
        repeat,
        PointTable(
            len(coordinates),
            (
                PointColumn(
                    scan_id,
                    "Detuning",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    coordinates,
                    "MHz",
                ),
            ),
        ),
        None,
        ValueSchema(
            (site, component),
            ValidityContract.components(site.axis_id, component.axis_id),
            values.dtype,
            value_unit="photoelectron",
        ),
    )
    block = DataBlock(
        BlockId("archive-curve-block"),
        DatasetRevision(5),
        values,
        DatasetComponentValidity((site.axis_id, component.axis_id), valid),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("archive-curve-generation")),
        block,
    )
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.CURVE,
        (
            SourceViewBinding(
                AxisSourceRef.tensor(repeat.axis_id),
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            SourceViewBinding(
                AxisSourceRef.point_coordinate(scan_id),
                AxisViewRole.X,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(site.axis_id),
                AxisViewRole.BATCH,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(component.axis_id),
                AxisViewRole.BATCH,
            ),
        ),
    )
    dataset_id = DatasetId("archive-curve-dataset")
    return DataFigure(
        FigureDocument(
            "archive-curve-document",
            3,
            (DatasetDescriptor(dataset_id, "site curves", schema.fingerprint),),
            (FigureLayer("curve", dataset_id, view),),
        ),
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )


def _saved_curve(path):
    from zlc_frontend.plot_kind import PlotKind
    from zlc_frontend.plot_panel import FigureIntent

    figure = _curve_figure()
    display = CurveDisplayState(
        revision=7,
        relim_mode=RelimMode.FIXED,
        fixed_y_limits=(0.0, 8.0),
        x_view=(-1.0, 1.0),
    )
    save_figure_archive(
        figure,
        path,
        figure_intent=FigureIntent(
            PlotKind.CURVE,
            "curve",
            "value",
            view=figure.document.layers[0].view,
        ),
        size_name="2x2",
        display=display,
        metadata={"device": "virtual"},
    )
    return figure, display


def _fit_ready_curve_figure():
    """One saved Fit whose non-fit axes are uniquely selected."""

    from zlc_data import (
        REPEAT,
        SCALAR_AXIS,
        SCAN_POINT,
        AxisId,
        AxisSourceRef,
        AxisSpec,
        BlockId,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointColumn,
        PointTable,
        StreamGenerationId,
        VALID,
        ValueSchema,
    )
    from zlc_data.fit import bind_fit
    from zlc_frontend import DataFigure, prepare_fit_authoring_options
    from zlc_frontend.figure import (
        AxisViewRole,
        DatasetDescriptor,
        DatasetId,
        FixedIndex,
        FigureDocument,
        FigureLayer,
        ResolvedDataset,
        ResolvedDatasetMap,
        SourceViewBinding,
        ViewIntent,
        ViewSpec,
    )

    repeat = AxisSpec(
        AxisId("archive-local.repeat"),
        "repeat",
        REPEAT,
        1,
        (0,),
    )
    scan = AxisSpec(
        AxisId("archive-local.detuning"),
        "detuning",
        SCAN_POINT,
        31,
        tuple(float(index) for index in range(31)),
        "MHz",
    )
    x = np.asarray(scan.coordinates, dtype=np.float64)
    values = (1.5 + 6.0 * np.exp(-((x - 14.0) / 3.5) ** 2))[None, :, None]
    schema = DatasetSchema(
        repeat,
        PointTable(
            scan.size,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                    scan.unit,
                    scan.coordinate_frame,
                ),
            ),
        ),
        None,
        ValueSchema.scalar(values.dtype, "count"),
    )
    block = DataBlock(
        BlockId("archive-local-curve"),
        DatasetRevision(1),
        values,
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("archive-local-generation")),
        block,
    )
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.CURVE,
        (
            SourceViewBinding(
                AxisSourceRef.tensor(repeat.axis_id),
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            SourceViewBinding(
                AxisSourceRef.point_coordinate(scan.axis_id),
                AxisViewRole.X,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(SCALAR_AXIS.axis_id),
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
        ),
    )
    dataset_id = DatasetId("archive-local-source")
    document = FigureDocument(
        "archive-local-document",
        0,
        (DatasetDescriptor(dataset_id, "fitted curve", schema.fingerprint),),
        (FigureLayer("curve", dataset_id, view),),
    )
    source = DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )
    options = prepare_fit_authoring_options(source, None)
    gaussian = next(
        option for option in options if option.spec.model_id == "gaussian_offset"
    )
    saved_result = bind_fit(gaussian.spec, schema).run(snapshot)
    return source.with_fit_results({"curve": saved_result})


def _pick_combo_value(combo, value, application) -> None:
    """Commit one visible combo choice through keyboard navigation."""

    index = combo.findData(value)
    assert index >= 0
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents()
    QtTest.QTest.keyClick(combo, QtCore.Qt.Key_Home)
    for _ in range(index):
        QtTest.QTest.keyClick(combo, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(combo, QtCore.Qt.Key_Return)
    assert combo.currentData() == value


def _archive_has_model(path, model_id: str) -> bool:
    try:
        archive = load_figure_archive(path)
    except Exception:
        return False
    results = tuple(archive.archive.figure.fit_results.values())
    return len(results) == 1 and results[0].spec.model_id == model_id


def test_archive_roundtrip_preserves_multidimensional_source_and_validity(tmp_path):
    path = tmp_path / "curve.npz"
    figure, display = _saved_curve(path)

    with np.load(path, allow_pickle=False) as raw:
        assert tuple(sorted(raw.files)) == ("payload", "schema")
        assert all(raw[name].dtype == np.uint8 for name in raw.files)

    loaded = load_figure_archive(path)
    source = figure.datasets.entries[0].snapshot
    reopened = loaded.archive.figure.datasets.entries[0].snapshot
    assert loaded.archive.display == display
    assert loaded.archive.size_name == "2x2"
    assert loaded.archive.metadata["device"] == "virtual"
    assert reopened.ref == source.ref
    assert reopened.block.schema == source.block.schema
    assert reopened.block.values.shape == (2, 21, 3, 2)
    np.testing.assert_array_equal(reopened.block.values, source.block.values)
    np.testing.assert_array_equal(
        reopened.block.validity.mask,
        source.block.validity.mask,
    )

    old = tmp_path / "old-shape.npz"
    np.savez(old, data_x=np.arange(2), data_y=np.arange(2))
    with pytest.raises(ValueError, match="exactly"):
        load_figure_archive(old)


def test_archive_codec_preserves_faceted_histogram_display():
    from zlc_data import AxisId, AxisSourceRef
    from zlc_frontend import (
        FacetedHistogramDisplayState,
        HistogramCellThresholds,
        HistogramDisplayState,
    )
    from zlc_frontend.figure_archive import (
        _display_state_from_tree,
        _display_state_to_tree,
    )

    display = FacetedHistogramDisplayState(
        HistogramDisplayState(revision=9),
        (
                HistogramCellThresholds(
                    ((AxisSourceRef.tensor(AxisId("archive.site")), 1),),
                (12.0, 24.0),
            ),
        ),
    )

    assert _display_state_from_tree(_display_state_to_tree(display)) == display


def test_standalone_launcher_passes_output_root_without_duplicate_figures(
    monkeypatch,
    tmp_path,
):
    import figure_viewer as launcher
    import zlc_frontend.qt_widgets as qt_widgets
    import zlc_workbench.figure_viewer.app as viewer_app

    class _App:
        @staticmethod
        def exec_():
            return 0

    observed = {}

    def open_viewer(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.delenv("ZLC_FIGURE_VIEWER_AUTO_CLOSE_MS", raising=False)
    monkeypatch.setattr(qt_widgets, "ensure_qt_app", lambda: _App())
    monkeypatch.setattr(viewer_app, "open_figure_viewer", open_viewer)

    assert launcher.main(["--workspace", str(tmp_path)]) == 0
    assert observed["output_root"] == tmp_path.resolve() / "_output"


def test_formal_viewer_loads_only_on_committed_human_path_and_keeps_good_pane(
    application,
    tmp_path,
):
    path = tmp_path / "curve.npz"
    _saved_curve(path)
    viewer = open_figure_viewer(output_root=tmp_path / "output")
    wrapper = viewer._zlc_window
    path_edit = viewer.info_pane.path_edit.edit
    status = viewer.info_pane.status
    try:
        wrapper.show()
        application.processEvents()
        QtTest.QTest.mouseClick(path_edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClicks(path_edit, str(path))
        assert viewer.archive is None
        QtTest.QTest.keyClick(path_edit, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: (
                viewer.archive is not None
                and (
                    board := viewer.findChild(
                        QtRasterBoard,
                        "figureViewerTypedBoard",
                    )
                )
                is not None
                and board.front_frame is not None
            ),
        )
        pane = viewer.figure_pane
        board = viewer.findChild(
            QtRasterBoard,
            "figureViewerTypedBoard",
        )
        assert pane is not None and pane.isVisible()
        assert board is not None and board.isVisible()
        accepted_archive = viewer.archive
        assert not wrapper.grab().isNull()

        missing = tmp_path / "missing.npz"
        QtTest.QTest.mouseClick(path_edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(
            path_edit,
            QtCore.Qt.Key_A,
            QtCore.Qt.ControlModifier,
        )
        QtTest.QTest.keyClicks(path_edit, str(missing))
        QtTest.QTest.keyClick(path_edit, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: viewer.worker_idle and status.severity == "error",
        )
        assert viewer.figure_pane is pane
        assert viewer.archive is accepted_archive
    finally:
        wrapper.close()
        _until(application, lambda: viewer._closed)


def test_formal_viewer_refits_and_reopens_the_same_archive(
    application,
    tmp_path,
):
    path = tmp_path / "refitted-curve.npz"
    original = _fit_ready_curve_figure()
    from zlc_frontend.plot_kind import PlotKind
    from zlc_frontend.plot_panel import FigureIntent

    save_figure_archive(
        original,
        path,
        figure_intent=FigureIntent(
            PlotKind.CURVE,
            "fitted curve",
            "count",
            view=original.document.layers[0].view,
        ),
        size_name="2x2",
        display=CurveDisplayState(),
        metadata={"source": "formal archive refit"},
    )
    original_ref = original.datasets.entries[0].snapshot.ref

    viewer = open_figure_viewer(output_root=tmp_path / "output")
    wrapper = viewer._zlc_window
    path_edit = viewer.info_pane.path_edit.edit
    try:
        wrapper.show()
        application.processEvents()
        QtTest.QTest.mouseClick(path_edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClicks(path_edit, str(path))
        QtTest.QTest.keyClick(path_edit, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: (
                viewer.archive is not None
                and viewer.figure_pane is not None
                and viewer.figure_pane.worker_idle
                and viewer.figure_pane.raster_ready
            ),
        )
        pane = viewer.figure_pane
        fit_entry = pane.findChild(
            QtWidgets.QPushButton,
            "figureViewerFitButton",
        )
        assert fit_entry is not None and fit_entry.isEnabled()
        QtTest.QTest.mouseClick(fit_entry, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: pane.worker_idle and bool(pane.fit_models),
        )

        model = pane.findChild(QtWidgets.QComboBox, "fitAuthoringModel")
        fit = pane.findChild(
            QtWidgets.QPushButton,
            "fitAuthoringFitButton",
        )
        save = pane.findChild(
            QtWidgets.QPushButton,
            "figureViewerSaveFitButton",
        )
        assert model is not None and fit is not None and save is not None
        assert model.currentData() == "gaussian_offset"
        _pick_combo_value(model, "lorentzian", application)
        _until(
            application,
            lambda: pane.worker_idle and fit.isEnabled(),
        )
        QtTest.QTest.mouseClick(fit, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                pane.worker_idle
                and pane.draft_ready
                and pane.raster_ready
                and save.isEnabled()
            ),
        )
        QtTest.QTest.mouseClick(save, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                pane.worker_idle
                and viewer.archive is not None
                and tuple(viewer.archive.archive.figure.fit_results.values())[0]
                .spec.model_id
                == "lorentzian"
                and _archive_has_model(path, "lorentzian")
            ),
        )

        reopened = load_figure_archive(path)
        assert reopened.archive.metadata["source"] == "formal archive refit"
        assert reopened.archive.figure.datasets.entries[0].snapshot.ref == original_ref
        assert pane.saved_reference is None
        assert not wrapper.grab().isNull()
    finally:
        wrapper.close()
        _until(application, lambda: viewer._closed)


def test_formal_viewer_keeps_old_generation_when_candidate_first_render_fails(
    application,
    monkeypatch,
    tmp_path,
):
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    _saved_curve(first_path)
    _saved_curve(second_path)
    viewer = open_figure_viewer(output_root=tmp_path / "output")
    wrapper = viewer._zlc_window
    path_edit = viewer.info_pane.path_edit.edit
    status = viewer.info_pane.status
    raw_info = viewer.info_pane.raw_info
    try:
        wrapper.show()
        application.processEvents()
        QtTest.QTest.mouseClick(path_edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClicks(path_edit, str(first_path))
        QtTest.QTest.keyClick(path_edit, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: viewer.archive is not None and viewer.worker_idle,
        )
        old_pane = viewer.figure_pane
        old_archive = viewer.archive
        old_info = raw_info.toPlainText()
        old_board = old_pane.findChild(
            QtRasterBoard,
            "figureViewerTypedBoard",
        )
        assert old_board is not None and old_board.front_frame is not None
        old_front = old_board.front_frame

        import zlc_workbench.data_figure.app as workbench

        original_create = workbench.create_data_figure_pane
        candidates = []

        def tracked_create(*args, **kwargs):
            pane = original_create(*args, **kwargs)
            candidates.append(pane)
            return pane

        def reject_initial_render(*_args, **_kwargs):
            raise RuntimeError("synthetic initial render failure")

        monkeypatch.setattr(workbench, "create_data_figure_pane", tracked_create)
        monkeypatch.setattr(
            workbench,
            "render_data_figure_front",
            reject_initial_render,
        )

        QtTest.QTest.mouseClick(path_edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(
            path_edit,
            QtCore.Qt.Key_A,
            QtCore.Qt.ControlModifier,
        )
        QtTest.QTest.keyClicks(path_edit, str(second_path))
        QtTest.QTest.keyClick(path_edit, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: (
                bool(candidates)
                and candidates[0].closed
                and viewer.worker_idle
                and status.severity == "error"
            ),
        )

        assert viewer.figure_pane is old_pane
        assert viewer.archive is old_archive
        assert viewer._current_path == first_path
        assert raw_info.toPlainText() == old_info
        assert old_board.front_frame is old_front
        assert "synthetic initial render failure" in status.text()
    finally:
        wrapper.close()
        _until(application, lambda: viewer._closed)


def test_notebook_no_argument_entry_opens_the_same_session_independent_viewer(
    application,
    tmp_path,
):
    import Zou_lab_control.api as zlc

    experiment = zlc.connect(
        "virtual",
        workspace=zlc.WorkspacePaths.for_workspace(
            (tmp_path / "workspace").resolve()
        ),
    )
    viewer = None
    try:
        viewer = experiment.figure_gui()
        assert viewer.figure_pane is None
        assert viewer._zlc_window is not None
    finally:
        if viewer is not None:
            viewer._zlc_window.close()
            _until(application, lambda: viewer._closed)
        experiment.close()
