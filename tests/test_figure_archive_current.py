"""Current typed Figure archive and formal FigureViewer operator path."""

from __future__ import annotations

import os

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest  # noqa: E402

from test_u03b_interactive_curve_figure import _curve_figure, _until  # noqa: E402
from zlc_frontend import CurveDisplayState, load_figure_archive  # noqa: E402
from zlc_frontend.display_range import RelimMode  # noqa: E402
from zlc_frontend.qt_widgets import QtRasterBoard, ensure_qt_app  # noqa: E402
from zlc_workbench.figure_viewer.app import open_figure_viewer  # noqa: E402


@pytest.fixture
def application():
    return ensure_qt_app()


def _saved_curve(path):
    figure = _curve_figure()
    display = CurveDisplayState(
        revision=7,
        relim_mode=RelimMode.FIXED,
        fixed_y_limits=(0.0, 8.0),
        x_view=(-1.0, 1.0),
    )
    figure.save_archive(
        path,
        display=display,
        metadata={"device": "virtual"},
    )
    return figure, display


def test_archive_roundtrip_preserves_multidimensional_source_and_validity(tmp_path):
    path = tmp_path / "curve.npz"
    figure, display = _saved_curve(path)

    with np.load(path, allow_pickle=False) as raw:
        assert tuple(sorted(raw.files)) == ("payload", "schema")
        assert all(raw[name].dtype == np.uint8 for name in raw.files)

    loaded = load_figure_archive(path)
    source = figure.datasets.entries[0].snapshot
    reopened = loaded.figure.datasets.entries[0].snapshot
    assert loaded.display == display
    assert loaded.metadata["device"] == "virtual"
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


def test_formal_viewer_loads_only_on_committed_human_path_and_keeps_good_pane(
    application,
    tmp_path,
):
    path = tmp_path / "curve.npz"
    _saved_curve(path)
    viewer = open_figure_viewer()
    wrapper = viewer._zlc_window
    try:
        wrapper.show()
        application.processEvents()
        QtTest.QTest.mouseClick(viewer.path_edit.edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClicks(viewer.path_edit.edit, str(path))
        assert viewer.archive is None
        QtTest.QTest.keyClick(viewer.path_edit.edit, QtCore.Qt.Key_Return)
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
        digest = viewer.archive.payload_digest
        assert not wrapper.grab().isNull()

        missing = tmp_path / "missing.npz"
        QtTest.QTest.mouseClick(viewer.path_edit.edit, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(viewer.path_edit.edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        QtTest.QTest.keyClicks(viewer.path_edit.edit, str(missing))
        QtTest.QTest.keyClick(viewer.path_edit.edit, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: viewer.worker_idle and viewer.status.severity == "error",
        )
        assert viewer.figure_pane is pane
        assert viewer.archive.payload_digest == digest
    finally:
        wrapper.close()
        _until(application, lambda: viewer._closed)


def test_notebook_no_argument_entry_opens_the_same_session_independent_viewer(
    application,
    tmp_path,
):
    import Zou_lab_control.notebook as zlc

    experiment = zlc.connect("virtual", repository=tmp_path / "repository")
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
