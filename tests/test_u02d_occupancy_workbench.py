"""Production-path oracles for the exact interactive occupancy Workbench."""

from __future__ import annotations

from dataclasses import replace
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pytest

from zlc_data import (
    AxisId, AxisLayout, AxisSpec, BlockId, CoordinateFrameId, DatasetRevision,
    DatasetRevisionRef, PointLayout, REPEAT, SCAN_POINT, SITE, SPATIAL_X,
    SPATIAL_Y, StreamGenerationId,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import DatasetId, EvaluatedAxis, EvaluatedImage, EvaluatedInput
from zlc_frontend.image_display import image_display_form_values
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.occupancy_render import OccupancyCellNavigation, OccupancyCellView
from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: F401
from zlc_frontend.qt_widgets import (
    FluentRevisionedFormEditor, FluentSwitch, QtRasterBoard,
)
from zlc_frontend.render import SiteMapPanelPayload
from zlc_frontend.selector import RectangleGesture
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef

from Zou_lab_control.workbench._occupancy import open_occupancy_cell_workbench
from Zou_lab_control.workbench._frozen_raster import FrozenRasterWindow


_APPLICATION = None


@pytest.fixture(scope="module")
def application():
    global _APPLICATION
    _APPLICATION = ensure_qt_app()
    return _APPLICATION


def _until(application, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt condition did not become true")


def _axis(name, role, size, coordinates, *, frame=None, unit=None):
    return AxisSpec(AxisId(name), name, role, size, tuple(coordinates), unit, frame)


def _fixture(point_count=3):
    reference = OccupancyArtifactRef("test-sites", "f" * 64)
    repeat = _axis("cell.repeat", REPEAT, 1, (0,))
    point = _axis("cell.point", SCAN_POINT, point_count, range(point_count))
    navigation = OccupancyCellNavigation(
        reference.target_ref, "a" * 64, StreamGenerationId("occupancy-final"),
        repeat, (point,), PointLayout.rect_c((point_count,)),
        AxisLayout.rect_c((1, point_count)),
    )

    def view(point_index):
        frame = CoordinateFrameId("qcmos.roi0.bin1")
        y = _axis("camera.y", SPATIAL_Y, 12, range(12), frame=frame, unit="pixel")
        x = _axis("camera.x", SPATIAL_X, 12, range(12), frame=frame, unit="pixel")
        revision = DatasetRevision(point_index + 1)

        def evaluated_input(name, schema):
            return EvaluatedInput(
                DatasetId(name),
                DatasetRevisionRef(
                    BlockId(f"{name}-block"), StreamGenerationId(f"{name}-stream"),
                    schema * 64, revision,
                ),
            )

        def evaluated_axis(axis):
            return EvaluatedAxis(
                axis.axis_id, axis.name, axis.role, axis.unit,
                tuple(range(axis.size)), axis.coordinates,
            )

        values = np.arange(144, dtype=np.uint16).reshape(12, 12) + point_index
        background = EvaluatedImage(
            evaluated_axis(x), evaluated_axis(y), values, np.ones((12, 12), bool),
        )
        return OccupancyCellView(
            background, evaluated_input("camera-frame", "b"),
            evaluated_input("occupancy-state", "c"), ImageViewportTransform((y, x)),
            _axis("readout.site", SITE, 3, ("A", "B", "C")), frame,
            np.asarray(((2.0, 2.0), (5.0, 5.0), (9.0, 9.0))),
            np.asarray((False, True, False)), np.asarray((True, True, False)),
            "calibration/final", f"cell-{point_index}",
            navigation.selection_for_indices(0, (point_index,)),
            "run-real", "epoch-real",
            f"exact point {point_index}",
        )

    return reference, navigation, view


def _open(application, *, gate=None):
    reference, navigation, view_for = _fixture()
    calls = []

    def cell_loader(_reference, selection, **options):
        point = navigation.resolve_selection(selection)[2][0]
        calls.append((point, threading.get_ident(), options))
        if gate is not None and len(calls) == 1:
            gate[0].set()
            gate[1].wait(5.0)
        return view_for(point)

    window = open_occupancy_cell_workbench(
        lambda *_args, **_options: navigation, cell_loader, reference,
        selection=navigation.selection_for_indices(0, (0,)),
    )
    return window, navigation, calls


def _close(application, window):
    window.shutdown()
    _until(application, lambda: window.closed)
    _until(application, lambda: not window.isVisible())
    window.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    application.processEvents()


def test_public_exact_front_freezes_both_inputs_and_display_only_rectangle(
    application, monkeypatch,
):
    owner = threading.get_ident()
    window, navigation, calls = _open(application)
    try:
        _until(application, lambda: window.raster_ready)
        assert isinstance(window, QtWidgets.QWidget)
        assert not isinstance(window, FrozenRasterWindow)
        board = window.findChild(QtRasterBoard, "occupancyCellBoard")
        selector = window.findChild(FluentSwitch, "occupancyCellSelectorSwitch")
        assert selector.isEnabled() and not selector.isChecked()
        assert not board.selectors_enabled
        selector.setChecked(True)
        assert board.selectors_enabled
        assert "DISPLAY ONLY" not in window._mode.text()
        frame = board.front_frame
        payload = frame.panels[0].display_payload
        assert isinstance(payload, SiteMapPanelPayload)
        assert calls[0][1] != owner
        assert calls[0][2]["expected_navigation"] == navigation
        assert frame.panels[0].source_identity.dataset_id == payload.occupancy_input.dataset_id
        assert frame.panels[0].coherence_stamp.inputs == tuple(sorted(
            (payload.background.evaluated_input, payload.occupancy_input),
            key=lambda item: item.dataset_id.value,
        ))
        assert frame.panels[0].coherence_stamp.run_id == "run-real"
        assert frame.panels[0].coherence_stamp.join_key_digest == payload.join_key_digest
        monkeypatch.setattr(
            QtRasterBoard, "selection_for_rectangle_gesture",
            lambda *_args: (_ for _ in ()).throw(AssertionError("SITE authority forged")),
        )
        bounds = (0.1, 0.2, 0.6, 0.8)
        window._accept_rectangle(RectangleGesture(
            "sites", frame.board_id, frame.layout_generation, frame.sequence,
            frame.panels[0].source_identity, bounds,
            payload.background.viewport.viewport_revision,
        ))
        assert "DISPLAY ONLY rectangle" in window._summary.text()
        board._image_bindings["sites"].fault = RuntimeError(
            "synthetic selector fault"
        )
        window._owner_cycle()
        assert not selector.isChecked() and not selector.isEnabled()
        assert not board.selectors_enabled
        assert not window._edit_display.isEnabled()
        assert "failed closed" in window._diagnostic.text()
    finally:
        _close(application, window)


def test_exact_cell_view_rejects_coerced_occupancy_and_validity_facts():
    _reference, _navigation, view_for = _fixture()
    view = view_for(0)

    with pytest.raises(TypeError, match="occupied must have bool dtype"):
        replace(view, occupied=np.asarray((0, 1, 0), dtype=np.int8))
    with pytest.raises(TypeError, match="site_validity must have bool dtype"):
        replace(view, site_validity=np.asarray((1.0, 1.0, np.nan)))


def test_cell_loader_fails_closed_on_selection_drift(application):
    reference, navigation, view_for = _fixture()

    def cell_loader(_reference, selection, **options):
        requested = navigation.resolve_selection(selection)[2][0]
        assert options["expected_navigation"] == navigation
        return view_for(1 if requested == 0 else 0)

    window = open_occupancy_cell_workbench(
        lambda *_args, **_options: navigation,
        cell_loader,
        reference,
        selection=navigation.selection_for_indices(0, (0,)),
    )
    try:
        _until(application, lambda: "FAILED" in window._status.text())
        assert not window.raster_ready
        assert not window._board.has_front
        assert "different exact selection" in window._diagnostic.text()
    finally:
        _close(application, window)


def test_setting_and_edit_share_one_state_and_reraster_without_reload(application):
    window, _navigation, calls = _open(application)
    try:
        _until(application, lambda: window.raster_ready)
        visible_limits = window._board.visible_image_payload().color_limits
        values = image_display_form_values(window._display)
        values["relim_mode"] = RelimMode.FIXED
        window._apply_display_form(window._edit_display, 0, values)
        assert not window.raster_ready
        assert len(calls) == 1
        assert window._display.fixed_color_limits == visible_limits
        editors = window.findChildren(FluentRevisionedFormEditor)
        assert len(editors) == 2
        assert {editor.base_revision for editor in editors} == {1}
        _until(application, lambda: window.raster_ready)
        assert len(calls) == 1
        assert window._board.front_frame.panels[0].coherence_stamp.presentations[
            0
        ].panel_revision == 1
        assert all(editor.isEnabled() for editor in editors)
    finally:
        _close(application, window)


def test_navigation_is_latest_only_and_stale_cell_never_presents(application):
    started, release = threading.Event(), threading.Event()
    window, navigation, calls = _open(application, gate=(started, release))
    try:
        _until(application, started.is_set)
        window._queue_cell(navigation.selection_for_indices(0, (1,)))
        window._queue_cell(navigation.selection_for_indices(0, (2,)))
        release.set()
        _until(application, lambda: window.raster_ready)
        assert [call[0] for call in calls] == [0, 2]
        payload = window._board.front_frame.panels[0].display_payload
        assert payload.cell_identity == "cell-2"
        assert "exact point 2" in window._summary.text()
    finally:
        release.set()
        _close(application, window)


def test_close_is_nonblocking_and_reaps_blocked_stale_work(application):
    started, release = threading.Event(), threading.Event()
    window, _navigation, _calls = _open(application, gate=(started, release))
    _until(application, started.is_set)
    began = time.monotonic()
    window.shutdown()
    assert time.monotonic() - began < 0.2
    assert not window.raster_ready and not window.closed
    release.set()
    _until(application, lambda: window.closed)
    _until(application, lambda: not window.isVisible())
    assert not window._board.has_front
    window.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    application.processEvents()
