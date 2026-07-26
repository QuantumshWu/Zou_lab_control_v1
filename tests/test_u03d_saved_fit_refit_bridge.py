"""U0.3d exact saved-fit cell -> unified Figure Fit bridge."""

from __future__ import annotations

import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app
import pytest

import Zou_lab_control.api as zlc
from zlc_workbench.data_figure.window import DataFigureWindow
from zlc_workbench.fit_grid.window import SavedFitGridWindow
from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    DataTransformSpec,
    FitNumericPolicy,
    FitParameterConstraint,
    IndexRangeSelection,
    Selection,
    commit_transform,
)
from zlc_frontend import ImagePanelPayload
from zlc_neutral_atom.artifacts import FitResultRepository


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "pulses" / "imaging_template.json"
PANEL_ID = "generic-typed"


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt condition did not become true")


def _close(application, window) -> None:
    window.close()
    _until(
        application,
        lambda: window.closed and not window.isVisible(),
        timeout=10.0,
    )


def _retained_data_figure(application, grid) -> DataFigureWindow | None:
    return next(
        (
            window
            for window in getattr(application, "_zlc_retained_windows", ())
            if window is not grid and isinstance(window, DataFigureWindow)
        ),
        None,
    )


def test_saved_fit_requires_explicit_cell_then_replays_exact_ref_for_refit(
    application,
    tmp_path,
    monkeypatch,
) -> None:
    with zlc.connect("virtual", repository=tmp_path / "saved-refit") as experiment:
        capture_ref = experiment.readout.capture(PULSE)
        schema = experiment.readout.load_capture(capture_ref).frame_source.schema
        spatial = {axis.role: axis for axis in schema.cell_schema.data_axes}
        y_axis = spatial[SPATIAL_Y]
        x_axis = spatial[SPATIAL_X]
        roi = Selection(
            (
                IndexRangeSelection(y_axis.axis_id, 1, y_axis.size - 1),
                IndexRangeSelection(x_axis.axis_id, 1, x_axis.size - 1),
            )
        )
        transform = commit_transform(schema, DataTransformSpec((roi,)))
        numeric_policy = FitNumericPolicy(
            max_evaluations=3_500,
            covariance_rcond=1e-10,
        )
        execution = experiment.fit(
            capture_ref,
            model="radial_gaussian_center",
            committed_transform=transform,
            constraints=(
                FitParameterConstraint("one_over_e_radius", lower=1e-12),
            ),
            numeric_policy=numeric_policy,
        )
        original_spec = execution.result.spec
        saved_ref = execution.save()

        execute_sources = []
        opened_sources = []
        original_execute = FitResultRepository.execute
        experiment_type = type(experiment)
        original_open = experiment_type._open_fit_capable_figure_gui

        def observed_execute(
            self,
            artifacts,
            source,
            spec,
            *args,
            **kwargs,
        ):
            execute_sources.append(source)
            return original_execute(
                self,
                artifacts,
                source,
                spec,
                *args,
                **kwargs,
            )

        def observed_open(self, display_source, fit_source, **kwargs):
            opened_sources.append((display_source, fit_source, dict(kwargs)))
            return original_open(self, display_source, fit_source, **kwargs)

        monkeypatch.setattr(FitResultRepository, "execute", observed_execute)
        monkeypatch.setattr(
            experiment_type,
            "_open_fit_capable_figure_gui",
            observed_open,
        )
        grid = experiment.figure_gui(saved_ref)
        fit_window = None
        try:
            assert isinstance(grid, SavedFitGridWindow)
            _until(application, lambda: grid.worker_idle and grid.raster_ready)
            assert grid._reference == saved_ref
            assert grid._showing_page
            assert grid._current_selection is None
            assert not grid._fit_button.isEnabled()
            assert _retained_data_figure(application, grid) is None
            assert execute_sources == []

            # A one-cell page is still not an implicit selection.  The same
            # user-facing panel activation used by the explorer must focus it.
            projected, rendered = next(
                (projected, rendered)
                for projected, rendered in zip(
                    grid._page_panels,
                    grid._page_front.frame.panels,
                    strict=True,
                )
                if projected.fit_storage_index is not None
            )
            grid._board_widget.imagePanelLeftDoubleClicked.emit(rendered.panel_id)
            _until(
                application,
                lambda: grid.worker_idle
                and not grid._showing_page
                and grid._current_selection == projected.selection,
            )
            assert grid._fit_button.isEnabled()

            grid._model.resolve_selection(projected.selection)
            QtTest.QTest.mouseClick(
                grid._fit_button,
                QtCore.Qt.LeftButton,
            )
            _until(
                application,
                lambda: _retained_data_figure(application, grid) is not None,
            )
            fit_window = _retained_data_figure(application, grid)
            assert fit_window is not None
            assert len(opened_sources) == 1
            display_source, fit_source, options = opened_sources[0]
            assert display_source == capture_ref
            assert fit_source == capture_ref
            assert options["selection"] == projected.selection
            assert options["initial_selection"] == roi
            assert options["initial_fit_spec"] == original_spec
            try:
                _until(
                    application,
                    lambda: fit_window.worker_idle
                    and fit_window.raster_ready
                    and bool(fit_window.fit_models),
                    timeout=15.0,
                )
            except AssertionError:
                pytest.fail(
                    f"refit window did not become ready: "
                    f"status={fit_window._status.text()!r}, "
                    f"diagnostic={fit_window._diagnostic.text()!r}, "
                    f"family={fit_window._view_family!r}, "
                    f"models={fit_window.fit_models!r}"
                )

            # Fit opens the existing unified Figure Fit host and prepares only; it does
            # not solve until the user presses that host's explicit Fit button.
            assert execute_sources == []
            assert fit_window._fit_pane is not None
            seeded = fit_window._fit_pane.current_option().spec
            assert seeded == original_spec
            assert seeded.model_id == original_spec.model_id
            assert seeded.committed_transform == transform
            assert tuple(seeded.committed_transform.spec.operations) == (roi,)
            assert seeded.batch_axis_ids == original_spec.batch_axis_ids
            assert seeded.constraints == original_spec.constraints
            assert seeded.numeric_policy == numeric_policy
            assert {
                term.axis_id for term in projected.selection.terms
            }.issubset(set(seeded.batch_axis_ids))
            assert {
                term.axis_id for term in projected.selection.terms
            }.isdisjoint({term.axis_id for term in roi.terms})

            payload = fit_window._board_widget.visible_image_payload(PANEL_ID)
            assert isinstance(payload, ImagePanelPayload)
            assert payload.fit_overlay is None
            assert payload.evaluated_input.ref == execution.result.source_ref

            QtTest.QTest.mouseClick(
                fit_window._fit_pane.fit_button,
                QtCore.Qt.LeftButton,
            )
            _until(
                application,
                lambda: fit_window.worker_idle and fit_window.draft_ready,
            )
            assert execute_sources == [capture_ref]
            assert fit_window._fit_draft.result.spec == original_spec

            grid._show_page()
            assert grid._showing_page
            assert grid._current_selection is None
            assert not grid._fit_button.isEnabled()
        finally:
            if fit_window is not None:
                _close(application, fit_window)
            _close(application, grid)
