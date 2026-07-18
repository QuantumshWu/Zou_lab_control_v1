"""W8b explicit Selection authority through Capture Fit authoring and Save."""

from __future__ import annotations

import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

import Zou_lab_control.notebook as zlc
import Zou_lab_control.notebook.facade as facade_impl
import Zou_lab_control.workbench._fit as fit_ui
from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    DataTransformSpec,
    IndexRangeSelection,
    MissingPolicy,
    ReductionMethod,
    ReductionSpec,
    Selection,
    ValidityPolicy,
    bind_fit,
    commit_transform,
    encode_fit_result_batch,
    fit_spec_for,
)


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.close()
    _until(application, lambda: window.closed and not window.isVisible())


def test_selection_authority_is_projected_fitted_and_saved_exactly(
    tmp_path,
    monkeypatch,
) -> None:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    figure_calls: list[tuple[object, Selection | None, object | None]] = []
    agg_calls: list[object] = []
    original_data_figure = facade_impl._data_figure_for_services
    original_render = fit_ui._render_figure

    def observed_data_figure(services, source, **options):
        figure_calls.append(
            (
                source,
                options.get("selection"),
                options.get("draft_fit_result"),
            )
        )
        return original_data_figure(services, source, **options)

    def observed_render(*args, **kwargs):
        agg_calls.append(args[0] if args else None)
        return original_render(*args, **kwargs)

    monkeypatch.setattr(
        facade_impl,
        "_data_figure_for_services",
        observed_data_figure,
    )
    monkeypatch.setattr(fit_ui, "_render_figure", observed_render)

    with zlc.connect("virtual", repository=tmp_path / "w8b") as experiment:
        capture_ref = experiment.readout.capture(PULSE)
        schema = experiment.readout.load_capture(capture_ref).frame_source.schema
        spatial = {axis.role: axis for axis in schema.cell_schema.data_axes}
        y_axis, x_axis = spatial[SPATIAL_Y], spatial[SPATIAL_X]
        roi = Selection(
            (
                IndexRangeSelection(y_axis.axis_id, 1, y_axis.size - 1),
                IndexRangeSelection(x_axis.axis_id, 1, x_axis.size - 1),
            )
        )
        committed = commit_transform(schema, DataTransformSpec((roi,)))
        window = experiment.fit_gui(
            capture_ref,
            model="radial_gaussian_center",
            committed_transform=committed,
        )
        try:
            _until(
                application,
                lambda: window.worker_idle
                and window.raster_ready
                and bool(window.fit_models),
            )
            source_previews = tuple(
                call
                for call in figure_calls
                if call[0] == capture_ref and call[2] is None
            )
            assert source_previews == ((capture_ref, roi, None),)
            assert window._source_selection == roi
            assert window._current_bound().spec.committed_transform == committed
            assert "AUTHORITATIVE" in window._authority_summary.text()
            assert x_axis.axis_id.value in window._authority_summary.text()
            assert y_axis.axis_id.value in window._authority_summary.text()

            form = window._constraint_form
            assert form is not None
            form.widget_for("amplitude.initial").setText("1")
            window._fit_button.click()
            _until(
                application,
                lambda: window.worker_idle
                and window.draft_ready
                and window.raster_ready,
            )
            draft = window._draft_result
            assert draft is not None
            assert draft.result.spec.committed_transform == committed
            assert any(
                constraint.parameter_name == "amplitude"
                and constraint.initial == 1.0
                for constraint in draft.result.spec.constraints
            )
            direct = experiment.fit(capture_ref, draft.result.spec)
            assert encode_fit_result_batch(direct.result) == encode_fit_result_batch(
                draft.result
            )
            draft_payload = encode_fit_result_batch(draft.result)

            window._save_button.click()
            _until(
                application,
                lambda: window.worker_idle and window.saved_reference is not None,
            )
            saved = experiment.load_fit(window.saved_reference)
            assert saved.result.spec.committed_transform == committed
            assert encode_fit_result_batch(saved.result) == draft_payload

            previews_before_clear = len(source_previews)
            window._clear_button.click()
            _until(
                application,
                lambda: window.worker_idle
                and window.raster_ready
                and window.saved_reference is None,
            )
            source_previews = tuple(
                call
                for call in figure_calls
                if call[0] == capture_ref and call[2] is None
            )
            assert len(source_previews) == previews_before_clear + 1
            assert source_previews[-1] == (capture_ref, roi, None)
        finally:
            _close(application, window)

        figure_baseline = len(figure_calls)
        agg_baseline = len(agg_calls)
        unsupported = commit_transform(
            schema,
            DataTransformSpec(
                (
                    roi,
                    ReductionSpec(
                        (schema.repeat_axis.axis_id,),
                        ReductionMethod.MEAN,
                        MissingPolicy.REQUIRE_ALL,
                        ValidityPolicy.OMIT_INVALID,
                    ),
                )
            ),
        )
        # The data kernel accepts this explicit authority.  W8b rejects it
        # solely because the current GUI cannot preview it without lying.
        bound = bind_fit(
            fit_spec_for(
                schema,
                "radial_gaussian_center",
                committed_transform=unsupported,
            ),
            schema,
        )
        assert bound.spec.committed_transform == unsupported

        rejected = experiment.fit_gui(
            capture_ref,
            model="radial_gaussian_center",
            committed_transform=unsupported,
        )
        try:
            _until(
                application,
                lambda: rejected.worker_idle
                and rejected._status.text() == "FIT PREPARATION FAILED",
            )
            assert not rejected.fit_models
            assert not rejected.raster_ready
            assert len(figure_calls) == figure_baseline
            assert len(agg_calls) == agg_baseline
        finally:
            _close(application, rejected)
