"""W5 typed committed-capture Fit draft, overlay, Save, and Clear oracles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
import Zou_lab_control.notebook.facade as facade_impl
from Zou_lab_control.workbench import open_capture_fit_workbench
from zlc_data import (
    FitCancelled,
    FitParameterConstraint,
    SPATIAL_X,
    SPATIAL_Y,
    bind_fit,
    encode_fit_result_batch,
    fit_spec_for,
)
from zlc_frontend import fit_constraint_form, fit_spec_from_form
from zlc_neutral_atom.artifacts import (
    CaptureFitResultArtifactRef,
    CaptureFitResultRepository,
    CaptureRepository,
)
from zlc_workbench.fit import CaptureFitDraftAuthority


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def capture_product(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("w5-fit-workspace")
    with zlc.connect("virtual", repository=workspace) as experiment:
        reference = experiment.readout.capture(PULSE)
        yield experiment, reference, workspace


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.close()
    _until(
        application,
        lambda: window.closed and not window.isVisible(),
        timeout=10.0,
    )
    assert window not in getattr(application, "_zlc_retained_windows", ())


def _manifest_count(workspace: Path) -> int:
    root = workspace / "fits" / "content" / "manifests" / "fit-result"
    return 0 if not root.exists() else len(tuple(root.iterdir()))


def test_fit_gui_public_imports_remain_headless() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import Zou_lab_control.notebook; "
                "import Zou_lab_control.workbench; import zlc_frontend; "
                "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') "
                "for name in sys.modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert result.returncode == 0, result.stderr


def test_fit_gui_is_worker_owned_stale_safe_explicitly_saved_and_reopenable(
    application,
    capture_product,
    monkeypatch,
) -> None:
    experiment, capture_ref, workspace = capture_product
    owner_thread = threading.get_ident()
    inspect_threads: list[int] = []
    execute_threads: list[int] = []
    execute_controls: list[tuple[object, object]] = []
    first_execute_finished = threading.Event()
    release_first_execute = threading.Event()
    save_reopen_started = threading.Event()
    release_save_reopen = threading.Event()
    inspect = CaptureRepository.inspect_final
    execute = CaptureFitResultRepository.execute
    execute_count = 0

    def observed_inspect(self, *args, **kwargs):
        inspect_threads.append(threading.get_ident())
        return inspect(self, *args, **kwargs)

    def observed_execute(self, *args, **kwargs):
        nonlocal execute_count
        execute_threads.append(threading.get_ident())
        execute_controls.append(
            (kwargs.get("cancel_check"), kwargs.get("deadline_monotonic"))
        )
        result = execute(self, *args, **kwargs)
        execute_count += 1
        if execute_count == 1:
            first_execute_finished.set()
            if not release_first_execute.wait(10.0):
                raise TimeoutError("test did not release the first Fit completion")
        return result

    monkeypatch.setattr(CaptureRepository, "inspect_final", observed_inspect)
    monkeypatch.setattr(CaptureFitResultRepository, "execute", observed_execute)
    window = experiment.fit_gui(
        capture_ref,
        model="radial_gaussian_center",
        timeout_seconds=30.0,
    )
    try:
        _until(
            application,
            lambda: window.worker_idle and bool(window.fit_models),
        )
        assert inspect_threads and inspect_threads == [inspect_threads[0]]
        assert inspect_threads[0] != owner_thread
        assert window.raster_ready
        assert "radial_gaussian_center" in window.fit_models
        assert window._model_combo.currentData() == "radial_gaussian_center"
        assert SPATIAL_X.value in window._axis_summary.text()
        assert SPATIAL_Y.value in window._axis_summary.text()
        bound = window._current_bound()
        fit_axis_specs = tuple(
            bound.effective_schema.axis(axis_id)
            for axis_id in bound.spec.fit_axis_ids
        )
        batch_axis_specs = tuple(
            bound.effective_schema.axis(axis_id)
            for axis_id in bound.spec.batch_axis_ids
        )
        assert tuple(axis.role for axis in fit_axis_specs) == (
            SPATIAL_X,
            SPATIAL_Y,
        )
        assert bound.spec.batch_axis_ids == tuple(
            axis.axis_id for axis in batch_axis_specs
        )
        assert len(batch_axis_specs) == 2
        assert not window.draft_ready
        assert window.saved_reference is None
        assert _manifest_count(workspace) == 0

        form = window._constraint_form
        assert form is not None
        assert form.keys == tuple(
            f"{parameter.name}.{field}"
            for parameter in bound.parameter_definitions
            for field in ("initial", "lower", "upper", "fixed")
        )
        form.widget_for("amplitude.lower").setText("1")
        form.widget_for("amplitude.upper").setText("0")
        window._fit_button.click()
        assert window.worker_idle
        assert window._status.text() == "FIT REQUEST INVALID"
        assert execute_threads == []
        assert _manifest_count(workspace) == 0
        form.widget_for("amplitude.lower").setText("")
        form.widget_for("amplitude.upper").setText("")

        # A disabled widget cannot be edited by a person, but a programmatic
        # automation can still mutate it.  The captured editor revision must
        # therefore revoke the late worker completion rather than accepting it.
        window._fit_button.click()
        _until(application, first_execute_finished.is_set)
        assert window._future is not None
        form.widget_for("amplitude.initial").setText("1")
        release_first_execute.set()
        _until(
            application,
            lambda: window.worker_idle
            and window._status.text() == "STALE FIT DISCARDED",
        )
        assert not window.draft_ready
        assert not window._save_button.isEnabled()
        assert _manifest_count(workspace) == 0
        form.widget_for("amplitude.initial").setText("")

        # Cancellation remains authoritative after the worker has completed
        # but before its queued Qt owner callback accepts the result.
        window._fit_button.click()
        deadline = time.monotonic() + 30.0
        while not window._future.done() and time.monotonic() < deadline:
            time.sleep(0.002)
        assert window._future.done()
        window._cancel_fit()
        _until(
            application,
            lambda: window.worker_idle and window._status.text() == "FIT CANCELLED",
        )
        assert not window.draft_ready
        assert _manifest_count(workspace) == 0

        window._fit_button.click()
        _until(application, lambda: window.worker_idle and window.draft_ready)
        assert window.raster_ready
        assert execute_threads[-1] != owner_thread
        assert all(callable(cancel) for cancel, _deadline in execute_controls)
        assert all(
            isinstance(deadline, float) and deadline > time.monotonic()
            for _cancel, deadline in execute_controls
        )
        assert not hasattr(window._draft_result, "save")
        assert not hasattr(window._draft_authority, "project")
        draft_payload = encode_fit_result_batch(window._draft_result.result)
        assert _manifest_count(workspace) == 0
        assert window._save_button.isEnabled()

        saved_render_sources = []
        original_figure_factory = window._figure_factory

        def observed_figure_factory(source, **options):
            saved_render_sources.append(source)
            if isinstance(source, CaptureFitResultArtifactRef):
                save_reopen_started.set()
                if not release_save_reopen.wait(10.0):
                    raise TimeoutError("test did not release saved artifact reopen")
            return original_figure_factory(source, **options)

        window._figure_factory = observed_figure_factory
        window._save_button.click()
        _until(application, save_reopen_started.is_set)
        assert _manifest_count(workspace) == 1
        window.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert window.isVisible()
        assert not window._closing
        assert window._status.text() == "SAVE IN PROGRESS · CLOSE DEFERRED"
        form.widget_for("amplitude.initial").setText("1")
        release_save_reopen.set()
        _until(
            application,
            lambda: window.worker_idle and window.saved_reference is not None,
        )
        saved = window.saved_reference
        assert isinstance(saved, CaptureFitResultArtifactRef)
        assert saved_render_sources == [saved]
        assert window.raster_ready
        assert window._status.text() == "FIT SAVED · EDITOR CHANGED · CLOSE AGAIN"
        assert "current editor differs" in window._summary.text()
        assert not window.draft_ready
        assert not window._save_button.isEnabled()
        assert _manifest_count(workspace) == 1
        admitted = experiment.load_fit(saved)
        assert admitted.source_capture_ref == capture_ref
        assert encode_fit_result_batch(admitted.result) == draft_payload
        form.widget_for("amplitude.initial").setText("")

        # Persistence is still successful when the post-save renderer fails;
        # the returned artifact, not the raster, is the authority.
        window._fit_button.click()
        _until(application, lambda: window.worker_idle and window.draft_ready)

        def fail_saved_figure(source, **options):
            if isinstance(source, CaptureFitResultArtifactRef):
                raise RuntimeError("deterministic saved-overlay failure")
            return original_figure_factory(source, **options)

        window._figure_factory = fail_saved_figure
        window._save_button.click()
        _until(
            application,
            lambda: window.worker_idle and window.saved_reference is not None,
        )
        failed_display_ref = window.saved_reference
        assert failed_display_ref == saved
        assert window._status.text() == "FIT SAVED · DISPLAY FAILED"
        assert not window.raster_ready
        assert encode_fit_result_batch(
            experiment.load_fit(failed_display_ref).result
        ) == draft_payload

        window._figure_factory = original_figure_factory
        window._clear_button.click()
        _until(
            application,
            lambda: window.worker_idle
            and window.raster_ready
            and window.saved_reference is None,
        )
        assert window._status.text() == "SOURCE READY"
        assert encode_fit_result_batch(experiment.load_fit(saved).result) == draft_payload
        assert _manifest_count(workspace) == 1
    finally:
        release_first_execute.set()
        release_save_reopen.set()
        _close(application, window)


def test_cancel_fit_is_cooperative_and_never_publishes(
    application,
    capture_product,
) -> None:
    experiment, capture_ref, workspace = capture_product
    schema = experiment.readout.load_capture(capture_ref).frame_source.schema
    bound = bind_fit(
        fit_spec_for(schema, "radial_gaussian_center"),
        schema,
    )
    started = threading.Event()
    worker_threads: list[int] = []
    before = _manifest_count(workspace)
    viewer = None

    def execute_fit(_spec, cancel_check, _deadline):
        worker_threads.append(threading.get_ident())
        started.set()
        while not cancel_check():
            time.sleep(0.002)
        raise FitCancelled("deterministic test cancellation")

    def unexpected_draft_figure(_result, **_options):
        raise AssertionError("a cancelled Fit must not request a draft preview")

    window = open_capture_fit_workbench(
        experiment.figure,
        unexpected_draft_figure,
        lambda: (bound,),
        execute_fit,
        lambda execution: execution.save(),
        capture_ref,
        selected_model="radial_gaussian_center",
    )
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        window._fit_button.click()
        _until(application, started.is_set)
        assert window._cancel_fit_button.isEnabled()
        viewer = experiment.figure_gui(capture_ref)
        _until(application, lambda: viewer.worker_idle and viewer.raster_ready)
        assert window._future is not None and not window._future.done()
        window._cancel_fit_button.click()
        _until(
            application,
            lambda: window.worker_idle and window._status.text() == "FIT CANCELLED",
        )
        assert worker_threads and worker_threads[0] != threading.get_ident()
        assert not window.draft_ready
        assert window.saved_reference is None
        assert _manifest_count(workspace) == before
    finally:
        if viewer is not None:
            _close(application, viewer)
        _close(application, window)


def test_real_fit_borrow_does_not_block_ordinary_figure(
    application,
    capture_product,
    monkeypatch,
) -> None:
    experiment, capture_ref, _workspace = capture_product
    entered_execute = threading.Event()
    release_execute = threading.Event()
    original_execute = CaptureFitResultRepository.execute
    window = None
    viewer = None

    def blocked_execute(self, *args, **kwargs):
        entered_execute.set()
        if not release_execute.wait(10.0):
            raise TimeoutError("test did not release the real Fit repository")
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(CaptureFitResultRepository, "execute", blocked_execute)
    try:
        window = experiment.fit_gui(
            capture_ref,
            model="radial_gaussian_center",
            timeout_seconds=30.0,
        )
        _until(application, lambda: window.worker_idle and window.raster_ready)
        window._fit_button.click()
        _until(application, entered_execute.is_set)

        # This exercises the public Experiment closures and both real worker
        # lanes.  A fake Workbench executor would not expose operation_lock
        # accidentally serializing Fit and ordinary Figure materialization.
        viewer = experiment.figure_gui(capture_ref)
        _until(application, lambda: viewer.worker_idle and viewer.raster_ready)
        assert window._future is not None and not window._future.done()
        _close(application, viewer)
        viewer = None

        release_execute.set()
        _until(
            application,
            lambda: window.worker_idle and window.draft_ready and window.raster_ready,
        )
        _close(application, window)
        window = None
    finally:
        release_execute.set()
        if viewer is not None:
            _close(application, viewer)
        if window is not None:
            _close(application, window)


def test_fit_service_borrow_rejects_close_before_touching_resources(
    capture_product,
) -> None:
    experiment, _capture_ref, _workspace = capture_product
    token = object()
    services = facade_impl._ExperimentServices(
        runtime=object(),
        capture_repository=object(),
        scan_repository=object(),
        calibration_repository_path=Path("unused-calibration"),
        calibration_repository=None,
        occupancy_repository_path=Path("unused-occupancy"),
        occupancy_repository=None,
        fit_repository=object(),
        catalog=experiment.device_catalog,
        operation_lock=threading.RLock(),
    )
    facade_impl._register(token, services)
    borrowed = threading.Event()
    release = threading.Event()
    borrow_errors: list[BaseException] = []

    def hold_fit_borrow() -> None:
        try:
            with facade_impl._fit_service_guard(token):
                borrowed.set()
                assert release.wait(10.0)
        except BaseException as error:
            borrow_errors.append(error)

    worker = threading.Thread(target=hold_fit_borrow, daemon=False)
    worker.start()
    assert borrowed.wait(1.0)
    facade = facade_impl.Experiment(
        token,
        name="lifecycle oracle",
        device_catalog=experiment.device_catalog,
    )
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="active Fit operation"):
            facade.close()
        assert time.monotonic() - started < 1.0
        assert services.state == "CLOSING"
        with pytest.raises(RuntimeError, match="closing or closed"):
            with facade_impl._fit_service_guard(token):
                raise AssertionError("a closing Experiment admitted a new Fit")
    finally:
        release.set()
        worker.join(2.0)
        with facade_impl._AUTHORITY_LOCK:
            facade_impl._AUTHORITIES.pop(token, None)
    assert not worker.is_alive()
    assert services.active_fit_operations == 0
    assert len(borrow_errors) == 1
    assert isinstance(borrow_errors[0], FitCancelled)


def test_draft_save_uses_the_injected_experiment_lifecycle_gate(
    capture_product,
) -> None:
    experiment, capture_ref, workspace = capture_product
    schema = experiment.readout.load_capture(capture_ref).frame_source.schema
    spec = fit_spec_for(schema, "radial_gaussian_center")
    execution = experiment.fit(capture_ref, spec)
    before = _manifest_count(workspace)
    save_calls = []

    def reject_after_close(candidate):
        save_calls.append(candidate)
        raise RuntimeError("Experiment is closing or closed")

    authority = CaptureFitDraftAuthority(
        lambda _spec, _cancel, _deadline: execution,
        reject_after_close,
    )
    draft = authority.execute(spec, lambda: False, time.monotonic() + 30.0)
    with pytest.raises(RuntimeError, match="closing or closed"):
        authority.save(draft)

    assert save_calls == [execution]
    assert _manifest_count(workspace) == before
    assert authority.discard(draft)


def test_fit_gui_rejects_non_capture_sources_before_opening(capture_product) -> None:
    experiment, _capture_ref, _workspace = capture_product
    with pytest.raises(TypeError, match="CaptureArtifactRef"):
        experiment.fit_gui(object())


def test_fit_constraint_presenter_round_trips_existing_authority(
    capture_product,
) -> None:
    experiment, capture_ref, _workspace = capture_product
    schema = experiment.readout.load_capture(capture_ref).frame_source.schema
    base = bind_fit(
        fit_spec_for(schema, "radial_gaussian_center"),
        schema,
    )
    constrained = bind_fit(
        replace(
            base.spec,
            constraints=(
                FitParameterConstraint(
                    "amplitude",
                    initial=2.0,
                    lower=-3.0,
                    upper=4.0,
                ),
                FitParameterConstraint("center_x", initial=1.0, fixed=1.0),
            ),
        ),
        schema,
    )
    form = fit_constraint_form(constrained)
    unchanged = {field.key: field.default for field in form.fields}

    assert unchanged["amplitude.initial"] == 2.0
    assert unchanged["amplitude.lower"] == -3.0
    assert unchanged["amplitude.upper"] == 4.0
    assert unchanged["center_x.fixed"] == 1.0
    assert fit_spec_from_form(constrained, unchanged) == constrained.spec
