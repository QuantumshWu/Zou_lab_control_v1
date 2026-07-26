"""W6 formal calibration creation/edit, commit receipt, and report oracles."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app
import pytest

import Zou_lab_control.notebook as zlc
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    build_calibration_analysis_request_from_authoring,
    calibration_analysis_authoring_schema,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_workbench.form_projection import project_authoring_form


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture(scope="module")
def calibration_product(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("w6-calibration-workspace")
    with zlc.connect("virtual", repository=workspace) as experiment:
        reference = experiment.readout.sitemap(frames=12)
        computation = experiment.readout.load_calibration_computation(reference)
        yield experiment, reference, computation, workspace


def _until(application, predicate, *, timeout: float = 60.0) -> None:
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
        timeout=15.0,
    )
    assert window not in getattr(application, "_zlc_retained_windows", ())


def _manifest_count(workspace: Path) -> int:
    root = workspace / "calibrations" / "content" / "manifests" / "calibration"
    return 0 if not root.exists() else len(tuple(root.glob("*.manifest")))


def test_calibration_owner_presenter_and_public_imports_remain_headless() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import zlc_neutral_atom.logic_nodes.readout.calibration.calibration\n"
                "import zlc_neutral_atom.logic_nodes.readout.calibration.declaration\n"
                "import zlc_neutral_atom.logic_nodes.readout.calibration.workbench_adapter\n"
                "import zlc_workbench.form_projection\n"
                "for prefix in ('PyQt5', 'matplotlib', 'scipy'):\n"
                "    assert not any(\n"
                "        name == prefix or name.startswith(prefix + '.')\n"
                "        for name in sys.modules\n"
                "    ), prefix\n"
                "import Zou_lab_control.notebook\n"
                "import Zou_lab_control.workbench\n"
                "for prefix in ('PyQt5', 'matplotlib'):\n"
                "    assert not any(\n"
                "        name == prefix or name.startswith(prefix + '.')\n"
                "        for name in sys.modules\n"
                "    ), prefix\n"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert result.returncode == 0, result.stderr


def test_calibration_form_round_trip_preserves_spatial_authority(
    calibration_product,
) -> None:
    request = calibration_product[2].report.request
    form = project_authoring_form(calibration_analysis_authoring_schema(request))
    rebuilt = build_calibration_analysis_request_from_authoring(
        request,
        form.default_values(),
    )
    assert rebuilt == request
    assert rebuilt.layout == request.layout
    assert rebuilt.grid_shape_yx == request.grid_shape_yx
    assert rebuilt.ordering is request.ordering
    assert rebuilt.maximum_site_residual_px == request.maximum_site_residual_px
    assert rebuilt.expected_centers_xy is not None
    assert request.expected_centers_xy is not None
    assert rebuilt.expected_centers_xy.shape == request.expected_centers_xy.shape
    assert not rebuilt.expected_centers_xy.flags.writeable
    assert not any(
        key in form.keys
        for key in (
            "layout",
            "grid_shape_yx",
            "ordering",
            "expected_centers_xy",
            "maximum_site_residual_px",
        )
    )
    values = form.default_values()
    for key in tuple(key for key in values if key.startswith("model.")):
        values[key] = False
    with pytest.raises(ValueError, match="default model must remain enabled"):
        build_calibration_analysis_request_from_authoring(request, values)


def test_formal_calibration_commit_wins_close_and_render_failure_keeps_receipt(
    application,
    calibration_product,
) -> None:
    experiment, _prior, computation, workspace = calibration_product
    original = computation.report.request
    request = experiment.readout.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        replace(original, split_seed=original.split_seed + 1),
    )
    before = _manifest_count(workspace)
    owner_thread = threading.get_ident()
    committed = threading.Event()
    release_receipt = threading.Event()
    run_threads: list[int] = []
    window = experiment.readout.calibration_gui(request)
    try:
        _until(application, lambda: window.worker_idle and window.editor_ready)
        assert window.saved_reference is None
        assert window._form is not None
        form = window._form
        form.widget_for("train_fraction").setText("1")
        window._calibrate_button.click()
        assert window.worker_idle
        assert window._status.text() == "CALIBRATION REQUEST INVALID"
        assert _manifest_count(workspace) == before
        form.widget_for("train_fraction").setText("0.8")

        original_starter = window._run_starter

        def delayed_receipt_starter(*args):
            run_threads.append(threading.get_ident())
            handle = original_starter(*args)
            original_result = handle.result
            original_snapshot = handle.snapshot

            def delayed_result(timeout=None):
                reference = original_result(timeout)
                committed.set()
                if not release_receipt.wait(15.0):
                    raise TimeoutError("test did not release committed receipt")
                return reference

            def warning_snapshot():
                snapshot = original_snapshot()
                if snapshot.final_committed:
                    return replace(
                        snapshot,
                        cleanup_errors=(
                            *snapshot.cleanup_errors,
                            "deterministic post-commit cleanup warning",
                        ),
                    )
                return snapshot

            handle.result = delayed_result
            handle.snapshot = warning_snapshot
            return handle

        window._run_starter = delayed_receipt_starter
        window._calibrate_button.click()
        _until(application, committed.is_set)
        assert run_threads and run_threads[0] != owner_thread

        # A disabled form can still be changed programmatically.  The formal
        # Run keeps its captured revision while the UI marks the receipt stale.
        form.widget_for("split_seed").setText(str(original.split_seed + 2))
        window.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert window.isVisible()
        assert window.saved_reference is None
        assert window._status.text() == "CANCELLING CALIBRATION · CLOSE DEFERRED"

        def fail_report(_reference, **_options):
            raise RuntimeError("deterministic post-commit report failure")

        window._computation_loader = fail_report
        release_receipt.set()
        _until(
            application,
            lambda: window.worker_idle and window.saved_reference is not None,
        )
        saved = window.saved_reference
        assert isinstance(saved, CalibrationArtifactRef)
        assert window.isVisible()
        assert window._status.text() == (
            "CALIBRATION SAVED · RUN WARNING · REPORT DISPLAY FAILED · "
            "EDITOR CHANGED · CLOSE AGAIN"
        )
        assert "post-commit cleanup warning" in window._diagnostic.text()
        assert "post-commit report failure" in window._diagnostic.text()
        assert _manifest_count(workspace) == before + 1
        admitted = experiment.readout.load_calibration_computation(saved)
        assert admitted.report.request.train_fraction == 0.8
        assert admitted.report.request.split_seed == original.split_seed + 1
        assert admitted.artifact.source_binding.source_capture_ref == (
            request.source_capture_ref
        )
    finally:
        release_receipt.set()
        _close(application, window)


def test_exact_saved_calibration_reopens_for_new_revision_without_mutation(
    application,
    calibration_product,
) -> None:
    experiment, reference, _computation, workspace = calibration_product
    before = _manifest_count(workspace)
    window = experiment.readout.calibration_edit_gui(reference)
    try:
        _until(
            application,
            lambda: window.worker_idle and window.editor_ready and window.raster_ready,
        )
        assert window.saved_reference == reference
        assert reference.target_ref in window._authority.text()
        assert "editing" in window._authority.text()
        assert _manifest_count(workspace) == before
        previous_boards = window._boards
        previous_bundle = window._bundle
        assert previous_boards and previous_bundle is not None
        assert window._present_bundle(previous_bundle)
        assert all(not board.has_front for board in previous_boards)
        assert all(window._tabs.indexOf(board) < 0 for board in previous_boards)
        assert window._form is not None
        window._form.widget_for("split_seed").setText("7")
        assert window._status.text() == "FINAL CALIBRATION · EDITOR CHANGED"
        window._reset_button.click()
        assert window._form.widget_for("split_seed").text() == "0"
        assert _manifest_count(workspace) == before
    finally:
        _close(application, window)


def test_committed_invalid_receipt_never_claims_failure_or_auto_closes(
    application,
    calibration_product,
) -> None:
    experiment, _reference, computation, workspace = calibration_product
    request = experiment.readout.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        replace(computation.report.request, split_seed=23),
    )
    before = _manifest_count(workspace)
    committed = threading.Event()
    release = threading.Event()
    actual_references: list[CalibrationArtifactRef] = []
    window = experiment.readout.calibration_gui(request)
    try:
        original_starter = window._run_starter

        def invalid_receipt_starter(*args):
            handle = original_starter(*args)
            original_result = handle.result

            def invalid_result(timeout=None):
                reference = original_result(timeout)
                actual_references.append(reference)
                committed.set()
                if not release.wait(15.0):
                    raise TimeoutError("test did not release invalid receipt")
                return object()

            handle.result = invalid_result
            return handle

        window._run_starter = invalid_receipt_starter
        window._calibrate_button.click()
        _until(application, committed.is_set)
        window.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert window.isVisible()
        release.set()
        _until(
            application,
            lambda: window.worker_idle
            and window._status.text() == "COMMIT RECEIPT INVALID · CLOSE AGAIN",
        )
        assert window.isVisible()
        assert window.saved_reference is None
        assert "Run reports FINAL commit" in window._summary.text()
        assert _manifest_count(workspace) == before + 1
        assert len(actual_references) == 1
        experiment.readout.load_calibration_computation(actual_references[0])
    finally:
        release.set()
        _close(application, window)


def test_typed_reference_without_commit_evidence_is_not_admitted_as_final(
    application,
    calibration_product,
) -> None:
    experiment, reference, computation, _workspace = calibration_product
    request = experiment.readout.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        computation.report.request,
    )
    window = experiment.readout.calibration_gui(request)
    try:
        generation = window._run_owner.begin_generation()
        window._run_active = True
        window._run_revision = window._editor_revision
        snapshot = RunSnapshot(
            RunId("synthetic-uncommitted-calibration"),
            RunState.SUCCEEDED,
            "terminal",
            False,
            None,
            None,
            None,
            (),
            None,
        )
        window._record_terminal_warnings = lambda: snapshot
        future = Future()
        future.set_result(reference)
        window._accept_terminal_completion(generation, future)
        assert window.worker_idle
        assert window.saved_reference is None
        assert window._status.text() == "COMMIT EVIDENCE MISSING"
        assert "not being admitted as SAVED or FINAL" in window._summary.text()
        assert "final_committed is false" in window._diagnostic.text()
    finally:
        _close(application, window)


def test_stop_before_finalize_publishes_no_calibration(
    application,
    calibration_product,
    monkeypatch,
) -> None:
    experiment, _reference, computation, workspace = calibration_product
    request = experiment.readout.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        replace(computation.report.request, split_seed=19),
    )
    before = _manifest_count(workspace)
    import zlc_neutral_atom.logic_nodes.readout.calibration.analysis as analysis_module

    original_analyze = analysis_module._analyze_calibration_resolved
    analyzed = threading.Event()
    release = threading.Event()

    def blocked_after_analysis(*args, **kwargs):
        result = original_analyze(*args, **kwargs)
        analyzed.set()
        if not release.wait(15.0):
            raise TimeoutError("test did not release calibration analysis")
        return result

    monkeypatch.setattr(
        analysis_module,
        "_analyze_calibration_resolved",
        blocked_after_analysis,
    )
    window = experiment.readout.calibration_gui(request)
    try:
        _until(application, lambda: window.worker_idle and window.editor_ready)
        window._calibrate_button.click()
        _until(application, analyzed.is_set)
        window._stop_button.click()
        release.set()
        _until(
            application,
            lambda: window.worker_idle
            and window._status.text() == "CALIBRATION CANCELLED",
        )
        assert window.saved_reference is None
        assert _manifest_count(workspace) == before
        experiment.readout.load_capture(request.source_capture_ref)
    finally:
        release.set()
        _close(application, window)
