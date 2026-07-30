"""Formal calibration creation, editing, direct save, and report behavior."""

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
from zlc_frontend.form import project_authoring_form
import pytest

import Zou_lab_control.api as zlc
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    build_calibration_analysis_request_from_authoring,
    calibration_analysis_authoring_schema,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import OccupancyArtifactRef
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture(scope="module")
def calibration_product(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("w6-calibration-workspace")
    with zlc.connect(
        "virtual",
        workspace=zlc.WorkspacePaths(
            ROOT,
            ROOT / "pulses",
            ROOT / "tasks",
            workspace.resolve() / "_output",
        ),
    ) as experiment:
        reference = experiment.nodes.calibration.sitemap(frames=12)
        computation = experiment.nodes.calibration.load_calibration_computation(reference)
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


def _calibration_record_count(workspace: Path) -> int:
    root = workspace / "_output" / "calibrations"
    return 0 if not root.exists() else len(tuple(root.glob("*/calibration.json")))


def test_calibration_owner_presenter_and_public_imports_remain_headless() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import zlc_neutral_atom.logic_nodes.readout.calibration.calibration\n"
                "import zlc_neutral_atom.logic_nodes.readout.calibration.declaration\n"
                "from zlc_neutral_atom.logic_node_package import "
                "discover_logic_node_packages\n"
                "packages = discover_logic_node_packages()\n"
                "calibration = next(value for value in packages "
                "if value.api_name == 'calibration')\n"
                "assert calibration.ui_contributions\n"
                "assert not any(value.module in sys.modules "
                "for value in calibration.ui_contributions)\n"
                "import zlc_frontend.form\n"
                "for prefix in ('PyQt5', 'matplotlib', 'scipy'):\n"
                "    assert not any(\n"
                "        name == prefix or name.startswith(prefix + '.')\n"
                "        for name in sys.modules\n"
                "    ), prefix\n"
                "import Zou_lab_control.api\n"
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


def test_formal_calibration_save_wins_close_and_render_failure_keeps_reference(
    application,
    calibration_product,
) -> None:
    experiment, _prior, computation, workspace = calibration_product
    original = computation.report.request
    request = experiment.nodes.calibration.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        replace(original, split_seed=original.split_seed + 1),
    )
    before = _calibration_record_count(workspace)
    owner_thread = threading.get_ident()
    saved = threading.Event()
    release_receipt = threading.Event()
    run_threads: list[int] = []
    window = experiment.nodes.calibration.calibration_gui(request)
    try:
        _until(application, lambda: window.worker_idle and window.editor_ready)
        assert window.saved_reference is None
        assert window._form is not None
        form = window._form
        form.widget_for("train_fraction").setValue(1.0)
        window._calibrate_button.click()
        assert window.worker_idle
        assert window._status.text() == "CALIBRATION REQUEST INVALID"
        assert _calibration_record_count(workspace) == before
        form.widget_for("train_fraction").setValue(0.8)

        original_starter = window._run_starter

        def delayed_receipt_starter(*args):
            run_threads.append(threading.get_ident())
            handle = original_starter(*args)
            original_result = handle.result
            original_snapshot = handle.snapshot

            def delayed_result(timeout=None):
                reference = original_result(timeout)
                saved.set()
                if not release_receipt.wait(15.0):
                    raise TimeoutError("test did not release saved reference")
                return reference

            def warning_snapshot():
                snapshot = original_snapshot()
                return replace(
                    snapshot,
                    cleanup_errors=(
                        *snapshot.cleanup_errors,
                        "deterministic post-FINAL cleanup warning",
                    ),
                )

            handle.result = delayed_result
            handle.snapshot = warning_snapshot
            return handle

        window._run_starter = delayed_receipt_starter
        window._calibrate_button.click()
        _until(application, saved.is_set)
        assert run_threads and run_threads[0] != owner_thread

        # A disabled form can still be changed programmatically.  The formal
        # Run keeps its captured revision while the UI marks the receipt stale.
        form.widget_for("split_seed").setValue(original.split_seed + 2)
        window.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert window.isVisible()
        assert window.saved_reference is None
        assert window._status.text() == "CANCELLING CALIBRATION · CLOSE DEFERRED"

        def fail_report(_reference, **_options):
            raise RuntimeError("deterministic post-FINAL report failure")

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
        assert "post-FINAL cleanup warning" in window._diagnostic.text()
        assert "post-FINAL report failure" in window._diagnostic.text()
        assert _calibration_record_count(workspace) == before + 1
        loaded = experiment.nodes.calibration.load_calibration_computation(saved)
        assert loaded.report.request.train_fraction == 0.8
        assert loaded.report.request.split_seed == original.split_seed + 1
        assert loaded.artifact.source_binding.source_capture_ref == (
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
    before = _calibration_record_count(workspace)
    window = experiment.nodes.calibration.calibration_edit_gui(reference)
    try:
        _until(
            application,
            lambda: window.worker_idle and window.editor_ready and window.raster_ready,
        )
        assert window.saved_reference == reference
        assert reference.target_ref in window._authority.text()
        assert "editing" in window._authority.text()
        assert _calibration_record_count(workspace) == before
        previous_boards = window._boards
        previous_bundle = window._bundle
        assert previous_boards and previous_bundle is not None
        assert window._present_bundle(previous_bundle)
        assert all(not board.has_front for board in previous_boards)
        assert all(window._tabs.indexOf(board) < 0 for board in previous_boards)
        assert window._form is not None
        window._form.widget_for("split_seed").setValue(7)
        assert window._status.text() == "FINAL CALIBRATION · EDITOR CHANGED"
        window._reset_button.click()
        assert window._form.widget_for("split_seed").value() == 0
        assert _calibration_record_count(workspace) == before
    finally:
        _close(application, window)


def test_non_calibration_success_result_is_a_protocol_failure(
    application,
    calibration_product,
) -> None:
    experiment, _reference, computation, workspace = calibration_product
    request = experiment.nodes.calibration.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        replace(computation.report.request, split_seed=23),
    )
    before = _calibration_record_count(workspace)
    saved = threading.Event()
    release = threading.Event()
    actual_references: list[CalibrationArtifactRef] = []
    window = experiment.nodes.calibration.calibration_gui(request)
    try:
        original_starter = window._run_starter

        def invalid_receipt_starter(*args):
            handle = original_starter(*args)
            original_result = handle.result

            def invalid_result(timeout=None):
                reference = original_result(timeout)
                actual_references.append(reference)
                saved.set()
                if not release.wait(15.0):
                    raise TimeoutError("test did not release invalid result")
                return object()

            handle.result = invalid_result
            return handle

        window._run_starter = invalid_receipt_starter
        window._calibrate_button.click()
        _until(application, saved.is_set)
        release.set()
        _until(
            application,
            lambda: window.worker_idle
            and window._status.text() == "CALIBRATION FAILED",
        )
        assert window.isVisible()
        assert window.saved_reference is None
        assert "returned no CalibrationArtifactRef" in window._summary.text()
        assert "invalid reference" in window._diagnostic.text()
        assert _calibration_record_count(workspace) == before + 1
        assert len(actual_references) == 1
        experiment.nodes.calibration.load_calibration_computation(actual_references[0])
    finally:
        release.set()
        _close(application, window)


def test_typed_success_reference_is_final_without_second_commit_evidence(
    application,
    calibration_product,
) -> None:
    experiment, reference, computation, _workspace = calibration_product
    request = experiment.nodes.calibration.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        computation.report.request,
    )
    window = experiment.nodes.calibration.calibration_gui(request)
    try:
        generation = window._run_owner.begin_generation()
        window._run_active = True
        window._run_revision = window._editor_revision
        snapshot = RunSnapshot(
            run_id=RunId("synthetic-final-calibration"),
            state=RunState.SUCCEEDED,
            phase="terminal",
            primary_error=None,
            cleanup_errors=(),
        )
        window._record_terminal_warnings = lambda: snapshot
        future = Future()
        future.set_result(reference)
        window._accept_terminal_completion(generation, future)
        assert window.saved_reference == reference
        assert window._status.text() == "CALIBRATION SAVED"
        assert reference.record_path in window._summary.text()
        assert window._diagnostic.text() == ""
    finally:
        _close(application, window)


def test_stop_before_finalize_publishes_no_calibration(
    application,
    calibration_product,
    monkeypatch,
) -> None:
    experiment, _reference, computation, workspace = calibration_product
    current_before = experiment.nodes.calibration.current_calibration_ref
    assert current_before is not None
    request = experiment.nodes.calibration.calibration_request(
        computation.artifact.source_binding.source_capture_ref,
        replace(computation.report.request, split_seed=19),
    )
    before = _calibration_record_count(workspace)
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
    window = experiment.nodes.calibration.calibration_gui(request)
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
        assert _calibration_record_count(workspace) == before
        assert experiment.nodes.calibration.current_calibration_ref == current_before
        experiment.readout.load_capture(request.source_capture_ref)
    finally:
        release.set()
        _close(application, window)


def test_experiment_close_retires_lazy_calibration_and_occupancy_windows(
    application,
    tmp_path,
) -> None:
    """Leaf GUI descriptors borrow exactly the owning Experiment lifetime."""

    experiment = zlc.connect(
        "virtual",
        workspace=zlc.WorkspacePaths(
            ROOT,
            ROOT / "pulses",
            ROOT / "tasks",
            tmp_path.resolve() / "_output",
        ),
    )
    closed = False
    try:
        calibration = CalibrationArtifactRef(
            "lazy-ui-lifecycle/calibration.json",
        )
        occupancy = OccupancyArtifactRef(
            "lazy-ui-lifecycle/occupancy.json",
        )

        calibration_window = (
            experiment.nodes.calibration.calibration_report_gui(calibration)
        )
        occupancy_window = experiment.nodes.occupancy.occupancy_cell_gui(occupancy)
        _until(
            application,
            lambda: calibration_window.isVisible()
            and occupancy_window.isVisible(),
            timeout=15.0,
        )

        experiment.close()
        closed = True
        assert calibration_window.permanently_closed
        assert occupancy_window.permanently_closed
        assert not calibration_window.isVisible()
        assert not occupancy_window.isVisible()
    finally:
        if not closed:
            experiment.close()
