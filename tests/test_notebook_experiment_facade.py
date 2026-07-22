"""The short notebook facade without leaking process-owned hardware authority."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import Zou_lab_control.notebook as zlc
import Zou_lab_control.notebook.facade as facade_impl
from zlc_data import FitNumericPolicy, SPATIAL_X, SPATIAL_Y, encode_fit_result_batch
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureRepository,
    FitResultArtifactRef,
    FitResultRepository,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.readout.calibration_repository import CalibrationRepository
from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
from zlc_neutral_atom.runtime.ports import BoundDevice
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_pulse import FrozenScanTable, RepeatRegion, load_pulse_document
from zlc_storage import RepositoryRootBusy


ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"
MOT_SCAN_PULSE = ROOT / "pulses" / "mot_field_template.json"


def _expect(error_type, text: str, operation):
    try:
        operation()
    except error_type as error:
        assert text in str(error), str(error)
        return error
    raise AssertionError(f"expected {error_type.__name__}: {text}")


def _assert_repository_roots_released(root: Path) -> None:
    CaptureRepository(root / "captures").close()
    CalibrationRepository(root / "calibrations").close()
    FitResultRepository(root / "fits").close()


def _case_capture_and_fit(root: Path) -> None:
    with zlc.connect("virtual", repository=root) as exp:
        request = exp.readout.capture_request(IMAGING_PULSE)
        assert request.camera_ref == exp.device_catalog["camera"].ref
        assert request.sequencer_ref == exp.device_catalog["sequencer"].ref

        descriptor = exp.inspect(request)
        assert descriptor.camera_role == "camera"
        assert descriptor.sequencer_role == "sequencer"
        assert descriptor.trigger_channel == "ch11"
        assert descriptor.expected_frames == 3
        assert descriptor.output_shape == (1, 3, 96, 128)
        assert descriptor.resource_claims == ("device/sequencer", "device/camera")

        reference = exp.run(request)
        assert isinstance(reference, CaptureArtifactRef)
        artifact = exp.readout.load_capture(reference)
        assert artifact.frame_source.schema.physical_shape == descriptor.output_shape
        evidence = artifact.pulse_evidence
        assert evidence is not None
        assert evidence.compiled_artifact.fingerprint == descriptor.compiled_pulse_digest
        assert evidence.expected_trigger_count == descriptor.expected_frames
        assert tuple(artifact.frame_source.iter_cell_schedule()) == tuple(
            evidence.join_contract.iter_cell_schedule(
                evidence.trigger_schedule,
                artifact.frame_source.schema,
            )
        )
        assert tuple(
            setting.event_index
            for setting in artifact.camera_provenance.descriptor.event_settings
        ) == (0, 1, 2)

        convenience_reference = exp.readout.capture(IMAGING_PULSE)
        assert exp.readout.load_capture(convenience_reference).pulse_evidence is not None
        execution = exp.fit(
            convenience_reference,
            model="radial_gaussian_center",
            numeric_policy=FitNumericPolicy(
                max_evaluations=500,
            ),
        )
        assert tuple(axis.role for axis in execution.result.fit_axis_specs) == (
            SPATIAL_X,
            SPATIAL_Y,
        )
        assert execution.result.spec.batch_axis_ids == tuple(
            axis.axis_id for axis in execution.result.batch_axis_specs
        )
        assert len(execution.result.batch_axis_specs) == 2

        fit_ref = execution.save()
        assert isinstance(fit_ref, FitResultArtifactRef)
        admitted = exp.load_fit(fit_ref)
        assert admitted.reference == fit_ref
        assert admitted.source_artifact_ref == convenience_reference
        assert encode_fit_result_batch(admitted.result) == encode_fit_result_batch(
            execution.result
        )


def _case_scan_and_fit(root: Path) -> None:
    document = load_pulse_document(MOT_SCAN_PULSE)
    columns = tuple(item.parameter_id for item in document.scan_parameters)
    document = replace(
        document,
        scan_table=FrozenScanTable(
            columns,
            ((0, 0, 0), (1, 0, 1), (0, 1, 1)),
        ),
        repeat=RepeatRegion(
            document.periods[0].period_id,
            document.periods[-1].period_id,
            2,
        ),
    )
    with zlc.connect("virtual", repository=root) as exp:
        scan_ref = exp.scan(
            exp.readout.scan_request(document, timeout_seconds=15.0)
        )
        source = exp.readout.materialize_scan(scan_ref)
        execution = exp.fit(
            scan_ref,
            model="radial_gaussian_center",
            numeric_policy=FitNumericPolicy(
                max_evaluations=500,
            ),
        )
        assert execution.source_artifact_ref == scan_ref
        assert execution.result.source_ref == source.snapshot.ref
        fit_ref = execution.save()
        admitted = exp.load_fit(fit_ref)
        assert admitted.source_artifact_ref == scan_ref
        assert encode_fit_result_batch(admitted.result) == encode_fit_result_batch(
            execution.result
        )


def _case_public_authority_and_validation(root: Path) -> None:
    exp = zlc.connect("virtual", repository=root)
    try:
        forbidden = (BoundDevice, RunPlan)
        public_values = (
            exp.name,
            exp.device_catalog,
            exp.readout,
            exp.pulse,
            exp.pulse.target,
        )
        assert not any(isinstance(value, forbidden) for value in public_values)
        for attribute in ("devices", "camera", "sequencer"):
            assert not hasattr(exp, attribute)
        assert not hasattr(exp.readout, "camera")
        assert not hasattr(exp.pulse, "sequencer")

        _expect(
            ValueError,
            "not 'camera'",
            lambda: exp.readout.capture_request(
                IMAGING_PULSE,
                camera_role="sequencer",
            ),
        )
        wiring_request = exp.readout.capture_request(
            IMAGING_PULSE,
            trigger_channel="mot_trigger",
        )
        _expect(ValueError, "not wired", lambda: exp.inspect(wiring_request))

        request = exp.readout.capture_request(IMAGING_PULSE)
        stale = DeviceRef(
            request.camera_ref.installation_id,
            request.camera_ref.runtime_instance_id + "-stale",
            request.camera_ref.role,
        )
        _expect(
            RuntimeError,
            "another runtime instance",
            lambda: exp.inspect(replace(request, camera_ref=stale)),
        )

        _expect(
            RepositoryRootBusy,
            "live owner",
            lambda: CaptureRepository(root / "captures"),
        )
        _expect(
            RepositoryRootBusy,
            "live owner",
            lambda: FitResultRepository(root / "fits"),
        )

        bound = exp.readout.for_binding(ReadoutBindingKey("camera"))
        assert not hasattr(bound, "current_calibration_ref")
        _expect(ValueError, "cannot switch", lambda: bound.for_binding("sequencer"))
    finally:
        exp.close()
        exp.close()
    _assert_repository_roots_released(root)


def _case_close_retry(root: Path) -> None:
    exp = zlc.connect("virtual", repository=root)
    token = exp._authority_token
    services = facade_impl._AUTHORITIES[token]
    borrow = services.capture_repository._root_lease.borrow()
    try:
        _expect(facade_impl._ResourceCleanupError, "close failed", exp.close)
        assert facade_impl._AUTHORITIES[token] is services
        assert services.state == "CLOSING"
        _expect(
            RuntimeError,
            "closing or closed",
            lambda: exp.readout.load_capture(object()),
        )
    finally:
        borrow.close()
    exp.close()
    assert token not in facade_impl._AUTHORITIES
    _assert_repository_roots_released(root)


def _case_close_race(root: Path, surface: str) -> None:
    exp = zlc.connect("virtual", repository=root)
    services = facade_impl._AUTHORITIES[exp._authority_token]
    passed_initial_lookup = threading.Event()
    backend_calls: list[str] = []
    failures: list[BaseException] = []
    original_guard = facade_impl._service_guard

    @contextmanager
    def observed_guard(token):
        passed_initial_lookup.set()
        with original_guard(token) as value:
            yield value

    facade_impl._service_guard = observed_guard
    if surface == "capture":
        type(services.capture_repository).load = (
            lambda _self, _reference: backend_calls.append("capture-load")
        )
        operation = lambda: exp.readout.load_capture(object())
    else:
        type(services.runtime).pulse_port = (
            lambda _self, _reference: backend_calls.append("pulse-port")
        )
        operation = lambda: exp.pulse.target

    def invoke() -> None:
        try:
            operation()
        except BaseException as error:
            failures.append(error)

    with services.operation_lock:
        thread = threading.Thread(target=invoke, daemon=False)
        thread.start()
        assert passed_initial_lookup.wait(1.0)
        exp.close()
    thread.join(2.0)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert str(failures[0]) == "Experiment is closing or closed"
    assert backend_calls == []


def _case_fit_close_drain(root: Path) -> None:
    exp = zlc.connect("virtual", repository=root)
    token = exp._authority_token
    services = facade_impl._AUTHORITIES[token]
    entered = threading.Barrier(3)
    close_started = threading.Barrier(3)
    releases = (threading.Event(), threading.Event())
    fit_failures: list[BaseException] = []
    close_failures: list[BaseException] = []

    def fit_worker(index: int) -> None:
        try:
            with facade_impl._fit_service_guard(token):
                entered.wait(timeout=2.0)
                assert releases[index].wait(2.0)
        except BaseException as error:
            fit_failures.append(error)

    def close_worker() -> None:
        try:
            close_started.wait(timeout=2.0)
            exp.close()
        except BaseException as error:
            close_failures.append(error)

    fit_threads = tuple(
        threading.Thread(target=fit_worker, args=(index,), daemon=False)
        for index in range(2)
    )
    for thread in fit_threads:
        thread.start()
    entered.wait(timeout=2.0)
    with services.operation_lock:
        assert services.active_fit_operations == 2

    close_threads = tuple(
        threading.Thread(target=close_worker, daemon=False) for _ in range(2)
    )
    for thread in close_threads:
        thread.start()
    close_started.wait(timeout=2.0)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with services.operation_lock:
            if services.state == "CLOSING":
                break
        time.sleep(0.005)
    else:
        raise AssertionError("concurrent close did not enter CLOSING")

    def attempt_late_fit() -> None:
        with facade_impl._fit_service_guard(token):
            raise AssertionError("Fit entered after Experiment began closing")

    _expect(RuntimeError, "closing or closed", attempt_late_fit)
    assert all(thread.is_alive() for thread in close_threads)

    releases[0].set()
    fit_threads[0].join(2.0)
    assert not fit_threads[0].is_alive()
    assert all(thread.is_alive() for thread in close_threads)

    releases[1].set()
    for thread in (*fit_threads[1:], *close_threads):
        thread.join(2.0)
        assert not thread.is_alive()
    assert close_failures == []
    assert len(fit_failures) == 2
    assert all(isinstance(error, facade_impl.FitCancelled) for error in fit_failures)
    assert token not in facade_impl._AUTHORITIES
    _assert_repository_roots_released(root)


def _case_fit_reentrant_close(root: Path) -> None:
    exp = zlc.connect("virtual", repository=root)
    token = exp._authority_token
    services = facade_impl._AUTHORITIES[token]
    with facade_impl._fit_service_guard(token):
        _expect(
            RuntimeError,
            "cannot close reentrantly",
            exp.close,
        )
        with services.operation_lock:
            assert services.state == "OPEN"
            assert services.active_fit_operations == 1
    with services.operation_lock:
        assert services.state == "OPEN"
        assert services.active_fit_operations == 0
        assert services.fit_operations_drained.is_set()
    exp.close()
    assert token not in facade_impl._AUTHORITIES
    _assert_repository_roots_released(root)


def _case_failed_public_root(root: Path) -> None:
    authority_count = len(facade_impl._AUTHORITIES)

    def reject_public_root(*_args, **_kwargs):
        raise RuntimeError("public root construction failed")

    facade_impl.Experiment = reject_public_root
    error = _expect(
        RuntimeError,
        "public root construction failed",
        lambda: zlc.connect("virtual", repository=root),
    )
    assert error.__cause__ is None
    assert len(facade_impl._AUTHORITIES) == authority_count
    _assert_repository_roots_released(root)


def _run_case(case: str, root_text: str) -> None:
    root = Path(root_text)
    cases = {
        "capture-and-fit": lambda: _case_capture_and_fit(root),
        "scan-and-fit": lambda: _case_scan_and_fit(root),
        "public-authority": lambda: _case_public_authority_and_validation(root),
        "close-retry": lambda: _case_close_retry(root),
        "close-race-capture": lambda: _case_close_race(root, "capture"),
        "close-race-pulse": lambda: _case_close_race(root, "pulse"),
        "fit-close-drain": lambda: _case_fit_close_drain(root),
        "fit-reentrant-close": lambda: _case_fit_reentrant_close(root),
        "failed-public-root": lambda: _case_failed_public_root(root),
    }
    cases[case]()


def _run_isolated(case: str, root: Path) -> subprocess.CompletedProcess[str]:
    command = (
        "import runpy,sys; "
        "runpy.run_path(sys.argv[1])['_run_case'](sys.argv[2], sys.argv[3])"
    )
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(
        (sys.executable, "-c", command, str(Path(__file__).resolve()), case, str(root)),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )


@pytest.mark.parametrize(
    "case",
    (
        "capture-and-fit",
        "scan-and-fit",
        "public-authority",
        "close-retry",
        "close-race-capture",
        "close-race-pulse",
        "fit-close-drain",
        "fit-reentrant-close",
        "failed-public-root",
    ),
)
def test_notebook_facade_in_process_lifetime_installation(
    case: str,
    tmp_path: Path,
) -> None:
    completed = _run_isolated(case, tmp_path / case)
    assert completed.returncode == 0, (
        f"isolated notebook case {case!r} failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_connect_rejects_implicit_or_non_string_target(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        zlc.connect("virtual")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="explicit target backend"):
        zlc.connect({}, repository=tmp_path / "workspace")  # type: ignore[arg-type]
