"""The short notebook facade without leaking process-owned hardware authority."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import Zou_lab_control.notebook as zlc
import Zou_lab_control.notebook.facade as facade_impl
from zlc_data import FitNumericPolicy, SPATIAL_X, SPATIAL_Y, encode_fit_result_batch
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureFitResultArtifactRef,
    CaptureFitResultRepository,
    CaptureRepository,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.readout.calibration_repository import CalibrationRepository
from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
from zlc_neutral_atom.runtime.ports import BoundDevice
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_storage import RepositoryRootBusy


ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


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
    CaptureFitResultRepository(root / "fits").close()


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
        assert descriptor.estimated_peak_bytes < request.pipeline_memory_limit_bytes

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
                sample_budget_per_batch=512,
                max_packed_observations=4_096,
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
        assert isinstance(fit_ref, CaptureFitResultArtifactRef)
        admitted = exp.load_fit(fit_ref)
        assert admitted.reference == fit_ref
        assert admitted.source_capture_ref == convenience_reference
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
            lambda: CaptureFitResultRepository(root / "fits"),
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
        "public-authority": lambda: _case_public_authority_and_validation(root),
        "close-retry": lambda: _case_close_retry(root),
        "close-race-capture": lambda: _case_close_race(root, "capture"),
        "close-race-pulse": lambda: _case_close_race(root, "pulse"),
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
        "public-authority",
        "close-retry",
        "close-race-capture",
        "close-race-pulse",
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
