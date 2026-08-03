"""The public Experiment API without leaking process-owned hardware authority."""

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

import Zou_lab_control.api as zlc
import Zou_lab_control.api.facade as facade_impl
import Zou_lab_control.api._readout_core as readout_core_impl
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.runtime.ports import BoundDevice
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_pulse import FrozenScanTable, RepeatRegion, load_pulse_document


ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = Path("imaging_template.json")


def _workspace(project_root: Path) -> zlc.WorkspacePaths:
    project_root = project_root.resolve()
    return zlc.WorkspacePaths(
        project_root=project_root,
        pulses_root=(ROOT / "pulses").resolve(),
        tasks_root=(ROOT / "tasks").resolve(),
        output_root=project_root / "_output",
    )


def _connect(project_root: Path):
    return zlc.connect("virtual", workspace=_workspace(project_root))


def _expect(error_type, text: str, operation):
    try:
        operation()
    except error_type as error:
        assert text in str(error), str(error)
        return error
    raise AssertionError(f"expected {error_type.__name__}: {text}")


def _assert_direct_output_root_writable(root: Path) -> None:
    output = root / "_output"
    output.mkdir(parents=True, exist_ok=True)
    probe = output / "close-probe.tmp"
    probe.write_bytes(b"ok")
    probe.unlink()


def _case_capture(root: Path) -> None:
    with _connect(root) as exp:
        request = exp.readout.capture_request(IMAGING_PULSE)
        assert request.camera_ref == exp.device_catalog["camera"].ref
        assert request.sequencer_ref == exp.device_catalog["sequencer"].ref

        descriptor = exp.inspect(request)
        assert descriptor.camera_role == "camera"
        assert descriptor.sequencer_role == "sequencer"
        assert descriptor.trigger_channel == "ch11"
        assert descriptor.expected_frames == 3
        assert descriptor.output_schema.physical_shape == (1, 3, 96, 128)

        reference = exp.run(request)
        assert isinstance(reference, CaptureArtifactRef)
        artifact = exp.readout.load_capture(reference)
        assert (
            artifact.frame_source.schema.physical_shape
            == descriptor.output_schema.physical_shape
        )
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
        convenience_capture = exp.readout.load_capture(convenience_reference)
        assert convenience_capture.pulse_evidence is not None


def _case_public_authority_and_validation(root: Path) -> None:
    exp = _connect(root)
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

        bound = exp.readout.for_binding(ReadoutBindingKey("camera"))
        _expect(ValueError, "cannot switch", lambda: bound.for_binding("sequencer"))
    finally:
        exp.close()
        exp.close()
    _assert_direct_output_root_writable(root)


class _ControlledWorkbenchHandle:
    """Structural WorkbenchHandle used to prove application close ordering."""

    def __init__(
        self,
        *,
        initially_acknowledged: bool = False,
        event_owner_thread_id: int | None = None,
    ) -> None:
        self._allow_ack = threading.Event()
        if initially_acknowledged:
            self._allow_ack.set()
        self.wait_entered = threading.Event()
        self.request_count = 0
        self.event_owner_thread_id = event_owner_thread_id

    @property
    def permanently_closed(self) -> bool:
        return self._allow_ack.is_set()

    def restore_window(self) -> None:
        raise AssertionError("close test must not restore its Workbench")

    def request_owner_close(self) -> None:
        self.request_count += 1

    def wait_owner_closed(self, timeout: float) -> bool:
        self.wait_entered.set()
        if threading.get_ident() == self.event_owner_thread_id:
            self._allow_ack.set()
        return self._allow_ack.wait(min(timeout, 0.4))

    def acknowledge_close(self) -> None:
        self._allow_ack.set()


def _case_concurrent_close_owner(root: Path) -> None:
    exp = _connect(root)
    services = exp._services
    handle = _ControlledWorkbenchHandle()
    assert exp._open_workbench_handle(None, lambda: handle) is handle

    failures: list[BaseException] = []

    def invoke_close() -> None:
        try:
            exp.close()
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=invoke_close, daemon=False)
    first.start()
    assert handle.wait_entered.wait(2.0)

    second = threading.Thread(target=invoke_close, daemon=False)
    second.start()
    time.sleep(0.05)
    assert second.is_alive()
    with services.operation_lock:
        assert services.state == "CLOSING"

    handle.acknowledge_close()
    first.join(2.0)
    second.join(2.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert handle.request_count == 2
    assert services.state == "CLOSED"
    _assert_direct_output_root_writable(root)


def _case_concurrent_gui_owner_keeps_pumping(root: Path) -> None:
    exp = _connect(root)
    services = exp._services
    handle = _ControlledWorkbenchHandle(
        event_owner_thread_id=threading.get_ident(),
    )
    with services.operation_lock:
        services.gui_handles["controlled"] = handle

    foreign_failures: list[BaseException] = []

    def foreign_close() -> None:
        try:
            exp.close()
        except BaseException as error:
            foreign_failures.append(error)

    foreign = threading.Thread(target=foreign_close, daemon=False)
    foreign.start()
    assert handle.wait_entered.wait(2.0)

    # The main thread models the Qt owner.  It is a concurrent caller, not the
    # teardown owner, but its Workbench wait port must keep processing the
    # close acknowledgement needed by the foreign owner.
    exp.close()
    foreign.join(2.0)
    assert not foreign.is_alive()
    assert foreign_failures == []
    assert services.state == "CLOSED"
    _assert_direct_output_root_writable(root)


def _case_gui_close_retry_preserves_data(root: Path) -> None:
    exp = _connect(root)
    services = exp._services
    handle = _ControlledWorkbenchHandle()
    with services.operation_lock:
        services.gui_handles["controlled"] = handle

    # Avoid spending the production acknowledgement deadline in this
    # deterministic contract case while still exercising the real close path.
    handle.wait_owner_closed = lambda _timeout: False  # type: ignore[method-assign]
    _expect(
        facade_impl._ResourceCleanupError,
        "Workbench close failed",
        exp.close,
    )
    assert services.state == "CLOSING"
    assert services.closing_gui_handles == (handle,)
    handle.wait_owner_closed = (  # type: ignore[method-assign]
        lambda timeout: handle._allow_ack.wait(timeout)
    )
    handle.acknowledge_close()
    exp.close()
    assert services.state == "CLOSED"
    _assert_direct_output_root_writable(root)


def _case_runtime_close_retry_preserves_handles(root: Path) -> None:
    exp = _connect(root)
    services = exp._services
    handle = _ControlledWorkbenchHandle(initially_acknowledged=True)
    with services.operation_lock:
        services.gui_handles["controlled"] = handle

    runtime_type = type(services.runtime)
    original_shutdown = runtime_type.shutdown
    shutdown_calls = 0

    def fail_first_shutdown(runtime, *, timeout: float):
        nonlocal shutdown_calls
        shutdown_calls += 1
        if shutdown_calls == 1:
            return False
        return original_shutdown(runtime, timeout=timeout)

    runtime_type.shutdown = fail_first_shutdown
    try:
        _expect(RuntimeError, "did not complete", exp.close)
        assert services.state == "CLOSING"
        assert services.closing_gui_handles == (handle,)
        assert handle.request_count == 1
        exp.close()
    finally:
        runtime_type.shutdown = original_shutdown

    assert shutdown_calls == 2
    assert handle.request_count == 3
    assert services.state == "CLOSED"
    _assert_direct_output_root_writable(root)


def _case_close_race(root: Path, surface: str) -> None:
    exp = _connect(root)
    services = exp._services
    passed_initial_lookup = threading.Event()
    backend_calls: list[str] = []
    failures: list[BaseException] = []
    guard_owner = readout_core_impl if surface == "capture" else facade_impl
    guard_name = "service_guard" if surface == "capture" else "_service_guard"
    original_guard = getattr(guard_owner, guard_name)

    @contextmanager
    def observed_guard(guarded_services):
        passed_initial_lookup.set()
        with original_guard(guarded_services) as value:
            yield value

    setattr(guard_owner, guard_name, observed_guard)
    if surface == "capture":
        readout_core_impl.load_capture_artifact = (  # type: ignore[assignment]
            lambda *_args, **_kwargs: backend_calls.append("capture-load")
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
    def reject_public_root(*_args, **_kwargs):
        raise RuntimeError("public root construction failed")

    facade_impl.Experiment = reject_public_root
    error = _expect(
        RuntimeError,
        "public root construction failed",
        lambda: _connect(root),
    )
    assert error.__cause__ is None
    _assert_direct_output_root_writable(root)


def _run_case(case: str, root_text: str) -> None:
    root = Path(root_text)
    cases = {
        "capture": lambda: _case_capture(root),
        "public-authority": lambda: _case_public_authority_and_validation(root),
        "concurrent-close-owner": lambda: _case_concurrent_close_owner(root),
        "concurrent-gui-owner": lambda: _case_concurrent_gui_owner_keeps_pumping(root),
        "gui-close-retry": lambda: _case_gui_close_retry_preserves_data(root),
        "runtime-close-retry": lambda: _case_runtime_close_retry_preserves_handles(root),
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
        "capture",
        "public-authority",
        "concurrent-close-owner",
        "concurrent-gui-owner",
        "gui-close-retry",
        "runtime-close-retry",
        "close-race-capture",
        "close-race-pulse",
        "failed-public-root",
    ),
)
def test_public_api_in_process_lifetime_installation(
    case: str,
    tmp_path: Path,
) -> None:
    completed = _run_isolated(case, tmp_path / case)
    assert completed.returncode == 0, (
        f"isolated public API case {case!r} failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_connect_rejects_implicit_or_non_string_target(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        zlc.connect("virtual")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="config must be"):
        zlc.connect(  # type: ignore[arg-type]
            {},
            workspace=_workspace(tmp_path / "workspace"),
        )
