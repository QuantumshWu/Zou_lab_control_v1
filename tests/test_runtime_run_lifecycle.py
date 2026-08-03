"""Current flat Run lifecycle and post-safety finalization contracts."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from Zou_lab_control.api._application_services import (
    ExperimentServices,
    WorkspacePaths,
    application_start_run,
)
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_neutral_atom.runtime._failure import detach_failure, safe_error_summary
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    DeviceBroker,
    SessionClosedAck,
    SafetyInterrupt,
    SafetyOperation,
    cleanup_device_session,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
)
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    PostSafetyContext,
    RunCancelled,
    RunContext,
    RunController,
    RunFailed,
    RunPlan,
    RunSnapshot,
    RunStartRejected,
    RunState,
)


def camera_key(name: str = "a") -> ResourceKey:
    return ResourceKey.parse(f"device/camera/{name}")


def identity(resource: ResourceKey) -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        stable_device_identity=str(resource),
        evidence_kind=DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
    )


@dataclass
class DeviceFixture:
    key: ResourceKey
    broker: DeviceBroker
    device: BoundDevice


def device_fixture(
    name: str = "a",
    *,
    identity_probe=None,
    execute_command=lambda command: command,
    interrupt=None,
) -> DeviceFixture:
    resource = camera_key(name)
    physical = identity(resource)
    probe = identity_probe or (lambda: physical)
    broker = DeviceBroker()
    verified = broker.verify_identity(probe)
    binding_instance_id = verified.binding_instance_id

    def close_session(command):
        return SessionClosedAck(
            session_id=command.session_id,
            binding_instance_id=binding_instance_id,
            source_stopped=True,
            no_more_work=True,
            joined=True,
        )

    device = broker.bind(
        key=resource,
        identity=verified,
        execute_command=execute_command,
        close_session=close_session,
        interrupt_operations=(
            {}
            if interrupt is None
            else {SafetyOperation.ABORT: interrupt}
        ),
    )
    return DeviceFixture(resource, broker, device)


def safe_cleanup(context: RunContext, resource: ResourceKey) -> CleanupReport:
    return cleanup_device_session(
        context.cleanup_device(resource),
        "test-session",
        1.0,
    )


def plan(
    item: DeviceFixture,
    *,
    preflight=lambda _context: "prepared",
    execute=lambda _context, prepared: prepared,
    cleanup=None,
    finalize=lambda _context, executed: executed,
    interrupt: bool = False,
    timeout_seconds: float | None = None,
) -> RunPlan:
    cleanup = cleanup or (
        lambda context, _prepared, _primary: safe_cleanup(context, item.key)
    )
    return RunPlan(
        name="test run",
        resource_claims=(ResourceClaim(item.key),),
        bound_devices=(item.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=(
            (SafetyInterrupt(item.key, SafetyOperation.ABORT),)
            if interrupt
            else ()
        ),
        timeout_seconds=timeout_seconds,
    )


def controller(_tmp_path) -> tuple[RunController, ResourceArbiter]:
    arbiter = ResourceArbiter()
    return RunController(arbiter), arbiter


def close_runtime(
    runtime: RunController,
    arbiter: ResourceArbiter,
    *items: DeviceFixture,
) -> None:
    assert runtime.shutdown(2.0)
    for item in items:
        item.broker.shutdown()
    arbiter.shutdown()


def wait_snapshot(handle, predicate, timeout: float = 2.0) -> RunSnapshot:
    deadline = time.monotonic() + timeout
    while True:
        snapshot = handle.snapshot()
        if predicate(snapshot):
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Run did not reach requested state: {snapshot}")
        time.sleep(0.002)


class _AdmissionRuntime:
    """Real RunController with one deterministic post-release test hook."""

    def __init__(self) -> None:
        self.resources = ResourceArbiter()
        self.controller = RunController(self.resources)
        self.after_release = None

    def start(self, plan: RunPlan):
        return self.controller.start(plan)

    def wait_until_released(
        self,
        run_ids: tuple[str, ...],
        *,
        timeout: float | None = None,
    ) -> bool:
        released = self.resources.wait_until_released(run_ids, timeout=timeout)
        callback = self.after_release
        if released and callback is not None:
            self.after_release = None
            callback()
        return released

    def shutdown(self) -> None:
        assert self.controller.shutdown(2.0)
        self.resources.shutdown()


@contextmanager
def _admission_environment(tmp_path):
    runtime = _AdmissionRuntime()
    plane = SignalDataPlane()
    drained = threading.Event()
    drained.set()
    services = ExperimentServices(
        workspace_paths=WorkspacePaths.for_workspace(
            tmp_path.resolve(),
        ),
        installation=object(),
        runtime=runtime,
        captures_root=tmp_path.resolve() / "_output" / "captures",
        calibrations_root=tmp_path.resolve() / "_output" / "calibrations",
        catalog=object(),
        installation_config=object(),
        pulse_application=object(),
        signal_plane=plane,
        operation_lock=threading.RLock(),
        admission_lock=threading.RLock(),
        operations_drained=drained,
        operation_thread_counts={},
        active_runs={},
        gui_handles={},
    )
    try:
        yield services, runtime
    finally:
        plane.close()
        runtime.shutdown()


def _admission_plan(
    name: str,
    owner: object,
    claims: tuple[ResourceClaim, ...],
    *,
    preemptible: bool,
) -> RunPlan:
    def execute(context, prepared):
        context.cancellation.wait_requested()
        return prepared

    return RunPlan(
        name=name,
        resource_claims=claims,
        bound_devices=(),
        preflight=lambda _context: name,
        execute=execute,
        cleanup=lambda _context, _prepared, _primary: CleanupReport.complete(),
        finalize=lambda _context, executed: executed,
    ).with_lifecycle(owner, preemptible=preemptible)


def _start_admission_run(
    services: ExperimentServices,
    name: str,
    claims: tuple[ResourceClaim, ...],
    *,
    preemptible: bool,
):
    handle = application_start_run(
        services,
        _admission_plan(name, object(), claims, preemptible=preemptible),
    )
    wait_snapshot(handle, lambda item: item.state is RunState.RUNNING)
    return handle


@pytest.mark.parametrize("second_preemptible", (False, True))
def test_application_retires_every_blocker_or_none(
    tmp_path,
    second_preemptible: bool,
) -> None:
    first_claim = ResourceClaim(ResourceKey.parse("application/camera"))
    second_claim = ResourceClaim(ResourceKey.parse("application/sequencer"))
    with _admission_environment(tmp_path) as (services, _runtime):
        first = _start_admission_run(
            services,
            f"first-plan-{second_preemptible}",
            (first_claim,),
            preemptible=True,
        )
        second = _start_admission_run(
            services,
            f"second-plan-{second_preemptible}",
            (second_claim,),
            preemptible=second_preemptible,
        )
        candidate_plan = _admission_plan(
            f"candidate-plan-{second_preemptible}",
            object(),
            (first_claim, second_claim),
            preemptible=False,
        )

        if not second_preemptible:
            with pytest.raises(RunStartRejected) as rejected:
                application_start_run(services, candidate_plan)
            assert {item.conflicting_run_id for item in rejected.value.blockers} == {
                first.run_id.value,
                second.run_id.value,
            }
            assert first.snapshot().state is RunState.RUNNING
            assert second.snapshot().state is RunState.RUNNING
        else:
            candidate = _start_admission_run(
                services,
                candidate_plan.name,
                candidate_plan.resource_claims,
                preemptible=False,
            )
            assert first.wait(2.0).state is RunState.CANCELLED
            assert second.wait(2.0).state is RunState.CANCELLED
            candidate.cancel("test cleanup")
            assert candidate.wait(2.0).state is RunState.CANCELLED

        first.cancel("test cleanup")
        second.cancel("test cleanup")
        assert first.wait(2.0).state is RunState.CANCELLED
        assert second.wait(2.0).state is RunState.CANCELLED


def test_cancel_during_retirement_never_starts_the_candidate(tmp_path) -> None:
    claim = (ResourceClaim(ResourceKey.parse("application/camera")),)
    with _admission_environment(tmp_path) as (services, runtime):
        old = _start_admission_run(
            services,
            "retiring-plan",
            claim,
            preemptible=True,
        )
        release_reached = threading.Event()
        continue_admission = threading.Event()

        def pause_before_final_admission() -> None:
            release_reached.set()
            assert continue_admission.wait(2.0)

        runtime.after_release = pause_before_final_admission
        cancelled = threading.Event()
        failures: list[BaseException] = []
        candidate_plan = _admission_plan(
            "cancelled-candidate-plan",
            object(),
            claim,
            preemptible=False,
        )

        def admit_candidate() -> None:
            try:
                application_start_run(
                    services,
                    candidate_plan,
                    cancel_requested=cancelled.is_set,
                )
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=admit_candidate, daemon=False)
        worker.start()
        assert release_reached.wait(2.0)
        cancelled.set()
        continue_admission.set()
        worker.join(2.0)
        assert not worker.is_alive()
        assert old.wait(2.0).state is RunState.CANCELLED
        assert len(failures) == 1
        assert isinstance(failures[0], CancellationRequested)

        probe = _start_admission_run(
            services,
            "post-cancel-probe",
            claim,
            preemptible=False,
        )
        probe.cancel("test cleanup")
        assert probe.wait(2.0).state is RunState.CANCELLED


def test_a_new_racer_rejects_the_frozen_plan_once(tmp_path) -> None:
    claim = (ResourceClaim(ResourceKey.parse("application/camera")),)
    with _admission_environment(tmp_path) as (services, runtime):
        old = _start_admission_run(
            services,
            "old-plan",
            claim,
            preemptible=True,
        )

        racers = []

        def admit_racer() -> None:
            racers.append(
                _start_admission_run(
                    services,
                    "racer-plan",
                    claim,
                    preemptible=True,
                )
            )

        runtime.after_release = admit_racer
        with pytest.raises(RunStartRejected) as rejected:
            application_start_run(
                services,
                _admission_plan(
                    "raced-candidate-plan",
                    object(),
                    claim,
                    preemptible=False,
                ),
            )

        assert old.wait(2.0).state is RunState.CANCELLED
        assert len(racers) == 1
        racer = racers[0]
        assert rejected.value.blockers == (
            ResourceBusy(claim[0], racer.run_id.value, claim[0]),
        )
        # The final racer is preemptible; remaining RUNNING proves there was
        # no hidden second retirement/retry pass.
        assert racer.snapshot().state is RunState.RUNNING
        racer.cancel("test cleanup")
        assert racer.wait(2.0).state is RunState.CANCELLED


def test_run_plan_has_distinct_prepared_executed_and_final_values(tmp_path):
    item = device_fixture("typed-values")
    runtime, arbiter = controller(tmp_path)
    observations = []

    def preflight(_context):
        observations.append("preflight")
        return ("prepared",)

    def execute(_context, prepared):
        assert prepared == ("prepared",)
        observations.append("execute")
        return {"executed": 3}

    def finalize(_context, executed):
        assert executed == {"executed": 3}
        observations.append("finalize")
        return "final"

    handle = runtime.start(
        plan(item, preflight=preflight, execute=execute, finalize=finalize)
    )
    result = handle.result(2.0)
    assert result == "final"
    assert observations == ["preflight", "execute", "finalize"]
    assert arbiter.wait_until_released((handle.run_id.value,), timeout=0.0)
    close_runtime(runtime, arbiter, item)
def test_resource_claim_is_held_until_cleanup_finishes(tmp_path):
    item = device_fixture("claim")
    runtime, arbiter = controller(tmp_path)
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()

    def cleanup(context, _prepared, _primary):
        cleanup_entered.set()
        assert cleanup_release.wait(1.0)
        return safe_cleanup(context, item.key)

    handle = runtime.start(plan(item, cleanup=cleanup))
    assert cleanup_entered.wait(1.0)
    assert not arbiter.wait_until_released((handle.run_id.value,), timeout=0.0)
    with pytest.raises(RunStartRejected) as caught:
        runtime.start(plan(item))
    assert caught.value.blockers == (
        ResourceBusy(
            ResourceClaim(item.key),
            handle.run_id.value,
            ResourceClaim(item.key),
        ),
    )
    cleanup_release.set()
    assert handle.result(2.0) == "prepared"
    assert arbiter.wait_until_released((handle.run_id.value,), timeout=0.0)
    close_runtime(runtime, arbiter, item)


def test_resource_claim_is_released_before_blocked_finalize(tmp_path):
    item = device_fixture("claim-before-finalize")
    runtime, arbiter = controller(tmp_path)
    finalize_entered = threading.Event()
    finalize_release = threading.Event()

    def finalize(_context, executed):
        finalize_entered.set()
        assert finalize_release.wait(2.0)
        return executed

    first = runtime.start(plan(item, finalize=finalize))
    assert finalize_entered.wait(1.0)
    assert arbiter.wait_until_released((first.run_id.value,), timeout=0.0)

    second = runtime.start(plan(item))
    assert second.result(2.0) == "prepared"
    finalize_release.set()
    assert first.result(2.0) == "prepared"
    close_runtime(runtime, arbiter, item)
def test_child_code_gets_read_only_cancellation_and_handle_owns_transition(tmp_path):
    item = device_fixture("readonly-cancel")
    runtime, arbiter = controller(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def execute(context, prepared):
        assert not hasattr(context.cancellation, "request")
        entered.set()
        assert release.wait(1.0)
        return prepared

    handle = runtime.start(plan(item, execute=execute))
    assert entered.wait(1.0)
    assert handle.cancel("user stop") is CancelOutcome.REQUESTED
    assert handle.snapshot().state is RunState.CANCELLING
    assert handle.cancel("duplicate") is CancelOutcome.ALREADY_REQUESTED
    release.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert handle.snapshot().state is RunState.CANCELLED
    close_runtime(runtime, arbiter, item)
def test_start_returns_before_blocking_identity_probe_and_can_cancel(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    resource = camera_key("blocking-probe")
    physical = identity(resource)

    def probe():
        nonlocal calls
        calls += 1
        if calls > 1:
            entered.set()
            assert release.wait(2.0)
        return physical

    item = device_fixture("blocking-probe", identity_probe=probe)
    executed = []
    runtime, arbiter = controller(tmp_path)
    started = time.monotonic()
    handle = runtime.start(plan(item, execute=lambda *_: executed.append(True)))
    assert time.monotonic() - started < 0.25
    assert entered.wait(1.0)
    assert handle.snapshot().phase == "binding-devices"
    assert handle.cancel("cancel while binding") is CancelOutcome.REQUESTED
    release.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert executed == []
    close_runtime(runtime, arbiter, item)

def test_interrupt_lane_unblocks_execution_before_cleanup(tmp_path):
    release = threading.Event()
    interrupted = threading.Event()

    def interrupt():
        interrupted.set()
        release.set()

    item = device_fixture("interrupt", interrupt=interrupt)
    runtime, arbiter = controller(tmp_path)

    def execute(_context, prepared):
        assert release.wait(2.0)
        return prepared

    handle = runtime.start(plan(item, execute=execute, interrupt=True))
    wait_snapshot(handle, lambda value: value.phase == "execute")
    handle.cancel("abort hardware")
    assert interrupted.wait(1.0)
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    close_runtime(runtime, arbiter, item)


def test_interrupt_failure_preserves_diagnostics_and_cannot_hide_cleanup(tmp_path):
    release = threading.Event()

    def interrupt():
        release.set()
        raise OSError("abort transport failed")

    item = device_fixture("interrupt-failure", interrupt=interrupt)
    runtime, arbiter = controller(tmp_path)
    handle = runtime.start(
        plan(
            item,
            execute=lambda _context, prepared: release.wait(2.0) and prepared,
            interrupt=True,
        )
    )
    wait_snapshot(handle, lambda value: value.phase == "execute")
    handle.cancel("abort hardware")
    with pytest.raises(RunFailed):
        handle.result(2.0)
    snapshot = handle.snapshot()
    assert any("abort transport failed" in value for value in snapshot.cleanup_errors)
    assert all(snapshot.cleanup_errors)
    close_runtime(runtime, arbiter, item)


def test_failure_detachment_preserves_complete_string_evidence():
    detail = "detail-" + "x" * 4096

    def fail(depth: int) -> None:
        if depth:
            fail(depth - 1)
        raise RuntimeError(detail)

    caught = None
    try:
        fail(12)
    except RuntimeError as error:
        for index in range(12):
            error.add_note(f"note-{index}-" + "y" * 600)
        caught = detach_failure(error, note_prefix="detached traceback")
    assert caught is not None
    assert detail in safe_error_summary(caught)
    assert len(caught.locations) >= 13
    assert len(caught.notes) == 13  # traceback note plus every original note
    assert all(
        caught.notes[index + 1].startswith(f"note-{index}-")
        for index in range(12)
    )
def test_cleanup_failure_fails_only_the_current_run(tmp_path):
    item = device_fixture("cleanup-failure")
    runtime, arbiter = controller(tmp_path)
    handle = runtime.start(
        plan(
            item,
            cleanup=lambda _context, _prepared, _primary: CleanupReport(
                errors=(RuntimeError("synthetic cleanup failure"),)
            ),
        )
    )
    with pytest.raises(RunFailed, match="synthetic cleanup failure"):
        handle.result(2.0)
    assert runtime.start(plan(item)).result(2.0) == "prepared"
    close_runtime(runtime, arbiter, item)


def test_finalize_start_is_linearization_point_for_late_cancel(tmp_path):
    item = device_fixture("finalize-gate")
    runtime, arbiter = controller(tmp_path)
    finalize_entered = threading.Event()
    release = threading.Event()
    observed = []

    def finalize(context: PostSafetyContext, executed):
        observed.append(context.run_id)
        finalize_entered.set()
        assert release.wait(1.0)
        return f"final:{executed}"

    handle = runtime.start(plan(item, finalize=finalize))
    assert finalize_entered.wait(1.0)
    assert observed == [handle.run_id]
    assert handle.cancel("too late") is CancelOutcome.TOO_LATE_FINALIZING
    release.set()
    assert handle.result(2.0) == "final:prepared"
    close_runtime(runtime, arbiter, item)


def test_finalize_failure_is_an_ordinary_run_failure_after_release(tmp_path):
    item = device_fixture("finalize-failure")
    runtime, arbiter = controller(tmp_path)

    def finalize(context, _executed):
        assert arbiter.wait_until_released((context.run_id.value,), timeout=0.0)
        raise RuntimeError("domain finalization failed")

    handle = runtime.start(plan(item, finalize=finalize))
    with pytest.raises(RunFailed, match="domain finalization failed"):
        handle.result(2.0)
    assert handle.snapshot().state is RunState.FAILED
    close_runtime(runtime, arbiter, item)


def test_cancel_before_finalize_skips_domain_output(tmp_path):
    item = device_fixture("cancel-before-finalize")
    runtime, arbiter = controller(tmp_path)
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()
    finalized = []

    def cleanup(context, _prepared, _primary):
        cleanup_entered.set()
        assert cleanup_release.wait(1.0)
        return safe_cleanup(context, item.key)

    handle = runtime.start(
        plan(
            item,
            cleanup=cleanup,
            finalize=lambda _context, _executed: finalized.append(True),
        )
    )
    assert cleanup_entered.wait(1.0)
    assert handle.cancel("before domain output") is CancelOutcome.REQUESTED
    cleanup_release.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert finalized == []
    assert arbiter.wait_until_released((handle.run_id.value,), timeout=0.0)
    close_runtime(runtime, arbiter, item)


def test_post_safety_context_only_exposes_run_identity(tmp_path):
    item = device_fixture("post-safety-context")
    runtime, arbiter = controller(tmp_path)
    observed = []

    def finalize(context: PostSafetyContext, executed):
        observed.append(context)
        assert arbiter.wait_until_released((context.run_id.value,), timeout=0.0)
        for removed_capability in (
            "cancellation",
            "deadline",
            "checkpoint",
            "device",
            "cleanup_device",
        ):
            assert not hasattr(context, removed_capability)
        return executed

    handle = runtime.start(plan(item, finalize=finalize))
    assert handle.result(2.0) == "prepared"
    assert observed[0].run_id == handle.run_id
    with pytest.raises(AttributeError, match="immutable"):
        observed[0].run_id = handle.run_id
    close_runtime(runtime, arbiter, item)


def test_controller_shutdown_closes_admission_then_cancels_and_drains(tmp_path):
    item = device_fixture("shutdown")
    runtime, arbiter = controller(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def execute(_context, prepared):
        entered.set()
        assert release.wait(2.0)
        return prepared

    handle = runtime.start(plan(item, execute=execute))
    assert entered.wait(1.0)
    assert not runtime.shutdown(0.01)
    with pytest.raises(RuntimeError, match="rejects new Runs"):
        runtime.start(plan(item))
    assert handle.snapshot().state is RunState.CANCELLING
    release.set()
    assert runtime.shutdown(2.0)
    assert handle.snapshot().state is RunState.CANCELLED
    item.broker.shutdown()
    arbiter.shutdown()


def test_owner_thread_start_failure_releases_unarmed_claim(tmp_path, monkeypatch):
    item = device_fixture("thread-start")
    runtime, arbiter = controller(tmp_path)
    original = threading.Thread.start

    def fail_owner_start(thread):
        if thread.name.startswith("zlc-run-"):
            raise OSError("synthetic owner thread start failure")
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_owner_start)
    with pytest.raises(RuntimeError, match="owner thread failed to start"):
        runtime.start(plan(item))
    probe = arbiter.acquire_all(
        "post-start-failure-probe",
        (ResourceClaim(item.key),),
    )
    assert isinstance(probe, ResourceLease)
    assert probe.release()
    close_runtime(runtime, arbiter, item)
