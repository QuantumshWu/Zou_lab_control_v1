"""Current flat Run lifecycle and FINAL commit contracts."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.commit import (
    CommitTarget,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
)
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
)
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    CapabilityRevoked,
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
from zlc_storage import RepositoryRootLease


def camera_key(name: str = "a") -> ResourceKey:
    return ResourceKey.parse(f"device/camera/{name}")


def identity(resource: ResourceKey) -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        stable_device_identity=str(resource),
        evidence_kind=DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
        evidence_digest=f"identity-readback:{resource}",
        asset_map_revision="test-assets-v1",
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
            acknowledgement_digest="test-session-closed",
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
    requires_final_commit: bool = False,
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
        requires_final_commit=requires_final_commit,
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


def open_commit_coordinator(root):
    root.mkdir()
    journal = PersistentCommitJournal(
        root / "commit-intents.zlcj",
        "test-repository",
    )
    lease = RepositoryRootLease(root)
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: None,
        root_lease=lease,
    )
    return lease, coordinator


def commit_final(
    context: PostSafetyContext,
    coordinator: RepositoryCommitCoordinator,
    *,
    commit_id: str,
    publish,
):
    target = CommitTarget(
        repository_id="test-repository",
        artifact_kind="test-artifact",
        artifact_format="tests.Artifact",
        target_ref=f"artifacts/{commit_id}",
        expected_manifest_digest="0" * 64,
    )
    run_id = context.authorize_commit_preparation()
    operation = coordinator.prepare(
        commit_id,
        run_id,
        target,
        lambda: PublishedManifest(
            target.target_ref,
            target.expected_manifest_digest,
            publish(),
        ),
    )
    context._track_prepared_commit(operation)
    return context.commit_final(operation)


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

    result = runtime.start(
        plan(item, preflight=preflight, execute=execute, finalize=finalize)
    ).result(2.0)
    assert result == "final"
    assert observations == ["preflight", "execute", "finalize"]
    assert not arbiter.active_claims()
    close_runtime(runtime, arbiter, item)
def test_resource_claim_is_held_through_cleanup_until_terminal_publication(tmp_path):
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
    assert arbiter.active_claims()
    with pytest.raises(RunStartRejected) as caught:
        runtime.start(plan(item))
    assert isinstance(caught.value.outcome, ResourceBusy)
    cleanup_release.set()
    assert handle.result(2.0) == "prepared"
    assert not arbiter.active_claims()
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


def test_interrupt_failure_is_bounded_and_cannot_hide_cleanup(tmp_path):
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
    assert len(snapshot.cleanup_errors) <= 2
    close_runtime(runtime, arbiter, item)


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


def test_final_commit_is_linearization_point_for_late_cancel(tmp_path):
    item = device_fixture("commit-gate")
    runtime, arbiter = controller(tmp_path)
    lease, coordinator = open_commit_coordinator(tmp_path / "repository")
    committed = threading.Event()
    release = threading.Event()

    def finalize(context, _executed):
        result = commit_final(
            context,
            coordinator,
            commit_id="final-linearization",
            publish=lambda: "committed-ref",
        )
        committed.set()
        assert release.wait(1.0)
        return result

    handle = runtime.start(
        plan(item, finalize=finalize, requires_final_commit=True)
    )
    assert committed.wait(1.0)
    assert handle.cancel("too late") is CancelOutcome.TOO_LATE_ALREADY_COMMITTED
    release.set()
    assert handle.result(2.0) == "committed-ref"
    coordinator.close()
    lease.close()
    close_runtime(runtime, arbiter, item)


def test_required_final_commit_cannot_succeed_with_return_value_only(tmp_path):
    item = device_fixture("missing-commit")
    runtime, arbiter = controller(tmp_path)
    handle = runtime.start(plan(item, requires_final_commit=True))
    with pytest.raises(RunFailed, match="requires a final commit"):
        handle.result(2.0)
    close_runtime(runtime, arbiter, item)


def test_leaked_post_safety_context_is_revoked_after_finalize(tmp_path):
    item = device_fixture("revoke-post")
    runtime, arbiter = controller(tmp_path)
    leaked = []

    def finalize(context, executed):
        leaked.append(context)
        return executed

    assert runtime.start(plan(item, finalize=finalize)).result(2.0) == "prepared"
    with pytest.raises(CapabilityRevoked, match="left finalize"):
        leaked[0].checkpoint()
    with pytest.raises(CapabilityRevoked, match="left finalize"):
        leaked[0].authorize_commit_preparation()
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
    assert not arbiter.active_claims()
    close_runtime(runtime, arbiter, item)
