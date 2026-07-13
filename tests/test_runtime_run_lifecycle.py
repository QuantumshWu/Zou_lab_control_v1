"""Truthful lifecycle contracts for RunController and RunHandle."""

from __future__ import annotations

import gc
import math
import threading
import time
import weakref
from dataclasses import fields
from itertools import count

import pytest

from zlc_neutral_atom.runtime import (
    CancelOutcome,
    CheckpointCommit,
    CheckpointDisposition,
    CheckpointReconciliationRequired,
    CheckpointSnapshot,
    BoundDevice,
    CapabilityRevoked,
    CommitRecovery,
    CommitIntent,
    CommitKind,
    CommitSubject,
    CommitTarget,
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
    CleanupReport,
    HazardClaim,
    MemoryCommitJournal,
    MemoryQuarantineJournal,
    FinalCommit,
    FinalCommitReconciliationRequired,
    PostSafetyContext,
    PublishVisibilityUnknown,
    PublishedManifest,
    QuarantineJournalError,
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    ResourceQuarantined,
    RunCancelled,
    RunContext,
    RunController,
    RunFailed,
    RunMode,
    RunPlan,
    RunStartRejected,
    RunState,
    SafeReceipt,
    SafetyDecision,
    CleanupStepAck,
    SafeStateAck,
    SafetyInterrupt,
    SafetyOperation,
)
from zlc_neutral_atom.runtime.commit import RepositoryCommitCoordinator
from zlc_neutral_atom.runtime.run import RunHandle, _MISSING


_COMMIT_IDS = count()


class FailFirstCommitMarkerJournal(MemoryCommitJournal):
    def __init__(self, *, fail_abort: bool = False, fail_commit: bool = False):
        super().__init__("test-repository")
        self.fail_abort = fail_abort
        self.fail_commit = fail_commit

    def mark_aborted(self, commit_id: str) -> None:
        if self.fail_abort:
            self.fail_abort = False
            raise OSError("abort marker acknowledgement lost")
        super().mark_aborted(commit_id)

    def mark_committed(self, commit_id: str) -> None:
        if self.fail_commit:
            self.fail_commit = False
            raise OSError("commit marker acknowledgement lost")
        super().mark_committed(commit_id)


def camera_key(name: str = "a") -> ResourceKey:
    return ResourceKey.parse(f"device/camera/{name}")


def new_arbiter() -> ResourceArbiter:
    return ResourceArbiter(MemoryQuarantineJournal())


def verified_identity(
    broker: DeviceBroker,
    key: ResourceKey,
    evidence_digest: str,
):
    return broker.verify_identity(
        lambda: DeviceIdentityAck(
            str(key),
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
            evidence_digest,
            "test-assets-v1",
        )
    )


def safe_ack(
    operation: SafetyOperation = SafetyOperation.SAFE_STATE,
    digest: str = "test-safe-acknowledgement",
) -> CleanupStepAck:
    return CleanupStepAck(operation=operation, acknowledgement_digest=digest)


def safe_cleanup_report(context: RunContext, key: ResourceKey) -> CleanupReport:
    device = context.cleanup_device(key)
    device.perform(SafetyOperation.SAFE_STATE)
    return CleanupReport.safe((device.verify_safe_state(),))


def commit(
    context: PostSafetyContext,
    publish,
    recover=lambda _intent: CommitRecovery(committed=False),
    *,
    journal=None,
    commit_id=None,
):
    commit_id = commit_id or f"test-commit-{next(_COMMIT_IDS)}"
    journal = journal or MemoryCommitJournal("test-repository")
    target = CommitTarget(
        repository_id="test-repository",
        artifact_kind="test-artifact",
        schema_version="1",
        target_ref=f"artifacts/{commit_id}",
        expected_manifest_digest="0" * 64,
    )
    coordinator = RepositoryCommitCoordinator(
        journal,
        recover,
        allow_ephemeral=True,
    )

    def publish_manifest():
        return PublishedManifest(
            target_ref=target.target_ref,
            manifest_digest=target.expected_manifest_digest,
            result=publish(),
        )

    subject = context.authorize_commit_preparation(CommitKind.FINAL)
    return context.commit_final(
        FinalCommit(
            coordinator.prepare(
                CommitKind.FINAL,
                commit_id,
                subject,
                target,
                publish_manifest,
            )
        )
    )


def checkpoint_commit(
    context: PostSafetyContext,
    publish,
    recover=lambda _intent: CommitRecovery(committed=False),
    *,
    journal=None,
    commit_id=None,
):
    commit_id = commit_id or f"test-checkpoint-{next(_COMMIT_IDS)}"
    journal = journal or MemoryCommitJournal("test-repository")
    target = CommitTarget(
        repository_id="test-repository",
        artifact_kind="test-checkpoint",
        schema_version="1",
        target_ref=f"artifacts/{commit_id}",
        expected_manifest_digest="0" * 64,
    )
    coordinator = RepositoryCommitCoordinator(
        journal,
        recover,
        allow_ephemeral=True,
    )

    def publish_manifest():
        return PublishedManifest(
            target_ref=target.target_ref,
            manifest_digest=target.expected_manifest_digest,
            result=publish(),
        )

    subject = context.authorize_commit_preparation(CommitKind.CHECKPOINT)
    return context.commit_checkpoint(
        CheckpointCommit(
            coordinator.prepare(
                CommitKind.CHECKPOINT,
                commit_id,
                subject,
                target,
                publish_manifest,
            )
        )
    )


def prepared_authority(
    context: PostSafetyContext,
    coordinator: RepositoryCommitCoordinator,
    kind: CommitKind,
    commit_id: str,
    target: CommitTarget,
    publish,
    *,
    subject: CommitSubject | None = None,
):
    if subject is None:
        subject = context.authorize_commit_preparation(kind)
    return coordinator.prepare(kind, commit_id, subject, target, publish)


def plan(
    *,
    key: ResourceKey,
    execute,
    preflight=lambda _ctx: "prepared",
    cleanup=None,
    finalize=lambda _ctx, result: result,
    interrupt=None,
    cleanup_ack=safe_ack,
    requires_final_commit=False,
    timeout_seconds=None,
) -> RunPlan:
    generation = "test-connection-generation"
    safety_operations = {}
    interrupt_operations = ()
    if interrupt is not None:
        safety_operations[SafetyOperation.ABORT] = lambda: interrupt(None)
        interrupt_operations = (SafetyInterrupt(key, SafetyOperation.ABORT),)
    if cleanup is None:
        cleanup = lambda context, _prepared, _primary: safe_cleanup_report(context, key)
    broker = DeviceBroker()
    device = broker.bind(
        key=key,
        identity=verified_identity(broker, key, generation),
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.SAFE_STATE: cleanup_ack},
        verify_safe_state=lambda: SafeStateAck("test-safe-state-readback"),
        interrupt_operations=safety_operations,
    )
    return RunPlan(
        name="test run",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(ResourceClaim(key),),
        hazard_claims=(
            HazardClaim(key, str(key), device.connection_generation),
        ),
        bound_devices=(device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=interrupt_operations,
        requires_final_commit=requires_final_commit,
        timeout_seconds=timeout_seconds,
    )


def test_success_returns_result_and_releases_claim_only_after_cleanup():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    cleaned = threading.Event()

    def cleanup(context, _prepared, _primary):
        assert arbiter.active_claims()
        cleaned.set()
        return safe_cleanup_report(context, camera_key())

    handle = controller.start(
        plan(key=camera_key(), execute=lambda _ctx, prepared: prepared + " result", cleanup=cleanup)
    )
    assert handle.result(2.0) == "prepared result"
    assert cleaned.is_set()
    assert handle.snapshot().state is RunState.SUCCEEDED
    assert not arbiter.active_claims()


def test_resource_conflict_rejects_start_without_stopping_owner():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    release = threading.Event()
    started = threading.Event()

    def execute(_ctx, _prepared):
        started.set()
        release.wait()
        return "done"

    first = controller.start(plan(key=camera_key(), execute=execute))
    assert started.wait(1.0)
    with pytest.raises(RunStartRejected) as caught:
        controller.start(plan(key=camera_key(), execute=lambda *_: None))
    assert isinstance(caught.value.outcome, ResourceBusy)
    assert first.snapshot().state is RunState.RUNNING
    release.set()
    assert first.result(2.0) == "done"


def test_cancel_does_not_release_claim_or_report_terminal_until_worker_exits():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    started = threading.Event()
    allow_exit = threading.Event()

    def execute(ctx: RunContext, _prepared):
        started.set()
        allow_exit.wait()
        ctx.checkpoint()

    handle = controller.start(plan(key=camera_key(), execute=execute))
    assert started.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    assert handle.snapshot().state is RunState.CANCELLING
    with pytest.raises(TimeoutError):
        handle.wait(0.01)
    with pytest.raises(RunStartRejected):
        controller.start(plan(key=camera_key(), execute=lambda *_: None))

    allow_exit.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert handle.snapshot().state is RunState.CANCELLED
    assert not arbiter.active_claims()


def test_execute_error_is_primary_and_cleanup_error_is_additional():
    controller = RunController(new_arbiter())

    def execute(_ctx, _prepared):
        raise ValueError("physics failed")

    def cleanup(context, _prepared, primary):
        assert isinstance(primary, ValueError)
        device = context.cleanup_device(camera_key())
        device.perform(SafetyOperation.SAFE_STATE)
        receipt = device.verify_safe_state()
        return CleanupReport.safe(
            (receipt,), errors=(OSError("close failed"),)
        )

    handle = controller.start(plan(key=camera_key(), execute=execute, cleanup=cleanup))
    with pytest.raises(RunFailed) as caught:
        handle.result(2.0)
    assert caught.value.primary is not None
    assert caught.value.primary.original_type == "ValueError"
    snapshot = handle.snapshot()
    assert snapshot.primary_error == "ValueError: physics failed"
    assert snapshot.cleanup_errors == ("OSError: close failed",)


def test_finalize_failure_drops_partial_result_and_traceback_graph_without_gc():
    class PartialResult:
        pass

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result_reference = None

        def execute(_context, _prepared):
            nonlocal result_reference
            result = PartialResult()
            result_reference = weakref.ref(result)
            return result

        def finalize(_context, partial_result):
            raise RuntimeError("synthetic finalize failure", partial_result)

        handle = RunController(new_arbiter()).start(
            plan(key=camera_key("partial-result"), execute=execute, finalize=finalize)
        )
        with pytest.raises(RunFailed) as caught:
            handle.result(2.0)

        assert result_reference is not None and result_reference() is None
        assert handle._result is _MISSING
        assert caught.value.primary is handle._primary
        assert caught.value.primary is not None
        assert caught.value.primary.original_type == "RuntimeError"
        assert caught.value.primary.__traceback__ is None
        assert any(
            note.startswith("detached run traceback: ")
            for note in getattr(caught.value.primary, "__notes__", ())
        )
    finally:
        if was_enabled:
            gc.enable()


def test_terminal_failure_evidence_cannot_retain_prepared_exception_payload():
    class Prepared:
        pass

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        prepared_reference = None

        def preflight(_context):
            nonlocal prepared_reference
            prepared = Prepared()
            prepared_reference = weakref.ref(prepared)
            return prepared

        def execute(_context, prepared):
            raise RuntimeError("failure embeds prepared", prepared)

        handle = RunController(new_arbiter()).start(
            plan(
                key=camera_key("prepared-payload"),
                preflight=preflight,
                execute=execute,
            )
        )
        with pytest.raises(RunFailed) as caught:
            handle.result(2.0)

        assert prepared_reference is not None and prepared_reference() is None
        assert caught.value.primary is not None
        assert caught.value.primary.original_type == "RuntimeError"
        assert all(type(argument) is str for argument in caught.value.primary.args)
    finally:
        if was_enabled:
            gc.enable()


def test_oversized_exception_group_keeps_root_identity_in_bounded_evidence():
    try:
        exception_group_type = ExceptionGroup
    except NameError:
        pytest.skip("ExceptionGroup requires Python 3.11+")
    failures = tuple(RuntimeError(f"child-{index}") for index in range(40))

    def execute(_context, _prepared):
        raise exception_group_type("many failures", failures)

    handle = RunController(new_arbiter()).start(
        plan(key=camera_key("large-exception-group"), execute=execute)
    )
    with pytest.raises(RunFailed) as caught:
        handle.result(2.0)

    assert caught.value.primary is not None
    assert caught.value.primary.original_type == "ExceptionGroup"
    assert caught.value.primary.truncated
    assert "exception graph exceeded 32 nodes" in str(caught.value.primary)


def test_hostile_exception_hooks_cannot_delay_terminal_publication(monkeypatch):
    import zlc_neutral_atom.runtime.run as run_module

    class HostileFailure(RuntimeError):
        @property
        def args(self):
            raise AssertionError("subclass args descriptor must not run")

        def __str__(self):
            raise AssertionError("subclass __str__ must not run")

        def __repr__(self):
            raise AssertionError("subclass __repr__ must not run")

        def add_note(self, _note):
            raise AssertionError("subclass add_note must not run")

    primary = HostileFailure("hostile primary")

    def finalize(_context, _result):
        raise primary

    def fail_abandonment(_context):
        raise HostileFailure("hostile secondary")

    monkeypatch.setattr(
        run_module.PostSafetyContext,
        "_abandon_unconsumed_commits",
        fail_abandonment,
    )
    arbiter = new_arbiter()
    handle = RunController(arbiter).start(
        plan(
            key=camera_key("hostile-terminal"),
            execute=lambda *_: "partial",
            finalize=finalize,
        )
    )
    with pytest.raises(RunFailed) as caught:
        handle.result(2.0)

    assert caught.value.primary is not None
    assert caught.value.primary.original_type == "HostileFailure"
    assert handle.snapshot().state is RunState.FAILED
    assert not arbiter.active_claims()


def test_internal_lifecycle_fault_uses_fail_closed_terminal_fallback(monkeypatch):
    import zlc_neutral_atom.runtime.run as run_module

    def fail_terminal_classification(*_args):
        raise RuntimeError("synthetic lifecycle implementation fault")

    monkeypatch.setattr(run_module, "_terminal_state", fail_terminal_classification)
    arbiter = new_arbiter()
    handle = RunController(arbiter).start(
        plan(key=camera_key("fail-closed-owner"), execute=lambda *_: "result")
    )
    with pytest.raises(RunFailed, match="synthetic lifecycle implementation fault"):
        handle.result(2.0)

    assert handle.snapshot().state is RunState.FAILED
    assert not arbiter.active_claims()


def test_terminal_handle_does_not_retain_controller_without_gc():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        controller = RunController(new_arbiter())
        controller_reference = weakref.ref(controller)
        handle = controller.start(
            plan(key=camera_key("observer-release"), execute=lambda *_: "done")
        )
        assert handle.result(2.0) == "done"
        assert handle._on_terminal is None

        del controller
        assert controller_reference() is None
    finally:
        if was_enabled:
            gc.enable()


def test_thread_start_failure_does_not_retain_hidden_run_graph_without_gc(monkeypatch):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        references = {}
        original_init = RunHandle.__init__

        def observed_init(handle, *args, **kwargs):
            original_init(handle, *args, **kwargs)
            references["handle"] = weakref.ref(handle)

        def fail_start(thread):
            references["thread"] = weakref.ref(thread)
            raise OSError("synthetic Run owner start failure")

        monkeypatch.setattr(RunHandle, "__init__", observed_init)
        monkeypatch.setattr(threading.Thread, "start", fail_start)
        controller = RunController(new_arbiter())
        controller_reference = weakref.ref(controller)

        with pytest.raises(RuntimeError, match="Run owner thread failed to start"):
            controller.start(
                plan(key=camera_key("owner-start-failure"), execute=lambda *_: None)
            )

        del controller
        assert controller_reference() is None
        assert references["handle"]() is None
        assert references["thread"]() is None
    finally:
        if was_enabled:
            gc.enable()


def test_unconfirmed_safe_state_quarantines_only_declared_failed_resource():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    key = camera_key("unsafe")

    def cleanup(_ctx, _prepared, _primary):
        return CleanupReport.unsafe(
            (key,),
            reason="camera disarm acknowledgement failed",
            recovery_action="verify camera idle",
        )

    handle = controller.start(plan(key=key, execute=lambda *_: "data", cleanup=cleanup))
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert isinstance(
        arbiter.acquire_all("retry", (ResourceClaim(key),)),
        ResourceQuarantined,
    )


def test_cancel_after_atomic_final_commit_is_too_late_and_run_succeeds():
    controller = RunController(new_arbiter())
    committed = threading.Event()
    finish = threading.Event()

    def finalize(ctx: PostSafetyContext, _result):
        result = commit(ctx, lambda: "artifact-ref")
        committed.set()
        finish.wait()
        return result

    handle = controller.start(
        plan(
            key=camera_key(),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert committed.wait(1.0)
    assert handle.cancel() is CancelOutcome.TOO_LATE_ALREADY_COMMITTED
    finish.set()
    assert handle.result(2.0) == "artifact-ref"


def test_cancel_wins_before_commit_gate_and_no_artifact_is_published():
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    finalize_ready = threading.Event()
    try_commit = threading.Event()
    artifact_exists = threading.Event()

    def finalize(ctx: PostSafetyContext, _result):
        finalize_ready.set()
        try_commit.wait()
        return commit(ctx, lambda: artifact_exists.set())

    handle = controller.start(
        plan(
            key=camera_key("cancel-first"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert finalize_ready.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    try_commit.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert not artifact_exists.is_set()


def test_cancel_remains_available_while_commit_intent_is_being_persisted():
    controller = RunController(new_arbiter())
    intent_started = threading.Event()
    allow_intent = threading.Event()
    published = threading.Event()

    class BlockingIntentJournal(MemoryCommitJournal):
        def begin(self, intent):
            intent_started.set()
            allow_intent.wait()
            super().begin(intent)

    journal = BlockingIntentJournal("test-repository")

    def finalize(context: PostSafetyContext, _result):
        return commit(
            context,
            lambda: published.set(),
            journal=journal,
            commit_id="cancel-during-intent",
        )

    handle = controller.start(
        plan(
            key=camera_key("cancel-during-intent"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert intent_started.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    allow_intent.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert not published.is_set()
    assert journal.pending() == ()


def test_commit_callback_and_marker_share_one_gate_against_cancel():
    controller = RunController(ResourceArbiter(MemoryQuarantineJournal()))
    publish_entered = threading.Event()
    allow_replace = threading.Event()
    artifact_exists = threading.Event()
    cancel_outcomes = []

    def publish():
        publish_entered.set()
        allow_replace.wait()
        artifact_exists.set()
        return "committed-ref"

    def finalize(ctx: PostSafetyContext, _result):
        return commit(ctx, publish)

    handle = controller.start(
        plan(
            key=camera_key("commit-first"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert publish_entered.wait(1.0)
    cancel_thread = threading.Thread(target=lambda: cancel_outcomes.append(handle.cancel()))
    cancel_thread.start()
    allow_replace.set()
    cancel_thread.join()
    assert cancel_outcomes[0] is CancelOutcome.TOO_LATE_FINALIZING
    assert handle.result(2.0) == "committed-ref"
    assert artifact_exists.is_set()


def test_cancel_during_cleanup_cannot_be_reported_as_success():
    controller = RunController(new_arbiter())
    cleanup_entered = threading.Event()
    finish_cleanup = threading.Event()

    def cleanup(context, _prepared, _primary):
        cleanup_entered.set()
        finish_cleanup.wait()
        return safe_cleanup_report(context, camera_key("cleanup-cancel"))

    handle = controller.start(
        plan(key=camera_key("cleanup-cancel"), execute=lambda *_: "result", cleanup=cleanup)
    )
    assert cleanup_entered.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    finish_cleanup.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)


def test_interrupt_failure_is_visible_and_cannot_hide_successfully_safe_cleanup():
    controller = RunController(new_arbiter())
    started = threading.Event()
    allow_exit = threading.Event()

    def execute(ctx: RunContext, _prepared):
        started.set()
        allow_exit.wait()
        ctx.checkpoint()

    def interrupt(_ctx):
        allow_exit.set()
        raise OSError("abort transport failed")

    handle = controller.start(plan(key=camera_key(), execute=execute, interrupt=interrupt))
    assert started.wait(1.0)
    handle.cancel()
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert any(
        "OSError: abort transport failed" in error
        for error in handle.snapshot().cleanup_errors
    )


def test_interrupt_inflight_blocks_terminal_and_resource_release():
    arbiter = ResourceArbiter(MemoryQuarantineJournal())
    controller = RunController(arbiter)
    execute_started = threading.Event()
    allow_execute_exit = threading.Event()
    interrupt_started = threading.Event()
    allow_interrupt_exit = threading.Event()
    key = camera_key("interrupt-barrier")

    def execute(ctx: RunContext, _prepared):
        execute_started.set()
        allow_execute_exit.wait()
        ctx.checkpoint()

    def interrupt(_ctx):
        interrupt_started.set()
        allow_execute_exit.set()
        allow_interrupt_exit.wait()
        raise OSError("late interrupt failure")

    handle = controller.start(plan(key=key, execute=execute, interrupt=interrupt))
    assert execute_started.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    assert interrupt_started.wait(1.0)
    with pytest.raises(TimeoutError):
        handle.wait(0.01)
    assert isinstance(
        arbiter.acquire_all("blocked-by-interrupt", (ResourceClaim(key),)),
        ResourceBusy,
    )

    allow_interrupt_exit.set()
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert any(
        "OSError: late interrupt failure" in error
        for error in handle.snapshot().cleanup_errors
    )


def test_cancel_after_cleanup_begins_does_not_race_a_second_hardware_interrupt():
    controller = RunController(new_arbiter())
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    interrupt_called = threading.Event()

    def cleanup(context, _prepared, _primary):
        cleanup_started.set()
        allow_cleanup.wait()
        return safe_cleanup_report(context, camera_key("cleanup-gate"))

    def interrupt(_ctx):
        interrupt_called.set()

    handle = controller.start(
        plan(
            key=camera_key("cleanup-gate"),
            execute=lambda *_: "done",
            cleanup=cleanup,
            interrupt=interrupt,
        )
    )
    assert cleanup_started.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    assert not interrupt_called.is_set()
    allow_cleanup.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)


def test_quarantine_journal_failure_holds_claim_until_explicit_retry():
    class RecoverableJournal(MemoryQuarantineJournal):
        fail = True

        def append_safety_bundle(self, bundle):
            if self.fail:
                raise OSError("safety disk offline")
            super().append_safety_bundle(bundle)

    journal = RecoverableJournal()
    arbiter = ResourceArbiter(journal)
    controller = RunController(arbiter)
    key = camera_key("journal")

    def cleanup(_ctx, _prepared, _primary):
        return CleanupReport.unsafe(
            (key,), reason="safe unknown", recovery_action="repair journal and verify camera"
        )

    handle = controller.start(plan(key=key, execute=lambda *_: None, cleanup=cleanup))
    stalled = handle.wait_for(lambda snap: snap.phase == "safety-journal-failed", timeout=2.0)
    assert stalled.state is RunState.RUNNING
    assert stalled.recovery_instruction is not None
    assert isinstance(
        arbiter.acquire_all("blocked", (ResourceClaim(key),)),
        ResourceBusy,
    )

    journal.fail = False
    assert handle.retry_recovery()
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert isinstance(
        arbiter.acquire_all("quarantined", (ResourceClaim(key),)),
        ResourceQuarantined,
    )


def test_contender_never_observes_claim_free_while_owner_is_nonterminal():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    execute_entered = threading.Event()
    finish_execute = threading.Event()
    acquired = threading.Event()
    observed_states = []
    key = camera_key("linearization")

    def execute(_ctx, _prepared):
        execute_entered.set()
        finish_execute.wait()
        return "done"

    handle = controller.start(plan(key=key, execute=execute))
    assert execute_entered.wait(1.0)

    def contend():
        while not acquired.is_set():
            outcome = arbiter.acquire_all("contender", (ResourceClaim(key),))
            if isinstance(outcome, ResourceBusy):
                continue
            assert not isinstance(outcome, ResourceQuarantined)
            observed_states.append(handle.snapshot().state)
            outcome._release_unarmed()
            acquired.set()

    thread = threading.Thread(target=contend)
    thread.start()
    finish_execute.set()
    assert handle.result(2.0) == "done"
    assert acquired.wait(1.0)
    thread.join()
    assert observed_states == [RunState.SUCCEEDED]


def test_hazard_journal_failure_prevents_preflight_until_explicit_recovery():
    class RecoverableHazardJournal(MemoryQuarantineJournal):
        fail = True

        def append_hazards(self, records):
            if self.fail:
                raise OSError("hazard ledger offline")
            return super().append_hazards(records)

    journal = RecoverableHazardJournal()
    arbiter = ResourceArbiter(journal)
    controller = RunController(arbiter)
    preflight_called = threading.Event()
    key = camera_key("hazard-start")

    def preflight(_ctx):
        preflight_called.set()
        return "prepared"

    handle = controller.start(
        plan(key=key, preflight=preflight, execute=lambda *_: "done")
    )
    stalled = handle.wait_for(lambda snap: snap.phase == "safety-journal-failed", 2.0)
    assert stalled.state is RunState.RUNNING
    assert not preflight_called.is_set()
    assert isinstance(
        arbiter.acquire_all("blocked", (ResourceClaim(key),)), ResourceBusy
    )

    journal.fail = False
    assert handle.retry_recovery()
    assert handle.result(2.0) == "done"
    assert preflight_called.is_set()
    assert not journal.unresolved_hazards()


def test_cancel_cannot_interrupt_before_hazard_active_is_durable():
    class BlockingHazardJournal(MemoryQuarantineJournal):
        entered = threading.Event()
        allow = threading.Event()

        def append_hazards(self, records):
            self.entered.set()
            self.allow.wait()
            return super().append_hazards(records)

    journal = BlockingHazardJournal()
    controller = RunController(ResourceArbiter(journal))
    interrupt_called = threading.Event()
    preflight_called = threading.Event()

    def interrupt(_ctx):
        interrupt_called.set()

    def preflight(_ctx):
        preflight_called.set()
        return "prepared"

    handle = controller.start(
        plan(
            key=camera_key("hazard-cancel-gate"),
            preflight=preflight,
            execute=lambda *_: "unused",
            interrupt=interrupt,
        )
    )
    assert journal.entered.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    assert not interrupt_called.is_set()
    assert not preflight_called.is_set()
    journal.allow.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert not interrupt_called.is_set()
    assert not preflight_called.is_set()


def test_safe_safety_bundle_journal_failure_retries_safe_path_not_quarantine():
    class RecoverableSafeBundleJournal(MemoryQuarantineJournal):
        fail = True

        def append_safety_bundle(self, bundle):
            if self.fail:
                raise OSError("safety bundle disk offline")
            super().append_safety_bundle(bundle)

    journal = RecoverableSafeBundleJournal()
    arbiter = ResourceArbiter(journal)
    controller = RunController(arbiter)
    key = camera_key("safe-bundle-retry")
    handle = controller.start(plan(key=key, execute=lambda *_: "result"))
    stalled = handle.wait_for(
        lambda snapshot: snapshot.phase == "safety-journal-failed",
        2.0,
    )
    assert stalled.state is RunState.RUNNING
    assert isinstance(
        arbiter.acquire_all("blocked-safe-retry", (ResourceClaim(key),)), ResourceBusy
    )
    journal.fail = False
    assert handle.retry_recovery()
    assert handle.result(2.0) == "result"
    assert not arbiter.quarantine_records()


def test_mixed_cleanup_releases_safe_device_and_quarantines_only_failed_device():
    journal = MemoryQuarantineJournal()
    arbiter = ResourceArbiter(journal)
    controller = RunController(arbiter)
    camera = camera_key("mixed")
    fpga = ResourceKey.parse("device/fpga/mixed")
    finalized = threading.Event()

    def cleanup(context, _prepared, _primary):
        device = context.cleanup_device(camera)
        device.perform(SafetyOperation.SAFE_STATE)
        camera_proof = device.verify_safe_state()
        return CleanupReport.mixed(
            (camera_proof,),
            (
                SafetyDecision.unsafe(
                    fpga,
                    reason="sequencer safe readback unavailable",
                    recovery_action="reconnect and verify safe",
                ),
            )
        )

    broker = DeviceBroker()
    camera_device = broker.bind(
        key=camera,
        identity=verified_identity(broker, camera, "camera-generation"),
        execute_command=lambda command: command,
        cleanup_operations={
            SafetyOperation.SAFE_STATE: lambda: safe_ack(digest="camera-safe")
        },
        verify_safe_state=lambda: SafeStateAck("camera-safe-readback"),
    )
    fpga_device = broker.bind(
        key=fpga,
        identity=verified_identity(broker, fpga, "fpga-generation"),
        execute_command=lambda command: command,
        cleanup_operations={
            SafetyOperation.SAFE_STATE: lambda: safe_ack(digest="fpga-safe")
        },
        verify_safe_state=lambda: SafeStateAck("fpga-safe-readback"),
    )
    run_plan = RunPlan(
        name="mixed cleanup",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(ResourceClaim(camera), ResourceClaim(fpga)),
        hazard_claims=(
            HazardClaim(camera, str(camera), camera_device.connection_generation),
            HazardClaim(fpga, str(fpga), fpga_device.connection_generation),
        ),
        preflight=lambda _ctx: "prepared",
        execute=lambda _ctx, _prepared: "result",
        cleanup=cleanup,
        bound_devices=(camera_device, fpga_device),
        finalize=lambda _ctx, result: finalized.set() or result,
    )
    handle = controller.start(run_plan)
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert not finalized.is_set()
    camera_lease = arbiter.acquire_all("camera-after-mixed", (ResourceClaim(camera),))
    assert isinstance(camera_lease, ResourceLease)
    camera_lease._release_unarmed()
    assert isinstance(
        arbiter.acquire_all("fpga-after-mixed", (ResourceClaim(fpga),)),
        ResourceQuarantined,
    )


def test_safety_bundle_revokes_hardware_before_finalize_but_retains_claim():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    key = camera_key("post-safety")
    finalized = threading.Event()

    def finalize(ctx: PostSafetyContext, _result):
        assert ctx.safety_bundle_id is not None
        assert arbiter.active_claims()
        finalized.set()
        return commit(ctx, lambda: "artifact-ref")

    handle = controller.start(
        plan(
            key=key,
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert handle.result(2.0) == "artifact-ref"
    assert finalized.is_set()
    assert handle.snapshot().safety_bundle_id is not None
    assert not arbiter.active_claims()


def test_cancel_after_safety_bundle_never_restarts_hardware_interrupt():
    controller = RunController(new_arbiter())
    finalize_entered = threading.Event()
    allow_finalize = threading.Event()
    interrupt_called = threading.Event()

    def finalize(ctx: PostSafetyContext, result):
        finalize_entered.set()
        allow_finalize.wait()
        ctx.checkpoint()
        return result

    handle = controller.start(
        plan(
            key=camera_key("post-safety-cancel"),
            execute=lambda *_: "result",
            finalize=finalize,
            interrupt=lambda _ctx: interrupt_called.set(),
        )
    )
    assert finalize_entered.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    assert not interrupt_called.is_set()
    allow_finalize.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert not interrupt_called.is_set()


def test_run_plan_rejects_non_finite_timeouts_and_terminal_history_is_bounded():
    base = dict(
        name="invalid timeout",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(),
        hazard_claims=(),
        preflight=lambda _ctx: None,
        execute=lambda _ctx, _prepared: None,
        cleanup=lambda _ctx, _prepared, _primary: CleanupReport(),
        bound_devices=(),
        finalize=lambda _ctx, result: result,
    )
    for invalid in (math.nan, math.inf, -math.inf, -1.0):
        with pytest.raises(ValueError):
            RunPlan(**base, timeout_seconds=invalid)
    assert RunPlan(**base, timeout_seconds=0.0).timeout_seconds == 0.0

    controller = RunController(new_arbiter(), terminal_history_limit=2)
    handles = [controller.start(RunPlan(**base)) for _ in range(4)]
    for handle in handles:
        assert handle.result(2.0) is None
    snapshots = controller.snapshots()
    assert len(snapshots) == 2
    with pytest.raises(KeyError):
        controller.lookup(handles[0].run_id)
    assert controller.forget_terminal(handles[-1].run_id)


def test_cleanup_rejects_untyped_acknowledgement_and_quarantines_device():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    key = camera_key("untyped-ack")
    handle = controller.start(
        plan(
            key=key,
            execute=lambda *_: "result",
            cleanup_ack=lambda: "free-text-is-not-a-safety-ack",
        )
    )

    with pytest.raises(RunFailed):
        handle.result(2.0)

    assert any(
        "must return CleanupStepAck" in error
        for error in handle.snapshot().cleanup_errors
    )
    assert isinstance(
        arbiter.acquire_all("blocked-untyped-ack", (ResourceClaim(key),)),
        ResourceQuarantined,
    )


def test_cleanup_cannot_forge_safe_decision_without_running_device_operation():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    key = camera_key("forged-safe")
    safety_called = threading.Event()

    def forged_cleanup(_context, _prepared, _primary):
        forged = SafeReceipt(
            key=key,
            stable_device_identity=str(key),
            connection_generation="test-connection-generation",
            operation_id="VERIFY_SAFE_STATE",
            acknowledgement_digest="fabricated",
        )
        return CleanupReport(decisions=(SafetyDecision.safe(forged),))

    def real_safety_operation():
        safety_called.set()
        return safe_ack()

    handle = controller.start(
        plan(
            key=key,
            execute=lambda *_: "result",
            cleanup=forged_cleanup,
            cleanup_ack=real_safety_operation,
        )
    )

    with pytest.raises(RunFailed):
        handle.result(2.0)

    assert not safety_called.is_set()
    assert any(
        "SAFE cleanup decisions require a RunContext SafetyProof" in error
        for error in handle.snapshot().cleanup_errors
    )
    assert isinstance(
        arbiter.acquire_all("blocked-forged-safe", (ResourceClaim(key),)),
        ResourceQuarantined,
    )


def test_read_status_ack_cannot_be_used_as_verified_safe_state():
    key = camera_key("status-is-not-safe")
    broker = DeviceBroker()
    verified = threading.Event()
    device = broker.bind(
        key=key,
        identity=verified_identity(broker, key, "generation"),
        execute_command=lambda command: command,
        cleanup_operations={
            SafetyOperation.READ_STATUS: lambda: CleanupStepAck(
                SafetyOperation.READ_STATUS,
                "status-only",
            )
        },
        verify_safe_state=lambda: verified.set() or SafeStateAck("safe"),
    )

    def cleanup(context, _prepared, _primary):
        status_ack = context.cleanup_device(key).perform(SafetyOperation.READ_STATUS)
        return CleanupReport.safe((status_ack,))

    handle = RunController(new_arbiter()).start(
        RunPlan(
            name="status is not safety proof",
            mode=RunMode.FINITE_EXACT,
            resource_claims=(ResourceClaim(key),),
            hazard_claims=(
                HazardClaim(key, str(key), device.connection_generation),
            ),
            bound_devices=(device,),
            preflight=lambda _context: None,
            execute=lambda _context, _prepared: None,
            cleanup=cleanup,
            finalize=lambda _context, result: result,
        )
    )
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert not verified.is_set()
    assert any("SafetyProof" in error for error in handle.snapshot().cleanup_errors)


def test_safety_proof_is_immutable_and_cannot_substitute_another_device():
    camera = camera_key("immutable-proof-camera")
    fpga = ResourceKey.parse("device/fpga/immutable-proof-fpga")
    camera_verified = threading.Event()
    fpga_verified = threading.Event()
    broker = DeviceBroker()
    camera_device = broker.bind(
        key=camera,
        identity=verified_identity(broker, camera, "camera-generation"),
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
        verify_safe_state=lambda: camera_verified.set() or SafeStateAck("camera-safe"),
    )
    fpga_device = broker.bind(
        key=fpga,
        identity=verified_identity(broker, fpga, "fpga-generation"),
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
        verify_safe_state=lambda: fpga_verified.set() or SafeStateAck("fpga-safe"),
    )

    def cleanup(context, _prepared, _primary):
        camera_device = context.cleanup_device(camera)
        fpga_device = context.cleanup_device(fpga)
        camera_device.perform(SafetyOperation.SAFE_STATE)
        fpga_device.perform(SafetyOperation.SAFE_STATE)
        camera_proof = camera_device.verify_safe_state()
        forged_receipt = SafeReceipt(
            key=fpga,
            stable_device_identity=str(fpga),
            connection_generation="fpga-generation",
            operation_id="VERIFY_SAFE_STATE",
            acknowledgement_digest="forged-by-cross-device-substitution",
        )
        with pytest.raises(AttributeError, match="immutable"):
            camera_proof.receipt = forged_receipt
        return CleanupReport.safe(
            (camera_proof, fpga_device.verify_safe_state())
        )

    run_plan = RunPlan(
        name="immutable safety proof",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(ResourceClaim(camera), ResourceClaim(fpga)),
        hazard_claims=(
            HazardClaim(camera, str(camera), camera_device.connection_generation),
            HazardClaim(fpga, str(fpga), fpga_device.connection_generation),
        ),
        bound_devices=(camera_device, fpga_device),
        preflight=lambda _context: None,
        execute=lambda _context, _prepared: "result",
        cleanup=cleanup,
        finalize=lambda _context, result: result,
    )

    assert RunController(new_arbiter()).run(run_plan) == "result"
    assert camera_verified.is_set()
    assert fpga_verified.is_set()


def test_cleanup_acknowledgement_must_match_requested_operation():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    key = camera_key("wrong-operation-ack")
    handle = controller.start(
        plan(
            key=key,
            execute=lambda *_: "result",
            cleanup_ack=lambda: safe_ack(SafetyOperation.DISARM),
        )
    )

    with pytest.raises(RunFailed):
        handle.result(2.0)

    assert any(
        "expected SAFE_STATE" in error
        for error in handle.snapshot().cleanup_errors
    )


def test_cleanup_phase_cannot_reenter_normal_device_execution():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    key = camera_key("cleanup-execution-gate")
    blocked = threading.Event()

    def cleanup(context, _prepared, _primary):
        with pytest.raises(CapabilityRevoked):
            context.device(key).execute("late-fire")
        blocked.set()
        return safe_cleanup_report(context, key)

    handle = controller.start(
        plan(key=key, execute=lambda *_: "result", cleanup=cleanup)
    )
    assert handle.result(2.0) == "result"
    assert blocked.is_set()


def test_run_plan_rejects_bound_device_generation_mismatch():
    key = camera_key("generation-mismatch")
    broker = DeviceBroker()
    with pytest.raises(ValueError, match="generation does not match"):
        RunPlan(
            name="generation mismatch",
            mode=RunMode.FINITE_EXACT,
            resource_claims=(ResourceClaim(key),),
            hazard_claims=(HazardClaim(key, str(key), "hazard-generation"),),
            bound_devices=(
                broker.bind(
                    key=key,
                    identity=verified_identity(broker, key, "binding-generation"),
                    execute_command=lambda command: command,
                    cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
                    verify_safe_state=lambda: SafeStateAck("safe-readback"),
                ),
            ),
            preflight=lambda _ctx: None,
            execute=lambda _ctx, _prepared: None,
            cleanup=lambda context, _prepared, _primary: safe_cleanup_report(
                context, key
            ),
            finalize=lambda _ctx, result: result,
        )


def test_interrupt_capability_must_be_declared_thread_safe_separately():
    key = camera_key("interrupt-capability")
    broker = DeviceBroker()
    device = broker.bind(
        key=key,
        identity=verified_identity(broker, key, "generation"),
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.ABORT: lambda: safe_ack(SafetyOperation.ABORT)},
        verify_safe_state=lambda: SafeStateAck("safe-readback"),
    )
    with pytest.raises(ValueError, match="thread-safe interrupt"):
        RunPlan(
            name="missing interrupt capability",
            mode=RunMode.FINITE_EXACT,
            resource_claims=(ResourceClaim(key),),
            hazard_claims=(
                HazardClaim(key, str(key), device.connection_generation),
            ),
            bound_devices=(device,),
            preflight=lambda _ctx: None,
            execute=lambda _ctx, _prepared: None,
            cleanup=lambda _ctx, _prepared, _primary: CleanupReport.unsafe(
                (key,),
                reason="not run",
                recovery_action="not run",
            ),
            finalize=lambda _ctx, result: result,
            interrupt_operations=(SafetyInterrupt(key, SafetyOperation.ABORT),),
        )


def test_multi_device_interrupt_attempts_every_device_after_one_failure():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    camera = camera_key("interrupt-all-camera")
    fpga = ResourceKey.parse("device/fpga/interrupt-all")
    execute_started = threading.Event()
    allow_execute_exit = threading.Event()
    fpga_interrupt_called = threading.Event()

    def camera_interrupt():
        raise OSError("camera abort failed")

    def fpga_interrupt():
        fpga_interrupt_called.set()
        allow_execute_exit.set()

    def execute(context, _prepared):
        execute_started.set()
        allow_execute_exit.wait()
        context.checkpoint()

    def cleanup(context, _prepared, _primary):
        devices = (context.cleanup_device(camera), context.cleanup_device(fpga))
        for device in devices:
            device.perform(SafetyOperation.SAFE_STATE)
        receipts = tuple(device.verify_safe_state() for device in devices)
        return CleanupReport.safe(receipts)

    broker = DeviceBroker()
    camera_device = broker.bind(
        key=camera,
        identity=verified_identity(broker, camera, "camera-generation"),
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
        verify_safe_state=lambda: SafeStateAck("camera-safe-readback"),
        interrupt_operations={SafetyOperation.ABORT: camera_interrupt},
    )
    fpga_device = broker.bind(
        key=fpga,
        identity=verified_identity(broker, fpga, "fpga-generation"),
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
        verify_safe_state=lambda: SafeStateAck("fpga-safe-readback"),
        interrupt_operations={SafetyOperation.ABORT: fpga_interrupt},
    )
    run_plan = RunPlan(
        name="all interrupts",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(ResourceClaim(camera), ResourceClaim(fpga)),
        hazard_claims=(
            HazardClaim(camera, str(camera), camera_device.connection_generation),
            HazardClaim(fpga, str(fpga), fpga_device.connection_generation),
        ),
        bound_devices=(camera_device, fpga_device),
        preflight=lambda _ctx: None,
        execute=execute,
        cleanup=cleanup,
        finalize=lambda _ctx, result: result,
        interrupt_operations=(
            SafetyInterrupt(camera, SafetyOperation.ABORT),
            SafetyInterrupt(fpga, SafetyOperation.ABORT),
        ),
    )
    handle = controller.start(run_plan)
    assert execute_started.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert fpga_interrupt_called.is_set()
    assert any(
        "camera abort failed" in error
        for error in handle.snapshot().cleanup_errors
    )


def test_finalize_receives_no_device_capability():
    controller = RunController(new_arbiter())
    observed = []

    def finalize(context: PostSafetyContext, result):
        observed.extend(
            (hasattr(context, "device"), hasattr(context, "cleanup_device"))
        )
        return result

    handle = controller.start(
        plan(
            key=camera_key("post-safety-context"),
            execute=lambda *_: "result",
            finalize=finalize,
        )
    )
    assert handle.result(2.0) == "result"
    assert observed == [False, False]


def test_checkpoint_is_durable_but_later_finalize_failure_keeps_run_failed():
    journal = MemoryCommitJournal("test-repository")

    def finalize(context: PostSafetyContext, _result):
        raw_ref = checkpoint_commit(
            context,
            lambda: "raw-capture-ref",
            journal=journal,
            commit_id="raw-before-analysis-failure",
        )
        assert raw_ref == "raw-capture-ref"
        snapshot = context._handle.snapshot()
        assert not snapshot.final_committed
        assert snapshot.committed_checkpoint is not None
        raise ArithmeticError("calibration failed after raw checkpoint")

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("checkpoint-analysis-failure"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    with pytest.raises(RunFailed, match="calibration failed"):
        handle.result(2.0)
    snapshot = handle.snapshot()
    assert snapshot.state is RunState.FAILED
    assert not snapshot.final_committed
    assert snapshot.committed_checkpoint.commit_id == "raw-before-analysis-failure"
    assert snapshot.committed_checkpoint.target.target_ref == (
        "artifacts/raw-before-analysis-failure"
    )
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED
    assert journal.pending() == ()


def test_checkpoint_does_not_satisfy_required_final_commit():
    def finalize(context: PostSafetyContext, _result):
        return checkpoint_commit(
            context,
            lambda: "raw-only-ref",
            commit_id="raw-without-final",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("checkpoint-is-not-final"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    with pytest.raises(RunFailed, match="requires a final artifact commit"):
        handle.result(2.0)
    snapshot = handle.snapshot()
    assert snapshot.state is RunState.FAILED
    assert not snapshot.final_committed
    assert snapshot.committed_checkpoint is not None
    assert snapshot.committed_checkpoint.target.target_ref == "artifacts/raw-without-final"
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED


def test_checkpoint_then_final_commit_has_two_distinct_authority_boundaries():
    def finalize(context: PostSafetyContext, _result):
        raw = checkpoint_commit(
            context,
            lambda: "raw-ref",
            commit_id="raw-then-final",
        )
        assert raw == "raw-ref"
        context.checkpoint()
        return commit(
            context,
            lambda: "calibration-ref",
            commit_id="calibration-final",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("checkpoint-then-final"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert handle.result(2.0) == "calibration-ref"
    snapshot = handle.snapshot()
    assert snapshot.final_committed
    assert snapshot.committed_checkpoint.commit_id == "raw-then-final"
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED


def test_cancel_before_checkpoint_publish_aborts_intent_and_never_publishes():
    controller = RunController(new_arbiter())
    intent_started = threading.Event()
    allow_intent = threading.Event()
    published = threading.Event()

    class BlockingCheckpointIntentJournal(MemoryCommitJournal):
        def begin(self, value):
            intent_started.set()
            allow_intent.wait()
            super().begin(value)

    journal = BlockingCheckpointIntentJournal("test-repository")

    def finalize(context: PostSafetyContext, _result):
        return checkpoint_commit(
            context,
            lambda: published.set() or "raw-ref",
            journal=journal,
            commit_id="cancel-before-checkpoint-publish",
        )

    handle = controller.start(
        plan(
            key=camera_key("cancel-before-checkpoint-publish"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    assert intent_started.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    allow_intent.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert not published.is_set()
    assert journal.pending() == ()
    assert handle.snapshot().committed_checkpoint is None


def test_cancel_before_checkpoint_call_writes_no_intent_and_discards_authority():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    operation_ready = threading.Event()
    try_checkpoint = threading.Event()

    def finalize(context: PostSafetyContext, _result):
        target = CommitTarget(
            "test-repository",
            "test-checkpoint",
            "1",
            "artifacts/cancel-before-checkpoint-call",
            "0" * 64,
        )
        operation = CheckpointCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.CHECKPOINT,
                "cancel-before-checkpoint-call",
                target,
                lambda: PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    "must-not-publish",
                ),
            ),
        )
        operation_ready.set()
        try_checkpoint.wait()
        return context.commit_checkpoint(operation)

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("cancel-before-checkpoint-call"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    assert operation_ready.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    try_checkpoint.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert journal._intents == {}
    assert coordinator._authorities == {}
    assert handle.snapshot().committed_checkpoint is None


def test_cancel_during_checkpoint_publish_preserves_raw_but_blocks_analysis():
    controller = RunController(new_arbiter())
    publish_started = threading.Event()
    allow_publish = threading.Event()
    analysis_started = threading.Event()

    def publish():
        publish_started.set()
        allow_publish.wait()
        return "raw-ref"

    def finalize(context: PostSafetyContext, _result):
        checkpoint_commit(
            context,
            publish,
            commit_id="cancel-during-checkpoint-publish",
        )
        analysis_started.set()
        return "should-not-return"

    handle = controller.start(
        plan(
            key=camera_key("cancel-during-checkpoint-publish"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    assert publish_started.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    allow_publish.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert not analysis_started.is_set()
    snapshot = handle.snapshot()
    assert not snapshot.final_committed
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED


def test_checkpoint_lost_ack_is_synchronously_recovered_before_analysis():
    visible = []
    analysis_saw_checkpoint = []

    def publish():
        visible.append("raw-ref")
        raise PublishVisibilityUnknown("checkpoint fsync acknowledgement lost")

    def recover(intent):
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=intent.target.target_ref,
                manifest_digest=intent.target.expected_manifest_digest,
                result=visible[-1],
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        raw = checkpoint_commit(
            context,
            publish,
            recover,
            commit_id="checkpoint-sync-recovery",
        )
        analysis_saw_checkpoint.append(context._handle.snapshot().committed_checkpoint)
        raise RuntimeError(f"analysis failed after {raw}")

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("checkpoint-sync-recovery"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    with pytest.raises(RunFailed, match="analysis failed"):
        handle.result(2.0)
    assert analysis_saw_checkpoint[0].disposition is CheckpointDisposition.COMMITTED
    assert "fsync acknowledgement lost" in handle.snapshot().commit_recovery_warning


def test_checkpoint_visibility_absent_is_durably_aborted_and_run_fails():
    journal = MemoryCommitJournal("test-repository")

    def finalize(context: PostSafetyContext, _result):
        return checkpoint_commit(
            context,
            lambda: (_ for _ in ()).throw(PublishVisibilityUnknown("raw replace lost")),
            lambda _intent: CommitRecovery(committed=False),
            journal=journal,
            commit_id="checkpoint-definitively-absent",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("checkpoint-definitively-absent"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    with pytest.raises(RunFailed, match="raw replace lost"):
        handle.result(2.0)
    assert journal.pending() == ()
    assert handle.snapshot().committed_checkpoint is None


def test_unknown_checkpoint_holds_claim_then_retry_commits_without_reentering_finalize():
    arbiter = new_arbiter()
    journal = MemoryCommitJournal("test-repository")
    finalize_calls = []
    analysis_calls = []
    recover_calls = 0
    second_recovery_failed = threading.Event()

    def recover(intent):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls <= 2:
            if recover_calls == 2:
                second_recovery_failed.set()
            raise OSError("checkpoint repository unavailable")
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=intent.target.target_ref,
                manifest_digest=intent.target.expected_manifest_digest,
                result="recovered-raw-ref",
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        finalize_calls.append("entered")
        raw = checkpoint_commit(
            context,
            lambda: (_ for _ in ()).throw(
                PublishVisibilityUnknown("checkpoint visibility unknown")
            ),
            recover,
            journal=journal,
            commit_id="unknown-checkpoint",
        )
        analysis_calls.append(raw)
        return raw

    key = camera_key("unknown-checkpoint")
    handle = RunController(arbiter).start(
        plan(key=key, execute=lambda *_: "staged", finalize=finalize)
    )
    snapshot = handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert snapshot.state is RunState.RUNNING
    assert snapshot.committed_checkpoint.disposition is (
        CheckpointDisposition.RECONCILIATION_REQUIRED
    )
    assert not hasattr(snapshot.committed_checkpoint, "ref")
    assert len(arbiter.active_claims()) == 1
    assert tuple(value.commit_id for value in journal.pending()) == (
        "unknown-checkpoint",
    )
    assert handle.retry_recovery()
    assert second_recovery_failed.wait(1.0)
    handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert len(arbiter.active_claims()) == 1
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="restart analysis"):
        handle.result(2.0)
    snapshot = handle.snapshot()
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED
    assert snapshot.committed_checkpoint.target.target_ref == "artifacts/unknown-checkpoint"
    assert finalize_calls == ["entered"]
    assert analysis_calls == []
    assert recover_calls == 3
    assert journal.pending() == ()
    assert not arbiter.active_claims()


def test_unknown_checkpoint_retry_absent_fails_and_clears_checkpoint_record():
    journal = MemoryCommitJournal("test-repository")
    recover_calls = 0

    def recover(_intent):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls == 1:
            raise OSError("repository unavailable")
        return CommitRecovery(committed=False)

    def finalize(context: PostSafetyContext, _result):
        return checkpoint_commit(
            context,
            lambda: (_ for _ in ()).throw(PublishVisibilityUnknown("unknown raw")),
            recover,
            journal=journal,
            commit_id="absent-checkpoint-retry",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("absent-checkpoint-retry"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    pending_snapshot = handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert not hasattr(pending_snapshot.committed_checkpoint, "ref")
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="unknown raw"):
        handle.result(2.0)
    assert handle.snapshot().committed_checkpoint is None
    assert journal.pending() == ()


def test_cancel_while_checkpoint_reconciliation_pending_ends_cancelled_with_raw_ref():
    journal = MemoryCommitJournal("test-repository")
    recover_calls = 0

    def recover(intent):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls == 1:
            raise OSError("repository unavailable")
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=intent.target.target_ref,
                manifest_digest=intent.target.expected_manifest_digest,
                result="raw-ref",
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        return checkpoint_commit(
            context,
            lambda: (_ for _ in ()).throw(PublishVisibilityUnknown("raw unknown")),
            recover,
            journal=journal,
            commit_id="cancel-pending-checkpoint",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("cancel-pending-checkpoint"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert handle.cancel() is CancelOutcome.REQUESTED
    assert handle.retry_recovery()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    snapshot = handle.snapshot()
    assert snapshot.committed_checkpoint.disposition is CheckpointDisposition.COMMITTED
    assert snapshot.committed_checkpoint.target.target_ref == (
        "artifacts/cancel-pending-checkpoint"
    )


def test_checkpoint_commit_marker_retry_fails_original_run_without_recovery_publish():
    journal = FailFirstCommitMarkerJournal(fail_commit=True)
    recover_calls = []
    finalize_calls = []

    def finalize(context: PostSafetyContext, _result):
        finalize_calls.append("entered")
        return checkpoint_commit(
            context,
            lambda: "raw-ref",
            lambda intent: recover_calls.append(intent) or CommitRecovery(committed=False),
            journal=journal,
            commit_id="checkpoint-marker-retry",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("checkpoint-marker-retry"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    pending_snapshot = handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert pending_snapshot.committed_checkpoint.target.target_ref == (
        "artifacts/checkpoint-marker-retry"
    )
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="restart analysis"):
        handle.result(2.0)
    assert recover_calls == []
    assert finalize_calls == ["entered"]
    assert handle.snapshot().committed_checkpoint.disposition is (
        CheckpointDisposition.COMMITTED
    )


def test_second_checkpoint_is_rejected_and_its_authority_is_discarded():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    published = []

    def operation(context, suffix):
        target = CommitTarget(
            "test-repository",
            "test-checkpoint",
            "1",
            f"artifacts/checkpoint-{suffix}",
            "0" * 64,
        )
        return CheckpointCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.CHECKPOINT,
                f"checkpoint-{suffix}",
                target,
                lambda: PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    published.append(suffix) or target.target_ref,
                ),
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        context.commit_checkpoint(operation(context, "first"))
        context.commit_checkpoint(operation(context, "second"))
        return "unreachable"

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("two-checkpoints"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    with pytest.raises(RunFailed, match="only once"):
        handle.result(2.0)
    assert published == ["first"]
    assert coordinator._authorities == {}
    assert handle.snapshot().committed_checkpoint.commit_id == "checkpoint-first"


def test_checkpoint_wrong_safety_bundle_discards_authority_before_intent():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )

    def finalize(context: PostSafetyContext, _result):
        target = CommitTarget(
            "test-repository",
            "test-checkpoint",
            "1",
            "artifacts/wrong-checkpoint-safety",
            "0" * 64,
        )
        return context.commit_checkpoint(
            CheckpointCommit(
                prepared_authority(
                    context,
                    coordinator,
                    CommitKind.CHECKPOINT,
                    "wrong-checkpoint-safety",
                    target,
                    lambda: PublishedManifest(
                        target.target_ref,
                        target.expected_manifest_digest,
                        "raw-ref",
                    ),
                    subject=CommitSubject(
                        context.run_id.value,
                        "wrong-safety-bundle",
                    ),
                ),
            )
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("wrong-checkpoint-safety"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    with pytest.raises(RunFailed, match="safety bundle"):
        handle.result(2.0)
    assert coordinator._authorities == {}
    assert journal._intents == {}


def test_cancelled_final_commit_rejection_discards_unconsumed_authority():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    operation_ready = threading.Event()
    try_commit = threading.Event()

    def finalize(context: PostSafetyContext, _result):
        target = CommitTarget(
            "test-repository",
            "final",
            "1",
            "artifacts/cancelled-final-authority",
            "0" * 64,
        )
        operation = FinalCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.FINAL,
                "cancelled-final-authority",
                target,
                lambda: PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    "must-not-publish",
                ),
            ),
        )
        operation_ready.set()
        try_commit.wait()
        return context.commit_final(operation)

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("cancelled-final-authority"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert operation_ready.wait(1.0)
    assert handle.cancel() is CancelOutcome.REQUESTED
    try_commit.set()
    with pytest.raises(RunCancelled):
        handle.result(2.0)
    assert coordinator._authorities == {}
    assert journal.pending() == ()


def test_nonowner_inflight_rejection_preserves_authority_for_owner_gate():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    second_errors = []

    def finalize(context: PostSafetyContext, _result):
        targets = tuple(
            CommitTarget(
                "test-repository",
                "final",
                "1",
                f"artifacts/inflight-final-{index}",
                "0" * 64,
            )
            for index in (1, 2)
        )
        second_operation = FinalCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.FINAL,
                "inflight-final-2",
                targets[1],
                lambda: PublishedManifest(
                    targets[1].target_ref,
                    targets[1].expected_manifest_digest,
                    "second",
                ),
            ),
        )

        def publish_first():
            thread = threading.Thread(
                target=lambda: _call_second_final(
                    context,
                    second_operation,
                    second_errors,
                )
            )
            thread.start()
            thread.join(1.0)
            assert not thread.is_alive()
            return PublishedManifest(
                targets[0].target_ref,
                targets[0].expected_manifest_digest,
                "first",
            )

        first_operation = FinalCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.FINAL,
                "inflight-final-1",
                targets[0],
                publish_first,
            ),
        )
        result = context.commit_final(first_operation)
        assert len(coordinator._authorities) == 1
        with pytest.raises(RuntimeError, match="only once"):
            context.commit_final(second_operation)
        return result

    def _call_second_final(context, operation, errors):
        try:
            context.commit_final(operation)
        except BaseException as error:
            errors.append(error)

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("inflight-final-authority"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert handle.result(2.0) == "first"
    assert len(second_errors) == 1
    assert "authoritative Run owner thread" in str(second_errors[0])
    assert coordinator._authorities == {}


def test_final_already_committed_rejection_discards_second_authority():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )

    def operation(context, index):
        target = CommitTarget(
            "test-repository",
            "final",
            "1",
            f"artifacts/already-final-{index}",
            "0" * 64,
        )
        return FinalCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.FINAL,
                f"already-final-{index}",
                target,
                lambda: PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    f"final-{index}",
                ),
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        result = context.commit_final(operation(context, 1))
        with pytest.raises(RuntimeError, match="only once"):
            context.commit_final(operation(context, 2))
        return result

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("already-final-authority"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert handle.result(2.0) == "final-1"
    assert coordinator._authorities == {}


def test_checkpoint_snapshot_contains_only_durable_repository_fact():
    assert {field.name for field in fields(CheckpointSnapshot)} == {
        "commit_id",
        "target",
        "disposition",
    }


@pytest.mark.parametrize("kind", ("checkpoint", "final"))
def test_running_nonowner_rejection_preserves_same_operation_for_owner(kind):
    journal = MemoryCommitJournal("test-repository")
    published = threading.Event()
    errors = []
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )

    def finalize(context: PostSafetyContext, result):
        target = CommitTarget(
            "test-repository",
            kind,
            "1",
            f"artifacts/nonowner-first-{kind}",
            "0" * 64,
        )
        operation_type = CheckpointCommit if kind == "checkpoint" else FinalCommit
        operation_kind = (
            CommitKind.CHECKPOINT if kind == "checkpoint" else CommitKind.FINAL
        )
        operation = operation_type(
            prepared_authority(
                context,
                coordinator,
                operation_kind,
                f"nonowner-first-{kind}",
                target,
                lambda: published.set()
                or PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    "owner-ref",
                ),
            ),
        )
        commit_operation = (
            context.commit_checkpoint if kind == "checkpoint" else context.commit_final
        )

        def escaped_commit():
            try:
                commit_operation(operation)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=escaped_commit)
        thread.start()
        thread.join(1.0)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], CapabilityRevoked)
        assert not published.is_set()
        assert journal.pending() == ()
        assert len(coordinator._authorities) == 1
        return commit_operation(operation)

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key(f"nonowner-first-{kind}"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=kind == "final",
        )
    )
    assert handle.result(2.0) == "owner-ref"
    assert published.is_set()
    assert journal.pending() == ()
    assert coordinator._authorities == {}
    assert handle.snapshot().final_committed is (kind == "final")
    assert (handle.snapshot().committed_checkpoint is not None) is (kind == "checkpoint")


def test_escaped_final_context_cannot_mutate_terminal_run_and_burns_authority():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    escaped = {}
    published = threading.Event()

    def finalize(context: PostSafetyContext, result):
        target = CommitTarget(
            "test-repository",
            "final",
            "1",
            "artifacts/terminal-escape",
            "0" * 64,
        )
        escaped["context"] = context
        escaped["operation"] = FinalCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.FINAL,
                "terminal-escape",
                target,
                lambda: published.set()
                or PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    "late-ref",
                ),
            ),
        )
        return result

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("terminal-escape"),
            execute=lambda *_: "terminal-result",
            finalize=finalize,
        )
    )
    assert handle.result(2.0) == "terminal-result"
    before = handle.snapshot()
    with pytest.raises(CapabilityRevoked, match="terminal"):
        escaped["context"].commit_final(escaped["operation"])
    assert handle.snapshot() == before
    assert not published.is_set()
    assert journal.pending() == ()
    assert coordinator._authorities == {}


def test_same_final_operation_loser_cannot_consume_authority_from_owner_winner():
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    loser_errors = []

    def finalize(context: PostSafetyContext, _result):
        target = CommitTarget(
            "test-repository",
            "final",
            "1",
            "artifacts/same-operation-race",
            "0" * 64,
        )
        operation_box = {}

        def publish_manifest():
            def loser():
                try:
                    context.commit_final(operation_box["operation"])
                except BaseException as error:
                    loser_errors.append(error)

            thread = threading.Thread(target=loser)
            thread.start()
            thread.join(1.0)
            assert not thread.is_alive()
            return PublishedManifest(
                target.target_ref,
                target.expected_manifest_digest,
                "winner-ref",
            )

        operation_box["operation"] = FinalCommit(
            prepared_authority(
                context,
                coordinator,
                CommitKind.FINAL,
                "same-operation-race",
                target,
                publish_manifest,
            ),
        )
        return context.commit_final(operation_box["operation"])

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("same-operation-race"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert handle.result(2.0) == "winner-ref"
    assert len(loser_errors) == 1
    assert isinstance(loser_errors[0], CapabilityRevoked)
    assert coordinator._authorities == {}


def test_swallowed_checkpoint_reconciliation_return_cannot_publish_success():
    recover_calls = 0

    def recover(intent):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls == 1:
            raise OSError("checkpoint fact unavailable")
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                intent.target.target_ref,
                intent.target.expected_manifest_digest,
                "raw-ref",
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        try:
            checkpoint_commit(
                context,
                lambda: (_ for _ in ()).throw(PublishVisibilityUnknown("raw unknown")),
                recover,
                commit_id="swallowed-checkpoint-return",
            )
        except CheckpointReconciliationRequired:
            return "pretend-success"
        raise AssertionError("checkpoint reconciliation should have interrupted finalize")

    arbiter = new_arbiter()
    handle = RunController(arbiter).start(
        plan(
            key=camera_key("swallowed-checkpoint-return"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    snapshot = handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert not snapshot.state.terminal
    assert len(arbiter.active_claims()) == 1
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="restart analysis"):
        handle.result(2.0)


def test_swallowed_final_reconciliation_return_waits_for_fact_before_success():
    recover_calls = 0

    def recover(intent):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls == 1:
            raise OSError("final fact unavailable")
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                intent.target.target_ref,
                intent.target.expected_manifest_digest,
                "final-ref",
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        try:
            commit(
                context,
                lambda: (_ for _ in ()).throw(PublishVisibilityUnknown("final unknown")),
                recover,
                commit_id="swallowed-final-return",
            )
        except FinalCommitReconciliationRequired:
            return "pretend-success"
        raise AssertionError("final reconciliation should have interrupted finalize")

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("swallowed-final-return"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    snapshot = handle.wait_for(
        lambda value: value.phase == "final-commit-reconciliation-failed",
        2.0,
    )
    assert not snapshot.state.terminal
    assert handle.retry_recovery()
    assert handle.result(2.0) == "final-ref"


def test_swallowed_checkpoint_then_finalize_error_reconciles_before_failure():
    recover_calls = 0

    def recover(intent):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls == 1:
            raise OSError("checkpoint fact unavailable")
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                intent.target.target_ref,
                intent.target.expected_manifest_digest,
                "raw-ref",
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        try:
            checkpoint_commit(
                context,
                lambda: (_ for _ in ()).throw(PublishVisibilityUnknown("raw unknown")),
                recover,
                commit_id="swallowed-checkpoint-error",
            )
        except CheckpointReconciliationRequired:
            raise ArithmeticError("analysis failed after swallowing checkpoint")

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("swallowed-checkpoint-error"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="analysis failed after swallowing"):
        handle.result(2.0)


def test_pending_checkpoint_reconciliation_rejects_and_discards_final_commit():
    checkpoint_recover_calls = 0
    final_published = threading.Event()
    final_journal = MemoryCommitJournal("test-repository")
    final_coordinator = RepositoryCommitCoordinator(
        final_journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )

    def recover(intent):
        nonlocal checkpoint_recover_calls
        checkpoint_recover_calls += 1
        if checkpoint_recover_calls == 1:
            raise OSError("checkpoint fact unavailable")
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                intent.target.target_ref,
                intent.target.expected_manifest_digest,
                "raw-ref",
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        try:
            checkpoint_commit(
                context,
                lambda: (_ for _ in ()).throw(PublishVisibilityUnknown("raw unknown")),
                recover,
                commit_id="checkpoint-blocks-final",
            )
        except CheckpointReconciliationRequired:
            target = CommitTarget(
                "test-repository",
                "final",
                "1",
                "artifacts/rejected-final-after-checkpoint",
                "0" * 64,
            )
            with pytest.raises(RuntimeError, match="reconciliation is pending"):
                prepared_authority(
                    context,
                    final_coordinator,
                    CommitKind.FINAL,
                    "rejected-final-after-checkpoint",
                    target,
                    lambda: final_published.set()
                    or PublishedManifest(
                        target.target_ref,
                        target.expected_manifest_digest,
                        "final-ref",
                    ),
                )
            return "swallowed"

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("checkpoint-blocks-final"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    handle.wait_for(
        lambda value: value.phase == "checkpoint-reconciliation-failed",
        2.0,
    )
    assert not final_published.is_set()
    assert final_journal.pending() == ()
    assert final_coordinator._authorities == {}
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="restart analysis"):
        handle.result(2.0)


def test_conflicting_existing_intent_is_never_aborted_by_begin_failure():
    class TrackingJournal(MemoryCommitJournal):
        def __init__(self):
            super().__init__("test-repository")
            self.aborted_ids = []

        def mark_aborted(self, commit_id):
            self.aborted_ids.append(commit_id)
            super().mark_aborted(commit_id)

    journal = TrackingJournal()
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    old_target = CommitTarget(
        "test-repository",
        "old-kind",
        "1",
        "artifacts/old-owner",
        "1" * 64,
    )
    old_intent = CommitIntent(
        CommitKind.CHECKPOINT,
        "conflicting-intent-id",
        "old-run",
        "old-safety",
        old_target,
        1.0,
    )
    journal.begin(old_intent)

    def finalize(context: PostSafetyContext, _result):
        new_target = CommitTarget(
            "test-repository",
            "new-kind",
            "1",
            "artifacts/new-owner",
            "2" * 64,
        )
        return context.commit_checkpoint(
            CheckpointCommit(
                prepared_authority(
                    context,
                    coordinator,
                    CommitKind.CHECKPOINT,
                    "conflicting-intent-id",
                    new_target,
                    lambda: PublishedManifest(
                        new_target.target_ref,
                        new_target.expected_manifest_digest,
                        "new-ref",
                    ),
                ),
            )
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("conflicting-intent"),
            execute=lambda *_: "staged",
            finalize=finalize,
        )
    )
    with pytest.raises(RunFailed, match="conflicting intent"):
        handle.result(2.0)
    assert journal.pending() == (old_intent,)
    assert journal.aborted_ids == []


@pytest.mark.parametrize("kind", ("checkpoint", "final"))
def test_admit_rechecks_deadline_after_outer_validation_before_any_io(kind):
    journal = MemoryCommitJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    published = threading.Event()

    def finalize(context: PostSafetyContext, _result):
        target = CommitTarget(
            "test-repository",
            kind,
            "1",
            f"artifacts/deadline-toctou-{kind}",
            "0" * 64,
        )
        operation_type = CheckpointCommit if kind == "checkpoint" else FinalCommit
        operation_kind = (
            CommitKind.CHECKPOINT if kind == "checkpoint" else CommitKind.FINAL
        )
        operation = operation_type(
            prepared_authority(
                context,
                coordinator,
                operation_kind,
                f"deadline-toctou-{kind}",
                target,
                lambda: published.set()
                or PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    "raw-ref",
                ),
            ),
        )
        context._validate_commit_entry(operation)
        assert context.deadline is not None
        while time.monotonic() < context.deadline:
            time.sleep(0.001)
        if kind == "checkpoint":
            return context._handle._commit_checkpoint(
                operation,
                deadline=context.deadline,
            )
        return context._handle._commit_final(operation, deadline=context.deadline)

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key(f"deadline-toctou-{kind}"),
            execute=lambda *_: "staged",
            finalize=finalize,
            timeout_seconds=0.25,
        )
    )
    with pytest.raises(RunFailed, match="monotonic deadline"):
        handle.result(2.0)
    assert not published.is_set()
    assert journal.pending() == ()
    assert coordinator._authorities == {}


@pytest.mark.parametrize("kind", ("checkpoint", "final"))
def test_durable_commit_rechecks_deadline_after_journal_begin_before_publish(kind):
    deadline_box = {}

    class BeginAcrossDeadlineJournal(MemoryCommitJournal):
        def begin(self, intent):
            super().begin(intent)
            deadline = deadline_box["deadline"]
            while time.monotonic() <= deadline:
                time.sleep(0.001)

    journal = BeginAcrossDeadlineJournal("test-repository")
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    published = threading.Event()
    commit_id = f"deadline-after-journal-begin-{kind}"

    def finalize(context: PostSafetyContext, _result):
        assert context.deadline is not None
        deadline_box["deadline"] = context.deadline
        target = CommitTarget(
            "test-repository",
            kind,
            "1",
            f"artifacts/{commit_id}",
            "0" * 64,
        )
        operation_type = CheckpointCommit if kind == "checkpoint" else FinalCommit
        operation_kind = (
            CommitKind.CHECKPOINT if kind == "checkpoint" else CommitKind.FINAL
        )
        operation = operation_type(
            prepared_authority(
                context,
                coordinator,
                operation_kind,
                commit_id,
                target,
                lambda: published.set()
                or PublishedManifest(
                    target.target_ref,
                    target.expected_manifest_digest,
                    "raw-ref",
                ),
            )
        )
        if kind == "checkpoint":
            return context.commit_checkpoint(operation)
        return context.commit_final(operation)

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key(f"deadline-after-journal-begin-{kind}"),
            execute=lambda *_: "staged",
            finalize=finalize,
            timeout_seconds=0.25,
            requires_final_commit=kind == "final",
        )
    )
    with pytest.raises(RunFailed, match="monotonic deadline"):
        handle.result(2.0)
    assert not published.is_set()
    assert journal.pending() == ()
    assert set(journal._intents) == {commit_id}
    assert journal._aborted == {commit_id}
    assert coordinator._authorities == {}


def test_final_commit_recovers_visible_manifest_after_lost_acknowledgement():
    controller = RunController(new_arbiter())
    visible_manifest = []

    def publish():
        visible_manifest.append("artifact-ref")
        raise PublishVisibilityUnknown("directory fsync acknowledgement lost")

    def recover(_intent):
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref="artifacts/lost-acknowledgement",
                manifest_digest="0" * 64,
                result=visible_manifest[-1],
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        return commit(
            context,
            publish,
            recover,
            commit_id="lost-acknowledgement",
        )

    handle = controller.start(
        plan(
            key=camera_key("commit-recovery"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    assert handle.result(2.0) == "artifact-ref"
    snapshot = handle.snapshot()
    assert snapshot.final_committed
    assert "fsync acknowledgement lost" in snapshot.commit_recovery_warning
    assert visible_manifest == ["artifact-ref"]


def test_publish_digest_must_match_repository_commit_target():
    controller = RunController(new_arbiter())
    journal = MemoryCommitJournal("test-repository")
    recovery_calls = []
    target = CommitTarget(
        repository_id="test-repository",
        artifact_kind="test-artifact",
        schema_version="1",
        target_ref="artifacts/digest-mismatch",
        expected_manifest_digest="0" * 64,
    )
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda value: recovery_calls.append(value)
        or CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=value.target.target_ref,
                manifest_digest=value.target.expected_manifest_digest,
                result="laundered-artifact-ref",
            ),
        ),
        allow_ephemeral=True,
    )

    def finalize(context: PostSafetyContext, _result):
        return context.commit_final(
            FinalCommit(
                prepared_authority(
                    context,
                    coordinator,
                    CommitKind.FINAL,
                    "digest-mismatch",
                    target,
                    lambda: PublishedManifest(
                        target_ref=target.target_ref,
                        manifest_digest="1" * 64,
                        result="wrong-artifact-ref",
                    ),
                ),
            )
        )

    handle = controller.start(
        plan(
            key=camera_key("digest-mismatch"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    with pytest.raises(RunFailed, match="digest differs"):
        handle.result(2.0)
    assert not handle.snapshot().final_committed
    assert journal.pending() == ()
    assert recovery_calls == []


def test_wrong_digest_remains_force_abort_after_abort_marker_failure():
    journal = FailFirstCommitMarkerJournal(fail_abort=True)
    target = CommitTarget(
        repository_id="test-repository",
        artifact_kind="test-artifact",
        schema_version="1",
        target_ref="artifacts/wrong-digest-marker-failure",
        expected_manifest_digest="0" * 64,
    )
    recovery_calls = []
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda value: recovery_calls.append(value)
        or CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref=value.target.target_ref,
                manifest_digest=value.target.expected_manifest_digest,
                result="laundered",
            ),
        ),
        allow_ephemeral=True,
    )

    def finalize(context: PostSafetyContext, _result):
        return context.commit_final(
            FinalCommit(
                prepared_authority(
                    context,
                    coordinator,
                    CommitKind.FINAL,
                    "wrong-digest-marker-failure",
                    target,
                    lambda: PublishedManifest(
                        target_ref=target.target_ref,
                        manifest_digest="1" * 64,
                        result="wrong",
                    ),
                ),
            )
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("wrong-digest-marker-failure"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    handle.wait_for(
        lambda value: value.phase == "final-commit-reconciliation-failed",
        2.0,
    )
    assert recovery_calls == []
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="digest differs"):
        handle.result(2.0)
    assert recovery_calls == []
    assert journal.pending() == ()


def test_uncommitted_visibility_resolution_remains_force_abort_after_marker_failure():
    journal = FailFirstCommitMarkerJournal(fail_abort=True)
    recovery_calls = []

    def publish():
        raise PublishVisibilityUnknown("replace acknowledgement lost")

    def recover(value):
        recovery_calls.append(value)
        return CommitRecovery(committed=False)

    def finalize(context: PostSafetyContext, _result):
        return commit(
            context,
            publish,
            recover,
            journal=journal,
            commit_id="uncommitted-marker-failure",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("uncommitted-marker-failure"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    handle.wait_for(
        lambda value: value.phase == "final-commit-reconciliation-failed",
        2.0,
    )
    assert len(recovery_calls) == 1
    assert handle.retry_recovery()
    with pytest.raises(RunFailed, match="replace acknowledgement lost"):
        handle.result(2.0)
    assert len(recovery_calls) == 1
    assert journal.pending() == ()


def test_validated_publish_remains_force_commit_after_commit_marker_failure():
    journal = FailFirstCommitMarkerJournal(fail_commit=True)
    recovery_calls = []

    def recover(value):
        recovery_calls.append(value)
        return CommitRecovery(committed=False)

    def finalize(context: PostSafetyContext, _result):
        return commit(
            context,
            lambda: "artifact-ref",
            recover,
            journal=journal,
            commit_id="committed-marker-failure",
        )

    handle = RunController(new_arbiter()).start(
        plan(
            key=camera_key("committed-marker-failure"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    handle.wait_for(
        lambda value: value.phase == "final-commit-reconciliation-failed",
        2.0,
    )
    assert recovery_calls == []
    assert handle.retry_recovery()
    assert handle.result(2.0) == "artifact-ref"
    assert recovery_calls == []
    assert journal.pending() == ()


def test_unknown_final_commit_visibility_holds_claim_until_retry_reconciles():
    arbiter = new_arbiter()
    controller = RunController(arbiter)
    journal = MemoryCommitJournal("test-repository")
    recover_calls = 0

    def publish():
        raise PublishVisibilityUnknown("replace acknowledgement lost")

    def recover(_intent):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls == 1:
            raise OSError("repository temporarily unavailable")
        return CommitRecovery(
            committed=True,
            result=PublishedManifest(
                target_ref="artifacts/unknown-visible-commit",
                manifest_digest="0" * 64,
                result="artifact-ref",
            ),
        )

    def finalize(context: PostSafetyContext, _result):
        return commit(
            context,
            publish,
            recover,
            journal=journal,
            commit_id="unknown-visible-commit",
        )

    handle = controller.start(
        plan(
            key=camera_key("unknown-visible"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    snapshot = handle.wait_for(
        lambda value: value.phase == "final-commit-reconciliation-failed",
        2.0,
    )
    assert snapshot.state is RunState.RUNNING
    assert len(arbiter.active_claims()) == 1
    assert tuple(value.commit_id for value in journal.pending()) == (
        "unknown-visible-commit",
    )

    assert handle.retry_recovery()
    assert handle.result(2.0) == "artifact-ref"
    assert journal.pending() == ()
    assert not arbiter.active_claims()
    assert "replace acknowledgement lost" in handle.snapshot().commit_recovery_warning


def test_definitively_uncommitted_publish_is_aborted_and_run_fails():
    controller = RunController(new_arbiter())
    journal = MemoryCommitJournal("test-repository")

    def publish():
        raise OSError("publish failed")

    def finalize(context: PostSafetyContext, _result):
        return commit(
            context,
            publish,
            lambda _intent: CommitRecovery(committed=False),
            journal=journal,
            commit_id="definitively-aborted-commit",
        )

    handle = controller.start(
        plan(
            key=camera_key("definitively-aborted"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    with pytest.raises(RunFailed, match="publish failed"):
        handle.result(2.0)
    assert journal.pending() == ()
    assert not handle.snapshot().final_committed


def test_required_final_commit_cannot_be_bypassed_by_plain_finalize_result():
    controller = RunController(new_arbiter())
    handle = controller.start(
        plan(
            key=camera_key("missing-final-commit"),
            execute=lambda *_: "staged",
            finalize=lambda _context, _result: "uncommitted-ref",
            requires_final_commit=True,
        )
    )
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert "requires a final artifact commit" in handle.snapshot().primary_error


def test_commit_authority_is_single_use_and_bound_to_its_minting_run():
    journal = MemoryCommitJournal("test-repository")
    target = CommitTarget(
        repository_id="test-repository",
        artifact_kind="test-artifact",
        schema_version="1",
        target_ref="artifacts/single-use-authority",
        expected_manifest_digest="0" * 64,
    )
    publish_calls = []
    coordinator = RepositoryCommitCoordinator(
        journal,
        lambda _intent: CommitRecovery(committed=False),
        allow_ephemeral=True,
    )
    operation_box = {}

    def finalize(context: PostSafetyContext, _result):
        if "operation" not in operation_box:
            operation_box["operation"] = FinalCommit(
                prepared_authority(
                    context,
                    coordinator,
                    CommitKind.FINAL,
                    "single-use-authority",
                    target,
                    lambda: PublishedManifest(
                        target_ref=target.target_ref,
                        manifest_digest=target.expected_manifest_digest,
                        result=publish_calls.append("published") or "artifact-ref",
                    ),
                )
            )
        return context.commit_final(operation_box["operation"])

    def run_plan():
        return RunPlan(
            name="single use commit authority",
            mode=RunMode.FINITE_EXACT,
            resource_claims=(),
            hazard_claims=(),
            bound_devices=(),
            preflight=lambda _context: None,
            execute=lambda _context, _prepared: "staged",
            cleanup=lambda _context, _prepared, _primary: CleanupReport(),
            finalize=finalize,
            requires_final_commit=True,
        )

    controller = RunController(new_arbiter())
    assert controller.run(run_plan()) == "artifact-ref"
    with pytest.raises(RunFailed, match="another Run"):
        controller.run(run_plan())
    assert publish_calls == ["published"]


def test_final_commit_must_reference_this_runs_safety_bundle():
    controller = RunController(new_arbiter())

    def finalize(context: PostSafetyContext, _result):
        journal = MemoryCommitJournal("test-repository")
        target = CommitTarget(
            repository_id="test-repository",
            artifact_kind="test-artifact",
            schema_version="1",
            target_ref="artifacts/wrong-safety-bundle-commit",
            expected_manifest_digest="0" * 64,
        )
        coordinator = RepositoryCommitCoordinator(
            journal,
            lambda _intent: CommitRecovery(committed=False),
            allow_ephemeral=True,
        )
        return context.commit_final(
            FinalCommit(
                prepared_authority(
                    context,
                    coordinator,
                    CommitKind.FINAL,
                    "wrong-safety-bundle-commit",
                    target,
                    lambda: PublishedManifest(
                        target_ref=target.target_ref,
                        manifest_digest=target.expected_manifest_digest,
                        result="artifact-ref",
                    ),
                    subject=CommitSubject(
                        context.run_id.value,
                        "another-safety-bundle",
                    ),
                ),
            )
        )

    handle = controller.start(
        plan(
            key=camera_key("wrong-safety-bundle"),
            execute=lambda *_: "staged",
            finalize=finalize,
            requires_final_commit=True,
        )
    )
    with pytest.raises(RunFailed):
        handle.result(2.0)
    assert "safety bundle differs from this Run" in handle.snapshot().primary_error


def test_captured_bound_device_reference_has_no_hardware_invocation_surface():
    key = camera_key("captured-binding")
    broker = DeviceBroker()
    device = broker.bind(
        key=key,
        identity=verified_identity(broker, key, "generation"),
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
        verify_safe_state=lambda: SafeStateAck("safe-readback"),
    )

    def execute(_context, _prepared):
        return device

    assert not hasattr(device, "execute")
    assert not hasattr(device, "_execute_command")
    run_plan = RunPlan(
        name="captured binding reference",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(ResourceClaim(key),),
        hazard_claims=(
            HazardClaim(key, str(key), device.connection_generation),
        ),
        bound_devices=(device,),
        preflight=lambda _ctx: None,
        execute=execute,
        cleanup=lambda context, _prepared, _primary: safe_cleanup_report(
            context, key
        ),
        finalize=lambda _ctx, result: result,
    )
    controller = RunController(new_arbiter())
    assert controller.run(run_plan) is device


def test_live_identity_change_rejects_start_before_any_hardware_command():
    key = camera_key("identity-changed-before-start")
    evidence = {"digest": "readback-at-bind"}
    commands = []
    broker = DeviceBroker()
    identity = broker.verify_identity(
        lambda: DeviceIdentityAck(
            str(key),
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
            evidence["digest"],
            "test-assets-v1",
        )
    )
    device = broker.bind(
        key=key,
        identity=identity,
        execute_command=lambda command: commands.append(command),
        cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
        verify_safe_state=lambda: SafeStateAck("safe-readback"),
    )
    evidence["digest"] = "different-live-readback"
    run_plan = RunPlan(
        name="identity must still match before run",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(ResourceClaim(key),),
        hazard_claims=(HazardClaim(key, str(key), device.connection_generation),),
        bound_devices=(device,),
        preflight=lambda _context: None,
        execute=lambda context, _prepared: context.device(key).execute("fire"),
        cleanup=lambda context, _prepared, _primary: safe_cleanup_report(context, key),
        finalize=lambda _context, result: result,
    )
    arbiter = new_arbiter()

    with pytest.raises(RuntimeError, match="explicit re-establishment"):
        RunController(arbiter).start(run_plan)

    assert commands == []
    assert not arbiter.active_claims()


def test_identity_change_during_run_cannot_authorize_safe_release():
    key = camera_key("identity-changed-during-run")
    evidence = {"digest": "readback-at-bind"}
    broker = DeviceBroker()
    identity = broker.verify_identity(
        lambda: DeviceIdentityAck(
            str(key),
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
            evidence["digest"],
            "test-assets-v1",
        )
    )
    device = broker.bind(
        key=key,
        identity=identity,
        execute_command=lambda command: command,
        cleanup_operations={SafetyOperation.SAFE_STATE: safe_ack},
        verify_safe_state=lambda: SafeStateAck("safe-readback"),
    )

    def execute(context, _prepared):
        context.device(key).execute("fire")
        evidence["digest"] = "different-live-readback"
        return "data"

    run_plan = RunPlan(
        name="identity must still match for safe proof",
        mode=RunMode.FINITE_EXACT,
        resource_claims=(ResourceClaim(key),),
        hazard_claims=(HazardClaim(key, str(key), device.connection_generation),),
        bound_devices=(device,),
        preflight=lambda _context: None,
        execute=execute,
        cleanup=lambda context, _prepared, _primary: safe_cleanup_report(context, key),
        finalize=lambda _context, result: result,
    )
    arbiter = new_arbiter()
    handle = RunController(arbiter).start(run_plan)

    with pytest.raises(RunFailed):
        handle.result(2.0)

    assert "explicit re-establishment" in handle.snapshot().cleanup_errors[0]
    assert isinstance(
        arbiter.acquire_all("retry", (ResourceClaim(key),)),
        ResourceQuarantined,
    )


def test_terminal_history_callback_runs_after_claim_release_and_keeps_owner_truthful():
    callback_entered = threading.Event()
    allow_callback_exit = threading.Event()

    class BlockingHistoryController(RunController):
        def _on_terminal(self, finished):
            callback_entered.set()
            assert allow_callback_exit.wait(2.0)
            super()._on_terminal(finished)

    arbiter = new_arbiter()
    controller = BlockingHistoryController(arbiter)
    key = camera_key("terminal-history-lock")
    handle = controller.start(plan(key=key, execute=lambda *_: "result"))
    assert callback_entered.wait(1.0)
    assert handle.snapshot().state is RunState.SUCCEEDED
    assert handle.owner_thread_alive()

    next_lease = arbiter.acquire_all("next-run", (ResourceClaim(key),))
    assert isinstance(next_lease, ResourceLease)
    next_lease._release_unarmed()

    allow_callback_exit.set()
    assert handle.result(2.0) == "result"
    deadline = threading.Event()
    for _ in range(100):
        if not handle.owner_thread_alive():
            break
        deadline.wait(0.001)
    assert not handle.owner_thread_alive()
