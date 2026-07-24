from __future__ import annotations

from concurrent.futures import Executor, Future
import time

import pytest

from zlc_frontend.curve_display import (
    CurveDisplayState,
    curve_display_with_x_view,
)
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunCancelled,
    RunId,
    RunSnapshot,
    RunState,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.reference import ScanArtifactRef
from zlc_workbench.scan import (
    FinalScanPresentation,
    PreparedScanPanelRun,
    ScanPanelController,
)


_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
)


def _ref(digit: str = "a") -> ScanArtifactRef:
    return ScanArtifactRef("scan-repository", digit * 64)


def _snapshot(
    run_id: RunId,
    state: RunState,
    *,
    final_committed: bool = False,
    phase: str = "execute",
) -> RunSnapshot:
    return RunSnapshot(
        run_id=run_id,
        state=state,
        phase=phase,
        final_committed=final_committed,
        commit_recovery_warning=None,
        primary_error=None,
        cleanup_errors=(),
        recovery_instruction=None,
    )


class _ManualExecutor(Executor):
    def __init__(self) -> None:
        self.pending: list[tuple[Future, object, tuple, dict]] = []
        self.shutdown_called = False

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        self.pending.append((future, fn, args, kwargs))
        return future

    def run_next(self) -> Future:
        future, fn, args, kwargs = self.pending.pop(0)
        assert future.set_running_or_notify_cancel()
        try:
            value = fn(*args, **kwargs)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(value)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_called = True


class _RejectingExecutor(_ManualExecutor):
    def __init__(self, reject_at: int) -> None:
        super().__init__()
        self._reject_at = reject_at
        self._submissions = 0

    def submit(self, fn, /, *args, **kwargs):
        self._submissions += 1
        if self._submissions == self._reject_at:
            raise RuntimeError(f"submission {self._submissions} rejected")
        return super().submit(fn, *args, **kwargs)


class _FlakyClosingPreview:
    def __init__(self) -> None:
        self.closed = False
        self.retired = False
        self.worker_done = True
        self.fault = None
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_calls == 1:
            raise RuntimeError("presenter clear failed once")
        self.retired = True


class _FakeHandle:
    def __init__(self, run_id: str, reference: ScanArtifactRef) -> None:
        self.run_id = RunId(run_id)
        self.reference = reference
        self.current = _snapshot(self.run_id, RunState.RUNNING)
        self.cancel_reasons: list[str] = []
        self.wait_calls = 0
        self.result_calls = 0

    def snapshot(self) -> RunSnapshot:
        return self.current

    def cancel(self, reason: str = "cancel") -> CancelOutcome:
        self.cancel_reasons.append(reason)
        if self.current.state.terminal:
            return CancelOutcome.ALREADY_TERMINAL
        if self.current.state is RunState.CANCELLING:
            return CancelOutcome.ALREADY_REQUESTED
        self.current = _snapshot(self.run_id, RunState.CANCELLING, phase="cancelling")
        return CancelOutcome.REQUESTED

    def wait(self, timeout=None) -> RunSnapshot:
        self.wait_calls += 1
        return self.current

    def result(self, timeout=None) -> ScanArtifactRef:
        self.result_calls += 1
        snapshot = self.wait(timeout)
        if snapshot.state is RunState.SUCCEEDED and snapshot.final_committed:
            return self.reference
        raise RunCancelled(snapshot)

    def finish(self, state: RunState, *, final_committed: bool = False) -> None:
        assert state.terminal
        self.current = _snapshot(
            self.run_id,
            state,
            final_committed=final_committed,
            phase="terminal",
        )


class _Application:
    def __init__(self, handle: _FakeHandle) -> None:
        self.handle = handle
        self.prepare_calls = 0
        self.start_calls = 0
        self.project_calls: list[ScanArtifactRef] = []
        self.project_error: BaseException | None = None
        self.presentation_ref: ScanArtifactRef | None = None
        self.png_bytes = _PNG

    def prepare(self):
        self.prepare_calls += 1
        return PreparedScanPanelRun(None, self._start)

    def _start(self, preview):
        assert preview is None
        self.start_calls += 1
        return self.handle

    def project_final(self, source_ref):
        self.project_calls.append(source_ref)
        if self.project_error is not None:
            raise self.project_error
        return FinalScanPresentation(
            self.presentation_ref or source_ref,
            self.png_bytes,
            "x=detuning · y=site counts · final-only",
        )


def _controller(*, digest: str = "a"):
    reference = _ref(digest)
    handle = _FakeHandle(f"run-{digest}", reference)
    application = _Application(handle)
    executor = _ManualExecutor()
    wakes: list[None] = []
    controller = ScanPanelController(
        application,
        lambda: wakes.append(None),
        executor=executor,
    )
    return controller, application, handle, executor, wakes


def _reach_committed_reference(controller, handle, executor):
    controller.start()
    executor.run_next()  # application.prepare
    controller.owner_cycle()
    executor.run_next()  # prepared.start
    controller.owner_cycle()
    handle.finish(RunState.SUCCEEDED, final_committed=True)
    controller.owner_cycle()  # observes terminal, schedules result
    assert handle.result_calls == 0
    executor.run_next()  # handle.result -> also joins/reaps
    controller.owner_cycle()  # accepts ref, schedules final projection


def _reach_active_run(controller, executor):
    controller.start()
    executor.run_next()  # application.prepare
    controller.owner_cycle()
    executor.run_next()  # prepared.start
    controller.owner_cycle()


def test_application_reconfigure_resets_unscoped_curve_display_state() -> None:
    controller, _application, _handle, _executor, _wakes = _controller()
    changed = curve_display_with_x_view(CurveDisplayState(), (1.0, 2.0))
    assert controller.reconfigure_progressive_curve_display(changed) == 1
    assert controller.progressive_curve_display == changed

    replacement_handle = _FakeHandle("replacement-run", _ref("b"))
    controller.reconfigure(_Application(replacement_handle))
    assert controller.progressive_curve_display == CurveDisplayState()


def test_live_runtime_poll_never_builds_a_full_panel_model() -> None:
    controller, _application, handle, executor, _wakes = _controller()
    _reach_active_run(controller, executor)
    original_model = controller.view_model

    def forbidden_full_projection():
        raise AssertionError("live runtime poll rebuilt the full panel model")

    controller._build_view_model = forbidden_full_projection
    assert controller.poll_runtime_change() is None
    assert controller.view_model is original_model

    handle.current = _snapshot(
        handle.run_id,
        RunState.RUNNING,
        phase="cleanup",
    )
    update = controller.poll_runtime_change()
    assert update is not None
    assert update.status == "RUNNING / cleanup · FINAL-ONLY"
    assert update.can_stop
    assert not update.terminal_boundary
    assert controller.view_model is original_model


def test_terminal_runtime_poll_publishes_one_real_boundary() -> None:
    controller, _application, handle, executor, _wakes = _controller()
    _reach_active_run(controller, executor)
    handle.finish(RunState.SUCCEEDED, final_committed=True)

    update = controller.poll_runtime_change()

    assert update is not None and update.terminal_boundary
    assert update.status == "FINAL COMMITTED · RETRIEVING RESULT"
    assert not update.can_stop
    assert not controller.needs_periodic_poll
    assert controller.view_model.status == update.status
    assert len(executor.pending) == 1


def test_final_only_controller_reaps_terminal_before_projecting() -> None:
    controller, application, handle, executor, wakes = _controller()

    _reach_committed_reference(controller, handle, executor)

    interim = controller.view_model
    assert interim.artifact_ref == handle.reference
    assert interim.presentation is None
    assert interim.status == "FINAL · BUILDING DISPLAY"
    assert handle.result_calls == 1
    assert handle.wait_calls == 1
    assert application.project_calls == []

    executor.run_next()
    final = controller.owner_cycle()
    assert final.status == "FINAL"
    assert final.final_only is True
    assert final.artifact_ref == handle.reference
    assert final.presentation is not None
    assert final.presentation.source_ref == handle.reference
    assert final.presentation.png_bytes == _PNG
    assert application.project_calls == [handle.reference]
    assert final.can_start is True
    assert final.can_stop is False
    assert wakes


def test_stop_while_preparing_discards_command_before_admission() -> None:
    controller, _, handle, executor, _ = _controller(digest="b")

    controller.start()
    assert controller.stop() is CancelOutcome.REQUESTED
    assert controller.view_model.status == "CANCELLING BEFORE ADMISSION · NOT FINAL"
    assert handle.cancel_reasons == []

    executor.run_next()
    model = controller.owner_cycle()
    assert not executor.pending
    assert model.status == "CANCELLED BEFORE ADMISSION · NOT FINAL"
    assert model.can_start is True
    assert handle.cancel_reasons == []


def test_stop_while_start_is_inflight_cancels_returned_handle() -> None:
    controller, _, handle, executor, _ = _controller(digest="b")

    controller.start()
    executor.run_next()  # prepare
    controller.owner_cycle()  # submits start
    assert controller.stop() is CancelOutcome.REQUESTED
    assert controller.view_model.status == "CANCELLATION PENDING HANDLE · NOT FINAL"
    executor.run_next()  # start returns handle
    controller.owner_cycle()
    assert handle.cancel_reasons == [
        "scan panel stop requested before admission returned"
    ]
    assert handle.current.state is RunState.CANCELLING

    handle.finish(RunState.CANCELLED)
    controller.owner_cycle()
    assert handle.wait_calls == 0
    executor.run_next()
    final = controller.owner_cycle()
    assert handle.wait_calls == 1
    assert final.status == "CANCELLED · NOT FINAL"
    assert final.artifact_ref is None
    assert final.can_start is True


@pytest.mark.parametrize(
    ("reject_at", "expected_status"),
    (
        (1, "PREPARATION FAILED · NOT FINAL"),
        (2, "FAILED BEFORE ADMISSION · NOT FINAL"),
    ),
)
def test_executor_submission_failure_cannot_stick_scan_state(
    reject_at: int,
    expected_status: str,
) -> None:
    reference = _ref("9")
    handle = _FakeHandle("run-9", reference)
    application = _Application(handle)
    executor = _RejectingExecutor(reject_at)
    controller = ScanPanelController(
        application,
        lambda: None,
        executor=executor,
    )

    controller.start()
    if reject_at == 2:
        executor.run_next()
    controller.owner_cycle()
    model = controller.owner_cycle()

    assert model.status == expected_status
    assert model.can_start is True
    assert model.worker_idle is True
    assert application.start_calls == 0


def test_projection_failure_cannot_erase_the_final_artifact_reference() -> None:
    controller, application, handle, executor, _ = _controller(digest="c")
    application.project_error = RuntimeError("renderer unavailable")

    _reach_committed_reference(controller, handle, executor)
    committed = controller.view_model.artifact_ref
    executor.run_next()
    failed_display = controller.owner_cycle()

    assert committed == handle.reference
    assert failed_display.artifact_ref == handle.reference
    assert failed_display.presentation is None
    assert failed_display.status == "FINAL · DISPLAY FAILED"
    assert "renderer unavailable" in (failed_display.diagnostic or "")
    assert failed_display.can_start is True


def test_projection_artifact_identity_is_gated() -> None:
    controller, application, handle, executor, _ = _controller(digest="d")
    application.presentation_ref = _ref("e")

    _reach_committed_reference(controller, handle, executor)
    executor.run_next()
    model = controller.owner_cycle()

    assert model.artifact_ref == handle.reference
    assert model.presentation is None
    assert model.status == "FINAL · DISPLAY FAILED"
    assert "another artifact" in (model.diagnostic or "")


def test_close_during_start_revokes_generation_cancels_and_reaps_stale_handle() -> None:
    controller, _, handle, executor, _ = _controller(digest="f")

    controller.start()
    executor.run_next()
    controller.owner_cycle()
    controller.close()
    assert controller.closed is False
    assert controller.view_model.closing is True

    executor.run_next()  # stale start result
    controller.owner_cycle()  # cancels it and queues detached wait
    assert handle.cancel_reasons == ["stale scan panel start result"]
    assert controller.closed is False
    handle.finish(RunState.CANCELLED)
    executor.run_next()
    model = controller.owner_cycle()

    assert handle.wait_calls == 1
    assert model.closed is True
    assert model.status == "CLOSED"
    assert controller.worker_idle is True
    # An injected executor is composition-owned and must not be shut down here.
    assert executor.shutdown_called is False


def test_close_of_admitted_run_is_nonblocking_and_drains_its_reap_future() -> None:
    controller, _, handle, executor, _ = _controller(digest="1")
    controller.start()
    executor.run_next()
    controller.owner_cycle()
    executor.run_next()
    controller.owner_cycle()

    controller.close()
    assert handle.cancel_reasons == ["scan panel is closing"]
    assert handle.wait_calls == 0
    assert controller.closed is False
    handle.finish(RunState.CANCELLED)
    executor.run_next()
    model = controller.owner_cycle()

    assert handle.wait_calls == 1
    assert model.closed is True
    assert model.worker_idle is True


def test_close_waits_for_failed_preview_clear_retry() -> None:
    controller, _, _, _, _ = _controller(digest="3")
    preview = _FlakyClosingPreview()
    controller._preview = preview

    controller.close()
    assert controller.closed is False
    assert controller.view_model.closing is True
    assert preview.close_calls == 1

    time.sleep(0.11)
    model = controller.owner_cycle()
    assert preview.close_calls == 2
    assert model.closed is True
    assert model.status == "CLOSED"


def test_final_presentation_rejects_mutable_or_non_png_payloads() -> None:
    with pytest.raises(TypeError, match="immutable bytes"):
        FinalScanPresentation(_ref("2"), bytearray(_PNG), "projection")
    with pytest.raises(ValueError, match="PNG"):
        FinalScanPresentation(_ref("2"), b"not-png", "projection")
