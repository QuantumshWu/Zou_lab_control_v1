"""Qt-free current Pulse editor controller and preview worker contracts."""

from __future__ import annotations

from concurrent.futures import Future
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.devices.sequencer.application import (
    AppliedPulseSnapshot,
    PulseRunRequest,
    PulseTargetDescriptor,
)
from zlc_neutral_atom.devices.sequencer.port import PulseScanProgress
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_pulse import (
    FIELD_DURATION,
    PORT_DIGITAL,
    ApiParameter,
    FrozenScanTable,
    PulseExecutionForm,
    PulseFieldRef,
    ScanParameter,
    load_deployed_pulse_target,
    pulse_target_manifest_from_lanes,
    resolve_api_parameters,
)
from zlc_workbench.pulse_editor.session import PulseEditorSession
from zlc_workbench.pulse_editor.controller import (
    PulseEditorController,
    PulseOwnerUpdate,
    PulseRuntimeUpdate,
)


def _pump_until(controller, predicate, *, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while not predicate(controller) and time.monotonic() < deadline:
        time.sleep(0.005)
        controller.pump()
    assert predicate(controller)
    return controller


def _controller() -> PulseEditorController:
    target = load_deployed_pulse_target()
    session = PulseEditorSession.new(target, time_step_ns=20)
    return PulseEditorController(session)


def _descriptor(name: str) -> PulseTargetDescriptor:
    return PulseTargetDescriptor(
        DeviceRef(
            runtime_instance_id=name,
            instance_id="sequencer",
            type_id="sequencer.test",
            role="sequencer",
        ),
        pulse_target_manifest_from_lanes(load_deployed_pulse_target()),
        50_000_000.0,
        0,
    )


class _ScanHandle:
    def __init__(self, pulse, run_id: RunId) -> None:
        self._pulse = pulse
        self.run_id = run_id

    def snapshot(self):
        return self._pulse.run

    def cancel(self, reason=""):
        return self._pulse.cancel_active(reason)

    def wait(self, timeout=None):
        return self.snapshot()


class _StrandedScanPulse:
    """A running scan whose replacement promise is never completed.

    This is not a hypothetical: the holding-pulse loop is the only completer,
    and a Run that is torn down drops its replacement queue without completing
    the futures it already handed out.
    """

    def __init__(self, document, descriptor: PulseTargetDescriptor) -> None:
        self._descriptor = descriptor
        self._document = document
        self.run = None
        self.applied = None
        self.replacements: list[Future] = []

    def request(
        self,
        source,
        execution_form,
        *,
        api_values=None,
        scan_sweep_count=1,
    ):
        return PulseRunRequest(
            source,
            execution_form,
            self._descriptor.sequencer_ref,
            None,
            (),
            scan_sweep_count,
        )

    def start(self, request):
        run_id = RunId("stranded-replacement")
        execution = resolve_api_parameters(request.document, {})
        self.applied = AppliedPulseSnapshot(
            run_id.value,
            request.document,
            (),
            request.scan_sweep_count,
            execution,
            request.execution_form,
            execution.fingerprint,
        )
        self.run = RunSnapshot(
            run_id=run_id,
            state=RunState.RUNNING,
            phase="holding-pulse",
            primary_error=None,
            cleanup_errors=(),
        )
        return _ScanHandle(self, run_id)

    def snapshot(self):
        return self.applied

    def observe_active(self):
        if self.run is None:
            return None
        return SimpleNamespace(request=None, run=self.run, applied=self.applied)

    def cancel_active(self, _reason=""):
        if self.run is not None and not self.run.state.terminal:
            self.run = replace(self.run, state=RunState.CANCELLED)
        return None

    def observe_scan_progress(self):
        assert self.applied is not None
        table = self.applied.source_document.scan_table
        return PulseScanProgress(
            self.run.run_id.value,
            self.applied.artifact_digest,
            len(table.rows),
            0,
            "RUNNING",
        )

    def replace_active(self, _request):
        stranded: Future = Future()
        self.replacements.append(stranded)
        return stranded


def _stranded_scan_controller() -> tuple[PulseEditorController, _StrandedScanPulse]:
    target = load_deployed_pulse_target()
    document = PulseEditorSession.new(target, time_step_ns=20).document
    field = PulseFieldRef(FIELD_DURATION, document.periods[0].period_id)
    document = replace(
        document,
        scan_parameters=(ScanParameter("duration", field, "Duration", "ns"),),
        scan_table=FrozenScanTable(("duration",), ((1000,), (2000,), (3000,))),
    )
    descriptor = _descriptor("test-stranded-replacement")
    pulse = _StrandedScanPulse(document, descriptor)
    controller = PulseEditorController(
        PulseEditorSession(document),
        pulse=pulse,
        descriptor=descriptor,
        initial_connection_mode="virtual",
    )
    controller.start(PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS)
    _pump_until(
        controller,
        lambda value: value.runtime_update().run_snapshot is not None
        and value.runtime_update().run_snapshot.state is RunState.RUNNING,
    )
    controller.request_scan_progress()
    _pump_until(
        controller,
        lambda value: value.runtime_update().scan_progress is not None,
    )
    return controller, pulse


def test_a_stranded_replacement_promise_never_occupies_the_editor_workers():
    """The editor's pool runs only work the editor itself can finish.

    Submitting ``Future.result`` as work parked a worker forever; two of them
    blocked every later file, connect, preview, and start command behind a
    queue nobody could drain.
    """

    controller, pulse = _stranded_scan_controller()
    try:
        controller.hold_scan_point()
        controller.step_scan_point(1)
        assert len(pulse.replacements) == 2
        assert not any(promise.done() for promise in pulse.replacements)
        assert controller.worker_idle

        # Both workers are still available, and the editor still reports the
        # truth about itself instead of a false "not idle".
        controller.request_preview()
        _pump_until(
            controller,
            lambda value: value.preview_update().plot is not None,
        )
        assert not controller.runtime_update().run_admission_pending
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)
    assert controller.worker_idle
    # Revocation lives with the fact: closing the editor made the queued point
    # untrue, so a late answer to it can no longer be admitted.  The promise
    # itself is never cancelled - the Run's lane may already have taken it, and
    # completing a cancelled future would raise inside the Run.
    for promise in pulse.replacements:
        assert not promise.cancelled()
    pulse.replacements[-1].set_result(pulse.applied)
    assert controller.pump() is None
    assert controller.runtime_update().held_scan_point is None


def test_a_stranded_hold_is_replaced_by_the_next_one_without_a_second_promise():
    controller, pulse = _stranded_scan_controller()
    try:
        controller.hold_scan_point()
        first = pulse.replacements[0]
        controller.step_scan_point(1)
        assert controller._pending_hold is not None
        assert controller._pending_hold.future is pulse.replacements[1]

        applied = pulse.applied
        assert applied is not None
        # The superseded promise answers late; it belongs to no hold, so the
        # editor cannot adopt a scan point it already replaced.
        first.set_result(applied)
        assert controller.pump() is None
        assert controller.runtime_update().held_scan_point is None

        pulse.replacements[1].set_result(applied)
        _pump_until(
            controller,
            lambda value: value.runtime_update().held_scan_point is not None,
        )
        assert controller.runtime_update().held_scan_point[0] == 1
        assert controller._pending_hold is None
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


def test_run_admission_is_derived_from_the_pending_work_not_a_stored_flag():
    """A started-but-undrained admission must not latch the editor busy."""

    target = load_deployed_pulse_target()
    descriptor = _descriptor("test-admission-level")
    admitted = threading.Event()

    class SlowPulse(_StrandedScanPulse):
        def start(self, request):
            admitted.wait(10.0)
            return super().start(request)

    document = PulseEditorSession.new(target, time_step_ns=20).document
    pulse = SlowPulse(document, descriptor)
    controller = PulseEditorController(
        PulseEditorSession(document),
        pulse=pulse,
        descriptor=descriptor,
        initial_connection_mode="virtual",
    )
    try:
        controller.start(PulseExecutionForm.CONTINUOUS_MONITOR)
        assert controller.runtime_update().run_admission_pending
        assert controller._run_busy()
        admitted.set()
        _pump_until(
            controller,
            lambda value: value.runtime_update().run_snapshot is not None,
        )
        assert not controller.runtime_update().run_admission_pending
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


def _gated_scan_source(gate: Path) -> str:
    """Trusted-local scan Python that occupies a worker until ``gate`` exists."""

    return (
        "import time\n"
        "from pathlib import Path\n"
        f"gate = Path({str(gate)!r})\n"
        "deadline = time.monotonic() + 20.0\n"
        "while not gate.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.005)\n"
        "scan_table = [[1000.0]]\n"
    )


def _scan_workspace_controller() -> PulseEditorController:
    target = load_deployed_pulse_target()
    document = PulseEditorSession.new(target, time_step_ns=20).document
    field = PulseFieldRef(FIELD_DURATION, document.periods[0].period_id)
    document = replace(
        document,
        scan_parameters=(ScanParameter("duration", field, "Duration", "ns"),),
    )
    return PulseEditorController(PulseEditorSession(document))


def test_scan_workspace_busy_is_derived_from_the_pending_work(tmp_path: Path):
    """Resetting the workspace must not tell the editor a worker stopped.

    ``new_document`` rebuilds the Scan workspace, and a stored busy flag was
    cleared there while the worker it described was still running.  The editor
    then admitted a second scan program, both pool workers were occupied by
    work it believed was not happening, and every later file, preview, connect,
    and start command queued behind them.
    """

    gate = tmp_path / "gate"
    program = tmp_path / "program.py"
    program.write_text("scan_table = [[1000.0]]\n", encoding="utf-8")
    controller = _scan_workspace_controller()
    try:
        controller.generate_scan_source(_gated_scan_source(gate))
        assert controller.current_scan_workspace.busy_operation == "generate"
        assert not controller.worker_idle

        controller.new_document()
        # The work is still running, so the workspace still says so.
        assert controller.current_scan_workspace.busy_operation == "generate"
        try:
            controller.load_scan_program(program)
        except RuntimeError as error:
            assert "generate" in str(error)
        else:
            raise AssertionError("a second scan program occupied the second worker")

        gate.write_text("go", encoding="utf-8")
        _pump_until(
            controller,
            lambda value: value.current_scan_workspace.busy_operation is None,
        )
        assert controller.worker_idle
        # The same removal that consumed the result cleared the busy fact, so
        # the next operation is admitted without anyone clearing a flag.
        controller.load_scan_program(program)
        _pump_until(
            controller,
            lambda value: value.current_scan_workspace.busy_operation is None,
        )
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


def test_promoting_a_stashed_start_is_published_without_a_stored_reap_flag():
    """The admission move must be visible from the levels that produce it.

    ``run_admission_pending`` stays true across the promotion of a stashed
    start into an in-flight one, so a change tuple carrying only that
    disjunction cannot see the move.  A stored ``_owner_reaped`` boolean used
    to supply the signal; the two levels it stood between supply it directly.
    """

    target = load_deployed_pulse_target()
    descriptor = _descriptor("test-promoted-start")
    admitted = threading.Event()

    class _StashingPulse(_StrandedScanPulse):
        def __init__(self, document, descriptor) -> None:
            super().__init__(document, descriptor)
            self.active = RunSnapshot(
                run_id=RunId("previous-owner"),
                state=RunState.RUNNING,
                phase="running",
                primary_error=None,
                cleanup_errors=(),
            )

        def observe_active(self):
            if self.active is None:
                return super().observe_active()
            return SimpleNamespace(request=None, run=self.active, applied=None)

        def start(self, request):
            admitted.wait(10.0)
            return super().start(request)

    document = PulseEditorSession.new(target, time_step_ns=20).document
    pulse = _StashingPulse(document, descriptor)
    controller = PulseEditorController(
        PulseEditorSession(document),
        pulse=pulse,
        descriptor=descriptor,
        initial_connection_mode="virtual",
    )
    try:
        assert not hasattr(controller, "_owner_reaped")
        controller.start(PulseExecutionForm.CONTINUOUS_MONITOR)
        assert controller._pending_start is not None
        assert not controller._in_flight("start")

        # Nothing else moves across the promotion: no handle, no snapshot, no
        # applied intent, no diagnostic - only which level holds the press.
        pulse.active = None
        controller._run_snapshot = None
        publication = controller.pump()
        assert controller._pending_start is None
        assert controller._in_flight("start")
        assert publication is not None
        assert publication.runtime is not None
        assert publication.runtime.run_admission_pending
    finally:
        admitted.set()
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


def test_api_slot_edit_dirties_applied_intent_and_on_pulse_clears_it():
    """API-slot units and applied identity have one exact model owner."""

    target = load_deployed_pulse_target()
    document = PulseEditorSession.new(target, time_step_ns=20).document
    field = PulseFieldRef(FIELD_DURATION, document.periods[0].period_id)
    document = replace(
        document,
        api_parameters=(ApiParameter("duration", field, "ns"),),
    )
    descriptor = PulseTargetDescriptor(
        DeviceRef(
            runtime_instance_id="test-api-slot",
            instance_id="sequencer",
            type_id="sequencer.test",
            role="sequencer",
        ),
        pulse_target_manifest_from_lanes(target),
        50_000_000.0,
        0,
    )

    class Handle:
        def __init__(self, pulse, run_id: RunId) -> None:
            self._pulse = pulse
            self.run_id = run_id

        def snapshot(self):
            return self._pulse._run

        def cancel(self, reason=""):
            return self._pulse.cancel_active(reason)

        def wait(self, timeout=None):
            return self.snapshot()

    class Pulse:
        def __init__(self) -> None:
            self._request = None
            self._run = None
            self._applied = None
            self.requests: list[PulseRunRequest] = []

        def request(
            self,
            source,
            execution_form,
            *,
            api_values=None,
            scan_sweep_count=1,
        ):
            supplied = {} if api_values is None else dict(api_values)
            request = PulseRunRequest(
                source,
                execution_form,
                descriptor.sequencer_ref,
                None,
                tuple(
                    (parameter.parameter_id, supplied[parameter.parameter_id])
                    for parameter in source.api_parameters
                ),
                scan_sweep_count,
            )
            self.requests.append(request)
            return request

        def start(self, request):
            self._request = request
            run_id = RunId(f"api-slot-{len(self.requests)}")
            execution = resolve_api_parameters(
                request.document,
                dict(request.api_values),
            )
            self._applied = AppliedPulseSnapshot(
                run_id.value,
                request.document,
                request.api_values,
                request.scan_sweep_count,
                execution,
                request.execution_form,
                execution.fingerprint,
            )
            self._run = RunSnapshot(
                run_id=run_id,
                state=RunState.RUNNING,
                phase="holding-pulse",
                primary_error=None,
                cleanup_errors=(),
            )
            return Handle(self, run_id)

        def snapshot(self):
            return self._applied

        def observe_active(self):
            if self._run is None:
                return None
            return SimpleNamespace(
                request=self._request,
                run=self._run,
                applied=self._applied,
            )

        def cancel_active(self, _reason=""):
            if self._run is not None and not self._run.state.terminal:
                self._run = replace(self._run, state=RunState.CANCELLED)
            return None

        def observe_scan_progress(self):
            return None

    pulse = Pulse()
    controller = PulseEditorController(
        PulseEditorSession(document),
        pulse=pulse,
        descriptor=descriptor,
        initial_connection_mode="virtual",
    )
    try:
        controller.start(PulseExecutionForm.CONTINUOUS_MONITOR)
        _pump_until(
            controller,
            lambda value: (
                value.runtime_update().applied_snapshot is not None
                and value.runtime_update().run_snapshot is not None
                and value.runtime_update().run_snapshot.state is RunState.RUNNING
            ),
        )
        first = controller.runtime_update().applied_snapshot
        assert first is not None
        assert controller.runtime_update().is_document_applied(
            controller.current_document
        )

        controller.set_period_duration(field.period_id, 1, "us")
        assert not controller.runtime_update().is_document_applied(
            controller.current_document
        )

        controller.start(PulseExecutionForm.CONTINUOUS_MONITOR)
        _pump_until(
            controller,
            lambda value: (
                len(pulse.requests) == 2
                and value.runtime_update().applied_snapshot is not None
                and value.runtime_update().applied_snapshot is not first
                and value.runtime_update().run_snapshot is not None
                and value.runtime_update().run_snapshot.state is RunState.RUNNING
            ),
        )
        assert pulse.requests[-1].api_values == (("duration", 1000),)
        applied = controller.runtime_update().applied_snapshot
        assert applied is not None
        assert applied.execution_document.field_value(field) == (1000, "ns")
        assert applied.authoring_document.field_value(field) == (1, "us")
        assert controller.runtime_update().is_document_applied(
            controller.current_document
        )
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


def test_owner_wake_without_new_fact_is_silent_and_the_clock_shares_that_entrance():
    offline = _controller()
    try:
        assert offline.pump() is None
    finally:
        offline.request_close()
        _pump_until(offline, lambda value: value.runtime_update().close_complete)

    target = load_deployed_pulse_target()
    running = RunSnapshot(
        run_id=RunId("narrow-runtime-update"),
        state=RunState.RUNNING,
        phase="execute",
        primary_error=None,
        cleanup_errors=(),
    )

    class Pulse:
        def observe_active(self):
            return SimpleNamespace(
                run=running,
                applied=None,
                request=SimpleNamespace(document=None),
            )

        def cancel_active(self, _reason=""):
            return None

        def snapshot(self):
            return None

    controller = PulseEditorController(
        PulseEditorSession.new(target, time_step_ns=20),
        pulse=Pulse(),
        descriptor=PulseTargetDescriptor(
            DeviceRef(
                runtime_instance_id="test-runtime",
                instance_id="sequencer",
                type_id="sequencer.test",
                role="sequencer",
            ),
            pulse_target_manifest_from_lanes(target),
            50_000_000.0,
            0,
        ),
        initial_connection_mode="virtual",
    )
    controller._run_snapshot = RunSnapshot(
        run_id=running.run_id,
        state=RunState.RUNNING,
        phase="queued",
        primary_error=None,
        cleanup_errors=(),
    )
    # The 40 ms clock is not a second, narrower advancer: it enters the same
    # pump every wake enters, so no fact can be observed by a path that cannot
    # also clear it.
    assert not hasattr(controller, "poll_runtime_change")
    publication = controller.pump()
    assert isinstance(publication, PulseOwnerUpdate)
    assert isinstance(publication.runtime, PulseRuntimeUpdate)
    assert publication.runtime.run_snapshot is running
    controller._pulse = None
    controller.request_close()
    controller.pump()


def test_preview_completion_publishes_only_preview_surface():
    controller = _controller()
    try:
        controller.request_preview()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with controller._lock:
                if any(entry.completion is not None for entry in controller._pending):
                    break
            time.sleep(0.005)
        else:
            raise AssertionError("preview worker did not complete")

        publication = controller.pump()
        assert isinstance(publication, PulseOwnerUpdate)
        assert publication.preview is not None
        assert publication.editor is None
        assert publication.file is None
        assert publication.runtime is None
        assert publication.scan_progress is None
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


def test_preview_worker_publishes_latest_typed_plot_projection_only():
    controller = _controller()
    try:
        controller.request_preview()
        _pump_until(
            controller,
            lambda value: value.preview_update().plot is not None,
        )
        first = controller.preview_update().plot
        assert first is not None
        assert first.editor_revision == 0
        assert not first.include_off_rows
        assert first.spec.labels.title == first.timeline.title
        assert not hasattr(first, "raster")
        assert not hasattr(first, "payload")

        controller.set_preview_include_off(True)
        _pump_until(
            controller,
            lambda value: value.preview_update().plot is not None
            and value.preview_update().plot.include_off_rows,
        )
        second = controller.preview_update().plot
        assert second is not None
        assert second.timeline is not first.timeline
        assert second.timeline.fingerprint == first.timeline.fingerprint
        assert "all channels" in second.status
        assert len(second.data.channels) >= len(first.data.channels)
        assert second.recommended_size

        controller.set_preview_include_off(False)
        _pump_until(
            controller,
            lambda value: value.preview_update().plot is not None
            and not value.preview_update().plot.include_off_rows,
        )
        third = controller.preview_update().plot
        assert third is not None
        assert third.data == first.data
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)
        assert controller.worker_idle


def test_clear_row_and_clear_all_are_domain_edits_not_widget_reconstruction():
    controller = _controller()
    try:
        initial = controller.current_document
        digital = next(
            port.key for port in initial.target.ports if port.kind == PORT_DIGITAL
        )
        controller.set_digital(initial.periods[0].period_id, digital, True)
        controller.clear_port(digital)
        row_cleared = controller.current_document
        lane = row_cleared.target.by_key[digital].lanes[0]
        lane_index = row_cleared.target.raw_lanes.index(lane)
        assert row_cleared.periods[0].states[lane_index] == 0

        controller.rename_document("kept name")
        controller.set_visible_ports([digital])
        controller.clear_all()
        cleared = controller.current_document
        assert cleared.name == "kept name"
        assert cleared.visible_ports == (digital,)
        assert len(cleared.periods) == 1
        assert (cleared.periods[0].duration, cleared.periods[0].unit) == (1, "us")
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


def test_editor_file_state_distinguishes_loaded_from_saved(tmp_path: Path):
    target = load_deployed_pulse_target()
    created = PulseEditorSession.new(target, time_step_ns=20)
    assert created.file_state == "new"

    path = tmp_path / "operator-pulse.json"
    created.save(path)
    assert created.file_state == "saved"

    loaded = PulseEditorSession.load(path)
    assert loaded.file_state == "loaded"
    loaded.replace_document(loaded.document)
    assert loaded.file_state == "loaded"


def test_borrowed_experiment_retirement_detaches_before_runtime_timer_poll():
    target = load_deployed_pulse_target()

    class ClosedExperimentPulse:
        observe_calls = 0

        def observe_active(self):
            self.observe_calls += 1
            raise RuntimeError("Experiment is closed")

        def cancel_active(self, _reason=""):
            return None

        def request(self, *_args, **_kwargs):
            raise AssertionError("request is not part of this close-race reproduction")

        def start(self, *_args, **_kwargs):
            raise AssertionError("start is not part of this close-race reproduction")

        def snapshot(self):
            return None

    pulse = ClosedExperimentPulse()
    descriptor = PulseTargetDescriptor(
        DeviceRef(
            runtime_instance_id="test-runtime",
            instance_id="sequencer",
            type_id="sequencer.test",
            role="sequencer",
        ),
        pulse_target_manifest_from_lanes(target),
        50_000_000.0,
        0,
    )
    controller = PulseEditorController(
        PulseEditorSession.new(target, time_step_ns=20),
        pulse=pulse,
        descriptor=descriptor,
        initial_connection_mode="virtual",
    )
    controller._run_snapshot = RunSnapshot(
        run_id=RunId("active-before-owner-close"),
        state=RunState.RUNNING,
        phase="execute",
        primary_error=None,
        cleanup_errors=(),
    )

    controller.retire_borrowed_authority()
    controller.pump()

    assert pulse.observe_calls == 0
    assert controller._pulse is None
    _pump_until(controller, lambda value: value.runtime_update().close_complete)
