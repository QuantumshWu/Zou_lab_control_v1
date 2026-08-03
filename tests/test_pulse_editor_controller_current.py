"""Qt-free current Pulse editor controller and preview worker contracts."""

from __future__ import annotations

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
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_pulse import (
    FIELD_DURATION,
    PORT_DIGITAL,
    ApiParameter,
    PulseExecutionForm,
    PulseFieldRef,
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

        def wait(self):
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


def test_owner_wake_without_new_fact_is_silent_and_runtime_poll_stays_narrow():
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
    update = controller.poll_runtime_change()
    assert isinstance(update, PulseRuntimeUpdate)
    assert update.run_snapshot is running
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
                if controller._results:
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
    controller.poll_runtime_change()

    assert pulse.observe_calls == 0
    assert controller._pulse is None
    _pump_until(controller, lambda value: value.runtime_update().close_complete)
