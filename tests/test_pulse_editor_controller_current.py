"""Qt-free current Pulse editor controller and preview worker contracts."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.pulse_application import PulseTargetDescriptor
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_pulse import (
    PORT_DIGITAL,
    load_deployed_pulse_target,
    pulse_target_manifest_from_lanes,
)
from zlc_workbench.pulse import PulseEditorSession
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


def test_owner_wake_without_new_fact_is_silent_and_runtime_poll_stays_narrow():
    offline = _controller()
    try:
        assert offline.pump() is None
    finally:
        offline.request_close()
        _pump_until(offline, lambda value: value.runtime_update().close_complete)

    target = load_deployed_pulse_target()
    running = RunSnapshot(
        RunId("narrow-runtime-update"),
        RunState.RUNNING,
        "execute",
        False,
        None,
        None,
        (),
        None,
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
            DeviceRef("test-installation", "test-runtime", "sequencer"),
            pulse_target_manifest_from_lanes(target),
            50_000_000.0,
            0,
            1,
        ),
        initial_connection_mode="virtual",
    )
    controller._run_snapshot = RunSnapshot(
        running.run_id,
        RunState.RUNNING,
        "queued",
        False,
        None,
        None,
        (),
        None,
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


def test_preview_worker_uses_document_identity_and_latest_presentation_revision():
    controller = _controller()
    try:
        controller.request_preview()
        _pump_until(
            controller,
            lambda value: value.preview_update().rendered_preview is not None,
        )
        first = controller.preview_update().rendered_preview
        assert first is not None
        assert first.payload.document_input.document_revision == 0
        assert first.payload.document_input.content_digest == first.timeline.fingerprint
        assert first.presentation_revision == 0
        assert not hasattr(first.payload, "evaluated_input")

        controller.set_preview_include_off(True)
        _pump_until(
            controller,
            lambda value: value.preview_update().rendered_preview is not None
            and value.preview_update().rendered_preview.presentation_revision == 1,
        )
        second = controller.preview_update().rendered_preview
        assert second is not None
        assert second.timeline is not first.timeline
        assert second.timeline.fingerprint == first.timeline.fingerprint
        assert "all channels" in second.status
        assert len(second.payload.row_keys) >= len(first.payload.row_keys)

        viewport = second.payload.viewport
        low, high = viewport.home_x_limits
        span = high - low
        controller.commit_preview_view(
            (low + 0.2 * span, high - 0.2 * span),
            presentation_revision=2,
        )
        _pump_until(
            controller,
            lambda value: value.preview_update().rendered_preview is not None
            and value.preview_update().rendered_preview.presentation_revision == 2,
        )
        third = controller.preview_update().rendered_preview
        assert third is not None
        assert third.payload.viewport.x_limits != third.payload.viewport.home_x_limits
        assert third.payload.document_input == second.payload.document_input
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


def test_preview_manual_size_is_transient_and_reset_on_reentry():
    controller = _controller()
    try:
        controller.set_preview_size("4x4")
        _pump_until(
            controller,
            lambda value: value.preview_update().rendered_preview is not None
            and value.preview_update().rendered_preview.size == "4x4",
        )
        pinned = controller.preview_update().rendered_preview
        assert pinned is not None

        controller.reset_preview_size()
        _pump_until(
            controller,
            lambda value: value.preview_update().rendered_preview is not None
            and value.preview_update().rendered_preview.presentation_revision
            > pinned.presentation_revision,
        )
        automatic = controller.preview_update().rendered_preview
        assert automatic is not None
        assert automatic.size != "4x4"
    finally:
        controller.request_close()
        _pump_until(controller, lambda value: value.runtime_update().close_complete)


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
        DeviceRef("test-installation", "test-runtime", "sequencer"),
        pulse_target_manifest_from_lanes(target),
        50_000_000.0,
        0,
        1,
    )
    controller = PulseEditorController(
        PulseEditorSession.new(target, time_step_ns=20),
        pulse=pulse,
        descriptor=descriptor,
        initial_connection_mode="virtual",
    )
    controller._run_snapshot = RunSnapshot(
        RunId("active-before-owner-close"),
        RunState.RUNNING,
        "execute",
        False,
        None,
        None,
        (),
        None,
    )

    controller.retire_borrowed_authority()
    controller.poll_runtime_change()

    assert pulse.observe_calls == 0
    assert controller._pulse is None
    _pump_until(controller, lambda value: value.runtime_update().close_complete)
