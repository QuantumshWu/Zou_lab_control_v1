from __future__ import annotations

from dataclasses import replace
import time

import pytest

from Zou_lab_control.notebook import PulseFacade, connect
from Zou_lab_control.notebook.facade import _service_guard
from zlc_neutral_atom.pulse_application import (
    PulseApplicationOwner,
    prepare_pulse_execution,
)
from zlc_neutral_atom.runtime.run import CancelOutcome, RunState
from zlc_pulse import (
    FIELD_DURATION,
    PORT_DAC,
    PORT_DIGITAL,
    ApiParameter,
    FrozenScanTable,
    PulseDocument,
    PulseExecutionForm,
    PulseFieldRef,
    PulsePeriod,
    RepeatRegion,
    ScanParameter,
)


def _authored_scan_document(experiment) -> tuple[PulseDocument, PulseFieldRef]:
    descriptor = experiment.pulse.target
    target = descriptor.target
    scan_field = PulseFieldRef(FIELD_DURATION, "scan_period")
    api_field = PulseFieldRef(FIELD_DURATION, "api_period")
    document = PulseDocument(
        "application state proof",
        target,
        descriptor.time_step_ns,
        (
            PulsePeriod(
                "scan_period",
                100,
                "ns",
                "scan period",
                tuple(0 for _ in target.raw_lanes),
            ),
            PulsePeriod(
                "api_period",
                100,
                "ns",
                "API period",
                tuple(0 for _ in target.raw_lanes),
            ),
        ),
        scan_parameters=(
            ScanParameter("scan_duration", scan_field, "Scan duration", "ns"),
        ),
        scan_table=FrozenScanTable(
            ("scan_duration",),
            ((100,), (200,)),
        ),
        api_parameters=(ApiParameter("api_duration", api_field, "ns"),),
        visible_ports=tuple(
            port.key
            for port in target.ports
            if port.kind in (PORT_DIGITAL, PORT_DAC)
        ),
        repeat=RepeatRegion("scan_period", "api_period", 2),
    )
    return document, api_field


def _wait_for_observation(pulse: PulseFacade, predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while True:
        observation = pulse.observe_active()
        if observation is not None and predicate(observation):
            return observation
        if time.monotonic() >= deadline:
            raise AssertionError("pulse observation did not reach the expected state")
        time.sleep(0.005)


def _wait_for_scan_progress(pulse: PulseFacade, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while True:
        progress = pulse.observe_scan_progress()
        if progress is not None and progress.available:
            return progress
        if time.monotonic() >= deadline:
            reason = None if progress is None else progress.unavailable_reason
            raise AssertionError(f"pulse scan cursor remained unavailable: {reason}")
        time.sleep(0.005)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    with connect(
        repository=tmp_path_factory.mktemp("pulse-application-state")
    ) as value:
        yield value


def test_applied_snapshot_preserves_source_api_intent_and_outer_scan_sweeps(
    experiment,
):
    document, api_field = _authored_scan_document(experiment)
    request = experiment.pulse.request(
        document,
        PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        api_values={"api_duration": 200},
        scan_sweep_count=3,
    )

    assert request.document is document
    assert request.document.api_parameters == document.api_parameters
    assert request.api_values == (("api_duration", 200),)
    assert experiment.pulse.snapshot() is None

    descriptor = experiment.pulse.inspect(request)
    assert descriptor.scan_point_count == 6
    assert experiment.pulse.snapshot() is None

    result = experiment.pulse.run(request)
    applied = experiment.pulse.snapshot()
    assert applied is not None
    assert applied.run_id == result.run_id
    assert applied.artifact_digest == result.artifact_digest
    assert applied.source_document is document
    assert applied.source_document.api_parameters == document.api_parameters
    assert applied.api_values == (("api_duration", 200),)
    assert applied.scan_sweep_count == 3
    assert applied.execution_document.api_parameters == ()
    assert applied.execution_document.field_value(api_field) == (200, "ns")
    assert applied.execution_document.repeat == document.repeat
    assert applied.execution_document.scan_table is not None
    assert applied.execution_document.scan_table.rows == (
        (100,),
        (200,),
        (100,),
        (200,),
        (100,),
        (200,),
    )
    assert applied.execution_document_digest == applied.execution_document.fingerprint
    assert applied.source_document_digest == document.fingerprint


def test_two_facades_share_run_observation_applied_identity_and_safe_cancel(
    experiment,
):
    document, _api_field = _authored_scan_document(experiment)
    hold_document = replace(
        document,
        scan_parameters=(),
        scan_table=None,
        api_parameters=(),
    )
    peer = PulseFacade(experiment.pulse._token)
    assert peer.snapshot() == experiment.pulse.snapshot()

    request = experiment.pulse.request(
        hold_document,
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    handle = experiment.pulse.start(request)
    holding = _wait_for_observation(
        peer,
        lambda observation: observation.applied is not None
        and observation.run.phase == "holding-pulse",
    )
    assert holding.run.run_id == handle.run_id
    assert holding.request is request
    assert peer.snapshot() == experiment.pulse.snapshot() == holding.applied

    with pytest.raises(RuntimeError, match="already active"):
        peer.start(request)

    assert peer.cancel_active("state owner cancellation proof") is CancelOutcome.REQUESTED
    terminal = _wait_for_observation(
        experiment.pulse,
        lambda observation: observation.run.state.terminal,
    )
    assert terminal.run.state is RunState.CANCELLED
    assert terminal.run.safety_bundle_id is not None
    assert terminal.run.final_committed is False
    assert peer.observe_active() == terminal
    assert peer.cancel_active() is CancelOutcome.ALREADY_TERMINAL


def test_run_rejects_both_continuous_forms_and_outer_k_is_finite_scan_only(
    experiment,
):
    document, _api_field = _authored_scan_document(experiment)
    api_values = {"api_duration": 200}
    continuous_scan = experiment.pulse.request(
        document,
        PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
        api_values=api_values,
    )
    with pytest.raises(ValueError, match="started and cancelled"):
        experiment.pulse.run(continuous_scan)

    hold_document = replace(
        document,
        scan_parameters=(),
        scan_table=None,
        api_parameters=(),
    )
    continuous_hold = experiment.pulse.request(
        hold_document,
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    with pytest.raises(ValueError, match="started and cancelled"):
        experiment.pulse.run(continuous_hold)

    with pytest.raises(ValueError, match="does not accept a finite outer"):
        experiment.pulse.request(
            document,
            PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
            api_values=api_values,
            scan_sweep_count=2,
        )
    with pytest.raises(ValueError, match="only to autonomous scan"):
        experiment.pulse.request(
            hold_document,
            PulseExecutionForm.STATIC_ONCE,
            scan_sweep_count=2,
        )

    handle = experiment.pulse.start(continuous_scan)
    cycling = _wait_for_observation(
        experiment.pulse,
        lambda observation: observation.applied is not None
        and observation.run.phase == "holding-pulse",
    )
    assert cycling.run.run_id == handle.run_id
    assert cycling.applied is not None
    assert (
        cycling.applied.execution_form
        is PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS
    )
    assert cycling.applied.execution_document.repeat == document.repeat
    assert cycling.applied.execution_document.scan_table == document.scan_table
    progress = _wait_for_scan_progress(experiment.pulse)
    assert progress.run_id == handle.run_id.value
    assert progress.artifact_digest == cycling.applied.artifact_digest
    assert progress.point_count == 2
    assert progress.current_point_index in (0, 1)
    assert not hasattr(progress, "completed_sweep_count")
    assert experiment.pulse.cancel_active("continuous scan proof") is CancelOutcome.REQUESTED
    cancelled = _wait_for_observation(
        experiment.pulse,
        lambda observation: observation.run.state.terminal,
    )
    assert cancelled.run.state is RunState.CANCELLED
    assert cancelled.run.safety_bundle_id is not None


def test_continuous_backend_failure_terminates_run_without_user_cancel(
    experiment,
):
    document, _api_field = _authored_scan_document(experiment)
    request = experiment.pulse.request(
        document,
        PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
        api_values={"api_duration": 200},
    )
    owner = PulseApplicationOwner()
    with _service_guard(experiment._authority_token) as services:
        port = replace(
            services.runtime.pulse_port(request.sequencer_ref),
            _continuous_failure_waiter=(
                lambda _session_id, _run_id, _digest, _timeout: (
                    "forced backend observer failure"
                )
            ),
        )
        prepared = prepare_pulse_execution(
            request,
            pulse_port=port,
            start_run=services.runtime.start,
            on_applied=owner._record_applied,
        )
        handle = owner.start(prepared)

    deadline = time.monotonic() + 5.0
    while not handle.snapshot().state.terminal:
        if time.monotonic() >= deadline:
            raise AssertionError("backend failure did not terminate the continuous Run")
        time.sleep(0.005)
    terminal = handle.snapshot()
    assert terminal.state is RunState.FAILED
    assert terminal.safety_bundle_id is not None
    assert terminal.final_committed is False
