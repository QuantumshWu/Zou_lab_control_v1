from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from Zou_lab_control.api import PulseFacade, WorkspacePaths, connect
from Zou_lab_control.api.facade import _service_guard
from zlc_neutral_atom.devices.sequencer import application as pulse_application
from zlc_neutral_atom.devices.sequencer.application import (
    PulseApplicationOwner,
    PulseReplacementLane,
    prepare_pulse_execution,
)
from zlc_neutral_atom.devices.sequencer.port import PulseSession
from zlc_neutral_atom.devices.simulation.sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
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


def _hold_document(experiment) -> PulseDocument:
    document, _api_field = _authored_scan_document(experiment)
    return replace(
        document,
        scan_parameters=(),
        scan_table=None,
        api_parameters=(),
    )


def _hold_request_of_duration(experiment, hold_document: PulseDocument, duration: int):
    return experiment.pulse.request(
        replace(
            hold_document,
            periods=(
                replace(hold_document.periods[0], duration=duration),
                *hold_document.periods[1:],
            ),
        ),
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )


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
        "virtual",
        workspace=WorkspacePaths.for_workspace(
            tmp_path_factory.mktemp("pulse-application-state").resolve()
        ),
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


def test_two_facades_share_unified_preemption_observation_and_safe_cancel(
    experiment,
):
    hold_document = _hold_document(experiment)
    peer = PulseFacade(experiment._services)
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
    assert holding.applied.source_document is hold_document
    assert peer.snapshot() == experiment.pulse.snapshot() == holding.applied

    replacement = peer.start(request)
    assert replacement.run_id != handle.run_id
    retired = handle.wait(5.0)
    assert retired.state is RunState.CANCELLED
    holding = _wait_for_observation(
        peer,
        lambda observation: observation.run.run_id == replacement.run_id
        and observation.applied is not None
        and observation.run.phase == "holding-pulse",
    )
    assert holding.applied.source_document is hold_document
    assert peer.snapshot() == experiment.pulse.snapshot() == holding.applied

    assert peer.cancel_active("state owner cancellation proof") is CancelOutcome.REQUESTED
    terminal = _wait_for_observation(
        experiment.pulse,
        lambda observation: observation.run.state.terminal,
    )
    assert terminal.run.state is RunState.CANCELLED
    assert terminal.run.run_id == replacement.run_id
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

    hold_document = _hold_document(experiment)
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
    with _service_guard(experiment._services) as services:
        port = replace(
            services.runtime.require_capability(
                request.sequencer_ref,
                "pulse.execute",
            ),
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


def test_replacement_is_refused_by_the_dying_run_before_it_is_terminal(
    experiment,
    monkeypatch,
):
    """A drained replacement lane refuses new work in the same breath.

    The Run stops consuming replacements when its plan cleanup runs, but the
    terminal state anyone else can see is published several hardware round
    trips later.  Anything admitted in that interval would be a promise with
    no consumer left, and its caller would wait on it forever.
    """

    request = experiment.pulse.request(
        _hold_document(experiment),
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    entered_cleanup = threading.Event()
    release_cleanup = threading.Event()
    owned_cleanup = PulseSession.cleanup

    def blocking_cleanup(self, context):
        entered_cleanup.set()
        assert release_cleanup.wait(5.0)
        return owned_cleanup(self, context)

    monkeypatch.setattr(PulseSession, "cleanup", blocking_cleanup)
    handle = experiment.pulse.start(request)
    try:
        _wait_for_observation(
            experiment.pulse,
            lambda observation: observation.run.phase == "holding-pulse",
        )
        assert (
            experiment.pulse.cancel_active("replacement lane revocation proof")
            is CancelOutcome.REQUESTED
        )
        assert entered_cleanup.wait(5.0)
        assert not handle.snapshot().state.terminal
        with pytest.raises(RuntimeError, match="replacement lane is closed"):
            experiment.pulse.replace_active(request)
    finally:
        release_cleanup.set()

    assert handle.wait(5.0).state is RunState.CANCELLED


def test_failure_watchdog_follows_the_artifact_the_session_applied_last(
    experiment,
):
    """The watchdog asks about the artifact that is on the sequencer now.

    Only the session knows which artifact it applied last.  A digest captured
    when the Run was compiled names an artifact the hardware dropped at the
    first in-place replacement, and every later observation would answer "no
    failure" about an execution that no longer exists.
    """

    hold_document = _hold_document(experiment)
    request = experiment.pulse.request(
        hold_document,
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    replacement = experiment.pulse.request(
        replace(
            hold_document,
            periods=(
                replace(hold_document.periods[0], duration=200),
                *hold_document.periods[1:],
            ),
        ),
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )

    owner = PulseApplicationOwner()
    observed: list[str] = []
    replacement_applied = threading.Event()

    def waiter(_session_id, _run_id, artifact_digest, timeout):
        observed.append(artifact_digest)
        if not replacement_applied.is_set():
            time.sleep(min(float(timeout), 0.02))
            return None
        return "forced backend observer failure"

    with _service_guard(experiment._services) as services:
        port = replace(
            services.runtime.require_capability(
                request.sequencer_ref,
                "pulse.execute",
            ),
            _continuous_failure_waiter=waiter,
        )
        prepared = prepare_pulse_execution(
            request,
            pulse_port=port,
            start_run=services.runtime.start,
            on_applied=owner._record_applied,
            on_replace_ready=owner._record_replace,
        )
        handle = owner.start(prepared)

    deadline = time.monotonic() + 5.0
    while handle.snapshot().phase != "holding-pulse":
        if time.monotonic() >= deadline:
            raise AssertionError("continuous pulse never reached its HOLD phase")
        time.sleep(0.005)
    started = owner.snapshot()
    assert started is not None

    applied = owner.replace_active(replacement).result(5.0)
    assert applied.artifact_digest != started.artifact_digest
    assert owner.snapshot() == applied
    replacement_applied.set()

    deadline = time.monotonic() + 5.0
    while not handle.snapshot().state.terminal:
        if time.monotonic() >= deadline:
            raise AssertionError(
                "the watchdog never reported a failure of the replaced artifact"
            )
        time.sleep(0.005)
    assert handle.snapshot().state is RunState.FAILED
    assert observed[-1] == applied.artifact_digest


def test_a_replacement_seam_that_names_another_run_is_never_installed(
    experiment,
):
    """The lane is bound to the Run whose thread drains it.

    An owner that accepted any published lane would hand callers the seam of a
    Run it does not report as active, so ``observe_active`` would describe one
    Run while ``replace_active`` fed another.  The binding is the identity of
    the Run, not the mere existence of a lane.
    """

    hold_document = _hold_document(experiment)
    request = experiment.pulse.request(
        hold_document,
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    replacement = _hold_request_of_duration(experiment, hold_document, 200)
    owner = PulseApplicationOwner()

    with _service_guard(experiment._services) as services:
        prepared = prepare_pulse_execution(
            request,
            pulse_port=services.runtime.require_capability(
                request.sequencer_ref,
                "pulse.execute",
            ),
            start_run=services.runtime.start,
            on_applied=owner._record_applied,
            on_replace_ready=owner._record_replace,
        )
        handle = owner.start(prepared)

    try:
        deadline = time.monotonic() + 5.0
        while handle.snapshot().phase != "holding-pulse":
            if time.monotonic() >= deadline:
                raise AssertionError("continuous pulse never reached its HOLD phase")
            time.sleep(0.005)

        foreign = PulseReplacementLane()
        with pytest.raises(RuntimeError, match="without an owned Pulse Run"):
            owner._record_replace("f" * 32, foreign)
        applied = owner.replace_active(replacement).result(5.0)
        assert owner.snapshot() == applied
        assert foreign.take() is None
    finally:
        owner.cancel_active("replacement seam identity proof")
        handle.wait(5.0)


def test_a_replacement_that_stopped_the_hardware_fails_the_run_it_emptied(
    experiment,
    monkeypatch,
):
    """A replacement that left nothing on the sequencer must end its Run.

    ``PulseSession.replace`` reaches verified SAFE before it prepares the new
    artifact, so a refused prepare leaves the hardware idle and the session
    unable to serve anything.  Reporting that only through the caller's future
    would leave the Run advertising ``holding-pulse`` for an artifact no
    hardware holds, and nothing observable could ever clear the claim: the
    backend answers "no failure" about a session it already sealed.
    """

    hold_document = _hold_document(experiment)
    request = experiment.pulse.request(
        hold_document,
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    replacement = _hold_request_of_duration(experiment, hold_document, 200)

    refuse_prepare = threading.Event()
    owned_prepare = VirtualSequencerExecutionEndpoint._backend_prepare

    def refusing_prepare(self, session):
        if refuse_prepare.is_set():
            raise RuntimeError("forced sequencer prepare refusal")
        return owned_prepare(self, session)

    monkeypatch.setattr(
        VirtualSequencerExecutionEndpoint,
        "_backend_prepare",
        refusing_prepare,
    )
    handle = experiment.pulse.start(request)
    try:
        _wait_for_observation(
            experiment.pulse,
            lambda observation: observation.applied is not None
            and observation.run.phase == "holding-pulse",
        )
        refuse_prepare.set()
        future = experiment.pulse.replace_active(replacement)
        with pytest.raises(RuntimeError, match="forced sequencer prepare refusal"):
            future.result(5.0)
        terminal = handle.wait(5.0)
    finally:
        if not handle.snapshot().state.terminal:
            experiment.pulse.cancel_active("emptied hold cleanup")
            handle.wait(5.0)
    assert terminal.state is RunState.FAILED
    with pytest.raises(RuntimeError, match="replacement lane is closed"):
        experiment.pulse.replace_active(replacement)


def test_a_replacement_whose_safe_interrupt_failed_fails_the_run_it_sealed(
    experiment,
    monkeypatch,
):
    """An unverified SAFE transition must end the Run, not be waited out.

    The endpoint marks its session closed the moment it is asked to go SAFE,
    before it proves it arrived, so an interrupt that raises leaves a session
    the backend will never report a failure about again.  A Run that kept
    holding here would advertise ``holding-pulse`` over hardware in an unknown
    state, and no observation could ever contradict it: the watchdog is not
    merely quiet, it is refused before it reaches the backend at all.
    """

    hold_document = _hold_document(experiment)
    request = experiment.pulse.request(
        hold_document,
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    replacement = _hold_request_of_duration(experiment, hold_document, 200)

    refuse_safe = threading.Event()
    owned_confirm = VirtualSequencerExecutionEndpoint._backend_safe_state_confirmed

    def refusing_confirm(self, snapshot):
        # Only the transition this replacement asks for is refused; the Run's
        # own cleanup must still be able to prove SAFE.
        if refuse_safe.is_set():
            refuse_safe.clear()
            return False
        return owned_confirm(self, snapshot)

    monkeypatch.setattr(
        VirtualSequencerExecutionEndpoint,
        "_backend_safe_state_confirmed",
        refusing_confirm,
    )
    handle = experiment.pulse.start(request)
    try:
        _wait_for_observation(
            experiment.pulse,
            lambda observation: observation.applied is not None
            and observation.run.phase == "holding-pulse",
        )
        refuse_safe.set()
        future = experiment.pulse.replace_active(replacement)
        with pytest.raises(RuntimeError, match="did not reach verified SAFE state"):
            future.result(5.0)
        terminal = handle.wait(5.0)
    finally:
        refuse_safe.clear()
        if not handle.snapshot().state.terminal:
            experiment.pulse.cancel_active("sealed hold cleanup")
            handle.wait(5.0)
    assert terminal.state is RunState.FAILED
    with pytest.raises(RuntimeError, match="replacement lane is closed"):
        experiment.pulse.replace_active(replacement)


def test_a_replacement_refused_before_hardware_leaves_the_hold_running(
    experiment,
    monkeypatch,
):
    """A request the board cannot accept must not end a healthy hold.

    Compilation happens on the Run's own lane, after the lane is drained and
    before the session is asked for anything.  The session still holds exactly
    what it fired, so the honest answer is to fail the one caller who asked and
    keep serving; ending the Run would let a rejected edit stop the hardware.
    """

    hold_document = _hold_document(experiment)
    request = experiment.pulse.request(
        hold_document,
        PulseExecutionForm.CONTINUOUS_MONITOR,
    )
    replacement = _hold_request_of_duration(experiment, hold_document, 200)
    accepted = _hold_request_of_duration(experiment, hold_document, 300)

    def refusing_compile(compiled_request, *, pulse_port):
        raise ValueError("compiled pulse exceeds live sequencer memory")

    handle = experiment.pulse.start(request)
    try:
        holding = _wait_for_observation(
            experiment.pulse,
            lambda observation: observation.applied is not None
            and observation.run.phase == "holding-pulse",
        )
        applied_before = holding.applied
        monkeypatch.setattr(
            pulse_application,
            "_compile_pulse_request",
            refusing_compile,
        )
        with pytest.raises(ValueError, match="exceeds live sequencer memory"):
            experiment.pulse.replace_active(replacement).result(5.0)
        monkeypatch.undo()
        still_holding = experiment.pulse.replace_active(accepted).result(5.0)
    finally:
        experiment.pulse.cancel_active("surviving hold cleanup")
        handle.wait(5.0)

    assert still_holding.artifact_digest != applied_before.artifact_digest
    assert handle.snapshot().state is RunState.CANCELLED
