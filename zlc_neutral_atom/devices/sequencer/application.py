"""Declarative Sequencer seam shared by the public API and Workbench."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
import math
import threading
from typing import Callable

from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    PostSafetyContext,
    RunContext,
    RunHandle,
    RunPlan,
    RunSnapshot,
)
from zlc_neutral_atom.devices.sequencer.port import (
    BoundPulsePort,
    ContinuousPulseExecutionRequest,
    FinitePulseExecutionRequest,
    PulseScanProgress,
    PulseSession,
    PulseTerminalEvidenceKind,
)
from zlc_pulse import (
    apply_api_values_to_authoring_document,
    CompiledPulseArtifact,
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    PulseTargetManifest,
    bind_pulse_document_target,
    compile_pulse_artifact,
)
from zlc_pulse.authoring import resolve_api_parameters
from zlc_pulse.scan_execution import materialize_scan_sweeps
from zlc_storage import canonical_text, positive_real


_AUTO_FINITE_TIMEOUT_FLOOR_SECONDS = 30.0
_AUTO_FINITE_TIMEOUT_FIXED_MARGIN_SECONDS = 5.0
_AUTO_FINITE_TIMEOUT_SCALE = 1.25
_TERMINAL_CALL_MARGIN_SECONDS = 2.0
_CONTINUOUS_FAILURE_WAIT_SLICE_SECONDS = 0.1
_CONTINUOUS_FORMS = (
    PulseExecutionForm.CONTINUOUS_MONITOR,
    PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
)
_SCAN_FORMS = (
    PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
)


def _ordered_api_values(
    document: PulseDocument,
    values: tuple[tuple[str, int | float], ...],
) -> tuple[tuple[str, int | float], ...]:
    frozen = tuple(values)
    if any(not isinstance(item, tuple) or len(item) != 2 for item in frozen):
        raise TypeError("api_values must contain (parameter_id, value) tuples")
    for parameter_id, _value in frozen:
        canonical_text(parameter_id, "API value parameter_id")
    expected = tuple(
        parameter.parameter_id for parameter in document.api_parameters
    )
    if tuple(parameter_id for parameter_id, _value in frozen) != expected:
        raise ValueError(
            "Pulse Run API values must exactly cover declared parameters "
            "in declaration order"
        )
    return frozen


def _resolve_finite_timeout_seconds(
    request: "PulseRunRequest",
    artifact: CompiledPulseArtifact,
    *,
    max_blocking_call_seconds: float,
) -> float | None:
    if request.execution_form in _CONTINUOUS_FORMS:
        return None
    physical_duration = (
        artifact.target_ir.duration_seconds
        + artifact.max_configured_output_delay_ticks / artifact.target_ir.clock_hz
    )
    blocking_limit = positive_real(
        max_blocking_call_seconds,
        "max_blocking_call_seconds",
    )
    if physical_duration + _TERMINAL_CALL_MARGIN_SECONDS >= blocking_limit:
        raise ValueError(
            "compiled pulse duration exceeds the sequencer completion-call window; "
            "reconnect with a larger transport timeout"
        )
    if request.timeout_seconds is None:
        return max(
            _AUTO_FINITE_TIMEOUT_FLOOR_SECONDS,
            physical_duration * _AUTO_FINITE_TIMEOUT_SCALE
            + _AUTO_FINITE_TIMEOUT_FIXED_MARGIN_SECONDS,
        )
    explicit = positive_real(request.timeout_seconds, "timeout_seconds")
    if explicit <= physical_duration:
        raise ValueError(
            "finite pulse timeout must exceed compiled physical duration"
        )
    return explicit


@dataclass(frozen=True, slots=True)
class PulseTargetDescriptor:
    """Read-only target facts; it contains no hardware drive capability."""

    sequencer_ref: DeviceRef
    manifest: PulseTargetManifest
    clock_hz: float
    geometry_fingerprint: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        if not isinstance(self.manifest, PulseTargetManifest):
            raise TypeError("manifest must be PulseTargetManifest")
        clock_hz = positive_real(self.clock_hz, "clock_hz")
        object.__setattr__(self, "clock_hz", clock_hz)
        if (
            isinstance(self.geometry_fingerprint, bool)
            or not isinstance(self.geometry_fingerprint, int)
            or not 0 <= self.geometry_fingerprint <= 0xFFFFFFFF
        ):
            raise ValueError("geometry_fingerprint must be an unsigned 32-bit integer")

    @property
    def target(self) -> PulseTarget:
        return self.manifest.target

    @property
    def time_step_ns(self) -> float:
        return 1e9 / self.clock_hz


@dataclass(frozen=True, slots=True)
class PulseRunRequest:
    """One immutable authored intent plus explicit execution-time values."""

    document: PulseDocument
    execution_form: PulseExecutionForm
    sequencer_ref: DeviceRef
    timeout_seconds: float | None
    api_values: tuple[tuple[str, int | float], ...] = ()
    scan_sweep_count: int = 1

    @property
    def requires_cancellation(self) -> bool:
        return self.execution_form in _CONTINUOUS_FORMS

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if self.execution_form is PulseExecutionForm.STATIC_REFERENCE_POINT:
            raise ValueError("STATIC_REFERENCE_POINT is a preview form, not a hardware Run")
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        api_values = _ordered_api_values(self.document, self.api_values)
        resolved = resolve_api_parameters(self.document, dict(api_values))
        if resolved.api_parameters:
            raise RuntimeError("Pulse Run retained unresolved API declarations")
        object.__setattr__(self, "api_values", api_values)

        sweep_count = self.scan_sweep_count
        if isinstance(sweep_count, bool) or not isinstance(sweep_count, int):
            raise TypeError("scan_sweep_count must be an integer")
        if sweep_count < 1:
            raise ValueError("scan_sweep_count must be positive")
        if self.execution_form in _SCAN_FORMS:
            if self.document.scan_table is None:
                raise ValueError("autonomous scan execution requires a frozen scan table")
            if (
                self.execution_form
                is PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS
                and sweep_count != 1
            ):
                raise ValueError(
                    "continuous autonomous scan cycles its frozen table and does not "
                    "accept a finite outer sweep count"
                )
        elif sweep_count != 1:
            raise ValueError("scan_sweep_count applies only to autonomous scan execution")

        if self.requires_cancellation:
            if self.timeout_seconds is not None:
                raise ValueError("continuous pulse execution ends by cancellation, not timeout")
        else:
            if self.timeout_seconds is not None:
                object.__setattr__(
                    self,
                    "timeout_seconds",
                    positive_real(self.timeout_seconds, "timeout_seconds"),
                )


@dataclass(frozen=True, slots=True)
class PulseRunDescriptor:
    name: str
    sequencer_role: str
    execution_form: PulseExecutionForm
    artifact_digest: str
    logical_duration_seconds: float
    scan_point_count: int
    resource_claim: str
    timeout_seconds: float | None

    def __post_init__(self) -> None:
        canonical_text(self.name, "pulse run name")
        canonical_text(self.sequencer_role, "sequencer_role")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        canonical_text(self.artifact_digest, "artifact_digest")
        duration = float(self.logical_duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("logical_duration_seconds must be finite and positive")
        object.__setattr__(self, "logical_duration_seconds", duration)
        if (
            isinstance(self.scan_point_count, bool)
            or not isinstance(self.scan_point_count, int)
            or self.scan_point_count < 0
        ):
            raise ValueError("scan_point_count must be a non-negative integer")
        canonical_text(self.resource_claim, "resource_claim")
        if self.execution_form in _CONTINUOUS_FORMS:
            if self.timeout_seconds is not None:
                raise ValueError("continuous PulseRunDescriptor cannot carry a timeout")
        else:
            object.__setattr__(
                self,
                "timeout_seconds",
                positive_real(self.timeout_seconds, "timeout_seconds"),
            )


@dataclass(frozen=True, slots=True)
class PulseRunResult:
    run_id: str
    execution_form: PulseExecutionForm
    artifact_digest: str
    terminal_evidence_kind: PulseTerminalEvidenceKind

    def __post_init__(self) -> None:
        canonical_text(self.run_id, "run_id")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        canonical_text(self.artifact_digest, "artifact_digest")
        if not isinstance(self.terminal_evidence_kind, PulseTerminalEvidenceKind):
            raise TypeError("terminal_evidence_kind must be PulseTerminalEvidenceKind")


@dataclass(frozen=True, slots=True)
class AppliedPulseSnapshot:
    """The authored intent and exact document accepted by ``session.prepare``."""

    run_id: str
    source_document: PulseDocument
    api_values: tuple[tuple[str, int | float], ...]
    scan_sweep_count: int
    execution_document: PulseDocument
    execution_form: PulseExecutionForm
    artifact_digest: str
    authoring_document: PulseDocument = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        canonical_text(self.run_id, "run_id")
        if not isinstance(self.source_document, PulseDocument):
            raise TypeError("source_document must be PulseDocument")
        if not isinstance(self.execution_document, PulseDocument):
            raise TypeError("execution_document must be PulseDocument")
        api_values = _ordered_api_values(self.source_document, self.api_values)
        object.__setattr__(self, "api_values", api_values)
        object.__setattr__(
            self,
            "authoring_document",
            apply_api_values_to_authoring_document(
                self.source_document,
                dict(api_values),
            ),
        )
        if (
            isinstance(self.scan_sweep_count, bool)
            or not isinstance(self.scan_sweep_count, int)
            or self.scan_sweep_count < 1
        ):
            raise ValueError("scan_sweep_count must be a positive integer")
        if self.execution_document.api_parameters:
            raise ValueError("an applied execution document cannot retain API parameters")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        canonical_text(self.artifact_digest, "artifact_digest")

    @property
    def source_document_digest(self) -> str:
        return self.source_document.fingerprint

    @property
    def execution_document_digest(self) -> str:
        return self.execution_document.fingerprint

    @property
    def authoring_document_digest(self) -> str:
        """Fingerprint of source intent with the exact accepted API values."""

        return self.authoring_document.fingerprint

    def matches_authoring_document(self, document: PulseDocument) -> bool:
        """Whether ``document`` is exactly the authoring intent now applied."""

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        return self.authoring_document_digest == document.fingerprint


@dataclass(frozen=True, slots=True)
class PulseRunObservation:
    """Immutable observation of the most recently started pulse Run.

    ``applied`` is the only account of what the Run put on the sequencer: it is
    republished by every in-place replacement.  The intent the Run was admitted
    with is deliberately absent, because it stops describing the hardware at the
    first replacement and nothing can update a copy of it here.
    """

    run: RunSnapshot
    applied: AppliedPulseSnapshot | None

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunSnapshot):
            raise TypeError("run must be RunSnapshot")
        if self.applied is not None:
            if not isinstance(self.applied, AppliedPulseSnapshot):
                raise TypeError("applied must be AppliedPulseSnapshot or None")
            if self.applied.run_id != self.run.run_id.value:
                raise ValueError("applied snapshot belongs to another Run")


class PreparedPulseExecution:
    """One-shot command hiding the generation-bound Port and RunPlan."""

    __slots__ = (
        "_descriptor",
        "_lock",
        "_plan",
        "_request",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        plan: RunPlan,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: PulseRunDescriptor,
        request: PulseRunRequest,
    ) -> None:
        if not isinstance(plan, RunPlan):
            raise TypeError("plan must be RunPlan")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        if not isinstance(descriptor, PulseRunDescriptor):
            raise TypeError("descriptor must be PulseRunDescriptor")
        if not isinstance(request, PulseRunRequest):
            raise TypeError("request must be PulseRunRequest")
        self._plan = plan
        self._start_run = start_run
        self._descriptor = descriptor
        self._request = request
        self._lock = threading.Lock()
        self._started = False

    @property
    def descriptor(self) -> PulseRunDescriptor:
        return self._descriptor

    def start(self) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedPulseExecution is one-shot")
            self._started = True
        return self._start_run(
            self._plan.with_lifecycle(
                owner=self,
                preemptible=self._request.execution_form in _CONTINUOUS_FORMS,
            )
        )


class PulseReplacementLane:
    """The one lane through which a live Run accepts in-place replacements.

    Admission and drain are the same fact, so they share one lock: closing is
    performed by the drain itself.  Without that, a request accepted after the
    only consumer stopped would return a promise nobody can ever complete, and
    the caller would wait on it for the rest of the process' life.
    """

    __slots__ = ("_closed_reason", "_lock", "_pending")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[tuple[PulseRunRequest, Future]] = []
        self._closed_reason: str | None = None

    def submit(self, request: PulseRunRequest) -> Future:
        """Accept one replacement, or refuse it because nobody can apply it."""

        if not isinstance(request, PulseRunRequest):
            raise TypeError("replacement request must be PulseRunRequest")
        if request.execution_form is not PulseExecutionForm.CONTINUOUS_MONITOR:
            raise ValueError("pulse replacement requires CONTINUOUS_MONITOR")
        future: Future = Future()
        with self._lock:
            if self._closed_reason is not None:
                raise RuntimeError(
                    f"pulse replacement lane is closed: {self._closed_reason}"
                )
            self._pending.append((request, future))
        return future

    def take(self) -> tuple[PulseRunRequest, Future] | None:
        with self._lock:
            return self._pending.pop(0) if self._pending else None

    def close(self, error: BaseException) -> None:
        with self._lock:
            if self._closed_reason is None:
                self._closed_reason = f"{type(error).__name__}: {error}"
            drained = tuple(self._pending)
            self._pending.clear()
        for _request, future in drained:
            if not future.done():
                future.set_exception(error)


class PulseApplicationOwner:
    """Single in-process owner of applied pulse identity and the latest Run."""

    __slots__ = (
        "_active_applied",
        "_active_descriptor",
        "_active_handle",
        "_active_progress_reader",
        "_active_replacement_lane",
        "_last_applied",
        "_lock",
        "_pending_applied",
        "_start_in_progress",
        "_start_lock",
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._start_in_progress = False
        self._pending_applied = None
        self._last_applied: AppliedPulseSnapshot | None = None
        self._active_handle: RunHandle | None = None
        self._active_applied: AppliedPulseSnapshot | None = None
        self._active_descriptor: PulseRunDescriptor | None = None
        self._active_progress_reader: Callable[[], PulseScanProgress] | None = None
        self._active_replacement_lane: tuple[str, PulseReplacementLane] | None = None

    def snapshot(self) -> AppliedPulseSnapshot | None:
        with self._lock:
            return self._last_applied

    def observe_active(self) -> PulseRunObservation | None:
        while True:
            with self._lock:
                handle = self._active_handle
            if handle is None:
                return None
            run = handle.snapshot()
            with self._lock:
                if self._active_handle is handle:
                    return PulseRunObservation(run, self._active_applied)

    def replace_active(self, request: PulseRunRequest) -> Future:
        """Queue one applied-pulse replacement on the existing Run owner.

        The returned future completes only after the Run's sequencer lane has
        performed the existing SAFE -> prepare -> FIRE seam.  It never starts
        a second Run and never performs transport I/O on the caller thread.
        Whether a replacement is still possible is answered by the lane of the
        Run that is active right now, not by a cached copy here: only the Run
        knows when it stopped consuming, and it revokes the lane in the same
        breath.  A lane published by a Run that never became the active one is
        therefore refused rather than fed work nobody observes.
        """

        with self._lock:
            handle = self._active_handle
            bound = self._active_replacement_lane
        if handle is None or bound is None or bound[0] != handle.run_id.value:
            raise RuntimeError("active pulse has no replacement seam")
        return bound[1].submit(request)

    def cancel_active(
        self,
        reason: str = "user requested pulse stop",
    ) -> CancelOutcome | None:
        with self._lock:
            handle = self._active_handle
        return None if handle is None else handle.cancel(reason)

    def observe_scan_progress(self) -> PulseScanProgress | None:
        """Observe the active continuous scan without acquiring drive authority.

        The reader published with each applied fact reads the live session, so
        it is the only thing asked which artifact and how many points are on
        the sequencer.  The prepared descriptor answers only until the Run has
        applied anything, when it is still the whole truth.
        """

        while True:
            with self._lock:
                handle = self._active_handle
                descriptor = self._active_descriptor
                reader = self._active_progress_reader
            if handle is None:
                return None
            if descriptor is None:
                raise RuntimeError("pulse application lost its active execution identity")
            run = handle.snapshot()
            if reader is None:
                progress = PulseScanProgress.unavailable(
                    run_id=run.run_id.value,
                    artifact_digest=descriptor.artifact_digest,
                    point_count=descriptor.scan_point_count,
                    backend_state=run.state.value,
                    reason="pulse scan has not reached hardware prepare",
                )
            else:
                progress = reader()
            with self._lock:
                if self._active_handle is handle:
                    if progress.run_id != run.run_id.value:
                        raise RuntimeError(
                            "pulse scan progress belongs to another active Run"
                        )
                    return progress

    def start(self, prepared: PreparedPulseExecution) -> RunHandle:
        if not isinstance(prepared, PreparedPulseExecution):
            raise TypeError("prepared must be PreparedPulseExecution")
        with self._start_lock:
            with self._lock:
                self._start_in_progress = True
                self._pending_applied = None
            try:
                handle = prepared.start()
            except BaseException:
                with self._lock:
                    self._start_in_progress = False
                    self._pending_applied = None
                raise
            with self._lock:
                pending = self._pending_applied
                mismatch = (
                    pending is not None
                    and pending[0].run_id != handle.run_id.value
                )
                if mismatch:
                    self._pending_applied = None
                    self._start_in_progress = False
                else:
                    self._active_handle = handle
                    self._active_descriptor = prepared.descriptor
                    self._active_applied = None if pending is None else pending[0]
                    self._active_progress_reader = (
                        None if pending is None else pending[1]
                    )
                    if pending is not None:
                        self._last_applied = pending[0]
                    self._pending_applied = None
                    self._start_in_progress = False
            if mismatch:
                handle.cancel("applied pulse identity mismatch")
                raise RuntimeError("applied pulse belongs to another starting Run")
            return handle

    def _record_applied(
        self,
        snapshot: AppliedPulseSnapshot,
        progress_reader: Callable[[], PulseScanProgress],
    ) -> None:
        if not isinstance(snapshot, AppliedPulseSnapshot):
            raise TypeError("snapshot must be AppliedPulseSnapshot")
        if not callable(progress_reader):
            raise TypeError("progress_reader must be callable")
        with self._lock:
            handle = self._active_handle
            if handle is None or handle.run_id.value != snapshot.run_id:
                if not self._start_in_progress:
                    raise RuntimeError(
                        "prepared pulse Run is not owned by this application"
                    )
                if self._pending_applied is not None:
                    raise RuntimeError("starting pulse produced more than one applied fact")
                self._pending_applied = (snapshot, progress_reader)
                return
            self._last_applied = snapshot
            self._active_applied = snapshot
            self._active_progress_reader = progress_reader

    def _record_replace(self, run_id: str, lane: PulseReplacementLane) -> None:
        """Install the one replacement lane, bound to the Run that owns it."""

        canonical_text(run_id, "run_id")
        if not isinstance(lane, PulseReplacementLane):
            raise TypeError("replacement lane must be PulseReplacementLane")
        with self._lock:
            handle = self._active_handle
            if handle is None or handle.run_id.value != run_id:
                if not self._start_in_progress:
                    raise RuntimeError(
                        "replacement seam was published without an owned Pulse Run"
                    )
            self._active_replacement_lane = (run_id, lane)


def _compile_pulse_request(
    request: PulseRunRequest,
    *,
    pulse_port: BoundPulsePort,
) -> tuple[
    PulseDocument,
    ContinuousPulseExecutionRequest | FinitePulseExecutionRequest,
    float | None,
]:
    """Compile one request through the sole existing pulse seam.

    Initial admission and an in-place hold/step replacement intentionally use
    this same function.  The caller decides whether the resulting execution is
    admitted as a new Run or applied to the already-owned session.
    """

    if not isinstance(request, PulseRunRequest):
        raise TypeError("request must be PulseRunRequest")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    document = bind_pulse_document_target(
        request.document,
        pulse_port.capability.target,
    )
    document = resolve_api_parameters(document, dict(request.api_values))
    if document.api_parameters:
        raise RuntimeError("Pulse Run retained unresolved API declarations")
    if request.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        document = materialize_scan_sweeps(document, request.scan_sweep_count)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=pulse_port.capability.clock_hz,
        execution_form=request.execution_form,
        live_target=pulse_port.capability.target,
    )
    timeout_seconds = _resolve_finite_timeout_seconds(
        request,
        artifact,
        max_blocking_call_seconds=pulse_port.capability.max_blocking_call_seconds,
    )
    if request.execution_form in _CONTINUOUS_FORMS:
        execution: ContinuousPulseExecutionRequest | FinitePulseExecutionRequest = (
            ContinuousPulseExecutionRequest(document, artifact)
        )
    else:
        execution = FinitePulseExecutionRequest(document, artifact)
    return document, execution, timeout_seconds


def prepare_pulse_execution(
    request: PulseRunRequest,
    *,
    pulse_port: BoundPulsePort,
    start_run: Callable[[RunPlan], RunHandle],
    on_applied: (
        Callable[[AppliedPulseSnapshot, Callable[[], PulseScanProgress]], None]
        | None
    ) = None,
    on_replace_ready: Callable[[str, PulseReplacementLane], None] | None = None,
) -> PreparedPulseExecution:
    """Bind, compile, and freeze one pulse-only Run without touching hardware."""

    if not isinstance(request, PulseRunRequest):
        raise TypeError("request must be PulseRunRequest")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    document, execution, timeout_seconds = _compile_pulse_request(
        request,
        pulse_port=pulse_port,
    )
    plan = _compile_pulse_run_plan(
        request,
        pulse_port=pulse_port,
        execution=execution,
        timeout_seconds=timeout_seconds,
        on_applied=on_applied,
        on_replace_ready=on_replace_ready,
    )
    descriptor = PulseRunDescriptor(
        f"Pulse {document.name}",
        request.sequencer_ref.role,
        request.execution_form,
        execution.artifact.fingerprint,
        execution.artifact.target_ir.duration_seconds,
        len(execution.artifact.target_ir.scan_points),
        str(pulse_port.resource_claim.key),
        timeout_seconds,
    )
    return PreparedPulseExecution(plan, start_run, descriptor, request)


def _compile_pulse_run_plan(
    request: PulseRunRequest,
    *,
    pulse_port: BoundPulsePort,
    execution: FinitePulseExecutionRequest | ContinuousPulseExecutionRequest,
    timeout_seconds: float | None,
    on_applied: (
        Callable[[AppliedPulseSnapshot, Callable[[], PulseScanProgress]], None]
        | None
    ),
    on_replace_ready: Callable[[str, PulseReplacementLane], None] | None,
) -> RunPlan[PulseSession, PulseRunResult, PulseRunResult]:
    replacement_lane = PulseReplacementLane()
    replacement_cache: dict[
        tuple[str, PulseExecutionForm, tuple[tuple[str, int | float], ...]],
        tuple[PulseDocument, ContinuousPulseExecutionRequest],
    ] = {}

    def preflight(_context: RunContext) -> PulseSession:
        return pulse_port.open_session(execution)

    def drive(context: RunContext, session: PulseSession) -> PulseRunResult:
        context.checkpoint()
        session.prepare(context)

        def observe_applied_scan() -> PulseScanProgress:
            # One read of the session answers both facts about the artifact on
            # the sequencer.  Capturing them here instead would describe the
            # artifact this Run began with and stop naming the hardware at the
            # first in-place replacement.
            applied_artifact = session.applied_artifact
            return pulse_port.observe_scan_progress(
                session_id=session.session_id,
                run_id=context.run_id.value,
                artifact_digest=applied_artifact.fingerprint,
                point_count=len(applied_artifact.target_ir.scan_points),
            )

        def publish_applied(
            source: PulseRunRequest,
            execution_document: PulseDocument,
        ) -> AppliedPulseSnapshot:
            applied = AppliedPulseSnapshot(
                context.run_id.value,
                source.document,
                source.api_values,
                source.scan_sweep_count,
                execution_document,
                source.execution_form,
                session.artifact_digest,
            )
            if on_applied is not None:
                on_applied(applied, observe_applied_scan)
            return applied

        publish_applied(request, execution.document)
        if (
            on_replace_ready is not None
            and isinstance(execution, ContinuousPulseExecutionRequest)
        ):
            on_replace_ready(context.run_id.value, replacement_lane)
        context.checkpoint()
        session.fire(context)
        if isinstance(execution, ContinuousPulseExecutionRequest):
            context.set_phase("holding-pulse")
            while True:
                context.checkpoint()
                queued = replacement_lane.take()
                if queued is not None:
                    replacement_request, replacement_future = queued
                    try:
                        replacement_key = (
                            replacement_request.document.fingerprint,
                            replacement_request.execution_form,
                            replacement_request.api_values,
                        )
                        cached = replacement_cache.get(replacement_key)
                        if cached is None:
                            replacement_document, compiled, _timeout = (
                                _compile_pulse_request(
                                    replacement_request,
                                    pulse_port=pulse_port,
                                )
                            )
                            if not isinstance(
                                compiled,
                                ContinuousPulseExecutionRequest,
                            ):
                                raise TypeError(
                                    "pulse replacement did not compile as continuous"
                                )
                            replacement_execution = compiled
                            replacement_cache[replacement_key] = (
                                replacement_document,
                                replacement_execution,
                            )
                        else:
                            replacement_document, replacement_execution = cached
                        session.replace(context, replacement_execution)
                        replacement_future.set_result(
                            publish_applied(
                                replacement_request,
                                replacement_document,
                            )
                        )
                    except BaseException as error:
                        replacement_future.set_exception(error)
                        if not session.holds_hardware:
                            # ``replace`` interrupts the fired artifact before
                            # it prepares the new one, so a refusal past that
                            # point leaves the sequencer holding nothing.  Only
                            # the session can say so, and it is asked here
                            # rather than inferred from the error: a request
                            # this lane refused before the session was touched
                            # is the caller's failure, not the Run's.  Staying
                            # in the loop would advertise "holding-pulse" over
                            # idle hardware forever, because the backend
                            # answers "no failure" about the session it sealed.
                            raise
                    continue
                failure = pulse_port.wait_continuous_failure(
                    session_id=session.session_id,
                    run_id=context.run_id.value,
                    artifact_digest=session.artifact_digest,
                    timeout=_CONTINUOUS_FAILURE_WAIT_SLICE_SECONDS,
                )
                # Cancellation wins races with a failure notification because its
                # interrupt path has already requested immediate hardware SAFE.
                context.checkpoint()
                if failure is not None:
                    raise RuntimeError(
                        f"continuous pulse backend failed: {failure}"
                    )
        terminal = session.complete(context)
        return PulseRunResult(
            context.run_id.value,
            request.execution_form,
            session.artifact_digest,
            terminal.evidence_kind,
        )

    def execute(context: RunContext, session: PulseSession) -> PulseRunResult:
        ended: BaseException | None = None
        try:
            return drive(context, session)
        except BaseException as error:
            ended = error
            raise
        finally:
            # This frame is where consumption stops, so this is where the lane
            # stops admitting work.  Revoking it in a later lifecycle callback
            # would leave a window in which a caller is handed a promise that
            # nobody is left to complete, and the terminal state that would
            # warn them is still several hardware round trips away.
            replacement_lane.close(
                ended
                if ended is not None
                else RuntimeError("pulse Run stopped consuming replacements")
            )

    def cleanup(
        context: RunContext,
        session: PulseSession | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        return (
            pulse_port.verify_idle(context)
            if session is None
            else session.cleanup(context)
        )

    def finalize(
        context: PostSafetyContext,
        result: PulseRunResult,
    ) -> PulseRunResult:
        if not isinstance(result, PulseRunResult):
            raise TypeError("pulse Run finalized an unexpected result")
        if result.run_id != context.run_id.value:
            raise ValueError("pulse Run result belongs to another Run")
        return result

    return RunPlan(
        name=f"Pulse {request.document.name}",
        resource_claims=(pulse_port.resource_claim,),
        bound_devices=(pulse_port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=pulse_port.interrupt_operations,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "AppliedPulseSnapshot",
    "PreparedPulseExecution",
    "PulseReplacementLane",
    "PulseRunDescriptor",
    "PulseRunObservation",
    "PulseRunRequest",
    "PulseRunResult",
    "PulseTargetDescriptor",
    "prepare_pulse_execution",
]
