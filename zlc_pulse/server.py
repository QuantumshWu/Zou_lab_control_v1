"""Current-only remote execution contract for compiled pulse artifacts."""

from __future__ import annotations

import logging
import math
import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
from zlc_storage import (
    canonical_text as _text,
    decode,
    encode,
    sha256_text as _sha256,
)

from .artifact import (
    CompiledPulseArtifact,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
)
from .fpga import pack_target_ir
from .evidence import (
    AutonomousTableTerminalEvidence,
    PostTerminalTailEvidence,
    PulseBackendCompletion,
    StaticOnceTerminalEvidence,
    hardware_terminal_evidence_from_tree,
    hardware_terminal_evidence_to_tree,
    post_terminal_tail_evidence_from_tree,
    post_terminal_tail_evidence_to_tree,
    validate_backend_completion_for_artifact,
    validate_backend_completion_intrinsic,
)
from .manifest import (
    PulseTargetManifest,
    pulse_target_manifest_from_tree,
    pulse_target_manifest_to_tree,
)
from .target import PulseTarget
from .validation import validate_target_ir_for_target


PREPARED_PULSE_REF_SCHEMA = "zlc_pulse.PreparedPulseRef"
PULSE_COMPLETION_SCHEMA = "zlc_pulse.PulseCompletion"
PULSE_CONTINUOUS_FAILURE_SCHEMA = "zlc_pulse.ContinuousFailure"
PULSE_SERVER_SNAPSHOT_SCHEMA = "zlc_pulse.PulseExecutionSnapshot"


def _typed_backend_observation(snapshot: object) -> dict[str, object]:
    """Rename one private session observation for the public value owner."""

    required = {
        "schema",
        "state",
        "prepared_artifact_digest",
        "scan_points",
        "last_confirmed_cursor",
        "cursor_sample_count",
        "underflow_observed",
        "safe_status_word",
        "safe_clock_enable_words",
    }
    if not isinstance(snapshot, dict) or not required.issubset(snapshot):
        raise ValueError("pulse backend snapshot lacks current typed observation fields")
    if snapshot["schema"] != "zlc_pulse.DeployedStreamerSessionSnapshot":
        raise ValueError("pulse backend snapshot schema differs")
    return {
        "physical_state": snapshot["state"],
        "physical_prepared_artifact_digest": snapshot[
            "prepared_artifact_digest"
        ],
        "physical_scan_point_count": snapshot["scan_points"],
        "physical_scan_cursor": snapshot["last_confirmed_cursor"],
        "physical_cursor_sample_count": snapshot["cursor_sample_count"],
        "physical_underflow_observed": snapshot["underflow_observed"],
        "safe_status_word": snapshot["safe_status_word"],
        "safe_clock_enable_words": snapshot["safe_clock_enable_words"],
    }


class _PulseExecutionBackend(Protocol):
    """The narrow existing-hardware seam consumed by :class:`PulseExecutionService`."""

    def prepare(self, artifact: CompiledPulseArtifact) -> None: ...

    def fire(self, artifact: CompiledPulseArtifact) -> None: ...

    def await_completion(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float | None,
    ) -> PulseBackendCompletion | None: ...

    def wait_continuous_failure(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float,
    ) -> str | None: ...

    def safe_state(self) -> None: ...

    def request_interrupt(self) -> None:
        """Request interruption without waiting for the ordinary backend I/O owner.

        Implementations must be thread-safe and non-blocking relative to
        ``prepare``/``fire``/``await_completion``.  The service calls this
        before it waits for the backend-operation gate needed by physical SAFE.
        """

        ...

    def snapshot(self) -> dict[str, object]:
        """Return a thread-safe, non-blocking observation of backend state."""

        ...


@dataclass(frozen=True)
class PreparedPulseRef:
    connection_generation: str
    artifact_digest: str

    def __post_init__(self) -> None:
        _text(self.connection_generation, "connection_generation")
        _sha256(self.artifact_digest, "artifact_digest")


@dataclass(frozen=True)
class PulseServerSnapshot:
    """One exact public observation of service and deployed physical state."""

    connection_generation: str
    manifest: PulseTargetManifest
    clock_hz: float
    geometry_fingerprint: int
    state: str
    prepared_ref: PreparedPulseRef | None
    physical_state: str
    physical_prepared_artifact_digest: str | None
    physical_scan_point_count: int
    physical_scan_cursor: int | None
    physical_cursor_sample_count: int
    physical_underflow_observed: bool
    safe_status_word: int | None
    safe_clock_enable_words: tuple[int, ...] | None

    def __post_init__(self) -> None:
        _text(self.connection_generation, "connection_generation")
        if not isinstance(self.manifest, PulseTargetManifest):
            raise TypeError("PulseServerSnapshot manifest has another type")
        if (
            isinstance(self.clock_hz, bool)
            or not isinstance(self.clock_hz, (int, float))
            or not math.isfinite(float(self.clock_hz))
            or self.clock_hz <= 0
        ):
            raise ValueError("PulseServerSnapshot clock is invalid")
        object.__setattr__(self, "clock_hz", float(self.clock_hz))
        if (
            isinstance(self.geometry_fingerprint, bool)
            or not isinstance(self.geometry_fingerprint, int)
            or not 0 <= self.geometry_fingerprint <= 0xFFFFFFFF
        ):
            raise ValueError("PulseServerSnapshot geometry fingerprint is invalid")
        for name in ("state", "physical_state"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"PulseServerSnapshot {name} is invalid")
        if self.prepared_ref is not None and not isinstance(
            self.prepared_ref,
            PreparedPulseRef,
        ):
            raise TypeError("PulseServerSnapshot prepared_ref has another type")
        if (
            self.prepared_ref is not None
            and self.prepared_ref.connection_generation
            != self.connection_generation
        ):
            raise ValueError(
                "PulseServerSnapshot prepared_ref belongs to another generation"
            )
        if self.physical_prepared_artifact_digest is not None:
            _sha256(
                self.physical_prepared_artifact_digest,
                "physical_prepared_artifact_digest",
            )
        for name in ("physical_scan_point_count", "physical_cursor_sample_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"PulseServerSnapshot {name} must be non-negative")
        cursor = self.physical_scan_cursor
        if self.physical_cursor_sample_count == 0:
            if cursor is not None:
                raise ValueError("unsampled pulse progress cannot publish a cursor")
        elif (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not 0 <= cursor < self.physical_scan_point_count
        ):
            raise ValueError("sampled pulse cursor is outside the scan table")
        if not isinstance(self.physical_underflow_observed, bool):
            raise TypeError("physical_underflow_observed must be bool")
        status = self.safe_status_word
        if status is not None and (
            isinstance(status, bool)
            or not isinstance(status, int)
            or not 0 <= status <= 0xFFFFFFFF
        ):
            raise ValueError("safe_status_word must be an unsigned 32-bit integer")
        clocks = self.safe_clock_enable_words
        if clocks is not None:
            clocks = tuple(clocks)
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 0xFFFFFFFF
                for value in clocks
            ):
                raise ValueError("safe_clock_enable_words must be unsigned words")
            expected_words = (len(self.manifest.target.raw_lanes) + 31) // 32
            if len(clocks) != expected_words:
                raise ValueError(
                    "safe_clock_enable_words cardinality differs from target geometry"
                )
            object.__setattr__(self, "safe_clock_enable_words", clocks)
        if (status is None) != (clocks is None):
            raise ValueError("SAFE readback facts must be present or absent together")
        if status is not None and (
            status != 0
            or any(clocks)
            or self.physical_state != "SAFE"
            or self.physical_prepared_artifact_digest is not None
        ):
            raise ValueError("SAFE readback facts do not prove current physical SAFE")
        if self.physical_state == "SAFE" and not self.safe_readback_confirmed:
            raise ValueError("physical SAFE state lacks exact readback")
        if self.state == "SAFE" and (
            not self.safe_readback_confirmed or self.prepared_ref is not None
        ):
            raise ValueError("server SAFE requires an exact physical SAFE readback")

    @property
    def target(self) -> PulseTarget:
        return self.manifest.target

    @property
    def safe_readback_confirmed(self) -> bool:
        return (
            self.safe_status_word == 0
            and self.safe_clock_enable_words is not None
            and not any(self.safe_clock_enable_words)
            and self.physical_state == "SAFE"
            and self.physical_prepared_artifact_digest is None
        )


@dataclass(frozen=True)
class PulseCompletion:
    prepared_ref: PreparedPulseRef
    hardware_terminal: StaticOnceTerminalEvidence | AutonomousTableTerminalEvidence
    post_terminal_tail: PostTerminalTailEvidence
    expected_trigger_counts_from_completed_schedule: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_ref, PreparedPulseRef):
            raise TypeError("prepared_ref must be PreparedPulseRef")
        validate_backend_completion_intrinsic(
            PulseBackendCompletion(self.hardware_terminal, self.post_terminal_tail)
        )
        counts = tuple(self.expected_trigger_counts_from_completed_schedule)
        if len({channel for channel, _count in counts}) != len(counts):
            raise ValueError("completion trigger channels must be unique")
        for channel, count in counts:
            _text(channel, "trigger channel")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("trigger count must be a non-negative integer")
        object.__setattr__(
            self,
            "expected_trigger_counts_from_completed_schedule",
            counts,
        )


class PulseExecutionService:
    """Single-owner prepare/FIRE/complete service for one deployed pulse target.

    The service never accepts PulseDocument or source-table data.  Compilation is
    complete before the RPC boundary; the server validates and executes the exact
    content-addressed artifact it received.
    """

    def __init__(
        self,
        manifest: PulseTargetManifest,
        *,
        clock_hz: float,
        backend: _PulseExecutionBackend,
        params: StreamerParams,
        connection_generation: str | None = None,
    ) -> None:
        if not isinstance(manifest, PulseTargetManifest):
            raise TypeError("manifest must be PulseTargetManifest")
        if not isinstance(clock_hz, (int, float)) or not math.isfinite(float(clock_hz)) or clock_hz <= 0:
            raise ValueError("clock_hz must be finite and positive")
        for method in (
            "prepare",
            "fire",
            "await_completion",
            "safe_state",
            "request_interrupt",
            "snapshot",
        ):
            if not callable(getattr(backend, method, None)):
                raise TypeError(f"pulse backend is missing {method}()")
        self._manifest = manifest
        self._target = manifest.target
        self._clock_hz = float(clock_hz)
        self._backend = backend
        if not isinstance(params, StreamerParams):
            raise TypeError("params must be StreamerParams")
        self._params = params
        self._geometry_fingerprint = build_fingerprint(self._params) & 0xFFFFFFFF
        self._generation = connection_generation or uuid.uuid4().hex
        _text(self._generation, "connection_generation")
        self._lock = threading.RLock()
        self._safe_state_lock = threading.Lock()
        self._backend_operation_lock = threading.Lock()
        self._state = "IDLE"
        self._artifact: CompiledPulseArtifact | None = None
        self._prepared_ref: PreparedPulseRef | None = None
        self._completion: PulseCompletion | None = None
        self._operation_epoch = 0

    @property
    def connection_generation(self) -> str:
        return self._generation

    def renew_connection_generation(self) -> str:
        """Invalidate every prepared reference before admitting a new RPC owner."""

        with self._lock:
            if self._state != "SAFE":
                raise RuntimeError(
                    f"cannot renew connection generation while pulse service is {self._state}"
                )
            self._snapshot_locked()
            self._generation = uuid.uuid4().hex
            self._artifact = None
            self._prepared_ref = None
            self._completion = None
            return self._generation

    def snapshot(self) -> PulseServerSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> PulseServerSnapshot:
        prepared = self._prepared_ref
        physical = _typed_backend_observation(self._backend.snapshot())
        return PulseServerSnapshot(
            connection_generation=self._generation,
            manifest=self._manifest,
            clock_hz=self._clock_hz,
            geometry_fingerprint=self._geometry_fingerprint,
            state=self._state,
            prepared_ref=prepared,
            **physical,
        )

    def prepare(self, artifact: CompiledPulseArtifact) -> PreparedPulseRef:
        with self._lock:
            if self._state not in {"IDLE", "SAFE", "DONE"}:
                raise RuntimeError(
                    f"pulse service state {self._state} requires completion or verified safe_state "
                    "before another prepare"
                )
            prior_state = self._state
            operation_epoch = self._operation_epoch
            self._state = "VALIDATING"
        try:
            self._validate_artifact(artifact)
        except BaseException:
            with self._lock:
                if (
                    operation_epoch == self._operation_epoch
                    and self._state == "VALIDATING"
                ):
                    self._state = prior_state
            raise
        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or self._state != "VALIDATING"
            ):
                raise RuntimeError(
                    "pulse prepare validation was superseded by interrupt-to-safe"
                )
            self._state = "PREPARING"
        try:
            with self._backend_operation_lock:
                with self._lock:
                    if (
                        operation_epoch != self._operation_epoch
                        or self._state != "PREPARING"
                    ):
                        raise RuntimeError(
                            "pulse prepare was superseded before backend admission"
                        )
                self._backend.prepare(artifact)
        except BaseException as error:
            with self._lock:
                superseded = operation_epoch != self._operation_epoch
                if not superseded:
                    self._state = "FAILED"
                    self._artifact = None
                    self._prepared_ref = None
                    self._completion = None
            if not superseded:
                self._best_effort_safe_after_failure(error)
            raise
        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or self._state != "PREPARING"
            ):
                raise RuntimeError(
                    "pulse prepare was superseded by an interrupt-to-safe operation"
                )
            reference = PreparedPulseRef(
                self._generation,
                artifact.fingerprint,
            )
            self._artifact = artifact
            self._prepared_ref = reference
            self._completion = None
            self._state = "PREPARED"
            return reference

    def fire(self, reference: PreparedPulseRef) -> None:
        with self._lock:
            artifact = self._require_prepared(reference, expected_state="PREPARED")
            self._state = "FIRING"
            operation_epoch = self._operation_epoch
        try:
            with self._backend_operation_lock:
                with self._lock:
                    if (
                        operation_epoch != self._operation_epoch
                        or self._state != "FIRING"
                    ):
                        raise RuntimeError(
                            "pulse FIRE was superseded before backend admission"
                        )
                self._backend.fire(artifact)
        except BaseException as error:
            with self._lock:
                superseded = operation_epoch != self._operation_epoch
                if not superseded:
                    self._state = "FAILED"
            if not superseded:
                self._best_effort_safe_after_failure(error)
            raise
        with self._lock:
            if operation_epoch != self._operation_epoch or self._state != "FIRING":
                raise RuntimeError(
                    "pulse FIRE was superseded by an interrupt-to-safe operation"
                )
            self._state = "RUNNING"

    def complete(
        self,
        reference: PreparedPulseRef,
        *,
        timeout: float | None,
    ) -> PulseCompletion:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive or None")
        with self._lock:
            if self._state == "DONE":
                if reference != self._prepared_ref or self._completion is None:
                    raise RuntimeError(
                        "completed pulse reference differs from cached completion"
                    )
                return self._completion
            if self._state not in {"RUNNING", "TIMEOUT"}:
                raise RuntimeError(
                    f"pulse service state is {self._state}, expected RUNNING or TIMEOUT"
                )
            artifact = self._require_prepared(
                reference,
                expected_state=self._state,
            )
            if artifact.target_ir.repeat_forever:
                raise RuntimeError("continuous pulse execution has no logical completion; use safe_state")
            self._state = "COMPLETING"
            operation_epoch = self._operation_epoch
        try:
            with self._backend_operation_lock:
                with self._lock:
                    if (
                        operation_epoch != self._operation_epoch
                        or self._state != "COMPLETING"
                    ):
                        raise RuntimeError(
                            "pulse completion was superseded before backend admission"
                        )
                backend_completion = self._backend.await_completion(
                    artifact,
                    timeout,
                )
            if backend_completion is not None:
                validate_backend_completion_for_artifact(
                    backend_completion,
                    artifact,
                )
        except BaseException as error:
            with self._lock:
                superseded = operation_epoch != self._operation_epoch
                if not superseded:
                    self._state = "FAILED"
                    self._artifact = None
                    self._prepared_ref = None
                    self._completion = None
            if not superseded:
                self._best_effort_safe_after_failure(error)
            raise
        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or self._state != "COMPLETING"
            ):
                raise RuntimeError(
                    "pulse completion was superseded by an interrupt-to-safe operation"
                )
            if backend_completion is None:
                self._state = "TIMEOUT"
            else:
                self._state = "DONE"
                completion = PulseCompletion(
                    reference,
                    backend_completion.hardware_terminal,
                    backend_completion.post_terminal_tail,
                    tuple(
                        (schedule.channel, schedule.total)
                        for schedule in artifact.trigger_schedules
                    ),
                )
                self._completion = completion
        if backend_completion is None:
            raise TimeoutError("pulse backend did not reach a validated terminal before timeout")
        return completion

    def wait_continuous_failure(
        self,
        reference: PreparedPulseRef,
        *,
        timeout: float,
    ) -> str | None:
        """Wait briefly for a backend-owned continuous execution failure."""

        wait_seconds = float(timeout)
        if not math.isfinite(wait_seconds) or wait_seconds <= 0.0:
            raise ValueError("continuous failure wait timeout must be positive")
        with self._lock:
            artifact = self._require_prepared(reference, expected_state="RUNNING")
            if not artifact.target_ir.repeat_forever:
                raise RuntimeError(
                    "continuous failure wait requires a cyclic prepared artifact"
                )
            operation_epoch = self._operation_epoch
        waiter = getattr(self._backend, "wait_continuous_failure", None)
        if not callable(waiter):
            raise RuntimeError(
                "pulse backend lacks continuous failure notification"
            )
        try:
            failure = waiter(artifact, wait_seconds)
        except BaseException as error:
            with self._lock:
                superseded = operation_epoch != self._operation_epoch
                if not superseded:
                    self._state = "FAILED"
            if not superseded:
                self._best_effort_safe_after_failure(error)
            raise
        with self._lock:
            if operation_epoch != self._operation_epoch:
                return None
            if self._state != "RUNNING":
                raise RuntimeError(
                    f"continuous failure wait resumed while service is {self._state}"
                )
            if failure is None:
                return None
            _text(failure, "continuous pulse failure")
            self._state = "FAILED"
            return failure

    def safe_state(self) -> None:
        self._safe_state(expected_generation=None)

    def _safe_state(
        self,
        *,
        expected_generation: str | None,
    ) -> tuple[PulseServerSnapshot, BaseException | None]:
        with self._safe_state_lock:
            return self._safe_state_owned(expected_generation=expected_generation)

    def _safe_state_owned(
        self,
        *,
        expected_generation: str | None,
    ) -> tuple[PulseServerSnapshot, BaseException | None]:
        with self._lock:
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                raise RuntimeError("interrupt connection generation is stale")
            if self._state == "SAFE" and self._prepared_ref is None:
                try:
                    current_safe = self._snapshot_locked()
                except Exception:
                    current_safe = None
                if (
                    current_safe is not None
                    and current_safe.safe_readback_confirmed
                ):
                    return current_safe, None
            self._operation_epoch += 1
            operation_epoch = self._operation_epoch
            self._state = "INTERRUPTING"
        interrupt_error: BaseException | None = None
        try:
            self._backend.request_interrupt()
        except BaseException as error:
            interrupt_error = error
        try:
            with self._backend_operation_lock:
                self._backend.safe_state()
        except BaseException as error:
            if interrupt_error is not None:
                error.add_note(
                    "non-blocking interrupt request also failed before safe_state: "
                    f"{type(interrupt_error).__name__}: {interrupt_error}"
                )
            with self._lock:
                if operation_epoch == self._operation_epoch:
                    self._artifact = None
                    self._prepared_ref = None
                    self._completion = None
                    self._state = "SAFE_FAILED"
            raise
        with self._lock:
            if operation_epoch != self._operation_epoch:
                raise RuntimeError("safe_state operation was superseded")
            self._artifact = None
            self._prepared_ref = None
            self._completion = None
            self._state = "SAFE"
            return self._snapshot_locked(), interrupt_error

    def safe_state_for_generation(self, generation: str) -> PulseServerSnapshot:
        """Authorize a separate abort connection without granting normal control."""

        _text(generation, "connection_generation")
        snapshot, _interrupt_error = self._safe_state(
            expected_generation=generation
        )
        return snapshot

    def _best_effort_safe_after_failure(self, primary: BaseException) -> None:
        try:
            _snapshot, interrupt_error = self._safe_state(
                expected_generation=None
            )
        except BaseException as safety_error:
            primary.add_note(
                "pulse backend safe_state also failed after the primary operation: "
                f"{type(safety_error).__name__}: {safety_error}"
            )
        else:
            if interrupt_error is not None:
                primary.add_note(
                    "pulse backend request_interrupt failed, but safe_state succeeded: "
                    f"{type(interrupt_error).__name__}: {interrupt_error}"
                )

    def _validate_artifact(self, artifact: CompiledPulseArtifact) -> None:
        if not isinstance(artifact, CompiledPulseArtifact):
            raise TypeError("artifact must be CompiledPulseArtifact")
        if artifact.target_abi_fingerprint != self._target.abi_fingerprint:
            raise ValueError("compiled artifact target differs from deployed target")
        validate_target_ir_for_target(artifact.target_ir, self._target)
        if artifact.target_ir.clock_hz != self._clock_hz:
            raise ValueError("compiled artifact clock differs from deployed clock")
        if artifact.wire_image.geometry_fingerprint != self._geometry_fingerprint:
            raise ValueError("compiled artifact geometry differs from deployed geometry")
        if artifact.wire_image != pack_target_ir(artifact.target_ir, self._params):
            raise ValueError("compiled artifact wire image differs from deterministic TargetIR packing")

    def _require_prepared(
        self,
        reference: PreparedPulseRef,
        *,
        expected_state: str,
    ) -> CompiledPulseArtifact:
        if not isinstance(reference, PreparedPulseRef):
            raise TypeError("reference must be PreparedPulseRef")
        with self._lock:
            if self._state != expected_state:
                raise RuntimeError(f"pulse service state is {self._state}, expected {expected_state}")
            if reference != self._prepared_ref or reference.connection_generation != self._generation:
                raise RuntimeError("prepared pulse reference is stale or belongs to another connection")
            artifact = self._artifact
            if artifact is None:
                raise RuntimeError("prepared pulse artifact is absent")
            return artifact


def prepared_pulse_ref_to_tree(value: PreparedPulseRef) -> dict[str, object]:
    if not isinstance(value, PreparedPulseRef):
        raise TypeError("value must be PreparedPulseRef")
    return {
        "schema": PREPARED_PULSE_REF_SCHEMA,
        "connection_generation": value.connection_generation,
        "artifact_digest": value.artifact_digest,
    }


def prepared_pulse_ref_from_tree(tree: object) -> PreparedPulseRef:
    fields = {
        "schema",
        "connection_generation",
        "artifact_digest",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PreparedPulseRef has an unknown field set")
    if tree["schema"] != PREPARED_PULSE_REF_SCHEMA:
        raise ValueError("PreparedPulseRef schema differs")
    return PreparedPulseRef(
        tree["connection_generation"],
        tree["artifact_digest"],
    )


def pulse_server_snapshot_to_tree(
    value: PulseServerSnapshot,
) -> dict[str, object]:
    if not isinstance(value, PulseServerSnapshot):
        raise TypeError("value must be PulseServerSnapshot")
    return {
        "schema": PULSE_SERVER_SNAPSHOT_SCHEMA,
        "connection_generation": value.connection_generation,
        "manifest": pulse_target_manifest_to_tree(value.manifest),
        "clock_hz": value.clock_hz,
        "geometry_fingerprint": value.geometry_fingerprint,
        "state": value.state,
        "prepared_ref": (
            None
            if value.prepared_ref is None
            else prepared_pulse_ref_to_tree(value.prepared_ref)
        ),
        "physical_state": value.physical_state,
        "physical_prepared_artifact_digest": (
            value.physical_prepared_artifact_digest
        ),
        "physical_scan_point_count": value.physical_scan_point_count,
        "physical_scan_cursor": value.physical_scan_cursor,
        "physical_cursor_sample_count": value.physical_cursor_sample_count,
        "physical_underflow_observed": value.physical_underflow_observed,
        "safe_status_word": value.safe_status_word,
        "safe_clock_enable_words": (
            None
            if value.safe_clock_enable_words is None
            else list(value.safe_clock_enable_words)
        ),
    }


def pulse_server_snapshot_from_tree(tree: object) -> PulseServerSnapshot:
    fields = {
        "schema",
        "connection_generation",
        "manifest",
        "clock_hz",
        "geometry_fingerprint",
        "state",
        "prepared_ref",
        "physical_state",
        "physical_prepared_artifact_digest",
        "physical_scan_point_count",
        "physical_scan_cursor",
        "physical_cursor_sample_count",
        "physical_underflow_observed",
        "safe_status_word",
        "safe_clock_enable_words",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulseExecutionSnapshot has an unknown field set")
    if tree["schema"] != PULSE_SERVER_SNAPSHOT_SCHEMA:
        raise ValueError("PulseExecutionSnapshot schema differs")
    raw_ref = tree["prepared_ref"]
    raw_clocks = tree["safe_clock_enable_words"]
    if raw_clocks is not None and not isinstance(raw_clocks, list):
        raise TypeError(
            "PulseExecutionSnapshot SAFE clock words must be a list or null"
        )
    return PulseServerSnapshot(
        connection_generation=tree["connection_generation"],
        manifest=pulse_target_manifest_from_tree(tree["manifest"]),
        clock_hz=tree["clock_hz"],
        geometry_fingerprint=tree["geometry_fingerprint"],
        state=tree["state"],
        prepared_ref=(
            None if raw_ref is None else prepared_pulse_ref_from_tree(raw_ref)
        ),
        physical_state=tree["physical_state"],
        physical_prepared_artifact_digest=(
            tree["physical_prepared_artifact_digest"]
        ),
        physical_scan_point_count=tree["physical_scan_point_count"],
        physical_scan_cursor=tree["physical_scan_cursor"],
        physical_cursor_sample_count=tree["physical_cursor_sample_count"],
        physical_underflow_observed=tree["physical_underflow_observed"],
        safe_status_word=tree["safe_status_word"],
        safe_clock_enable_words=(
            None if raw_clocks is None else tuple(raw_clocks)
        ),
    )


def pulse_completion_to_tree(value: PulseCompletion) -> dict[str, object]:
    if not isinstance(value, PulseCompletion):
        raise TypeError("value must be PulseCompletion")
    return {
        "schema": PULSE_COMPLETION_SCHEMA,
        "prepared_ref": prepared_pulse_ref_to_tree(value.prepared_ref),
        "hardware_terminal": hardware_terminal_evidence_to_tree(
            value.hardware_terminal
        ),
        "post_terminal_tail": post_terminal_tail_evidence_to_tree(
            value.post_terminal_tail
        ),
        "expected_trigger_counts_from_completed_schedule": [
            list(item)
            for item in value.expected_trigger_counts_from_completed_schedule
        ],
    }


def pulse_completion_from_tree(tree: object) -> PulseCompletion:
    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "prepared_ref",
        "hardware_terminal",
        "post_terminal_tail",
        "expected_trigger_counts_from_completed_schedule",
    }:
        raise ValueError("PulseCompletion has an unknown field set")
    if tree["schema"] != PULSE_COMPLETION_SCHEMA:
        raise ValueError("PulseCompletion schema differs")
    raw_counts = tree["expected_trigger_counts_from_completed_schedule"]
    if not isinstance(raw_counts, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in raw_counts
    ):
        raise ValueError("PulseCompletion trigger counts must be [channel, count] rows")
    return PulseCompletion(
        prepared_pulse_ref_from_tree(tree["prepared_ref"]),
        hardware_terminal_evidence_from_tree(tree["hardware_terminal"]),
        post_terminal_tail_evidence_from_tree(tree["post_terminal_tail"]),
        tuple((item[0], item[1]) for item in raw_counts),
    )


def encode_artifact_message(value: CompiledPulseArtifact) -> bytes:
    return encode_compiled_pulse_artifact(value)


def decode_artifact_message(payload: bytes) -> CompiledPulseArtifact:
    return decode_compiled_pulse_artifact(payload)


def encode_prepared_ref_message(value: PreparedPulseRef) -> bytes:
    return encode(prepared_pulse_ref_to_tree(value))


def decode_prepared_ref_message(payload: bytes) -> PreparedPulseRef:
    return prepared_pulse_ref_from_tree(decode(payload))


def encode_completion_message(value: PulseCompletion) -> bytes:
    return encode(pulse_completion_to_tree(value))


def decode_completion_message(payload: bytes) -> PulseCompletion:
    return pulse_completion_from_tree(decode(payload))


def encode_continuous_failure_message(value: str | None) -> bytes:
    if value is not None:
        _text(value, "continuous pulse failure")
    return encode(
        {
            "schema": PULSE_CONTINUOUS_FAILURE_SCHEMA,
            "failure": value,
        }
    )


def decode_continuous_failure_message(payload: bytes) -> str | None:
    tree = decode(payload)
    if not isinstance(tree, dict) or set(tree) != {"schema", "failure"}:
        raise ValueError("ContinuousFailure has an unknown field set")
    if tree["schema"] != PULSE_CONTINUOUS_FAILURE_SCHEMA:
        raise ValueError("ContinuousFailure schema differs")
    failure = tree["failure"]
    if failure is not None:
        _text(failure, "continuous pulse failure")
    return failure


@dataclass
class AdmittedPulseSession:
    """The one admitted client: a control channel and its paired abort channel.

    Roles are declared, never inferred from arrival order, so the server can
    distinguish one client's second channel from a second client entirely.
    """

    control: object
    peer: str
    generation: str = ""
    abort: object | None = None
    abort_peer: str = ""


def describe_rpyc_peer(conn: object) -> str:
    """Name a connection for operator-facing admission records."""

    channel = getattr(conn, "_channel", None)
    stream = getattr(channel, "stream", None)
    socket = getattr(stream, "sock", None)
    try:
        host, port = socket.getpeername()[:2]
    except Exception:
        return "unknown peer"
    return f"{host}:{port}"


def serve_pulse_execution_service(
    service: PulseExecutionService,
    *,
    host: str = "0.0.0.0",
    port: int = 18861,
    start: bool = True,
):
    """Expose only current canonical messages over RPyC."""

    if not isinstance(service, PulseExecutionService):
        raise TypeError("service must be PulseExecutionService")
    try:
        import rpyc
        from rpyc.utils.server import ThreadedServer
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("pulse server requires rpyc on the FPGA computer") from exc

    session_lock = threading.Lock()
    admitted: list[AdmittedPulseSession] = []

    def held_session() -> AdmittedPulseSession | None:
        with session_lock:
            return admitted[0] if admitted else None

    class RPyCCurrentPulseService(rpyc.Service):
        """Admit one paired session; every other connection is refused by name.

        A connection carries no authority from having arrived: it declares a
        role, and the server admits at most one control channel plus the single
        abort channel that presents that control channel's generation.
        """

        def on_connect(self, conn):
            self._conn = conn

        def on_disconnect(self, conn):
            with session_lock:
                held = admitted[0] if admitted else None
                if held is None:
                    return
                if held.abort is conn:
                    held.abort = None
                    role = "abort"
                elif held.control is conn:
                    admitted.clear()
                    role = "control"
                else:
                    return
            logging.info(
                "%s channel from %s closed", role, describe_rpyc_peer(conn)
            )
            if role != "control":
                return
            # The physical state belongs to the session, not to the socket:
            # losing the control channel ends both together.
            try:
                service.safe_state()
            except Exception:
                logging.exception("SAFE after control channel loss failed")

        def exposed_open_control_session(self):
            """Admit this connection as the one control channel and re-arm SAFE."""

            conn = self._conn
            peer = describe_rpyc_peer(conn)
            with session_lock:
                if admitted:
                    held = admitted[0]
                    raise RuntimeError(
                        "pulse control is already held by "
                        f"{held.peer}; close that client before opening another"
                    )
                admitted.append(AdmittedPulseSession(conn, peer))
            try:
                # Admission is a live recovery attempt.  A prior disconnect SAFE
                # failure blocks only until this serialized owner obtains a
                # current receipt; no failure state survives successful readback.
                service.safe_state()
                generation = service.renew_connection_generation()
            except BaseException:
                with session_lock:
                    if admitted and admitted[0].control is conn:
                        admitted.clear()
                logging.exception("control channel admission from %s failed", peer)
                raise
            with session_lock:
                if not admitted or admitted[0].control is not conn:
                    raise RuntimeError(
                        "pulse control admission was revoked while it was opening"
                    )
                admitted[0].generation = generation
            logging.info("control channel opened by %s (generation %s)", peer, generation)
            return generation

        def exposed_attach_abort_channel(self, connection_generation):
            """Pair this connection to the live session as its only abort channel.

            The abort path exists because ``complete`` occupies the control
            channel for the whole run; pairing it explicitly is what lets the
            server tell one client's second channel from a second client.
            """

            conn = self._conn
            peer = describe_rpyc_peer(conn)
            generation = str(connection_generation)
            with session_lock:
                held = admitted[0] if admitted else None
                if held is None or not held.generation:
                    raise RuntimeError("no open pulse control session to attach to")
                if held.control is conn:
                    raise RuntimeError(
                        "the control channel cannot also serve as its abort channel"
                    )
                if held.generation != generation:
                    raise RuntimeError(
                        "abort channel presented another session's generation"
                    )
                if held.abort is not None:
                    raise RuntimeError(
                        f"this pulse session already has an abort channel ({held.abort_peer})"
                    )
                held.abort = conn
                held.abort_peer = peer
            logging.info("abort channel attached by %s (generation %s)", peer, generation)
            return True

        def _require_control(self) -> None:
            held = held_session()
            if held is not None and held.control is getattr(self, "_conn", None):
                return
            raise RuntimeError(
                "this connection is not the pulse control channel"
                + (f"; control is held by {held.peer}" if held is not None else "")
            )

        def _require_abort(self) -> str:
            held = held_session()
            if held is None or held.abort is not getattr(self, "_conn", None):
                raise RuntimeError(
                    "this connection is not the abort channel of the open pulse session"
                )
            return held.generation

        def exposed_current_snapshot(self):
            self._require_control()
            return encode(pulse_server_snapshot_to_tree(service.snapshot()))

        def exposed_current_prepare(self, artifact_bytes):
            self._require_control()
            return encode_prepared_ref_message(
                service.prepare(decode_artifact_message(bytes(artifact_bytes)))
            )

        def exposed_current_fire(self, reference_bytes):
            self._require_control()
            service.fire(decode_prepared_ref_message(bytes(reference_bytes)))
            return True

        def exposed_current_complete(self, reference_bytes, timeout=None):
            self._require_control()
            return encode_completion_message(
                service.complete(
                    decode_prepared_ref_message(bytes(reference_bytes)),
                    timeout=timeout,
                )
            )

        def exposed_current_wait_continuous_failure(
            self,
            reference_bytes,
            timeout,
        ):
            self._require_control()
            return encode_continuous_failure_message(
                service.wait_continuous_failure(
                    decode_prepared_ref_message(bytes(reference_bytes)),
                    timeout=float(timeout),
                )
            )

        def exposed_current_interrupt_safe_state(self):
            # The attached channel is the authority; a generation argument would
            # only re-admit the forgeable identity this pairing replaced.
            return encode(
                pulse_server_snapshot_to_tree(
                    service.safe_state_for_generation(self._require_abort())
                )
            )

    server = ThreadedServer(
        RPyCCurrentPulseService,
        hostname=host,
        port=int(port),
        protocol_config={"allow_public_attrs": True, "sync_request_timeout": None},
    )
    if start:
        server.start()
    return server


__all__ = [
    "PREPARED_PULSE_REF_SCHEMA",
    "PULSE_COMPLETION_SCHEMA",
    "PULSE_CONTINUOUS_FAILURE_SCHEMA",
    "PULSE_SERVER_SNAPSHOT_SCHEMA",
    "PreparedPulseRef",
    "PulseCompletion",
    "PulseExecutionService",
    "PulseServerSnapshot",
    "decode_artifact_message",
    "decode_completion_message",
    "decode_continuous_failure_message",
    "decode_prepared_ref_message",
    "encode_artifact_message",
    "encode_completion_message",
    "encode_continuous_failure_message",
    "encode_prepared_ref_message",
    "prepared_pulse_ref_from_tree",
    "prepared_pulse_ref_to_tree",
    "pulse_completion_from_tree",
    "pulse_completion_to_tree",
    "pulse_server_snapshot_from_tree",
    "pulse_server_snapshot_to_tree",
    "serve_pulse_execution_service",
]
