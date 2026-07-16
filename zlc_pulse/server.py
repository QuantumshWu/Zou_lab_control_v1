"""Current-only remote execution contract for compiled pulse artifacts."""

from __future__ import annotations

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
    PulseExecutionForm,
    admit_compiled_pulse_payload_size,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
)
from .fpga import pack_target_ir
from .deployment import (
    resident_scan_point_capacity,
    validate_resident_scan_capacity,
)
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
from .target import PulseTarget, pulse_target_to_tree
from .validation import validate_target_ir_for_target


PREPARED_PULSE_REF_SCHEMA = "zlc_pulse.PreparedPulseRef"
PULSE_COMPLETION_SCHEMA = "zlc_pulse.PulseCompletion"


class PulseExecutionBackend(Protocol):
    """The narrow existing-hardware seam consumed by :class:`PulseExecutionService`."""

    def prepare(self, artifact: CompiledPulseArtifact) -> None: ...

    def fire(self, artifact: CompiledPulseArtifact) -> None: ...

    def await_completion(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float | None,
    ) -> PulseBackendCompletion | None: ...

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
        target: PulseTarget,
        *,
        clock_hz: float,
        backend: PulseExecutionBackend,
        params: StreamerParams | None = None,
        connection_generation: str | None = None,
    ) -> None:
        if not isinstance(target, PulseTarget):
            raise TypeError("target must be PulseTarget")
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
        self._target = target
        self._clock_hz = float(clock_hz)
        self._backend = backend
        self._params = params or StreamerParams()
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
            if self._state not in {"IDLE", "SAFE"}:
                raise RuntimeError(
                    f"cannot renew connection generation while pulse service is {self._state}"
                )
            self._generation = uuid.uuid4().hex
            self._artifact = None
            self._prepared_ref = None
            self._completion = None
            return self._generation

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        prepared = self._prepared_ref
        return {
            "schema": "zlc_pulse.PulseExecutionSnapshot",
            "connection_generation": self._generation,
            "target": pulse_target_to_tree(self._target),
            "clock_hz": self._clock_hz,
            "geometry_fingerprint": self._geometry_fingerprint,
            "resident_scan_point_capacity": resident_scan_point_capacity(
                self._params
            ),
            "state": self._state,
            "prepared_ref": (
                None if prepared is None else prepared_pulse_ref_to_tree(prepared)
            ),
            "backend": dict(self._backend.snapshot()),
        }

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
            if artifact.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
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

    def safe_state(self) -> None:
        self._safe_state(expected_generation=None)

    def _safe_state(
        self,
        *,
        expected_generation: str | None,
    ) -> tuple[dict[str, object], BaseException | None]:
        with self._safe_state_lock:
            return self._safe_state_owned(expected_generation=expected_generation)

    def _safe_state_owned(
        self,
        *,
        expected_generation: str | None,
    ) -> tuple[dict[str, object], BaseException | None]:
        with self._lock:
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                raise RuntimeError("interrupt connection generation is stale")
            if self._state == "SAFE" and self._prepared_ref is None:
                return self._snapshot_locked(), None
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

    def safe_state_for_generation(self, generation: str) -> dict[str, object]:
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
        validate_resident_scan_capacity(artifact, self._params)
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

    connection_lock = threading.Lock()
    active_connection: list[object] = []

    class RPyCCurrentPulseService(rpyc.Service):
        def on_connect(self, conn):
            with connection_lock:
                owns_connection = not active_connection
                if owns_connection:
                    active_connection.append(conn)
            self._owner_connection = conn if owns_connection else None
            if not owns_connection:
                return
            try:
                service.renew_connection_generation()
            except BaseException:
                with connection_lock:
                    if active_connection and active_connection[0] is conn:
                        active_connection.clear()
                raise

        def on_disconnect(self, conn):
            with connection_lock:
                owns_connection = bool(active_connection and active_connection[0] is conn)
            if not owns_connection:
                return
            try:
                service.safe_state()
            except Exception:
                pass
            finally:
                with connection_lock:
                    if active_connection and active_connection[0] is conn:
                        active_connection.clear()

        def _require_owner(self):
            connection = getattr(self, "_owner_connection", None)
            with connection_lock:
                if connection is None or not active_connection or active_connection[0] is not connection:
                    raise RuntimeError("RPC connection does not own the pulse execution service")

        def exposed_current_snapshot(self):
            self._require_owner()
            return encode(service.snapshot())

        def exposed_current_prepare(self, artifact_bytes):
            self._require_owner()
            admit_compiled_pulse_payload_size(len(artifact_bytes))
            return encode_prepared_ref_message(
                service.prepare(decode_artifact_message(bytes(artifact_bytes)))
            )

        def exposed_current_fire(self, reference_bytes):
            self._require_owner()
            service.fire(decode_prepared_ref_message(bytes(reference_bytes)))
            return True

        def exposed_current_complete(self, reference_bytes, timeout=None):
            self._require_owner()
            return encode_completion_message(
                service.complete(
                    decode_prepared_ref_message(bytes(reference_bytes)),
                    timeout=timeout,
                )
            )

        def exposed_current_interrupt_safe_state(self, connection_generation):
            return encode(
                service.safe_state_for_generation(str(connection_generation))
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
    "PreparedPulseRef",
    "PulseCompletion",
    "PulseExecutionBackend",
    "PulseExecutionService",
    "decode_artifact_message",
    "decode_completion_message",
    "decode_prepared_ref_message",
    "encode_artifact_message",
    "encode_completion_message",
    "encode_prepared_ref_message",
    "prepared_pulse_ref_from_tree",
    "prepared_pulse_ref_to_tree",
    "pulse_completion_from_tree",
    "pulse_completion_to_tree",
    "serve_pulse_execution_service",
]
