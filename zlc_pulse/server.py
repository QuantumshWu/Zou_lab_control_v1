"""Current-only remote execution contract for compiled pulse artifacts."""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
from zlc_storage import canonical_digest, decode, encode

from .artifact import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    compiled_pulse_artifact_from_tree,
    compiled_pulse_artifact_to_tree,
)
from .fpga import pack_target_ir
from .target import PulseTarget, pulse_target_to_tree


PREPARED_PULSE_REF_SCHEMA = "zlc_pulse.PreparedPulseRef/v1"
PULSE_COMPLETION_SCHEMA = "zlc_pulse.PulseCompletion/v1"


class PulseExecutionBackend(Protocol):
    """The narrow existing-hardware seam consumed by :class:`PulseExecutionService`."""

    def prepare(self, artifact: CompiledPulseArtifact) -> None: ...

    def fire(self, artifact: CompiledPulseArtifact) -> None: ...

    def wait_done(self, artifact: CompiledPulseArtifact, timeout: float | None) -> bool: ...

    def safe_state(self) -> None: ...

    def snapshot(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class PreparedPulseRef:
    connection_generation: str
    artifact_digest: str
    source_ir_digest: str
    wire_image_digest: str

    def __post_init__(self) -> None:
        _text(self.connection_generation, "connection_generation")
        for field in ("artifact_digest", "source_ir_digest", "wire_image_digest"):
            _sha256(getattr(self, field), field)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(prepared_pulse_ref_to_tree(self))


@dataclass(frozen=True)
class PulseCompletion:
    prepared_ref: PreparedPulseRef
    logical_done: bool
    completed_schedule_trigger_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_ref, PreparedPulseRef):
            raise TypeError("prepared_ref must be PreparedPulseRef")
        if type(self.logical_done) is not bool:
            raise TypeError("logical_done must be bool")
        counts = tuple(self.completed_schedule_trigger_counts)
        if len({channel for channel, _count in counts}) != len(counts):
            raise ValueError("completion trigger channels must be unique")
        for channel, count in counts:
            _text(channel, "trigger channel")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("trigger count must be a non-negative integer")
        object.__setattr__(self, "completed_schedule_trigger_counts", counts)


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
        for method in ("prepare", "fire", "wait_done", "safe_state", "snapshot"):
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
        self._state = "IDLE"
        self._artifact: CompiledPulseArtifact | None = None
        self._prepared_ref: PreparedPulseRef | None = None

    @property
    def connection_generation(self) -> str:
        return self._generation

    def renew_connection_generation(self) -> str:
        """Invalidate every prepared reference before admitting a new RPC owner."""

        with self._lock:
            if self._state not in {"IDLE", "SAFE", "SAFE_FAILED"}:
                raise RuntimeError(
                    f"cannot renew connection generation while pulse service is {self._state}"
                )
            self._generation = uuid.uuid4().hex
            self._artifact = None
            self._prepared_ref = None
            return self._generation

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            prepared = self._prepared_ref
            state = self._state
            backend = dict(self._backend.snapshot())
            return {
                "schema": "zlc_pulse.PulseExecutionSnapshot/v1",
                "connection_generation": self._generation,
                "target": pulse_target_to_tree(self._target),
                "clock_hz": self._clock_hz,
                "geometry_fingerprint": self._geometry_fingerprint,
                "state": state,
                "prepared_ref": None if prepared is None else prepared_pulse_ref_to_tree(prepared),
                "backend": backend,
            }

    def prepare(self, artifact: CompiledPulseArtifact) -> PreparedPulseRef:
        self._validate_artifact(artifact)
        with self._lock:
            if self._state not in {"IDLE", "SAFE", "DONE"}:
                raise RuntimeError(
                    f"pulse service state {self._state} requires completion or verified safe_state "
                    "before another prepare"
                )
            self._state = "PREPARING"
            try:
                self._backend.prepare(artifact)
            except BaseException as error:
                self._best_effort_safe_after_failure(error)
                self._state = "FAILED"
                self._artifact = None
                self._prepared_ref = None
                raise
            reference = PreparedPulseRef(
                self._generation,
                artifact.fingerprint,
                artifact.target_ir.fingerprint,
                artifact.wire_image.digest,
            )
            self._artifact = artifact
            self._prepared_ref = reference
            self._state = "PREPARED"
            return reference

    def fire(self, reference: PreparedPulseRef) -> None:
        with self._lock:
            artifact = self._require_prepared(reference, expected_state="PREPARED")
            self._state = "FIRING"
            try:
                self._backend.fire(artifact)
            except BaseException as error:
                self._best_effort_safe_after_failure(error)
                self._state = "FAILED"
                raise
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
            artifact = self._require_prepared(reference, expected_state="RUNNING")
            if artifact.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
                raise RuntimeError("continuous pulse execution has no logical completion; use safe_state")
            self._state = "COMPLETING"
            try:
                done = bool(self._backend.wait_done(artifact, timeout))
            except BaseException as error:
                self._best_effort_safe_after_failure(error)
                self._state = "FAILED"
                raise
            self._state = "DONE" if done else "TIMEOUT"
            return PulseCompletion(
                reference,
                done,
                tuple(
                    (schedule.channel, schedule.total)
                    for schedule in artifact.trigger_schedules
                ) if done else (),
            )

    def safe_state(self) -> None:
        with self._lock:
            try:
                self._backend.safe_state()
            except BaseException:
                self._artifact = None
                self._prepared_ref = None
                self._state = "SAFE_FAILED"
                raise
            self._artifact = None
            self._prepared_ref = None
            self._state = "SAFE"

    def _best_effort_safe_after_failure(self, primary: BaseException) -> None:
        try:
            self._backend.safe_state()
        except BaseException as safety_error:
            primary.add_note(
                "pulse backend safe_state also failed after the primary operation: "
                f"{type(safety_error).__name__}: {safety_error}"
            )

    def _validate_artifact(self, artifact: CompiledPulseArtifact) -> None:
        if not isinstance(artifact, CompiledPulseArtifact):
            raise TypeError("artifact must be CompiledPulseArtifact")
        if artifact.target_abi_fingerprint != self._target.abi_fingerprint:
            raise ValueError("compiled artifact target differs from deployed target")
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
            if artifact is None or artifact.fingerprint != reference.artifact_digest:
                raise RuntimeError("prepared pulse artifact is absent or changed")
            return artifact


def prepared_pulse_ref_to_tree(value: PreparedPulseRef) -> dict[str, object]:
    if not isinstance(value, PreparedPulseRef):
        raise TypeError("value must be PreparedPulseRef")
    return {
        "schema": PREPARED_PULSE_REF_SCHEMA,
        "connection_generation": value.connection_generation,
        "artifact_digest": value.artifact_digest,
        "source_ir_digest": value.source_ir_digest,
        "wire_image_digest": value.wire_image_digest,
    }


def prepared_pulse_ref_from_tree(tree: object) -> PreparedPulseRef:
    fields = {
        "schema",
        "connection_generation",
        "artifact_digest",
        "source_ir_digest",
        "wire_image_digest",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PreparedPulseRef has an unknown field set")
    if tree["schema"] != PREPARED_PULSE_REF_SCHEMA:
        raise ValueError("PreparedPulseRef schema differs")
    return PreparedPulseRef(
        tree["connection_generation"],
        tree["artifact_digest"],
        tree["source_ir_digest"],
        tree["wire_image_digest"],
    )


def pulse_completion_to_tree(value: PulseCompletion) -> dict[str, object]:
    if not isinstance(value, PulseCompletion):
        raise TypeError("value must be PulseCompletion")
    return {
        "schema": PULSE_COMPLETION_SCHEMA,
        "prepared_ref": prepared_pulse_ref_to_tree(value.prepared_ref),
        "logical_done": value.logical_done,
        "completed_schedule_trigger_counts": [list(item) for item in value.completed_schedule_trigger_counts],
    }


def pulse_completion_from_tree(tree: object) -> PulseCompletion:
    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "prepared_ref",
        "logical_done",
        "completed_schedule_trigger_counts",
    }:
        raise ValueError("PulseCompletion has an unknown field set")
    if tree["schema"] != PULSE_COMPLETION_SCHEMA:
        raise ValueError("PulseCompletion schema differs")
    raw_counts = tree["completed_schedule_trigger_counts"]
    if not isinstance(raw_counts, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in raw_counts
    ):
        raise ValueError("PulseCompletion trigger counts must be [channel, count] rows")
    return PulseCompletion(
        prepared_pulse_ref_from_tree(tree["prepared_ref"]),
        tree["logical_done"],
        tuple((item[0], item[1]) for item in raw_counts),
    )


def encode_artifact_message(value: CompiledPulseArtifact) -> bytes:
    return encode(compiled_pulse_artifact_to_tree(value))


def decode_artifact_message(payload: bytes) -> CompiledPulseArtifact:
    return compiled_pulse_artifact_from_tree(decode(payload))


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
                if active_connection:
                    raise RuntimeError("pulse execution service already has an active control owner")
                active_connection.append(conn)
            try:
                service.renew_connection_generation()
            except BaseException:
                with connection_lock:
                    if active_connection and active_connection[0] is conn:
                        active_connection.clear()
                raise
            self._owner_connection = conn

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

        def exposed_current_safe_state(self):
            self._require_owner()
            service.safe_state()
            return True

    server = ThreadedServer(
        RPyCCurrentPulseService,
        hostname=host,
        port=int(port),
        protocol_config={"allow_public_attrs": True, "sync_request_timeout": None},
    )
    if start:
        server.start()
    return server


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


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
