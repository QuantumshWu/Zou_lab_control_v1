"""Bounded exact acquisition streams and non-blocking monitor taps."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Callable, Generic, Protocol, TypeVar

from zlc_data import DataBlock, DataPatch, StreamGenerationId, Value
from zlc_storage import (
    canonical_digest,
    canonical_text as _canonical_text,
    decode,
    encode,
    finite_real,
    nonnegative_integer as _nonnegative_int,
    positive_integer as _positive_int,
    sha256_digest,
    sha256_text as _digest,
)


PayloadT = TypeVar("PayloadT")
_EOS_TOKEN = object()
_DELIVERY_TOKEN = object()
_CURSOR_TOKEN = object()
_RESERVATION_TOKEN = object()
_MONITOR_TOKEN = object()
_STREAM_TOKEN = object()
_PRODUCER_TOKEN = object()
_READINESS_TOKEN = object()
_MAX_ARTIFACT_REFERENCE_BYTES = 64 * 1024
_MAX_DIRECT_ARTIFACT_INPUTS = 32
_MAX_DIRECT_ARTIFACT_INPUT_BYTES = 1024 * 1024
_MAX_EXACT_PROCESSOR_STAGES = 64
_MAX_CHAIN_ARTIFACT_INPUT_OCCURRENCES = 256
_MAX_CHAIN_ARTIFACT_INPUT_BYTES = 4 * 1024 * 1024
_MAX_TRACE_CAUSATION_REFS = 1 + _MAX_DIRECT_ARTIFACT_INPUTS


class PayloadContract(Protocol[PayloadT]):
    fingerprint: str
    max_retained_nbytes: int

    def snapshot(self, payload: PayloadT) -> PayloadT: ...

    def retained_nbytes(self, payload: PayloadT) -> int: ...

    def digest(self, payload: PayloadT) -> str: ...


class JoinKeyContract(Protocol):
    """Generation owner whose snapshot validates and freezes one join key."""

    fingerprint: str

    def snapshot(self, key: object) -> object: ...


def _contains_materialization(value: object, seen: set[int] | None = None) -> bool:
    if isinstance(value, (DataBlock, DataPatch)):
        return True
    if type(value) is Value:
        return False
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return False
    identity = id(value)
    seen = set() if seen is None else seen
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(
            _contains_materialization(item, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_materialization(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_materialization(getattr(value, field.name), seen)
            for field in fields(value)
        )
    return False


@dataclass(frozen=True, order=True)
class StreamId:
    value: str

    def __post_init__(self) -> None:
        _canonical_text(self.value, "StreamId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class EventId:
    value: str

    def __post_init__(self) -> None:
        _canonical_text(self.value, "EventId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EventRef:
    stream_id: StreamId
    generation: StreamGenerationId
    sequence: int
    event_id: EventId
    payload_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        if not isinstance(self.event_id, EventId):
            raise TypeError("event_id must be EventId")
        _digest(self.payload_digest, "payload_digest")


def event_id_for_sequence(
    stream_id: StreamId,
    generation: StreamGenerationId,
    sequence: int,
) -> EventId:
    """Return the one owner-defined identity for a generated stream event."""

    if not isinstance(stream_id, StreamId):
        raise TypeError("stream_id must be StreamId")
    if not isinstance(generation, StreamGenerationId):
        raise TypeError("generation must be StreamGenerationId")
    sequence = _nonnegative_int(sequence, "sequence")
    return EventId(f"{stream_id.value}:{generation.value}:{sequence}")


def event_ref_to_tree(reference: EventRef) -> dict[str, object]:
    """Canonical field projection owned beside ``EventRef`` itself."""

    if not isinstance(reference, EventRef):
        raise TypeError("reference must be EventRef")
    return {
        "stream_id": reference.stream_id.value,
        "generation": reference.generation.value,
        "sequence": reference.sequence,
        "event_id": reference.event_id.value,
        "payload_digest": reference.payload_digest,
    }


@dataclass(frozen=True)
class EventSpanRef:
    stream_id: StreamId
    generation: StreamGenerationId
    start_sequence: int
    end_sequence: int
    count: int
    ordered_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        start = _nonnegative_int(self.start_sequence, "start_sequence")
        end = _nonnegative_int(self.end_sequence, "end_sequence")
        count = _nonnegative_int(self.count, "count")
        if end < start or count != end - start:
            raise ValueError("EventSpanRef count must equal end_sequence - start_sequence")
        _digest(self.ordered_digest, "ordered_digest")
        object.__setattr__(self, "start_sequence", start)
        object.__setattr__(self, "end_sequence", end)
        object.__setattr__(self, "count", count)


class OrderedEventSpanHasher:
    """Shared O(1)-memory owner for one contiguous ordered EventRef span."""

    __slots__ = (
        "_stream_id",
        "_generation",
        "_start_sequence",
        "_next_sequence",
        "_hasher",
        "_sealed",
    )

    def __init__(
        self,
        stream_id: StreamId,
        generation: StreamGenerationId,
        start_sequence: int,
    ) -> None:
        if not isinstance(stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        start = _nonnegative_int(start_sequence, "start_sequence")
        self._stream_id = stream_id
        self._generation = generation
        self._start_sequence = start
        self._next_sequence = start
        self._hasher = hashlib.sha256()
        self._hasher.update(b"zlc_neutral_atom.OrderedEventRefs\x00")
        self._sealed = False

    def update(self, reference: EventRef) -> None:
        if self._sealed:
            raise RuntimeError("ordered event span is already sealed")
        if not isinstance(reference, EventRef):
            raise TypeError("reference must be EventRef")
        if (
            reference.stream_id != self._stream_id
            or reference.generation != self._generation
            or reference.sequence != self._next_sequence
        ):
            raise ValueError(
                "EventRef differs from the ordered span stream/generation/sequence"
            )
        encoded = encode(event_ref_to_tree(reference))
        self._hasher.update(len(encoded).to_bytes(8, "big"))
        self._hasher.update(encoded)
        self._next_sequence += 1

    def seal(self, end_sequence: int) -> EventSpanRef:
        if self._sealed:
            raise RuntimeError("ordered event span is already sealed")
        end = _nonnegative_int(end_sequence, "end_sequence")
        if end != self._next_sequence:
            raise ValueError("ordered event span does not cover the requested interval")
        self._sealed = True
        return EventSpanRef(
            self._stream_id,
            self._generation,
            self._start_sequence,
            end,
            end - self._start_sequence,
            self._hasher.hexdigest(),
        )


@dataclass(frozen=True)
class ArtifactInputRef:
    """Immutable, owner-encoded identity of one direct artifact dependency.

    Runtime deliberately does not interpret foreign artifact reference types.
    The owning domain serializes its typed reference with its canonical codec;
    this value snapshots those bytes and binds both their schema and content.
    """

    reference_schema_id: str
    canonical_reference: bytes
    content_digest: str

    def __post_init__(self) -> None:
        schema_id = _canonical_text(self.reference_schema_id, "reference_schema_id")
        if not isinstance(self.canonical_reference, bytes):
            raise TypeError("canonical_reference must be immutable bytes")
        raw = self.canonical_reference
        if not raw or len(raw) > _MAX_ARTIFACT_REFERENCE_BYTES:
            raise ValueError("canonical_reference must contain at most 64 KiB")
        try:
            tree = decode(raw)
        except Exception as error:
            raise ValueError("canonical_reference is not canonical owner data") from error
        if not isinstance(tree, dict) or tree.get("schema") != schema_id:
            raise ValueError(
                "canonical_reference schema does not match reference_schema_id"
            )
        if encode(tree) != raw:
            raise ValueError("canonical_reference is not byte-canonical")
        object.__setattr__(self, "reference_schema_id", schema_id)
        _digest(self.content_digest, "content_digest")

    @property
    def reference_digest(self) -> str:
        return sha256_digest(self.canonical_reference)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.ArtifactInputRef",
                "reference_schema_id": self.reference_schema_id,
                "reference_digest": self.reference_digest,
                "content_digest": self.content_digest,
            }
        )


def _owned_artifact_input_ref(reference: ArtifactInputRef) -> ArtifactInputRef:
    """Reconstruct a direct dependency so caller reflection cannot rewrite history."""

    if not isinstance(reference, ArtifactInputRef):
        raise TypeError("artifact input must be ArtifactInputRef")
    return ArtifactInputRef(
        reference.reference_schema_id,
        reference.canonical_reference,
        reference.content_digest,
    )


@dataclass(frozen=True)
class ProcessorStageProvenance:
    """Bounded, inspectable identity of one direct processor dependency set."""

    processor_binding_digest: str
    direct_artifact_inputs: tuple[ArtifactInputRef, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.processor_binding_digest, "processor_binding_digest")
        if not isinstance(self.direct_artifact_inputs, tuple):
            raise TypeError("direct_artifact_inputs must be an immutable tuple")
        inputs = tuple(item for item in self.direct_artifact_inputs)
        if any(not isinstance(item, ArtifactInputRef) for item in inputs):
            raise TypeError(
                "direct_artifact_inputs must contain ArtifactInputRef values"
            )
        if len(inputs) > _MAX_DIRECT_ARTIFACT_INPUTS:
            raise ValueError("processor stage has too many direct artifact inputs")
        if (
            sum(len(item.canonical_reference) for item in inputs)
            > _MAX_DIRECT_ARTIFACT_INPUT_BYTES
        ):
            raise ValueError("processor stage artifact inputs exceed the byte budget")
        identities = tuple(item.fingerprint for item in inputs)
        if len(set(identities)) != len(identities):
            raise ValueError("processor stage repeats an artifact input")
        # Frozen dataclasses are an API contract, not a security boundary:
        # ``object.__setattr__`` can still rewrite a caller-owned instance.  The
        # provenance owner therefore retains reconstructed references only.
        object.__setattr__(
            self,
            "direct_artifact_inputs",
            tuple(_owned_artifact_input_ref(item) for item in inputs),
        )


CausationRef = EventRef | EventSpanRef | ArtifactInputRef


@dataclass(frozen=True)
class TraceContext:
    run_id: str | None
    source_id: str
    correlation_id: str
    causation_refs: tuple[CausationRef, ...] = ()
    config_revision: int | None = None
    control_revision: int | None = None

    def __post_init__(self) -> None:
        if self.run_id is not None:
            _canonical_text(self.run_id, "run_id")
        _canonical_text(self.source_id, "source_id")
        _canonical_text(self.correlation_id, "correlation_id")
        refs = tuple(self.causation_refs)
        if any(not isinstance(ref, (EventRef, EventSpanRef, ArtifactInputRef)) for ref in refs):
            raise TypeError("causation_refs contains an unsupported reference")
        if len(refs) > _MAX_TRACE_CAUSATION_REFS:
            raise ValueError("causation_refs exceeds the direct lineage budget")
        object.__setattr__(self, "causation_refs", refs)
        for field in ("config_revision", "control_revision"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _nonnegative_int(value, field))


@dataclass(frozen=True)
class TraceBinding:
    """Stable formal-run identity shared by every event in one reservation."""

    run_id: str
    source_id: str

    def __post_init__(self) -> None:
        _canonical_text(self.run_id, "run_id")
        _canonical_text(self.source_id, "source_id")

    def validate(self, trace: TraceContext) -> None:
        if not isinstance(trace, TraceContext):
            raise TypeError("trace must be TraceContext")
        if trace.run_id != self.run_id or trace.source_id != self.source_id:
            raise StreamError("event trace differs from the reserved formal run/source")


@dataclass(frozen=True)
class Envelope(Generic[PayloadT]):
    event_ref: EventRef
    emitted_at: float
    captured_at: float
    payload_contract_fingerprint: str
    trace: TraceContext
    payload: PayloadT
    join_key: object | None = None
    join_key_schema_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if _contains_materialization(self.payload):
            raise TypeError("DataBlock/DataPatch are materialization values, not stream payloads")
        if not isinstance(self.event_ref, EventRef):
            raise TypeError("event_ref must be EventRef")
        object.__setattr__(self, "emitted_at", finite_real(self.emitted_at, "emitted_at"))
        object.__setattr__(self, "captured_at", finite_real(self.captured_at, "captured_at"))
        _digest(self.payload_contract_fingerprint, "payload_contract_fingerprint")
        if not isinstance(self.trace, TraceContext):
            raise TypeError("trace must be TraceContext")
        if (self.join_key is None) != (self.join_key_schema_fingerprint is None):
            raise ValueError("join_key and join_key_schema_fingerprint must be supplied together")
        if self.join_key is not None:
            try:
                hash(self.join_key)
            except TypeError as exc:
                raise TypeError("join_key must be frozen and hashable") from exc
            _digest(self.join_key_schema_fingerprint, "join_key_schema_fingerprint")

    @property
    def ref(self) -> EventRef:
        return self.event_ref

    @property
    def event_id(self) -> EventId:
        return self.event_ref.event_id

    @property
    def stream_id(self) -> StreamId:
        return self.event_ref.stream_id

    @property
    def stream_generation(self) -> StreamGenerationId:
        return self.event_ref.generation

    @property
    def sequence(self) -> int:
        return self.event_ref.sequence

    @property
    def payload_digest(self) -> str:
        return self.event_ref.payload_digest


class EndOfStream:
    """Opaque terminal receipt minted exactly once by an AcquisitionStream."""

    __slots__ = (
        "_stream_id",
        "_stream_generation",
        "_end_sequence",
        "_ended_at",
        "_owner_ref",
        "_nonce",
    )

    def __init__(
        self,
        token: object,
        *,
        stream_id: StreamId,
        stream_generation: StreamGenerationId,
        end_sequence: int,
        ended_at: float,
        owner: object,
        nonce: object,
    ) -> None:
        if token is not _EOS_TOKEN:
            raise PermissionError("EndOfStream can only be minted by AcquisitionStream")
        if not isinstance(stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(stream_generation, StreamGenerationId):
            raise TypeError("stream_generation must be StreamGenerationId")
        object.__setattr__(self, "_stream_id", stream_id)
        object.__setattr__(self, "_stream_generation", stream_generation)
        object.__setattr__(self, "_end_sequence", _nonnegative_int(end_sequence, "end_sequence"))
        object.__setattr__(self, "_ended_at", finite_real(ended_at, "ended_at"))
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        object.__setattr__(self, "_nonce", nonce)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("EndOfStream is immutable")

    @property
    def stream_id(self) -> StreamId:
        return self._stream_id

    @property
    def stream_generation(self) -> StreamGenerationId:
        return self._stream_generation

    @property
    def end_sequence(self) -> int:
        return self._end_sequence

    @property
    def ended_at(self) -> float:
        return self._ended_at

    @property
    def _owner(self) -> object | None:
        """Return the live stream owner without making the receipt own it."""

        return self._owner_ref()


class StreamError(RuntimeError):
    pass


class StreamGap(StreamError):
    def __init__(self, expected: int, earliest_retained: int, next_sequence: int) -> None:
        self.expected = expected
        self.earliest_retained = earliest_retained
        self.next_sequence = next_sequence
        super().__init__(
            f"stream history gap: expected {expected}, earliest retained "
            f"{earliest_retained}, next sequence {next_sequence}"
        )


class StreamEndedEarly(StreamError):
    pass


class SchemaChanged(StreamError):
    def __init__(
        self,
        previous: StreamGenerationId,
        replacement: StreamGenerationId,
    ) -> None:
        self.previous = previous
        self.replacement = replacement
        super().__init__(
            f"stream generation changed from {previous.value!r} to {replacement.value!r}"
        )


class StreamBackpressure(StreamError):
    pass


class RetentionOverrun(StreamError):
    pass


class SourceFailed(StreamError):
    pass


class ReservationCapacityExceeded(StreamError):
    pass


class ReservationStateError(StreamError):
    pass


class ReservationState(str, Enum):
    RESERVED = "RESERVED"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RELEASED = "RELEASED"


class ProducerFlowControl(str, Enum):
    BACKPRESSURE_CAPABLE = "BACKPRESSURE_CAPABLE"
    NON_BACKPRESSURE_CAPTURED = "NON_BACKPRESSURE_CAPTURED"


@dataclass(frozen=True)
class _Stored(Generic[PayloadT]):
    envelope: Envelope[PayloadT]
    payload_bytes: int


class Delivery(Generic[PayloadT]):
    """Single-use delivery minted by one registered cursor authority."""

    __slots__ = ("_cursor", "_envelope", "_acked")

    def __init__(
        self,
        token: object,
        *,
        cursor: "AcquisitionCursor[PayloadT]",
        envelope: Envelope[PayloadT],
    ) -> None:
        if token is not _DELIVERY_TOKEN:
            raise PermissionError("Delivery can only be minted by AcquisitionCursor")
        object.__setattr__(self, "_cursor", cursor)
        object.__setattr__(self, "_envelope", envelope)
        object.__setattr__(self, "_acked", False)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("Delivery is immutable")

    @property
    def envelope(self) -> Envelope[PayloadT]:
        return self._envelope

    @property
    def payload(self) -> PayloadT:
        return self._envelope.payload

    @property
    def is_exact(self) -> bool:
        return self._cursor._reservation_token is not None

    @property
    def acknowledged(self) -> bool:
        return self._acked

    def ack(self) -> None:
        self._cursor._ack_delivery(self)


class ExactReservation(Generic[PayloadT]):
    """One finite retention claim; acknowledgement is its only moving watermark."""

    def __init__(
        self,
        authority: object,
        *,
        stream: "AcquisitionStream[PayloadT]",
        token: object,
        start_sequence: int,
        end_sequence: int,
        max_inflight_events: int,
        max_inflight_bytes: int,
        trace_binding: TraceBinding,
    ) -> None:
        if authority is not _RESERVATION_TOKEN:
            raise PermissionError("ExactReservation can only be minted by AcquisitionStream")
        self._stream = stream
        self._token = token
        self.start_sequence = start_sequence
        self.end_sequence = end_sequence
        self.max_inflight_events = max_inflight_events
        self.max_inflight_bytes = max_inflight_bytes
        self.trace_binding = trace_binding
        self._ack_sequence = start_sequence
        self._unacked_bytes = 0
        self._state = ReservationState.RESERVED
        self._cursor: AcquisitionCursor[PayloadT] | None = None
        self._consumer_owner: object | None = None

    @property
    def state(self) -> ReservationState:
        with self._stream._condition:
            return self._state

    @property
    def acknowledged_sequence(self) -> int:
        with self._stream._condition:
            return self._ack_sequence

    @property
    def consumer_bound(self) -> bool:
        """Whether the reservation has its one required exact consumer."""

        with self._stream._condition:
            return self._consumer_owner is not None

    def activate(self) -> "AcquisitionCursor[PayloadT]":
        with self._stream._condition:
            if self._state is not ReservationState.RESERVED:
                raise ReservationStateError("only a reserved exact stream may be activated")
            self._state = ReservationState.ACTIVE
            self._cursor = AcquisitionCursor(
                _CURSOR_TOKEN,
                stream=self._stream,
                start_sequence=self.start_sequence,
                end_sequence=self.end_sequence,
                reservation_token=self._token,
            )
            return self._cursor

    def complete(self) -> None:
        with self._stream._condition:
            if self._consumer_owner is not None:
                raise ReservationStateError(
                    "reservation completion belongs to its bound exact consumer"
                )
            if self._state not in (ReservationState.ACTIVE, ReservationState.DRAINING):
                raise ReservationStateError("reservation is not active or draining")
            if self._ack_sequence != self.end_sequence:
                raise ReservationStateError("reservation cannot complete before every event is acked")
            self._state = ReservationState.COMPLETED
            self._stream._condition.notify_all()

    def abort(self, *, cancelled: bool = False) -> None:
        with self._stream._condition:
            if self._consumer_owner is not None:
                raise ReservationStateError(
                    "bound reservation abort belongs to its exact consumer"
                )
            if self._state in (ReservationState.RELEASED, ReservationState.COMPLETED):
                raise ReservationStateError("completed/released reservation cannot be aborted")
            self._state = ReservationState.CANCELLED if cancelled else ReservationState.FAILED
            self._stream._trim_locked()
            self._stream._condition.notify_all()

    def release(self) -> None:
        self._stream._release_reservation(self._token)


class _ObjectReference:
    """Weak reverse ownership when possible, identity-preserving otherwise.

    Runtime owners such as DatasetBuilder, CaptureSession, and processor workers
    are weak-referenceable.  Keeping only a weak reverse edge prevents the
    authority proof from extending their lifetime after a reservation is
    released.  The strong fallback preserves the low-level stream API for
    identity tokens such as ``object()``, which cannot participate in a cycle.
    """

    __slots__ = ("_weak", "_strong")

    def __init__(self, owner: object) -> None:
        try:
            reference = weakref.ref(owner)
        except TypeError:
            reference = None
        self._weak = reference
        self._strong = owner if reference is None else None

    def get(self) -> object | None:
        return self._strong if self._weak is None else self._weak()

    def matches(self, owner: object) -> bool:
        current = self.get()
        return current is not None and current is owner


class _CallbackReference:
    """Weakly retain bound owner callbacks used only while a graph is live."""

    __slots__ = ("_weak", "_strong")

    def __init__(self, callback: Callable) -> None:
        if not callable(callback):
            raise TypeError("exact owner callback must be callable")
        bound_owner = getattr(callback, "__self__", None)
        if bound_owner is not None:
            try:
                reference = weakref.WeakMethod(callback)
            except TypeError:
                reference = None
        else:
            reference = None
        self._weak = reference
        # Free functions have no reverse owner edge and must remain live for
        # the duration of the proof.  Bound methods use WeakMethod above.
        self._strong = callback if reference is None else None

    def resolve(self) -> Callable:
        callback = self._strong if self._weak is None else self._weak()
        if callback is None:
            raise ReservationStateError("exact chain owner is no longer live")
        return callback


class ExactConsumerReadiness:
    """Opaque proof that a source reaches one live terminal DatasetBuilder.

    The proof is identity-bound to both the source consumer and the terminal
    dataset consumer.  Every intermediate processor contributes private
    liveness, completion, and cancellation capabilities.  Liveness is
    rechecked at bind, prepare, and start; completion recursively means the
    one terminal dataset has sealed, not merely that one stage emitted EOS.
    This is a process-local preflight capability, not a serializable plan flag
    and not a user-constructible ``is_exact`` boolean.
    """

    __slots__ = (
        "_source_reservation",
        "_source_consumer",
        "_source_contract_digest",
        "_source_schedule_digest",
        "_source_key_sequence_digest",
        "_chain_contract_digest",
        "_terminal_reservation",
        "_terminal_consumer",
        "_binding_owner",
        "_owner_liveness",
        "_owner_completion",
        "_owner_cancel",
        "_processor_stages",
    )

    def __init__(
        self,
        authority: object,
        *,
        source_reservation: ExactReservation,
        source_consumer: object,
        source_contract_digest: str,
        source_schedule_digest: str,
        source_key_sequence_digest: str,
        chain_contract_digest: str,
        terminal_reservation: ExactReservation,
        terminal_consumer: object,
        owner_liveness: tuple[Callable[[], None], ...],
        owner_completion: Callable[[float], object] | None,
        owner_cancel: Callable[[str | None], bool] | None,
        processor_stages: tuple[ProcessorStageProvenance, ...],
    ) -> None:
        if authority is not _READINESS_TOKEN:
            raise PermissionError(
                "ExactConsumerReadiness is minted by an exact stream authority"
            )
        _digest(source_contract_digest, "source contract digest")
        _digest(source_schedule_digest, "source schedule digest")
        _digest(source_key_sequence_digest, "source key-sequence digest")
        _digest(chain_contract_digest, "chain contract digest")
        object.__setattr__(self, "_source_reservation", source_reservation)
        object.__setattr__(
            self,
            "_source_consumer",
            _ObjectReference(source_consumer),
        )
        object.__setattr__(self, "_source_contract_digest", source_contract_digest)
        object.__setattr__(self, "_source_schedule_digest", source_schedule_digest)
        object.__setattr__(
            self,
            "_source_key_sequence_digest",
            source_key_sequence_digest,
        )
        object.__setattr__(self, "_chain_contract_digest", chain_contract_digest)
        object.__setattr__(self, "_terminal_reservation", terminal_reservation)
        object.__setattr__(
            self,
            "_terminal_consumer",
            _ObjectReference(terminal_consumer),
        )
        object.__setattr__(self, "_binding_owner", None)
        object.__setattr__(
            self,
            "_owner_liveness",
            tuple(_CallbackReference(callback) for callback in owner_liveness),
        )
        object.__setattr__(
            self,
            "_owner_completion",
            None if owner_completion is None else _CallbackReference(owner_completion),
        )
        object.__setattr__(
            self,
            "_owner_cancel",
            None if owner_cancel is None else _CallbackReference(owner_cancel),
        )
        stages = tuple(processor_stages)
        if any(not isinstance(stage, ProcessorStageProvenance) for stage in stages):
            raise TypeError("processor_stages contains an unsupported value")
        if len(stages) > _MAX_EXACT_PROCESSOR_STAGES:
            raise ValueError("exact processor chain exceeds the stage budget")
        chain_inputs = tuple(
            reference
            for stage in stages
            for reference in stage.direct_artifact_inputs
        )
        if len(chain_inputs) > _MAX_CHAIN_ARTIFACT_INPUT_OCCURRENCES:
            raise ValueError("exact processor chain has too many artifact input edges")
        if (
            sum(len(reference.canonical_reference) for reference in chain_inputs)
            > _MAX_CHAIN_ARTIFACT_INPUT_BYTES
        ):
            raise ValueError("exact processor chain artifact inputs exceed the byte budget")
        object.__setattr__(self, "_processor_stages", stages)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ExactConsumerReadiness is immutable")

    @property
    def chain_contract_digest(self) -> str:
        return self._chain_contract_digest

    @property
    def processor_stages(self) -> tuple[ProcessorStageProvenance, ...]:
        return self._processor_stages

    def _resolved_owner_liveness(self) -> tuple[Callable[[], None], ...]:
        return tuple(reference.resolve() for reference in self._owner_liveness)

    def _require_live_source_locked(
        self,
        reservation: ExactReservation,
        context: str,
    ) -> None:
        """Validate the shared live source invariant while its stream lock is held."""

        stream = reservation._stream
        if stream._reservations.get(reservation._token) is not reservation:
            raise ReservationStateError(f"{context} reservation is not registered")
        if not self._source_consumer.matches(reservation._consumer_owner):
            raise ReservationStateError(f"{context} lost its source consumer")
        if reservation._state is not ReservationState.ACTIVE:
            raise ReservationStateError(f"{context} source is not ACTIVE")
        if stream._terminal_error is not None or stream._closed:
            raise ReservationStateError(f"{context} source is no longer live")

    def _claim_binding(self, owner: object) -> None:
        """Consume this process-local proof for exactly one enclosing owner."""

        stream = self._source_reservation._stream
        with stream._condition:
            if self._binding_owner is not None:
                raise ReservationStateError("exact readiness was already bound")
            reservation = self._source_reservation
            self._require_live_source_locked(reservation, "exact readiness")
            self._validate_terminal_sink()
            object.__setattr__(self, "_binding_owner", _ObjectReference(owner))

    def validate_source(
        self,
        *,
        reservation: ExactReservation,
        trace_binding: TraceBinding,
        payload_contract_fingerprint: str,
        join_key_contract_fingerprint: str,
        source_contract_digest: str,
        source_schedule_digest: str,
        source_key_sequence_digest: str,
        total_events: int,
    ) -> None:
        """Revalidate every source identity plus the live terminal sink."""

        if reservation is not self._source_reservation:
            raise ReservationStateError("exact readiness belongs to another reservation")
        _digest(payload_contract_fingerprint, "payload contract fingerprint")
        _digest(join_key_contract_fingerprint, "join key contract fingerprint")
        _digest(source_contract_digest, "source contract digest")
        _digest(source_schedule_digest, "source schedule digest")
        _digest(source_key_sequence_digest, "source key-sequence digest")
        expected_total = _positive_int(total_events, "total_events")
        stream = reservation._stream
        with stream._condition:
            self._require_live_source_locked(reservation, "exact readiness")
            if reservation.trace_binding != trace_binding:
                raise ReservationStateError("exact readiness trace binding differs")
            if reservation.end_sequence - reservation.start_sequence != expected_total:
                raise ReservationStateError("exact readiness event interval differs")
            if stream.payload_contract_fingerprint != payload_contract_fingerprint:
                raise ReservationStateError("exact readiness payload contract differs")
            key_contract = stream._join_key_contract
            if key_contract is None or key_contract.fingerprint != join_key_contract_fingerprint:
                raise ReservationStateError("exact readiness join-key contract differs")
            if self._source_contract_digest != source_contract_digest:
                raise ReservationStateError("exact readiness source contract differs")
            if self._source_schedule_digest != source_schedule_digest:
                raise ReservationStateError("exact readiness source schedule differs")
            if self._source_key_sequence_digest != source_key_sequence_digest:
                raise ReservationStateError("exact readiness source key sequence differs")
        self._validate_terminal_sink()

    def _validate_terminal_sink(self) -> None:
        reservation = self._terminal_reservation
        stream = reservation._stream
        with stream._condition:
            if stream._reservations.get(reservation._token) is not reservation:
                raise ReservationStateError("terminal dataset reservation is not registered")
            if not self._terminal_consumer.matches(reservation._consumer_owner):
                raise ReservationStateError("exact chain lost its terminal dataset consumer")
            if reservation._state not in (
                ReservationState.ACTIVE,
                ReservationState.DRAINING,
            ):
                raise ReservationStateError("terminal dataset consumer is not live")
            if stream._terminal_error is not None or stream._closed:
                raise ReservationStateError("terminal dataset stream is no longer live")
        for validate_liveness in self._resolved_owner_liveness():
            validate_liveness()

    def _validate_emitter(
        self,
        *,
        stream: "AcquisitionStream",
        trace_binding: TraceBinding,
        payload_contract_fingerprint: str,
        join_key_contract_fingerprint: str,
        source_key_sequence_digest: str,
        total_events: int,
    ) -> None:
        """Cross-bind one producer to this proof's immediate source interval."""

        _digest(payload_contract_fingerprint, "payload contract fingerprint")
        _digest(join_key_contract_fingerprint, "join key contract fingerprint")
        _digest(source_key_sequence_digest, "source key-sequence digest")
        expected_total = _positive_int(total_events, "total_events")
        reservation = self._source_reservation
        if reservation._stream is not stream:
            raise ReservationStateError(
                "downstream readiness belongs to another output stream"
            )
        with stream._condition:
            self._require_live_source_locked(reservation, "downstream readiness")
            if reservation.trace_binding != trace_binding:
                raise ReservationStateError(
                    "downstream readiness trace binding differs from output producer"
                )
            if reservation.end_sequence - reservation.start_sequence != expected_total:
                raise ReservationStateError(
                    "downstream readiness event interval differs"
                )
            if stream.payload_contract_fingerprint != payload_contract_fingerprint:
                raise ReservationStateError(
                    "downstream readiness payload contract differs"
                )
            key_contract = stream._join_key_contract
            if (
                key_contract is None
                or key_contract.fingerprint != join_key_contract_fingerprint
            ):
                raise ReservationStateError(
                    "downstream readiness join-key contract differs"
                )
            if self._source_key_sequence_digest != source_key_sequence_digest:
                raise ReservationStateError(
                    "downstream readiness source key sequence differs"
                )
        self._validate_terminal_sink()

    def _await_source_ack(
        self,
        owner: object,
        event_ref: EventRef,
        *,
        deadline_monotonic: float,
        checkpoint: Callable[[], None],
    ) -> None:
        """Wait until the immediate consumer really acknowledges one emitted event."""

        if not isinstance(event_ref, EventRef):
            raise TypeError("event_ref must be EventRef")
        if not callable(checkpoint):
            raise TypeError("checkpoint must be callable")
        deadline = finite_real(deadline_monotonic, "deadline_monotonic")
        reservation = self._source_reservation
        stream = reservation._stream
        if (
            event_ref.stream_id != stream.stream_id
            or event_ref.generation != stream.generation
            or event_ref.sequence < reservation.start_sequence
            or event_ref.sequence >= reservation.end_sequence
        ):
            raise ReservationStateError(
                "emitted event is outside downstream readiness source interval"
            )
        while True:
            checkpoint()
            terminal_before_ack = False
            with stream._condition:
                binding_owner = self._binding_owner
                if binding_owner is None or not binding_owner.matches(owner):
                    raise PermissionError(
                        "downstream acknowledgement belongs to another bound owner"
                    )
                if reservation._ack_sequence > event_ref.sequence:
                    return
                registered = stream._reservations.get(reservation._token) is reservation
                terminal_before_ack = (
                    not registered
                    or reservation._state
                    not in (ReservationState.ACTIVE, ReservationState.DRAINING)
                    or stream._terminal_error is not None
                )
                if not terminal_before_ack:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "downstream processor did not acknowledge output before deadline"
                        )
                    stream._condition.wait(min(0.05, remaining))
            if terminal_before_ack:
                # The completion callback preserves the downstream worker's real
                # exception instead of reducing it to a reservation-state symptom.
                self._await_bound_completion(owner, deadline_monotonic=deadline)
                raise ReservationStateError(
                    "downstream processor completed without acknowledging output"
                )

    def _await_bound_completion(
        self,
        owner: object,
        *,
        deadline_monotonic: float,
    ) -> object:
        """Await the immediate processor, which itself includes terminal completion."""

        deadline = finite_real(deadline_monotonic, "deadline_monotonic")
        reservation = self._source_reservation
        stream = reservation._stream
        with stream._condition:
            binding_owner = self._binding_owner
            if binding_owner is None or not binding_owner.matches(owner):
                raise PermissionError(
                    "downstream completion belongs to another bound owner"
                )
            completion_reference = self._owner_completion
        if completion_reference is None:
            raise ReservationStateError(
                "terminal readiness has no intermediate processor completion"
            )
        completion = completion_reference.resolve()
        return completion(deadline)

    def _cancel_bound_owner(self, owner: object, reason: str | None) -> bool:
        """Propagate fail-closed teardown to the immediate downstream processor."""

        reservation = self._source_reservation
        stream = reservation._stream
        with stream._condition:
            binding_owner = self._binding_owner
            if binding_owner is None or not binding_owner.matches(owner):
                raise PermissionError(
                    "downstream cancellation belongs to another bound owner"
                )
            cancel_reference = self._owner_cancel
        if cancel_reference is None:
            raise ReservationStateError(
                "terminal readiness has no intermediate processor cancellation"
            )
        cancel = cancel_reference.resolve()
        return cancel(reason)

class AcquisitionCursor(Generic[PayloadT]):
    """Opaque cursor with at most one unacknowledged delivery."""

    def __init__(
        self,
        authority: object,
        *,
        stream: "AcquisitionStream[PayloadT]",
        start_sequence: int,
        end_sequence: int | None,
        reservation_token: object | None,
    ) -> None:
        if authority is not _CURSOR_TOKEN:
            raise PermissionError("AcquisitionCursor can only be minted by AcquisitionStream")
        self._stream = stream
        self._generation = stream.generation
        self._next_sequence = start_sequence
        self._end_sequence = end_sequence
        self._reservation_token = reservation_token
        self._inflight: Delivery[PayloadT] | None = None

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def next(self, timeout: float | None = None) -> Delivery[PayloadT]:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._stream._condition:
            if self._inflight is not None:
                return self._inflight
            if self._end_sequence is not None and self._next_sequence >= self._end_sequence:
                raise StopIteration
            while True:
                if self._generation != self._stream.generation:
                    raise StreamEndedEarly("stream generation changed under cursor")
                if self._stream._terminal_error is not None:
                    raise self._stream._terminal_error
                stored = self._stream._records.get(self._next_sequence)
                if stored is not None:
                    self._inflight = Delivery(
                        _DELIVERY_TOKEN,
                        cursor=self,
                        envelope=stored.envelope,
                    )
                    return self._inflight
                if self._next_sequence < self._stream._next_sequence:
                    raise StreamGap(
                        self._next_sequence,
                        self._stream._earliest_retained_locked(),
                        self._stream._next_sequence,
                    )
                if self._stream._closed:
                    raise StreamEndedEarly(
                        f"stream ended at {self._stream._next_sequence} before sequence "
                        f"{self._next_sequence}"
                    )
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for acquisition event")
                    self._stream._condition.wait(remaining)
                else:
                    self._stream._condition.wait()

    def _ack_delivery(
        self,
        delivery: Delivery[PayloadT],
        consumer: object | None = None,
    ) -> None:
        with self._stream._condition:
            if self._inflight is None or delivery is not self._inflight:
                raise ValueError("ack must consume this cursor's current delivery")
            if delivery._acked:
                raise ValueError("delivery acknowledgement is single-use")
            envelope = delivery.envelope
            if envelope.sequence != self._next_sequence:
                raise ValueError("delivery sequence does not match cursor")
            if self._reservation_token is not None:
                reservation = self._stream._reservations.get(self._reservation_token)
                if reservation is None:
                    raise ReservationStateError("reservation is not registered")
                if (
                    reservation._consumer_owner is not None
                    and reservation._consumer_owner is not consumer
                ):
                    raise PermissionError(
                        "delivery acknowledgement belongs to the bound exact consumer"
                    )
                self._stream._ack(self._reservation_token, envelope.sequence)
            object.__setattr__(delivery, "_acked", True)
            self._next_sequence += 1
            self._inflight = None


@dataclass(frozen=True, slots=True)
class MonitorUpdate(Generic[PayloadT]):
    """One atomic delivery from a concrete tap, including local overwrite loss."""

    envelope: Envelope[PayloadT]
    missed: int

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, Envelope):
            raise TypeError("envelope must be Envelope")
        object.__setattr__(self, "missed", _nonnegative_int(self.missed, "missed"))


class MonitorTap(Generic[PayloadT]):
    """A bounded overwrite queue which never participates in exact retention."""

    def __init__(
        self,
        authority: object,
        *,
        stream: "AcquisitionStream[PayloadT]",
        max_events: int,
        max_bytes: int,
    ) -> None:
        if authority is not _MONITOR_TOKEN:
            raise PermissionError("MonitorTap can only be minted by AcquisitionStream")
        self._stream = stream
        self.max_events = _positive_int(max_events, "monitor max_events")
        self.max_bytes = _positive_int(max_bytes, "monitor max_bytes")
        if self.max_bytes < stream.max_payload_bytes:
            raise ValueError("monitor max_bytes must retain one maximum payload")
        self._condition = threading.Condition(threading.Lock())
        self._queue: deque[_Stored[PayloadT]] = deque()
        self._retained_bytes = 0
        self._missed = 0
        self._closed = False
        self._source_finished = False
        self._terminal_error: StreamError | None = None
        self._consumer_owner: object | None = None

    @property
    def retained_bytes(self) -> int:
        with self._condition:
            return self._retained_bytes

    def _offer(self, stored: _Stored[PayloadT]) -> None:
        with self._condition:
            if self._closed or self._source_finished:
                return
            while self._queue and (
                len(self._queue) >= self.max_events
                or self._retained_bytes + stored.payload_bytes > self.max_bytes
            ):
                removed = self._queue.popleft()
                self._retained_bytes -= removed.payload_bytes
                self._missed += 1
            self._queue.append(stored)
            self._retained_bytes += stored.payload_bytes
            self._condition.notify_all()

    def _claim_consumer(self, owner: object) -> None:
        if owner is None:
            raise TypeError("monitor consumer owner cannot be None")
        # Match the stream -> tap lock order used by publication and terminal
        # delivery. Ownership and the first publication then have one
        # linearization point.
        with self._stream._condition:
            if self._stream._next_sequence != 0:
                raise ReservationStateError(
                    "monitor consumer must bind before the first publication"
                )
            with self._condition:
                if self._closed or self._source_finished:
                    raise StreamEndedEarly("cannot bind a terminal monitor tap")
                if self._consumer_owner is not None:
                    raise PermissionError("monitor tap already belongs to another consumer")
                self._consumer_owner = owner
                # A raw reader may already be asleep in next(). Wake it so it
                # rechecks authority rather than stealing the owner's first event.
                self._condition.notify_all()

    def _raise_terminal_if_empty_locked(self) -> None:
        if self._closed:
            raise StreamEndedEarly("monitor tap is closed")
        if self._source_finished:
            if self._terminal_error is not None:
                raise self._terminal_error
            raise StreamEndedEarly("monitor source reached end-of-stream")

    def _take(
        self,
        owner: object | None,
        *,
        latest: bool,
        timeout: float | None = None,
    ) -> MonitorUpdate[PayloadT]:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._queue:
                self._raise_terminal_if_empty_locked()
                if self._consumer_owner is not owner:
                    raise PermissionError("monitor tap belongs to another consumer")
                if latest:
                    raise LookupError("monitor tap has no retained event")
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for monitor event")
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()
            if self._consumer_owner is not owner:
                raise PermissionError("monitor tap belongs to another consumer")
            if latest:
                while len(self._queue) > 1:
                    removed = self._queue.popleft()
                    self._retained_bytes -= removed.payload_bytes
                    self._missed += 1
            stored = self._queue.popleft()
            self._retained_bytes -= stored.payload_bytes
            missed, self._missed = self._missed, 0
            return MonitorUpdate(stored.envelope, missed)

    def next(self, timeout: float | None = None) -> MonitorUpdate[PayloadT]:
        return self._take(None, latest=False, timeout=timeout)

    def _next_for(
        self,
        owner: object,
        timeout: float | None = None,
    ) -> MonitorUpdate[PayloadT]:
        return self._take(owner, latest=False, timeout=timeout)

    def latest(self) -> MonitorUpdate[PayloadT]:
        return self._take(None, latest=True)

    def _latest_for(self, owner: object) -> MonitorUpdate[PayloadT]:
        return self._take(owner, latest=True)

    def _source_ended(self, error: StreamError | None) -> None:
        with self._condition:
            self._source_finished = True
            self._terminal_error = error
            if error is not None:
                self._queue.clear()
                self._retained_bytes = 0
            self._condition.notify_all()

    def close(self) -> None:
        self._stream._remove_monitor(self)
        with self._condition:
            self._closed = True
            self._consumer_owner = None
            self._queue.clear()
            self._retained_bytes = 0
            self._condition.notify_all()


class AcquisitionProducer(Generic[PayloadT]):
    """Exclusive write/terminal authority retained by the source owner lane."""

    __slots__ = ("_stream", "__weakref__")

    def __init__(self, authority: object, stream: "AcquisitionStream[PayloadT]") -> None:
        if authority is not _PRODUCER_TOKEN:
            raise PermissionError("AcquisitionProducer can only be minted with its stream")
        self._stream = stream

    def emit(
        self,
        payload: PayloadT,
        *,
        captured_at: float,
        trace: TraceContext,
        join_key: object | None = None,
    ) -> Envelope[PayloadT]:
        return self._stream._emit(
            payload,
            captured_at=captured_at,
            trace=trace,
            join_key=join_key,
        )

    def finish(self) -> EndOfStream:
        return self._stream._finish(self)

    def supersede(self, replacement: StreamGenerationId) -> None:
        self._stream._supersede(self, replacement)

    def fail(self, error: StreamError) -> None:
        self._stream._fail(self, error)


class AcquisitionStream(Generic[PayloadT]):
    """One finite generation with shared exact retention and monitor fan-out."""

    def __init__(
        self,
        authority: object,
        *,
        stream_id: StreamId,
        generation: StreamGenerationId,
        payload_contract: PayloadContract[PayloadT],
        flow_control: ProducerFlowControl,
        retention_events: int,
        retention_bytes: int,
        join_key_contract: JoinKeyContract | None = None,
    ) -> None:
        if authority is not _STREAM_TOKEN:
            raise PermissionError("use AcquisitionStream.create()")
        if not isinstance(stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        if not isinstance(flow_control, ProducerFlowControl):
            raise TypeError("flow_control must be ProducerFlowControl")
        try:
            payload_contract_fingerprint = payload_contract.fingerprint
            max_payload_bytes = payload_contract.max_retained_nbytes
        except AttributeError as exc:
            raise TypeError("payload_contract does not implement PayloadContract") from exc
        _digest(payload_contract_fingerprint, "payload contract fingerprint")
        self.stream_id = stream_id
        self.generation = generation
        self.payload_contract_fingerprint = payload_contract_fingerprint
        self.flow_control = flow_control
        self.retention_events = _positive_int(retention_events, "retention_events")
        self.retention_bytes = _positive_int(retention_bytes, "retention_bytes")
        self.max_payload_bytes = _positive_int(max_payload_bytes, "max_payload_bytes")
        if self.max_payload_bytes > self.retention_bytes:
            raise ValueError("max_payload_bytes cannot exceed retention_bytes")
        for method in ("snapshot", "validate", "retained_nbytes", "digest"):
            if not callable(getattr(payload_contract, method, None)):
                raise TypeError(f"payload_contract.{method} must be callable")
        if join_key_contract is not None:
            _digest(join_key_contract.fingerprint, "join key contract fingerprint")
            if not callable(getattr(join_key_contract, "snapshot", None)):
                raise TypeError("join_key_contract.snapshot must be callable")
        self._payload_contract = payload_contract
        self._join_key_contract = join_key_contract
        self._condition = threading.Condition(threading.RLock())
        self._records: dict[int, _Stored[PayloadT]] = {}
        self._order: deque[int] = deque()
        self._retained_bytes = 0
        self._next_sequence = 0
        self._reservations: dict[object, ExactReservation[PayloadT]] = {}
        self._formal_consumer_claimed = False
        self._formal_rebind_required = False
        self._formal_interval_start: int | None = None
        self._formal_interval_end: int | None = None
        self._monitors: set[MonitorTap[PayloadT]] = set()
        self._closed = False
        self._terminal_error: StreamError | None = None
        self._eos: EndOfStream | None = None
        producer = AcquisitionProducer(_PRODUCER_TOKEN, self)
        # Keep construction authority alive until create() can return it; the
        # public factory then replaces this temporary edge with a weak reverse
        # reference so producer -> stream remains the sole ownership direction.
        self._producer_owner: AcquisitionProducer[PayloadT] | None = producer
        self._producer_ref: weakref.ReferenceType | None = None

    @classmethod
    def create(
        cls,
        stream_id: StreamId,
        payload_contract: PayloadContract[PayloadT],
        *,
        flow_control: ProducerFlowControl,
        retention_events: int,
        retention_bytes: int,
        join_key_contract: JoinKeyContract | None = None,
    ) -> tuple["AcquisitionStream[PayloadT]", AcquisitionProducer[PayloadT]]:
        stream = cls(
            _STREAM_TOKEN,
            stream_id=stream_id,
            generation=StreamGenerationId(uuid.uuid4().hex),
            payload_contract=payload_contract,
            flow_control=flow_control,
            retention_events=retention_events,
            retention_bytes=retention_bytes,
            join_key_contract=join_key_contract,
        )
        producer = stream._producer_owner
        assert producer is not None
        stream._producer_ref = weakref.ref(producer)
        stream._producer_owner = None
        return stream, producer

    @property
    def _producer(self) -> AcquisitionProducer[PayloadT] | None:
        """Observe the live producer authority without creating an owner cycle."""

        owner = self._producer_owner
        if owner is not None:
            return owner
        reference = self._producer_ref
        return None if reference is None else reference()

    @property
    def next_sequence(self) -> int:
        with self._condition:
            return self._next_sequence

    @property
    def retained_bytes(self) -> int:
        with self._condition:
            return self._retained_bytes

    @property
    def retained_events(self) -> int:
        with self._condition:
            return len(self._order)

    def reserve(
        self,
        *,
        total_events: int,
        max_inflight_events: int,
        max_inflight_bytes: int,
        trace_binding: TraceBinding,
    ) -> ExactReservation[PayloadT]:
        total = _positive_int(total_events, "total_events")
        inflight_events = _positive_int(max_inflight_events, "max_inflight_events")
        inflight_bytes = _positive_int(max_inflight_bytes, "max_inflight_bytes")
        if not isinstance(trace_binding, TraceBinding):
            raise TypeError("trace_binding must be TraceBinding")
        if inflight_events > total:
            raise ValueError("max_inflight_events cannot exceed total_events")
        required_worst_case_bytes = inflight_events * self.max_payload_bytes
        if inflight_bytes < required_worst_case_bytes:
            raise ValueError(
                "max_inflight_bytes must cover max_inflight_events * max_payload_bytes"
            )
        if inflight_bytes > self.retention_bytes:
            raise ReservationCapacityExceeded("reservation byte budget exceeds stream capacity")
        with self._condition:
            if self._closed:
                raise StreamEndedEarly("cannot reserve a closed stream")
            if self._reservations:
                raise ReservationCapacityExceeded(
                    "one stream generation has exactly one formal exact consumer"
                )
            if self._formal_consumer_claimed:
                raise ReservationCapacityExceeded(
                    "this stream generation already had its formal exact consumer"
                )
            token = object()
            reservation = ExactReservation(
                _RESERVATION_TOKEN,
                stream=self,
                token=token,
                start_sequence=self._next_sequence,
                end_sequence=self._next_sequence + total,
                max_inflight_events=inflight_events,
                max_inflight_bytes=inflight_bytes,
                trace_binding=trace_binding,
            )
            self._reservations[token] = reservation
            return reservation

    def subscribe(self, start_sequence: int | None = None) -> AcquisitionCursor[PayloadT]:
        with self._condition:
            start = self._next_sequence if start_sequence is None else _nonnegative_int(
                start_sequence, "start_sequence"
            )
            return AcquisitionCursor(
                _CURSOR_TOKEN,
                stream=self,
                start_sequence=start,
                end_sequence=None,
                reservation_token=None,
            )

    def monitor(self, *, max_events: int, max_bytes: int) -> MonitorTap[PayloadT]:
        tap = MonitorTap(
            _MONITOR_TOKEN,
            stream=self,
            max_events=max_events,
            max_bytes=max_bytes,
        )
        with self._condition:
            if self._closed:
                raise StreamEndedEarly("cannot monitor a closed stream")
            if self._next_sequence != 0:
                raise ReservationStateError(
                    "monitor topology must be admitted before the first publication"
                )
            self._monitors.add(tap)
        return tap

    def _emit(
        self,
        payload: PayloadT,
        *,
        captured_at: float,
        trace: TraceContext,
        join_key: object | None = None,
    ) -> Envelope[PayloadT]:
        payload = self._payload_contract.snapshot(payload)
        payload_digest = _digest(
            self._payload_contract.digest(payload),
            "payload digest",
        )
        size = _positive_int(
            self._payload_contract.retained_nbytes(payload),
            "measured payload bytes",
        )
        if size > self.max_payload_bytes:
            raise ValueError("payload exceeds the stream contract max_payload_bytes")
        if self._join_key_contract is None:
            if join_key is not None:
                raise ValueError("this stream generation does not declare a join key")
            join_key_schema_fingerprint = None
        else:
            join_key = self._join_key_contract.snapshot(join_key)
            join_key_schema_fingerprint = self._join_key_contract.fingerprint
        with self._condition:
            if self._closed:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise StreamEndedEarly("cannot emit after end-of-stream")
            sequence = self._next_sequence
            if self._formal_consumer_claimed or self._formal_rebind_required:
                covering = tuple(
                    reservation
                    for reservation in self._reservations.values()
                    if reservation._state
                    in (ReservationState.ACTIVE, ReservationState.DRAINING)
                    and reservation._consumer_owner is not None
                    and reservation.start_sequence <= sequence < reservation.end_sequence
                )
                if len(covering) != 1:
                    raise ReservationStateError(
                        "formal stream emission requires one live bound reservation "
                        "covering the next sequence"
                    )
                reservation = covering[0]
                if (
                    self._formal_interval_start != reservation.start_sequence
                    or self._formal_interval_end != reservation.end_sequence
                ):
                    raise ReservationStateError(
                        "formal stream reservation differs from its frozen interval"
                    )
                reservation.trace_binding.validate(trace)
            for reservation in self._reservations.values():
                if (
                    reservation._state
                    in (
                        ReservationState.RESERVED,
                        ReservationState.ACTIVE,
                        ReservationState.DRAINING,
                    )
                    and reservation.start_sequence <= sequence < reservation.end_sequence
                    and reservation._consumer_owner is None
                ):
                    raise ReservationStateError(
                        "exact data cannot be emitted before its formal consumer is bound"
                    )
            selected_event_id = event_id_for_sequence(
                self.stream_id,
                self.generation,
                sequence,
            )
            event_ref = EventRef(
                self.stream_id,
                self.generation,
                sequence,
                selected_event_id,
                payload_digest,
            )
            envelope = Envelope(
                event_ref=event_ref,
                emitted_at=time.time(),
                captured_at=captured_at,
                payload_contract_fingerprint=self.payload_contract_fingerprint,
                trace=trace,
                payload=payload,
                join_key=join_key,
                join_key_schema_fingerprint=join_key_schema_fingerprint,
            )
            try:
                for reservation in self._reservations.values():
                    if reservation._state not in (
                        ReservationState.RESERVED,
                        ReservationState.ACTIVE,
                        ReservationState.DRAINING,
                    ):
                        continue
                    if not reservation.start_sequence <= sequence < reservation.end_sequence:
                        continue
                    unacked_events = sequence - reservation._ack_sequence + 1
                    if unacked_events > reservation.max_inflight_events:
                        raise StreamBackpressure(
                            "exact consumer exceeded its event backlog budget"
                        )
                    if reservation._unacked_bytes + size > reservation.max_inflight_bytes:
                        raise StreamBackpressure(
                            "exact consumer exceeded its byte backlog budget"
                        )
                self._trim_locked(extra_events=1, extra_bytes=size)
            except StreamBackpressure as error:
                if self.flow_control is ProducerFlowControl.BACKPRESSURE_CAPABLE:
                    raise
                overrun = RetentionOverrun(
                    "non-backpressure source exceeded frozen retention budget; "
                    "the generation is permanently invalid"
                )
                self._terminal_error = overrun
                for reservation in self._reservations.values():
                    if reservation._state in (
                        ReservationState.RESERVED,
                        ReservationState.ACTIVE,
                        ReservationState.DRAINING,
                    ):
                        reservation._state = ReservationState.FAILED
                self._close_generation_locked(overrun)
                raise overrun from error
            stored = _Stored(envelope, size)
            self._records[sequence] = stored
            self._order.append(sequence)
            self._retained_bytes += size
            self._next_sequence += 1
            for reservation in self._reservations.values():
                if (
                    reservation._state
                    in (ReservationState.RESERVED, ReservationState.ACTIVE, ReservationState.DRAINING)
                    and reservation.start_sequence <= sequence < reservation.end_sequence
                ):
                    reservation._unacked_bytes += size
                    if self._next_sequence >= reservation.end_sequence and reservation._state is ReservationState.ACTIVE:
                        reservation._state = ReservationState.DRAINING
            for monitor in tuple(self._monitors):
                monitor._offer(stored)
            self._condition.notify_all()
        return envelope

    def _close_generation_locked(self, error: StreamError | None) -> None:
        """Publish one terminal state to every monitor while the stream lock is held."""

        self._closed = True
        monitors = tuple(self._monitors)
        self._monitors.clear()
        for monitor in monitors:
            monitor._source_ended(error)
        self._condition.notify_all()

    def _finish(self, producer: AcquisitionProducer[PayloadT]) -> EndOfStream:
        with self._condition:
            if producer is not self._producer:
                raise PermissionError("terminal authority belongs to another stream")
            if self._terminal_error is not None:
                raise self._terminal_error
            if self._eos is not None:
                return self._eos
            if self._formal_rebind_required:
                raise StreamEndedEarly(
                    "formal stream cannot finish while replacement binding is required"
                )
            if (
                self._formal_interval_end is not None
                and self._next_sequence != self._formal_interval_end
            ):
                raise StreamEndedEarly(
                    "formal stream terminal sequence differs from its frozen interval"
                )
            nonce = object()
            self._eos = EndOfStream(
                _EOS_TOKEN,
                stream_id=self.stream_id,
                stream_generation=self.generation,
                end_sequence=self._next_sequence,
                ended_at=time.time(),
                owner=self,
                nonce=nonce,
            )
            self._close_generation_locked(None)
            return self._eos

    def _supersede(
        self,
        producer: AcquisitionProducer[PayloadT],
        replacement: StreamGenerationId,
    ) -> None:
        if producer is not self._producer:
            raise PermissionError("terminal authority belongs to another stream")
        if not isinstance(replacement, StreamGenerationId):
            raise TypeError("replacement must be StreamGenerationId")
        if replacement == self.generation:
            raise ValueError("replacement generation must differ from the active generation")
        with self._condition:
            if self._eos is not None:
                raise StreamEndedEarly("completed stream generation cannot be superseded")
            if isinstance(self._terminal_error, SchemaChanged):
                if self._terminal_error.replacement == replacement:
                    return
                raise StreamEndedEarly("stream generation was already superseded")
            if self._terminal_error is not None:
                raise StreamEndedEarly("stream already has a terminal failure")
            self._terminal_error = SchemaChanged(self.generation, replacement)
            self._close_generation_locked(self._terminal_error)

    def _fail(
        self,
        producer: AcquisitionProducer[PayloadT],
        error: StreamError,
    ) -> None:
        if producer is not self._producer:
            raise PermissionError("terminal authority belongs to another stream")
        if not isinstance(error, StreamError):
            raise TypeError("source failure must be a StreamError")
        with self._condition:
            if self._eos is not None:
                raise StreamEndedEarly("completed stream cannot fail")
            if self._terminal_error is not None:
                if self._terminal_error is error:
                    return
                raise StreamEndedEarly("stream already has a terminal failure")
            self._terminal_error = error
            self._close_generation_locked(error)

    def _owns_eos(self, eos: EndOfStream) -> bool:
        with self._condition:
            return (
                isinstance(eos, EndOfStream)
                and eos._owner is self
                and self._eos is eos
                and eos._nonce is self._eos._nonce
            )

    def _await_terminal(self, timeout: float) -> EndOfStream | None:
        """Observe source EOS/failure without depending on an external finish callback."""

        timeout = max(0.0, float(timeout))
        with self._condition:
            if self._terminal_error is not None:
                raise self._terminal_error
            if self._eos is not None:
                return self._eos
            if timeout:
                self._condition.wait(timeout)
            if self._terminal_error is not None:
                raise self._terminal_error
            return self._eos

    def _consume_exact(
        self,
        reservation: ExactReservation[PayloadT],
        delivery: Delivery[PayloadT],
        consumer: object,
        commit: Callable[[Envelope[PayloadT]], object],
    ) -> object:
        """Atomically validate authority, commit one cell, then advance its watermark."""

        with self._condition:
            cursor = self._validate_consumer_delivery_locked(
                reservation,
                delivery,
                consumer,
            )
            result = commit(delivery.envelope)
            cursor._ack_delivery(delivery, consumer)
            return result

    def _validate_consumer_delivery(
        self,
        reservation: ExactReservation[PayloadT],
        delivery: Delivery[PayloadT],
        consumer: object,
    ) -> None:
        """Validate one delivery without committing or advancing its watermark."""

        with self._condition:
            self._validate_consumer_delivery_locked(reservation, delivery, consumer)

    def _validate_consumer_delivery_locked(
        self,
        reservation: ExactReservation[PayloadT],
        delivery: Delivery[PayloadT],
        consumer: object,
    ) -> AcquisitionCursor[PayloadT]:
        self._require_consumer_locked(reservation, consumer)
        if reservation._state not in (ReservationState.ACTIVE, ReservationState.DRAINING):
            raise ReservationStateError("exact consumer reservation is not active")
        cursor = reservation._cursor
        if cursor is None or delivery._cursor is not cursor:
            raise PermissionError("Delivery belongs to another exact reservation")
        if cursor._stream is not self or cursor._inflight is not delivery:
            raise PermissionError("Delivery belongs to another stream authority")
        if delivery.acknowledged:
            raise ReservationStateError("Delivery was already acknowledged")
        envelope = delivery.envelope
        stored = self._records.get(envelope.sequence)
        if stored is None or stored.envelope is not envelope:
            raise PermissionError("Delivery no longer names its retained stream event")
        reservation.trace_binding.validate(envelope.trace)
        return cursor

    def _ack_consumer(
        self,
        reservation: ExactReservation[PayloadT],
        delivery: Delivery[PayloadT],
        consumer: object,
    ) -> None:
        """Advance an exact watermark only for its identity-bound consumer."""

        with self._condition:
            cursor = self._validate_consumer_delivery_locked(
                reservation,
                delivery,
                consumer,
            )
            cursor._ack_delivery(delivery, consumer)

    def _complete_consumer(
        self,
        reservation: ExactReservation[PayloadT],
        eos: EndOfStream,
        consumer: object,
        commit: Callable[[], object],
    ) -> object:
        """Complete a fully acknowledged exact consumer from its source receipt."""

        with self._condition:
            self._validate_consumer_completion_locked(reservation, eos, consumer)
            result = commit()
            reservation._state = ReservationState.COMPLETED
            self._trim_locked()
            self._condition.notify_all()
            return result

    def _validate_consumer_completion(
        self,
        reservation: ExactReservation[PayloadT],
        eos: EndOfStream,
        consumer: object,
    ) -> None:
        """Stage all fallible source-terminal checks without changing reservation state."""

        with self._condition:
            self._validate_consumer_completion_locked(reservation, eos, consumer)

    def _validate_consumer_completion_locked(
        self,
        reservation: ExactReservation[PayloadT],
        eos: EndOfStream,
        consumer: object,
    ) -> None:
        self._require_consumer_locked(reservation, consumer)
        if reservation._state not in (ReservationState.ACTIVE, ReservationState.DRAINING):
            raise ReservationStateError("exact consumer reservation cannot be completed")
        if not self._owns_eos(eos):
            raise PermissionError("EndOfStream belongs to another source authority")
        if eos.end_sequence != reservation.end_sequence:
            raise StreamEndedEarly(
                "source terminal sequence differs from the reserved formal interval"
            )
        if reservation._ack_sequence != reservation.end_sequence:
            raise ReservationStateError("formal interval is not fully acknowledged")

    def _claim_consumer(
        self,
        reservation: ExactReservation[PayloadT],
        consumer: object,
        *,
        source_contract_digest: str,
        source_schedule_digest: str,
        source_key_sequence_digest: str,
        chain_contract_digest: str,
        terminal: bool = False,
        downstream: ExactConsumerReadiness | None = None,
        owner_liveness: Callable[[], None] | None = None,
        owner_completion: Callable[[float], object] | None = None,
        owner_cancel: Callable[[str | None], bool] | None = None,
        processor_stage: ProcessorStageProvenance | None = None,
    ) -> ExactConsumerReadiness:
        _digest(source_contract_digest, "source contract digest")
        _digest(source_schedule_digest, "source schedule digest")
        _digest(source_key_sequence_digest, "source key-sequence digest")
        _digest(chain_contract_digest, "chain contract digest")
        if terminal == (downstream is not None):
            raise ValueError(
                "exact consumer must be either a terminal dataset sink or bind one downstream"
            )
        if terminal and any(
            callback is not None
            for callback in (owner_liveness, owner_completion, owner_cancel)
        ):
            raise ValueError(
                "terminal dataset sink does not accept processor lifecycle callbacks"
            )
        if terminal and processor_stage is not None:
            raise ValueError("terminal dataset sink does not declare a processor stage")
        if downstream is not None and not all(
            callable(callback)
            for callback in (owner_liveness, owner_completion, owner_cancel)
        ):
            raise TypeError(
                "processor exact consumer requires liveness, completion, and cancel callbacks"
            )
        if downstream is not None and not isinstance(
            processor_stage,
            ProcessorStageProvenance,
        ):
            raise TypeError("processor exact consumer requires processor_stage")
        with self._condition:
            if self._reservations.get(reservation._token) is not reservation:
                raise ReservationStateError("exact consumer reservation is not registered")
            if reservation._consumer_owner is not None:
                raise ReservationStateError("reservation already has an exact consumer")
            if self._formal_consumer_claimed:
                raise ReservationStateError(
                    "stream generation already claimed its formal exact consumer"
                )
            if reservation._state is not ReservationState.ACTIVE:
                raise ReservationStateError("exact consumer requires an ACTIVE reservation")
            if downstream is not None:
                downstream._validate_terminal_sink()
                terminal_reservation = downstream._terminal_reservation
                terminal_consumer = downstream._terminal_consumer.get()
                if terminal_consumer is None:
                    raise ReservationStateError(
                        "downstream terminal consumer is no longer live"
                    )
                owner_liveness = (
                    owner_liveness,
                    *downstream._resolved_owner_liveness(),
                )
                processor_stages = (
                    processor_stage,
                    *downstream._processor_stages,
                )
            else:
                terminal_reservation = reservation
                terminal_consumer = consumer
                owner_liveness = ()
                owner_completion = None
                owner_cancel = None
                processor_stages = ()
            readiness = ExactConsumerReadiness(
                _READINESS_TOKEN,
                source_reservation=reservation,
                source_consumer=consumer,
                source_contract_digest=source_contract_digest,
                source_schedule_digest=source_schedule_digest,
                source_key_sequence_digest=source_key_sequence_digest,
                chain_contract_digest=chain_contract_digest,
                terminal_reservation=terminal_reservation,
                terminal_consumer=terminal_consumer,
                owner_liveness=owner_liveness,
                owner_completion=owner_completion,
                owner_cancel=owner_cancel,
                processor_stages=processor_stages,
            )
            if downstream is not None:
                downstream._claim_binding(consumer)
            # Final no-fail authority commit.  Failed validation before this
            # point is retryable.  A later zero-event failed preflight may also
            # release the claim; publishing the first event makes it a permanent
            # generation-lifetime tombstone.
            reservation._consumer_owner = consumer
            self._formal_consumer_claimed = True
            self._formal_rebind_required = False
            self._formal_interval_start = reservation.start_sequence
            self._formal_interval_end = reservation.end_sequence
            return readiness

    def _abort_consumer(
        self,
        reservation: ExactReservation[PayloadT],
        consumer: object,
        commit: Callable[[], None],
        *,
        cancelled: bool = False,
    ) -> None:
        with self._condition:
            self._require_consumer_locked(reservation, consumer)
            if reservation._state is ReservationState.COMPLETED:
                raise ReservationStateError("completed reservation cannot be aborted")
            commit()
            if reservation._state not in (
                ReservationState.FAILED,
                ReservationState.CANCELLED,
            ):
                reservation._state = (
                    ReservationState.CANCELLED if cancelled else ReservationState.FAILED
                )
            self._trim_locked()
            self._condition.notify_all()

    def _require_consumer_locked(
        self,
        reservation: ExactReservation[PayloadT],
        consumer: object,
    ) -> None:
        """Validate the shared registered-consumer invariant under the stream lock."""

        if self._reservations.get(reservation._token) is not reservation:
            raise ReservationStateError("exact consumer reservation is not registered")
        if reservation._consumer_owner is not consumer:
            raise PermissionError("reservation belongs to another exact consumer")

    def _ack(self, token: object, sequence: int) -> None:
        try:
            reservation = self._reservations[token]
        except KeyError as exc:
            raise ReservationStateError("reservation is not active") from exc
        if reservation._state not in (ReservationState.ACTIVE, ReservationState.DRAINING):
            raise ReservationStateError("reservation cannot acknowledge in its current state")
        if sequence != reservation._ack_sequence:
            raise ValueError("exact acknowledgement must be strictly ordered")
        stored = self._records.get(sequence)
        if stored is None:
            raise StreamGap(sequence, self._earliest_retained_locked(), self._next_sequence)
        reservation._ack_sequence += 1
        reservation._unacked_bytes -= stored.payload_bytes
        self._trim_locked()
        self._condition.notify_all()

    def _release_reservation(self, token: object) -> None:
        with self._condition:
            try:
                reservation = self._reservations[token]
            except KeyError as exc:
                raise ReservationStateError("unknown exact reservation") from exc
            if reservation._state not in (
                ReservationState.COMPLETED,
                ReservationState.FAILED,
                ReservationState.CANCELLED,
            ):
                raise ReservationStateError("reservation must complete or abort before release")
            retryable_zero_event_preflight = (
                reservation._consumer_owner is not None
                and reservation._state
                in (ReservationState.FAILED, ReservationState.CANCELLED)
                and self._next_sequence == reservation.start_sequence
            )
            reservation._state = ReservationState.RELEASED
            self._reservations.pop(token)
            if retryable_zero_event_preflight:
                # A fully bound graph that failed before publishing its first
                # event never became the generation's data consumer.  Reusing
                # that still-empty generation is safe; any emitted event makes
                # the lifetime tombstone permanent, even if it was never acked.
                self._formal_consumer_claimed = False
                self._formal_rebind_required = True
            # A released capability remains inspectable as a tombstone, but it
            # must no longer retain the live consumer graph or cursor/Delivery
            # chain.  All identity checks above happen before this no-fail
            # lifetime commit.
            cursor = reservation._cursor
            if cursor is not None:
                cursor._inflight = None
            reservation._consumer_owner = None
            reservation._cursor = None
            self._trim_locked()
            self._condition.notify_all()

    def _remove_monitor(self, monitor: MonitorTap[PayloadT]) -> None:
        with self._condition:
            self._monitors.discard(monitor)

    def _earliest_retained_locked(self) -> int:
        return self._order[0] if self._order else self._next_sequence

    def _protected_sequence_locked(self) -> int | None:
        watermarks = [
            reservation._ack_sequence
            for reservation in self._reservations.values()
            if reservation._state
            in (ReservationState.RESERVED, ReservationState.ACTIVE, ReservationState.DRAINING)
        ]
        return min(watermarks) if watermarks else None

    def _trim_locked(self, *, extra_events: int = 0, extra_bytes: int = 0) -> None:
        protected = self._protected_sequence_locked()
        prospective_order = deque(self._order)
        prospective_bytes = self._retained_bytes
        removals: list[int] = []
        while prospective_order and protected is not None and prospective_order[0] < protected:
            oldest = prospective_order.popleft()
            prospective_bytes -= self._records[oldest].payload_bytes
            removals.append(oldest)
        while prospective_order and (
            len(prospective_order) + extra_events > self.retention_events
            or prospective_bytes + extra_bytes > self.retention_bytes
        ):
            oldest = prospective_order[0]
            if protected is not None and oldest >= protected:
                raise StreamBackpressure("stream retention is pinned by an unacknowledged exact cursor")
            prospective_order.popleft()
            prospective_bytes -= self._records[oldest].payload_bytes
            removals.append(oldest)
        if len(prospective_order) + extra_events > self.retention_events:
            raise StreamBackpressure("stream event retention capacity is exhausted")
        if prospective_bytes + extra_bytes > self.retention_bytes:
            raise StreamBackpressure("stream byte retention capacity is exhausted")
        for oldest in removals:
            actual = self._order.popleft()
            if actual != oldest:
                raise RuntimeError("stream retention order changed while locked")
            removed = self._records.pop(oldest)
            self._retained_bytes -= removed.payload_bytes


__all__ = [
    "AcquisitionCursor",
    "AcquisitionProducer",
    "AcquisitionStream",
    "ArtifactInputRef",
    "Delivery",
    "Envelope",
    "EndOfStream",
    "EventId",
    "EventRef",
    "EventSpanRef",
    "event_id_for_sequence",
    "event_ref_to_tree",
    "ExactConsumerReadiness",
    "ExactReservation",
    "JoinKeyContract",
    "MonitorTap",
    "MonitorUpdate",
    "OrderedEventSpanHasher",
    "PayloadContract",
    "ProducerFlowControl",
    "ProcessorStageProvenance",
    "ReservationCapacityExceeded",
    "RetentionOverrun",
    "ReservationState",
    "ReservationStateError",
    "SchemaChanged",
    "SourceFailed",
    "StreamBackpressure",
    "StreamEndedEarly",
    "StreamError",
    "StreamGap",
    "StreamId",
    "TraceContext",
    "TraceBinding",
]
