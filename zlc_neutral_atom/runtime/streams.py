"""Bounded exact acquisition streams and non-blocking monitor taps."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral
from typing import Callable, Generic, Protocol, TypeVar

from zlc_data import DataBlock, DataPatch, StreamGenerationId


PayloadT = TypeVar("PayloadT")
_EOS_TOKEN = object()
_DELIVERY_TOKEN = object()
_CURSOR_TOKEN = object()
_RESERVATION_TOKEN = object()
_MONITOR_TOKEN = object()
_MONITOR_UPDATE_TOKEN = object()
_STREAM_TOKEN = object()
_PRODUCER_TOKEN = object()


class PayloadContract(Protocol[PayloadT]):
    fingerprint: str
    max_retained_nbytes: int

    def snapshot(self, payload: PayloadT) -> PayloadT: ...

    def validate(self, payload: PayloadT) -> None: ...

    def retained_nbytes(self, payload: PayloadT) -> int: ...


class JoinKeyContract(Protocol):
    fingerprint: str

    def snapshot(self, key: object) -> object: ...

    def validate(self, key: object) -> None: ...


def _canonical_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _digest(value: str, field: str) -> str:
    value = _canonical_text(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _positive_int(value: int, field: str) -> int:
    value = _nonnegative_int(value, field)
    if value == 0:
        raise ValueError(f"{field} must be positive")
    return value


def _finite_time(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite timestamp")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite timestamp")
    return value


def _contains_materialization(value: object, seen: set[int] | None = None) -> bool:
    if isinstance(value, (DataBlock, DataPatch)):
        return True
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

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        if not isinstance(self.event_id, EventId):
            raise TypeError("event_id must be EventId")


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


@dataclass(frozen=True)
class ArtifactInputRef:
    typed_ref: object
    content_digest: str

    def __post_init__(self) -> None:
        _digest(self.content_digest, "content_digest")


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
    event_id: EventId
    stream_id: StreamId
    stream_generation: StreamGenerationId
    sequence: int
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
        if not isinstance(self.event_id, EventId):
            raise TypeError("event_id must be EventId")
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(self.stream_generation, StreamGenerationId):
            raise TypeError("stream_generation must be StreamGenerationId")
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        object.__setattr__(self, "emitted_at", _finite_time(self.emitted_at, "emitted_at"))
        object.__setattr__(self, "captured_at", _finite_time(self.captured_at, "captured_at"))
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
        return EventRef(
            self.stream_id,
            self.stream_generation,
            self.sequence,
            self.event_id,
        )


class EndOfStream:
    """Opaque terminal receipt minted exactly once by an AcquisitionStream."""

    __slots__ = (
        "_stream_id",
        "_stream_generation",
        "_end_sequence",
        "_ended_at",
        "_owner",
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
        object.__setattr__(self, "_ended_at", _finite_time(ended_at, "ended_at"))
        object.__setattr__(self, "_owner", owner)
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
        self._materializer: object | None = None

    @property
    def state(self) -> ReservationState:
        with self._stream._condition:
            return self._state

    @property
    def acknowledged_sequence(self) -> int:
        with self._stream._condition:
            return self._ack_sequence

    @property
    def materializer_bound(self) -> bool:
        """Whether the exact reservation already has its single dataset owner."""

        with self._stream._condition:
            return self._materializer is not None

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
            if self._materializer is not None:
                raise ReservationStateError(
                    "reservation completion belongs to its bound DatasetBuilder"
                )
            if self._state not in (ReservationState.ACTIVE, ReservationState.DRAINING):
                raise ReservationStateError("reservation is not active or draining")
            if self._ack_sequence != self.end_sequence:
                raise ReservationStateError("reservation cannot complete before every event is acked")
            self._state = ReservationState.COMPLETED
            self._stream._condition.notify_all()

    def abort(self, *, cancelled: bool = False) -> None:
        with self._stream._condition:
            if self._materializer is not None:
                raise ReservationStateError("bound reservation abort belongs to DatasetBuilder")
            if self._state in (ReservationState.RELEASED, ReservationState.COMPLETED):
                raise ReservationStateError("completed/released reservation cannot be aborted")
            self._state = ReservationState.CANCELLED if cancelled else ReservationState.FAILED
            self._stream._trim_locked()
            self._stream._condition.notify_all()

    def release(self) -> None:
        self._stream._release_reservation(self._token)

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

    def _ack_delivery(self, delivery: Delivery[PayloadT]) -> None:
        with self._stream._condition:
            if self._inflight is None or delivery is not self._inflight:
                raise ValueError("ack must consume this cursor's current delivery")
            if delivery._acked:
                raise ValueError("delivery acknowledgement is single-use")
            envelope = delivery.envelope
            if envelope.sequence != self._next_sequence:
                raise ValueError("delivery sequence does not match cursor")
            if self._reservation_token is not None:
                self._stream._ack(self._reservation_token, envelope.sequence)
            object.__setattr__(delivery, "_acked", True)
            self._next_sequence += 1
            self._inflight = None


class MonitorUpdate(Generic[PayloadT]):
    """Opaque update issued by one concrete monitor tap."""

    __slots__ = ("_tap", "_envelope", "_missed")

    def __init__(
        self,
        authority: object,
        *,
        tap: "MonitorTap[PayloadT]",
        envelope: Envelope[PayloadT],
        missed: int,
    ) -> None:
        if authority is not _MONITOR_UPDATE_TOKEN:
            raise PermissionError("MonitorUpdate can only be minted by MonitorTap")
        if not isinstance(envelope, Envelope):
            raise TypeError("envelope must be Envelope")
        object.__setattr__(self, "_tap", tap)
        object.__setattr__(self, "_envelope", envelope)
        object.__setattr__(self, "_missed", _nonnegative_int(missed, "missed"))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("MonitorUpdate is immutable")

    @property
    def envelope(self) -> Envelope[PayloadT]:
        return self._envelope

    @property
    def missed(self) -> int:
        return self._missed


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
        self._condition = threading.Condition(threading.Lock())
        self._queue: deque[_Stored[PayloadT]] = deque()
        self._retained_bytes = 0
        self._missed = 0
        self._closed = False
        self._source_finished = False
        self._terminal_error: StreamError | None = None

    @property
    def retained_bytes(self) -> int:
        with self._condition:
            return self._retained_bytes

    def _offer(self, stored: _Stored[PayloadT]) -> None:
        with self._condition:
            if self._closed or self._source_finished:
                return
            if stored.payload_bytes > self.max_bytes:
                self._missed += len(self._queue) + 1
                self._queue.clear()
                self._retained_bytes = 0
                self._condition.notify_all()
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

    def next(self, timeout: float | None = None) -> MonitorUpdate[PayloadT]:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._queue:
                if self._closed:
                    raise StreamEndedEarly("monitor tap is closed")
                if self._source_finished:
                    if self._terminal_error is not None:
                        raise self._terminal_error
                    raise StreamEndedEarly("monitor source reached end-of-stream")
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for monitor event")
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()
            stored = self._queue.popleft()
            self._retained_bytes -= stored.payload_bytes
            missed, self._missed = self._missed, 0
            return MonitorUpdate(
                _MONITOR_UPDATE_TOKEN,
                tap=self,
                envelope=stored.envelope,
                missed=missed,
            )

    def latest(self) -> MonitorUpdate[PayloadT]:
        with self._condition:
            if not self._queue:
                if self._source_finished:
                    if self._terminal_error is not None:
                        raise self._terminal_error
                    raise StreamEndedEarly("monitor source reached end-of-stream")
                raise LookupError("monitor tap has no retained event")
            while len(self._queue) > 1:
                removed = self._queue.popleft()
                self._retained_bytes -= removed.payload_bytes
                self._missed += 1
            stored = self._queue.popleft()
            self._retained_bytes -= stored.payload_bytes
            missed, self._missed = self._missed, 0
            return MonitorUpdate(
                _MONITOR_UPDATE_TOKEN,
                tap=self,
                envelope=stored.envelope,
                missed=missed,
            )

    def _owns_update(self, update: MonitorUpdate[PayloadT]) -> bool:
        return isinstance(update, MonitorUpdate) and update._tap is self

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
            self._queue.clear()
            self._retained_bytes = 0
            self._condition.notify_all()


class AcquisitionProducer(Generic[PayloadT]):
    """Exclusive write/terminal authority retained by the source owner lane."""

    __slots__ = ("_stream",)

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
        for method in ("snapshot", "validate", "retained_nbytes"):
            if not callable(getattr(payload_contract, method, None)):
                raise TypeError(f"payload_contract.{method} must be callable")
        if join_key_contract is not None:
            _digest(join_key_contract.fingerprint, "join key contract fingerprint")
            for method in ("snapshot", "validate"):
                if not callable(getattr(join_key_contract, method, None)):
                    raise TypeError(f"join_key_contract.{method} must be callable")
        self._payload_contract = payload_contract
        self._join_key_contract = join_key_contract
        self._condition = threading.Condition(threading.RLock())
        self._records: dict[int, _Stored[PayloadT]] = {}
        self._order: deque[int] = deque()
        self._retained_bytes = 0
        self._next_sequence = 0
        self._event_namespace = f"{self.stream_id.value}:{self.generation.value}"
        self._reservations: dict[object, ExactReservation[PayloadT]] = {}
        self._monitors: set[MonitorTap[PayloadT]] = set()
        self._closed = False
        self._terminal_error: StreamError | None = None
        self._eos: EndOfStream | None = None
        self._producer = AcquisitionProducer(_PRODUCER_TOKEN, self)

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
        return stream, stream._producer

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
                    "one stream generation has exactly one formal materializer"
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
        self._payload_contract.validate(payload)
        if _contains_materialization(payload):
            raise TypeError("DataBlock/DataPatch are materialization values, not stream payloads")
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
            self._join_key_contract.validate(join_key)
            join_key_schema_fingerprint = self._join_key_contract.fingerprint
        with self._condition:
            if self._closed:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise StreamEndedEarly("cannot emit after end-of-stream")
            sequence = self._next_sequence
            selected_event_id = EventId(f"{self._event_namespace}:{sequence}")
            envelope = Envelope(
                event_id=selected_event_id,
                stream_id=self.stream_id,
                stream_generation=self.generation,
                sequence=sequence,
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
                self._closed = True
                for reservation in self._reservations.values():
                    if reservation._state in (
                        ReservationState.RESERVED,
                        ReservationState.ACTIVE,
                        ReservationState.DRAINING,
                    ):
                        reservation._state = ReservationState.FAILED
                for monitor in tuple(self._monitors):
                    monitor._source_ended(overrun)
                self._condition.notify_all()
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

    def _finish(self, producer: AcquisitionProducer[PayloadT]) -> EndOfStream:
        with self._condition:
            if producer is not self._producer:
                raise PermissionError("terminal authority belongs to another stream")
            if self._terminal_error is not None:
                raise self._terminal_error
            if self._eos is not None:
                return self._eos
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
            self._closed = True
            for monitor in tuple(self._monitors):
                monitor._source_ended(None)
            self._condition.notify_all()
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
            self._closed = True
            for monitor in tuple(self._monitors):
                monitor._source_ended(self._terminal_error)
            self._condition.notify_all()

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
            self._closed = True
            for monitor in tuple(self._monitors):
                monitor._source_ended(error)
            self._condition.notify_all()

    def _owns_eos(self, eos: EndOfStream) -> bool:
        with self._condition:
            return (
                isinstance(eos, EndOfStream)
                and eos._owner is self
                and self._eos is eos
                and eos._nonce is self._eos._nonce
            )

    def _consume_exact(
        self,
        reservation: ExactReservation[PayloadT],
        delivery: Delivery[PayloadT],
        materializer: object,
        commit: Callable[[Envelope[PayloadT]], object],
    ) -> object:
        """Atomically validate authority, commit one cell, then advance its watermark."""

        with self._condition:
            registered = self._reservations.get(reservation._token)
            if registered is not reservation:
                raise ReservationStateError("DatasetBuilder reservation is not registered")
            if reservation._materializer is not materializer:
                raise PermissionError("reservation belongs to another DatasetBuilder")
            if reservation._state not in (ReservationState.ACTIVE, ReservationState.DRAINING):
                raise ReservationStateError("DatasetBuilder reservation is not active")
            cursor = reservation._cursor
            if cursor is None or delivery._cursor is not cursor:
                raise PermissionError("Delivery belongs to another exact reservation")
            if cursor._stream is not self or cursor._inflight is not delivery:
                raise PermissionError("Delivery belongs to another stream authority")
            if delivery.acknowledged:
                raise ReservationStateError("Delivery was already acknowledged")
            reservation.trace_binding.validate(delivery.envelope.trace)
            result = commit(delivery.envelope)
            cursor._ack_delivery(delivery)
            return result

    def _seal_exact(
        self,
        reservation: ExactReservation[PayloadT],
        eos: EndOfStream,
        materializer: object,
        commit: Callable[[], object],
    ) -> object:
        """Seal only the fully acknowledged reservation owned by this terminal receipt."""

        with self._condition:
            if self._reservations.get(reservation._token) is not reservation:
                raise ReservationStateError("DatasetBuilder reservation is not registered")
            if reservation._materializer is not materializer:
                raise PermissionError("reservation belongs to another DatasetBuilder")
            if reservation._state not in (ReservationState.ACTIVE, ReservationState.DRAINING):
                raise ReservationStateError("DatasetBuilder reservation cannot be sealed")
            if not self._owns_eos(eos):
                raise PermissionError("EndOfStream belongs to another source authority")
            if eos.end_sequence != reservation.end_sequence:
                raise StreamEndedEarly(
                    "source terminal sequence differs from the reserved formal interval"
                )
            if reservation._ack_sequence != reservation.end_sequence:
                raise ReservationStateError("formal interval is not fully acknowledged")
            result = commit()
            reservation._state = ReservationState.COMPLETED
            self._trim_locked()
            self._condition.notify_all()
            return result

    def _claim_materializer(
        self,
        reservation: ExactReservation[PayloadT],
        materializer: object,
    ) -> None:
        with self._condition:
            if self._reservations.get(reservation._token) is not reservation:
                raise ReservationStateError("DatasetBuilder reservation is not registered")
            if reservation._materializer is not None:
                raise ReservationStateError("reservation already has a DatasetBuilder")
            reservation._materializer = materializer

    def _abort_materializer(
        self,
        reservation: ExactReservation[PayloadT],
        materializer: object,
        commit: Callable[[], None],
    ) -> None:
        with self._condition:
            if self._reservations.get(reservation._token) is not reservation:
                raise ReservationStateError("DatasetBuilder reservation is not registered")
            if reservation._materializer is not materializer:
                raise PermissionError("reservation belongs to another DatasetBuilder")
            if reservation._state is ReservationState.COMPLETED:
                raise ReservationStateError("completed reservation cannot be aborted")
            commit()
            if reservation._state not in (
                ReservationState.FAILED,
                ReservationState.CANCELLED,
            ):
                reservation._state = ReservationState.FAILED
            self._trim_locked()
            self._condition.notify_all()

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
            reservation._state = ReservationState.RELEASED
            self._reservations.pop(token)
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
    "ExactReservation",
    "JoinKeyContract",
    "MonitorTap",
    "MonitorUpdate",
    "PayloadContract",
    "ProducerFlowControl",
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
