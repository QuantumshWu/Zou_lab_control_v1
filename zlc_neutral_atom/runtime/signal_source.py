"""Source-neutral live signal events over an already-running producer stream.

The source owns event selection and Value projection.  Consumers see only a
declared ``ValueSchema`` and fresh immutable values with their original stream
identity/trace; they never receive the producer's physical payload type.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from zlc_data import (
    INVALID,
    VALID,
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    CommittedTransform,
    ComponentValidity,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    Invalid,
    OwnedSnapshot,
    PointLayout,
    RowComponentValidity,
    Selection,
    Valid,
    ValidityContract,
    Value,
    ValueSchema,
    apply_transform,
    commit_transform,
    committed_transform_from_tree,
    committed_transform_to_tree,
    resolve_transformed_schema,
    value_schema_from_tree,
    value_schema_to_tree,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    Envelope,
    EventRef,
    FollowTap,
    ProcessorStageProvenance,
    TraceContext,
)
from zlc_storage import (
    canonical_text,
    decode,
    encode,
    exact_mapping,
    finite_real,
    positive_integer,
    sha256_digest,
    sha256_text,
)


PayloadT = TypeVar("PayloadT")
_CURSOR_TOKEN = object()
SIGNAL_ASSOCIATION_EVIDENCE_SCHEMA = (
    "zlc_neutral_atom.runtime.signal-association-evidence"
)
_SIGNAL_REPEAT_AXIS = AxisSpec(
    AxisId("zlc_neutral_atom.signal-event.repeat"),
    "signal event repeat",
    REPEAT,
    1,
    (0,),
)
_SIGNAL_POINT_AXIS = AxisSpec(
    AxisId("zlc_neutral_atom.signal-event.point"),
    "signal event point",
    SCAN_POINT,
    1,
    (0,),
)


class SignalAssociationUnavailable(RuntimeError):
    """The producer cannot prove that selected events belong to a pulse cause."""


@dataclass(frozen=True, slots=True)
class SignalAssociationScheduleRequirement:
    """Physical trigger schedules an associated producer needs preserved.

    This value does not identify a Camera or prescribe how a producer proves
    causation.  It only tells a pulse compiler which physical trigger lanes
    must remain in the compiled artifact and terminal evidence.  Grouping and
    cardinality remain producer-owned admission checks.
    """

    trigger_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trigger_channels, tuple):
            raise TypeError("association trigger_channels must be a tuple")
        channels = tuple(
            canonical_text(channel, "association trigger channel")
            for channel in self.trigger_channels
        )
        if len(channels) != len(set(channels)):
            raise ValueError("association trigger channels must be unique")
        object.__setattr__(self, "trigger_channels", channels)


@dataclass(frozen=True, slots=True)
class SignalProjectionAuthority:
    """One schema-committed authoritative projection of live signal Values."""

    input_value_schema: ValueSchema
    committed_transform: CommittedTransform | None
    output_value_schema: ValueSchema

    def __post_init__(self) -> None:
        if not isinstance(self.input_value_schema, ValueSchema):
            raise TypeError("input_value_schema must be ValueSchema")
        if not isinstance(self.output_value_schema, ValueSchema):
            raise TypeError("output_value_schema must be ValueSchema")
        transform = self.committed_transform
        if transform is None:
            if self.output_value_schema != self.input_value_schema:
                raise ValueError(
                    "identity signal projection must preserve its ValueSchema"
                )
            return
        if not isinstance(transform, CommittedTransform):
            raise TypeError(
                "committed_transform must be CommittedTransform or None"
            )
        expected = _resolve_signal_projection_output(
            self.input_value_schema,
            transform,
        )
        if expected != self.output_value_schema:
            raise ValueError(
                "signal projection output differs from its committed transform"
            )

    @property
    def input_schema_fingerprint(self) -> str:
        return self.input_value_schema.fingerprint

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_value_schema.fingerprint


def signal_projection_authority_to_tree(
    value: SignalProjectionAuthority,
) -> dict[str, object]:
    if not isinstance(value, SignalProjectionAuthority):
        raise TypeError("value must be SignalProjectionAuthority")
    return {
        "input_value_schema": value_schema_to_tree(value.input_value_schema),
        "input_schema_fingerprint": value.input_schema_fingerprint,
        "committed_transform": (
            None
            if value.committed_transform is None
            else committed_transform_to_tree(value.committed_transform)
        ),
        "output_value_schema": value_schema_to_tree(value.output_value_schema),
        "output_schema_fingerprint": value.output_schema_fingerprint,
    }


def signal_projection_authority_from_tree(tree: object) -> SignalProjectionAuthority:
    data = exact_mapping(
        tree,
        {
            "input_value_schema",
            "input_schema_fingerprint",
            "committed_transform",
            "output_value_schema",
            "output_schema_fingerprint",
        },
        "SignalProjectionAuthority",
        discriminator=None,
    )
    value = SignalProjectionAuthority(
        value_schema_from_tree(data["input_value_schema"]),
        (
            None
            if data["committed_transform"] is None
            else committed_transform_from_tree(data["committed_transform"])
        ),
        value_schema_from_tree(data["output_value_schema"]),
    )
    if signal_projection_authority_to_tree(value) != tree:
        raise ValueError("SignalProjectionAuthority tree is non-canonical")
    return value


@dataclass(frozen=True, slots=True)
class SignalAssociationRequest:
    """Pulse cause that a producer must associate with an exact event group."""

    association_id: str
    cause_id: str
    cause_digest: str
    expected_event_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "association_id",
            canonical_text(self.association_id, "signal association_id"),
        )
        object.__setattr__(
            self,
            "cause_id",
            canonical_text(self.cause_id, "signal association cause_id"),
        )
        object.__setattr__(
            self,
            "cause_digest",
            sha256_text(self.cause_digest, "signal association cause_digest"),
        )
        object.__setattr__(
            self,
            "expected_event_count",
            positive_integer(
                self.expected_event_count,
                "signal association expected_event_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class SignalAssociationEvidence:
    """Canonical producer evidence bound to one request and terminal receipt."""

    request: SignalAssociationRequest
    terminal_evidence_digest: str
    evidence_schema_id: str
    canonical_evidence: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.request, SignalAssociationRequest):
            raise TypeError("signal association evidence requires its request")
        schema_id = canonical_text(
            self.evidence_schema_id,
            "signal association evidence_schema_id",
        )
        terminal_digest = sha256_text(
            self.terminal_evidence_digest,
            "signal association terminal_evidence_digest",
        )
        if not isinstance(self.canonical_evidence, bytes):
            raise TypeError("canonical_evidence must be immutable canonical bytes")
        payload = decode(self.canonical_evidence)
        if not isinstance(payload, dict):
            raise TypeError("canonical_evidence must decode to a mapping")
        required = {
            "schema": schema_id,
            "association_id": self.request.association_id,
            "cause_id": self.request.cause_id,
            "cause_digest": self.request.cause_digest,
            "expected_event_count": self.request.expected_event_count,
            "terminal_evidence_digest": terminal_digest,
        }
        if any(payload.get(name) != value for name, value in required.items()):
            raise ValueError(
                "canonical_evidence is not self-bound to its request and terminal"
            )
        object.__setattr__(self, "evidence_schema_id", schema_id)
        object.__setattr__(self, "terminal_evidence_digest", terminal_digest)
        object.__setattr__(self, "canonical_evidence", bytes(self.canonical_evidence))

    @property
    def evidence_digest(self) -> str:
        return sha256_digest(self.canonical_evidence)


def _signal_association_request_to_tree(
    value: SignalAssociationRequest,
) -> dict[str, object]:
    if not isinstance(value, SignalAssociationRequest):
        raise TypeError("value must be SignalAssociationRequest")
    return {
        "association_id": value.association_id,
        "cause_id": value.cause_id,
        "cause_digest": value.cause_digest,
        "expected_event_count": value.expected_event_count,
    }


def _signal_association_request_from_tree(tree: object) -> SignalAssociationRequest:
    data = exact_mapping(
        tree,
        {
            "association_id",
            "cause_id",
            "cause_digest",
            "expected_event_count",
        },
        "SignalAssociationRequest",
        discriminator=None,
    )
    value = SignalAssociationRequest(
        data["association_id"],
        data["cause_id"],
        data["cause_digest"],
        data["expected_event_count"],
    )
    if _signal_association_request_to_tree(value) != tree:
        raise ValueError("SignalAssociationRequest tree is non-canonical")
    return value


def signal_association_evidence_to_tree(
    value: SignalAssociationEvidence,
) -> dict[str, object]:
    if not isinstance(value, SignalAssociationEvidence):
        raise TypeError("value must be SignalAssociationEvidence")
    return {
        "schema": SIGNAL_ASSOCIATION_EVIDENCE_SCHEMA,
        "request": _signal_association_request_to_tree(value.request),
        "terminal_evidence_digest": value.terminal_evidence_digest,
        "evidence_schema_id": value.evidence_schema_id,
        "canonical_evidence": value.canonical_evidence,
    }


def signal_association_evidence_from_tree(tree: object) -> SignalAssociationEvidence:
    data = exact_mapping(
        tree,
        {
            "schema",
            "request",
            "terminal_evidence_digest",
            "evidence_schema_id",
            "canonical_evidence",
        },
        SIGNAL_ASSOCIATION_EVIDENCE_SCHEMA,
    )
    value = SignalAssociationEvidence(
        _signal_association_request_from_tree(data["request"]),
        data["terminal_evidence_digest"],
        data["evidence_schema_id"],
        data["canonical_evidence"],
    )
    if signal_association_evidence_to_tree(value) != tree:
        raise ValueError("SignalAssociationEvidence tree is non-canonical")
    return value


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """One fresh source value and the provenance of that exact publication."""

    value: Value
    event_ref: EventRef
    trace: TraceContext
    captured_at: float
    processor_stages: tuple[ProcessorStageProvenance, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.value, Value):
            raise TypeError("signal event value must be Value")
        if not isinstance(self.event_ref, EventRef):
            raise TypeError("signal event reference must be EventRef")
        if not isinstance(self.trace, TraceContext):
            raise TypeError("signal event trace must be TraceContext")
        object.__setattr__(
            self,
            "captured_at",
            finite_real(self.captured_at, "captured_at"),
        )
        object.__setattr__(
            self,
            "processor_stages",
            _snapshot_processor_stages(self.processor_stages),
        )


def _snapshot_processor_stages(
    values: tuple[ProcessorStageProvenance, ...],
) -> tuple[ProcessorStageProvenance, ...]:
    if not isinstance(values, tuple):
        raise TypeError("processor_stages must be an immutable tuple")
    stages = tuple(values)
    if any(not isinstance(stage, ProcessorStageProvenance) for stage in stages):
        raise TypeError(
            "processor_stages must contain ProcessorStageProvenance values"
        )
    return tuple(
        ProcessorStageProvenance(
            stage.processor_binding_digest,
            stage.direct_artifact_inputs,
        )
        for stage in stages
    )


@dataclass(frozen=True, slots=True)
class SignalOutputProjection(Generic[PayloadT]):
    """Owner-declared schema and event selector/projector for one output."""

    value_schema: ValueSchema
    project: Callable[[Envelope[PayloadT]], Value | None]

    def __post_init__(self) -> None:
        if not isinstance(self.value_schema, ValueSchema):
            raise TypeError("signal output value_schema must be ValueSchema")
        if not callable(self.project):
            raise TypeError("signal output project must be callable")


class SignalEventCursor(Generic[PayloadT]):
    """Software-order cursor that never exposes the producer payload type.

    Being opened before another operation proves only subscription order.  It
    deliberately has no API for claiming that a later event was caused by a
    pulse or by any other physical action.
    """

    __slots__ = ("_closed", "_processor_stages", "_projection", "_tap")

    def __init__(
        self,
        authority: object,
        *,
        tap: FollowTap[PayloadT],
        projection: SignalOutputProjection[PayloadT],
        processor_stages: tuple[ProcessorStageProvenance, ...],
    ) -> None:
        if authority is not _CURSOR_TOKEN:
            raise PermissionError(
                "SignalEventCursor can only be minted by StreamSignalEventSource"
            )
        if not isinstance(tap, FollowTap):
            raise TypeError("signal cursor requires a FollowTap")
        if not isinstance(projection, SignalOutputProjection):
            raise TypeError("signal cursor requires a SignalOutputProjection")
        self._tap = tap
        self._projection = projection
        self._processor_stages = _snapshot_processor_stages(processor_stages)
        self._closed = False

    @property
    def value_schema(self) -> ValueSchema:
        return self._projection.value_schema

    @property
    def stream_id(self):
        return self._tap.stream_id

    @property
    def stream_generation(self):
        return self._tap.stream_generation

    @property
    def start_sequence(self) -> int:
        return self._tap.start_sequence

    def next(self, timeout: float | None = None) -> SignalEvent:
        if self._closed:
            raise RuntimeError("signal event cursor is closed")
        if timeout is None:
            deadline = None
        else:
            seconds = finite_real(timeout, "signal event timeout", minimum=0.0)
            deadline = time.monotonic() + seconds
        while True:
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            envelope = self._tap.next(remaining)
            try:
                value = self._projection.project(envelope)
            except BaseException:
                self.close()
                raise
            if value is None:
                continue
            if not isinstance(value, Value):
                self.close()
                raise TypeError("signal output projector must return Value or None")
            if value.schema is not self._projection.value_schema:
                self.close()
                raise TypeError(
                    "projected Value must share the output's declared ValueSchema"
                )
            return SignalEvent(
                value,
                envelope.event_ref,
                envelope.trace,
                envelope.captured_at,
                self._processor_stages,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tap.close()

    def __enter__(self) -> "SignalEventCursor[PayloadT]":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.close()
        return False


@runtime_checkable
class SignalEventSource(Protocol):
    """Ordering-only source capability consumed by signal-driven applications."""

    def value_schema(self, output_name: str) -> ValueSchema: ...

    def open_signal_cursor(self, output_name: str) -> SignalEventCursor: ...


@runtime_checkable
class SignalEventAssociationCursor(Protocol):
    """Producer-owned cursor that can prove pulse association for event groups.

    ``arm_signal_association`` is a pre-FIRE admission boundary: the producer
    must atomically freeze the next ``expected_event_count`` group boundary and
    its exact token-owned grouping, or reject the request.  This is a causal
    boundary, not a host-memory reservation or capacity policy.
    ``next_associated_signal``
    may return only members of that admitted group.  ``finish`` may succeed
    only after the group is complete and its evidence is bound to the exact
    terminal object supplied to ``bind_signal_association``.
    """

    @property
    def value_schema(self) -> ValueSchema: ...

    @property
    def stream_id(self): ...

    @property
    def stream_generation(self): ...

    @property
    def start_sequence(self) -> int: ...

    def arm_signal_association(self, request: SignalAssociationRequest) -> object:
        """Freeze the next exact group boundary before FIRE, or reject it."""

        ...

    def bind_signal_association(
        self,
        token: object,
        terminal_evidence: object,
    ) -> None:
        """Bind the admitted group to the exact producer-observed terminal."""

        ...

    def next_associated_signal(
        self,
        token: object,
        timeout: float | None = None,
    ) -> SignalEvent:
        """Return only an event admitted to this token's exact group."""

        ...

    def finish_signal_association(
        self,
        token: object,
    ) -> SignalAssociationEvidence:
        """Return group-complete evidence self-bound to the terminal digest."""

        ...

    def close(self) -> None: ...


@runtime_checkable
class SignalEventAssociationSource(SignalEventSource, Protocol):
    """Explicit producer capability; ordinary ``SignalEventSource`` is not enough."""

    def signal_association_schedule_requirement(
        self,
        output_name: str,
    ) -> SignalAssociationScheduleRequirement: ...

    def open_associated_signal_cursor(
        self,
        output_name: str,
    ) -> SignalEventAssociationCursor: ...


class StreamSignalEventSource(Generic[PayloadT]):
    """Bind named, declared output projections to one live stream generation."""

    __slots__ = ("_outputs", "_processor_stages", "_stream")

    def __init__(
        self,
        stream: AcquisitionStream[PayloadT],
        outputs: Mapping[str, SignalOutputProjection[PayloadT]],
        *,
        processor_stages: tuple[ProcessorStageProvenance, ...] = (),
    ) -> None:
        if not isinstance(stream, AcquisitionStream):
            raise TypeError("signal source stream must be AcquisitionStream")
        owned: dict[str, SignalOutputProjection[PayloadT]] = {}
        for raw_name, projection in outputs.items():
            name = canonical_text(raw_name, "signal output name")
            if name in owned:
                raise ValueError("signal output names must be unique")
            if not isinstance(projection, SignalOutputProjection):
                raise TypeError(
                    "signal outputs must contain SignalOutputProjection values"
                )
            owned[name] = projection
        if not owned:
            raise ValueError("signal event source must declare at least one output")
        self._stream = stream
        self._outputs = owned
        self._processor_stages = _snapshot_processor_stages(processor_stages)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self._outputs)

    def value_schema(self, output_name: str) -> ValueSchema:
        return self._projection(output_name).value_schema

    def open_signal_cursor(self, output_name: str) -> SignalEventCursor[PayloadT]:
        projection = self._projection(output_name)
        tap = self._stream.follow()
        return SignalEventCursor(
            _CURSOR_TOKEN,
            tap=tap,
            projection=projection,
            processor_stages=self._processor_stages,
        )

    def _projection(self, output_name: str) -> SignalOutputProjection[PayloadT]:
        name = canonical_text(output_name, "signal output name")
        try:
            return self._outputs[name]
        except KeyError as error:
            raise KeyError(f"producer has no signal output {name!r}") from error


class AuthoritativeSignalEventCursor:
    """Apply one schema-committed data transform to each fresh source event."""

    __slots__ = (
        "_closed",
        "_committed_transform",
        "_cursor",
        "_input_dataset_schema",
        "_input_value_schema",
        "_output_value_schema",
    )

    def __init__(
        self,
        cursor,
        *,
        input_value_schema: ValueSchema,
        input_dataset_schema: DatasetSchema,
        output_value_schema: ValueSchema,
        committed_transform: CommittedTransform,
    ) -> None:
        if getattr(cursor, "value_schema", None) is not input_value_schema:
            raise TypeError(
                "source cursor schema differs from the committed projection input"
            )
        self._cursor = cursor
        self._input_value_schema = input_value_schema
        self._input_dataset_schema = input_dataset_schema
        self._output_value_schema = output_value_schema
        self._committed_transform = committed_transform
        self._closed = False

    @property
    def value_schema(self) -> ValueSchema:
        return self._output_value_schema

    @property
    def stream_id(self):
        return self._cursor.stream_id

    @property
    def stream_generation(self):
        return self._cursor.stream_generation

    @property
    def start_sequence(self) -> int:
        return self._cursor.start_sequence

    def next(self, timeout: float | None = None) -> SignalEvent:
        if self._closed:
            raise RuntimeError("signal event cursor is closed")
        event = self._cursor.next(timeout)
        return self._project_event(event)

    def _project_event(self, event: object) -> SignalEvent:
        if not isinstance(event, SignalEvent):
            self.close()
            raise TypeError("source cursor must return SignalEvent")
        if event.value.schema is not self._input_value_schema:
            self.close()
            raise TypeError("source event ValueSchema changed after projection commit")
        try:
            value = _apply_signal_value_transform(
                event,
                input_dataset_schema=self._input_dataset_schema,
                output_value_schema=self._output_value_schema,
                committed_transform=self._committed_transform,
            )
        except BaseException:
            self.close()
            raise
        return SignalEvent(
            value,
            event.event_ref,
            event.trace,
            event.captured_at,
            event.processor_stages,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cursor.close()

    def __enter__(self) -> "AuthoritativeSignalEventCursor":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.close()
        return False


class AuthoritativeAssociatedSignalEventCursor(AuthoritativeSignalEventCursor):
    """Preserve producer-owned association while projecting associated values."""

    __slots__ = ()

    def arm_signal_association(self, request: SignalAssociationRequest) -> object:
        return self._cursor.arm_signal_association(request)

    def bind_signal_association(
        self,
        token: object,
        terminal_evidence: object,
    ) -> None:
        self._cursor.bind_signal_association(token, terminal_evidence)

    def next_associated_signal(
        self,
        token: object,
        timeout: float | None = None,
    ) -> SignalEvent:
        if self._closed:
            raise RuntimeError("signal event cursor is closed")
        event = self._cursor.next_associated_signal(token, timeout)
        return self._project_event(event)

    def finish_signal_association(
        self,
        token: object,
    ) -> SignalAssociationEvidence:
        evidence = self._cursor.finish_signal_association(token)
        if not isinstance(evidence, SignalAssociationEvidence):
            self.close()
            raise TypeError(
                "associated source cursor returned another evidence type"
            )
        return evidence


class AuthoritativeSignalEventSource:
    """One named signal output with a frozen authoritative data transform."""

    __slots__ = (
        "_committed_transform",
        "_input_dataset_schema",
        "_input_value_schema",
        "_output_name",
        "_output_value_schema",
        "_projection_authority",
        "_source",
    )

    def __init__(
        self,
        source: SignalEventSource,
        output_name: str,
        authoritative_spec: DataTransformSpec,
    ) -> None:
        if not isinstance(source, SignalEventSource):
            raise TypeError("authoritative projection requires SignalEventSource")
        if not isinstance(authoritative_spec, DataTransformSpec):
            raise TypeError("authoritative_spec must be DataTransformSpec")
        name = canonical_text(output_name, "signal output name")
        input_value_schema = source.value_schema(name)
        if not isinstance(input_value_schema, ValueSchema):
            raise TypeError("signal source value_schema() must return ValueSchema")
        _reject_outer_axis_transform(authoritative_spec)
        input_dataset_schema = _signal_dataset_schema(input_value_schema)
        committed = commit_transform(input_dataset_schema, authoritative_spec)
        output_value_schema = _resolve_signal_projection_output(
            input_value_schema,
            committed,
        )
        self._source = source
        self._output_name = name
        self._input_value_schema = input_value_schema
        self._input_dataset_schema = input_dataset_schema
        self._output_value_schema = output_value_schema
        self._projection_authority = SignalProjectionAuthority(
            input_value_schema,
            committed,
            output_value_schema,
        )
        self._committed_transform = committed

    @property
    def committed_transform(self) -> CommittedTransform:
        return self._committed_transform

    @property
    def projection_authority(self) -> SignalProjectionAuthority:
        return self._projection_authority

    def value_schema(self, output_name: str) -> ValueSchema:
        self._require_output(output_name)
        return self._output_value_schema

    def open_signal_cursor(
        self,
        output_name: str,
    ) -> AuthoritativeSignalEventCursor:
        self._require_output(output_name)
        cursor = self._source.open_signal_cursor(self._output_name)
        try:
            return AuthoritativeSignalEventCursor(
                cursor,
                input_value_schema=self._input_value_schema,
                input_dataset_schema=self._input_dataset_schema,
                output_value_schema=self._output_value_schema,
                committed_transform=self._committed_transform,
            )
        except BaseException:
            cursor.close()
            raise

    def _require_output(self, output_name: str) -> None:
        name = canonical_text(output_name, "signal output name")
        if name != self._output_name:
            raise KeyError(f"projected source exposes only {self._output_name!r}")


class AuthoritativeAssociatedSignalEventSource(AuthoritativeSignalEventSource):
    """Authoritative projection that transparently preserves source association."""

    __slots__ = ()

    def __init__(
        self,
        source: SignalEventAssociationSource,
        output_name: str,
        authoritative_spec: DataTransformSpec,
    ) -> None:
        if not isinstance(source, SignalEventAssociationSource):
            raise TypeError("associated projection requires association capability")
        super().__init__(source, output_name, authoritative_spec)

    def open_associated_signal_cursor(
        self,
        output_name: str,
    ) -> AuthoritativeAssociatedSignalEventCursor:
        self._require_output(output_name)
        cursor = self._source.open_associated_signal_cursor(self._output_name)
        try:
            return AuthoritativeAssociatedSignalEventCursor(
                cursor,
                input_value_schema=self._input_value_schema,
                input_dataset_schema=self._input_dataset_schema,
                output_value_schema=self._output_value_schema,
                committed_transform=self._committed_transform,
            )
        except BaseException:
            cursor.close()
            raise

    def signal_association_schedule_requirement(
        self,
        output_name: str,
    ) -> SignalAssociationScheduleRequirement:
        self._require_output(output_name)
        source = self._source
        if not isinstance(source, SignalEventAssociationSource):
            raise RuntimeError("projected source lost association capability")
        return source.signal_association_schedule_requirement(self._output_name)


def authoritative_signal_event_source(
    source: SignalEventSource,
    output_name: str,
    authoritative_spec: DataTransformSpec | None,
) -> SignalEventSource:
    """Return identity for ``None`` or a schema-first authoritative wrapper."""

    if authoritative_spec is None:
        return source
    source_type = (
        AuthoritativeAssociatedSignalEventSource
        if isinstance(source, SignalEventAssociationSource)
        else AuthoritativeSignalEventSource
    )
    return source_type(source, output_name, authoritative_spec)


def _reject_outer_axis_transform(spec: DataTransformSpec) -> None:
    protected = {
        _SIGNAL_REPEAT_AXIS.axis_id,
        _SIGNAL_POINT_AXIS.axis_id,
    }
    for operation in spec.operations:
        axis_ids = (
            tuple(term.axis_id for term in operation.terms)
            if isinstance(operation, Selection)
            else operation.axis_ids
        )
        if any(axis_id in protected for axis_id in axis_ids):
            raise ValueError(
                "signal projection cannot select or reduce synthetic repeat/point axes"
            )


def _signal_dataset_schema(value_schema: ValueSchema) -> DatasetSchema:
    if any(
        axis.axis_id
        in {_SIGNAL_REPEAT_AXIS.axis_id, _SIGNAL_POINT_AXIS.axis_id}
        for axis in value_schema.data_axes
    ):
        raise ValueError("signal ValueSchema uses a reserved carrier AxisId")
    return DatasetSchema(
        _SIGNAL_REPEAT_AXIS,
        (_SIGNAL_POINT_AXIS,),
        PointLayout.rect_c((1,)),
        value_schema,
    )


def _resolve_signal_projection_output(
    input_value_schema: ValueSchema,
    transform: CommittedTransform,
) -> ValueSchema:
    input_dataset_schema = _signal_dataset_schema(input_value_schema)
    transformed = resolve_transformed_schema(input_dataset_schema, transform)
    if (
        transformed.cell_axes != (_SIGNAL_REPEAT_AXIS, _SIGNAL_POINT_AXIS)
        or transformed.cell_layout != input_dataset_schema.cell_layout
    ):
        raise ValueError(
            "signal projection must preserve its synthetic repeat/point carrier"
        )
    validity_contract = (
        ValidityContract.components(*transformed.validity_axis_ids)
        if transformed.validity_axis_ids
        else ValidityContract.value()
    )
    return ValueSchema(
        transformed.data_axes,
        validity_contract,
        transformed.dtype,
        transformed.value_unit,
    )


def _apply_signal_value_transform(
    event: SignalEvent,
    *,
    input_dataset_schema: DatasetSchema,
    output_value_schema: ValueSchema,
    committed_transform: CommittedTransform,
) -> Value:
    source_validity = event.value.validity
    dataset_validity = (
        ComponentValidity(
            source_validity.axis_ids,
            source_validity.mask.reshape(1, 1, *source_validity.mask.shape),
        )
        if isinstance(source_validity, ComponentValidity)
        else source_validity
    )
    revision = DatasetRevision(event.event_ref.sequence)
    block = DataBlock(
        BlockId(f"signal-event:{event.event_ref.event_id.value}"),
        revision,
        event.value.values.reshape(input_dataset_schema.physical_shape),
        dataset_validity,
        input_dataset_schema,
    )
    transformed = apply_transform(
        OwnedSnapshot(
            DatasetRevisionRef(
                block.block_id,
                event.event_ref.generation,
                input_dataset_schema.fingerprint,
                revision,
            ),
            block,
        ),
        committed_transform,
    )
    transformed_validity = transformed.validity
    if isinstance(transformed_validity, (Valid, Invalid)):
        value_validity = transformed_validity
    elif not isinstance(transformed_validity, RowComponentValidity):
        raise TypeError("transformed signal validity has an unsupported type")
    elif transformed_validity.axis_ids:
        value_validity = ComponentValidity(
            transformed_validity.axis_ids,
            transformed_validity.mask[0],
        )
    else:
        value_validity = (
            VALID if bool(transformed_validity.mask[0]) else INVALID
        )
    return Value(
        transformed.values[0],
        value_validity,
        output_value_schema,
    )


__all__ = [
    "SIGNAL_ASSOCIATION_EVIDENCE_SCHEMA",
    "SignalAssociationEvidence",
    "SignalAssociationRequest",
    "SignalAssociationScheduleRequirement",
    "SignalAssociationUnavailable",
    "SignalProjectionAuthority",
    "SignalEvent",
    "SignalEventAssociationCursor",
    "SignalEventAssociationSource",
    "SignalEventCursor",
    "SignalEventSource",
    "SignalOutputProjection",
    "StreamSignalEventSource",
    "AuthoritativeSignalEventCursor",
    "AuthoritativeSignalEventSource",
    "AuthoritativeAssociatedSignalEventCursor",
    "AuthoritativeAssociatedSignalEventSource",
    "authoritative_signal_event_source",
    "signal_association_evidence_from_tree",
    "signal_association_evidence_to_tree",
    "signal_projection_authority_from_tree",
    "signal_projection_authority_to_tree",
]
