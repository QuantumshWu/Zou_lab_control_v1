"""Lossless live Camera-event to Occupancy-event publication.

This is Occupancy's authoritative live signal owner.  It follows one already
running Camera output, classifies every future event in source order, and emits
one atomic payload containing ``counts``, ``occupied``, and ``rate``.  GUI
snapshot/coalescing policy is deliberately absent from this module.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from zlc_data import (
    INVALID,
    VALID,
    ComponentValidity,
    Invalid,
    Valid,
    Value,
    ValuePayloadContract,
    ValueSchema,
    StreamGenerationId,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ReadoutModel,
    apply_readout_model,
)
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationRequest,
    SignalAssociationScheduleRequirement,
    SignalEvent,
    SignalEventAssociationCursor,
    SignalEventAssociationSource,
    SignalEventSource,
    SignalOutputProjection,
    StreamSignalEventSource,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionProducer,
    AcquisitionStream,
    SourceFailed,
    StreamEndedEarly,
    StreamError,
    StreamId,
)
from zlc_storage import canonical_text

from .processor import (
    OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
    OCCUPANCY_PROCESSOR_KEY,
    _output_schemas,
    _require_output_value_schemas,
    _validate_sample_fields,
)
from ..contracts import FrameContract
from zlc_neutral_atom.logic_nodes.camera_measurement.output_binding import (
    CameraFrameOutputBinding,
)


_OUTPUT_NAMES = ("counts", "occupied", "rate")


@dataclass(frozen=True, slots=True)
class OccupancySignalValues:
    """All Occupancy values derived from one and the same Camera event."""

    counts: Value
    occupied: Value
    rate: Value

    __hash__ = None


@dataclass(frozen=True, slots=True)
class OccupancySignalValuesContract:
    """Generation-owned payload contract for one atomic derived publication."""

    counts_schema: ValueSchema
    occupied_schema: ValueSchema
    rate_schema: ValueSchema

    def __post_init__(self) -> None:
        if not isinstance(self.counts_schema, ValueSchema) or not isinstance(
            self.occupied_schema,
            ValueSchema,
        ):
            raise TypeError("Occupancy signal schemas must be ValueSchema")
        if not isinstance(self.rate_schema, ValueSchema):
            raise TypeError("Occupancy rate schema must be ValueSchema")
        _require_output_value_schemas(
            self.counts_schema,
            self.occupied_schema,
        )
        if self.rate_schema != ValueSchema.scalar(np.dtype("<f8"), None):
            raise ValueError("Occupancy rate requires the canonical scalar schema")

    @property
    def _counts(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.counts_schema)

    @property
    def _occupied(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.occupied_schema)

    @property
    def _rate(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.rate_schema)

    def snapshot(self, payload: OccupancySignalValues) -> OccupancySignalValues:
        self.validate(payload)
        return payload

    def validate(self, payload: object) -> None:
        if not isinstance(payload, OccupancySignalValues):
            raise TypeError("Occupancy live payload must be OccupancySignalValues")
        self._counts.validate(payload.counts)
        self._occupied.validate(payload.occupied)
        self._rate.validate(payload.rate)
        _validate_sample_fields(payload.counts, payload.occupied)
        _validate_rate(payload.occupied, payload.rate)

def _validate_rate(occupied: Value, rate: Value) -> None:
    validity = occupied.validity
    if not isinstance(validity, ComponentValidity):
        raise TypeError("Occupancy signal requires per-site validity")
    usable = np.asarray(validity.mask, dtype=bool)
    denominator = int(np.count_nonzero(usable))
    expected = 0.0
    if denominator:
        expected = float(
            np.count_nonzero(np.asarray(occupied.values, dtype=bool) & usable)
            / denominator
        )
        if not isinstance(rate.validity, Valid):
            raise ValueError("Occupancy rate must be valid when a site is usable")
    elif not isinstance(rate.validity, Invalid):
        raise ValueError("Occupancy rate must be invalid when no site is usable")
    if float(rate.values[0]) != expected:
        raise ValueError("Occupancy rate differs from its same-shot occupied value")


def _rate_value(occupied: Value, schema: ValueSchema) -> Value:
    validity = occupied.validity
    if not isinstance(validity, ComponentValidity):
        raise TypeError("Occupancy signal requires per-site validity")
    usable = np.asarray(validity.mask, dtype=bool)
    denominator = int(np.count_nonzero(usable))
    if denominator:
        value = float(
            np.count_nonzero(np.asarray(occupied.values, dtype=bool) & usable)
            / denominator
        )
        rate_validity = VALID
    else:
        value = 0.0
        rate_validity = INVALID
    return Value(np.asarray((value,), dtype="<f8"), rate_validity, schema)


def _occupancy_output_value(
    payload: OccupancySignalValues,
    output_name: str,
) -> Value:
    if output_name == "counts":
        return payload.counts
    if output_name == "occupied":
        return payload.occupied
    if output_name == "rate":
        return payload.rate
    raise KeyError(f"Occupancy has no signal output {output_name!r}")


def _occupancy_output_schema(
    contract: OccupancySignalValuesContract,
    output_name: str,
) -> ValueSchema:
    if output_name == "counts":
        return contract.counts_schema
    if output_name == "occupied":
        return contract.occupied_schema
    if output_name == "rate":
        return contract.rate_schema
    raise KeyError(f"Occupancy has no signal output {output_name!r}")


@dataclass(slots=True, eq=False)
class _OccupancyAssociationToken:
    request: SignalAssociationRequest
    upstream_token: object
    terminal_bound: bool = False
    delivered: int = 0
    finished: bool = False


class _OccupancyAssociatedCursor:
    """Strict 1:1 transformation over one dedicated upstream associated cursor."""

    __slots__ = (
        "_active",
        "_classify",
        "_closed",
        "_contract",
        "_frame_schema",
        "_output_name",
        "_producer",
        "_stream",
        "_upstream",
        "_value_schema",
    )

    def __init__(
        self,
        upstream: SignalEventAssociationCursor,
        *,
        output_name: str,
        frame_schema: ValueSchema,
        contract: OccupancySignalValuesContract,
        classify: Callable[[Value], OccupancySignalValues],
    ) -> None:
        if not isinstance(upstream, SignalEventAssociationCursor):
            raise TypeError("Occupancy requires an upstream association cursor")
        if upstream.value_schema != frame_schema:
            raise ValueError("upstream association cursor changed Camera schema")
        name = canonical_text(output_name, "Occupancy signal output name")
        value_schema = _occupancy_output_schema(contract, name)
        stream, producer = AcquisitionStream.create(
            StreamId(f"occupancy-associated:{uuid.uuid4().hex}"),
            contract,
        )
        self._upstream = upstream
        self._output_name = name
        self._value_schema = value_schema
        self._frame_schema = frame_schema
        self._contract = contract
        self._classify = classify
        self._stream = stream
        self._producer = producer
        self._active: _OccupancyAssociationToken | None = None
        self._closed = False

    @property
    def value_schema(self) -> ValueSchema:
        return self._value_schema

    @property
    def stream_id(self):
        return self._stream.stream_id

    @property
    def stream_generation(self):
        return self._stream.generation

    @property
    def start_sequence(self) -> int:
        return 0

    def arm_signal_association(self, request: SignalAssociationRequest) -> object:
        self._require_open()
        if not isinstance(request, SignalAssociationRequest):
            raise TypeError("Occupancy association requires SignalAssociationRequest")
        active = self._active
        if active is not None and not active.finished:
            raise RuntimeError("Occupancy association groups cannot overlap")
        upstream_token = self._upstream.arm_signal_association(request)
        token = _OccupancyAssociationToken(request, upstream_token)
        self._active = token
        return token

    def bind_signal_association(
        self,
        token: object,
        terminal_evidence: object,
    ) -> None:
        state = self._require_token(token)
        if state.terminal_bound:
            raise RuntimeError("Occupancy association was already terminal-bound")
        self._upstream.bind_signal_association(
            state.upstream_token,
            terminal_evidence,
        )
        state.terminal_bound = True

    def next_associated_signal(
        self,
        token: object,
        timeout: float | None = None,
    ) -> SignalEvent:
        state = self._require_token(token)
        if not state.terminal_bound:
            raise RuntimeError("Occupancy association is not terminal-bound")
        if state.delivered >= state.request.expected_event_count:
            raise RuntimeError("Occupancy association group is exhausted")
        upstream_event = self._upstream.next_associated_signal(
            state.upstream_token,
            timeout,
        )
        if not isinstance(upstream_event, SignalEvent):
            raise TypeError("upstream association returned another event type")
        if upstream_event.value.schema != self._frame_schema:
            raise RuntimeError("upstream associated Camera schema changed")
        payload = self._classify(upstream_event.value)
        self._contract.validate(payload)
        envelope = self._producer.emit(
            payload,
            captured_at=upstream_event.captured_at,
            direct_parent_refs=(upstream_event.event_ref,),
        )
        state.delivered += 1
        return SignalEvent(
            _occupancy_output_value(payload, self._output_name),
            envelope.event_ref,
            envelope.direct_parent_refs,
            envelope.captured_at,
        )

    def finish_signal_association(
        self,
        token: object,
    ) -> None:
        state = self._require_token(token)
        if not state.terminal_bound:
            raise RuntimeError("Occupancy association is not terminal-bound")
        if state.delivered != state.request.expected_event_count:
            raise RuntimeError("Occupancy association group is incomplete")
        self._upstream.finish_signal_association(state.upstream_token)
        state.finished = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._upstream.close()
        finally:
            self._producer.finish()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Occupancy association cursor is closed")

    def _require_token(self, token: object) -> _OccupancyAssociationToken:
        self._require_open()
        active = self._active
        if active is None or token is not active:
            raise RuntimeError("unknown Occupancy association token")
        if active.finished:
            raise RuntimeError("Occupancy association group is already finished")
        return active


class OccupancySignalProcessor:
    """Frozen classifier/provenance owner that can start one live derived source."""

    __slots__ = (
        "_contract",
        "_frame_schema",
        "_model",
        "_source_binding",
    )

    def __init__(
        self,
        *,
        frame_contract: FrameContract,
        model: ReadoutModel,
        source_binding: CameraFrameOutputBinding,
    ) -> None:
        if not isinstance(frame_contract, FrameContract):
            raise TypeError("frame_contract must be FrameContract")
        if not isinstance(model, ReadoutModel):
            raise TypeError("model must be ReadoutModel")
        if not isinstance(source_binding, CameraFrameOutputBinding):
            raise TypeError("source_binding must be CameraFrameOutputBinding")
        counts_schema, occupied_schema = _output_schemas(
            frame_contract,
            model.feature.site_axis,
        )
        self._frame_schema = frame_contract.frame_schema
        self._model = model
        self._source_binding = source_binding
        self._contract = OccupancySignalValuesContract(
            counts_schema,
            occupied_schema,
            ValueSchema.scalar(np.dtype("<f8"), None),
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return _OUTPUT_NAMES

    def value_schema(self, output_name: str) -> ValueSchema:
        name = canonical_text(output_name, "Occupancy signal output name")
        if name == "counts":
            return self._contract.counts_schema
        if name == "occupied":
            return self._contract.occupied_schema
        if name == "rate":
            return self._contract.rate_schema
        raise KeyError(f"Occupancy has no signal output {name!r}")

    def start(
        self,
        source: SignalEventSource,
        source_output_name: str,
    ) -> "RunningOccupancySignalSource":
        """Open the Camera future cursor before starting the processor worker."""

        name = canonical_text(source_output_name, "Camera signal output name")
        source_type = (
            AssociatedRunningOccupancySignalSource
            if isinstance(source, SignalEventAssociationSource)
            else RunningOccupancySignalSource
        )
        return source_type(
            source,
            source_output_name=name,
            frame_schema=self._frame_schema,
            contract=self._contract,
            classify=self._classify,
            expected_source_stream_id=self._source_binding.event_stream_id,
            expected_source_stream_generation=(
                self._source_binding.event_stream_generation
            ),
        )

    def _classify(self, frame: Value) -> OccupancySignalValues:
        if not isinstance(frame, Value) or frame.schema != self._frame_schema:
            raise TypeError("Occupancy received a Camera value with another schema")
        result = apply_readout_model(
            self._model,
            frame,
            expected_frame_schema=self._frame_schema,
        )
        validity = result.occupied.validity
        if not isinstance(validity, ComponentValidity):
            raise TypeError("readout result requires ComponentValidity")
        counts = Value(
            result.signals.values,
            validity,
            self._contract.counts_schema,
        )
        occupied = Value(
            result.occupied.values,
            validity,
            self._contract.occupied_schema,
        )
        return OccupancySignalValues(
            counts,
            occupied,
            _rate_value(occupied, self._contract.rate_schema),
        )


class RunningOccupancySignalSource:
    """Live derived stream whose only input authority is a Camera future cursor."""

    __slots__ = (
        "_condition",
        "_contract",
        "_cursor",
        "_classify",
        "_done",
        "_error",
        "_frame_schema",
        "_output_source_id",
        "_producer",
        "_signal_source",
        "_source",
        "_source_output_name",
        "_thread",
    )

    def __init__(
        self,
        source: SignalEventSource,
        *,
        source_output_name: str,
        frame_schema: ValueSchema,
        contract: OccupancySignalValuesContract,
        classify: Callable[[Value], OccupancySignalValues],
        expected_source_stream_id: StreamId,
        expected_source_stream_generation: StreamGenerationId,
    ) -> None:
        if not isinstance(source, SignalEventSource):
            raise TypeError("Occupancy live input must implement SignalEventSource")
        name = canonical_text(source_output_name, "Camera signal output name")
        source_schema = source.value_schema(name)
        if source_schema != frame_schema:
            raise ValueError("Camera signal schema differs from the calibration FrameContract")
        if not isinstance(contract, OccupancySignalValuesContract):
            raise TypeError("contract must be OccupancySignalValuesContract")
        if not callable(classify):
            raise TypeError("classify must be callable")
        if not isinstance(expected_source_stream_id, StreamId):
            raise TypeError("expected_source_stream_id must be StreamId")
        if not isinstance(expected_source_stream_generation, StreamGenerationId):
            raise TypeError(
                "expected_source_stream_generation must be StreamGenerationId"
            )

        cursor = source.open_signal_cursor(name)
        if cursor.value_schema is not source_schema:
            cursor.close()
            raise TypeError("Camera cursor lost its declared ValueSchema owner")
        if (
            cursor.stream_id != expected_source_stream_id
            or cursor.stream_generation != expected_source_stream_generation
        ):
            cursor.close()
            raise ValueError(
                "Camera cursor belongs to another bound stream generation"
            )
        try:
            stream_id = StreamId(f"occupancy-live:{uuid.uuid4().hex}")
            stream, producer = AcquisitionStream.create(
                stream_id,
                contract,
            )
            signal_source = StreamSignalEventSource(
                stream,
                {
                    "counts": SignalOutputProjection(
                        contract.counts_schema,
                        lambda envelope: envelope.payload.counts,
                    ),
                    "occupied": SignalOutputProjection(
                        contract.occupied_schema,
                        lambda envelope: envelope.payload.occupied,
                    ),
                    "rate": SignalOutputProjection(
                        contract.rate_schema,
                        lambda envelope: envelope.payload.rate,
                    ),
                },
            )
        except BaseException:
            cursor.close()
            raise
        self._condition = threading.Condition(threading.Lock())
        self._source = source
        self._source_output_name = name
        self._frame_schema = frame_schema
        self._contract = contract
        self._classify = classify
        self._cursor = cursor
        self._output_source_id = stream_id.value
        self._producer: AcquisitionProducer[OccupancySignalValues] = producer
        self._signal_source = signal_source
        self._done = False
        self._error: BaseException | None = None

        def process() -> None:
            self._process_events(classify)

        thread = threading.Thread(
            target=process,
            name="occupancy-live-signal",
            daemon=False,
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException as error:
            cursor.close()
            failure = SourceFailed(
                "Occupancy live signal worker failed to start: "
                f"{safe_error_summary(error)}"
            )
            producer.fail(failure)
            with self._condition:
                self._done = True
                self._error = failure
                self._condition.notify_all()
            raise

    @property
    def output_names(self) -> tuple[str, ...]:
        return _OUTPUT_NAMES

    @property
    def definition_key(self):
        return OCCUPANCY_PROCESSOR_KEY

    @property
    def dataset_output_declarations(self):
        return OCCUPANCY_LIVE_OUTPUT_DECLARATIONS

    @property
    def worker_idle(self) -> bool:
        with self._condition:
            return self._done

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    def value_schema(self, output_name: str) -> ValueSchema:
        return self._signal_source.value_schema(output_name)

    def open_signal_cursor(self, output_name: str):
        return self._signal_source.open_signal_cursor(output_name)

    def request_close(self) -> None:
        """Interrupt the upstream cursor without joining on the caller thread."""

        self._cursor.close()

    def join_closed(self) -> None:
        """Join only after the worker has published its terminal idle fact."""

        if not self.worker_idle:
            raise RuntimeError("Occupancy signal worker is not idle")
        thread = self._thread
        if thread is not threading.current_thread():
            thread.join()

    def _process_events(
        self,
        classify: Callable[[Value], OccupancySignalValues],
    ) -> None:
        failure: BaseException | None = None
        try:
            while True:
                event = self._cursor.next()
                if not isinstance(event, SignalEvent):
                    raise TypeError("Camera signal cursor returned another event type")
                payload = classify(event.value)
                self._contract.validate(payload)
                self._producer.emit(
                    payload,
                    captured_at=event.captured_at,
                    direct_parent_refs=(event.event_ref,),
                )
        except StreamEndedEarly:
            self._producer.finish()
        except StreamError as error:
            failure = error
            self._producer.fail(error)
        except BaseException as error:
            failure = SourceFailed(
                "Occupancy live signal processing failed: "
                f"{safe_error_summary(error)}"
            )
            self._producer.fail(failure)
        finally:
            self._cursor.close()
            with self._condition:
                self._error = failure
                self._done = True
                self._condition.notify_all()


class AssociatedRunningOccupancySignalSource(RunningOccupancySignalSource):
    """Occupancy source that transparently preserves upstream association proof."""

    __slots__ = ()

    def __init__(
        self,
        source: SignalEventAssociationSource,
        **kwargs,
    ) -> None:
        if not isinstance(source, SignalEventAssociationSource):
            raise TypeError(
                "associated Occupancy requires producer association capability"
            )
        super().__init__(source, **kwargs)

    def open_associated_signal_cursor(
        self,
        output_name: str,
    ) -> SignalEventAssociationCursor:
        name = canonical_text(output_name, "Occupancy signal output name")
        _occupancy_output_schema(self._contract, name)
        source = self._source
        if not isinstance(source, SignalEventAssociationSource):
            raise RuntimeError("Occupancy lost upstream association capability")
        upstream = source.open_associated_signal_cursor(self._source_output_name)
        try:
            return _OccupancyAssociatedCursor(
                upstream,
                output_name=name,
                frame_schema=self._frame_schema,
                contract=self._contract,
                classify=self._classify,
            )
        except BaseException:
            upstream.close()
            raise

    def signal_association_schedule_requirement(
        self,
        output_name: str,
    ) -> SignalAssociationScheduleRequirement:
        name = canonical_text(output_name, "Occupancy signal output name")
        _occupancy_output_schema(self._contract, name)
        source = self._source
        if not isinstance(source, SignalEventAssociationSource):
            raise RuntimeError("Occupancy lost upstream association capability")
        requirement = source.signal_association_schedule_requirement(
            self._source_output_name
        )
        if not isinstance(requirement, SignalAssociationScheduleRequirement):
            raise TypeError("upstream returned another schedule requirement type")
        return requirement


__all__ = [
    "AssociatedRunningOccupancySignalSource",
    "OccupancySignalProcessor",
    "RunningOccupancySignalSource",
]
