"""Synchronous one-to-one operator hosted by one bounded worker thread."""

from __future__ import annotations

import threading
import time
import math

from zlc_storage import canonical_digest
from zlc_data import DatasetSchema

from zlc_neutral_atom.runtime.cancellation import CancellationRequested, CancellationToken
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetMode,
    SealedDatasetArtifact,
    dataset_cell_key_fingerprint,
    dataset_cell_permutation_digest,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    Delivery,
    EndOfStream,
    ExactConsumerReadiness,
    ExactReservation,
    ReservationState,
    SourceFailed,
    StreamEndedEarly,
    TraceContext,
)

from .contract import BoundStreamProcessor


class StreamProcessorError(RuntimeError):
    pass


class ExactStreamProcessorWorker:
    """Consume one exact interval and retain every output in its terminal builder.

    The worker owns no scheduler and no secondary input queue.  One input delivery
    remains unacknowledged until the pass-through-key output has been emitted and
    committed by the downstream ``DatasetBuilder``.

    All stream waits obey one absolute run deadline.  One additional second is
    reserved only for fail-closed teardown and thread join after that deadline.
    """

    def __init__(
        self,
        bound: BoundStreamProcessor,
        input_reservation: ExactReservation,
        input_cursor: AcquisitionCursor,
        *,
        source_schema: DatasetSchema,
        source_contract_digest: str,
        source_schedule_digest: str,
        expected_keys: tuple[object, ...],
        output_producer: AcquisitionProducer,
        output_cursor: AcquisitionCursor,
        output_builder: DatasetBuilder,
        deadline_monotonic: float,
        cancellation: CancellationToken | None = None,
    ) -> None:
        if not isinstance(bound, BoundStreamProcessor):
            raise TypeError("bound must be BoundStreamProcessor")
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise TypeError("deadline_monotonic must be a finite absolute timestamp")
        deadline_monotonic = float(deadline_monotonic)
        if deadline_monotonic <= time.monotonic():
            raise TimeoutError("processor absolute deadline already expired")
        if not isinstance(input_reservation, ExactReservation):
            raise TypeError("input_reservation must be ExactReservation")
        if not isinstance(input_cursor, AcquisitionCursor):
            raise TypeError("input_cursor must be AcquisitionCursor")
        if input_reservation._cursor is not input_cursor:
            raise ValueError("input_cursor does not belong to input_reservation")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        input_stream = input_reservation._stream
        if input_stream._payload_contract is not bound.input_payload_contract:
            raise ValueError("processor must share the input stream PayloadContract owner")
        if input_stream.payload_contract_fingerprint != (
            bound.definition.input_payload_contract_fingerprint
        ):
            raise ValueError("input stream payload contract differs from processor")
        input_key_contract = input_stream._join_key_contract
        if not isinstance(input_key_contract, DatasetCellKeyContract):
            raise TypeError("exact dataset processor requires DatasetCellKeyContract")
        if input_key_contract is not bound.join_key_contract:
            raise ValueError("processor must share the input join-key contract owner")
        if input_key_contract.fingerprint != bound.definition.join_key_contract_fingerprint:
            raise ValueError("input stream join-key contract differs from processor")
        if input_key_contract.fingerprint != dataset_cell_key_fingerprint(source_schema):
            raise ValueError("input join-key domain differs from source_schema")
        keys = tuple(expected_keys)
        total = input_reservation.end_sequence - input_reservation.start_sequence
        if len(keys) != total:
            raise ValueError("expected_keys length differs from exact reservation")
        if any(not isinstance(key, DatasetCellAddress) for key in keys):
            raise TypeError("expected_keys must contain DatasetCellAddress values")
        for key in keys:
            input_key_contract.validate(key)
            try:
                hash(key)
            except TypeError as error:
                raise TypeError("expected keys must be frozen and hashable") from error
        if dataset_cell_permutation_digest(source_schema, keys) != source_schedule_digest:
            raise ValueError("source_schedule_digest differs from expected_keys/source_schema")
        if not isinstance(output_producer, AcquisitionProducer):
            raise TypeError("output_producer must be AcquisitionProducer")
        if not isinstance(output_cursor, AcquisitionCursor):
            raise TypeError("output_cursor must be AcquisitionCursor")
        output_stream = output_producer._stream
        if output_stream._payload_contract is not bound.output_payload_contract:
            raise ValueError("processor must share the output stream PayloadContract owner")
        if output_cursor._stream is not output_stream:
            raise ValueError("output_cursor and output_producer belong to different streams")
        if output_builder.mode is not DatasetMode.FINITE_EXACT:
            raise ValueError("processor output requires a FINITE_EXACT DatasetBuilder")
        if output_builder._source is not output_stream:
            raise ValueError("output_builder belongs to another stream")
        if output_stream.stream_id != bound.output_stream_id:
            raise ValueError("output stream id differs from processor binding")
        if output_stream.payload_contract_fingerprint != (
            bound.definition.output_payload_contract_fingerprint
        ):
            raise ValueError("output stream payload contract differs from processor")
        output_key_contract = output_stream._join_key_contract
        if output_key_contract is not bound.join_key_contract:
            raise ValueError("processor must share the output join-key contract owner")
        if output_key_contract.fingerprint != input_key_contract.fingerprint:
            raise ValueError("processor must preserve its join-key fingerprint exactly")
        output_reservation = output_builder._reservation
        if output_reservation is None or output_reservation._cursor is not output_cursor:
            raise ValueError("output cursor is not the output builder reservation cursor")
        if output_cursor._reservation_token is not output_reservation._token:
            raise ValueError("output cursor reservation identity differs from builder")
        expected_output_trace = input_reservation.trace_binding.__class__(
            input_reservation.trace_binding.run_id,
            bound.output_source_id,
        )
        if output_reservation.trace_binding != expected_output_trace:
            raise ValueError("output reservation TraceBinding differs from processor lineage")
        if output_builder._expected_cells != keys:
            raise ValueError("output builder schedule differs from expected_keys")
        if output_builder._join_plan_digest != dataset_cell_permutation_digest(
            output_builder.schema,
            keys,
        ):
            raise ValueError("output builder schedule digest is inconsistent")
        downstream = output_builder.exact_readiness()
        chain_digest = canonical_digest(
            {
                "contract": "zlc_neutral_atom.ExactStreamProcessorChain/v1",
                "processor": bound.fingerprint,
                "source_contract": source_contract_digest,
                "source_schedule": source_schedule_digest,
                "downstream": downstream.chain_contract_digest,
                "events": total,
            }
        )
        readiness = input_stream._claim_consumer(
            input_reservation,
            self,
            source_contract_digest=source_contract_digest,
            source_schedule_digest=source_schedule_digest,
            chain_contract_digest=chain_digest,
            downstream=downstream,
            owner_liveness=self._validate_readiness_liveness,
        )
        self._bound = bound
        self._source_schema = source_schema
        self._input_reservation = input_reservation
        self._input_cursor = input_cursor
        self._input_stream = input_stream
        self._expected_keys = keys
        self._output_producer = output_producer
        self._output_cursor = output_cursor
        self._output_builder = output_builder
        self._cancellation = cancellation or CancellationToken()
        self._deadline_monotonic = deadline_monotonic
        self._readiness = readiness
        self._condition = threading.Condition(threading.Lock())
        self._thread: threading.Thread | None = None
        self._input_eos: EndOfStream | None = None
        self._result: SealedDatasetArtifact | None = None
        self._error: BaseException | None = None
        self._done = False
        self._closing = False

    def exact_readiness(self) -> ExactConsumerReadiness:
        self._validate_readiness_liveness()
        self._readiness._validate_terminal_sink()
        return self._readiness

    def _validate_readiness_liveness(self) -> None:
        with self._condition:
            thread = self._thread
            if thread is None:
                raise StreamProcessorError("processor worker has not started")
            if (
                self._done
                or self._closing
                or self._error is not None
                or self._cancellation.is_cancelled
                or time.monotonic() >= self._deadline_monotonic
                or not thread.is_alive()
            ):
                raise StreamProcessorError("processor chain owner is not live")

    @property
    def is_alive(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    def start(self) -> None:
        self._checkpoint()
        with self._condition:
            if self._thread is not None or self._done or self._closing:
                raise StreamProcessorError("processor worker can start only once")
            thread = threading.Thread(
                target=self._run,
                name=f"stream-processor:{self._bound.definition.key}",
                daemon=False,
            )
            self._thread = thread
            thread.start()

    def finish(self, eos: EndOfStream, timeout: float | None = None) -> SealedDatasetArtifact:
        if not isinstance(eos, EndOfStream):
            raise TypeError("eos must be EndOfStream")
        if not self._input_stream._owns_eos(eos):
            raise PermissionError("EOS belongs to another source authority")
        with self._condition:
            if self._thread is None:
                raise StreamProcessorError("processor worker has not started")
            if self._input_eos is not None and self._input_eos is not eos:
                raise StreamProcessorError("processor already received another terminal receipt")
            self._input_eos = eos
            self._condition.notify_all()
        self.wait(timeout)
        with self._condition:
            if self._error is not None:
                raise StreamProcessorError("exact stream processor failed") from self._error
            if self._result is None:
                raise StreamProcessorError("processor ended without a sealed dataset")
            return self._result

    def wait(self, timeout: float | None = None) -> None:
        deadline = self._deadline_monotonic + 1.0
        if timeout is not None:
            deadline = min(deadline, time.monotonic() + max(0.0, timeout))
        with self._condition:
            if self._thread is None and not self._done and not self._closing:
                raise StreamProcessorError("processor worker has not started")
            while not self._done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for exact stream processor")
                self._condition.wait(remaining)
        thread = self._thread
        if thread is not None:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)
            if thread.is_alive():
                raise TimeoutError("processor signalled done but its worker thread did not exit")

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise StreamProcessorError("exact stream processor failed") from error

    def cancel(self, reason: str | None = None) -> bool:
        return self._cancellation.request(reason)

    def close(self, timeout: float | None = None) -> None:
        """Idempotently cancel/join, including the preflight-before-start path."""

        with self._condition:
            if self._done or self._closing:
                wait_for_existing_close = True
                preflight_only = False
            else:
                wait_for_existing_close = False
                self._closing = True
                if self._thread is None:
                    self._error = CancellationRequested("processor closed before start")
                    preflight_only = True
                else:
                    preflight_only = False
        if wait_for_existing_close:
            self.wait(timeout)
            return
        if preflight_only:
            self._cleanup(False)
            with self._condition:
                self._done = True
                self._condition.notify_all()
            return
        self.cancel("processor closed")
        self.wait(timeout)

    def _run(self) -> None:
        succeeded = False
        try:
            for index, expected_key in enumerate(self._expected_keys):
                self._checkpoint()
                delivery = self._next_input()
                self._process_one(index, expected_key, delivery)
            eos = self._wait_for_eos()
            self._input_stream._validate_consumer_completion(
                self._input_reservation,
                eos,
                self,
            )
            self._checkpoint()
            output_eos = self._output_producer.finish()
            self._checkpoint()
            result = self._output_builder.seal(output_eos)
            self._checkpoint()
            self._input_stream._complete_consumer(
                self._input_reservation,
                eos,
                self,
                lambda: None,
            )
            with self._condition:
                self._result = result
            succeeded = True
        except BaseException as error:
            with self._condition:
                self._error = error
            try:
                self._output_producer.fail(SourceFailed(f"processor failed: {error}"))
            except StreamEndedEarly:
                pass
            except BaseException as cleanup_error:
                error.add_note(f"output stream failure also failed: {cleanup_error!r}")
        finally:
            self._cleanup(succeeded)
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def _next_input(self) -> Delivery:
        while True:
            self._checkpoint()
            try:
                return self._input_cursor.next(timeout=min(0.05, self._remaining()))
            except TimeoutError:
                continue

    def _process_one(self, index: int, expected_key: object, delivery: Delivery) -> None:
        envelope = delivery.envelope
        if envelope.join_key != expected_key:
            raise StreamProcessorError(
                f"input key at ordinal {index} differs from frozen exact schedule"
            )
        invocation_payload = self._bound.input_payload_contract.snapshot(envelope.payload)
        self._bound.input_payload_contract.validate(invocation_payload)
        started = time.monotonic()
        output_payload = self._bound.operator(invocation_payload, self._bound.config)
        elapsed = time.monotonic() - started
        if elapsed > self._bound.definition.operator_deadline_seconds:
            raise TimeoutError(
                "trusted synchronous processor exceeded operator_deadline_seconds"
            )
        self._checkpoint()
        output_trace = TraceContext(
            run_id=envelope.trace.run_id,
            source_id=self._bound.output_source_id,
            correlation_id=envelope.trace.correlation_id,
            causation_refs=(envelope.ref, *self._bound.artifact_inputs),
            config_revision=envelope.trace.config_revision,
            control_revision=envelope.trace.control_revision,
        )
        emitted = self._output_producer.emit(
            output_payload,
            captured_at=envelope.captured_at,
            trace=output_trace,
            join_key=envelope.join_key,
        )
        if emitted.join_key != expected_key:
            raise StreamProcessorError("PASS_THROUGH output join key changed during emit")
        if emitted.join_key_schema_fingerprint != (
            self._bound.definition.join_key_contract_fingerprint
        ):
            raise StreamProcessorError("PASS_THROUGH output join-key fingerprint changed")
        output_delivery = self._output_cursor.next(timeout=0.0)
        if output_delivery.envelope.event_id != emitted.event_id:
            raise StreamProcessorError("output cursor did not retain the emitted event")
        self._output_builder.consume(output_delivery)
        if not output_delivery.acknowledged:
            raise StreamProcessorError("downstream builder did not acknowledge output")
        self._input_stream._ack_consumer(
            self._input_reservation,
            delivery,
            self,
        )

    def _wait_for_eos(self) -> EndOfStream:
        deadline = min(
            self._deadline_monotonic,
            time.monotonic() + self._bound.definition.terminal_wait_seconds,
        )
        while True:
            self._checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("source did not publish a terminal fact within its contract")
            eos = self._input_stream._await_terminal(min(0.05, remaining))
            if eos is None:
                continue
            with self._condition:
                supplied = self._input_eos
            if supplied is not None and supplied is not eos:
                raise StreamProcessorError("supplied EOS belongs to another source receipt")
            return eos

    def _remaining(self) -> float:
        remaining = self._deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("processor absolute deadline expired")
        return remaining

    def _checkpoint(self) -> None:
        self._cancellation.checkpoint()
        self._remaining()

    def _cleanup(self, succeeded: bool) -> None:
        cancelled = isinstance(self._error, CancellationRequested)
        if not succeeded and self._input_reservation.state in (
            ReservationState.ACTIVE,
            ReservationState.DRAINING,
        ):
            try:
                self._input_stream._abort_consumer(
                    self._input_reservation,
                    self,
                    lambda: None,
                    cancelled=cancelled,
                )
            except BaseException as cleanup_error:
                if self._error is not None:
                    self._error.add_note(
                        f"input reservation abort also failed: {cleanup_error!r}"
                    )
        try:
            self._output_builder.close()
        except BaseException as cleanup_error:
            if self._error is None:
                self._error = cleanup_error
            else:
                self._error.add_note(f"output builder close also failed: {cleanup_error!r}")
        if self._input_reservation.state in (
            ReservationState.COMPLETED,
            ReservationState.FAILED,
            ReservationState.CANCELLED,
        ):
            try:
                self._input_reservation.release()
            except BaseException as cleanup_error:
                if self._error is None:
                    self._error = cleanup_error
                else:
                    self._error.add_note(
                        f"input reservation release also failed: {cleanup_error!r}"
                    )


__all__ = ["ExactStreamProcessorWorker", "StreamProcessorError"]
