"""Synchronous one-to-one operator hosted by one bounded worker thread."""

from __future__ import annotations

import threading
import time
import math

from zlc_storage import canonical_digest
from zlc_neutral_atom.runtime.cancellation import CancellationRequested, CancellationToken
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetMode,
    FrozenDatasetEdge,
    SealedDatasetArtifact,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    Delivery,
    EndOfStream,
    Envelope,
    ExactConsumerReadiness,
    ExactReservation,
    EventSpanRef,
    OrderedEventSpanHasher,
    ProcessorStageProvenance,
    ReservationState,
    SourceFailed,
    StreamEndedEarly,
    TraceContext,
)

from .contract import BoundStreamProcessor


class StreamProcessorError(RuntimeError):
    pass


class _TerminalDatasetOutput:
    """Immediate exact sink used only by the tail processor stage."""

    def __init__(
        self,
        cursor: AcquisitionCursor,
        builder: DatasetBuilder,
    ) -> None:
        self.readiness = builder.exact_readiness()
        self._cursor = cursor
        self._builder = builder

    def acknowledge(
        self,
        _owner: object,
        emitted: Envelope,
        *,
        deadline_monotonic: float,
        checkpoint,
    ) -> None:
        checkpoint()
        output_delivery = self._cursor.next(
            timeout=max(0.0, deadline_monotonic - time.monotonic())
        )
        if output_delivery.envelope.event_id != emitted.event_id:
            raise StreamProcessorError("output cursor did not retain the emitted event")
        self._builder.consume(output_delivery)
        if not output_delivery.acknowledged:
            raise StreamProcessorError("downstream builder did not acknowledge output")

    def complete(
        self,
        _owner: object,
        output_eos: EndOfStream,
        *,
        deadline_monotonic: float,
    ) -> SealedDatasetArtifact:
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError("processor absolute deadline expired before dataset seal")
        return self._builder.seal(output_eos)

    def cancel_and_join(
        self,
        _owner: object,
        _reason: str,
        *,
        deadline_monotonic: float,
    ) -> None:
        del deadline_monotonic
        return None

    def close(self) -> None:
        self._builder.close()


class _ProcessorStageOutput:
    """Immediate live processor sink backed by the existing exact readiness proof."""

    def __init__(self, readiness: ExactConsumerReadiness) -> None:
        self.readiness = readiness

    def acknowledge(
        self,
        owner: object,
        emitted: Envelope,
        *,
        deadline_monotonic: float,
        checkpoint,
    ) -> None:
        self.readiness._await_source_ack(
            owner,
            emitted.ref,
            deadline_monotonic=deadline_monotonic,
            checkpoint=checkpoint,
        )

    def complete(
        self,
        owner: object,
        _output_eos: EndOfStream,
        *,
        deadline_monotonic: float,
    ) -> SealedDatasetArtifact:
        result = self.readiness._await_bound_completion(
            owner,
            deadline_monotonic=deadline_monotonic,
        )
        if not isinstance(result, SealedDatasetArtifact):
            raise StreamProcessorError(
                "downstream exact chain completed without a sealed dataset"
            )
        return result

    def cancel_and_join(
        self,
        owner: object,
        reason: str,
        *,
        deadline_monotonic: float,
    ) -> None:
        self.readiness._cancel_bound_owner(owner, reason)
        try:
            self.readiness._await_bound_completion(
                owner,
                deadline_monotonic=deadline_monotonic,
            )
        except StreamProcessorError:
            # Failure is the expected terminal state while unwinding.  The
            # completion callback has still joined the downstream worker.
            pass

    @staticmethod
    def close() -> None:
        return None


class ExactStreamProcessorWorker:
    """Consume one exact interval in a bounded live processor chain.

    The worker owns no scheduler and no secondary input queue.  One input delivery
    remains unacknowledged until the pass-through-key output has been emitted and
    acknowledged by its immediate downstream processor or terminal
    ``DatasetBuilder``.  Only the tail owns materialization; every upstream stage
    propagates EOS and waits for the same sealed terminal artifact.

    All stream waits obey one absolute run deadline.  One additional second is
    reserved only for fail-closed teardown and thread join after that deadline.
    """

    def __init__(
        self,
        bound: BoundStreamProcessor,
        input_reservation: ExactReservation,
        input_cursor: AcquisitionCursor,
        *,
        input_edge: FrozenDatasetEdge,
        output_producer: AcquisitionProducer,
        deadline_monotonic: float,
        output_cursor: AcquisitionCursor | None = None,
        output_builder: DatasetBuilder | None = None,
        downstream_readiness: ExactConsumerReadiness | None = None,
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
        if not isinstance(input_edge, FrozenDatasetEdge):
            raise TypeError("input_edge must be FrozenDatasetEdge")
        if input_edge.expected_cells is None:
            raise ValueError("exact processor input_edge requires a frozen schedule")
        input_stream = input_reservation._stream
        input_edge.validate_stream(input_stream)
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
        if input_key_contract.fingerprint != input_edge.key_contract_fingerprint:
            raise ValueError("input join-key domain differs from input_edge")
        keys = input_edge.expected_cells
        assert keys is not None
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
        if not isinstance(output_producer, AcquisitionProducer):
            raise TypeError("output_producer must be AcquisitionProducer")
        output_stream = output_producer._stream
        if output_stream._payload_contract is not bound.output_payload_contract:
            raise ValueError("processor must share the output stream PayloadContract owner")
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
        expected_output_trace = input_reservation.trace_binding.__class__(
            input_reservation.trace_binding.run_id,
            bound.output_source_id,
        )
        terminal_output = output_cursor is not None or output_builder is not None
        chained_output = downstream_readiness is not None
        if terminal_output == chained_output:
            raise ValueError(
                "processor output must bind exactly one terminal builder or downstream worker"
            )
        if terminal_output:
            if not isinstance(output_cursor, AcquisitionCursor):
                raise TypeError("output_cursor must be AcquisitionCursor")
            if not isinstance(output_builder, DatasetBuilder):
                raise TypeError("output_builder must be DatasetBuilder")
            if output_builder.mode is not DatasetMode.FINITE_EXACT:
                raise ValueError("processor output requires a FINITE_EXACT DatasetBuilder")
            if output_cursor._stream is not output_stream:
                raise ValueError(
                    "output_cursor and output_producer belong to different streams"
                )
            if output_builder._source is not output_stream:
                raise ValueError("output_builder belongs to another stream")
            output_reservation = output_builder._reservation
            if output_reservation is None or output_reservation._cursor is not output_cursor:
                raise ValueError("output cursor is not the output builder reservation cursor")
            if output_cursor._reservation_token is not output_reservation._token:
                raise ValueError("output cursor reservation identity differs from builder")
            if output_reservation.trace_binding != expected_output_trace:
                raise ValueError(
                    "output reservation TraceBinding differs from processor lineage"
                )
            if output_builder._expected_cells != keys:
                raise ValueError("output builder schedule differs from expected_keys")
            if (
                output_builder.edge.exact_key_sequence_digest
                != input_edge.exact_key_sequence_digest
            ):
                raise ValueError("output builder key sequence differs from input edge")
            output_sink = _TerminalDatasetOutput(output_cursor, output_builder)
            downstream = output_sink.readiness
        else:
            if output_cursor is not None or output_builder is not None:
                raise ValueError(
                    "chained processor output cannot also own a terminal builder"
                )
            if not isinstance(downstream_readiness, ExactConsumerReadiness):
                raise TypeError(
                    "downstream_readiness must be ExactConsumerReadiness"
                )
            downstream_readiness._validate_emitter(
                stream=output_stream,
                trace_binding=expected_output_trace,
                payload_contract_fingerprint=output_stream.payload_contract_fingerprint,
                join_key_contract_fingerprint=output_key_contract.fingerprint,
                source_key_sequence_digest=input_edge.exact_key_sequence_digest,
                total_events=total,
            )
            output_sink = _ProcessorStageOutput(downstream_readiness)
            downstream = downstream_readiness
        chain_digest = canonical_digest(
            {
                "contract": "zlc_neutral_atom.ExactStreamProcessorChain/v1",
                "processor": bound.fingerprint,
                "source_contract": input_edge.consumer_contract_digest,
                "source_schedule": input_edge.schedule_digest,
                "downstream": downstream.chain_contract_digest,
                "events": total,
            }
        )
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be CancellationToken or None")
        # Initialize every callback-visible field before the final authority
        # claim.  After _claim_consumer succeeds there are no ordinary
        # construction steps left that could strand a half-built owner.
        self._bound = bound
        self._input_edge = input_edge
        self._input_reservation = input_reservation
        self._input_cursor = input_cursor
        self._input_stream = input_stream
        self._expected_keys = keys
        self._output_producer = output_producer
        self._output_sink = output_sink
        self._cancellation = cancellation or CancellationToken()
        self._deadline_monotonic = deadline_monotonic
        self._condition = threading.Condition(threading.Lock())
        self._thread: threading.Thread | None = None
        self._input_eos: EndOfStream | None = None
        self._result: SealedDatasetArtifact | None = None
        self._error: BaseException | None = None
        self._done = False
        self._closing = False
        self._readiness: ExactConsumerReadiness | None = None
        self._ordered_input_refs = OrderedEventSpanHasher(
            input_stream.stream_id,
            input_stream.generation,
            input_reservation.start_sequence,
        )
        readiness = input_stream._claim_consumer(
            input_reservation,
            self,
            source_contract_digest=input_edge.consumer_contract_digest,
            source_schedule_digest=input_edge.schedule_digest,
            source_key_sequence_digest=input_edge.exact_key_sequence_digest,
            chain_contract_digest=chain_digest,
            downstream=downstream,
            owner_liveness=self._validate_readiness_liveness,
            owner_completion=self._await_completion_for_upstream,
            owner_cancel=self.cancel,
            processor_stage=ProcessorStageProvenance(
                bound.fingerprint,
                bound.artifact_inputs,
            ),
        )
        self._readiness = readiness

    def exact_readiness(self) -> ExactConsumerReadiness:
        self._validate_readiness_liveness()
        readiness = self._readiness
        if readiness is None:
            raise StreamProcessorError("processor lost its exact readiness proof")
        readiness._validate_terminal_sink()
        return readiness

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
        # Recheck the immediate downstream after construction and before this
        # thread becomes observable.  The worker's own readiness cannot be
        # checked until its thread is alive, but the already-started tail can.
        self._output_sink.readiness._validate_terminal_sink()
        start_failure: BaseException | None = None
        failure_traceback = None
        with self._condition:
            if self._thread is not None or self._done or self._closing:
                raise StreamProcessorError("processor worker can start only once")
            thread = threading.Thread(
                target=self._run,
                name=f"stream-processor:{self._bound.definition.key}",
                daemon=False,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException as error:
                # Thread.start is the worker's atomic liveness commit.  An OS
                # failure must not leave a non-running Thread object owning an
                # ACTIVE exact graph that close() can never join.
                self._thread = None
                self._error = error
                self._closing = True
                start_failure = error
                failure_traceback = error.__traceback__
        if start_failure is not None:
            self._propagate_output_failure(start_failure)
            self._cleanup(False)
            with self._condition:
                self._done = True
                self._condition.notify_all()
            raise start_failure.with_traceback(failure_traceback)

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

    def _await_completion_for_upstream(
        self,
        deadline_monotonic: float,
    ) -> SealedDatasetArtifact:
        """Identity-bound completion used by exactly one enclosing processor stage."""

        remaining = max(0.0, float(deadline_monotonic) - time.monotonic())
        self.wait(remaining)
        with self._condition:
            error = self._error
            result = self._result
        if error is not None:
            raise StreamProcessorError("downstream exact processor failed") from error
        if result is None:
            raise StreamProcessorError(
                "downstream exact processor ended without a sealed dataset"
            )
        return result

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
            self._propagate_output_failure(self._error)
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
            result = self._output_sink.complete(
                self,
                output_eos,
                deadline_monotonic=self._deadline_monotonic,
            )
            self._checkpoint()
            readiness = self._readiness
            if readiness is None:
                raise StreamProcessorError("processor lost its exact readiness proof")
            result = result._with_derivation(
                readiness,
                self._input_event_span(),
            )
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
            self._propagate_output_failure(error)
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
        self._input_stream._validate_consumer_delivery(
            self._input_reservation,
            delivery,
            self,
        )
        if envelope.join_key != expected_key:
            raise StreamProcessorError(
                f"input key at ordinal {index} differs from frozen exact schedule"
            )
        self._ordered_input_refs.update(envelope.ref)
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
            causation_refs=(
                envelope.ref,
                *self._bound.artifact_inputs,
            ),
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
        self._output_sink.acknowledge(
            self,
            emitted,
            deadline_monotonic=self._deadline_monotonic,
            checkpoint=self._checkpoint,
        )
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

    def _input_event_span(self) -> EventSpanRef:
        reservation = self._input_reservation
        return self._ordered_input_refs.seal(reservation.end_sequence)

    def _remaining(self) -> float:
        remaining = self._deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("processor absolute deadline expired")
        return remaining

    def _checkpoint(self) -> None:
        self._cancellation.checkpoint()
        self._remaining()

    def _propagate_output_failure(self, error: BaseException | None) -> None:
        reason = "processor failed" if error is None else f"processor failed: {error}"
        try:
            self._output_producer.fail(SourceFailed(reason))
        except StreamEndedEarly:
            pass
        except BaseException as cleanup_error:
            if error is not None:
                error.add_note(f"output stream failure also failed: {cleanup_error!r}")
        teardown_deadline = max(self._deadline_monotonic, time.monotonic()) + 1.0
        try:
            self._output_sink.cancel_and_join(
                self,
                reason,
                deadline_monotonic=teardown_deadline,
            )
        except BaseException as cleanup_error:
            if error is not None:
                error.add_note(
                    f"downstream processor teardown also failed: {cleanup_error!r}"
                )

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
            self._output_sink.close()
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
