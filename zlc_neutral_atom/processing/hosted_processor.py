"""Headless hosted lifecycle for one source-driven Processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable
import uuid

from zlc_data import ValueSchema
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from .signal_plane import SignalDataPlane, SignalPublication, SignalValue

__all__ = [
    "HostedProcessor",
    "ProcessorPublication",
]


@dataclass(frozen=True, slots=True)
class ProcessorPublication:
    """One complete typed Processor result transaction."""

    outputs: Mapping[str, LiveDatasetOutput]

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, Mapping) or not self.outputs:
            raise ValueError("Processor publication requires typed outputs")
        outputs = dict(self.outputs)
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in outputs
        ):
            raise ValueError("Processor publication names must be canonical text")
        if any(not isinstance(value, LiveDatasetOutput) for value in outputs.values()):
            raise TypeError("Processor publication values must be LiveDatasetOutput")
        object.__setattr__(self, "outputs", MappingProxyType(outputs))


class HostedProcessor:
    """Domain-neutral Processor identity over one admitted live source.

    The node owns binding, lifecycle validation, and atomic publication.  Its
    application owns evaluation, while a composition adapter converts the
    typed result into already-materialized outputs. Delivery is delegated to
    the signal plane's latest-only worker lane.
    """

    def __init__(
        self,
        *,
        definition_key: DefinitionKey,
        request: object,
        instance_id: str,
        dataset_output_declarations: tuple[DatasetOutputDeclaration, ...],
        source_signal: str,
        initial_publication: SignalPublication,
        prepare_application: Callable[[], object],
        materialize_publication: Callable[
            [object, SignalValue], ProcessorPublication
        ],
        qualify_output: Callable[[str], str],
        data_plane: SignalDataPlane,
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(definition_key, DefinitionKey):
            raise TypeError("definition_key must be DefinitionKey")
        if not isinstance(initial_publication, SignalPublication):
            raise TypeError("initial_publication must be SignalPublication")
        source_signal = str(source_signal).strip()
        if not source_signal:
            raise ValueError("source_signal must be non-empty")
        initial_source = initial_publication.value(source_signal)
        if not isinstance(initial_source, SignalValue):
            raise ValueError("initial publication lacks the selected signal")
        if initial_source.name != source_signal:
            raise ValueError("initial source differs from the selected signal")
        if not isinstance(initial_source.coverage, MonitorCoverage):
            raise ValueError(
                "latest-only Processor input requires typed monitor coverage"
            )
        if not callable(prepare_application):
            raise TypeError("prepare_application must be callable")
        if not callable(materialize_publication):
            raise TypeError("materialize_publication must be callable")
        if not callable(qualify_output):
            raise TypeError("qualify_output must be callable")
        if not isinstance(data_plane, SignalDataPlane):
            raise TypeError("data_plane must be SignalDataPlane")
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        identity = str(instance_id).strip()
        if not identity:
            raise ValueError("hosted instance id must be non-empty")
        declarations = tuple(dataset_output_declarations)
        if any(
            not isinstance(value, DatasetOutputDeclaration)
            for value in declarations
        ):
            raise TypeError(
                "dataset_output_declarations must contain "
                "DatasetOutputDeclaration values"
            )
        output_names = tuple(value.name for value in declarations)
        if len(set(output_names)) != len(output_names):
            raise ValueError("Processor output names must be unique")
        if not output_names:
            raise ValueError("Processor must declare at least one Dataset output")
        self._definition_key = definition_key
        self.instance_id = identity
        self._request = request
        self._source_signal = source_signal
        self._dataset_output_declarations = declarations
        self._output_names = output_names
        self._signal_event_source: SignalEventSource | None = None
        self._signal_events_close_requested = False
        self._signal_events_closed = False
        self._source_signal_generation = initial_publication.generation
        self._initial_publication: SignalPublication | None = initial_publication
        self._source_run_id = initial_source.run_id
        self._source_epoch_id = initial_source.epoch_id
        self._prepare_application = prepare_application
        self._prepared_application: object | None = None
        self._materialize_publication = materialize_publication
        self._qualify_output = qualify_output
        self._data_plane = data_plane
        self._request_owner_wake = request_owner_wake
        self._run_id = RunId(f"processor-{uuid.uuid4().hex}")
        self._state: RunState | None = None
        self._phase = "not started"
        self._error: str | None = None
        self._cancel_requested = False
        self._processor_lane_retired = False
        self._pending_terminal_state: RunState | None = None
        self._pending_terminal_error: str | None = None

    @property
    def definition_key(self) -> DefinitionKey:
        return self._definition_key

    @property
    def request(self) -> object:
        return self._request

    @property
    def dataset_output_declarations(
        self,
    ) -> tuple[DatasetOutputDeclaration, ...]:
        return self._dataset_output_declarations

    @property
    def output_names(self) -> tuple[str, ...]:
        """Exact owner-declared names available through the event seam."""

        return self._output_names

    @property
    def source_signal(self) -> str:
        return self._source_signal

    @property
    def prepared_application(self) -> object:
        application = self._prepared_application
        if application is None:
            raise RuntimeError("Processor application is not ready")
        return application

    @property
    def running(self) -> bool:
        return self._state is RunState.RUNNING

    @property
    def last_error(self) -> str | None:
        return self._error

    def signal_key(self, output_name: str) -> str:
        return self._qualify_output(output_name)

    def published_signals(self) -> tuple[str, ...]:
        return tuple(
            self.signal_key(output.name)
            for output in self._dataset_output_declarations
        )

    @property
    def worker_idle(self) -> bool:
        """Whether the optional derived-event worker has fully stopped."""

        source = self._signal_event_source
        return source is None or bool(getattr(source, "worker_idle", False))

    def start(self) -> None:
        if self.running:
            return
        if self._state is not None:
            raise RuntimeError("Processor nodes are one-shot")
        self._state = RunState.RUNNING
        self._phase = "preparing Processor application"
        publication, self._initial_publication = self._initial_publication, None
        if publication is None:
            self._fail(RuntimeError("Processor lost its initial publication"))
            return
        try:
            self._data_plane.attach_latest_only_processor(
                self,
                source_name=self._source_signal,
                initial_publication=publication,
            )
        except Exception as error:
            self._fail(error)

    def cancel(self, _reason: str = "Host requested stop") -> None:
        if self._state is None or self._state.terminal:
            return
        self._cancel_requested = True
        self._phase = "stopping after current Processor evaluation"
        if self._pending_terminal_state is None:
            self._pending_terminal_state = RunState.CANCELLED
        self._request_signal_events_close()
        idle = self._data_plane.cancel_latest_only_processor(self)
        if idle and not self._processor_lane_retired:
            self._processor_lane_retired = True
        self._finish_pending_terminal()

    def poll(self) -> RunSnapshot | None:
        if self._state is None:
            return None
        if self._state.terminal:
            return self._snapshot()
        if self._pending_terminal_state is not None:
            self._finish_pending_terminal()
            return self._snapshot()
        signal_source = self._signal_event_source
        signal_error = (
            None if signal_source is None else getattr(signal_source, "error", None)
        )
        if signal_error is not None:
            if not isinstance(signal_error, Exception):
                signal_error = RuntimeError(str(signal_error))
            self._fail(signal_error)
            return self._snapshot()
        return self._snapshot()

    def shutdown(self) -> None:
        self.cancel("Host is closing")
        self._finish_pending_terminal()

    def prepare_processor_application(self) -> object:
        return self._prepare_application()

    def validate_processor_source(self, source: SignalValue) -> None:
        if not isinstance(source, SignalValue):
            raise TypeError("Processor source must be SignalValue")
        if source.name != self._source_signal:
            raise ValueError("Processor received another selected signal")
        if not isinstance(source.coverage, MonitorCoverage):
            raise ValueError("Processor source requires typed monitor coverage")
        if (
            source.run_id != self._source_run_id
            or source.epoch_id != self._source_epoch_id
        ):
            raise RuntimeError(
                "selected signal now belongs to another run or epoch; "
                "restart the Processor to bind the new producer generation"
            )

    def processor_application_ready(self, application: object) -> None:
        if not callable(getattr(application, "evaluate", None)):
            raise TypeError("Processor prepare returned no evaluable application")
        if self._state is not RunState.RUNNING or self._cancel_requested:
            return
        self._prepared_application = application
        start_signal_events = getattr(application, "start_signal_events", None)
        if start_signal_events is not None:
            if not callable(start_signal_events):
                raise TypeError(
                    "Processor application start_signal_events must be callable"
                )
            (
                _generation,
                upstream,
                _output_name,
                transform,
            ) = self._data_plane.signal_event_binding(
                self._source_signal,
                expected_generation=self._source_signal_generation,
            )
            if transform is not None:
                raise ValueError(
                    "latest-only Processor event input must be a direct output"
                )
            derived = start_signal_events(upstream)
            try:
                self._validate_signal_event_source(derived)
            except BaseException:
                request_close = getattr(derived, "request_close", None)
                if callable(request_close):
                    request_close()
                raise
            self._signal_event_source = derived
            self._signal_events_close_requested = False
            self._signal_events_closed = False
            self._data_plane.bind_processor_event_source(self, derived)
        self._phase = "waiting for a new source revision"

    def processor_work_started(self, source: SignalValue) -> None:
        self.validate_processor_source(source)
        if self._state is RunState.RUNNING and not self._cancel_requested:
            self._phase = (
                "processing source revision "
                f"{source.snapshot.ref.revision.value}"
            )

    def accept_processor_result(
        self,
        source: SignalValue,
        source_publication: SignalPublication,
        result: object,
    ) -> None:
        if self._state is not RunState.RUNNING or self._cancel_requested:
            return
        self.validate_processor_source(source)
        publication = self._materialize_publication(result, source)
        if not isinstance(publication, ProcessorPublication):
            raise TypeError("Processor result adapter returned another value")
        self._data_plane.publish_processor(
            self,
            publication.outputs,
            source_publication=source_publication,
        )
        self._phase = "waiting for a new source revision"

    def accept_processor_failure(self, error: Exception) -> None:
        if self._state is None or self._state.terminal:
            return
        self._pending_terminal_state = RunState.FAILED
        self._pending_terminal_error = f"{type(error).__name__}: {error}"
        self._processor_lane_retired = True
        self._request_signal_events_close()
        self._data_plane.withdraw_processor(self)
        self._finish_pending_terminal()

    def accept_processor_cancelled(self) -> None:
        if self._state is RunState.RUNNING:
            if self._pending_terminal_state is None:
                self._pending_terminal_state = RunState.CANCELLED
            self._processor_lane_retired = True
            self._request_signal_events_close()
            self._finish_pending_terminal()

    def request_processor_owner_wake(self) -> None:
        self._request_owner_wake()

    def _fail(self, error: Exception) -> None:
        if self._state is None or self._state.terminal:
            return
        self._pending_terminal_state = RunState.FAILED
        self._pending_terminal_error = f"{type(error).__name__}: {error}"
        self._request_signal_events_close()
        idle = self._data_plane.cancel_latest_only_processor(self)
        if idle:
            self._processor_lane_retired = True
        self._finish_pending_terminal()

    def _validate_signal_event_source(self, source: object) -> None:
        if not isinstance(source, SignalEventSource):
            raise TypeError(
                "Processor start_signal_events returned no SignalEventSource"
            )
        request_close = getattr(source, "request_close", None)
        join_closed = getattr(source, "join_closed", None)
        if not callable(request_close) or not callable(join_closed):
            raise TypeError(
                "Processor signal source must expose request_close()/join_closed()"
            )
        if type(getattr(source, "worker_idle", None)) is not bool:
            raise TypeError(
                "Processor signal source must expose boolean worker_idle"
            )
        output_names = getattr(source, "output_names", None)
        if output_names is not None and tuple(output_names) != self._output_names:
            raise ValueError(
                "Processor signal source differs from the declared output vocabulary"
            )
        for name in self._output_names:
            if not isinstance(source.value_schema(name), ValueSchema):
                raise TypeError(
                    "Processor signal source value_schema() must return ValueSchema"
                )

    def _request_signal_events_close(self) -> None:
        source = self._signal_event_source
        if (
            source is None
            or self._signal_events_close_requested
            or self._signal_events_closed
        ):
            return
        request_close = getattr(source, "request_close", None)
        if not callable(request_close):
            raise TypeError("Processor signal source lost its request_close() seam")
        request_close()
        self._signal_events_close_requested = True

    def _join_signal_events_if_idle(self) -> bool:
        source = self._signal_event_source
        if source is None or self._signal_events_closed:
            return True
        if not bool(getattr(source, "worker_idle", False)):
            return False
        join_closed = getattr(source, "join_closed", None)
        if not callable(join_closed):
            raise TypeError("Processor signal source lost its join_closed() seam")
        join_closed()
        self._signal_events_closed = True
        return True

    def _finish_pending_terminal(self) -> bool:
        state = self._pending_terminal_state
        if state is None or not self._processor_lane_retired:
            return False
        if not self._join_signal_events_if_idle():
            self._phase = "waiting for Processor event worker to stop"
            return False
        self._state = state
        self._error = self._pending_terminal_error
        self._phase = "failed" if state is RunState.FAILED else "cancelled"
        self._pending_terminal_state = None
        self._pending_terminal_error = None
        return True

    def _snapshot(self) -> RunSnapshot:
        state = self._state
        if state is None:
            raise RuntimeError("Processor has not started")
        return RunSnapshot(
            self._run_id,
            state,
            self._phase,
            False,
            None,
            self._error,
            (),
            (),
        )
