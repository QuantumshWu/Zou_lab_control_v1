"""Headless hosted lifecycle for one source-driven Processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Callable, Protocol, runtime_checkable
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
from .signal_plane import SignalDataPlane, SignalValue

__all__ = [
    "HostedProcessor",
    "HostedProcessorSource",
    "ProcessorPublication",
]


_PROCESSOR_LIFECYCLE_GENERATIONS = count(1)


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


@runtime_checkable
class HostedProcessorSource(Protocol):
    """Lifecycle facts required to pin a Processor to one producer run."""

    @property
    def lifecycle_generation(self) -> int: ...

    @property
    def running(self) -> bool: ...

    @property
    def handle(self) -> object: ...


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
        source_node: HostedProcessorSource,
        initial_source: SignalValue,
        source_event_source: SignalEventSource | None = None,
        prepare_application: Callable[[], object],
        materialize_publication: Callable[
            [object, SignalValue], ProcessorPublication
        ],
        qualify_output: Callable[[str], str],
        publication_observer: Callable[
            [object, SignalValue, Mapping[str, SignalValue]], None
        ]
        | None,
        data_plane: SignalDataPlane,
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(definition_key, DefinitionKey):
            raise TypeError("definition_key must be DefinitionKey")
        if not isinstance(source_node, HostedProcessorSource):
            raise TypeError("source_node must implement HostedProcessorSource")
        if not isinstance(initial_source, SignalValue):
            raise TypeError("initial_source must be SignalValue")
        source_signal = str(source_signal).strip()
        if not source_signal:
            raise ValueError("source_signal must be non-empty")
        if initial_source.name != source_signal:
            raise ValueError("initial source differs from the selected signal")
        if not isinstance(initial_source.coverage, MonitorCoverage):
            raise ValueError(
                "latest-only Processor input requires typed monitor coverage"
            )
        if source_event_source is not None and not isinstance(
            source_event_source,
            SignalEventSource,
        ):
            raise TypeError(
                "source_event_source must implement SignalEventSource or be None"
            )
        source_generation = source_node.lifecycle_generation
        if (
            isinstance(source_generation, bool)
            or not isinstance(source_generation, int)
            or source_generation <= 0
        ):
            raise ValueError("initial source has no accepted lifecycle generation")
        source_handle = source_node.handle
        source_run_id = getattr(getattr(source_handle, "run_id", None), "value", None)
        if source_run_id is not None and source_run_id != initial_source.run_id:
            raise ValueError("initial source belongs to another producer instance")
        if not callable(prepare_application):
            raise TypeError("prepare_application must be callable")
        if not callable(materialize_publication):
            raise TypeError("materialize_publication must be callable")
        if not callable(qualify_output):
            raise TypeError("qualify_output must be callable")
        if publication_observer is not None and not callable(publication_observer):
            raise TypeError("publication_observer must be callable or None")
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
        self._lifecycle_generation = next(_PROCESSOR_LIFECYCLE_GENERATIONS)
        self._request = request
        self._source_signal = source_signal
        self._dataset_output_declarations = declarations
        self._output_names = output_names
        self._source_node = source_node
        self._source_event_source = source_event_source
        self._signal_event_source: SignalEventSource | None = None
        self._signal_events_close_requested = False
        self._signal_events_closed = False
        self._source_lifecycle_generation = source_generation
        self._initial_source: SignalValue | None = initial_source
        self._source_run_id = initial_source.run_id
        self._source_epoch_id = initial_source.epoch_id
        self._prepare_application = prepare_application
        self._materialize_publication = materialize_publication
        self._qualify_output = qualify_output
        self._publication_observer = publication_observer
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
    def input_nodes(self) -> tuple[object, ...]:
        return (self._source_node,)

    @property
    def handle(self):
        return None

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
    def running(self) -> bool:
        return self._state is RunState.RUNNING

    @property
    def lifecycle_generation(self) -> int:
        """Stable positive generation for this concrete node instance."""

        return self._lifecycle_generation

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

    def value_schema(self, output_name: str) -> ValueSchema:
        """Return one running Processor output's owner-defined event schema."""

        name = self._require_output_name(output_name)
        source = self._running_signal_source()
        schema = source.value_schema(name)
        if not isinstance(schema, ValueSchema):
            raise TypeError("Processor signal source returned another schema type")
        return schema

    def open_signal_cursor(self, output_name: str):
        """Open a future-only cursor over one running Processor output."""

        name = self._require_output_name(output_name)
        return self._running_signal_source().open_signal_cursor(name)

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
        source, self._initial_source = self._initial_source, None
        if source is None:
            self._fail(RuntimeError("Processor lost its initial source"))
            return
        try:
            self._data_plane.attach_latest_only_processor(
                self,
                source_name=self._source_signal,
                initial_source=source,
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
            self._data_plane.withdraw_processor(self)
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
        if (
            self._source_node.lifecycle_generation
            != self._source_lifecycle_generation
            or not self._source_node.running
        ):
            self._fail(RuntimeError("selected producer instance has stopped"))
            return self._snapshot()
        source_handle = self._source_node.handle
        handle_run_id = getattr(
            getattr(source_handle, "run_id", None),
            "value",
            None,
        )
        if handle_run_id is not None and handle_run_id != self._source_run_id:
            self._fail(
                RuntimeError(
                    "selected producer RunHandle differs from the bound lineage"
                )
            )
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
        if (
            self._source_node.lifecycle_generation
            != self._source_lifecycle_generation
        ):
            raise RuntimeError("selected producer generation changed")

    def processor_application_ready(self, application: object) -> None:
        if not callable(getattr(application, "evaluate", None)):
            raise TypeError("Processor prepare returned no evaluable application")
        if self._state is not RunState.RUNNING or self._cancel_requested:
            return
        start_signal_events = getattr(application, "start_signal_events", None)
        if start_signal_events is not None:
            if not callable(start_signal_events):
                raise TypeError(
                    "Processor application start_signal_events must be callable"
                )
            upstream = self._source_event_source
            if upstream is None:
                raise TypeError(
                    "Processor live events require a SignalEventSource input"
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
        result: object,
    ) -> None:
        if self._state is not RunState.RUNNING or self._cancel_requested:
            return
        self.validate_processor_source(source)
        publication = self._materialize_publication(result, source)
        if not isinstance(publication, ProcessorPublication):
            raise TypeError("Processor result adapter returned another value")
        published = self._data_plane.publish_processor(
            self,
            publication.outputs,
            source=source,
        )
        observer = self._publication_observer
        if observer is not None:
            observer(result, source, published)
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
            self._data_plane.withdraw_processor(self)
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
        self._data_plane.withdraw_processor(self)
        self._finish_pending_terminal()

    def _require_output_name(self, output_name: str) -> str:
        if (
            not isinstance(output_name, str)
            or not output_name
            or output_name.strip() != output_name
        ):
            raise ValueError("Processor output name must be canonical text")
        if output_name not in self._output_names:
            raise KeyError(f"Processor has no output {output_name!r}")
        return output_name

    def _running_signal_source(self) -> SignalEventSource:
        if self._state is not RunState.RUNNING or self._cancel_requested:
            raise RuntimeError("Processor signal events are not running")
        source = self._signal_event_source
        if source is None:
            raise TypeError("this Processor does not expose live signal events")
        return source

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

    def signal_event_source(self) -> SignalEventSource:
        """Return the application-owned source without changing this node's type."""

        return self._running_signal_source()

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
            None,
        )
