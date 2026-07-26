"""Generic TaskConsole lifecycle node for one source-driven Processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Callable
import uuid

from zlc_data import ValueSchema
from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from .input_binding import ResolvedDatasetInput

from .console_records import console_signal_key
from .data_plane import ConsoleDataPlane, ConsoleSignalValue

__all__ = [
    "ConsoleProcessorNode",
    "ConsoleProcessorPublication",
]


_PROCESSOR_LIFECYCLE_GENERATIONS = count(1)


@dataclass(frozen=True, slots=True)
class ConsoleProcessorPublication:
    """One typed result transaction plus optional frontend presentations."""

    outputs: Mapping[str, LiveDatasetOutput]
    presentations: Mapping[str, object]

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
        if not isinstance(self.presentations, Mapping):
            raise TypeError("Processor presentations must be a mapping")
        presentations = dict(self.presentations)
        if not set(presentations).issubset(outputs):
            raise ValueError("Processor presentation has no matching output")
        object.__setattr__(self, "outputs", MappingProxyType(outputs))
        object.__setattr__(
            self,
            "presentations",
            MappingProxyType(presentations),
        )


class ConsoleProcessorNode:
    """Domain-neutral Processor identity over one admitted live source.

    The node owns binding, lifecycle validation, and atomic publication.  Its
    application owns evaluation, while a narrow Workbench adapter converts the
    typed result into already-materialized outputs and presentation values.
    Delivery is delegated to the data plane's latest-only worker lane.
    """

    def __init__(
        self,
        spec,
        values,
        *,
        instance_id: str,
        instance_label: str,
        request: object,
        source_input: ResolvedDatasetInput,
        initial_source: ConsoleSignalValue,
        source_event_source: SignalEventSource | None = None,
        prepare_application: Callable[[], object],
        materialize_publication: Callable[
            [object, ConsoleSignalValue], ConsoleProcessorPublication
        ],
        data_plane: ConsoleDataPlane,
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(source_input, ResolvedDatasetInput):
            raise TypeError("source_input must be ResolvedDatasetInput")
        if source_input.transform_spec is not None:
            raise ValueError("latest-only Processor input must be a direct output")
        source_node = source_input.producer.run_node
        if source_node is None:
            raise ValueError("Processor input requires a running producer node")
        if not isinstance(initial_source, ConsoleSignalValue):
            raise TypeError("initial_source must be ConsoleSignalValue")
        source_signal = source_input.selection.signal_key
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
        source_generation = getattr(source_node, "lifecycle_generation", None)
        if (
            isinstance(source_generation, bool)
            or not isinstance(source_generation, int)
            or source_generation <= 0
        ):
            raise ValueError("initial source has no accepted lifecycle generation")
        source_handle = getattr(source_node, "handle", None)
        source_run_id = getattr(getattr(source_handle, "run_id", None), "value", None)
        if source_run_id is not None and source_run_id != initial_source.run_id:
            raise ValueError("initial source belongs to another producer instance")
        if not callable(prepare_application):
            raise TypeError("prepare_application must be callable")
        if not callable(materialize_publication):
            raise TypeError("materialize_publication must be callable")
        if not isinstance(data_plane, ConsoleDataPlane):
            raise TypeError("data_plane must be ConsoleDataPlane")
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        identity = str(instance_id).strip()
        label = str(instance_label).strip()
        if not identity or not label:
            raise ValueError("console instance id and label must be non-empty")
        self._spec = spec
        self.instance_id = identity
        self.instance_label = label
        self._lifecycle_generation = next(_PROCESSOR_LIFECYCLE_GENERATIONS)
        self._values = dict(values)
        self._request = request
        self._source_input = source_input
        self._source_signal = source_signal
        self._output_declarations = tuple(spec.outputs_for(request))
        self._output_names = tuple(
            declaration.name for declaration in self._output_declarations
        )
        self._source_node = source_node
        self._source_event_source = source_event_source
        self._signal_event_source: SignalEventSource | None = None
        self._signal_events_close_requested = False
        self._signal_events_closed = False
        self._source_lifecycle_generation = source_generation
        self._initial_source: ConsoleSignalValue | None = initial_source
        self._source_run_id = initial_source.run_id
        self._source_epoch_id = initial_source.epoch_id
        self._prepare_application = prepare_application
        self._materialize_publication = materialize_publication
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
    def spec(self):
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def display_label(self) -> str:
        return self.instance_label

    @property
    def layer(self) -> str:
        return self._spec.kind

    @property
    def prefix(self) -> str:
        return ""

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
    def output_declarations(self) -> tuple:
        return self._output_declarations

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
        return console_signal_key(self.instance_id, output_name)

    def published_signals(self) -> tuple[str, ...]:
        return tuple(
            self.signal_key(output.name)
            for output in self._output_declarations
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

    def cancel(self, _reason: str = "Console user requested stop") -> None:
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
            getattr(self._source_node, "lifecycle_generation", None)
            != self._source_lifecycle_generation
            or not bool(getattr(self._source_node, "running", False))
        ):
            self._fail(RuntimeError("selected producer instance has stopped"))
            return self._snapshot()
        source_handle = getattr(self._source_node, "handle", None)
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
        self.cancel("TaskConsole is closing")
        self._finish_pending_terminal()

    def _prepare_processor_application(self) -> object:
        return self._prepare_application()

    def _validate_processor_source(self, source: ConsoleSignalValue) -> None:
        if not isinstance(source, ConsoleSignalValue):
            raise TypeError("Processor source must be ConsoleSignalValue")
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
            getattr(self._source_node, "lifecycle_generation", None)
            != self._source_lifecycle_generation
        ):
            raise RuntimeError("selected producer generation changed")

    def _processor_application_ready(self, application: object) -> None:
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

    def _processor_work_started(self, source: ConsoleSignalValue) -> None:
        self._validate_processor_source(source)
        if self._state is RunState.RUNNING and not self._cancel_requested:
            self._phase = (
                "processing source revision "
                f"{source.snapshot.ref.revision.value}"
            )

    def _accept_processor_result(
        self,
        source: ConsoleSignalValue,
        result: object,
    ) -> None:
        if self._state is not RunState.RUNNING or self._cancel_requested:
            return
        self._validate_processor_source(source)
        publication = self._materialize_publication(result, source)
        if not isinstance(publication, ConsoleProcessorPublication):
            raise TypeError("Processor result adapter returned another value")
        self._data_plane.publish_processor(
            self,
            publication.outputs,
            source=source,
            presentations=publication.presentations,
        )
        self._phase = "waiting for a new source revision"

    def _accept_processor_failure(self, error: Exception) -> None:
        if self._state is None or self._state.terminal:
            return
        self._pending_terminal_state = RunState.FAILED
        self._pending_terminal_error = f"{type(error).__name__}: {error}"
        self._processor_lane_retired = True
        self._request_signal_events_close()
        self._data_plane.withdraw_processor(self)
        self._finish_pending_terminal()

    def _accept_processor_cancelled(self) -> None:
        if self._state is RunState.RUNNING:
            if self._pending_terminal_state is None:
                self._pending_terminal_state = RunState.CANCELLED
            self._processor_lane_retired = True
            self._request_signal_events_close()
            self._data_plane.withdraw_processor(self)
            self._finish_pending_terminal()

    def _request_processor_owner_wake(self) -> None:
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
