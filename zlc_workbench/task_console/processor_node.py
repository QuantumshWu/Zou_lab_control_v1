"""Generic TaskConsole lifecycle node for one source-driven Processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Callable
import uuid

from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_workbench.input_binding import ResolvedDatasetInput

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
        self._source_node = source_node
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
    def running(self) -> bool:
        return self._state is RunState.RUNNING

    @property
    def lifecycle_generation(self) -> int:
        """Stable positive generation for this concrete node instance."""

        return self._lifecycle_generation

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def final_result(self):
        return None

    @property
    def final_result_resolved(self) -> bool:
        return False

    @property
    def final_outputs_resolved(self) -> bool:
        return False

    @property
    def materialized_final_outputs(self):
        return None

    @property
    def materialized_final_presentations(self):
        return None

    @property
    def final_output_error(self) -> None:
        return None

    def signal_key(self, output_name: str) -> str:
        return console_signal_key(self.instance_id, output_name)

    def published_signals(self) -> tuple[str, ...]:
        return tuple(
            self.signal_key(output.name)
            for output in self._output_declarations
        )

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
        if self._data_plane.cancel_latest_only_processor(self):
            self._accept_processor_cancelled()

    def poll(self) -> RunSnapshot | None:
        if self._state is None:
            return None
        if self._state.terminal:
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
        if self._state is RunState.RUNNING and not self._cancel_requested:
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
            run_id=source.run_id,
            epoch_id=source.epoch_id,
            presentations=publication.presentations,
        )
        self._phase = "waiting for a new source revision"

    def _accept_processor_failure(self, error: Exception) -> None:
        if self._state is None or self._state.terminal:
            return
        self._error = f"{type(error).__name__}: {error}"
        self._state = RunState.FAILED
        self._phase = "failed"

    def _accept_processor_cancelled(self) -> None:
        if self._state is RunState.RUNNING:
            self._state = RunState.CANCELLED
            self._phase = "cancelled"

    def _request_processor_owner_wake(self) -> None:
        self._request_owner_wake()

    def _fail(self, error: Exception) -> None:
        self._accept_processor_failure(error)
        self._data_plane.cancel_latest_only_processor(self)

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
