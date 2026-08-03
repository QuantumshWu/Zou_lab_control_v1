"""One headless lifecycle and publication owner for every Logic node."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading

from zlc_neutral_atom.artifact_output import ArtifactOutputDeclaration
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
    LiveDatasetOutputOwner,
)
from zlc_neutral_atom.input_spec import DatasetInputSpec
from zlc_neutral_atom.logic_node import (
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
)
from zlc_neutral_atom.processing.signal_plane import (
    SignalDataPlane,
    SignalPublication,
    SignalValue,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
    LiveDatasetViewSpec,
)
from zlc_neutral_atom.runtime.signal_source import SignalEventSource

from .live_dataset import LiveDatasetPort, _ExactDeltaLivePort
from .owner_mailbox import RunOwnerMailbox
from .run import RunCancelled, RunFailed, RunHandle, RunSnapshot


__all__ = ["LogicNodeExecutionContext", "LogicNodeHost", "LogicNodeObservation"]


_UNRESOLVED = object()


class _StartSuppressed(Exception):
    """Stop won before a finite operation acquired a RunHandle."""


@dataclass(frozen=True, slots=True)
class LogicNodeObservation:
    """The common non-blocking status projection for every Logic node."""

    running: bool
    terminal: bool
    phase: str
    error: str | None = None
    warnings: tuple[str, ...] = ()
    run_snapshot: RunSnapshot | None = None

    def __post_init__(self) -> None:
        if type(self.running) is not bool or type(self.terminal) is not bool:
            raise TypeError("Logic-node observation flags must be bool")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("Logic-node observation phase must be non-empty")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise ValueError("Logic-node observation error must be text or None")
        warnings = tuple(self.warnings)
        if any(not isinstance(value, str) or not value.strip() for value in warnings):
            raise ValueError("Logic-node observation warnings must be non-empty text")
        object.__setattr__(self, "warnings", warnings)
        if self.run_snapshot is not None and not isinstance(
            self.run_snapshot,
            RunSnapshot,
        ):
            raise TypeError("run_snapshot must be RunSnapshot or None")


class LogicNodeExecutionContext:
    """The sole runtime capability passed to a finite leaf operation.

    It contains lifecycle and publication mechanics only.  Device resolution,
    authored fields and project paths were already captured when the descriptor
    bound its one operation against ``LogicNodeApplicationContext``.
    """

    __slots__ = ("_host",)

    def __init__(self, host: "LogicNodeHost") -> None:
        self._host = host

    def cancel_requested(self) -> bool:
        return self._host.cancel_requested

    def start_and_wait(self, starter: Callable[[], RunHandle]) -> object:
        return self._host._start_and_wait(starter)

    def open_live_dataset(
        self,
        spec: LiveDatasetViewSpec,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool = True,
        event_source: SignalEventSource | None = None,
    ) -> LiveDatasetPort:
        return self._host._open_live_dataset(
            spec,
            output_owner=output_owner,
            retain_on_terminal=retain_on_terminal,
            event_source=event_source,
        )

    def open_exact_dataset(
        self,
        spec: ExactDatasetPreviewSpec,
        *,
        projection: object,
    ) -> ExactDatasetPreviewPort:
        return self._host._open_exact_dataset(spec, projection=projection)

    def publish_final(
        self,
        outputs: Mapping[str, FinalDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        return self._host._publish_final(outputs)

    def warn(self, message: str) -> None:
        self._host._warn(message)


class LogicNodeHost:
    """The only public runtime owner for Task, Measurement and Processor.

    ``create`` binds a descriptor exactly once.  Task/Measurement operations run
    on one owner thread and may sequentially wait for flat Runs.  Processor
    operations run on SignalDataPlane's one shared latest-only lane.  These are
    private policies behind the same start/cancel/poll/shutdown surface.
    """

    @classmethod
    def create(
        cls,
        descriptor: LogicNodeDescriptor,
        request: object,
        application_context: LogicNodeApplicationContext,
        instance_id: str,
        request_owner_wake: Callable[[], None],
    ) -> "LogicNodeHost":
        if not isinstance(descriptor, LogicNodeDescriptor):
            raise TypeError("descriptor must be LogicNodeDescriptor")
        identity = str(instance_id).strip()
        if not identity:
            raise ValueError("Logic-node instance id must be non-empty")
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        data_plane = application_context.signal_plane
        if not isinstance(data_plane, SignalDataPlane):
            raise TypeError("application context must expose SignalDataPlane")

        operation = descriptor.bind_execute(request, application_context)
        if not callable(operation):
            raise TypeError("Logic-node bind_execute() must return one callable")
        output_specs = descriptor.outputs_for(request)
        dataset_outputs = tuple(value.declaration for value in output_specs)
        artifact_outputs = tuple(
            value.declaration for value in descriptor.artifact_outputs
        )

        source_signal: str | None = None
        if descriptor.definition.kind == "processor":
            dataset_inputs = tuple(
                value
                for value in descriptor.input_specs
                if isinstance(value, DatasetInputSpec)
            )
            if len(dataset_inputs) != 1:
                raise ValueError(
                    "Processor baseline requires exactly one Dataset input"
                )
            source = application_context.input(dataset_inputs[0])
            if not isinstance(source, str) or not source.strip():
                raise TypeError(
                    "Processor Dataset input must resolve to one signal key"
                )
            source_signal = source.strip()

        return cls(
            descriptor=descriptor,
            request=request,
            instance_id=identity,
            dataset_outputs=dataset_outputs,
            artifact_outputs=artifact_outputs,
            operation=operation,
            source_signal=source_signal,
            data_plane=data_plane,
            request_owner_wake=request_owner_wake,
        )

    def __init__(
        self,
        *,
        descriptor: LogicNodeDescriptor,
        request: object,
        instance_id: str,
        dataset_outputs: tuple[DatasetOutputDeclaration, ...],
        artifact_outputs: tuple[ArtifactOutputDeclaration, ...],
        operation: Callable[..., object],
        source_signal: str | None,
        data_plane: SignalDataPlane,
        request_owner_wake: Callable[[], None],
    ) -> None:
        self._descriptor = descriptor
        self._definition_key = descriptor.definition.key
        self._kind = descriptor.definition.kind
        self._request = request
        self.instance_id = instance_id
        self._dataset_outputs = dataset_outputs
        self._artifact_outputs = artifact_outputs
        self._operation = operation
        self._source_signal = source_signal
        self._data_plane = data_plane
        self._request_owner_wake = request_owner_wake
        self._execution_context = LogicNodeExecutionContext(self)

        self._owner = (
            None
            if self._kind == "processor"
            else RunOwnerMailbox(
                request_owner_wake,
                thread_name_prefix=(
                    f"logic-{self._definition_key.stable_definition_id}"
                ),
                max_workers=1,
            )
        )
        self._closed = False
        self._active = False
        self._terminal = False
        self._phase = "not started"
        self._error: str | None = None
        self._warnings: list[str] = []
        self._result: object = _UNRESOLVED
        self._handle: RunHandle | None = None
        self._snapshot: RunSnapshot | None = None
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._stop_reason = "Host requested stop"
        self._plane_state = False
        self._live_opened = False
        self._final_published = False

    @property
    def descriptor(self) -> LogicNodeDescriptor:
        return self._descriptor

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
        return self._dataset_outputs

    @property
    def artifact_output_declarations(
        self,
    ) -> tuple[ArtifactOutputDeclaration, ...]:
        return self._artifact_outputs

    def signal_key(self, output_name: str) -> str:
        name = str(output_name).strip()
        declared = {
            value.name for value in (*self._dataset_outputs, *self._artifact_outputs)
        }
        if name not in declared:
            raise KeyError(f"undeclared Logic-node output {name!r}")
        return f"@logic/{self.instance_id}/{name}"

    def published_signals(self) -> tuple[str, ...]:
        return tuple(self.signal_key(value.name) for value in self._dataset_outputs)

    def published_artifacts(self) -> tuple[str, ...]:
        return tuple(self.signal_key(value.name) for value in self._artifact_outputs)

    @property
    def running(self) -> bool:
        return self._active

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def handle(self) -> RunHandle | None:
        return self._handle

    @property
    def final_result(self) -> object | None:
        return None if self._result is _UNRESOLVED else self._result

    @property
    def final_result_resolved(self) -> bool:
        return self._result is not _UNRESOLVED

    @property
    def cancel_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def worker_idle(self) -> bool:
        if self._kind == "processor":
            return not self._active
        owner = self._require_owner()
        return owner.worker_idle and owner.owner_reaped

    @property
    def observation(self) -> LogicNodeObservation:
        with self._start_lock:
            warnings = tuple(self._warnings)
        return LogicNodeObservation(
            running=self.running,
            terminal=self.terminal,
            phase=self.phase,
            error=self.last_error,
            warnings=warnings,
            run_snapshot=self._snapshot,
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Logic-node host is closed")
        if self._active:
            return
        self._retire_plane_state()
        self._reset_generation()
        if self._kind == "processor":
            self._start_processor()
        else:
            self._start_finite()

    def cancel(self, reason: str = "Host requested stop") -> None:
        if not self._active:
            return
        if self._kind == "processor":
            self._cancel_processor()
            return
        with self._start_lock:
            self._stop_reason = str(reason)
            self._stop_event.set()
            handle = self._handle
        self._phase = "stopping"
        if handle is not None and not handle.snapshot().state.terminal:
            handle.cancel(reason)

    def poll(self) -> LogicNodeObservation:
        if self._kind != "processor":
            self._poll_finite()
        return self.observation

    def shutdown(self) -> None:
        if self._closed:
            return
        if self._active:
            self.cancel("Host is closing")
            self.poll()
            if self._active:
                raise RuntimeError("cannot close Logic-node host before terminal")
        if self._kind != "processor":
            self._require_owner().shutdown()
        self._data_plane.detach_live(self)
        self._plane_state = False
        self._closed = True

    def _reset_generation(self) -> None:
        self._active = False
        self._terminal = False
        self._phase = "starting"
        self._error = None
        with self._start_lock:
            self._warnings.clear()
        self._result = _UNRESOLVED
        self._handle = None
        self._snapshot = None
        self._stop_event.clear()
        self._stop_reason = "Host requested stop"
        self._live_opened = False
        self._final_published = False

    def _warn(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Logic-node warning must be non-empty text")
        with self._start_lock:
            if not self._active:
                raise RuntimeError("inactive Logic-node cannot publish a warning")
            self._warnings.append(message.strip())
        self._request_owner_wake()

    def _require_owner(self) -> RunOwnerMailbox:
        owner = self._owner
        if owner is None:
            raise RuntimeError("finite Run owner is unavailable for Processor")
        return owner

    def _start_finite(self) -> None:
        if self._dataset_outputs:
            self._data_plane.reserve(self)
            self._plane_state = True
        owner = self._require_owner()
        generation = owner.begin_generation()
        self._active = True

        def execute() -> object:
            if self._stop_event.is_set():
                raise _StartSuppressed()
            result = self._operation(self._execution_context)
            if isinstance(result, RunHandle):
                raise TypeError(
                    "Logic-node operation must wait through execution_context"
                )
            return result

        try:
            owner.submit("execute", execute, generation=generation)
        except BaseException:
            self._active = False
            self._terminal = True
            owner.mark_owner_reaped()
            self._retire_plane_state()
            raise

    def _poll_finite(self) -> None:
        owner = self._require_owner()
        if self._active and self._handle is not None:
            self._snapshot = self._handle.snapshot()
            self._phase = self._snapshot.phase
        for completion in owner.drain_completions():
            if completion.generation != owner.generation:
                continue
            error = completion.future.exception()
            if error is None:
                self._result = completion.future.result()
                self._finish_finite_success()
            else:
                self._finish_finite_failure(error)
            owner.mark_owner_reaped()

    def _finish_finite_success(self) -> None:
        if self._dataset_outputs and not self._final_published:
            if self._live_opened:
                # A successful live operation may have replaced its transient
                # preview with a terminal FINAL publication.  Detaching the
                # live slot must preserve that immutable FINAL generation;
                # retiring it here would make a completed Task disappear from
                # every consumer immediately after it reports ``done``.
                self._detach_plane_state()
            else:
                self._error = (
                    "Logic-node operation returned without publishing its "
                    "declared Dataset outputs"
                )
                self._retire_plane_state()
                self._phase = "failed"
                self._active = False
                self._terminal = True
                return
        elif self._live_opened:
            # FINAL-only producers (or a live producer whose final publication
            # was already recorded) follow the same retain-on-terminal rule.
            self._detach_plane_state()
        self._phase = "done"
        self._active = False
        self._terminal = True

    def _finish_finite_failure(self, error: BaseException) -> None:
        if isinstance(error, (RunCancelled, RunFailed)):
            self._snapshot = error.snapshot
        if isinstance(error, (_StartSuppressed, RunCancelled)):
            self._phase = "cancelled"
            self._error = None
        else:
            self._phase = "failed"
            self._error = f"{type(error).__name__}: {error}"
        self._active = False
        self._terminal = True
        self._result = _UNRESOLVED
        self._retire_plane_state()

    def _start_and_wait(self, starter: Callable[[], RunHandle]) -> object:
        if not callable(starter):
            raise TypeError("Run starter must be callable")
        with self._start_lock:
            if self._stop_event.is_set():
                raise _StartSuppressed()
        handle = starter()
        if not isinstance(handle, RunHandle):
            raise TypeError("Run starter returned no RunHandle")
        with self._start_lock:
            self._handle = handle
            self._require_owner().set_handle(handle)
            cancelled = self._stop_event.is_set()
            reason = self._stop_reason
        if cancelled:
            handle.cancel(reason)
        try:
            return handle.result()
        finally:
            self._snapshot = handle.snapshot()

    def _publish_final(
        self,
        outputs: Mapping[str, FinalDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        if self._kind == "processor":
            raise RuntimeError("Processor outputs publish through their exact parent")
        if not self._active or not self._plane_state:
            raise RuntimeError("Logic-node Dataset generation is not active")
        if self._final_published:
            raise RuntimeError("Logic-node FINAL outputs were already published")
        published = self._data_plane.publish_final(self, outputs)
        self._final_published = True
        return published

    def _attach_live(
        self,
        slot: object,
        *,
        event_source: SignalEventSource | None = None,
    ) -> None:
        if self._kind == "processor":
            raise RuntimeError("Processor output is owned by its shared event lane")
        if self._live_opened:
            raise RuntimeError("one Logic-node generation may open one live Dataset")
        if not self._plane_state:
            raise RuntimeError("Logic-node Dataset generation is not reserved")
        if not callable(getattr(slot, "freeze_live_outputs", None)):
            raise TypeError("live Dataset slot has no typed output materializer")
        if event_source is not None and not isinstance(
            event_source,
            SignalEventSource,
        ):
            raise TypeError("event_source must implement SignalEventSource")
        try:
            slot.set_change_listener(
                lambda: self._data_plane.mark_changed(self, slot)
            )
            self._data_plane.attach(self, slot, event_source=event_source)
        except BaseException:
            slot.close()
            raise
        self._live_opened = True

    def _open_live_dataset(
        self,
        spec: LiveDatasetViewSpec,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool,
        event_source: SignalEventSource | None,
    ) -> LiveDatasetPort:
        slot = LiveDatasetPort(
            spec,
            retain_on_terminal=retain_on_terminal,
            output_owner=output_owner,
        )
        self._attach_live(slot, event_source=event_source)
        return slot

    def _open_exact_dataset(
        self,
        spec: ExactDatasetPreviewSpec,
        *,
        projection: object,
    ) -> ExactDatasetPreviewPort:
        slot = _ExactDeltaLivePort(spec, projection)
        self._attach_live(slot)
        return slot

    def _retire_plane_state(self) -> None:
        if not self._plane_state:
            return
        self._data_plane.retire(self)
        self._plane_state = False

    def _detach_plane_state(self) -> None:
        """End a successful generation without withdrawing a FINAL front."""

        if not self._plane_state:
            return
        self._data_plane.detach_live(self)
        self._plane_state = False

    def _start_processor(self) -> None:
        signal = self._source_signal
        if signal is None:
            raise RuntimeError("Processor has no Dataset input")
        publication = self._data_plane.latest_publication(signal)
        if publication is None:
            raise LookupError(f"Processor input signal {signal!r} is not active")
        self.validate_processor_source(publication.value(signal))
        self._active = True
        self._phase = "starting"
        try:
            self._data_plane.attach_latest_only_processor(
                self,
                source_name=signal,
                initial_publication=publication,
            )
            self._plane_state = True
            self._phase = "running"
        except BaseException as error:
            self._active = False
            self._terminal = True
            self._phase = "failed"
            self._error = f"{type(error).__name__}: {error}"
            raise

    def _cancel_processor(self) -> None:
        self._stop_event.set()
        self._phase = "stopping"
        idle = self._data_plane.cancel_latest_only_processor(self)
        self._plane_state = False
        if idle and self._active:
            self.accept_processor_cancelled()

    def validate_processor_source(self, source: SignalValue | None) -> None:
        if not isinstance(source, SignalValue):
            raise TypeError("Processor input must be SignalValue")
        if source.name != self._source_signal:
            raise ValueError("Processor received a different input signal")
        if not isinstance(source.coverage, MonitorCoverage):
            raise ValueError("latest-only Processor requires monitor coverage")

    def evaluate_processor(
        self,
        source: SignalValue,
    ) -> Mapping[str, LiveDatasetOutput]:
        self.validate_processor_source(source)
        outputs = self._operation(source)
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError("Processor operation must return a non-empty mapping")
        return dict(outputs)

    def accept_processor_result(
        self,
        source: SignalValue,
        source_publication: SignalPublication,
        outputs: Mapping[str, LiveDatasetOutput],
    ) -> None:
        if not self._active or self.cancel_requested:
            return
        self.validate_processor_source(source)
        self._data_plane.publish_processor(
            self,
            outputs,
            source_publication=source_publication,
        )

    def accept_processor_failure(self, error: Exception) -> None:
        if not self._active:
            return
        self._data_plane.withdraw_processor(self)
        self._plane_state = False
        self._active = False
        self._terminal = True
        self._phase = "failed"
        self._error = f"{type(error).__name__}: {error}"

    def accept_processor_cancelled(self) -> None:
        if not self._active:
            return
        self._active = False
        self._terminal = True
        self._phase = "cancelled"
        self._error = None

    def request_processor_owner_wake(self) -> None:
        self._request_owner_wake()
