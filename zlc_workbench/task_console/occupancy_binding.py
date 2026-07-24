"""Typed binding and lifecycle for the reactive occupancy Processor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import uuid

from .console_records import console_signal_key
from zlc_frontend.site_map_render import build_occupancy_cell_view
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.readout.calibration import ReadoutModelKind
from zlc_neutral_atom.readout.occupancy import (
    OCCUPANCY_LIVE_OUTPUT_NAMES,
    ReactiveOccupancyMonitorEvaluation,
)
from zlc_neutral_atom.readout.reactive_occupancy_application import (
    PreparedReactiveOccupancyMonitor,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_storage import canonical_text

from .data_plane import ConsoleDataPlane, ConsoleSignalValue

__all__ = [
    "ConsoleProducerBinding",
    "OccupancyBindingIntent",
    "ReactiveOccupancyNode",
]


@dataclass(frozen=True, slots=True)
class OccupancyBindingIntent:
    """Exact Camera input plus one explicit calibration source."""

    camera_frame_signal: str
    calibration_signal: str | None = None
    calibration_ref_path: str | None = None
    model_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        canonical_text(self.camera_frame_signal, "camera_frame_signal")
        signal = self.calibration_signal
        path = self.calibration_ref_path
        if (signal is None) == (path is None):
            raise ValueError(
                "occupancy requires exactly one calibration task output or saved file"
            )
        if signal is not None:
            canonical_text(signal, "calibration_signal")
        if path is not None:
            canonical_text(path, "calibration_ref_path")
        if signal is not None and self.camera_frame_signal == signal:
            raise ValueError(
                "occupancy source and calibration must be distinct console outputs"
            )
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")


@dataclass(frozen=True, slots=True)
class ConsoleProducerBinding:
    """One exact output resolved against one row in this TaskConsole."""

    signal_key: str
    producer_label: str
    definition_key: DefinitionKey
    output_name: str
    request: object
    run_node: object | None
    final_result_resolved: bool
    final_result: object | None

    def __post_init__(self) -> None:
        canonical_text(self.signal_key, "signal_key")
        canonical_text(self.producer_label, "producer_label")
        if not isinstance(self.definition_key, DefinitionKey):
            raise TypeError("definition_key must be DefinitionKey")
        canonical_text(self.output_name, "output_name")
        if not isinstance(self.final_result_resolved, bool):
            raise TypeError("final_result_resolved must be bool")
        if not self.final_result_resolved and self.final_result is not None:
            raise ValueError(
                "an unresolved producer cannot expose a FINAL result"
            )

    @property
    def running(self) -> bool:
        return bool(
            self.run_node is not None
            and getattr(self.run_node, "running", False)
        )


class ReactiveOccupancyNode:
    """Latest-revision Processor over one already-running Camera signal.

    The node owns no device and never starts the selected producer.  Its sole
    shared TaskConsole monitor lane admits the frozen neutral application once,
    then passes each accepted immutable Camera revision to that application's
    ``evaluate`` operation.  This node owns only binding/lifecycle checks and
    atomic result publication.  Counts, occupied, and the validity-aware
    loading rate retain the Camera revision/run/epoch lineage.  The occupied
    value also carries the typed same-shot SiteMap presentation; no geometry or
    judged-frame side signal is manufactured.
    """

    def __init__(
        self,
        spec,
        values,
        *,
        instance_id: str,
        instance_label: str,
        intent: OccupancyBindingIntent,
        source_node: object,
        initial_source: ConsoleSignalValue,
        prepare_application: Callable[[], PreparedReactiveOccupancyMonitor],
        data_plane: ConsoleDataPlane,
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(intent, OccupancyBindingIntent):
            raise TypeError("intent must be OccupancyBindingIntent")
        if source_node is None:
            raise ValueError("reactive occupancy requires a running Camera node")
        if not isinstance(initial_source, ConsoleSignalValue):
            raise TypeError("initial_source must be ConsoleSignalValue")
        if initial_source.name != intent.camera_frame_signal:
            raise ValueError("initial Camera value differs from the selected signal")
        if not isinstance(initial_source.coverage, MonitorCoverage):
            raise ValueError(
                "reactive occupancy requires Camera's typed current-frame view; "
                "it cannot guess a current cell from a formal finite dataset"
            )
        source_generation = getattr(source_node, "lifecycle_generation", None)
        if (
            isinstance(source_generation, bool)
            or not isinstance(source_generation, int)
            or source_generation <= 0
        ):
            raise ValueError(
                "initial Camera value has no accepted source lifecycle generation"
            )
        source_handle = getattr(source_node, "handle", None)
        source_run_id = getattr(getattr(source_handle, "run_id", None), "value", None)
        if source_run_id is not None and source_run_id != initial_source.run_id:
            raise ValueError(
                "initial Camera value does not belong to the selected producer instance"
            )
        if not callable(prepare_application):
            raise TypeError("prepare_application must be callable")
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
        self._values = dict(values)
        self._request = intent
        self._output_declarations = tuple(spec.outputs_for(self._request))
        self._source_node = source_node
        self._source_lifecycle_generation = source_generation
        self._initial_source: ConsoleSignalValue | None = initial_source
        self._source_run_id = initial_source.run_id
        self._source_epoch_id = initial_source.epoch_id
        self._prepare_application = prepare_application
        self._data_plane = data_plane
        self._request_owner_wake = request_owner_wake
        self._run_id = RunId(f"reactive-occupancy-{uuid.uuid4().hex}")
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
    def request(self) -> OccupancyBindingIntent:
        return self._request

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
            raise RuntimeError("reactive occupancy nodes are one-shot")
        self._state = RunState.RUNNING
        self._phase = "admitting calibration"
        source, self._initial_source = self._initial_source, None
        if source is None:
            self._fail(RuntimeError("reactive occupancy lost its initial source"))
            return
        try:
            self._data_plane.attach_monitor_processor(
                self,
                source_name=self._request.camera_frame_signal,
                initial_source=source,
            )
        except Exception as error:
            self._fail(error)

    def cancel(self, _reason: str = "Console user requested stop") -> None:
        if self._state is None or self._state.terminal:
            return
        self._cancel_requested = True
        self._phase = "stopping after current pure transform"
        if self._data_plane.cancel_monitor_processor(self):
            self._accept_monitor_cancelled()

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
            self._fail(RuntimeError("selected Camera producer instance has stopped"))
            return self._snapshot()
        source_handle = getattr(self._source_node, "handle", None)
        handle_run_id = getattr(getattr(source_handle, "run_id", None), "value", None)
        if handle_run_id is not None and handle_run_id != self._source_run_id:
            self._fail(
                RuntimeError(
                    "selected Camera RunHandle differs from the bound frame lineage"
                )
            )
            return self._snapshot()
        return self._snapshot()

    def shutdown(self) -> None:
        self.cancel("TaskConsole is closing")

    def _prepare_monitor_application(self) -> PreparedReactiveOccupancyMonitor:
        return self._prepare_application()

    def _validate_monitor_source(self, source: ConsoleSignalValue) -> None:
        if not isinstance(source, ConsoleSignalValue):
            raise TypeError("occupancy source must be ConsoleSignalValue")
        if source.name != self._request.camera_frame_signal:
            raise ValueError("occupancy received another Camera signal")
        if not isinstance(source.coverage, MonitorCoverage):
            raise ValueError("occupancy requires Camera monitor coverage")
        if (
            source.run_id != self._source_run_id
            or source.epoch_id != self._source_epoch_id
        ):
            raise RuntimeError(
                "selected Camera signal now belongs to another Run/epoch; "
                "restart Occupancy to bind the new producer generation"
            )
        if (
            getattr(self._source_node, "lifecycle_generation", None)
            != self._source_lifecycle_generation
        ):
            raise RuntimeError("selected Camera producer generation changed")

    def _monitor_application_ready(self, application: object) -> None:
        if not isinstance(application, PreparedReactiveOccupancyMonitor):
            raise TypeError(
                "Occupancy prepare did not return its application command"
            )
        if self._state is RunState.RUNNING and not self._cancel_requested:
            self._phase = "waiting for a new Camera revision"

    def _monitor_work_started(self, source: ConsoleSignalValue) -> None:
        self._validate_monitor_source(source)
        if self._state is RunState.RUNNING and not self._cancel_requested:
            self._phase = (
                "classifying Camera revision "
                f"{source.snapshot.ref.revision.value}"
            )

    def _accept_monitor_result(
        self,
        source: ConsoleSignalValue,
        result: object,
    ) -> None:
        if self._state is not RunState.RUNNING or self._cancel_requested:
            return
        self._validate_monitor_source(source)
        if not isinstance(result, ReactiveOccupancyMonitorEvaluation):
            raise TypeError(
                "occupancy worker returned an invalid neutral evaluation"
            )
        evaluation = result
        cell = evaluation.cell
        site_map = cell.site_map
        calibration_identity = cell.calibration_ref.target_ref
        presentation = build_occupancy_cell_view(
            cell.background_value,
            cell.background_ref,
            cell.occupied_value,
            cell.occupied_ref,
            cell.selection,
            site_axis=site_map.site_axis,
            coordinate_frame=site_map.coordinate_frame,
            centers_xy=site_map.coordinates_xy,
            calibration_site_validity=site_map.validity.mask,
            calibration_identity=calibration_identity,
            run_id=source.run_id,
            provenance_epoch_id=source.epoch_id,
            summary=(
                f"Camera run={source.run_id} | "
                f"calibration={calibration_identity} | "
                f"revision={cell.background_ref.revision.value} | "
                f"logical point={cell.logical_point}"
            ),
        )
        self._data_plane.publish_processor(
            self,
            evaluation.outputs,
            run_id=source.run_id,
            epoch_id=source.epoch_id,
            presentations={OCCUPANCY_LIVE_OUTPUT_NAMES[1]: presentation},
        )
        self._phase = "waiting for a new Camera revision"

    def _accept_monitor_failure(self, error: Exception) -> None:
        if self._state is None or self._state.terminal:
            return
        self._error = f"{type(error).__name__}: {error}"
        self._state = RunState.FAILED
        self._phase = "failed"

    def _accept_monitor_cancelled(self) -> None:
        if self._state is RunState.RUNNING:
            self._state = RunState.CANCELLED
            self._phase = "cancelled"

    def _request_monitor_owner_wake(self) -> None:
        self._request_owner_wake()

    def _fail(self, error: Exception) -> None:
        self._accept_monitor_failure(error)
        self._data_plane.cancel_monitor_processor(self)

    def _snapshot(self) -> RunSnapshot:
        state = self._state
        if state is None:
            raise RuntimeError("reactive occupancy has not started")
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
