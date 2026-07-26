"""Future-only stream following and source-neutral Camera output events."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    VALID,
    AxisId,
    AxisSpec,
    DataTransformSpec,
    IndexRangeSelection,
    ReductionMethod,
    ReductionSpec,
    Selection,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraFrameMetadata,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.catalog import DefinitionKey, ProcessorDefinition
from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.input_spec import DatasetInputSpec
from zlc_neutral_atom.logic_node_declaration import (
    LogicNodeDeclaration,
    OutputPresentation,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CameraMeasurementRequest,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.signal_source import (
    camera_signal_event_source,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    SourceFailed,
    StreamEndedEarly,
    StreamId,
    TraceContext,
)
from zlc_neutral_atom.runtime.run import RunId, RunState
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationScheduleRequirement,
    SignalEventAssociationSource,
    SignalOutputProjection,
    StreamSignalEventSource,
    authoritative_signal_event_source,
)
from zlc_workbench.task_console.input_binding import (
    ConsoleDatasetProducerBinding,
    DatasetInputSelection,
    ResolvedDatasetInput,
)
from zlc_workbench.task_console import attachment_builders
from zlc_workbench.task_console.capability import (
    ConsoleNodeHost,
    ConsoleSignalEventSourceProvider,
)
from zlc_workbench.task_console.declaration_projection import (
    project_declaration_spec,
)
from zlc_workbench.task_console.data_plane import ConsoleDataPlane
from zlc_workbench.task_console.processor_node import ConsoleProcessorNode
from zlc_workbench.task_console.run_bridge import ConsoleRunNode


SCHEMA = ValueSchema.scalar(np.dtype("<f8"))


def _trace(sequence: int) -> TraceContext:
    return TraceContext("run", "camera", f"frame-{sequence}")


def _camera_sample(sequence: int) -> CameraSample:
    return CameraSample(
        Value(np.asarray([float(sequence)]), VALID, SCHEMA),
        CameraFrameMetadata(
            source_ordinal=sequence,
            produced_count=sequence + 1,
            frame_stamp=sequence,
            camera_stamp=sequence,
            timestamp_seconds=None,
            timestamp_microseconds=None,
            host_received_at_ns=sequence + 1,
            driver_buffer_index=sequence,
            correlation_id=f"frame-{sequence}",
        ),
    )


def _emit(producer, sequence: int):
    payload = _camera_sample(sequence)
    return producer.emit(
        payload,
        captured_at=payload.metadata.captured_at,
        trace=_trace(sequence),
    )


def test_follow_tap_starts_at_subscription_and_drains_before_terminal() -> None:
    contract = CameraSampleContract(SCHEMA)
    stream, producer = AcquisitionStream.create(StreamId("camera-live"), contract)
    _emit(producer, 0)

    tap = stream.follow()
    first = _emit(producer, 1)
    second = _emit(producer, 2)
    failure = SourceFailed("camera stopped")
    producer.fail(failure)

    assert tap.start_sequence == 1
    assert tap.next() is first
    assert tap.next() is second
    with pytest.raises(SourceFailed) as caught:
        tap.next()
    assert caught.value is failure
    tap.close()


def test_camera_named_cursor_filters_readout_phase_without_owning_camera() -> None:
    contract = CameraSampleContract(SCHEMA)
    stream, producer = AcquisitionStream.create(StreamId("camera-cycle"), contract)
    request = CameraMeasurementRequest(
        DeviceRef("installation", "runtime", "camera"),
        repeat=0,
        frames_per_cycle=3,
    )
    source = camera_signal_event_source(stream, request, contract)
    assert source.value_schema("frame_0") is SCHEMA

    _emit(producer, 0)
    cursor = source.open_signal_cursor("frame_0")
    _emit(producer, 1)
    _emit(producer, 2)
    selected = _emit(producer, 3)

    event = cursor.next(timeout=0.1)
    assert event.value.values.tolist() == [3.0]
    assert event.value.schema is SCHEMA
    assert event.event_ref is selected.event_ref
    assert event.trace is selected.trace
    cursor.close()
    producer.finish()

    with pytest.raises(StreamEndedEarly):
        source.open_signal_cursor("frame_1")


def test_authoritative_signal_projection_commits_multidimensional_area_and_reduce() -> None:
    y_axis = AxisSpec(AxisId("image.y"), "y", SPATIAL_Y, 4, tuple(range(4)))
    x_axis = AxisSpec(AxisId("image.x"), "x", SPATIAL_X, 5, tuple(range(5)))
    image_schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.value(),
        np.dtype("<f8"),
        "count",
    )
    stream, producer = AcquisitionStream.create(
        StreamId("generic-image"),
        ValuePayloadContract(image_schema),
    )
    source = StreamSignalEventSource(
        stream,
        {
            "image": SignalOutputProjection(
                image_schema,
                lambda envelope: envelope.payload,
            )
        },
    )
    spec = DataTransformSpec(
        (
            Selection(
                (
                    IndexRangeSelection(y_axis.axis_id, 1, 3),
                    IndexRangeSelection(x_axis.axis_id, 1, 4),
                )
            ),
            ReductionSpec(
                (y_axis.axis_id, x_axis.axis_id),
                ReductionMethod.SUM,
            ),
        )
    )
    projected = authoritative_signal_event_source(source, "image", spec)
    assert projected is not source
    assert projected.value_schema("image").is_scalar
    assert authoritative_signal_event_source(source, "image", None) is source

    cursor = projected.open_signal_cursor("image")
    raw = np.arange(20, dtype=np.float64).reshape(4, 5)
    envelope = producer.emit(
        Value(raw, VALID, image_schema),
        captured_at=12.5,
        trace=TraceContext("run", "processor", "image-0"),
    )
    event = cursor.next(timeout=0.1)

    assert event.value.values.tolist() == [float(raw[1:3, 1:4].sum())]
    assert event.event_ref is envelope.event_ref
    assert event.trace is envelope.trace
    assert event.captured_at == 12.5
    assert cursor.stream_id == stream.stream_id
    assert cursor.stream_generation == stream.generation
    assert cursor.start_sequence == 0
    cursor.close()
    producer.finish()


def test_console_processor_starts_and_owns_the_optional_event_source() -> None:
    """The generic host exposes owner-derived events without owning upstream."""

    output_names = ("counts", "occupied", "rate")
    schemas = {
        name: ValueSchema.scalar(np.dtype("<f8"))
        for name in output_names
    }
    lifecycle: list[str] = []

    class Upstream:
        cancel_calls = 0

        @staticmethod
        def value_schema(_output_name: str) -> ValueSchema:
            return SCHEMA

        @staticmethod
        def open_signal_cursor(output_name: str):
            return ("upstream", output_name)

        def cancel(self) -> None:
            self.cancel_calls += 1

    class Derived:
        error = None

        def __init__(self) -> None:
            self.output_names = output_names
            self.worker_idle = False
            self.request_close_calls = 0
            self.join_calls = 0

        @staticmethod
        def value_schema(output_name: str) -> ValueSchema:
            return schemas[output_name]

        @staticmethod
        def open_signal_cursor(output_name: str):
            return ("derived", output_name)

        @staticmethod
        def open_associated_signal_cursor(output_name: str):
            return ("derived-associated", output_name)

        @staticmethod
        def signal_association_schedule_requirement(output_name: str):
            return SignalAssociationScheduleRequirement()

        def request_close(self) -> None:
            lifecycle.append("derived.request_close")
            self.request_close_calls += 1

        def join_closed(self) -> None:
            assert self.worker_idle
            lifecycle.append("derived.join_closed")
            self.join_calls += 1

        def finish_worker(self) -> None:
            self.worker_idle = True

    upstream = Upstream()
    derived = Derived()

    class Application:
        @staticmethod
        def evaluate(*_args, **_kwargs):
            return None

        @staticmethod
        def start_signal_events(source):
            assert source is upstream
            lifecycle.append("derived.start")
            return derived

    class DataPlane:
        @staticmethod
        def cancel_latest_only_processor(_node) -> bool:
            lifecycle.append("latest.cancel")
            return True

        @staticmethod
        def withdraw_processor(_node) -> None:
            lifecycle.append("latest.withdraw")

    node = object.__new__(ConsoleProcessorNode)
    node._run_id = RunId("processor-test")
    node._state = RunState.RUNNING
    node._cancel_requested = False
    node._phase = "preparing"
    node._error = None
    node._output_names = output_names
    node._source_event_source = upstream
    node._signal_event_source = None
    node._signal_events_close_requested = False
    node._signal_events_closed = False
    node._processor_lane_retired = False
    node._pending_terminal_state = None
    node._pending_terminal_error = None
    node._data_plane = DataPlane()

    node._processor_application_ready(Application())

    assert lifecycle == ["derived.start"]
    assert node.output_names == output_names
    assert node.value_schema("counts") is schemas["counts"]
    assert node.open_signal_cursor("occupied") == ("derived", "occupied")
    assert isinstance(node, ConsoleSignalEventSourceProvider)
    exposed = node.signal_event_source()
    assert exposed is derived
    assert isinstance(exposed, SignalEventAssociationSource)
    assert exposed.open_associated_signal_cursor("rate") == (
        "derived-associated",
        "rate",
    )
    assert exposed.signal_association_schedule_requirement(
        "occupied"
    ) == SignalAssociationScheduleRequirement()
    assert not node.worker_idle

    node.cancel()

    assert lifecycle == [
        "derived.start",
        "derived.request_close",
        "latest.cancel",
        "latest.withdraw",
    ]
    assert derived.request_close_calls == 1
    assert derived.join_calls == 0
    assert upstream.cancel_calls == 0
    assert not node.worker_idle
    assert node._state is RunState.RUNNING
    assert isinstance(node, ConsoleSignalEventSourceProvider)
    with pytest.raises(RuntimeError, match="not running"):
        node.signal_event_source()

    derived.finish_worker()
    assert node.poll().state is RunState.CANCELLED
    assert lifecycle[-1] == "derived.join_closed"
    assert derived.join_calls == 1
    assert node.worker_idle

    node.shutdown()
    assert derived.request_close_calls == 1
    assert derived.join_calls == 1


def test_console_run_node_exposes_one_stable_typed_source_provider() -> None:
    class AssociatedCommand:
        @staticmethod
        def value_schema(_output_name: str) -> ValueSchema:
            return SCHEMA

        @staticmethod
        def open_signal_cursor(output_name: str):
            return ("ordered", output_name)

        @staticmethod
        def open_associated_signal_cursor(output_name: str):
            return ("associated", output_name)

        @staticmethod
        def signal_association_schedule_requirement(output_name: str):
            return SignalAssociationScheduleRequirement()

    class OrderingOnlyCommand:
        @staticmethod
        def value_schema(_output_name: str) -> ValueSchema:
            return SCHEMA

        @staticmethod
        def open_signal_cursor(output_name: str):
            return ("ordered", output_name)

    node = object.__new__(ConsoleRunNode)
    node._start_pending = True
    node._snapshot = None
    node._handle = None
    node._prepared_command = AssociatedCommand()

    assert isinstance(node, ConsoleSignalEventSourceProvider)
    exposed = node.signal_event_source()
    assert isinstance(exposed, SignalEventAssociationSource)
    assert exposed.open_associated_signal_cursor("frame_0") == (
        "associated",
        "frame_0",
    )
    assert exposed.signal_association_schedule_requirement(
        "frame_0"
    ) == SignalAssociationScheduleRequirement()

    node._prepared_command = OrderingOnlyCommand()
    exposed = node.signal_event_source()
    assert not isinstance(exposed, SignalEventAssociationSource)


def test_console_processor_keeps_live_events_optional() -> None:
    class Application:
        @staticmethod
        def evaluate(*_args, **_kwargs):
            return None

    node = object.__new__(ConsoleProcessorNode)
    node._state = RunState.RUNNING
    node._cancel_requested = False
    node._phase = "preparing"
    node._output_names = ("result",)
    node._source_event_source = None
    node._signal_event_source = None
    node._signal_events_close_requested = False
    node._signal_events_closed = False

    node._processor_application_ready(Application())

    assert node._phase == "waiting for a new source revision"
    assert node.worker_idle
    with pytest.raises(TypeError, match="does not expose live signal events"):
        node.open_signal_cursor("result")


def test_processor_attachment_injects_only_the_structural_source_capability(
    monkeypatch,
) -> None:
    input_spec = DatasetInputSpec("camera", "Camera", ("test.frame",))
    camera_output = DatasetOutputDeclaration("frame", "test.frame")

    class SourceNode:
        running = True

        @staticmethod
        def value_schema(_output_name: str) -> ValueSchema:
            raise AssertionError("attachment must not inspect physical schemas")

        @staticmethod
        def open_signal_cursor(_output_name: str):
            raise AssertionError("attachment must not subscribe")

        @staticmethod
        def cancel() -> None:
            raise AssertionError("attachment must not stop its source")

    source_node = SourceNode()
    resolved = ResolvedDatasetInput(
        DatasetInputSelection(input_spec, "camera/frame"),
        ConsoleDatasetProducerBinding(
            "camera/frame",
            "Camera",
            DefinitionKey("test", "camera"),
            camera_output,
            object(),
            source_node,
        ),
    )
    spec = project_declaration_spec(
        LogicNodeDeclaration(
            definition=ProcessorDefinition(
                DefinitionKey("test", "processor"),
                "Processor",
                "test.processor.config",
            ),
            description="test processor",
            authoring_schema=AuthoringSchema(
                (
                    AuthoringField(
                        "threshold",
                        "float",
                        "Threshold",
                        default=1.0,
                    ),
                )
            ),
            input_specs=(input_spec,),
            outputs=(
                OutputPresentation(
                    DatasetOutputDeclaration("result", "test.result"),
                    "result",
                    "Result",
                    "",
                ),
            ),
            build_request=lambda values: values["threshold"],
            bind_request=lambda request, _inputs: request,
        )
    )
    plane = ConsoleDataPlane()
    host = ConsoleNodeHost(
        plane,
        lambda _spec, _values: {"camera": resolved},
        lambda: None,
    )
    captured = {}

    class CapturingProcessorNode:
        def __init__(self, *_args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        ConsoleNodeHost,
        "current_value",
        lambda _host, _binding: object(),
    )
    monkeypatch.setattr(
        attachment_builders,
        "ConsoleProcessorNode",
        CapturingProcessorNode,
    )
    try:
        attachment = attachment_builders.processor_attachment(
            spec,
            bind_request=lambda config, _inputs: config,
            prepare=lambda request: request,
        )
        attachment.create_node(
            host,
            spec,
            {"threshold": 1.0, "camera": "camera/frame"},
            "processor-instance",
            "Processor",
        )
    finally:
        plane.close()

    assert captured["source_event_source"] is source_node
