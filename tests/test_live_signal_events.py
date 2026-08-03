"""Future-only stream following and exact event-association routing."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    VALID,
    AxisId,
    AxisSourceRef,
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
from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.catalog import DefinitionKey, ProcessorDefinition
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.devices.camera.contract import (
    CameraFrameMetadata,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.input_spec import DatasetInputSpec
from zlc_neutral_atom.installation import DeviceRef
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
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_neutral_atom.runtime.signal_source import (
    SignalOutputProjection,
    StreamSignalEventSource,
    authoritative_signal_event_source,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    SourceFailed,
    StreamEndedEarly,
    StreamId,
)
from zlc_workbench.task_console import attachment_builders
from zlc_workbench.task_console.capability import ConsoleNodeHost
from zlc_workbench.task_console.declaration_projection import (
    project_declaration_spec,
)
from zlc_workbench.task_console.input_binding import (
    ConsoleDatasetProducerBinding,
    DatasetInputSelection,
    ResolvedDatasetInput,
)


SCHEMA = ValueSchema.scalar(np.dtype("<f8"))


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
        ),
    )


def _emit(producer, sequence: int):
    payload = _camera_sample(sequence)
    return producer.emit(
        payload,
        captured_at=payload.metadata.captured_at,
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
        DeviceRef(
            runtime_instance_id="runtime",
            instance_id="camera",
            type_id="camera.test",
            role="camera",
        ),
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
                (
                    AxisSourceRef.tensor(y_axis.axis_id),
                    AxisSourceRef.tensor(x_axis.axis_id),
                ),
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
    )
    event = cursor.next(timeout=0.1)

    assert event.value.values.tolist() == [float(raw[1:3, 1:4].sum())]
    assert event.event_ref is envelope.event_ref
    assert event.captured_at == 12.5
    assert cursor.stream_id == stream.stream_id
    assert cursor.stream_generation == stream.generation
    assert cursor.start_sequence == 0
    cursor.close()
    producer.finish()


def test_processor_attachment_accepts_an_exact_projected_publication(
    monkeypatch,
) -> None:
    input_spec = DatasetInputSpec("camera", "Camera", ("test.frame",))
    camera_output = DatasetOutputDeclaration("frame", "test.frame")

    class SourceNode:
        running = True

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
        DataTransformSpec(
            (
                Selection(
                    (IndexRangeSelection(AxisId("selector.axis"), 0, 1),)
                ),
            )
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
                (AuthoringField("threshold", "float", "Threshold", default=1.0),)
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
    plane = SignalDataPlane()
    host = ConsoleNodeHost(
        data_plane=plane,
        resolve_inputs=lambda _spec, _values: {"camera": resolved},
        request_owner_wake=lambda: None,
    )
    captured = {}
    exact_publication = object()

    class CapturingProcessorNode:
        def __init__(self, *_args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        ConsoleNodeHost,
        "current_publication",
        lambda _host, _binding: exact_publication,
    )
    monkeypatch.setattr(
        attachment_builders,
        "HostedProcessor",
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
        )
    finally:
        plane.close()

    assert captured["source_signal"] == "camera/frame"
    assert captured["initial_publication"] is exact_publication
    assert captured["data_plane"] is plane
    assert "source_node" not in captured
    assert "source_event_source" not in captured
