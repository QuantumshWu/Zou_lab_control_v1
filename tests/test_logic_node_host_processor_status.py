"""The one LogicNodeHost projects Processor status without inventing a Run."""

from __future__ import annotations

import numpy as np

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.input_spec import DatasetInputSpec
from zlc_neutral_atom.logic_node import DatasetOutputSpec, LogicNodeDescriptor
from zlc_neutral_atom.processing.signal_plane import (
    SignalDataPlane,
    SignalPublication,
    SignalValue,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.hosted_run import LogicNodeHost
from zlc_neutral_atom.runtime.streams import EventRef, StreamId


SOURCE = DatasetInputSpec("source", "Source", None)
OUTPUT = DatasetOutputDeclaration("occupied", "test.occupied")


class _ProcessorPlane(SignalDataPlane):
    def __init__(self, publication: SignalPublication) -> None:
        self.publication = publication
        self.withdrawn = False

    def latest_publication(self, _name: str) -> SignalPublication:
        return self.publication

    def attach_latest_only_processor(self, node, **_kwargs) -> None:
        self.node = node

    def cancel_latest_only_processor(self, _node) -> bool:
        return True

    def withdraw_processor(self, _node) -> None:
        self.withdrawn = True

    def detach_live(self, _node) -> None:
        return None


class _Context:
    def __init__(self, plane: _ProcessorPlane) -> None:
        self.signal_plane = plane

    def input(self, spec: DatasetInputSpec) -> str:
        assert spec is SOURCE
        return "camera/frame"


def _publication() -> SignalPublication:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId("point"), "point", SCAN_POINT, 1, (0,))
    schema = DatasetSchema(
        repeat,
        PointTable(
            1,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    point.coordinates,
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("uint8")),
    )
    block = DataBlock(
        BlockId("frame"),
        DatasetRevision(1),
        np.ones((1, 1, 1), dtype=np.uint8),
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        schema,
    )
    value = SignalValue(
        "camera/frame",
        OwnedSnapshot(block.ref(StreamGenerationId("camera-generation")), block),
        MonitorCoverage(1, 1, 0, False),
    )
    return SignalPublication(
        EventRef(StreamId("camera-events"), StreamGenerationId("events"), 0),
        {value.name: value},
        object(),
    )


def _host(plane: _ProcessorPlane) -> LogicNodeHost:
    descriptor = LogicNodeDescriptor(
        api_name="probe",
        definition=LogicNodeDefinition(
            DefinitionKey("tests.probe", "processor"),
            "Probe",
            "processor",
        ),
        description="",
        authoring_schema=AuthoringSchema(()),
        input_specs=(SOURCE,),
        outputs=(DatasetOutputSpec(OUTPUT, "Occupied"),),
        build_request=lambda _values: object(),
        bind_execute=lambda _request, _context: lambda _source: {},
    )
    return LogicNodeHost.create(
        descriptor,
        object(),
        _Context(plane),
        "processor-instance",
        lambda: None,
    )


def test_logic_node_host_reports_only_processor_status() -> None:
    publication = _publication()
    cancelled = _host(_ProcessorPlane(publication))
    assert cancelled.observation.phase == "not started"
    cancelled.start()
    assert cancelled.running and not cancelled.terminal
    assert cancelled.phase == "running"
    cancelled.cancel()
    assert cancelled.terminal and not cancelled.running
    assert cancelled.phase == "cancelled"
    cancelled.shutdown()

    plane = _ProcessorPlane(publication)
    failed = _host(plane)
    failed.start()
    failed.accept_processor_failure(RuntimeError("bad source"))
    assert failed.terminal and failed.phase == "failed"
    assert failed.last_error == "RuntimeError: bad source"
    assert plane.withdrawn
    failed.shutdown()

    for removed_run_fact in ("run_id", "state", "final_committed"):
        assert not hasattr(failed, removed_run_fact)
