"""Hosted Processor status is node-local and never impersonates a domain Run."""

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
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.processing.hosted_processor import HostedProcessor
from zlc_neutral_atom.processing.signal_plane import (
    SignalDataPlane,
    SignalPublication,
    SignalValue,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_neutral_atom.runtime.streams import EventRef, StreamId


class _ProcessorPlane(SignalDataPlane):
    """Only the public host calls needed to exercise node-owned status."""

    def __init__(self) -> None:
        self.withdrawn = False

    def attach_latest_only_processor(self, node, **_kwargs) -> None:
        self.node = node

    def cancel_latest_only_processor(self, _node) -> bool:
        return True

    def withdraw_processor(self, _node) -> None:
        self.withdrawn = True


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
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("camera-generation")),
        block,
    )
    name = "camera/frame"
    value = SignalValue(name, snapshot, MonitorCoverage(1, 1, 0, False))
    return SignalPublication(
        EventRef(StreamId("camera-events"), StreamGenerationId("events"), 0),
        {name: value},
        object(),
    )


def _node(plane: _ProcessorPlane) -> HostedProcessor:
    publication = _publication()
    return HostedProcessor(
        definition_key=DefinitionKey("test", "processor"),
        request=object(),
        instance_id="processor-instance",
        dataset_output_declarations=(
            DatasetOutputDeclaration("occupied", "test.occupied"),
        ),
        source_signal="camera/frame",
        initial_publication=publication,
        prepare_application=lambda: object(),
        materialize_publication=lambda _result, _source: None,
        qualify_output=lambda name: f"processor/{name}",
        data_plane=plane,
        request_owner_wake=lambda: None,
    )


def test_hosted_processor_reports_only_node_status_for_cancel_and_failure() -> None:
    cancelled = _node(_ProcessorPlane())
    assert cancelled.poll() is None
    cancelled.start()
    assert cancelled.poll() is cancelled
    assert cancelled.running and not cancelled.terminal
    assert cancelled.phase == "preparing Processor application"
    cancelled.cancel()
    assert cancelled.poll() is cancelled
    assert cancelled.terminal and not cancelled.running
    assert cancelled.phase == "cancelled"

    plane = _ProcessorPlane()
    failed = _node(plane)
    failed.start()
    failed.accept_processor_failure(RuntimeError("bad source"))
    assert failed.poll() is failed
    assert failed.terminal and failed.phase == "failed"
    assert failed.last_error == "RuntimeError: bad source"
    assert plane.withdrawn

    for removed_run_fact in ("run_id", "state", "final_committed"):
        assert not hasattr(failed, removed_run_fact)
