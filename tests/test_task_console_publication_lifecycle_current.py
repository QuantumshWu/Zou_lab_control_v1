"""Current bounded provenance contract for TaskConsole Rolling fronts."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_neutral_atom.processing.signal_plane import SignalPublication, SignalValue
from zlc_neutral_atom.runtime.streams import EventRef, StreamId
from zlc_plot import RasterOperation, RasterPlotHost, RollingPlot
from zlc_workbench.task_console.panel_card import PanelCard, PanelSurfaceUpdate


_SIGNAL = "camera/count"
_DATA_GENERATION = StreamGenerationId("rolling-data")
_EVENT_GENERATION = StreamGenerationId("rolling-events")


def _schema() -> DatasetSchema:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    point = PointColumn(
        AxisId("scan"),
        "scan",
        SCAN_POINT,
        PointColumn.NUMERIC,
        (0.0,),
    )
    return DatasetSchema(
        repeat,
        PointTable(1, (point,)),
        None,
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )


def _publication(revision: int, schema: DatasetSchema) -> SignalPublication:
    block = DataBlock(
        BlockId(f"rolling-{revision}"),
        DatasetRevision(revision),
        np.asarray([[[float(revision)]]], dtype="<f8"),
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(block.ref(_DATA_GENERATION), block)
    value = SignalValue(_SIGNAL, snapshot, None)
    return SignalPublication(
        EventRef(StreamId("rolling-publications"), _EVENT_GENERATION, revision),
        {_SIGNAL: value},
        object(),
    )


class _AssociationHarness:
    """Only PanelCard's revision/publication association state and transitions."""

    _prune_publications = PanelCard._prune_publications
    _track_surface_submission = PanelCard._track_surface_submission
    _surface_update_revision = staticmethod(PanelCard._surface_update_revision)
    _observe_host_front = PanelCard._observe_host_front
    observe_surface_result = PanelCard.observe_surface_result
    reject_surface_update = PanelCard.reject_surface_update
    _publications_for = PanelCard._publications_for

    def __init__(self) -> None:
        self.panel_id = "rolling-panel"
        self._publication_by_host_revision = {}
        self._unresolved_revisions_by_host = {}
        self._latest_host_revisions = {}
        self._latest_host_sequence = {}
        self._presented_revisions_by_host = {}


def _update(
    harness: _AssociationHarness,
    host: RasterPlotHost,
    publication: SignalPublication,
) -> PanelSurfaceUpdate:
    value = publication.value(_SIGNAL)
    assert value is not None
    revision = value.snapshot.ref.revision.value
    harness._track_surface_submission(host, revision, publication)
    return PanelSurfaceUpdate(
        harness.panel_id,
        revision,
        host,
        publication,
        value,
        Future(),
        False,
    )


def _operation(initial, sequence: int, revisions: tuple[int, ...]) -> RasterOperation:
    identity = replace(
        initial.identity,
        sequence=sequence,
        data_revision=revisions[-1],
    )
    return RasterOperation(
        None,
        replace(initial, identity=identity, source_revisions=revisions),
    )


@pytest.fixture
def rolling_host():
    schema = _schema()
    first = _publication(1, schema).value(_SIGNAL)
    assert first is not None
    host = RasterPlotHost.from_plot(
        first.snapshot,
        RollingPlot(),
        parameters={"window": 3},
    )
    try:
        yield host, host.wait_for_front(timeout=5.0), schema
    finally:
        host.close(timeout=5.0)


def test_publication_association_is_exactly_presented_latest_and_unresolved(
    rolling_host,
) -> None:
    host, initial, schema = rolling_host
    harness = _AssociationHarness()
    publications = {
        revision: _publication(revision, schema) for revision in range(1, 6)
    }
    updates = {
        revision: _update(harness, host, publication)
        for revision, publication in publications.items()
    }

    harness.observe_surface_result(updates[1], _operation(initial, 1, (1,)))
    harness.observe_surface_result(updates[2], _operation(initial, 2, (1, 2)))
    harness._presented_revisions_by_host[host.host_id] = (1, 2)
    harness.observe_surface_result(updates[3], _operation(initial, 3, (1, 2, 3)))
    harness.observe_surface_result(updates[4], _operation(initial, 4, (2, 3, 4)))

    # The visible old window, newest worker window, and unresolved revision are
    # all still legitimate provenance owners; nothing else is retained.
    assert set(harness._publication_by_host_revision[host.host_id]) == {
        1,
        2,
        3,
        4,
        5,
    }

    harness.reject_surface_update(updates[5])
    assert set(harness._publication_by_host_revision[host.host_id]) == {1, 2, 3, 4}

    # Once the latest worker front is also the presented front, the former
    # visible-only revision is released immediately.
    harness._presented_revisions_by_host[host.host_id] = (2, 3, 4)
    harness._prune_publications(host.host_id)
    assert harness._publications_for(host, (2, 3, 4)) == tuple(
        publications[revision] for revision in (2, 3, 4)
    )
    assert set(harness._publication_by_host_revision[host.host_id]) == {2, 3, 4}


def test_stale_success_releases_its_resolved_out_of_window_publication(
    rolling_host,
) -> None:
    host, initial, schema = rolling_host
    harness = _AssociationHarness()
    publications = {
        revision: _publication(revision, schema) for revision in range(3, 7)
    }
    updates = {
        revision: _update(harness, host, publication)
        for revision, publication in publications.items()
    }
    harness._presented_revisions_by_host[host.host_id] = (5, 6)

    # Revision 3 remains unresolved while newer successful fronts advance and
    # trim it out of both the worker and presented windows.
    harness.observe_surface_result(updates[4], _operation(initial, 4, (3, 4)))
    harness.observe_surface_result(updates[5], _operation(initial, 5, (4, 5)))
    harness.observe_surface_result(updates[6], _operation(initial, 6, (5, 6)))
    assert set(harness._publication_by_host_revision[host.host_id]) == {3, 5, 6}

    # A delayed success is no longer unresolved, and its stale front must not
    # replace the newest window or leave its publication retained forever.
    harness.observe_surface_result(updates[3], _operation(initial, 3, (2, 3)))
    assert harness._latest_host_revisions[host.host_id] == (5, 6)
    assert harness._publications_for(host, (5, 6)) == (
        publications[5],
        publications[6],
    )
    assert set(harness._publication_by_host_revision[host.host_id]) == {5, 6}
