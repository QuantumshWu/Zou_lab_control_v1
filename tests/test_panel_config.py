"""Current TaskConsole panel-record contract.

A panel binds one typed dataset.  Multi-producer calculations belong to an
explicit Processor/join and cannot be smuggled into a saved GUI expression.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from zlc_data import (
    MONITOR_HISTORY,
    REPEAT,
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
    VALID,
    ValueSchema,
)
from zlc_frontend.plot_kind import PlotKind
from zlc_frontend.plot_panel import plot_panel_input
from zlc_workbench.task_console.console_records import (
    DEFAULT_UPDATE_MS,
    PANEL_KINDS,
    UPDATE_INTERVALS,
    PanelConfig,
)


KINDS = (
    PlotKind.IMAGE,
    PlotKind.SITE_MAP,
    PlotKind.CURVE,
    PlotKind.ROLLING,
    PlotKind.HISTOGRAM,
    PlotKind.GRID,
)
DEFAULT = {
    "panel_id": "panel-test",
    "title": "",
    "row": 0,
    "col": 0,
    "size": "2x2",
    "signal": "",
    "params": {},
}


def _scalar_snapshot(*, rolling: bool) -> OwnedSnapshot:
    repeat = AxisSpec(AxisId("test.repeat"), "repeat", REPEAT, 1, (0,))
    history = (
        AxisSpec(
            AxisId("test.monitor-history"),
            "history",
            MONITOR_HISTORY,
            2,
            (0, 1),
        ),
    ) if rolling else ()
    schema = DatasetSchema(
        repeat,
        (
            PointTable(
                history[0].size,
                (
                    PointColumn(
                        history[0].axis_id,
                        history[0].name,
                        history[0].role,
                        PointColumn.NUMERIC,
                        history[0].coordinates,
                        history[0].unit,
                        history[0].coordinate_frame,
                    ),
                ),
            )
            if history
            else PointTable(1)
        ),
        None,
        ValueSchema.scalar(np.dtype("<f8")),
    )
    block = DataBlock(
        BlockId("test-rolling" if rolling else "test-scalar"),
        DatasetRevision(0),
        np.zeros(schema.physical_shape, dtype="<f8"),
        VALID,
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("test-generation")), block)


def test_a_fresh_panel_is_unbound_for_every_kind() -> None:
    for kind in KINDS:
        assert PanelConfig(panel_id="panel-test", kind=kind).to_dict() == {
            "kind": kind.value,
            **DEFAULT,
        }


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"kind": PlotKind.CURVE, "signal": "cam/frame"}, {"signal": "cam/frame"}),
        (
            {
                "kind": PlotKind.CURVE,
                "title": "T",
                "row": 7,
                "col": 9,
                "size": "1x2",
            },
            {"title": "T", "row": 7, "col": 9, "size": "1x2"},
        ),
    ],
)
def test_construction_keeps_the_declared_single_binding(kwargs, expected) -> None:
    record = PanelConfig(**kwargs).to_dict()
    for key, value in expected.items():
        assert record[key] == value


def test_refresh_interval_is_exact_and_invalid_values_fail_closed() -> None:
    assert PanelConfig(kind=PlotKind.CURVE).update_ms == DEFAULT_UPDATE_MS
    assert PanelConfig(kind=PlotKind.CURVE, params={"update_ms": 200}).update_ms == 200
    for invalid in (999, 0, True, "200"):
        with pytest.raises(ValueError):
            PanelConfig(kind=PlotKind.CURVE, params={"update_ms": invalid}).update_ms


def test_the_record_round_trips_through_its_codec() -> None:
    payload = PanelConfig(kind=PlotKind.CURVE, signal="cam/frame").to_dict()
    assert PanelConfig.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda data: data.pop("size"), ValueError),
        (lambda data: data.update(row="x"), TypeError),
        (lambda data: data.update(row=-1), ValueError),
        (lambda data: data.update(col=-1), ValueError),
        (lambda data: data.update(signal=1), TypeError),
        (lambda data: data.update(kind="zzz"), ValueError),
        # The obsolete expression schema is rejected, not converted.
        (lambda data: data.update(source="value = signal"), ValueError),
    ],
)
def test_invalid_or_obsolete_records_fail_closed(mutate, error) -> None:
    payload = PanelConfig(kind=PlotKind.CURVE, signal="cam/frame").to_dict()
    mutate(payload)
    with pytest.raises(error):
        PanelConfig.from_dict(payload)


def test_negative_positions_are_never_silently_rewritten() -> None:
    with pytest.raises(ValueError):
        PanelConfig(kind=PlotKind.CURVE, row=-5)


def test_only_end_to_end_live_renderers_are_addable() -> None:
    assert tuple(PANEL_KINDS) == KINDS
    for kind in (PlotKind.METER, PlotKind.PULSE):
        with pytest.raises(ValueError):
            PanelConfig(kind=kind)


def test_meter_and_monitor_are_distinct_frontend_contracts() -> None:
    scalar = _scalar_snapshot(rolling=False)
    history = _scalar_snapshot(rolling=True)

    assert plot_panel_input(PlotKind.METER, scalar).snapshot is scalar
    assert plot_panel_input(PlotKind.ROLLING, history).snapshot is history
    with pytest.raises(ValueError, match="use Meter for a scalar dataset"):
        plot_panel_input(PlotKind.ROLLING, scalar)


def test_the_record_module_reaches_for_no_toolkit_and_no_renderer() -> None:
    import zlc_workbench.task_console.console_records as records

    tree = ast.parse(pathlib.Path(records.__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(
        name.split(".")[0] in {"matplotlib", "PyQt5", "PySide2", "PySide6"}
        for name in modules
    )
