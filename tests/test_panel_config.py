"""Current TaskConsole panel-record contract.

A panel binds one typed dataset.  Multi-producer calculations belong to an
explicit Processor/join and cannot be smuggled into a saved GUI expression.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_plot import PlotKind, RollingPlot
from zlc_workbench.task_console.console_records import (
    DEFAULT_UPDATE_MS,
    PANEL_KINDS,
    UPDATE_INTERVALS,
    PanelConfig,
)
from zlc_workbench.task_console.console_state import TaskConsoleState


KINDS = (
    PlotKind.IMAGE,
    PlotKind.CURVE,
    PlotKind.ROLLING,
    PlotKind.HISTOGRAM,
    PlotKind.FACET_GRID,
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


def test_a_fresh_panel_is_unbound_for_every_kind() -> None:
    for kind in KINDS:
        assert PanelConfig(panel_id="panel-test", plot=kind).to_dict() == {
            "kind": kind.value,
            **DEFAULT,
        }


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"plot": PlotKind.CURVE, "signal": "cam/frame"}, {"signal": "cam/frame"}),
        (
            {
                "plot": PlotKind.CURVE,
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
    assert PanelConfig(plot=PlotKind.CURVE).update_ms == DEFAULT_UPDATE_MS
    assert PanelConfig(plot=PlotKind.CURVE, params={"update_ms": 200}).update_ms == 200
    for invalid in (999, 0, True, "200"):
        with pytest.raises(ValueError):
            PanelConfig(plot=PlotKind.CURVE, params={"update_ms": invalid}).update_ms


def test_the_record_round_trips_through_its_codec() -> None:
    payload = PanelConfig(plot=PlotKind.CURVE, signal="cam/frame").to_dict()
    assert PanelConfig.from_dict(payload).to_dict() == payload


def test_exact_plot_spec_is_typed_in_memory_and_encoded_only_at_io() -> None:
    config = PanelConfig(plot=RollingPlot(), signal="cam/frame")
    assert config.plot == RollingPlot()
    assert "plot_spec" not in config.params
    payload = config.to_dict()
    assert "plot_spec" in payload["params"]
    restored = PanelConfig.from_dict(payload)
    assert restored.plot == config.plot
    assert "plot_spec" not in restored.params


def test_console_runtime_copies_typed_plot_without_a_layout_codec_round_trip(
    monkeypatch,
) -> None:
    config = PanelConfig(plot=RollingPlot(), signal="cam/frame")

    def fail_codec(*_args, **_kwargs):
        raise AssertionError("layout codec entered the in-memory console path")

    monkeypatch.setattr(PanelConfig, "to_dict", fail_codec)
    monkeypatch.setattr(PanelConfig, "from_dict", fail_codec)
    state = TaskConsoleState(panels=(config,))

    assert state.panels[0] is not config
    assert state.panels[0].plot == RollingPlot()
    assert "plot_spec" not in state.panels[0].params


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
    payload = PanelConfig(plot=PlotKind.CURVE, signal="cam/frame").to_dict()
    mutate(payload)
    with pytest.raises(error):
        PanelConfig.from_dict(payload)


def test_negative_positions_are_never_silently_rewritten() -> None:
    with pytest.raises(ValueError):
        PanelConfig(plot=PlotKind.CURVE, row=-5)


def test_only_end_to_end_live_renderers_are_addable() -> None:
    assert tuple(PANEL_KINDS) == KINDS
    with pytest.raises(ValueError):
        PanelConfig(plot=PlotKind.PULSE_TIMELINE)


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
