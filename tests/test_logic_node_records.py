"""Current TaskConsole logic-row record and Qt presentation contract."""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_neutral_atom.catalog import DefinitionKey, definition_key_to_tree
from zlc_workbench.task_console.console_records import (
    LogicNodeConfig,
)
from zlc_workbench.task_console.logic_node_row import LogicNodeRow
from zlc_frontend.qt_widgets import ensure_qt_app


DEFINITION_TREE = definition_key_to_tree(
    DefinitionKey("zlc_neutral_atom", "test.analysis")
)


def _record(**changes) -> LogicNodeConfig:
    values = {
        "node_id": "logic-test-analysis",
        "definition_key": DEFINITION_TREE,
        "title": "Analysis #1",
        "authored": {},
        "inputs": {},
    }
    values.update(changes)
    return LogicNodeConfig(**values)


def _row() -> LogicNodeRow:
    ensure_qt_app()
    return LogicNodeRow(_record(), "processor")


def test_row_buttons_follow_run_state() -> None:
    row = _row()
    assert row.start_button.isEnabled()
    assert not row.stop_button.isEnabled()
    row.set_state("running")
    assert not row.start_button.isEnabled()
    assert row.stop_button.isEnabled()
    row.set_state("error", status="failed")
    assert row.start_button.isEnabled()
    assert not row.stop_button.isEnabled()
    assert row.status_label.text() == "failed"


def test_publishes_uses_declared_name_shape_and_description() -> None:
    row = _row()
    row.set_publishes(
        [
            ("occupied", "1 × 1 × (35)", "per-site occupancy"),
            ("rate", "1 × 1 × (1)", "loading rate"),
        ]
    )
    assert row.publishes_label.text() == (
        "publishes:\n"
        "  occupied  1 × 1 × (35)\n"
        "  rate      1 × 1 × (1)"
    )
    assert "per-site occupancy" in row.publishes_label.toolTip()
    row.set_publishes([])
    assert row.publishes_label.text() == "publishes: (no declared outputs)"


def test_record_round_trip_preserves_three_distinct_identities() -> None:
    record = _record(
        authored={"action": "judge"},
        inputs={"source": "@logic/camera/frame"},
    )
    payload = record.to_dict()
    assert payload == {
        "node_id": "logic-test-analysis",
        "definition_key": DEFINITION_TREE,
        "title": "Analysis #1",
        "authored": {"action": "judge"},
        "inputs": {"source": "@logic/camera/frame"},
    }
    assert LogicNodeConfig.from_dict(payload).to_dict() == payload


def test_kind_is_not_mirrored_and_noncanonical_record_is_rejected() -> None:
    assert "kind" not in _record().to_dict()
    payload = _record().to_dict()
    payload["title"] = " Analysis #1 "
    with pytest.raises(ValueError, match="canonical"):
        LogicNodeConfig.from_dict(payload)


def test_record_owner_imports_no_qt_or_renderer() -> None:
    import zlc_workbench.task_console.console_records as records

    tree = ast.parse(pathlib.Path(records.__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(
        module.split(".")[0] in {"matplotlib", "PyQt5"}
        for module in modules
    )
