"""Current TaskConsole layout value and atomic repository contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zlc_neutral_atom.catalog import DefinitionKey, definition_key_to_tree
from zlc_workbench.task_console.console_records import (
    CONSOLE_STATE_SCHEMA,
    LogicNodeConfig,
    PanelConfig,
)
from zlc_workbench.task_console.console_state import TaskConsoleState, default_console_state
from zlc_workbench.task_console.layout_repository import (
    TASK_FILES_ENV,
    load_task_console_state,
    resolve_task_state,
    save_task_console_state,
    task_files_dir,
)


@pytest.fixture
def layouts(tmp_path, monkeypatch):
    monkeypatch.setenv(TASK_FILES_ENV, str(tmp_path / "tasks"))
    return tmp_path


def _state() -> TaskConsoleState:
    return TaskConsoleState(
        name="rb87",
        interval_ms=200,
        panels=[PanelConfig(panel_id="panel-main", kind="1d", signal="cam/frame")],
        logic=[
            LogicNodeConfig(
                node_id="logic-analysis-1",
                kind="processor",
                definition_key=definition_key_to_tree(
                    DefinitionKey("zlc_neutral_atom", "test.analysis")
                ),
                title="Analysis #1",
            )
        ],
    )


def test_an_empty_layout_is_the_current_default() -> None:
    assert default_console_state().to_dict() == {
        "schema": CONSOLE_STATE_SCHEMA,
        "name": "task",
        "interval_ms": 400,
        "panels": [],
        "logic": [],
    }


def test_layout_publication_is_readable_and_round_trips(layouts) -> None:
    path = layouts / "rb87.json"
    state = _state()
    assert save_task_console_state(state, path) == path.resolve()
    text = path.read_text(encoding="utf-8")
    assert list(json.loads(text)) == ["schema", "name", "interval_ms", "panels", "logic"]
    assert text.endswith("\n")
    assert load_task_console_state(path).to_dict() == state.to_dict()
    assert not tuple(path.parent.glob(f".{path.name}.*.tmp"))


def test_a_name_resolves_under_the_explicit_layout_directory(layouts) -> None:
    save_task_console_state(_state(), task_files_dir() / "my_layout.json")
    elsewhere = layouts / "elsewhere.json"
    save_task_console_state(_state(), elsewhere)
    assert resolve_task_state("  my_layout  ").name == "rb87"
    assert resolve_task_state(str(elsewhere)).name == "rb87"


@pytest.mark.parametrize("task", ["nope", "missing.json"])
def test_an_unknown_layout_fails_closed(task, layouts) -> None:
    with pytest.raises(ValueError):
        resolve_task_state(task if task == "nope" else str(layouts / task))


@pytest.mark.parametrize(
    "write, error",
    [
        (None, FileNotFoundError),
        ("{not json", json.JSONDecodeError),
        ('{"schema": "other"}', ValueError),
    ],
)
def test_loading_refuses_missing_corrupt_or_foreign_files(write, error, layouts) -> None:
    path = layouts / "x.json"
    if write is not None:
        path.write_text(write, encoding="utf-8")
    with pytest.raises(error):
        load_task_console_state(path)


def test_duplicate_logic_identity_is_rejected_but_titles_are_presentation(layouts) -> None:
    duplicate = _state().logic[0]
    with pytest.raises(ValueError, match="duplicate logic node_id"):
        TaskConsoleState(logic=[duplicate, duplicate])

    first = _state().logic[0]
    second = LogicNodeConfig(
        node_id="logic-analysis-2",
        kind=first.kind,
        definition_key=first.definition_key,
        title=first.title,
    )
    state = TaskConsoleState(logic=[first, second])
    assert [node.title for node in state.logic] == ["Analysis #1", "Analysis #1"]
