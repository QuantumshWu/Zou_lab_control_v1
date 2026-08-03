"""The current saved-console format belongs to the TaskConsole product."""

from __future__ import annotations

import ast
import pathlib

from zlc_workbench.task_console.console_records import (
    CONSOLE_STATE_SCHEMA,
    LOGIC_NODE_CONFIG_FIELDS,
    PANEL_CONFIG_FIELDS,
    TASK_CONSOLE_STATE_FIELDS,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_schema_is_a_semantic_format_name_not_a_module_path():
    assert CONSOLE_STATE_SCHEMA == "zlc.task_console.layout"


def test_all_three_console_record_field_specs_live_together():
    """The point of the module: one place answers 'what is a saved console made of'."""

    assert TASK_CONSOLE_STATE_FIELDS == {
        "schema": str, "name": str, "interval_ms": int, "panels": list, "logic": list}
    assert set(PANEL_CONFIG_FIELDS) == {
        "panel_id", "kind", "title", "row", "col", "size", "signal", "params"}
    assert set(LOGIC_NODE_CONFIG_FIELDS) == {
        "node_id", "definition_key", "title", "authored", "inputs"}
    # ``schema`` is a FIELD here, not merely a check: it round-trips into the file, and the
    # exact-key rule means a payload missing it is refused rather than defaulted.
    assert "schema" in TASK_CONSOLE_STATE_FIELDS
    assert "schema" not in PANEL_CONFIG_FIELDS and "schema" not in LOGIC_NODE_CONFIG_FIELDS


def test_the_record_module_touches_no_filesystem():
    """Records remain renderer-free values; the workbench repository owns I/O."""

    text = (REPO / "zlc_workbench" / "task_console" / "console_records.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(text)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not {m for m in modules if m.split(".")[0] in {"pathlib", "os", "io", "shutil"}}
    for reader_writer in ("open(", "read_text", "write_text", "read_bytes", "write_bytes"):
        assert reader_writer not in text, reader_writer
