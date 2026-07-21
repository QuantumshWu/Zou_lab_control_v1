"""The saved-console FILE FORMAT lives in ``zlc_data.console_records`` -- and only there.

The persisted schema identifier ``Zou_lab_control.frontend.TaskConsoleState`` reads like a
module path, and it is not one: it is a format NAME that every layout a user has saved
carries, and ``exact_mapping`` refuses a payload whose ``schema`` differs.  The module it
was once named after is deleted (directive 2026-07-21); re-deriving the identifier from
any code location would make every saved dashboard unopenable -- silently, at the moment
an operator tries to load their work.  Hence the literal below.
"""

from __future__ import annotations

import ast
import pathlib

from zlc_data.console_records import (
    CONSOLE_STATE_SCHEMA,
    LOGIC_NODE_CONFIG_FIELDS,
    PANEL_CONFIG_FIELDS,
    TASK_CONSOLE_STATE_FIELDS,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_schema_is_a_format_name_not_a_module_path():
    """Stated as a literal, so moving the class cannot quietly move the file format."""

    assert CONSOLE_STATE_SCHEMA == "Zou_lab_control.frontend.TaskConsoleState"


def test_all_three_console_record_field_specs_live_together():
    """The point of the module: one place answers 'what is a saved console made of'."""

    assert TASK_CONSOLE_STATE_FIELDS == {
        "schema": str, "name": str, "interval_ms": int, "panels": list, "logic": list}
    assert set(PANEL_CONFIG_FIELDS) == {
        "kind", "title", "row", "col", "size", "source", "params", "inputs"}
    assert set(LOGIC_NODE_CONFIG_FIELDS) == {"kind", "name", "title", "values"}
    # ``schema`` is a FIELD here, not merely a check: it round-trips into the file, and the
    # exact-key rule means a payload missing it is refused rather than defaulted.
    assert "schema" in TASK_CONSOLE_STATE_FIELDS
    assert "schema" not in PANEL_CONFIG_FIELDS and "schema" not in LOGIC_NODE_CONFIG_FIELDS


def test_the_record_module_still_touches_no_filesystem():
    """The line this slice declined to cross, asserted rather than left as prose.

    ``zlc_data`` serialises; it does not open files.  If a later slice moves save/load in
    here, this is the assertion it has to argue with."""

    text = (REPO / "zlc_data" / "console_records.py").read_text(encoding="utf-8")
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
