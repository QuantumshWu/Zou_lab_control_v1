"""The third console record's FORMAT declaration joins the other two -- but its class stays.

``layout_record``'s own docstring names three console records: panel, logic node and
console state.  The first two sank in S5-shell(q) and (t).  The third could not follow,
and the reason is the design document rather than a preference:

  L303  ``zlc_data`` 只容纳领域中立、headless、可序列化的数据语义和值上的纯算法
  L320  zlc_data -> zlc_storage.canonical(只允许纯 canonical 模块,不允许 repository/I/O)
  L506  ``zlc_data`` base 只依赖 NumPy/必要 solver 与 ``zlc_storage.canonical``,不加载 repository I/O

``TaskConsoleState`` carries ``save(path)`` / ``load(path)``, which write and read a file.
The only JSON in ``zlc_data`` today (``plot_region``) serialises in-memory bytes and never
touches the filesystem, so moving the class whole would either cross that line or change a
public API (``state.save(...)``) that the GUI and several frozen tests call.  What DID move
is the part that is unambiguously the same concern as the other two: the field spec and the
persisted schema identifier.

That identifier is the trap this file exists for.  ``Zou_lab_control.frontend.TaskConsoleState``
reads like a module path, and it is not one: every layout a user has saved carries it, and
``exact_mapping`` refuses a payload whose ``schema`` differs.  Re-deriving it from the class's
new home would make every saved dashboard unopenable -- silently, at the moment an operator
tries to load their work.  ``test_the_schema_is_a_format_name_not_a_module_path`` states the
literal and the reason; ``test_a_payload_stamped_with_the_new_module_path_is_refused``
demonstrates the breakage directly rather than describing it.

The classes' own round-trip is covered by frozen tests (test_frontend_smoke.py,
test_panel_view_logic_split.py) outside ``tests/migration_active_tests.txt``, so the cases
below were captured from the shell immediately before the change.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

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


def test_the_shell_takes_both_from_the_new_home():
    """Structural: a shell keeping its own copy would pass every value assertion above."""

    tree = ast.parse((REPO / "Zou_lab_control" / "frontend" / "task_console.py")
                     .read_text(encoding="utf-8"))
    sources: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                sources.setdefault(alias.asname or alias.name, set()).add(node.module)
    assert sources.get("CONSOLE_STATE_SCHEMA") == {"zlc_data.console_records"}
    assert sources.get("_TASK_CONSOLE_STATE_FIELDS") == {"zlc_data.console_records"}


def test_the_record_still_stamps_and_accepts_that_exact_schema():
    from Zou_lab_control.frontend.task_console import TaskConsoleState

    assert TaskConsoleState.schema == CONSOLE_STATE_SCHEMA
    payload = TaskConsoleState(name="rb87").to_dict()
    assert payload["schema"] == CONSOLE_STATE_SCHEMA
    assert TaskConsoleState.from_dict(payload).to_dict() == payload


def test_a_payload_stamped_with_the_new_module_path_is_refused():
    """The breakage, demonstrated rather than described.

    This is exactly what a later 'the class lives in zlc_data now, fix the string' edit
    would produce for every layout already on disk."""

    from Zou_lab_control.frontend.task_console import TaskConsoleState

    payload = TaskConsoleState(name="rb87").to_dict()
    payload["schema"] = "zlc_data.console_records.TaskConsoleState"
    with pytest.raises(ValueError, match="expected schema"):
        TaskConsoleState.from_dict(payload)


@pytest.mark.parametrize("mutate, error", [
    (lambda d: d.pop("schema"), ValueError),          # exact key set, schema included
    (lambda d: d.update(extra=1), ValueError),
    (lambda d: d.update(interval_ms="x"), TypeError),
    (lambda d: d.update(interval_ms=10), ValueError),  # below the 50 ms floor
    (lambda d: d.update(panels=[{"kind": "zzz"}]), ValueError),  # nested record validates too
])
def test_every_captured_rejection_path_still_refuses(mutate, error):
    from Zou_lab_control.frontend.task_console import TaskConsoleState

    payload = TaskConsoleState(name="rb87").to_dict()
    mutate(payload)
    with pytest.raises(error):
        TaskConsoleState.from_dict(payload)


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
