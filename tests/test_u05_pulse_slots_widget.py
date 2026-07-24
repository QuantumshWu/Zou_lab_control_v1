"""Current PulseSlots ownership: deferred values plus stable keyed Qt rows.

``ParamWidgetContext`` carried a zero-arg factory whose only purpose was to build a widget
that happened to live in ``task_console``.  ``pulse_slots_factory`` is gone; the context now
holds no callback that reaches back into the legacy tree at all.

Seeding is deferred until a matching committed program arrives.  Reconcile is keyed by
``(program_id, slot_id)`` and never reconstructs the program editor.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).

import ast
import os
import pathlib
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from zlc_data.param_decl import ParamDecl
from zlc_data.scan_template import ScanColumnSpec
from zlc_frontend.domain_ports import PulseTemplateRows
from zlc_frontend.qt_widgets import MeasurementPanel, PulseSlotsWidget, ensure_qt_app

COLUMNS = (ScanColumnSpec(name="s0", lo=-512.0, hi=512.0, is_dac=True, unit=""),)
API_ROWS = (("h0", "c0", "duration", "t_probe", "ns", "100"),
            ("h1", "c1", "duration", "t_hold", "ns", "50"))
API_COLUMNS = (
    ScanColumnSpec(name="c0", lo=1, hi=200, unit="ns", quantum=1),
    ScanColumnSpec(name="c1", lo=1, hi=100, unit="ns", quantum=1),
)
SCAN_ROWS = (("c0", "dac", "s0", "", "s0"),)

TEMPLATE = {"api": {"h0": 100.0, "h1": 50.0}, "sweep_kind": "scan_slot", "program": "col"}


def _built(seed=None, *, api=API_ROWS, scan=SCAN_ROWS, reconciles=1):
    ensure_qt_app()
    widget = PulseSlotsWidget()
    if seed is not None:
        widget.seed_value(seed)
    for _ in range(reconciles):
        widget.reconcile(api, scan, api_columns=API_COLUMNS if api else (),
                         scan_columns=COLUMNS if scan else (),
                         hardware_program="col" if scan else "", program_id="tpl")
    return widget


def test_a_fresh_reconcile_shows_the_template_values():
    assert _built().values_dict() == {"program_id": "tpl", **TEMPLATE}


def test_a_seed_for_this_program_is_restored_on_the_next_reconcile():
    """The deferred contract: stored at seed time, applied at reconcile time."""

    widget = _built({"program_id": "tpl", "api": {"h0": 250.0, "h1": 7.0},
                     "sweep_kind": "api_slot", "program": "grid"})
    assert widget.values_dict() == {"program_id": "tpl", "api": {"h0": 250.0, "h1": 7.0},
                                    "sweep_kind": "api_slot", "program": "grid"}


def test_a_seed_for_a_different_program_is_ignored():
    """Loading a workspace saved against another template must not poison this one."""

    widget = _built({"program_id": "OTHER", "api": {"h0": 999.0}})
    assert widget.values_dict() == {"program_id": "tpl", **TEMPLATE}


def test_a_non_numeric_seed_drops_that_key_rather_than_writing_text():
    """The easy-to-break case: h0 disappears; the untouched h1 keeps its template value."""

    widget = _built({"program_id": "tpl", "api": {"h0": "abc"}})
    assert widget.values_dict() == {"program_id": "tpl", "api": {"h1": 50.0},
                                    "sweep_kind": "scan_slot", "program": "col"}


def test_reconciling_the_same_program_twice_changes_nothing():
    assert _built(reconciles=2).values_dict() == {"program_id": "tpl", **TEMPLATE}


def test_clean_program_source_updates_in_the_same_editor():
    widget = _built()
    editor = widget._program_code
    widget.reconcile(
        API_ROWS,
        SCAN_ROWS,
        api_columns=API_COLUMNS,
        scan_columns=COLUMNS,
        hardware_program="updated source",
        program_id="tpl",
    )
    assert widget._program_code is editor
    assert editor.toPlainText() == "updated source"


def test_a_template_with_no_slots_reads_back_empty():
    widget = _built(api=(), scan=())
    assert widget.values_dict() == {"program_id": "tpl", "api": {},
                                    "sweep_kind": "", "program": ""}


def test_real_code_and_api_input_survive_slot_add_remove_move_and_update():
    """Human input stays in the unaffected keyed editors during structural reconcile."""

    app = ensure_qt_app()
    widget = _built()
    widget.show()
    app.processEvents()

    h0 = widget._api_rows[("tpl", "h0")].edit
    code = widget._program_code
    scan_row = widget._column_rows[widget._scan_slot_kind][("tpl", "c0")]

    h0.setFocus()
    QtTest.QTest.keyClick(h0, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(h0, "123.75")
    code.setFocus()
    QtTest.QTest.keyClick(code, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(code, "operator_code_keeps_cursor")
    cursor = code.textCursor()
    cursor.setPosition(3)
    cursor.setPosition(11, cursor.KeepAnchor)
    code.setTextCursor(cursor)
    selection = (cursor.selectionStart(), cursor.selectionEnd(), cursor.selectedText())

    api = (
        ("h2", "c2", "delay", "probe", "us", 9),
        ("h0", "c0", "duration", "renamed_probe", "us", 777),
    )
    scan = (
        ("c1", "duration", "1", "ns", "hold"),
        SCAN_ROWS[0],
    )
    columns = (
        ScanColumnSpec(name="c1", lo=1, hi=10, unit="ns", quantum=1),
        COLUMNS[0],
    )
    widget.reconcile(
        api,
        scan,
        api_columns=(
            ScanColumnSpec(name="c2", lo=1, hi=10, unit="us", quantum=1),
            ScanColumnSpec(name="c0", lo=1, hi=10, unit="us", quantum=1),
        ),
        scan_columns=columns,
        hardware_program="replacement must not overwrite a local draft",
        program_id="tpl",
    )
    app.processEvents()

    assert widget._api_rows[("tpl", "h0")].edit is h0
    assert h0.text() == "123.75"
    assert widget._program_code is code
    assert code.toPlainText() == "operator_code_keeps_cursor"
    after = code.textCursor()
    assert (after.selectionStart(), after.selectionEnd(), after.selectedText()) == selection
    assert widget._column_rows[widget._scan_slot_kind][("tpl", "c0")] is scan_row
    assert widget._api_order == [("tpl", "h2"), ("tpl", "h0")]
    assert ("tpl", "h1") not in widget._api_rows


def test_path_typing_is_only_a_draft_and_commits_keep_slot_and_code_identity(monkeypatch):
    """Real key/mouse input covers typing, Return, Browse, and keyed slot movement."""

    calls: list[str] = []

    def read_template(path: str) -> PulseTemplateRows:
        calls.append(path)
        if len(calls) == 1:
            api_rows = API_ROWS
            api_columns = API_COLUMNS
            scan_rows = SCAN_ROWS
            scan_columns = COLUMNS
        elif len(calls) == 2:
            api_rows = (API_ROWS[1], ("h2", "c2", "delay", "probe", "us", 9), API_ROWS[0])
            api_columns = (
                ScanColumnSpec(name="c1", lo=1, hi=10, unit="ns", quantum=1),
                ScanColumnSpec(name="c2", lo=1, hi=10, unit="us", quantum=1),
                ScanColumnSpec(name="c0", lo=1, hi=10, unit="us", quantum=1),
            )
            scan_rows = (("c1", "duration", "1", "ns", "hold"), SCAN_ROWS[0])
            scan_columns = (
                ScanColumnSpec(name="c1", lo=1, hi=10, unit="ns", quantum=1),
                COLUMNS[0],
            )
        else:
            api_rows = (("h2", "c2", "delay", "probe", "us", 9), API_ROWS[0])
            api_columns = (
                ScanColumnSpec(name="c2", lo=1, hi=10, unit="us", quantum=1),
                ScanColumnSpec(name="c0", lo=1, hi=10, unit="ns", quantum=1),
            )
            scan_rows = (SCAN_ROWS[0],)
            scan_columns = COLUMNS
        return PulseTemplateRows(
            api_rows=api_rows,
            api_columns=api_columns,
            scan_rows=scan_rows,
            scan_columns=scan_columns,
            program="backend_program_must_not_replace_the_local_draft",
            program_id="stable-program",
        )

    import zlc_frontend.domain_ports as ports

    monkeypatch.setattr(ports, "pulse_template_rows", read_template)
    spec = SimpleNamespace(
        name="Pulse scan",
        params=(
            ParamDecl(key="template", label="Pulse template", kind="path", default="initial.json"),
            ParamDecl(
                key="slots",
                label="Pulse slots",
                kind="pulse_slots",
                depends_on="template",
            ),
        ),
    )
    app = ensure_qt_app()
    panel = MeasurementPanel((spec,))
    panel.show()
    app.processEvents()
    assert len(calls) == 1

    path = panel._widgets["template"]
    slots = panel._widgets["slots"]
    h0 = slots._api_rows[("stable-program", "h0")].edit
    code = slots._program_code
    scan_row = slots._column_rows[slots._scan_slot_kind][("stable-program", "c0")]

    h0.setFocus()
    QtTest.QTest.keyClick(h0, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(h0, "321.5")
    code.setFocus()
    QtTest.QTest.keyClick(code, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(code, "local_code_selection")
    cursor = code.textCursor()
    cursor.setPosition(2)
    cursor.setPosition(9, cursor.KeepAnchor)
    code.setTextCursor(cursor)
    selected = (cursor.selectionStart(), cursor.selectionEnd(), cursor.selectedText())

    path.edit.setFocus()
    QtTest.QTest.keyClick(path.edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(path.edit, "typed-template.json")
    app.processEvents()
    assert len(calls) == 1, "textChanged reached the template reader"
    assert slots._api_rows[("stable-program", "h0")].edit is h0
    assert slots._program_code is code

    QtTest.QTest.keyClick(path.edit, QtCore.Qt.Key_Return)
    app.processEvents()
    assert len(calls) == 2
    assert slots._api_rows[("stable-program", "h0")].edit is h0
    assert h0.text() == "321.5"
    assert slots._program_code is code
    after = code.textCursor()
    assert (after.selectionStart(), after.selectionEnd(), after.selectedText()) == selected
    assert slots._column_rows[slots._scan_slot_kind][("stable-program", "c0")] is scan_row

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: ("browse-template.json", ""),
    )
    QtTest.QTest.mouseClick(path.browse, QtCore.Qt.LeftButton)
    app.processEvents()
    assert len(calls) == 3
    assert slots._api_rows[("stable-program", "h0")].edit is h0
    assert slots._program_code is code
    panel.close()


def test_hot_path_structure_has_no_full_tree_rebuild_or_text_changed_parse():
    """Mechanical ratchet for the two C47 call chains fixed by this slice."""

    from zlc_frontend.qt_widgets import measurement_panel, pulse_slots_widget

    widget_tree = ast.parse(
        pathlib.Path(pulse_slots_widget.__file__).read_text(encoding="utf-8")
    )
    widget_class = next(
        node for node in widget_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PulseSlotsWidget"
    )
    method_names = {
        node.name for node in widget_class.body if isinstance(node, ast.FunctionDef)
    }
    assert not {"rebuild", "_drop_layout", "_render_program"}.intersection(method_names)
    assert not [
        node for node in ast.walk(widget_class)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rebuild"
    ]
    program_editor_assignments = [
        node for node in ast.walk(widget_class)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_program_code"
            for target in node.targets
        )
    ]
    assert len(program_editor_assignments) == 1

    panel_tree = ast.parse(
        pathlib.Path(measurement_panel.__file__).read_text(encoding="utf-8")
    )
    panel_class = next(
        node for node in panel_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MeasurementPanel"
    )
    methods = {
        node.name: node for node in panel_class.body if isinstance(node, ast.FunctionDef)
    }
    mark_dump = ast.dump(methods["_mark_pulse_path_dirty"])
    assert "pulse_template_rows" not in mark_dump
    assert "reconcile" not in mark_dump
    build = methods["_rebuild_form"]
    changed_wires = [
        node for node in ast.walk(build)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "changed"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "src"
    ]
    assert len(changed_wires) == 1
    callback = ast.dump(changed_wires[0].args[0])
    assert "_mark_pulse_path_dirty" in callback
    assert "_commit_pulse_slots_path" not in callback


def test_the_context_carries_no_factory_into_the_legacy_tree_any_more():
    """Both fields retired, not merely unused - this is the point of the two slices."""

    from zlc_frontend.qt_widgets.param_widgets import ParamWidgetContext

    fields = set(ParamWidgetContext.__dataclass_fields__)
    assert not [name for name in fields if name.endswith("_factory")], fields


def test_pulse_slots_handler_builds_its_own_widget():
    """Structural: a handler still calling a factory would pass every behaviour test."""

    from zlc_frontend.qt_widgets import param_widgets

    tree = ast.parse(pathlib.Path(param_widgets.__file__).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "PulseSlotsHandler")
    build = next(n for n in node.body
                 if isinstance(n, ast.FunctionDef) and n.name == "build")
    constructed = {c.func.id for c in ast.walk(build)
                   if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "PulseSlotsWidget" in constructed
    assert "_factory" not in ast.dump(build)


def test_the_widget_module_pulls_in_no_renderer():
    from zlc_frontend.qt_widgets import pulse_slots_widget

    tree = ast.parse(pathlib.Path(pulse_slots_widget.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(m.split(".")[0] == "matplotlib" for m in modules), modules


