"""The logic-node record sinks to the data layer; its row card moves to qt_widgets.

A logic node is what PRODUCES data on the console's Logic tab.  Its record was a
plain value with a codec, and its row card a Qt leaf that only displays the record
and emits what the operator asked for - the console keeps every start/stop decision.
Neither needed the legacy shell, and nothing in the target packages already owned
either concern (only ``FluentStatusDot``, the leaf the row uses, which had migrated
long ago).

``layout_record`` came along, and that is the part worth stating: it is the shared
validator behind THREE console records - panel, logic node and console state.  Moving
only the logic-node half would have split it from its other two callers, so it sinks
with the first record that moved and the shell imports it back.

The goldens below were captured from the legacy classes immediately before the move.
The row's is deliberately observed generically - every label's text and every button's
enabled state - rather than through attribute names, so it pins what the operator
actually sees.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).
DIES_WITH_PARTIAL = {
    "test_the_shell_reads_all_three_names_back_from_their_new_homes": 'Zou_lab_control/frontend/task_console.py',
}

import ast
import pathlib

import pytest
from PyQt5 import QtWidgets

from zlc_data.console_records import LOGIC_KINDS, LogicNodeConfig, layout_record
from zlc_frontend.qt_widgets import LogicNodeRow, ensure_qt_app

ROW_GOLDEN = {
    "initial": (["", "A #1", "(processor)", "stopped", ""],
                [("Start", True), ("Stop", False), ("Edit", True), ("Remove", True)]),
    "running": (["", "A #1", "(processor)", "running", ""],
                [("Start", False), ("Stop", True), ("Edit", True), ("Remove", True)]),
    "error": (["", "A #1", "(processor)", "error", ""],
              [("Start", True), ("Stop", False), ("Edit", True), ("Remove", True)]),
    "stopped": (["", "A #1", "(processor)", "stopped", ""],
                [("Start", True), ("Stop", False), ("Edit", True), ("Remove", True)]),
}


def _row():
    ensure_qt_app()
    return LogicNodeRow(LogicNodeConfig(kind="processor", name="Analysis", title="A #1"))


def _snapshot(row):
    return ([label.text() for label in row.findChildren(QtWidgets.QLabel)],
            [(b.text(), b.isEnabled()) for b in row.findChildren(QtWidgets.QPushButton)])


def test_a_fresh_row_shows_the_stopped_card():
    assert _snapshot(_row()) == ROW_GOLDEN["initial"]


@pytest.mark.parametrize("state", ["running", "error", "stopped"])
def test_each_state_drives_the_same_labels_and_buttons(state):
    """Start/Stop enablement is the operator-visible consequence of the state."""

    row = _row()
    row.set_state(state)
    assert _snapshot(row) == ROW_GOLDEN[state]


def test_an_unknown_state_is_shown_verbatim_and_left_startable():
    """No silent mapping to 'stopped' - an unexpected state must be visible as itself."""

    row = _row()
    row.set_state("nonsense")
    labels, buttons = _snapshot(row)
    assert labels[3] == "nonsense"
    assert buttons == ROW_GOLDEN["stopped"][1]


def test_publishes_renders_the_short_names_and_says_so_when_empty():
    row = _row()
    row.set_publishes(["a/b", "c/d"])
    assert _snapshot(row)[0][4] == "publishes:\n  a  /\n  c  /"
    row.set_publishes([])
    assert _snapshot(row)[0][4] == "publishes: (nothing on the hub)"


@pytest.mark.parametrize("payload, expected", [
    ({"kind": "camera", "name": "live"},
     {"kind": "camera", "name": "live", "title": "live", "values": {}}),
    ({"kind": "processor", "name": "Analysis", "title": "A #1", "values": {"action": "roi"}},
     {"kind": "processor", "name": "Analysis", "title": "A #1", "values": {"action": "roi"}}),
])
def test_the_record_round_trips_through_its_codec(payload, expected):
    """A missing title defaults to the name; the dict is the canonical form."""

    record = LogicNodeConfig(**payload)
    assert record.to_dict() == expected
    assert LogicNodeConfig.from_dict(expected).to_dict() == expected


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        LogicNodeConfig(kind="bogus", name="x")
    assert LOGIC_KINDS == ("camera", "measurement", "processor", "task")


def test_the_shared_validator_refuses_a_wrong_type_rather_than_coercing():
    """The reason layout_record moved WITH the record: three callers, one rule."""

    with pytest.raises(TypeError):
        layout_record({"kind": "camera", "name": "live", "title": "live", "values": "not-a-dict"},
                      {"kind": str, "name": str, "title": str, "values": dict},
                      "LogicNodeConfig", discriminator=None)


def test_the_record_module_reaches_for_no_toolkit_and_no_renderer():
    import zlc_data.console_records as record

    tree = ast.parse(pathlib.Path(record.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(m.split(".")[0] in {"matplotlib", "PyQt5"} for m in modules), modules


def test_the_shell_reads_all_three_names_back_from_their_new_homes():
    """Structural, so a shell that quietly kept a copy shows up in the import graph."""

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "Zou_lab_control" / "frontend" / "task_console.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    sources = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                sources.setdefault(alias.name, set()).add(node.module)
    assert sources.get("LogicNodeConfig") == {"zlc_data.console_records"}
    assert sources.get("layout_record") == {"zlc_data.console_records"}
    assert sources.get("LogicNodeRow") == {"zlc_frontend.qt_widgets"}
