"""Current TaskConsole render/edit ownership contracts.

These are deliberately narrow migration tests: one human Qt edit flow and one
mechanical ownership ratchet.  They do not preserve any legacy renderer API.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


ROOT = Path(__file__).resolve().parents[1]


def test_panel_title_is_a_local_draft_until_one_semantic_commit() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_data.console_records import PanelConfig
    from zlc_frontend.console_state import TaskConsoleState
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.plot_bridge_console import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="1d", title="before"),),
        ),
        window_px=(900, 650),
    )
    try:
        console.show()
        application.processEvents()
        card = console.cards[0]
        requests: list[bool] = []
        card._render_request = (
            lambda _card, *, force=False: requests.append(bool(force)) or True
        )

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        application.processEvents()
        assert card.title_edit.isVisible()

        card.title_edit.setFocus()
        QtTest.QTest.keyClick(
            card.title_edit,
            QtCore.Qt.Key_A,
            QtCore.Qt.ControlModifier,
        )
        QtTest.QTest.keyClicks(card.title_edit, "one semantic title")
        application.processEvents()

        assert card.config.title == "before"
        assert requests == []

        QtTest.QTest.keyClick(card.title_edit, QtCore.Qt.Key_Return)
        application.processEvents()

        assert card.config.title == "one semantic title"
        assert requests == [False]
    finally:
        assert console.shutdown()
        console.close()
        application.processEvents()


def test_only_the_task_console_worker_lane_may_compose() -> None:
    bridge = ROOT / "zlc_workbench" / "task_console" / "plot_bridge.py"
    console = ROOT / "zlc_workbench" / "task_console" / "plot_bridge_console.py"
    editor = ROOT / "zlc_workbench" / "task_console" / "plot_bridge_editor.py"

    bridge_text = bridge.read_text(encoding="utf-8")
    editor_text = editor.read_text(encoding="utf-8")
    assert "_rerender_now" not in bridge_text
    assert "PanelComposer" not in editor_text
    assert ".compose(" not in editor_text
    assert "title_edit.textChanged" not in bridge_text
    assert "title_edit.textChanged" not in editor_text

    tree = ast.parse(console.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    compose_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compose"
    ]
    assert compose_calls, "worker render lane must compose accepted requests"
    for call in compose_calls:
        owner = parents.get(call)
        while owner is not None and not isinstance(
            owner, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            owner = parents.get(owner)
        assert owner is not None
        assert owner.name == "_compose_render_requests"
