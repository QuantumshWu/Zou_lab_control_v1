"""Current TaskConsole render/edit ownership contracts.

These are deliberately narrow migration tests: one human Qt edit flow and one
mechanical ownership ratchet.  They do not preserve any legacy renderer API.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


ROOT = Path(__file__).resolve().parents[1]


def test_panel_title_is_a_local_draft_until_one_semantic_commit() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_data.console_records import PanelConfig
    from zlc_frontend.console_state import TaskConsoleState
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.window import TaskConsole

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
        console._timer.stop()
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


def test_timer_never_rediscovers_signal_topology_but_binding_commit_does() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_data.console_records import PanelConfig
    from zlc_frontend.console_state import TaskConsoleState
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.window import TaskConsole

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(
                PanelConfig(
                    kind="1d",
                    title="source",
                    signal="source_a",
                ),
            ),
        ),
        window_px=(900, 650),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        provider_reads = 0
        real_signal_providers = console._signal_providers

        def counted_signal_providers():
            nonlocal provider_reads
            provider_reads += 1
            return real_signal_providers()

        console._signal_providers = counted_signal_providers
        card.names_provider = lambda: ("source_a", "source_b")
        card.sources_provider = lambda: {
            "source_a": ("producer",),
            "source_b": ("producer",),
        }
        card.formats_provider = lambda: {}

        for _ in range(4):
            console._tick()
        application.processEvents()
        assert provider_reads == 0

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        application.processEvents()
        data_freeze = console._data.freeze

        def unexpected_data_freeze():
            raise AssertionError("a view/binding commit advanced the data plane")

        console._data.freeze = unexpected_data_freeze
        try:
            combo = card.signal_combo
            combo.showPopup()
            view = combo.view()
            model = combo.model()
            target = None
            for row in range(model.rowCount()):
                parent = model.index(row, 0)
                for child_row in range(model.rowCount(parent)):
                    child = model.index(child_row, 0, parent)
                    if child.data(QtCore.Qt.UserRole) == "source_b":
                        view.expand(parent)
                        target = child
                        break
                if target is not None:
                    break
            assert target is not None
            application.processEvents()
            QtTest.QTest.mouseClick(
                view.viewport(),
                QtCore.Qt.LeftButton,
                pos=view.visualRect(target).center(),
            )
            application.processEvents()
        finally:
            console._data.freeze = data_freeze
        assert card.config.signal == "source_b"
        assert provider_reads > 0
        reads_after_commit = provider_reads

        for _ in range(4):
            console._tick()
        application.processEvents()
        assert provider_reads == reads_after_commit
    finally:
        assert console.shutdown()
        console.close()
        application.processEvents()


def test_repeat_choice_commits_one_typed_view_spec_from_real_qt_input() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        DatasetSchema,
        PointLayout,
        ValidityContract,
        ValueSchema,
    )
    from zlc_data.console_records import PanelConfig
    from zlc_frontend.console_state import TaskConsoleState
    from zlc_frontend.figure import (
        AxisViewRole,
        RepeatViewMode,
        view_spec_from_tree,
    )
    from zlc_frontend.panel_render import PanelComposer
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.window import TaskConsole

    repeat = AxisSpec(
        AxisId("task-console.repeat"),
        "repeat",
        REPEAT,
        3,
        (0, 1, 2),
    )
    scan = AxisSpec(
        AxisId("task-console.scan"),
        "detuning",
        SCAN_POINT,
        4,
        (-1.0, 0.0, 1.0, 2.0),
        "MHz",
    )
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema((), ValidityContract.value(), np.dtype("float64"), "V"),
    )

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(PanelConfig(kind="1d", title="typed repeat"),),
        ),
        window_px=(900, 650),
    )
    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        card._last_value = SimpleNamespace(
            snapshot=SimpleNamespace(block=SimpleNamespace(schema=schema))
        )
        requests: list[bool] = []
        card._render_request = (
            lambda _card, *, force=False: requests.append(bool(force)) or True
        )

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        application.processEvents()
        card._refresh_repeat_mode_control()
        combo = card.repeat_mode_combo
        assert combo.isVisible()
        assert combo.findData(RepeatViewMode.FACET) == -1
        assert combo.findData(RepeatViewMode.BATCH) >= 0

        combo.setFocus()
        QtTest.QTest.keyClick(combo, QtCore.Qt.Key_End)
        application.processEvents()

        view = view_spec_from_tree(card.config.params["view_spec"])
        assert view.binding(repeat.axis_id).role is AxisViewRole.BATCH
        assert "repeat_mode" not in card.config.params
        assert requests == [False]
        composer = PanelComposer(
            "typed-repeat",
            intent=card.view_intent(),
            view=view,
        )
        try:
            assert composer.document_for(schema).layers[0].view == view
        finally:
            composer.close()
    finally:
        assert console.shutdown()
        console.close()
        application.processEvents()


def test_grid_repeat_facet_focus_and_overview_follow_real_qt_input() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        BlockId,
        CellValidity,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointLayout,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )
    from zlc_data.console_records import PanelConfig
    from zlc_frontend.console_state import TaskConsoleState
    from zlc_frontend.figure import AxisViewRole, view_spec_from_tree
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.task_console.data_plane import ConsoleSignalValue
    from zlc_workbench.task_console.window import TaskConsole

    repeat = AxisSpec(
        AxisId("task-console.grid.repeat"),
        "repeat",
        REPEAT,
        3,
        (0, 1, 2),
    )
    scan = AxisSpec(
        AxisId("task-console.grid.scan"),
        "detuning",
        SCAN_POINT,
        4,
        (-1.0, 0.0, 1.0, 2.0),
        "MHz",
    )
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema((), ValidityContract.value(), np.dtype("float64"), "V"),
    )
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    block = DataBlock(
        BlockId("task-console-grid-block"),
        DatasetRevision(1),
        values,
        CellValidity(np.ones(values.shape, dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("task-console-grid-generation")),
        block,
    )
    value = ConsoleSignalValue(
        "scan",
        "virtual scan",
        snapshot,
        1,
        None,
        "run",
        "epoch",
        "0" * 64,
    )

    application = ensure_qt_app()
    console = TaskConsole(
        state=TaskConsoleState(
            panels=(
                PanelConfig(kind="grid", title="repeat facets", signal="scan"),
            ),
        ),
        window_px=(900, 650),
    )

    def wait_until(predicate, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            time.sleep(0.005)
        assert predicate()

    try:
        console.show()
        application.processEvents()
        console._timer.stop()
        card = console.cards[0]
        # Producer injection stops at the immutable monitor boundary.  Every
        # product action below is real Qt input on the formal TaskConsole.
        assert card._freeze_value_render_request(value, 1, force=True) is None

        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        application.processEvents()
        facet = card.grid_facet_combo
        assert facet.count() == 1
        assert facet.itemData(0) == repeat.axis_id
        facet.setFocus()
        QtTest.QTest.keyClick(facet, QtCore.Qt.Key_Down)
        QtTest.QTest.keyClick(facet, QtCore.Qt.Key_Return)

        wait_until(
            lambda: card.board is not None
            and card.board.showing_overview
            and card.board.overview_png is not None
        )
        view = view_spec_from_tree(card.config.params["view_spec"])
        assert view.binding(repeat.axis_id).role is AxisViewRole.FACET
        board = card.board
        assert len(board._regions) == repeat.size

        region = board._regions[0]
        target = board._overview._target_rect()
        assert target is not None
        position = QtCore.QPoint(
            int(target.x() + 0.5 * (region.left + region.right) * target.width()),
            int(target.y() + 0.5 * (region.top + region.bottom) * target.height()),
        )
        QtTest.QTest.mouseDClick(
            board._overview,
            QtCore.Qt.LeftButton,
            pos=position,
        )
        wait_until(
            lambda: not board.showing_overview and board.front_frame is not None
        )
        QtTest.QTest.mouseClick(
            board._overview_button,
            QtCore.Qt.LeftButton,
        )
        wait_until(lambda: board.showing_overview)
    finally:
        assert console.shutdown()
        console.close()
        application.processEvents()


def test_only_the_task_console_worker_lane_may_compose() -> None:
    card = ROOT / "zlc_workbench" / "task_console" / "panel_card.py"
    console = ROOT / "zlc_workbench" / "task_console" / "window.py"
    editor = ROOT / "zlc_workbench" / "task_console" / "panel_editor.py"
    render_lane = ROOT / "zlc_workbench" / "task_console" / "render_lane.py"

    card_text = card.read_text(encoding="utf-8")
    editor_text = editor.read_text(encoding="utf-8")
    for path in (card, editor, console):
        owner_tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_names = {
            alias.name
            for node in ast.walk(owner_tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        compose_calls = {
            node.func.attr
            for node in ast.walk(owner_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        assert "PanelComposer" not in imported_names
        assert "compose" not in compose_calls
    assert "title_edit.textChanged" not in card_text
    assert "title_edit.textChanged" not in editor_text

    tree = ast.parse(console.read_text(encoding="utf-8"))
    tick = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_tick"
    )
    tick_calls = {
        node.func.attr
        for node in ast.walk(tick)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_refresh_signal_info" not in tick_calls
    assert "_signal_providers" not in tick_calls
    assert "_signal_info_sig" not in console.read_text(encoding="utf-8")
    view_request = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_request_card_render"
    )
    view_request_calls = {
        node.func.attr
        for node in ast.walk(view_request)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "freeze" not in view_request_calls
    assert "freeze_render_request" not in view_request_calls
    assert "freeze_current_view_request" in view_request_calls

    render_tree = ast.parse(render_lane.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(render_tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    compose_calls = [
        node
        for node in ast.walk(render_tree)
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
        assert owner.name == "_compose"
