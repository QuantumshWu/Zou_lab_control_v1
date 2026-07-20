"""The TaskConsole is a board of N panels, not a single-card holder.

`main`'s console is a multi-panel board: Add Panel adds another panel every
time, each panel carries its own Remove, and closing the window waits for every
embedded panel.  The migrated console had hard-coded exactly one card - a second
Add Panel raised `RuntimeError("TaskConsole currently owns exactly one card")`
and the Add button disabled itself after the first card - which is a behaviour
deletion, not a design decision.  These tests pin the board semantics from the
outside: how many cards exist, which one Analysis targets, and what Remove and
close do.

The refusal and close-wait cases install a stand-in panel object on a card.  The
console never inspects a panel beyond the `idle`/`closed` questions asked here,
so driving a real acquisition would only make the test slow, not stronger.
"""

from __future__ import annotations

import pytest

import Zou_lab_control.notebook as zlc


class _StubPanel:
    """The minimum a card's panel must answer for the console's own rules."""

    def __init__(self, *, can_reconfigure: bool = True, closed: bool = True) -> None:
        self.can_reconfigure = can_reconfigure
        self.closed = closed
        self.final_reference = None
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.closed = True


def _add_panel(window, QtCore, QtTest, QtWidgets) -> None:
    button = window.findChild(QtWidgets.QPushButton, "addTaskPanelButton")
    assert button is not None
    QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    from PyQt5 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # One installation runtime per process, so every console here shares it.
    return zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("repository") / "workspace",
    )


@pytest.fixture
def console(experiment):
    window = experiment.task_console()
    try:
        yield window
    finally:
        for card in window.cards:
            card._panel = None
        window.close()


def test_add_panel_keeps_adding_panels(console):
    from PyQt5 import QtCore, QtTest, QtWidgets

    _add_panel(console, QtCore, QtTest, QtWidgets)
    _add_panel(console, QtCore, QtTest, QtWidgets)
    _add_panel(console, QtCore, QtTest, QtWidgets)

    assert len(console.cards) == 3
    # Every panel is separately identifiable, the way `main` names panels.
    assert [card.name for card in console.cards] == [
        "Pulse scan #1",
        "Pulse scan #2",
        "Pulse scan #3",
    ]
    add = console.findChild(QtWidgets.QPushButton, "addTaskPanelButton")
    catalog = console.findChild(QtWidgets.QComboBox, "taskCatalogCombo")
    assert add.isEnabled() and catalog.isEnabled()
    # `scan_card` stays the entry point for Save/Load: the newest panel.
    assert console.scan_card is console.cards[-1]


def test_every_card_carries_its_own_remove_button(console):
    from PyQt5 import QtCore, QtTest, QtWidgets

    _add_panel(console, QtCore, QtTest, QtWidgets)
    _add_panel(console, QtCore, QtTest, QtWidgets)
    first, second = console.cards

    remove = first.findChild(QtWidgets.QPushButton, "taskCardRemoveButton")
    assert remove is not None
    QtTest.QTest.mouseClick(remove, QtCore.Qt.LeftButton)

    assert console.cards == (second,)
    assert second.name == "Pulse scan #2"


def test_removing_the_last_card_restores_the_empty_state(console):
    from PyQt5 import QtCore, QtTest, QtWidgets

    _add_panel(console, QtCore, QtTest, QtWidgets)
    empty = console.findChild(QtWidgets.QLabel, "taskConsoleEmptyState")
    assert empty is not None and not empty.isVisible()

    console._remove_card(console.cards[0])

    assert console.cards == ()
    assert empty.isVisible()
    # A fresh Add Panel still works after the board has been emptied.
    _add_panel(console, QtCore, QtTest, QtWidgets)
    assert len(console.cards) == 1
    assert not empty.isVisible()


def test_remove_refuses_a_running_card_instead_of_killing_it(console):
    from PyQt5 import QtCore, QtTest, QtWidgets

    _add_panel(console, QtCore, QtTest, QtWidgets)
    card = console.cards[0]
    panel = _StubPanel(can_reconfigure=False, closed=False)
    card._panel = panel

    remove = card.findChild(QtWidgets.QPushButton, "taskCardRemoveButton")
    QtTest.QTest.mouseClick(remove, QtCore.Qt.LeftButton)

    assert console.cards == (card,)
    assert panel.shutdown_calls == 0
    status = console.findChild(QtWidgets.QLabel, "taskConsoleDiagnostics")
    assert "must be stopped and idle before Remove" in status.text()

    # Once the panel goes idle the very same click removes it.
    panel.can_reconfigure = True
    QtTest.QTest.mouseClick(remove, QtCore.Qt.LeftButton)
    assert console.cards == ()
    assert panel.shutdown_calls == 1


def test_a_foreign_card_cannot_be_removed_through_this_console(console):
    from PyQt5 import QtCore, QtTest, QtWidgets
    from Zou_lab_control.workbench._task_console import TaskScanCard

    _add_panel(console, QtCore, QtTest, QtWidgets)
    foreign = TaskScanCard(console._experiment, None, None)
    try:
        with pytest.raises(ValueError, match="does not belong to this TaskConsole"):
            console._remove_card(foreign)
    finally:
        foreign.shutdown()
        foreign.deleteLater()


def test_analysis_targets_the_newest_card_that_owns_a_final_artifact(console):
    from PyQt5 import QtCore, QtTest, QtWidgets

    _add_panel(console, QtCore, QtTest, QtWidgets)
    _add_panel(console, QtCore, QtTest, QtWidgets)
    first, second = console.cards
    add_analysis = console.findChild(
        QtWidgets.QPushButton,
        "addTaskAnalysisButton",
    )
    assert not add_analysis.isEnabled()

    first._panel = _StubPanel()
    first._panel.final_reference = "artifact-A"
    console._sync_analysis_entry()
    assert add_analysis.isEnabled()
    assert console._analysis_card() is first

    second._panel = _StubPanel()
    second._panel.final_reference = "artifact-B"
    console._sync_analysis_entry()
    assert console._analysis_card() is second

    second._panel.final_reference = None
    console._sync_analysis_entry()
    assert console._analysis_card() is first


def test_close_waits_for_every_embedded_panel(console):
    from PyQt5 import QtCore, QtTest, QtWidgets

    _add_panel(console, QtCore, QtTest, QtWidgets)
    _add_panel(console, QtCore, QtTest, QtWidgets)
    first, second = console.cards
    first._panel = _StubPanel(closed=True)
    lagging = _StubPanel(can_reconfigure=False, closed=False)
    second._panel = lagging
    # A shutdown that does not immediately close, the way a running panel drains.
    lagging.shutdown = lambda: setattr(lagging, "shutdown_calls", 1)

    console.close()

    assert console.isVisible()
    assert console._close_timer.isActive()
    assert lagging.shutdown_calls == 1

    lagging.closed = True
    console._poll_close()
    assert not console._close_timer.isActive()
