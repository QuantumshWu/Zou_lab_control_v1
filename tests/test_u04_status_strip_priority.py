"""The console's status surface is one persistent strip with a priority ladder.

`main` mounts `FluentStatusStrip` once and feeds it every tick through a single
ladder - a wedged node's red error outranks even a running task's progress line,
the display-behind advisory is amber because the RUN is unaffected, and idle
leaves the strip empty *at its fixed height* so a message never shifts the board
under the pointer.  The migrated console had regressed to a bare word-wrapping
`FluentLabel`: no severity, no ranking, and a height that grew with the text.

`FluentStatusStrip` itself was already migrated (`display_editor.py` uses it), so
the missing piece was the ladder, which now lives beside the strip instead of
being retyped by every window that grows one.
"""

from __future__ import annotations

import pytest

import Zou_lab_control.notebook as zlc

from zlc_frontend.qt_widgets import FluentStatusStrip, arbitrate_status_line


def test_the_ladder_ranks_error_over_task_over_warning_over_notice():
    assert arbitrate_status_line(
        error="boom", task="scanning", warning="behind", notice="saved"
    ) == ("boom", "error")
    assert arbitrate_status_line(
        task="scanning", warning="behind", notice="saved"
    ) == ("scanning", "task")
    assert arbitrate_status_line(warning="behind", notice="saved") == (
        "behind",
        "warning",
    )
    assert arbitrate_status_line(notice="saved") == ("saved", "info")


def test_nothing_to_say_is_an_empty_info_line_not_a_hidden_strip():
    assert arbitrate_status_line() == ("", "info")


def test_the_ladder_refuses_a_non_string_input():
    with pytest.raises(TypeError, match="status inputs must be str"):
        arbitrate_status_line(task=object())


@pytest.fixture(scope="module")
def console(tmp_path_factory):
    from PyQt5 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    experiment = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("repository") / "workspace",
    )
    window = experiment.task_console()
    try:
        yield window
    finally:
        for card in window.cards:
            card._panel = None
        window.close()


def _strip(console) -> FluentStatusStrip:
    strip = console.findChild(FluentStatusStrip, "taskConsoleStatusStrip")
    assert strip is not None, "the console must mount the persistent strip once"
    return strip


def test_the_console_mounts_one_persistent_strip_at_a_fixed_height(console):
    strip = _strip(console)
    tall = "x" * 4000
    before = strip.height()
    console._notice(tall)

    assert strip.text() == tall
    assert strip.height() == before, "a long message must elide, never grow the strip"
    assert strip.severity == "info"


def test_an_error_outranks_a_notice_until_the_next_success(console):
    console._notice("")
    console._show_error(RuntimeError("device refused"))
    strip = _strip(console)
    assert strip.severity == "error"
    assert "RuntimeError: device refused" in strip.text()

    # A notice is only emitted by an action that completed, so it retires the
    # failure instead of being silently outranked by it forever.
    console._notice("Saved current intent: somewhere")
    assert strip.severity == "info"
    assert strip.text() == "Saved current intent: somewhere"


def test_a_running_card_outranks_a_notice_and_names_itself(console):
    from PyQt5 import QtCore, QtTest, QtWidgets

    class _Running:
        can_reconfigure = False
        closed = False
        final_reference = None

        def shutdown(self) -> None:
            self.closed = True

    add = console.findChild(QtWidgets.QPushButton, "addTaskPanelButton")
    QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
    card = console.cards[-1]
    card._panel = _Running()
    card._refresh_state()

    strip = _strip(console)
    assert strip.severity == "task"
    assert card.name in strip.text()

    # A plain notice must not hide the run that is still going.
    console._notice("Removed something else")
    assert strip.severity == "task"

    # An error still outranks the running task: a wedged panel cannot go quiet.
    console._show_error(ValueError("wedged"))
    assert strip.severity == "error"

    card._panel = None
    card._refresh_state()
    console._remove_card(card)
    assert strip.severity == "info"


def test_a_refusal_stays_visible_while_a_card_is_running(console):
    """Refusing a click must never be buried under the ambient task line."""

    from PyQt5 import QtCore, QtTest, QtWidgets

    class _Running:
        can_reconfigure = False
        closed = False
        final_reference = None

        def shutdown(self) -> None:
            self.closed = True

    add = console.findChild(QtWidgets.QPushButton, "addTaskPanelButton")
    QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
    card = console.cards[-1]
    card._panel = _Running()
    card._refresh_state()
    strip = _strip(console)
    assert strip.severity == "task"

    remove = card.findChild(QtWidgets.QPushButton, "taskCardRemoveButton")
    QtTest.QTest.mouseClick(remove, QtCore.Qt.LeftButton)

    # The card is still there AND the operator can see why.
    assert card in console.cards
    assert "must be stopped and idle before Remove" in strip.text()
    assert strip.severity == "error"

    card._panel = None
    card._refresh_state()
    console._remove_card(card)
