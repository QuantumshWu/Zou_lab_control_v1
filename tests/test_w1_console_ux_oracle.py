"""W1's behaviour gate: the Task console's UX surface, frozen from the oracle.

Section 2.1 of the design document makes this the PREREQUISITE for the window
work, not a report on it: before target code may be touched, the implementer
reads ``main``'s implementation and freezes what the user can actually do.  Rule
9 makes ``main`` the UX authority, so the numbers below are not an opinion about
the console - they were measured by BUILDING it.

Provenance, so the next reader can redo it rather than trust it::

    git archive main | tar -x -C <tmp>        # main is 6c337d4, the declared base
    QT_QPA_PLATFORM=offscreen python -c "build TaskConsole, count the surface"

The same probe was then run against this branch's legacy console, and the two
agree on every dimension.  That is the finding that makes the salvage safe: the
branch's copy of ``frontend/task_console.py`` has grown 575 lines of fixes but
has NOT drifted from the oracle's user-visible surface, so salvaging the branch
copy is equivalent to salvaging ``main``'s.

This file is therefore both the D1 gate and the D2 harness.  Today it asserts the
LEGACY console matches the oracle.  When the salvaged window takes over the
entry point, the same constants are asserted against IT - and because they are
equalities rather than a frozen gap list, a window that quietly offers fewer
plot kinds cannot pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Measured from ``main`` (6c337d4) by constructing the window, then confirmed
#: identical on this branch.  Every entry is something the operator sees.
ORACLE_WIDGETS = 49
ORACLE_BUTTONS = frozenset({
    "Add Panel", "Devices", "Load", "Pause", "Save", "Save image",
    "Selectors", "Stop task", "⋯",
})
ORACLE_TABS = ("Monitor", "Logic")
ORACLE_ADD_PANEL_KINDS = frozenset({
    "Plot: 1D vector", "Plot: 2D image", "Plot: Distribution",
    "Plot: Rolling trace", "Plot: Site grid", "Plot: Site map",
})


@pytest.fixture(scope="module")
def legacy_console():
    from PyQt5 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from Zou_lab_control.frontend.task_console import (
        TaskConsole,
        default_console_state,
    )
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    window = TaskConsole(hub=SignalHub(), state=default_console_state())
    try:
        yield window
    finally:
        window.close()


def _buttons(window) -> frozenset[str]:
    from PyQt5 import QtWidgets

    return frozenset(
        button.text()
        for button in window.findChildren(QtWidgets.QAbstractButton)
        if button.text().strip()
    )


def _tabs(window) -> tuple[str, ...]:
    from PyQt5 import QtWidgets

    return tuple(
        tabs.tabText(index)
        for tabs in window.findChildren(QtWidgets.QTabWidget)
        for index in range(tabs.count())
    )


def _combo_items(window) -> frozenset[str]:
    from PyQt5 import QtWidgets

    return frozenset(
        combo.itemText(index)
        for combo in window.findChildren(QtWidgets.QComboBox)
        for index in range(combo.count())
    )


def test_the_header_offers_exactly_the_oracle_controls(legacy_console):
    assert _buttons(legacy_console) == ORACLE_BUTTONS, (
        "the console's header controls no longer match main's. Rule 9 makes "
        "main the UX authority; a control may not be dropped, renamed or "
        "invented without a row in the design document's deviation ledger."
    )


def test_the_permanent_two_tab_skeleton_is_present(legacy_console):
    assert _tabs(legacy_console) == ORACLE_TABS, (
        "Monitor and Logic are the console's permanent skeleton - the operator's "
        "daily workflow starts by switching between them."
    )


def test_add_panel_offers_every_plot_kind(legacy_console):
    offered = {item for item in _combo_items(legacy_console) if item.startswith("Plot: ")}
    assert offered == ORACLE_ADD_PANEL_KINDS, (
        "the Add Panel catalogue changed. Missing kinds are the difference "
        "between a console that can show an experiment and one that cannot: "
        f"missing {sorted(ORACLE_ADD_PANEL_KINDS - offered)}, "
        f"unexpected {sorted(offered - ORACLE_ADD_PANEL_KINDS)}."
    )


def test_the_window_is_not_quietly_shrinking(legacy_console):
    """A blunt scalar that catches wholesale loss the named checks would miss.

    Not an equality: widget counts move for legitimate reasons (a layout gains a
    spacer).  A floor catches the failure that matters - a window rebuilt with a
    fraction of the original's content, which is exactly what happened once.
    """

    from PyQt5 import QtWidgets

    count = len(legacy_console.findChildren(QtWidgets.QWidget))
    assert count >= ORACLE_WIDGETS, (
        f"the console builds {count} widgets against the oracle's "
        f"{ORACLE_WIDGETS}. A large drop means content was lost, not tidied."
    )
