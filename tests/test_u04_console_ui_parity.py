"""The migrated TaskConsole is compared against the OLD one, control by control.

Rule 9 makes `main`'s implementation the UX authority, and S2 promises the daily
workflow stays familiar.  Nothing enforced it.  Every slice so far ran the whole
active manifest green while the window the operator actually opens drifted until
it shared almost nothing with the old one:

    old TaskConsole              new TaskConsoleWindow
    49 widgets                   20 widgets
    9 header buttons             4 header buttons
    Monitor / Logic tabs         no tabs at all
    6 addable plot kinds         1 catalog entry ("Task: Pulse scan")

Test-green is not acceptance.  This file is the missing gate: it builds BOTH
consoles in one process and diffs their control inventories.  The legacy tree is
still present on this branch, so the comparison is against the real old window,
not a description of it.

It is a two-way ratchet, not a snapshot:

  * every control the new console is missing must be listed below AND carry a
    row in the design doc's UX deviation ledger.  Dropping one more control
    fails immediately.
  * restoring a control also fails, until its entry is removed from the lists
    here and from the ledger.  Parity work cannot be done silently either.

The lists shrink to empty as checklist 4 lands.  When they are empty this test
becomes a plain equality assertion between the two windows.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt5 import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs" / "SYSTEM_ARCHITECTURE_DESIGN_zh.md"

#: Header controls the OLD console has and the new one does not.  Each name is
#: also a row under UX-014 in the design doc.
MISSING_CONTROLS_PENDING_APPROVAL = frozenset(
    {"Selectors", "Devices", "Pause", "Save image", "Stop task", "⋯"}
)

#: Controls the new console grew that the old one never had.  An addition is a
#: deviation too - it is a button the operator has to learn.
ADDED_CONTROLS_PENDING_APPROVAL = frozenset({"Add Analysis → Fit"})

#: The old console's permanent two-tab skeleton, absent entirely today.
MISSING_TABS_PENDING_APPROVAL = ("Monitor", "Logic")

#: Plot kinds the old Add Panel dropdown offers.  The new dropdown offers none
#: of them, so no plot panel can be added at all.
MISSING_PLOT_KINDS_PENDING_APPROVAL = frozenset(
    {
        "Plot: 2D image",
        "Plot: Site map",
        "Plot: 1D vector",
        "Plot: Rolling trace",
        "Plot: Distribution",
        "Plot: Site grid",
    }
)


def _button_texts(window) -> frozenset[str]:
    return frozenset(
        button.text()
        for button in window.findChildren(QtWidgets.QAbstractButton)
        if button.text().strip()
    )


def _tab_titles(window) -> tuple[str, ...]:
    titles: list[str] = []
    for tabs in window.findChildren(QtWidgets.QTabWidget):
        titles.extend(tabs.tabText(index) for index in range(tabs.count()))
    return tuple(titles)


def _combo_items(window) -> frozenset[str]:
    items: set[str] = set()
    for combo in window.findChildren(QtWidgets.QComboBox):
        items.update(combo.itemText(index) for index in range(combo.count()))
    return frozenset(items)


@pytest.fixture(scope="module")
def consoles(tmp_path_factory):
    """Both windows, in this order: the old one needs no installation runtime."""

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from Zou_lab_control.frontend.task_console import (
        TaskConsole,
        default_console_state,
    )
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    old = TaskConsole(hub=SignalHub(), state=default_console_state())

    import Zou_lab_control.notebook as zlc

    experiment = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("parity") / "workspace",
    )
    new = experiment.task_console()
    try:
        yield old, new
    finally:
        for card in getattr(new, "cards", ()):
            card._panel = None
        new.close()


def test_the_legacy_console_is_still_here_to_compare_against(consoles):
    """Without the old window this file would be asserting against a memory."""

    old, _new = consoles
    assert _button_texts(old), "the legacy console must still build on this branch"
    assert _tab_titles(old) == MISSING_TABS_PENDING_APPROVAL


def test_no_header_control_may_be_dropped_beyond_the_registered_ones(consoles):
    old, new = consoles
    dropped = _button_texts(old) - _button_texts(new)

    assert dropped == MISSING_CONTROLS_PENDING_APPROVAL, (
        "the set of header controls missing from the new console changed.\n"
        f"  dropped now      : {sorted(dropped)}\n"
        f"  registered       : {sorted(MISSING_CONTROLS_PENDING_APPROVAL)}\n"
        "If you RESTORED one, delete it from this list and from UX-014.\n"
        "If you DROPPED one, put it back - Rule 9 does not allow it."
    )


def test_no_control_may_be_invented_beyond_the_registered_ones(consoles):
    old, new = consoles
    added = _button_texts(new) - _button_texts(old)

    assert added == ADDED_CONTROLS_PENDING_APPROVAL, (
        "the new console grew a control the old one never had.\n"
        f"  added now  : {sorted(added)}\n"
        f"  registered : {sorted(ADDED_CONTROLS_PENDING_APPROVAL)}"
    )


def test_the_permanent_tab_skeleton_is_tracked(consoles):
    old, new = consoles
    assert _tab_titles(old) == MISSING_TABS_PENDING_APPROVAL
    # Today the new console has no tabs at all.  When Monitor/Logic land, this
    # assertion is what forces the list above to be emptied in the same commit.
    present = set(_tab_titles(new))
    still_missing = tuple(
        title for title in MISSING_TABS_PENDING_APPROVAL if title not in present
    )
    assert still_missing == MISSING_TABS_PENDING_APPROVAL, (
        "a permanent tab appeared; remove it from MISSING_TABS_PENDING_APPROVAL "
        "and from UX-014 in the same commit."
    )


def test_the_add_panel_catalog_gap_is_tracked(consoles):
    old, new = consoles
    old_kinds = {item for item in _combo_items(old) if item.startswith("Plot: ")}
    assert old_kinds == MISSING_PLOT_KINDS_PENDING_APPROVAL

    new_items = _combo_items(new)
    restored = old_kinds & new_items
    assert not restored, (
        f"plot kinds {sorted(restored)} are now offered; remove them from "
        "MISSING_PLOT_KINDS_PENDING_APPROVAL and from UX-014."
    )


def test_every_registered_gap_has_a_row_in_the_ux_ledger():
    """A gap that is not written down is a silent downgrade, which Rule 9 bans."""

    text = DESIGN.read_text(encoding="utf-8")
    assert "UX-014" in text, "the console parity deviation must be registered"
    section = text[text.index("UX-014") :]
    section = section[: section.index("\n\n") if "\n\n" in section else len(section)]
    for name in (
        *MISSING_CONTROLS_PENDING_APPROVAL,
        *ADDED_CONTROLS_PENDING_APPROVAL,
        *MISSING_TABS_PENDING_APPROVAL,
        *MISSING_PLOT_KINDS_PENDING_APPROVAL,
    ):
        assert name in section, f"UX-014 does not mention {name!r}"


def test_the_new_console_is_measurably_smaller_and_that_is_recorded():
    """A blunt scalar the next reader cannot miss."""

    text = DESIGN.read_text(encoding="utf-8")
    assert "UX-014" in text
    # The ledger states the raw counts so the size of the gap is not left to
    # anyone's impression of it.
    assert "49" in text and "20" in text
