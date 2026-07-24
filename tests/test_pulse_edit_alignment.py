"""The Edit tab's three columns line up channel-for-channel.

Port Catalog (pin -> name), Delay/Scan (per-channel delay) and each Period card
(per-channel checkbox) all show one row per channel, and the operator reads them
ACROSS: cooling's pin, cooling's delay, cooling's on/off -- so those rows must sit
at the same Y.  The mechanism is that every card wraps its header in a fixed-height
top region (``_panel_top_height``); a period card that skipped it had a shorter
header and floated its checkboxes ~93 px above the matching delay row -- the
"period / delay / name alignment is wrong" complaint.

Measured on the real editor window through the authoritative widget lists.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore
import pytest

from zlc_frontend.qt_widgets import ensure_qt_app

#: Sub-pixel rounding across DPR tiers leaves a couple of px; the regression this
#: guards was a whole header row (~50-93 px), so a tight bound still discriminates.
MAX_ROW_SKEW_PX = 8


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture
def editor(application):
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    editor = open_pulse_editor()
    window = editor.window()
    window.show()
    for _ in range(6):
        application.processEvents()
    yield editor
    try:
        window.close()
    except Exception:                                    # pragma: no cover - teardown only
        pass
    application.processEvents()


def _top_y(widget) -> int:
    return widget.mapToGlobal(QtCore.QPoint(0, 0)).y()


def test_period_checkboxes_line_up_with_the_channel_delay_rows(editor):
    card = editor.drag_container.pulse_cards()[0]
    panel = editor.channel_panel
    channels = [key for key in panel.delay_edits if key in card.checks]
    assert channels, "no channels are shown in both the Period card and the Delay column"

    skewed = []
    for key in channels:
        dy = _top_y(card.checks[key]) - _top_y(panel.delay_edits[key])
        if abs(dy) > MAX_ROW_SKEW_PX:
            skewed.append(f"{key}: checkbox is {dy:+d} px off its delay row")
    assert not skewed, (
        "the Period checkboxes do not line up with the channel delay rows:\n"
        + "\n".join(skewed))


def _row_centre_y(widget) -> float:
    return _top_y(widget) + widget.height() / 2.0


def test_every_row_stays_aligned_in_the_show_all_compact_view(editor, application):
    """ALL 22 rows -- 18 digital channels AND the 4 DAC bus rows -- line up in Show All.

    The regression this pins: the panels advanced 25 px per row while the (compact)
    period card advanced 23 -- its own margins/spacing literals -- so the columns
    drifted 2 px per row and the LAST row was 47 px off.  The row-region vertical
    geometry is now one source (``_row_region_vmetrics``) and every card row is pinned
    to the shared row height, so the skew must stay within rounding for every row.
    """

    from zlc_pulse import PORT_CLOCK

    state = editor.read_state()
    state.visible_ports = [
        port.key for port in state.port_catalog.ports if port.kind != PORT_CLOCK]
    editor.load_state(state)
    for _ in range(10):
        application.processEvents()

    names = editor.names_panel.raw_label_widgets
    card = editor.drag_container.pulse_cards()[0]
    rows_checked = 0
    skewed = []
    for key, name_widget in names.items():
        if key in card.checks:
            other = card.checks[key]
        elif key.startswith("bus:") and key[4:] in card.bus_mode_combos:
            other = card.bus_mode_combos[key[4:]].parentWidget()
        else:
            continue
        rows_checked += 1
        dy = _row_centre_y(other) - _row_centre_y(name_widget)
        if abs(dy) > 2.5:
            skewed.append(f"{key}: {dy:+.1f} px")
    assert rows_checked >= 20, f"only {rows_checked} rows were compared -- the probe is broken"
    assert not skewed, (
        "rows drift apart in the Show All (compact) view:\n" + "\n".join(skewed))
