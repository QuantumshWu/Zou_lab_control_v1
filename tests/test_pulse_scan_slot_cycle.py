"""Clicking a period's scan dot cycles the field through its three states, visibly.

The dot on a duration (or DAC) field cycles the binding none -> SCAN (sN) -> API
(aN) -> none, exactly as the reference does.  The regression this pins: the click
mutated the model but the REBUILT card never re-applied the marker, so the field
looked untouched after every click -- the whole 3-state effect was invisible.

Driven the way a person drives it: open the editor, take the live card from the
drag container (NOT ``findChildren``, which also returns just-deleted cards that are
pending garbage collection and would report a stale, unbound field), click the dot,
and read the marker the render produced.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from zlc_frontend.qt_widgets import ensure_qt_app


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


def _duration_field(editor):
    """The live first-period duration field, from the authoritative card list."""

    card = editor.drag_container.pulse_cards()[0]
    return card.duration_edit, card.duration_dot


def _settle(application):
    for _ in range(4):
        application.processEvents()


def test_duration_dot_cycles_none_scan_api_off(editor, application):
    field, _dot = _duration_field(editor)
    # OFF to begin with: a hollow dot, a plain numeric value, editable.
    assert not field.dot.isChecked() and not getattr(field.dot, "_api", False), (
        "a fresh duration field must start unbound")

    # none -> SCAN: the field goes read-only and shows its slot expression, and the
    # dot fills orange with the 1-based slot number.
    editor.drag_container.pulse_cards()[0].duration_dot.click()
    _settle(application)
    field, _dot = _duration_field(editor)
    assert field.dot.isChecked() and not getattr(field.dot, "_api", False), (
        "first click must bind the field to a SCAN slot (orange dot)")
    assert field.dot.number() == 1, "the scan dot must show the 1-based slot number"
    assert field.isReadOnly(), "a scan-bound value moves to the scan table -> read-only"
    from zlc_workbench.pulse_editor.plot_bridge_pulse_gui import _is_slot_expr
    assert _is_slot_expr(field.text()), (
        f"a scan-bound duration must show its slot expression, not {field.text()!r}")

    # SCAN -> API: the value comes back (editable), and the dot turns violet.
    editor.drag_container.pulse_cards()[0].duration_dot.click()
    _settle(application)
    field, _dot = _duration_field(editor)
    assert getattr(field.dot, "_api", False) and not field.dot.isChecked(), (
        "second click must move the field to an API slot (violet dot)")
    assert not field.isReadOnly(), "an API-bound value stays editable (the API sets it by name)"

    # API -> off: back to a hollow dot, no binding.
    editor.drag_container.pulse_cards()[0].duration_dot.click()
    _settle(application)
    field, _dot = _duration_field(editor)
    assert not field.dot.isChecked() and not getattr(field.dot, "_api", False), (
        "third click must return the field to unbound")


def test_delay_dot_cycles_none_api_none(editor, application):
    """A channel delay is not scannable, so its dot cycles none -> API -> none.

    This exercises the exact regression that hid on the channel side: a delay API
    binding lives in ``api_slots`` (not ``delays``), so the panel-rebuild cache key
    must include it -- otherwise cycling the dot binds the model but never rebuilds
    the panel, and the violet marker never shows.
    """

    def delay_dot():
        panel = editor.channel_panel
        key = list(panel.delay_edits.keys())[0]
        return panel.delay_edits[key]

    field = delay_dot()
    assert not getattr(field.dot, "_api", False), "a fresh delay field starts unbound"

    delay_dot().dot.click()                                          # none -> API
    _settle(application)
    field = delay_dot()
    assert getattr(field.dot, "_api", False) and field.dot.number() == 1, (
        "clicking the delay dot must bind an API slot and show its violet marker")
    assert not field.isReadOnly(), "an API delay stays editable (the API sets it by name)"

    delay_dot().dot.click()                                          # API -> none
    _settle(application)
    field = delay_dot()
    assert not getattr(field.dot, "_api", False), "a second click must clear the API binding"


def test_duration_dot_binding_survives_a_rebuild(editor, application):
    """Binding is state, not a widget flag: a full reload must re-show the marker.

    This is the exact path that was broken -- ``load_state`` rebuilds the card, and the
    rebuilt duration field must re-derive its scan marker from the state it was handed.
    """

    editor.drag_container.pulse_cards()[0].duration_dot.click()      # bind SCAN
    _settle(application)
    editor.load_state(editor.read_state())                           # full rebuild
    _settle(application)
    field, _dot = _duration_field(editor)
    assert field.dot.isChecked() and field.dot.number() == 1, (
        "a rebuilt card lost the scan marker -- the binding is in the state, "
        "so the render must re-apply it")
