"""#H3s-F7 contract tests for the unified pulse-scan param form (`_PulseSlotsWidget`).

ONE scan table + a ``Scan: [None | API | Scan]`` mode toggle (the two always-visible scan tables
are gone).  The toggle is the SINGLE place that picks WHAT is swept; the table's columns / legend /
templates adapt to the mode; a mode whose kind is absent is disabled; switching API<->Scan keeps
each mode's own program buffer; ``values_dict`` emits the new single-``scan_code`` schema.

Offscreen, no real window -- the widget is built directly with synthetic slot rows.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def _app(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _widget_both():
    """A widget rebuilt with BOTH one api slot and one scan slot."""
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget
    w = _PulseSlotsWidget()
    w.rebuild(api_rows=[("a1", "duration", "1", "us", 5.0)],
              scan_rows=[("s0", "duration", "2", "ns", "probe")])
    return w


def _set_mode(w, mode: str) -> None:
    w._mode_combo.setCurrentIndex(w._mode_combo.findData(mode))


def test_default_mode_is_scan_when_scan_slots_exist(_app):
    """With both api + scan slots, the default mode is Scan (scan slots win the default)."""
    from Zou_lab_control.frontend.task_console import _scan_modes
    none, api, scan = _scan_modes()
    w = _widget_both()
    assert w.values_dict()["scan_mode"] == scan


def test_switch_to_api_shows_api_columns_and_extra_delay(_app):
    """Switching to API mode shows the api-slot columns + the extra-settle row; that row is hidden
    again in Scan mode (it is meaningful only for the software api sweep)."""
    from Zou_lab_control.frontend.task_console import _scan_modes
    none, api, scan = _scan_modes()
    w = _widget_both()
    assert w._extra_delay is None                     # default Scan mode: no extra-delay row
    _set_mode(w, api)
    assert w.values_dict()["scan_mode"] == api
    assert w._extra_delay is not None                 # API mode: extra-settle row shown
    # the legend names the api column (a1) in API mode
    legend = _legend_text(w)
    assert "a1" in legend
    _set_mode(w, scan)
    assert w._extra_delay is None                     # back to Scan: extra-delay row gone


def _legend_text(w) -> str:
    from PyQt5 import QtWidgets
    return "\n".join(lbl.text() for lbl in w.findChildren(QtWidgets.QLabel))


def test_values_dict_schema_per_mode(_app):
    """``values_dict`` emits exactly ``{api, scan_mode, scan_code, extra_delay}`` -- ONE scan_code
    (the active mode's program), and no separate api_scan key."""
    from Zou_lab_control.frontend.task_console import _scan_modes
    none, api, scan = _scan_modes()
    w = _widget_both()
    v = w.values_dict()
    assert set(v.keys()) == {"api", "scan_mode", "scan_code", "extra_delay"}
    assert "api_scan" not in v                          # the old second program key is gone
    assert v["scan_mode"] == scan
    _set_mode(w, none)
    v = w.values_dict()
    assert v["scan_mode"] == none
    assert v["scan_code"] == ""                         # None mode -> no table -> empty program


def test_absent_kind_disables_that_mode(_app):
    """A template with no scan slot disables Scan (and defaults to API); no api slot disables API
    (defaults to Scan); neither -> only None is selectable."""
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget, _scan_modes
    none, api, scan = _scan_modes()

    # api-only -> Scan disabled, default API
    w_api = _PulseSlotsWidget()
    w_api.rebuild(api_rows=[("a1", "duration", "1", "us", 5.0)], scan_rows=[])
    assert w_api.values_dict()["scan_mode"] == api
    assert not w_api._mode_combo.model().item(w_api._mode_combo.findData(scan)).isEnabled()

    # scan-only -> API disabled, default Scan
    w_scan = _PulseSlotsWidget()
    w_scan.rebuild(api_rows=[], scan_rows=[("s0", "duration", "2", "ns", "probe")])
    assert w_scan.values_dict()["scan_mode"] == scan
    assert not w_scan._mode_combo.model().item(w_scan._mode_combo.findData(api)).isEnabled()

    # neither -> only None
    w_none = _PulseSlotsWidget()
    w_none.rebuild(api_rows=[], scan_rows=[])
    assert w_none.values_dict()["scan_mode"] == none
    model = w_none._mode_combo.model()
    assert not model.item(w_none._mode_combo.findData(api)).isEnabled()
    assert not model.item(w_none._mode_combo.findData(scan)).isEnabled()
    assert model.item(w_none._mode_combo.findData(none)).isEnabled()


def test_switching_api_scan_preserves_each_mode_buffer(_app):
    """Each mode keeps its OWN last-typed program: type one program in API, another in Scan, then
    switch back and forth -- each mode's editor restores its own text (the columns differ, so the
    two buffers are never collapsed into one string)."""
    from Zou_lab_control.frontend.task_console import _scan_modes
    none, api, scan = _scan_modes()
    w = _widget_both()                                  # starts in Scan
    w._scan_code.setPlainText("scan_table = [[2000.0]]")
    _set_mode(w, api)
    w._scan_code.setPlainText("scan_table = [[3.0]]")
    _set_mode(w, scan)
    assert "2000.0" in w._scan_code.toPlainText()        # scan buffer restored
    _set_mode(w, api)
    assert "3.0" in w._scan_code.toPlainText()           # api buffer restored


def test_seed_value_round_trips_mode_and_program_and_api(_app):
    """A SAVED blob (a load) restores the api fixed values + the scan_mode + the active program:
    seed_value stashes them and the next rebuild (the template-driven repopulation) applies them."""
    from Zou_lab_control.frontend.task_console import _scan_modes
    none, api, scan = _scan_modes()
    w = _widget_both()
    # simulate a load that ran in API mode with a specific program + api value
    w.seed_value({"api": {"a1": 7.5}, "scan_mode": api,
                  "scan_code": "scan_table = [[4.0], [5.0]]", "extra_delay": 0.02})
    # the form re-builds from the template (same slots) -- the round-trip restores the saved state
    w.rebuild(api_rows=[("a1", "duration", "1", "us", 5.0)],
              scan_rows=[("s0", "duration", "2", "ns", "probe")])
    v = w.values_dict()
    assert v["scan_mode"] == api
    assert v["api"].get("a1") == pytest.approx(7.5)
    assert "4.0" in v["scan_code"] and "5.0" in v["scan_code"]
    assert v["extra_delay"] == pytest.approx(0.02)
