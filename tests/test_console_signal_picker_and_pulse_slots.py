"""MECHANICAL guards for the five GUI fixes the user demanded in the "巨量问题" round:

  #1  an API slot can be put on a channel DELAY through the GUI (the delay field has the
      same cycling dot the duration/DAC fields have), not only via the data model;
  #2  removing a period no longer raises when an API slot binds a later/removed period --
      ``_shift_slot_targets`` drops the slot on the removed period and re-indexes the rest;
  #3  the logic-node Edit's ``source`` picker reuses the SAME nested-by-producer signal
      dropdown the plot panels use (grouped headers + indented bare names, read by data);
  #4  the plot panel's source expression has an "expand" affordance that opens a large
      floating editor (so it never has to be typed in the cramped inline field);
  #5  the Logic tab shows SHORT signal names ("rate"), never the prefixed "judge_occupancy_rate".

Run on the offscreen Qt platform; build widgets directly (no flaky demo fixture).
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing import default_imaging_template


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


# --------------------------------------------------------------------------- #2 remove period
def test_remove_period_with_api_slot_does_not_raise_and_remaps():
    """The default imaging template binds a1 to TWO frames (periods 1 and 5) and a2 to period 3.
    Removing a MIDDLE period must drop any slot on it and shift the later ones down so the
    surviving slots still bind real periods (the regression was a validate() ValueError)."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor

    state = default_imaging_template()
    assert state.api_names() == ["a1", "a2", "a3"]   # one unique handle per exposure cell
    editor = PulseSequenceEditor(state=state)
    try:
        # remove period 1 (image_0, carries a1): a1 is dropped, a2@3 -> 2, a3@5 -> 4
        editor._selected_period = 1
        editor.remove_period()                         # must not raise
        after = editor.read_state()
        assert len(after.periods) == len(state.periods) - 1
        targets = {s.name: int(s.target) for s in after.api_slots}
        assert "a1" not in targets                      # the only a1 was on removed period 1
        assert targets["a2"] == 2 and targets["a3"] == 4   # later slots re-indexed
        after.validate()                               # the bug: this used to raise

        # and removing the LAST period (the other reproduction) is also clean
        editor._clear_period_selection()
        editor.remove_period()
        editor.read_state().validate()
    finally:
        editor.deleteLater()


# --------------------------------------------------------------------------- #1 delay api dot
def test_delay_field_can_become_an_api_slot_via_gui():
    """A non-bus channel's delay field is a cycling-dot edit; toggling it to API binds a delay
    API slot in the read-back state, and ``set_api`` then sets that channel's delay by name."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor

    editor = PulseSequenceEditor(state=default_imaging_template())
    try:
        state = editor.read_state()
        channel = next(c for c in state.channels if c != "emCCD")   # any non-bus TTL channel
        assert editor.state.api_slot_for("delay", channel) is None
        editor._toggle_delay_api(channel)                            # the dot's none -> api step
        bound = editor.read_state()
        name = bound.api_slot_for("delay", channel)
        assert name is not None and name.startswith("a")            # a delay API handle now exists
        bound.set_api(name, 1.0e-6)                                  # set the delay BY NAME
        assert bound.delays[channel] == pytest.approx(1.0e-6)
    finally:
        editor.deleteLater()


# --------------------------------------------------------------------------- #3 grouped picker
def test_logic_node_source_picker_is_grouped_and_read_by_data():
    """A signal-kind param renders the SAME nested picker the plot panels use: a disabled bold
    producer header + the bare signal indented under it, the pick read via ``currentData()``."""
    from Zou_lab_control.frontend.task_console import MeasurementPanel
    from Zou_lab_control.neutral_atom.operations.measurement import ParamDecl

    spec = SimpleNamespace(
        name="judge_occupancy",
        params=(ParamDecl("source", "Frame signal", "signal", default="frame"),))
    panel = MeasurementPanel(
        [spec], single=True, controls=False,
        signals_provider=lambda: ["frame", "occupied", "rate"],
        sources_provider=lambda: {"frame": ["live_image"], "occupied": ["occupancy"], "rate": ["occupancy"]},
        formats_provider=lambda: {"frame": "(48, 60)", "occupied": "(N,)", "rate": "scalar"})
    try:
        tag, combo = panel._widgets["source"]
        assert tag == "signal"
        # grouped: at least one disabled (non-selectable) header row, signals carry the bare name
        from PyQt5.QtCore import Qt
        model = combo.model()
        headers = [i for i in range(combo.count())
                   if not (model.item(i).flags() & Qt.ItemIsEnabled)]
        datas = [combo.itemData(i) for i in range(combo.count())]
        assert headers, "expected at least one disabled producer header"
        assert "occupancy" in [combo.itemText(i).strip() for i in headers]
        assert {"frame", "occupied", "rate"} <= {d for d in datas if d}
        # picking an indented signal reads back its BARE name (label is indented; data is clean)
        idx = next(i for i in range(combo.count()) if combo.itemData(i) == "rate")
        combo.setCurrentIndex(idx)
        assert panel.collect_values()["source"] == "rate"
    finally:
        panel.deleteLater()


# ---------------------------------------------------- #3 editable-combo read robustness (review)
def test_signal_picker_keeps_a_freshly_typed_not_yet_published_name():
    """Selecting a published signal then TYPING a new (not-yet-published) name must read back the
    TYPED name -- not the stale previously-selected item's data.  (Qt does not move currentIndex on
    free text, so a plain currentData() would silently drop the new name.)"""
    from Zou_lab_control.frontend.task_console import MeasurementPanel
    from Zou_lab_control.neutral_atom.operations.measurement import ParamDecl

    spec = SimpleNamespace(name="judge_occupancy",
                           params=(ParamDecl("source", "Frame signal", "signal", default="frame"),))
    panel = MeasurementPanel(
        [spec], single=True, controls=False,
        signals_provider=lambda: ["frame", "rate"],
        sources_provider=lambda: {"frame": ["camera"], "rate": ["occupancy"]},
        formats_provider=lambda: {})
    try:
        combo = panel._widgets["source"][1]
        idx = next(i for i in range(combo.count()) if combo.itemData(i) == "rate")
        combo.setCurrentIndex(idx)                 # a real selection
        combo.setEditText("future_signal")         # then the user types a new name
        assert panel.collect_values()["source"] == "future_signal"
    finally:
        panel.deleteLater()


def test_empty_signal_pick_never_reads_back_a_group_header():
    """With signals present, no leading none-row and an empty current, the picker must NOT land on
    a disabled producer HEADER (whose label would otherwise read back as if it were a chosen
    signal); an empty pick reads back as ''."""
    from Zou_lab_control.frontend.task_console import fill_grouped_signal_combo, read_editable_combo
    from Zou_lab_control.frontend.qt_fluent import FluentComboBox

    combo = FluentComboBox()
    combo.setEditable(True)
    fill_grouped_signal_combo(
        combo, names=["rate", "occupied"],
        sources={"rate": ["occupancy"], "occupied": ["occupancy"]}, formats={},
        current="")                                # empty current, NO none_label
    assert read_editable_combo(combo) == ""        # not "occupancy" (the group header)


# --------------------------------------------------------------------------- #4 floating editor
def test_plot_panel_has_floating_expression_editor():
    """The Source row carries an expand affordance plus the method that opens the large floating
    editor -- so an expression need never be typed in the cramped inline field."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", title="t", row=0, col=0, size="2x2", source="value"))
    try:
        assert hasattr(card, "expand_button") and card.expand_button.text() == "Edit…"
        assert callable(getattr(card, "_open_expr_editor", None))
    finally:
        card.deleteLater()


# --------------------------------------------------------------------------- #5 short logic names
def test_logic_tab_shows_short_signal_names():
    """``_live_node_formats`` strips the node's disambiguating prefix so the Logic row legend
    shows "rate"/"occupied", never "judge_occupancy_rate"; the meaning still resolves because
    output_specs are keyed by the short name."""
    from Zou_lab_control.frontend.task_console import TaskConsole

    node = SimpleNamespace(
        layer="processor",
        prefix="judge_occupancy_",
        published_signals=lambda: ["judge_occupancy_rate", "judge_occupancy_occupied"],
        output_specs=lambda: [SimpleNamespace(name="rate", description="cumulative loading"),
                              SimpleNamespace(name="occupied", description="per-site occupancy")])
    stub = SimpleNamespace(hub=SimpleNamespace(latest=lambda name: None))
    rows = TaskConsole._live_node_formats(stub, node)
    names = [r[0] for r in rows]
    assert names == ["occupied", "rate"]                       # short + sorted, no prefix
    assert all("judge_occupancy_" not in n for n in names)
    assert dict((r[0], r[2]) for r in rows)["rate"] == "cumulative loading"   # desc still resolves
