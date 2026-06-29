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
def test_logic_node_source_picker_groups_by_producer_with_human_labels_and_bare_data():
    """A signal-kind param renders the SAME nested collapsible-tree picker the plot panels use (G2):
    one producer GROUP, each leaf shown by its HUMAN label (never the raw ``judge_occupancy_rate`` key,
    #H3w-4) but carrying the BARE signal name as its bind data.  Asserted on the single-source grouping
    (`signal_tree_groups`) the widget is built from, + that the panel reads a pick back by bare name."""
    from Zou_lab_control.frontend.task_console import (MeasurementPanel, signal_tree_groups, read_editable_combo)
    from Zou_lab_control.neutral_atom.operations.measurement import ParamDecl
    from Zou_lab_control.frontend.qt_fluent import FluentTreeComboBox

    names = ["occupancy_occupied", "occupancy_rate"]
    sources = {"occupancy_occupied": ["Judge occupancy #1"], "occupancy_rate": ["Judge occupancy #1"]}
    formats = {"occupancy_occupied": "(35,)", "occupancy_rate": "scalar"}
    labels = {"occupancy_occupied": "Occupied", "occupancy_rate": "Loading rate"}
    groups = signal_tree_groups(names, sources, formats, labels)
    assert [g[0] for g in groups] == ["Judge occupancy #1"]              # one producer group
    leaves = groups[0][1]
    leaf_labels = [lbl for lbl, _bare, _full in leaves]
    bare = [bare for _lbl, bare, _full in leaves]
    assert any(l.startswith("Occupied") for l in leaf_labels) and any(l.startswith("Loading rate") for l in leaf_labels)
    assert not any("occupancy_" in l for l in leaf_labels)               # NO raw key leaks into the label
    assert set(bare) == set(names)                                       # bind data stays the bare name

    # and the widget reads a pick back by its BARE name (the bind key), not the human label shown
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()
    combo = FluentTreeComboBox()
    combo.set_signal_tree(groups, current="occupancy_rate", none_label="(none)")
    assert read_editable_combo(combo) == "occupancy_rate"
    combo.deleteLater()


def test_logic_source_picker_uses_short_name_labels_like_the_plot_picker():
    """#combo-parity: the occupancy / logic-node SOURCE picker must render IDENTICALLY to the plot
    Setting picker -- a camera leaf reads "frame_0", NOT the prefix-stripped "0" the
    _common_token_prefix fallback yields when NO short-name labels are threaded.  Guards the
    labels_provider wiring (MeasurementPanel.short_names_provider -> signal_expr_factory ->
    _SignalExprWidget._labels); without it the two pickers diverged ("camera 0/1/2" vs "camera
    frame_0")."""
    from PyQt5 import QtCore
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import _SignalExprWidget
    ensure_qt_app()
    names = ["frame_0", "frame_1", "frame_2"]
    sources = {n: ["camera"] for n in names}
    formats = {n: "(1,1,40,50)" for n in names}
    short = {n: n for n in names}                          # camera prefix "" -> short name == name

    def leaves(w):
        m = w.slot_combos[0]._model
        out = []
        for r in range(m.rowCount()):
            it = m.item(r)
            for c in range(it.rowCount()):
                ch = it.child(c)
                if ch.data(QtCore.Qt.UserRole):
                    out.append(ch.text().strip().split("  ")[0])
        return out

    with_labels = _SignalExprWidget(signals_provider=lambda: names, sources_provider=lambda: sources,
                                    formats_provider=lambda: formats, labels_provider=lambda: short)
    assert leaves(with_labels) == ["frame_0", "frame_1", "frame_2"]   # short NAME, like the plot picker
    without = _SignalExprWidget(signals_provider=lambda: names, sources_provider=lambda: sources,
                                formats_provider=lambda: formats)
    assert leaves(without) == ["0", "1", "2"]                         # the OLD bug (no labels -> stripped)
    with_labels.deleteLater(); without.deleteLater()


# ---------------------------------------------------- #3 a WAITING (declared, not-yet-published) signal
def test_signal_picker_can_select_a_declared_not_yet_published_signal():
    """A signal a node DECLARES but has not published yet (waiting) is still listed in the tree -- via
    the providers -- and reads back by its bare name, so you can wire a panel to a node before it runs
    (no 'must publish first' dead-end).  Read via ``read_editable_combo`` (the tree's documented path)."""
    from Zou_lab_control.frontend.task_console import signal_tree_groups, read_editable_combo
    from Zou_lab_control.frontend.qt_fluent import FluentTreeComboBox, ensure_qt_app
    ensure_qt_app()
    # 'rate' is declared (in names) but waiting (no live format) -- it must still be pickable
    groups = signal_tree_groups(["frame", "rate"],
                                {"frame": ["camera"], "rate": ["Judge occupancy #1"]},
                                {"frame": "(48, 60)"},               # 'rate' has no format -> waiting
                                {"rate": "Loading rate"})
    combo = FluentTreeComboBox()
    combo.set_signal_tree(groups, current="rate", none_label="(none)")
    assert read_editable_combo(combo) == "rate"                       # the waiting signal is bound
    combo.deleteLater()


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
