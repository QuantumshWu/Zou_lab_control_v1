# -*- coding: utf-8 -*-
"""Contracts for the first-class Analysis processor + persisted panel<->row association (#1/#2/#7).

* The association is a PERSISTED single source (panel ``params['region_signal']`` == row
  ``values['region']``), derived at read time -- save/load re-associates, rename never decouples.
* STOP semantics: a re-drag on a STOPPED analysis row retargets in place -- never deletes the row,
  never auto-starts it, never purges its lingering signals.
* Role classification: every hub signal either has a provider or is ``role='control'``; a cleared
  analysis removes its region signal, so no ``(unbound)`` picker group appears.
"""

import numpy as np

from Zou_lab_control.neutral_atom.core.fitting import FitRequest
from Zou_lab_control.neutral_atom.core.selection import Selection
from Zou_lab_control.neutral_atom.operations.processors.analysis import (
    ANALYSIS_ACTIONS, ANALYSIS_SPEC_NAME)

from conftest import add_logic_row, make_console


def _live_camera(con):
    row = add_logic_row(con, ("camera", "live"))
    con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
    con._start_logic_node(row)
    node = con._logic_nodes[id(row)]
    node.step()
    return row, sorted(node.published_signals())[0]


def _add_2d(con, signal, title=None):
    kc = con.kind_combo
    kc.setCurrentIndex(next(i for i in range(kc.count()) if kc.itemText(i) == "Plot: 2D image"))
    con._add_panel()
    card = con.cards[-1]
    if title:
        card.config.title = title
    card.config.inputs = [signal]
    card._apply_source()
    con.refresh_once()
    return card


def test_every_hub_signal_has_a_provider_or_is_control():
    """#7 invariant: the hub carries ONLY node-provided data signals and role='control' regions --
    nothing unowned, so the picker can never grow an '(unbound)' group from an analysis."""
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera(con)
        card = _add_2d(con, sig, title="Panel A")
        card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        con._refresh_signal_info()
        providers = con._signal_providers()
        for name in con.hub.names():
            assert name in providers or con._is_control_signal(name), (
                f"hub signal {name!r} has no provider and no control role")
        # and the control signal is NOT in the bindable pool (one choke point -> every picker)
        region = card.config.params["region_signal"]
        assert region not in con._signal_names()
    finally:
        con.shutdown()
        exp.close()


def test_clearing_the_analysis_removes_the_region_no_unbound_group():
    """#7 symmetric lifecycle: analysis cleared -> its region signal leaves the hub AND the persisted
    association is dropped; the picker grows no '(unbound)' group."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.param_widgets import grouped_signal_items
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera(con)
        card = _add_2d(con, sig, title="Panel A")
        card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        region = card.config.params["region_signal"]
        assert region in set(con.hub.names())
        con._on_panel_selection_clear(card, "roi")           # the explicit clear seam
        assert region not in set(con.hub.names())            # region left WITH the analysis
        assert not card.config.params.get("region_signal")   # persisted association dropped
        con._refresh_signal_info()
        items = grouped_signal_items(con._signal_names(), con._signal_providers(),
                                     con._signal_formats())
        assert not any("(unbound)" in str(label) for label, _v in items), items
    finally:
        con.shutdown()
        exp.close()


def test_stopped_row_redrag_retargets_without_delete_or_autostart():
    """#1/#2 stop semantics: a user's Stop is respected -- a re-drag updates the SAME row's region
    and values, does NOT delete it, does NOT auto-start it, and keeps its lingering hub signals."""
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        cam_row, sig = _live_camera(con)
        card = _add_2d(con, sig, title="Panel A")
        card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        row = con._panel_analysis_row(card)
        node = con._logic_nodes[id(row)]
        con._logic_nodes[id(cam_row)].step(); node.step()
        lingering = set(node.published_signals())
        con._stop_logic_node(row)
        con._on_panel_area_select(card, Selection.rectangle(0.0, 60.0, 0.0, 40.0))
        assert con._panel_analysis_row(card) is row          # same row, still listed
        assert row in con.logic_nodes
        assert con._logic_nodes.get(id(row)) is None         # NOT auto-started
        assert lingering <= set(con.hub.names())             # lingering data kept
        # Logic-tab Start replays the freshly republished region
        con._start_logic_node(row)
        node2 = con._logic_nodes[id(row)]
        con._logic_nodes[id(cam_row)].step(); node2.step()
        crop = con.hub.latest([s for s in node2.published_signals() if s.endswith("roi_frame")][0])
        assert crop.shape[-2:] == (40, 60)
    finally:
        con.shutdown()
        exp.close()


def test_fit_roi_switch_is_one_row_one_node():
    """The action switch (fit <-> roi) re-uses the SAME Analysis row/node/region -- a parameter
    switch, never a delete-and-recreate (the single-catalog-entry point of the redesign)."""
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        cam_row, sig = _live_camera(con)
        card = _add_2d(con, sig, title="Panel A")
        card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        row = con._panel_analysis_row(card)
        node = con._logic_nodes[id(row)]
        assert node.action == "roi"
        card.set_fit_request(FitRequest("center", selection=Selection.rectangle(0, 320, 0, 240)))
        assert con._panel_analysis_row(card) is row          # SAME row
        assert con._logic_nodes.get(id(row)) is node         # SAME node
        node._apply_pending_params()                         # the worker's between-shots apply
        assert node.action == "fit"
        assert row.node.name == ANALYSIS_SPEC_NAME
        assert row.node.values.get("action") == "fit"
    finally:
        con.shutdown()
        exp.close()


def test_save_load_keeps_association_and_region():
    """Persistence: region_signal + region payload ride the panel config; load republishes the region
    and the derived association is immediately live (no runtime dict to rebuild)."""
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera(con)
        card = _add_2d(con, sig, title="Panel A")
        card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        region = card.config.params["region_signal"]
        state = con.read_state()
        con.load_state(state)
        loaded = con.cards[-1]
        assert loaded.config.params.get("region_signal") == region
        assert con._panel_analysis_row(loaded) is not None
        assert region in set(con.hub.registered_names())     # replayed as a control signal
        assert con._is_control_signal(region)
    finally:
        con.shutdown()
        exp.close()


def test_panel_combo_actions_are_the_processor_vocabulary():
    """The Setting Analysis combo's data values derive from ANALYSIS_ACTIONS (single source with the
    node's dispatch) -- 'none' plus exactly the processor's action vocabulary."""
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera(con)
        card = _add_2d(con, sig, title="Panel A")
        card._open_settings()
        combo = card.analysis_combo
        values = {combo.itemData(i) for i in range(combo.count())}
        card.settings_popup.hide()
        assert values == {"none", *ANALYSIS_ACTIONS}
    finally:
        con.shutdown()
        exp.close()
