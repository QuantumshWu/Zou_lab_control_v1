# -*- coding: utf-8 -*-
"""Contract: one semantic selection feeds explicitly chosen Fit or ROI actions.

* A rectangle is data only until ``selection_action='roi'``; that explicit action creates or
  retargets a RoiProcessor whose canonical outputs appear on the hub.
* Fit results are hub signals: FitProcessor consumes a frame and publishes the shared
  gaussian2d_center model's parameters (fit_x0/fit_y0/...) as scalars -- usable as a rolling
  monitor or a scan loss, exactly like roi_value.
"""
import numpy as np

from Zou_lab_control._readout_math import gaussian2d_center
from Zou_lab_control.neutral_atom.core.fitting import FitRequest, fit_image
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.processors.fit import (
    FIT_SPEC_NAME, FitProcessor)
from Zou_lab_control.neutral_atom.operations.processors.roi import ROI_SPEC_NAME

from conftest import add_logic_row, make_console


def _gaussian_frame(h=240, w=320, x0=200.0, y0=90.0, amp=150.0, size=18.0, offset=12.0):
    ys, xs = np.mgrid[0:h, 0:w]
    img = gaussian2d_center((xs, ys), amp, offset, size, x0, y0)
    return img.astype(np.float64)


def test_shared_image_fit_recovers_a_synthetic_gaussian():
    result = fit_image(_gaussian_frame(), FitRequest("center"))
    assert result.valid, result.status
    params = result.parameter_map()
    assert abs(params["x0"] - 200.0) < 2.0
    assert abs(params["y0"] - 90.0) < 2.0
    assert abs(params["radius"] - 18.0) < 4.0


def test_fit_processor_publishes_fit_scalars_on_the_hub():
    hub = SignalHub()
    node = FitProcessor(hub, source_expr={"inputs": ["frame_0"], "source": "value = signal"})
    hub.publish({"frame_0": _gaussian_frame()})
    out = node.step()
    assert set(out) == set(FitProcessor.provides)
    assert hub.latest("fit_x0").shape == (1, 1, 1)
    assert abs(hub.latest("fit_x0")[0, 0, 0] - 200.0) < 2.0
    assert abs(hub.latest("fit_y0")[0, 0, 0] - 90.0) < 2.0
    assert hub.latest("fit_valid")[0, 0, 0]

    # Failure is an explicit same-transaction overwrite, never stale success.
    hub.publish({"frame_0": np.ones_like(_gaussian_frame())})
    node.step()
    assert not hub.latest("fit_valid")[0, 0, 0]
    assert hub.latest("fit_status")[0, 0, 0] == 0
    assert np.isnan(hub.latest("fit_x0")[0, 0, 0])
    assert np.isnan(hub.latest("fit_rmse")[0, 0, 0])


def test_explicit_roi_action_creates_then_retargets_a_roi_node():
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row = add_logic_row(con, ("camera", "live"))
        con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
        con._start_logic_node(row)
        node = con._logic_nodes[id(row)]
        node.step()
        sig = sorted(node.published_signals())[0]
        kc = con.kind_combo
        kc.setCurrentIndex(next(i for i in range(kc.count())
                                if kc.itemText(i) == "Plot: 2D image"))
        con._add_panel()
        card = con.cards[-1]
        card.config.inputs = [sig]
        card._apply_source()
        con.refresh_once()

        n_rows = len(con.logic_nodes)
        from Zou_lab_control.neutral_atom.core.selection import Selection
        card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        assert len(con.logic_nodes) == n_rows + 1                       # one ROI row created...
        roi_row = con.logic_nodes[-1]
        assert roi_row.node.name == ROI_SPEC_NAME
        roi_node = con._logic_nodes.get(id(roi_row))
        assert roi_node is not None                                     # ...and STARTED
        node.step(); roi_node.step()
        crop = con.hub.latest([s for s in roi_node.published_signals()
                               if s.endswith("roi_frame")][0])
        assert crop.shape[-2:] == (200, 300)

        # second drag: RETARGETS the running consumer, never stacks another row
        con._on_panel_area_select(card, Selection.rectangle(0.0, 50.0, 0.0, 40.0))
        assert len(con.logic_nodes) == n_rows + 1
        node.step(); roi_node.step()
        crop = con.hub.latest([s for s in roi_node.published_signals()
                               if s.endswith("roi_frame")][0])
        assert crop.shape[-2:] == (40, 50)

        # #10 SYMMETRIC TEARDOWN: leaving the ROI action STOPS + REMOVES the RoiProcessor through the
        # ONE selection-teardown seam, exactly as clearing a fit removes its FitProcessor -- no orphan
        # that keeps consuming after the operator switched the selector off.  The card wires this seam
        # to _select_analysis_action (an Analysis action change away from "roi") and to set_fit_request(None).
        assert callable(card.selection_clear_sink)
        card.selection_clear_sink(card, "roi")
        assert len(con.logic_nodes) == n_rows                           # the ROI row is gone
        assert con._logic_nodes.get(id(roi_row)) is None
    finally:
        con.shutdown()
        exp.close()


def test_selector_wiring_converts_axis_coords_to_the_frames_own_pixels():
    """The REAL drag wiring, end to end, through the region-signal model: a drag on a live 2d panel
    publishes its ``<slug>_region`` signal and creates a per-panel RoiProcessor consuming it; binding a
    2d panel to that ``roi_frame`` gives axes carrying the crop's ORIGIN; and a release on THAT panel is
    translated into the displayed frame's OWN pixels before its region is published (the ROI-of-ROI case
    -- unconverted axis pixels would clamp to a corner sliver)."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.selection import Selection
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        cam_row, frame_sig = _live_camera_signal(con)
        cam = con._logic_nodes[id(cam_row)]

        # A drag on the live frame panel: the console publishes the region signal + creates the ROI node.
        src_card = _add_2d_panel(con, frame_sig, title="frame")
        src_card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(src_card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        roi = con._logic_nodes[id(con._panel_analysis[id(src_card)])]
        cam.step(); roi.step()                    # roi_frame on the hub, region consumed
        crop_sig = [s for s in roi.published_signals() if s.endswith("roi_frame")][0]

        from Zou_lab_control.frontend.task_console import PanelConfig
        card = con._new_panel_card(PanelConfig(kind="2d", title="crop", size="2x2",
                                               source=f"value = {crop_sig}", params={}))
        con._attach_card(card)
        con.refresh_once()
        assert card._roi_built == [100, 400, 50, 250]        # axes carry the crop's origin

        card.set_selectors_enabled(True)
        card.config.params["selection_action"] = "roi"
        area = card.plotter.fig._zlc_tools.area
        assert area.callback == card._forward_area_select    # the sink IS wired on the selector

        area.range = [150.0, 200.0, 80.0, 120.0]             # AXIS coords on the crop's panel
        area._call()                                          # the selector's own release dispatch
        nested = con._logic_nodes[id(con._panel_analysis[id(card)])]
        cam.step(); roi.step(); nested.step()
        crop = con.hub.latest([s for s in nested.published_signals()
                               if s.endswith("roi_frame")][0])
        assert crop.shape[-2:] == (40, 50)        # local (y 30..70, x 50..100), NOT a clamped sliver
    finally:
        con.shutdown()
        exp.close()


def _live_camera_signal(con):
    """Start a live camera logic node and return (its row, its published frame signal)."""
    row = add_logic_row(con, ("camera", "live"))
    con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
    con._start_logic_node(row)
    node = con._logic_nodes[id(row)]
    node.step()
    return row, sorted(node.published_signals())[0]


def _add_2d_panel(con, signal, *, title=None):
    """Add a 2D image panel bound to ``signal`` through the real Add-Panel path and return its card."""
    kc = con.kind_combo
    kc.setCurrentIndex(next(i for i in range(kc.count())
                            if kc.itemText(i) == "Plot: 2D image"))
    con._add_panel()
    card = con.cards[-1]
    if title:
        card.config.title = title
    card.config.inputs = [signal]
    card._apply_source()
    con.refresh_once()
    return card


def test_two_panels_on_one_source_get_distinct_per_panel_analysis_nodes():
    """#3 root fix: an analysis node is keyed by the PANEL, never the source signal.  TWO panels bound
    to the SAME frame each drag their own ROI -> TWO distinct RoiProcessor rows with DISTINCT published
    output names.  The old source-signal keying made the second drag retarget the FIRST panel's node,
    so the two panels shared one node and one region."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.selection import Selection
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera_signal(con)
        card_a = _add_2d_panel(con, sig, title="Panel A")
        card_b = _add_2d_panel(con, sig, title="Panel B")
        card_a.config.params["selection_action"] = "roi"
        card_b.config.params["selection_action"] = "roi"

        con._on_panel_area_select(card_a, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        con._on_panel_area_select(card_b, Selection.rectangle(0.0, 60.0, 0.0, 40.0))

        row_a = con._panel_analysis[id(card_a)]
        row_b = con._panel_analysis[id(card_b)]
        assert row_a is not row_b                       # TWO distinct nodes, one per panel
        node_a = con._logic_nodes[id(row_a)]
        node_b = con._logic_nodes[id(row_b)]
        frame_a = [s for s in node_a.published_signals() if s.endswith("roi_frame")][0]
        frame_b = [s for s in node_b.published_signals() if s.endswith("roi_frame")][0]
        assert frame_a != frame_b                       # DISTINCT output names (no collision/sharing)

        # each node reduces ITS OWN region -> distinct crop shapes prove they are not shared
        _cam_node = con._logic_nodes[id(_cam)]
        _cam_node.step(); node_a.step(); node_b.step()
        assert con.hub.latest(frame_a).shape[-2:] == (200, 300)
        assert con.hub.latest(frame_b).shape[-2:] == (40, 60)
    finally:
        con.shutdown()
        exp.close()


def test_per_panel_teardown_removes_only_that_panels_node():
    """#1 root fix: clearing ONE panel's analysis (or turning its Selectors switch off, or removing the
    panel) tears down ONLY that panel's node; a second panel on the same source keeps running.  The old
    source-signal teardown removed EVERY node on the source, and the ROI branch had no teardown at all."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.selection import Selection
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera_signal(con)
        card_a = _add_2d_panel(con, sig, title="Panel A")
        card_b = _add_2d_panel(con, sig, title="Panel B")
        card_a.config.params["selection_action"] = "roi"
        card_b.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card_a, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        con._on_panel_area_select(card_b, Selection.rectangle(0.0, 60.0, 0.0, 40.0))
        row_a, row_b = con._panel_analysis[id(card_a)], con._panel_analysis[id(card_b)]

        # clear ONLY panel A's analysis
        card_a.selection_clear_sink(card_a, "roi")
        assert id(card_a) not in con._panel_analysis and row_a not in con.logic_nodes
        assert con._panel_analysis.get(id(card_b)) is row_b and row_b in con.logic_nodes  # B survives

        # turning panel B's Selectors switch OFF tears B's node down (symmetric, ROI included)
        card_b.set_selectors_enabled(True)
        card_b.set_selectors_enabled(False)
        assert id(card_b) not in con._panel_analysis and row_b not in con.logic_nodes
    finally:
        con.shutdown()
        exp.close()


def test_one_source_one_panel_fits_another_rois_distinct_nodes():
    """The acceptance scenario: TWO panels on the SAME source, one running a 2-D centre FIT and the
    other an ROI, own DISTINCT per-panel nodes -- a FitProcessor (publishing fit_x0/...) and a
    RoiProcessor (publishing roi_frame/roi_value) -- with distinct output names."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.fitting import FitRequest
    from Zou_lab_control.neutral_atom.core.selection import Selection
    from Zou_lab_control.neutral_atom.operations.processors.fit import FitProcessor
    from Zou_lab_control.neutral_atom.operations.processors.roi import RoiProcessor
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera_signal(con)
        fit_card = _add_2d_panel(con, sig, title="Fit panel")
        roi_card = _add_2d_panel(con, sig, title="ROI panel")

        fit_card.set_fit_request(FitRequest("center", selection=Selection.rectangle(0, 320, 0, 240)))
        roi_card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(roi_card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))

        fit_node = con._logic_nodes[id(con._panel_analysis[id(fit_card)])]
        roi_node = con._logic_nodes[id(con._panel_analysis[id(roi_card)])]
        assert isinstance(fit_node, FitProcessor) and isinstance(roi_node, RoiProcessor)
        fit_outs = set(fit_node.published_signals())
        roi_outs = set(roi_node.published_signals())
        assert any(s.endswith("fit_x0") for s in fit_outs)
        assert any(s.endswith("roi_frame") for s in roi_outs)
        assert fit_outs.isdisjoint(roi_outs)             # distinct per-panel output namespaces
    finally:
        con.shutdown()
        exp.close()


def test_removing_a_panel_removes_its_analysis_node():
    """#1: removing a panel from the board tears down the analysis node it owned (no orphan
    republishing after its panel is gone)."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.selection import Selection
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam, sig = _live_camera_signal(con)
        card = _add_2d_panel(con, sig, title="Panel A")
        card.config.params["selection_action"] = "roi"
        con._on_panel_area_select(card, Selection.rectangle(100.0, 400.0, 50.0, 250.0))
        row = con._panel_analysis[id(card)]
        assert row in con.logic_nodes
        con._remove_panel(card)
        assert id(card) not in con._panel_analysis and row not in con.logic_nodes
    finally:
        con.shutdown()
        exp.close()


def test_fit_center_spec_is_registered():
    from Zou_lab_control.neutral_atom.operations.processor_registry import discovered_processor_specs
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    try:
        names = {spec.name for spec in discovered_processor_specs(exp.readout)}
        assert FIT_SPEC_NAME in names and ROI_SPEC_NAME in names
    finally:
        exp.close()
