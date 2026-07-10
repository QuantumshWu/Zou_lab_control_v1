# -*- coding: utf-8 -*-
"""Contract: the selector/fit -> SignalHub chain works as ONE GESTURE.

* Drawing an area rectangle on a LIVE image panel (selectors ON) creates -- or, when one
  already consumes that signal, retargets -- a RoiProcessor: roi_frame/roi_value appear on
  the hub with the drawn region, no Edit-tab round trip.
* Fit results are hub signals: FitProcessor consumes a frame and publishes the shared
  gaussian2d_center model's parameters (fit_x0/fit_y0/...) as scalars -- usable as a rolling
  monitor or a scan loss, exactly like roi_value.
"""
import numpy as np

from Zou_lab_control._readout_math import gaussian2d_center
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.processors.fit import (
    FIT_SPEC_NAME, FitProcessor, fit_frame_center)
from Zou_lab_control.neutral_atom.operations.processors.roi import ROI_SPEC_NAME

from conftest import add_logic_row, make_console


def _gaussian_frame(h=240, w=320, x0=200.0, y0=90.0, amp=150.0, size=18.0, offset=12.0):
    ys, xs = np.mgrid[0:h, 0:w]
    img = gaussian2d_center((xs, ys), amp, offset, size, x0, y0)
    return img.astype(np.float64)


def test_fit_frame_center_recovers_a_synthetic_gaussian():
    result = fit_frame_center(_gaussian_frame())
    assert result is not None
    assert abs(result["fit_x0"] - 200.0) < 2.0
    assert abs(result["fit_y0"] - 90.0) < 2.0
    assert abs(result["fit_size"] - 18.0) < 4.0


def test_fit_processor_publishes_fit_scalars_on_the_hub():
    hub = SignalHub()
    node = FitProcessor(hub, source_expr={"inputs": ["frame_0"], "source": "value = signal"})
    hub.publish({"frame_0": _gaussian_frame()})
    out = node.step()
    assert set(out) == set(FitProcessor.provides)
    assert abs(hub.latest("fit_x0") - 200.0) < 2.0
    assert abs(hub.latest("fit_y0") - 90.0) < 2.0


def test_area_drag_on_a_live_panel_creates_then_retargets_a_roi_node():
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
        con._on_panel_area_select(card, (100.0, 400.0, 50.0, 250.0))   # the drag's rectangle
        assert len(con.logic_nodes) == n_rows + 1                       # one ROI row created...
        roi_row = con.logic_nodes[-1]
        assert roi_row.node.name == ROI_SPEC_NAME
        roi_node = con._logic_nodes.get(id(roi_row))
        assert roi_node is not None                                     # ...and STARTED
        node.step(); roi_node.step()
        crop = con.hub.latest([s for s in roi_node.published_signals()
                               if s.endswith("roi_frame")][0])
        assert crop.shape == (200, 300)                                 # (y, x) of the drawn box

        # second drag: RETARGETS the running consumer, never stacks another row
        con._on_panel_area_select(card, (0.0, 50.0, 0.0, 40.0))
        assert len(con.logic_nodes) == n_rows + 1
        node.step(); roi_node.step()
        crop = con.hub.latest([s for s in roi_node.published_signals()
                               if s.endswith("roi_frame")][0])
        assert crop.shape == (40, 50)
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
