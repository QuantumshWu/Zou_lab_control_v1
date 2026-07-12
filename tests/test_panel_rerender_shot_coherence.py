"""#shot-coherence-on-switch: an immediate panel re-render (signal switch / colormap / resize) must draw
from the console's CURRENT shared namespace (the SAME shot every other panel is on this tick), NOT the
panel's own stale ``_last_namespace`` (a PAST tick).

The user reported: after switching a panel's signal (or Stop/Start), the sitemap and the 2D image show
DIFFERENT shots ("明显不是同一次的signal").  Root cause: ``_apply_source`` -> ``_rerender_last`` replayed
the panel's cached namespace from an OLDER tick, so the switched panel lagged its siblings.  Fix:
``_rerender_last`` prefers the ``live_namespace_provider`` (a fresh ``hub`` snapshot) when it carries the
panel's bound signal, falling back to the cache only for a stopped producer.
"""

from __future__ import annotations

from conftest import raw_device_set

import os
import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")


def test_rerender_uses_current_shot_not_stale_cache(tmp_path):
    import Zou_lab_control.neutral_atom as na
    from tests.conftest import fire_live_imaging
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    exp = na.connect("virtual")
    con = None
    try:
        exp.readout.sitemap(method="box", frames=6, display=False)
        exp.readout.thresholds(frames=40, display=False)
        con = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
        con._timer.stop()
        kc = con.kind_combo

        def add(data):
            i = next(j for j in range(kc.count()) if kc.itemData(j) == data)
            kc.setCurrentIndex(i); con._add_panel(); return con.logic_nodes[-1]

        cam = add(("camera", "live")); jud = add(("processor", "Judge occupancy"))
        # The explicit default is session-owned calibration; no filesystem
        # artifact or fallback can change the geometry according to suite order.
        assert con._logic_editors[id(jud)].collect_values()["calibration_origin"] == "session"
        i = next(j for j in range(kc.count()) if kc.itemData(j) == "2d")
        kc.setCurrentIndex(i); con._add_panel()
        twod = con.cards[-1]; twod.config.inputs = ["frame_judged"]

        con._start_logic_node(cam); con._start_logic_node(jud)
        fire_live_imaging(exp, exposure=raw_device_set(exp).camera.exposure)
        camN = con._logic_nodes[id(cam)]; judN = con._logic_nodes[id(jud)]
        camN.step(); judN.step()

        # the panel cached an OLD shot; the hub then advances several shots WITHOUT a _tick
        twod._last_namespace = {"frame_judged": np.full((1, 96, 128), -999.0, dtype=float)}
        for _ in range(4):
            camN.step(); judN.step()
        current = np.asarray(con.hub.latest("frame_judged"))

        captured = {}
        twod.refresh = lambda ns: captured.setdefault("ns", ns)   # capture what _rerender_last draws from
        twod._rerender_last()                                       # the immediate re-render (signal switch path)

        ns = captured.get("ns")
        assert ns is not None, "_rerender_last must re-render"
        assert "frame_judged" in ns and np.array_equal(np.asarray(ns["frame_judged"]), current), (
            "an immediate re-render must use the CURRENT shared shot, not the panel's stale cache")
        assert float(np.asarray(ns["frame_judged"]).reshape(-1)[0]) != -999.0, "must NOT replay the stale cache"
    finally:
        if con is not None:
            con.shutdown()
        exp.close()
