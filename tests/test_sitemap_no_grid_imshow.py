"""Regression: the per-site ``occupied`` signal must NEVER be reshapeable into a 2D grid heatmap.

The user reported the site map (connected to the OccupancyProcessor 'occupied' signal) rendering as a
7x7 GRID IMSHOW -- each cell brightness = occupancy -- instead of "camera frame underlay + occupancy
rings".  Root cause: PanelCard._signal_structure attached the producing node's grid_shape (the trap-array
(5,7)) to 'occupied' even though occupied is per-site DATA (points_shape=(), data_shape=(n_sites,)), so a
2D panel's _coerce reshaped (n_sites,) -> (5,7) and Live2DDis imshow'd it.

A node's grid_shape un-flattens a 2D SCAN's swept points; it must apply ONLY to a signal that HAS such
points.  This guards: occupied's structure carries grid_shape=() (no reshape eligibility), so a 2D panel
on occupied raises an explicit "needs an image" error instead of silently heatmapping -- while a real 2D
pulse scan still keeps its grid_shape (covered by tests/test_pulse_param_scan.py).
"""

from __future__ import annotations

import os
import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")


def test_occupied_structure_has_no_grid_shape_so_a_2d_panel_cannot_heatmap_it():
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

        camrow = add(("camera", "live"))
        judrow = add(("processor", "Judge occupancy"))
        con._start_logic_node(camrow); con._start_logic_node(judrow)
        fire_live_imaging(exp, exposure=exp.devices.camera.exposure)
        con._logic_nodes[id(camrow)].step()
        con._logic_nodes[id(judrow)].step()

        # The OccupancyProcessor carries the trap grid_shape (e.g. (5,7)) on the NODE, but 'occupied'
        # is PER-SITE DATA -- so its structure must expose grid_shape=() (NOT the trap grid), or a 2D
        # panel could reshape (n_sites,) -> (5,7) and imshow a heatmap.
        st = con._signal_structure("occupied")
        assert st is not None, "occupied must resolve a producing-node structure"
        assert tuple(st["points_shape"]) == (1,), st               # no scan -> exactly one data-point (H4c iron law)
        data = tuple(st["data_shape"])
        assert len(data) == 1 and data[0] >= 2, st                 # per-site (n_sites,) DATA
        assert tuple(st["grid_shape"]) == (), (
            f"occupied must carry grid_shape=() so it can never become a 2D grid imshow; got {st}")
    finally:
        if con is not None:
            con.shutdown()
        exp.close()
