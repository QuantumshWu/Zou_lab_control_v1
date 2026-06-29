"""#4 edit-resize: the per-panel Edit tab's snapshot canvas must be a FIXED size = the figure's design
size, so repeated rebuilds (e.g. scrolling the bins param, which is a structure knob that re-snapshots)
keep it identical -- it can neither be stretched taller by the scroll-area layout (the reported "image
balloons / cuts off after a few bin turns") nor squished below the figure."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")

from Zou_lab_control.frontend.qt_fluent import ensure_qt_app  # noqa: E402


def test_edit_snapshot_canvas_is_fixed_size_and_idempotent_across_bin_rebuilds():
    ensure_qt_app()
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelEditor, PanelConfig, GAP)

    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
    console._timer.stop()
    try:
        from Zou_lab_control.frontend import panel_plot
        vals = np.concatenate([np.random.default_rng(0).normal(300.0, 20.0, 400),
                               np.random.default_rng(1).normal(460.0, 20.0, 300)])
        cfg = PanelConfig(kind="hist", title="dis", source="counts", row=GAP, col=GAP, size="2x4")
        card = console._new_panel_card(cfg)
        console._attach_card(card)
        # give the live card a real hist plotter (the render pipeline is not what we test here), and
        # stop the editor's pre-snapshot refresh from clearing it -- we test rebuild()'s SIZE behaviour.
        card.plotter = panel_plot(vals, kind="hist", size=cfg.size, bins=60, title="dis")
        console.refresh_once = lambda: None
        assert card.plotter is not None

        editor = PanelEditor(card, console)
        editor.rebuild()
        cv = editor._canvas
        assert cv is not None
        # FIXED: the snapshot canvas can neither grow nor squish (min == max == the design sizeHint)
        assert cv.minimumSize() == cv.maximumSize() == cv.sizeHint()
        size0 = (cv.width(), cv.height())

        # scrolling bins re-snapshots through rebuild(); the size must NOT drift across turns
        for b in (80, 120, 160, 200):
            card.config.params["bins"] = b
            editor.rebuild()
        cv = editor._canvas
        assert (cv.width(), cv.height()) == size0, "Edit snapshot canvas changed size across bin rebuilds"
        assert cv.minimumSize() == cv.maximumSize(), "snapshot canvas must stay FIXED, not growable"
    finally:
        console.shutdown()
        exp.close()
