"""Contract: figure SAVE has ONE source of truth -- the notebook (``plot.save()`` on a source-bound
plot) and the GUI (``PanelEditor.save``) both capture ``info['signals']`` + ``info['provenance']``
through the ONE frontend-neutral core (``operations.figure_capture``), so the two paths write
byte-identical rich npz.

Two mechanical guards:

1. **frontend-neutral core** -- ``neutral_atom.operations.figure_capture`` must NEVER import the frontend
   or Qt (it lives in the sealed analysis layer, next to ``test_virtual_equals_real_contract``); it is
   pure over the ``SignalHub`` + ``LogicNode`` contracts and reads device state only through the public
   ``.snapshot()``.

2. **notebook == GUI single source** -- for the SAME virtual session / node / signal, a source-bound
   notebook ``plot.save()`` and the GUI ``PanelEditor.save`` produce the SAME ``info['signals']`` (keys /
   roles / block shapes / block bytes) and the SAME ``info['provenance']`` (top-level ``devices`` +
   the top-level key set).  Both npz then ``load_figure`` + ``plot()`` (build) without error.  A plot with
   NO source binding degrades to the basic figure+npz save (no signals / provenance) -- the old behaviour.

Runs headless (``QT_QPA_PLATFORM=offscreen``); virtual sleeps are fast-forwarded.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import Zou_lab_control.neutral_atom as na  # noqa: E402
from Zou_lab_control.neutral_atom.operations import figure_capture  # noqa: E402


# --------------------------------------------------------------------- guard 1: frontend-neutral core
def test_figure_capture_never_imports_frontend_or_qt():
    """The core capture module is in the sealed analysis layer: it must import NO frontend / Qt symbol
    (it is pure over SignalHub + LogicNode).  A static import scan (so the rule holds even for a lazily
    imported path a runtime test would miss)."""
    src = Path(figure_capture.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = ("frontend", "PyQt5", "PyQt6", "PySide2", "PySide6", "matplotlib", "qt_fluent", "qt_canvas")
    offenders: list[str] = []
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for m in mods:
            if any(b in m for b in banned):
                offenders.append(m)
    assert offenders == [], f"figure_capture must not import frontend / Qt: {offenders}"


# --------------------------------------------------------------------- shared virtual fixture
def _camera_hub_node(exp):
    """A virtual camera measurement publishing ``cam_frame_0`` to a hub, run a few shots (the ONE fixture
    both the notebook and the GUI path bind to)."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement
    from conftest import fire_live_imaging

    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                            prefix="cam_", repeat=2)
    fire_live_imaging(exp)
    for _ in range(4):
        cam.step()
    cam.refresh_provenance()
    return hub, cam


def _signals_shape(info) -> dict:
    """The comparable shape of ``info['signals']``: ``{bare -> (role, block.shape, points, data)}``."""
    return {bare: (d["role"], tuple(np.asarray(d["block"]).shape), d["points_shape"], d["data_shape"])
            for bare, d in sorted((info.get("signals") or {}).items())}


# --------------------------------------------------------------------- guard 2: notebook == GUI
def test_notebook_save_and_gui_save_capture_identically(tmp_path):
    """One virtual session / camera node / ``cam_frame_0`` signal: the GUI ``PanelEditor.save`` and a
    source-bound notebook ``plot.save()`` write the SAME rich ``info`` -- identical ``signals`` (roles /
    shapes / bytes) and ``provenance`` (top-level devices + key set)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelConfig, PanelEditor, GAP)
    from Zou_lab_control.frontend import load_figure, panel_plot

    ensure_qt_app()
    exp = na.connect("virtual")
    console = None
    try:
        hub, cam = _camera_hub_node(exp)
        console = TaskConsole(hub=hub, state=default_console_state(), session=exp,
                              running_nodes=[cam], window_px=(900, 600))
        console._timer.stop()
        console.refresh_once = lambda: None

        # ----- GUI path: a panel wired to cam_frame_0, saved through PanelEditor.save (verbatim path)
        card = console._new_panel_card(PanelConfig(kind="hist", title="frame", source="value = cam_frame_0",
                                                   inputs=["cam_frame_0"], row=GAP, col=GAP, size="2x4"))
        console._attach_card(card)
        card.plotter = panel_plot(np.random.default_rng(0).normal(300.0, 20.0, 500),
                                  kind="hist", size="2x4", bins=40, title="frame")
        editor = PanelEditor(card, console)
        editor.rebuild()
        editor.save_dir_edit.setText(str(tmp_path / "gui"))
        editor.save_autoname.setChecked(False)
        editor.save()
        gui_info = load_figure(next(tmp_path.glob("gui*.npz"))).info

        # ----- NOTEBOOK path: a plot BOUND to the same hub + producing node, saved through plot.save()
        p = panel_plot(np.random.default_rng(0).normal(300.0, 20.0, 500),
                       kind="hist", size="2x4", bins=40, title="frame")
        p.bind_source(hub, cam, inputs=["cam_frame_0"],
                      resolve_node=console._node_for_signal, session=exp)
        p.save(str(tmp_path / "nb.png"))
        nb_info = load_figure(next(tmp_path.glob("nb*.npz"))).info

        # signals: identical keys / roles / shapes AND byte-for-byte identical blocks
        assert _signals_shape(gui_info) == _signals_shape(nb_info) != {}, "signals capture must match + be non-empty"
        for bare, gd in (gui_info["signals"]).items():
            np.testing.assert_array_equal(np.asarray(gd["block"]),
                                          np.asarray(nb_info["signals"][bare]["block"]))
        # provenance: identical top-level devices + identical top-level key set
        gp, npv = gui_info["provenance"], nb_info["provenance"]
        assert set(gp.get("devices", {})) == set(npv.get("devices", {})) == {"camera", "sequencer"}
        assert set(gp) == set(npv), "provenance top-level keys must match across the two paths"

        # both npz load + build a figure
        for info_npz in tmp_path.glob("*.npz"):
            df = load_figure(info_npz).plot()
            assert df.fig is not None
            plt.close("all")
    finally:
        if console is not None:
            console.shutdown()
        exp.close()
        plt.close("all")


def test_unbound_plot_save_degrades_to_basic(tmp_path):
    """A plot with NO source binding writes just the basic figure + npz (no ``signals`` / ``provenance``)
    -- the old behaviour, preserved for a plain array figure."""
    from Zou_lab_control.frontend import load_figure, panel_plot

    p = panel_plot(np.arange(50.0), kind="hist", size="2x4", bins=10, title="bare")
    p.save(str(tmp_path / "bare.png"))
    info = load_figure(next(tmp_path.glob("bare*.npz"))).info
    assert "signals" not in info and "provenance" not in info
    plt.close("all")
