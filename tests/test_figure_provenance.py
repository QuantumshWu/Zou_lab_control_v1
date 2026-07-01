"""Contract: a SAVED figure records the PROVENANCE of the data it plotted -- "what the apparatus was
doing when this data was taken" -- and the FigureViewer shows it.

The data flow (all through public paths):
  * an acquiring logic NODE (camera measurement / scan) refreshes ``provenance_snapshot()`` each shot:
    its held devices' PUBLIC ``.snapshot()`` (the backend-neutral contract) + acquisition params +
    calibration fingerprint + node/layer identity + a timestamp -- reading NO simulation ground truth,
    so it is the SAME record a real qCMOS run keeps (guarded by test_virtual_equals_real_contract);
  * the console's Save resolves the producing node of the panel's signal
    (``PanelEditor._provenance_for_save`` -> ``node.provenance_snapshot()``) and folds it into
    ``extra_info['provenance']``, which ``DataFigure.save`` stores under the npz's top-level ``info``;
  * ``load_figure`` reads ``info['provenance']`` back; the FigureViewer expands it readably.

These pin:
1. ``LogicNode.provenance_snapshot()`` unit behaviour: a node HOLDING a camera includes the camera's
   snapshot + acquisition params; a bare node (no device) returns no ``devices`` and never crashes;
2. end-to-end through the CONSOLE Save path: connect virtual -> a camera measurement publishes
   ``frame_0`` -> save its panel -> ``load_figure`` back -> ``info['provenance']`` carries the camera
   device snapshot + key acquisition params;
3. the fallback: no producing node but a session attached -> the whole-session device snapshot;
4. backward compatibility: an OLD npz with no ``provenance`` loads + the viewer expands nothing (no crash).

Runs headless (``QT_QPA_PLATFORM=offscreen``); virtual sleeps are fast-forwarded.
"""

from __future__ import annotations

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

pytest.importorskip("PyQt5")

import Zou_lab_control.neutral_atom as na  # noqa: E402
from Zou_lab_control.neutral_atom.core.signals import SignalHub  # noqa: E402
from Zou_lab_control.neutral_atom.operations.logic import (  # noqa: E402
    CameraMeasurement,
    OccupancyProcessor,
)
from Zou_lab_control.frontend import load_figure  # noqa: E402
from Zou_lab_control.frontend.qt_fluent import ensure_qt_app  # noqa: E402
from conftest import fire_live_imaging  # noqa: E402


# --------------------------------------------------------------------------- unit: the node snapshot
def test_provenance_snapshot_includes_held_camera_and_params():
    """A node that HOLDS a camera + sequencer records BOTH devices' public ``.snapshot()`` plus its own
    acquisition parameters -- the record captured at the last acquisition (refresh on each shot)."""
    exp = na.connect("virtual")
    try:
        hub = SignalHub()
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer, repeat=2)
        fire_live_imaging(exp)
        for _ in range(4):
            cam.step()
        prov = cam.provenance_snapshot()
        assert isinstance(prov, dict)
        # identity + timestamp
        assert prov["node"] == cam.display_label and prov["layer"] == "measurement"
        assert isinstance(prov.get("captured_at"), str) and prov["captured_at"]
        # the held devices' PUBLIC snapshots (camera contract), keyed by role
        devices = prov["devices"]
        assert "camera" in devices and "sequencer" in devices
        assert devices["camera"]["type"] == type(exp.devices.camera).__name__
        assert "exposure" in devices["camera"]                       # a real snapshot key, not a placeholder
        # the node's own acquisition params travelled too (the exposure / frames the operator set)
        assert "acquisition_parameters" in prov
        assert "exposure" in prov["acquisition_parameters"]
    finally:
        exp.close()


def test_provenance_snapshot_of_a_bare_node_has_no_devices_and_never_crashes():
    """A node holding NO acquisition device (a pure processor) yields a provenance dict with identity +
    timestamp but no ``devices`` section -- and never raises (the generic device probe finds nothing)."""
    hub = SignalHub()
    proc = OccupancyProcessor(hub)               # a processor holds no camera / sequencer
    prov = proc.provenance_snapshot()
    assert isinstance(prov, dict)
    assert prov["layer"] == "processor"
    assert "devices" not in prov, "a node with no held device declares no devices section"


# ------------------------------------------------------------------ end-to-end through the console save
def _camera_console(exp):
    """A real TaskConsole running a camera MEASUREMENT node that publishes ``frame_0`` -- the node is in
    ``running_nodes`` so the console's ``_node_for_signal`` resolves it (the Save provenance source)."""
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelConfig, GAP)
    from Zou_lab_control.frontend import panel_plot

    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                            prefix="cam_", repeat=2)
    fire_live_imaging(exp)
    for _ in range(4):
        cam.step()                                # publish cam_frame_0 + refresh provenance
    console = TaskConsole(hub=hub, state=default_console_state(), session=exp,
                          running_nodes=[cam], window_px=(900, 600))
    console._timer.stop()
    console.refresh_once = lambda: None
    # The panel is WIRED to cam_frame_0 (so Save resolves the camera node); the plotter kind is
    # incidental to provenance -- a hist over a flat sample keeps the fixture simple.
    card = console._new_panel_card(PanelConfig(kind="hist", title="frame", source="value = cam_frame_0",
                                               inputs=["cam_frame_0"], row=GAP, col=GAP, size="2x4"))
    console._attach_card(card)
    card.plotter = panel_plot(np.random.default_rng(0).normal(300.0, 20.0, 500),
                              kind="hist", size="2x4", bins=40, title="frame")
    return console, cam, card


def test_console_save_writes_producing_node_provenance(tmp_path):
    """The console Save resolves the node PRODUCING the panel's signal and stores its device snapshot in
    the npz.  Reopened with ``load_figure``, ``info['provenance']`` carries the camera snapshot + the
    acquisition params -- so a saved figure records the device state it was taken under."""
    ensure_qt_app()
    from Zou_lab_control.frontend.task_console import PanelEditor

    exp = na.connect("virtual")
    console = None
    try:
        console, cam, card = _camera_console(exp)
        editor = PanelEditor(card, console)
        editor.rebuild()
        # the console-side resolution: the panel's signal -> its producing node -> the node's snapshot
        prov = editor._provenance_for_save()
        assert prov is not None and prov["devices"]["camera"]["type"] == type(exp.devices.camera).__name__

        # drive the REAL Save into tmp (auto-name off = verbatim path) and read the npz back
        editor.save_dir_edit.setText(str(tmp_path / "frame"))
        editor.save_autoname.setChecked(False)
        editor.save()
        npz = next(tmp_path.glob("*.npz"), None)
        assert npz is not None, f"the console Save must write an .npz (status: {editor.status.text()})"

        saved = load_figure(npz)
        stored = saved.info.get("provenance")
        assert isinstance(stored, dict), "info['provenance'] round-trips through the npz"
        assert stored["devices"]["camera"]["type"] == type(exp.devices.camera).__name__
        assert "exposure" in stored["devices"]["camera"]
        assert "acquisition_parameters" in stored and "exposure" in stored["acquisition_parameters"]
    finally:
        if console is not None:
            console.shutdown()
        exp.close()
        plt.close("all")


def test_save_falls_back_to_session_devices_when_no_producing_node(tmp_path):
    """A panel wired to a signal NO node publishes (a derived/static value) still records provenance: the
    Save falls back to the whole-session device snapshot when a session is attached."""
    ensure_qt_app()
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelConfig, PanelEditor, GAP)
    from Zou_lab_control.frontend import panel_plot

    exp = na.connect("virtual")
    console = None
    try:
        hub = SignalHub()
        console = TaskConsole(hub=hub, state=default_console_state(), session=exp,
                              running_nodes=[], window_px=(900, 600))
        console._timer.stop()
        console.refresh_once = lambda: None
        card = console._new_panel_card(PanelConfig(kind="hist", title="dis", source="mystery",
                                                   inputs=["mystery_signal"], row=GAP, col=GAP, size="2x4"))
        console._attach_card(card)
        card.plotter = panel_plot(np.random.default_rng(0).normal(300.0, 20.0, 500),
                                  kind="hist", size="2x4", bins=40, title="dis")
        editor = PanelEditor(card, console)
        editor.rebuild()
        prov = editor._provenance_for_save()
        # no producing node -> the session's device snapshot (keyed by device role name)
        assert isinstance(prov, dict) and "camera" in prov, \
            "with no producing node, provenance falls back to the session device snapshot"
    finally:
        if console is not None:
            console.shutdown()
        exp.close()
        plt.close("all")


# ----------------------------------------------------------------- viewer: expand + backward compat
def test_viewer_expands_provenance_and_tolerates_its_absence(tmp_path):
    """The FigureViewer's Info column expands a stored ``provenance`` into readable rows, and an OLD npz
    with none loads + shows nothing extra (no crash) -- backward compatible."""
    ensure_qt_app()
    from Zou_lab_control.frontend import plot
    from Zou_lab_control.frontend.figure_viewer import show_figure_viewer

    # a save WITH provenance (nested devices dict, exactly the node snapshot shape)
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(300.0, 20.0, 400), rng.normal(460.0, 20.0, 300)])
    p = plot(vals, kind="hist", display=False, update="once", data_figure=True)
    df = p.to_data_figure()
    prov = {"node": "camera", "layer": "measurement", "captured_at": "2026-06-30 12:00:00",
            "devices": {"camera": {"type": "VirtualCamera", "exposure": 0.003, "roi": None}},
            "acquisition_parameters": {"exposure": 0.003, "frames_per_cycle": 1}}
    out = df.save(str(tmp_path / "withprov"),
                  extra_info={"source": "counts", "kind": "hist", "provenance": prov})
    plt.close(p.fig)

    v = show_figure_viewer(Path(out["data"]))
    try:
        # the raw-info field shows the whole dict, including the nested provenance
        assert "provenance" in v.raw_info.text() and "VirtualCamera" in v.raw_info.text()
        # the Info column expanded it into rows/sections (more rows than the flat facts alone)
        assert v.info_layout.count() > 0
        texts = _info_column_texts(v)
        assert any("VirtualCamera" in t for t in texts), "the camera device state is shown in the Info column"
    finally:
        win = v.window()
        if win is not None:
            win.close(); win.deleteLater()
        v.teardown()
        plt.close("all")

    # an OLD npz with NO provenance: loads and expands nothing, no crash
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, data_x=x, data_y=np.sin(x * 6),
             info={"labels": ["X", "Y", "Z"], "name": "legacy", "kind": "1d"})
    v2 = show_figure_viewer(legacy)
    try:
        texts = _info_column_texts(v2)
        assert not any("Provenance" == t for t in texts), "an old npz shows no Provenance section"
    finally:
        win = v2.window()
        if win is not None:
            win.close(); win.deleteLater()
        v2.teardown()
        plt.close("all")


def _info_column_texts(viewer) -> list[str]:
    """Every visible text in the Info column (section labels + row labels + read-only field values)."""
    from PyQt5 import QtWidgets

    out: list[str] = []
    for w in viewer.info_layout.parentWidget().findChildren(QtWidgets.QWidget):
        getter = getattr(w, "text", None)
        if callable(getter):
            try:
                t = getter()
            except Exception:
                continue
            if isinstance(t, str) and t:
                out.append(t)
    return out
