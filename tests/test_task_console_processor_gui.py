"""Contract: a processor is a LOGIC NODE that publishes to the hub end-to-end.

A processor is a logic node (Logic tab), added STOPPED, with an auto-generated Edit
form from the spec's ParamDecls.  Two execution styles share the model:

* ONE-SHOT (``Readout fidelity``): Start runs the action once on its own thread (a
  ``ProcessorRun``) over a saved frames folder and self-stops.
* REACTIVE (``Judge occupancy``): Start launches a live node that consumes each
  ``frame`` signal and republishes per-site occupancy via the real
  ``calibration.detect`` -- it keeps running until Stop.

Either way display is suppressed (no auto plot); you add a Plot panel on the Monitor
board pointed at a published signal to view it.

Offscreen Qt + the virtual==real path (na.write_virtual_run / a virtual session); no
demo GUI fixtures.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _add_processor_node(console, name):
    kc = console.kind_combo
    idx = next(j for j in range(kc.count()) if kc.itemData(j) == ("processor", name))
    assert kc.itemText(idx).startswith("Processor:")
    kc.setCurrentIndex(idx)
    console._add_panel()
    row = console.logic_nodes[-1]
    return row, console._logic_editors[id(row)]


def test_processor_node_runs_and_publishes(tmp_path):
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    data_dir = tmp_path / "run"
    na.write_virtual_run(str(data_dir), prefix="img", groups=40, shots_per_group=4,
                         short_shot=3, ref_shots=(1, 2, 4), grid_shape=(4, 5),
                         loading_probability=0.5, seed=4)
    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5)})
    exp.readout.sitemap_from_dir(str(data_dir), prefix="img", method="psf")

    console = TaskConsole(hub=SignalHub(), state=default_console_state(),
                          processors=exp.readout.processor_specs(), session=exp)
    console._timer.stop()
    try:
        row, editor = _add_processor_node(console, "Readout fidelity")
        assert row.node.kind == "processor"
        # the Edit form opened, auto-generated incl. the 'text' folder-path field
        assert editor.form is not None
        assert editor.form._widgets["data_dir"][0] == "text"
        # default STOPPED: no node built, nothing on the hub
        assert console._logic_nodes[id(row)] is None

        # fill the folder + Start: the action runs ONCE on its own thread, publishes
        editor.form._widgets["data_dir"][1].setText(str(data_dir))
        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 8.0
        while not getattr(node, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert node.finished
        assert float(console.hub.latest("processor_done")) == 1.0
        assert np.asarray(console.hub.latest("fidelity_site")).shape == (20,)
        # display suppressed: running the processor created NO plot panel
        assert console.cards == []
    finally:
        console.shutdown()
        exp.close()


def test_judge_occupancy_node_reacts_to_frames(tmp_path):
    """The REACTIVE 'Judge occupancy' processor consumes each live ``frame`` and
    republishes per-site occupancy/centers via the REAL ``calibration.detect`` (no
    folder, no one-shot).  Its calibration param is an EXPLICIT saved file (defaulting to
    the canonical calibration.json the Calibrate task writes); a Plot panel can then read
    its published signals."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5), "image_shape": (40, 50)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=16, display=False)

    hub = SignalHub()
    console = TaskConsole(hub=hub, state=default_console_state(),
                          processors=exp.readout.processor_specs(), session=exp)
    console._timer.stop()
    try:
        row, editor = _add_processor_node(console, "Judge occupancy")
        assert row.node.kind == "processor"
        # default STOPPED + defaults PREFILLED from the spec (never blank-to-guess): the
        # calibration field shows the canonical calibration.json path (NOT a blank mystery),
        # source shows its declared default.  There is no invented `ema` smoothing param.
        assert console._logic_nodes[id(row)] is None
        vals = editor.collect_values()
        assert vals["calibration"].replace("\\", "/").endswith("calibrations/calibration.json")
        assert vals["source"] == "frame"
        assert "ema" not in vals

        # point the detector at a REAL saved calibration file (the reference's explicit-file
        # model) -- the canonical default path may not exist in a fresh checkout.
        cal_file = tmp_path / "cal.json"
        exp.readout.save(str(cal_file))
        editor.form._widgets["calibration"][1].setText(str(cal_file))

        # a camera measurement publishes `frame`; the occupancy processor reacts to it
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        cam.step()                                  # first frame on the hub
        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        # a console-built occupancy node carries a per-INSTANCE prefix (#2: multiple occupancy
        # judges publish DISTINCT signals), so read its published names off the node, never a
        # hard-coded bare name.
        assert node.prefix                          # namespaced, not the bare (colliding) form
        occ_name, cen_name = node.prefix + "occupied", node.prefix + "centers"
        deadline = time.monotonic() + 8.0
        while occ_name not in hub.names() and time.monotonic() < deadline:
            cam.step()                              # keep frames coming for the reactive node
            time.sleep(0.03)
        assert occ_name in hub.names()
        assert np.asarray(hub.latest(occ_name)).shape == (20,)
        assert np.asarray(hub.latest(cen_name)).shape == (20, 2)
        # reactive: it keeps running (NOT a one-shot finished node)
        assert getattr(node, "finished", False) is False
        cam.stop()
    finally:
        console.shutdown()
        exp.close()
