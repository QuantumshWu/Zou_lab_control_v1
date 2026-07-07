"""#issue-1 + #issue-2 end-to-end (no headless workaround): drive the REAL TaskConsole through three
Stop/Start cycles with the imaging pulse actually FIRED, and assert
  (1) the occupancy signal name is STABLE across cycles -- never reprefixed to judge_occupancy_2_occupied,
      i.e. panels stay bound after Stop/Start (the unbound bug), AND
  (2) the LIVE occupied classification, measured per-frame against the SAME frame's ground truth, equals
      the calibration fidelity (~0.84 box @2ms) -- NOT the ~0.5 chance the user saw when the readout ran at
      a different exposure than the calibration (the exposure-mismatch bug, now fixed by the camera-exposure
      pin + detect threshold_exposure).

The virtual camera re-randomises the loading every frame, so truth MUST be read in lockstep with the judged
frame (read trap.occupancy right after the camera step, before the next frame) -- comparing across frames
gives a spurious ~0.5.  Routes through the same TaskConsole + calibration.detect the GUI uses.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")


def test_console_three_startstop_cycles_stable_name_and_live_accuracy():
    import Zou_lab_control.neutral_atom as na
    from tests.conftest import fire_live_imaging
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    exp = na.connect("virtual")
    con = None
    try:
        trap = exp.devices.trap_array
        exp.devices.camera.exposure = 0.020                      # stale default != readout gate
        task = exp.readout.calibrate_task(SignalHub(), threshold_method="otsu",
                                          threshold_frames=200, readout_exposure=0.002)
        task.run_to_completion()
        exp.readout._session.calibration_data = task.calibration
        assert exp.devices.camera.exposure == pytest.approx(0.002)   # #2 pin: live readout self-matches cal

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

        def cycle():
            con._start_logic_node(camrow); con._start_logic_node(judrow)
            fire_live_imaging(exp, exposure=0.002)
            camN = con._logic_nodes[id(camrow)]; judN = con._logic_nodes[id(judrow)]
            accs = []; name = None
            for _ in range(30):
                camN.step()                                   # one frame -> trap.occupancy is ITS truth
                truth = np.asarray(trap.occupancy, dtype=bool).reshape(-1)
                judN.step()                                   # judge THAT frame
                names = [n for n in con.hub.signal_versions() if n.endswith("occupied")]
                if names:
                    name = names[0]
                    pred = np.asarray(con.hub.latest(name), dtype=bool).reshape(-1)
                    if pred.shape == truth.shape:
                        accs.append(float((pred == truth).mean()))
            con._stop_logic_node(judrow); con._stop_logic_node(camrow)
            return name, (float(np.mean(accs)) if accs else float("nan"))

        results = [cycle(), cycle(), cycle()]
        names = {n for n, _ in results}
        accs = [a for _, a in results]
        assert names == {"occupied"}, f"signal name must stay 'occupied' across Stop/Start, got {names}"
        assert all(a >= 0.78 for a in accs), f"live occupied accuracy must equal the calibration (~0.84), got {accs}"
    finally:
        if con is not None:
            con.shutdown()
        exp.close()
