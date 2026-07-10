"""#issue-1 + #issue-2 end-to-end (no headless workaround): drive the REAL TaskConsole through three
Stop/Start cycles with the imaging pulse actually FIRED, and assert
  (1) the occupancy signal name is STABLE across cycles -- never reprefixed to judge_occupancy_2_occupied,
      i.e. panels stay bound after Stop/Start (the unbound bug), AND
  (2) the LIVE occupied classification, measured per-frame against the SAME frame's ground truth, equals
      the calibration fidelity (~0.84 box @2ms) -- NOT the ~0.5 chance the user saw when the readout ran at
      a different exposure than the calibration (the exposure-mismatch bug, now fixed by the camera-exposure
      pin + detect the typed calibration exposure).

The virtual camera re-randomises the loading every frame, so truth MUST be read in lockstep with the judged
frame (read trap.occupancy right after the camera step, before the next frame) -- comparing across frames
gives a spurious ~0.5.  Routes through the same TaskConsole + calibration.detect the GUI uses.
"""

from __future__ import annotations

import os
import time
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
            deadline = time.monotonic() + 3.0
            names = []
            while not names and time.monotonic() < deadline:
                names = [n for n in con.hub.signal_versions() if n.endswith("occupied")]
                time.sleep(0.01)
            con._stop_logic_node(judrow); con._stop_logic_node(camrow)
            return names[0] if names else None

        names = {cycle(), cycle(), cycle()}
        assert names == {"occupied"}, f"signal name must stay 'occupied' across Stop/Start, got {names}"

        # Accuracy is a separate synchronous transaction.  Calling ``step`` on a node
        # while its console-owned background thread is also stepping it races two frames
        # and compares one prediction with another frame's truth.
        from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, OccupancyProcessor
        hub = SignalHub()
        camera = CameraMeasurement(
            hub, exp.devices.camera, sequencer=exp.devices.sequencer, repeat=0)
        judge = OccupancyProcessor(
            hub, calibration=task.calibration,
            source_expr={"inputs": ["frame_0"], "source": "value = signal"},
            method="box")
        fire_live_imaging(exp, exposure=0.002)
        accs = []
        for _ in range(90):
            camera.step()
            truth = np.asarray(trap.occupancy, dtype=bool).reshape(-1)
            judge.step()
            pred = np.asarray(hub.latest("occupied"), dtype=bool).reshape(-1)
            accs.append(float((pred == truth).mean()))
        accuracy = float(np.mean(accs))
        assert accuracy >= 0.78, (
            f"live occupied accuracy must equal the calibration (~0.84), got {accuracy}")
    finally:
        if con is not None:
            con.shutdown()
        exp.close()
