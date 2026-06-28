"""#shot-clock: SOURCE-shot provenance contract.

Every value the hub stores carries the source-shot id it derives from: an acquiring node mints a fresh id
per real shot; a reactive processor INHERITS the id of the input it consumed.  This is what lets the
console assemble one coherent cross-producer display shot (the sitemap underlay == the 2D image ==
occupancy), instead of mixing the latest of each independently-threaded producer.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")


def test_hub_provenance_primitives_and_coherent_read():
    from Zou_lab_control.neutral_atom.core.signals import SignalHub, NO_LINEAGE
    h = SignalHub()
    assert h.latest_provenance("missing") == NO_LINEAGE          # soft-fail, never raises
    s1, s2 = h.next_source_shot(), h.next_source_shot()
    assert (s1, s2) == (1, 2)                                     # strictly monotonic, distinct
    h.publish({"frame": np.full((2, 2), 1.0)}, provenance=s1)
    h.publish({"derived": np.array([1.0])}, provenance=s1)        # a consumer inheriting s1
    h.publish({"rate": 7.0})                                      # free-running scalar -> NO_LINEAGE
    h.publish({"frame": np.full((2, 2), 2.0)}, provenance=s2)     # frame advances; derived lags at s1
    assert h.latest_provenance("frame") == s2
    assert h.latest_provenance("derived") == s1
    assert h.latest_provenance("rate") == NO_LINEAGE
    assert h.provenance_map() == {"frame": s2, "derived": s1, "rate": NO_LINEAGE}
    # coherent read at s1: the ahead frame is held back to s1, the scalar shows latest
    snap = h.snapshot_at(s1)
    assert snap["frame"][0, 0] == 1.0 and snap["derived"][0] == 1.0 and snap["rate"] == 7.0
    assert h.snapshot_at(None)["frame"][0, 0] == 2.0             # None == latest of each
    # clear() / remove_signals() drop the parallel provenance store in lockstep
    h.remove_signals(["frame"])
    assert "frame" not in h.provenance_map()
    h.clear()
    assert h.provenance_map() == {}


def test_camera_mints_and_occupancy_inherits():
    import Zou_lab_control.neutral_atom as na
    from tests.conftest import fire_live_imaging
    from Zou_lab_control.neutral_atom.core.signals import SignalHub, NO_LINEAGE
    from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor
    exp = na.connect("virtual")
    try:
        exp.readout.sitemap(method="box", frames=6, display=False)
        exp.readout.thresholds(frames=40, display=False)
        hub = SignalHub()
        cam = exp.readout.camera_measurement(hub, prefix="")
        occ = OccupancyProcessor(hub, calibration=exp.readout.current,
                                 session_calibration=lambda: exp.readout.current,
                                 source_expr={"inputs": ["frame"], "source": "value = signal"},
                                 method="box", grid_shape=(5, 7), prefix="")
        # two camera shots mint DISTINCT ids
        fire_live_imaging(exp, exposure=exp.devices.camera.exposure); cam.step()
        first = hub.latest_provenance("frame")
        fire_live_imaging(exp, exposure=exp.devices.camera.exposure); cam.step()
        second = hub.latest_provenance("frame")
        assert second > first > 0
        # the occupancy DERIVED signals inherit the SAME id as the frame they were judged from
        occ.step()
        fid = hub.latest_provenance("frame")
        assert hub.latest_provenance("occupied") == fid
        assert hub.latest_provenance("frame_judged") == fid
        # camera ahead -> the coherent read at the occupancy's shot holds the ahead frame back
        fire_live_imaging(exp, exposure=exp.devices.camera.exposure); cam.step()
        t = hub.latest_provenance("occupied")
        assert hub.latest_provenance("frame") > t                # camera is ahead
        held = np.squeeze(np.asarray(hub.snapshot_at(t)["frame"]))
        latest = np.squeeze(np.asarray(hub.snapshot_at(None)["frame"]))
        assert not np.array_equal(held, latest)                  # ahead frame held back at the coherent shot
    finally:
        exp.close()
