"""Processor outputs obey their complete per-signal schemas."""

from __future__ import annotations

from conftest import raw_device_set

import numpy as np

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.frontend.live import reduce_repeat
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, OccupancyProcessor

from conftest import fire_live_imaging


def test_occupancy_preserves_every_physical_repeat_and_point_cell():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    exp.readout.sitemap(frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    hub = SignalHub()
    camera = CameraMeasurement(
        hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer, repeat=5)
    processor = OccupancyProcessor(
        hub,
        calibration=exp.readout.current,
        source_expr={"inputs": ["frame_0"], "source": "value = signal"},
        method="box",
        grid_shape=(3, 4),
    )
    fire_live_imaging(exp)
    try:
        for _ in range(6):
            camera.step()
            processor.step()

        occupied = hub.latest_tensor("occupied")
        n_sites = occupied.schema.data_shape[0]
        assert occupied.schema.point_shape == (1,)
        assert occupied.data.shape == (5, 1, n_sites)
        assert occupied.valid.shape == (5, 1)

        for name in ("occupied", "counts"):
            tensor = hub.latest_tensor(name)
            assert tensor.data.shape == (5, 1, n_sites)
            assert tensor.schema.data_shape == (n_sites,)
        frame = hub.latest_tensor("frame_judged")
        assert frame.data.ndim == 4 and frame.data.shape[:2] == (5, 1)
        assert frame.schema.data_shape == frame.data.shape[2:]

        assert hub.latest("centers").shape == (1, 1, n_sites, 2)
        assert hub.schema("centers").data_shape == (n_sites, 2)
        assert hub.latest("thresholds").shape == (1, 1, n_sites)
        assert hub.latest("rate").shape == (5, 1, 1)

        probability = np.squeeze(
            reduce_repeat(occupied.data, "average", valid=occupied.valid))
        assert probability.shape == (n_sites,)
        assert np.all((probability >= 0) & (probability <= 1))
    finally:
        exp.close()
