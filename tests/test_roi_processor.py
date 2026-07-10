"""ROI processor: crop + reduce a live frame block through the stock func-node contract.

The user assembles "select a region -> watch its distribution / total counts" from stock parts
(camera measurement -> ROI processor -> dis / monitor panels).  These tests pin the data contract:
native-dtype crop, per-acquisition scalar, the region round-trip with the plot selector, and the
registry entry -- all through the same Processor path the real console uses (virtual == real)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.core.signal_tensor import SignalSchema
from Zou_lab_control.neutral_atom.operations.processors.roi import ROI_REDUCERS, RoiProcessor


def _hub_with_frame(block):
    hub = SignalHub()
    hub.register_signal("frame_0", SignalSchema(
        point_shape=(block.shape[1],), data_shape=block.shape[2:], dtype=block.dtype,
        repeat_capacity=block.shape[0]))
    hub.publish({"frame_0": block})
    return hub


def _frame_block(h=40, w=60, repeats=1, dtype=np.uint8):
    rng = np.random.default_rng(0)
    return rng.integers(5, 11, size=(repeats, 1, h, w), dtype=dtype)


def test_roi_crop_is_native_dtype_and_value_matches_reduce():
    block = _frame_block()
    block[0, 0, 10:20, 30:40] = 100                       # a bright patch inside the region
    hub = _hub_with_frame(block)
    node = RoiProcessor(hub, x_min=30, x_max=40, y_min=10, y_max=20, reduce="mean")
    hub.publish({"frame_0": block})    # react to a frame published AFTER the node subscribes
    node.step()
    crop, value = hub.latest("roi_frame"), hub.latest("roi_value")
    assert crop.shape == (1, 1, 10, 10)
    assert crop.dtype == block.dtype
    assert np.array_equal(crop[0, 0], block[0, 0, 10:20, 30:40])
    assert value[0, 0, 0] == 100.0
    # every declared reduce verb dispatches through the ONE table
    for verb, fn in ROI_REDUCERS.items():
        node.set_acquisition_parameters(reduce=verb)
        hub.publish({"frame_0": block})
        node.step()
        assert hub.latest("roi_value")[0, 0, 0] == float(
            fn(block[0, 0, 10:20, 30:40].astype(float)))


def test_roi_defaults_to_full_frame_and_reduces_repeats():
    block = _frame_block(repeats=3)
    hub = _hub_with_frame(block)
    node = RoiProcessor(hub)                               # all-zero region = the whole frame
    hub.publish({"frame_0": block})    # react to a frame published AFTER the node subscribes
    node.step()
    assert np.array_equal(hub.latest("roi_frame"), block)
    expected = np.mean(block.astype(float), axis=(-2, -1))[..., None]
    np.testing.assert_allclose(hub.latest("roi_value"), expected)


def test_roi_region_round_trips_with_the_plot_selector():
    hub = _hub_with_frame(_frame_block())
    node = RoiProcessor(hub)
    # the 2-D panel's rectangle selector hands four endpoints in frame pixels; they map 1:1
    params = node.region_to_acquisition_parameters(30.2, 39.8, 10.1, 19.9)
    node.set_acquisition_parameters(**params)
    assert (node.x_min, node.x_max, node.y_min, node.y_max) == (30.2, 39.8, 10.1, 19.9)
    node.step()
    # the declared spatial region (real source pixels) drives the roi_frame panel's axes
    region = node.acquisition_parameters()["region"]
    assert region == [30, 40, 10, 20]


def test_roi_registered_in_processor_catalog():
    from Zou_lab_control.neutral_atom.operations.processor_registry import discovered_processor_specs

    class _Readout:                                        # the factory only needs an object handle
        current = None

    names = {spec.name for spec in discovered_processor_specs(_Readout())}
    assert "ROI crop" in names
