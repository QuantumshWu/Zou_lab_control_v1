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
from Zou_lab_control.neutral_atom.operations.processors.roi import ROI_REDUCERS, RoiProcessor


def _hub_with_frame(block):
    hub = SignalHub()
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
    out = node.transform({"frame_0": hub.latest("frame_0")})
    assert out["roi_frame"].shape == (10, 10)
    assert out["roi_frame"].dtype == block.dtype           # native passthrough, no float64 balloon
    assert np.array_equal(out["roi_frame"], block[0, 0, 10:20, 30:40])
    assert out["roi_value"] == 100.0                       # the patch fills the whole region
    # every declared reduce verb dispatches through the ONE table
    for verb, fn in ROI_REDUCERS.items():
        node.set_acquisition_parameters(reduce=verb)
        out = node.transform({"frame_0": hub.latest("frame_0")})
        assert out["roi_value"] == float(fn(block[0, 0, 10:20, 30:40].astype(float)))


def test_roi_defaults_to_full_frame_and_reduces_repeats():
    block = _frame_block(repeats=3)
    hub = _hub_with_frame(block)
    node = RoiProcessor(hub)                               # all-zero region = the whole frame
    out = node.transform({"frame_0": hub.latest("frame_0")})
    assert out["roi_frame"].shape == block.shape[-2:]      # full frame
    assert np.array_equal(out["roi_frame"], block[-1, 0])  # the NEWEST repeat slice
    expected = float(np.mean([block[r, 0].mean() for r in range(3)]))
    assert out["roi_value"] == expected                    # per-slice reduce, then average


def test_roi_region_round_trips_with_the_plot_selector():
    hub = _hub_with_frame(_frame_block())
    node = RoiProcessor(hub)
    # the 2-D panel's rectangle selector hands four endpoints in frame pixels; they map 1:1
    params = node.region_to_acquisition_parameters(30.2, 39.8, 10.1, 19.9)
    node.set_acquisition_parameters(**params)
    assert (node.x_min, node.x_max, node.y_min, node.y_max) == (30.2, 39.8, 10.1, 19.9)
    node.transform({"frame_0": hub.latest("frame_0")})     # first frame teaches the shape
    # the declared spatial region (real source pixels) drives the roi_frame panel's axes
    region = node.acquisition_parameters()["region"]
    assert region == [30, 40, 10, 20]


def test_roi_registered_in_processor_catalog():
    from Zou_lab_control.neutral_atom.operations.processor_registry import discovered_processor_specs

    class _Readout:                                        # the factory only needs an object handle
        current = None

    names = {spec.name for spec in discovered_processor_specs(_Readout())}
    assert "ROI crop" in names
