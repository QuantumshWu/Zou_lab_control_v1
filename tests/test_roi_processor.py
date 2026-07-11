"""ROI processor: reduce a region of a live block through the stock func-node contract, GENERIC over
plot kind, driven by a per-panel REGION signal.

The user assembles "select a region -> watch its distribution / total counts" from stock parts
(measurement -> ROI processor -> dis / monitor panels).  These tests pin the data contract: the ROI
consumes its source block PLUS a ``<slug>_region`` signal carrying a plot-independent
:class:`Selection` (+ its axis binding) -- NOT pixel endpoints -- so an image rectangle, a 1-D x-range,
a distribution value-range and a site-centre rectangle all collapse to ONE finite ``roi_value`` with
zero per-kind branch, plus a native crop / NaN-passthrough ``roi_frame``.  A re-drag is just a
republish of the region.  All through the same Processor path the real console uses (virtual == real)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.selection import (
    Selection, axis_crop_binding, decode_region, encode_region, region_tensor,
    scatter_mask_binding, value_mask_binding)
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.core.signal_tensor import SignalSchema
from Zou_lab_control.neutral_atom.operations.processors.roi import ROI_REDUCERS, RoiProcessor


def _publish_region(hub, name, selection, *, bins=None):
    """Publish a per-panel region signal the console produces from a drag -- region_tensor is the ONE
    publisher payload (byte-stable schema, so a republish never needs replace)."""
    tensor = region_tensor(selection, bins=bins)
    hub.register_signal(name, tensor.schema)
    hub.publish({name: tensor})
    return name


def _hub_with(name, block):
    hub = SignalHub()
    hub.register_signal(name, SignalSchema(
        point_shape=(block.shape[1],), data_shape=block.shape[2:], dtype=block.dtype,
        repeat_capacity=block.shape[0]))
    hub.publish({name: block})
    return hub


def _frame_block(h=40, w=60, repeats=1, dtype=np.uint8):
    rng = np.random.default_rng(0)
    return rng.integers(5, 11, size=(repeats, 1, h, w), dtype=dtype)


def _image_selection(x0, x1, y0, y1):
    """A 2-D image rectangle in the frame's own pixels: the binding the console's ``region_binding``
    produces for a ``2d`` panel (x -> the last data axis, y -> the 2nd-last, pixel step 1)."""
    return Selection.rectangle(
        x0, x1, y0, y1,
        metadata={"binding": axis_crop_binding([("x", -1, 0.0, 1.0), ("y", -2, 0.0, 1.0)])})


def _region_roi(hub, source, selection, *, reduce="mean", bins=None):
    """Publish ``<source>_region`` from ``selection`` and build a RoiProcessor consuming (source, region)."""
    region = _publish_region(hub, f"{source}_region", selection, bins=bins)
    return RoiProcessor(hub, source_expr={"inputs": [source], "source": "value = signal"},
                        region=region, reduce=reduce)


def test_roi_crop_is_native_dtype_and_value_matches_reduce():
    block = _frame_block()
    block[0, 0, 10:20, 30:40] = 100                       # a bright patch inside the region
    hub = _hub_with("frame_0", block)
    node = _region_roi(hub, "frame_0", _image_selection(30, 40, 10, 20), reduce="mean")
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
    hub = _hub_with("frame_0", block)
    node = RoiProcessor(hub)                               # no region signal = the whole frame
    hub.publish({"frame_0": block})    # react to a frame published AFTER the node subscribes
    node.step()
    assert np.array_equal(hub.latest("roi_frame"), block)
    expected = np.mean(block.astype(float), axis=(-2, -1))[..., None]
    np.testing.assert_allclose(hub.latest("roi_value"), expected)


def test_roi_region_round_trips_with_the_plot_selector():
    hub = _hub_with("frame_0", _frame_block())
    # a rectangle dragged on the roi_frame panel is a Selection with the image axis-binding; the console
    # publishes it as the region signal, and the node declares the SAME region back (real source pixels).
    node = _region_roi(hub, "frame_0", _image_selection(30.2, 39.8, 10.1, 19.9))
    hub.publish({"frame_0": _frame_block()})
    node.step()
    assert node.acquisition_parameters()["region"] == [30, 40, 10, 20]


def test_roi_reduces_a_1d_vector_region():
    """1-D ROI: the array within the boxed x-range, reduced to one scalar (issue #4 -- '1d 的 roi 就是
    框范围内 data 的 array').  A contiguous index crop yields a native fixed-shape sub-array + a finite
    reduce."""
    vec = np.arange(20, dtype=np.float64).reshape(1, 1, 20)      # one point, a 20-sample vector
    hub = _hub_with("y", vec)
    sel = Selection.x(5, 9, metadata={"binding": axis_crop_binding([("x", -1, 0.0, 1.0)])})
    node = _region_roi(hub, "y", sel, reduce="sum")
    hub.publish({"y": vec})
    node.step()
    assert hub.latest("roi_frame").shape == (1, 1, 4)           # indices 5,6,7,8 (half-open, pixel rule)
    value = hub.latest("roi_value")
    assert value.shape == (1, 1, 1) and np.isfinite(value[0, 0, 0])
    assert value[0, 0, 0] == float(sum(range(5, 9)))


def test_roi_reduces_a_distribution_value_range():
    """hist ROI: the samples whose VALUE is in range (issue #4 -- 'hist 同理 count-range 内的样本').  No
    fixed sub-frame -> roi_frame is the stable NaN passthrough; roi_value is the finite reduce."""
    samples = np.array([[[1.0, 2.0, 3.0, 10.0, 11.0, 12.0]]])
    hub = _hub_with("s", samples)
    sel = Selection.value(9.0, 13.0, metadata={"binding": value_mask_binding()})
    node = _region_roi(hub, "s", sel, reduce="mean")
    hub.publish({"s": samples})
    node.step()
    assert hub.latest("roi_frame").shape == samples.shape       # stable shape (NaN outside the range)
    value = hub.latest("roi_value")
    assert np.isfinite(value[0, 0, 0]) and abs(value[0, 0, 0] - 11.0) < 1e-9


def test_roi_reduces_a_site_centre_rectangle():
    """sites ROI: the per-site values whose CENTRE falls in the rectangle, reduced to one scalar (a
    scatter mask over the (N,) data axis)."""
    occ = np.array([[[0.0, 1.0, 0.0, 1.0, 0.0]]])              # occupancy of 5 sites
    cx = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    cy = np.zeros(5)
    hub = _hub_with("occ", occ)
    sel = Selection.rectangle(5, 35, -5, 5,
                              metadata={"binding": scatter_mask_binding(-1, {"x": cx, "y": cy})})
    node = _region_roi(hub, "occ", sel, reduce="sum")
    hub.publish({"occ": occ})
    node.step()
    value = hub.latest("roi_value")
    # sites 1,2,3 (centres 10,20,30) are inside [5,35]; their occupancy is 1,0,1 -> sum 2
    assert np.isfinite(value[0, 0, 0]) and value[0, 0, 0] == 2.0


def test_roi_re_drag_republishes_the_region_and_retargets():
    """A re-drag is a REPUBLISH of the region signal (no set_selection plumbing); the reactive node
    re-computes on the next frame with the new region."""
    block = _frame_block()
    block[0, 0, 10:20, 30:40] = 100
    hub = _hub_with("frame_0", block)
    node = _region_roi(hub, "frame_0", _image_selection(30, 40, 10, 20))
    hub.publish({"frame_0": block}); node.step()
    assert hub.latest("roi_frame").shape == (1, 1, 10, 10)
    _publish_region(hub, "frame_0_region", _image_selection(0, 60, 0, 40))   # re-drag: whole frame
    hub.publish({"frame_0": block}); node.step()
    assert hub.latest("roi_frame").shape == (1, 1, 40, 60)


def test_roi_region_signal_round_trips_the_selection():
    """The region persists as a serializable signal (save/load + Edit reopen): encode -> decode gives
    back the identical Selection, and a fresh node consuming it reproduces the crop + its pixel region."""
    hub = _hub_with("frame_0", _frame_block())
    sel = _image_selection(30, 40, 10, 20)
    restored = decode_region(encode_region(sel))
    assert restored.bounds("x") == (30.0, 40.0) and restored.bounds("y") == (10.0, 20.0)
    assert restored.metadata["binding"]["mode"] == "axis-crop"
    node = _region_roi(hub, "frame_0", sel, reduce="sum")
    assert node.acquisition_parameters()["reduce"] == "sum"
    hub.publish({"frame_0": _frame_block()})
    node.step()
    assert node.acquisition_parameters()["region"] == [30, 40, 10, 20]
    assert node.region_name == "frame_0_region" and "frame_0_region" in node.consumes


def test_analysis_registered_in_processor_catalog():
    from Zou_lab_control.neutral_atom.operations.processor_registry import discovered_processor_specs

    class _Readout:                                        # the factory only needs an object handle
        current = None

    names = {spec.name for spec in discovered_processor_specs(_Readout())}
    assert "Analysis" in names
    assert "ROI crop" not in names and "Frame fit" not in names
