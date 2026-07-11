"""#4: a SMALL non-empty count-band hist ROI -> a 2-D ``roi_frame`` panel that renders its IN-BAND
pixels, NOT a blank frame.

The value-mask ``roi_frame`` is the NaN passthrough of the source frame with out-of-band pixels set to
NaN.  On a LARGE frame the side-distribution / colour-limit reads an even-STRIDED sample (a perf cap);
a NARROW band leaves only a FEW finite pixels, and the stride can miss them ALL -> the sample is
all-NaN -> the colour limit collapses to NaN -> the image renders BLANK.  The fix
(``Live2DDis._distribution_values`` falls back to the FULL frame's finite pixels when the strided
sample has none) makes the clim come from the in-band pixels, so a non-empty roi_frame shows its pixels
(out-of-band = the cmap's bad colour, i.e. the axes background).  Verified on a REAL imshow.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.frontend.live import Live2DDis, _image_axes_px_budget, panel_plot_spec
from Zou_lab_control.neutral_atom.core.raster import RegularRaster
from Zou_lab_control.neutral_atom.core.selection import Selection, encode_region, value_mask_binding
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.core.signal_tensor import SignalSchema, SignalTensor
from Zou_lab_control.neutral_atom.operations.processors.roi import RoiProcessor

# A frame just ABOVE the display sample cap (Live2DDis._DIST_SAMPLE_CAP = 200_000) so the colour-limit
# path takes the even-stride sample -- the regime where a sparse band can be missed.
H, W = 420, 500          # 210_000 pixels -> stride factor 2 (samples the EVEN flat indices)
_LO, _HI = 595.0, 605.0  # a narrow in-band window


def _sparse_in_band_source():
    """A source frame whose ONLY in-band pixels sit at ODD flat indices -- so the even-stride display
    sample of the value-mask roi_frame misses them all (the #4 all-NaN-sample regime), deterministically."""
    frame = np.full(H * W, 100.0, dtype=np.float64)          # everything OUT of [595, 605]
    odd = np.arange(1, H * W, 2)[:80]                        # 80 in-band pixels, all at ODD positions
    frame[odd] = np.linspace(_LO + 1.0, _HI - 1.0, odd.size)
    return frame.reshape(H, W)


def _value_mask_roi_frame(frame, lo, hi):
    """Run the real RoiProcessor value-mask (a hist count-band ROI) and return the (H,W) roi_frame."""
    block = frame.reshape(1, 1, *frame.shape)
    hub = SignalHub()
    hub.register_signal("frame_0", SignalSchema(
        point_shape=(1,), data_shape=frame.shape, dtype=block.dtype, repeat_capacity=1))
    hub.publish({"frame_0": block})
    sel = Selection.value(lo, hi, metadata={"binding": value_mask_binding()})
    values, meta = encode_region(sel)
    rschema = SignalSchema(point_shape=(1,), data_shape=(1, 2), dtype=np.float64,
                           repeat_capacity=1, metadata=meta)
    hub.register_signal("frame_0_region", rschema)
    hub.publish({"frame_0_region": SignalTensor(values.reshape(1, 1, 1, 2), rschema)})
    node = RoiProcessor(hub, source_expr={"inputs": ["frame_0"], "source": "value = signal"},
                        region="frame_0_region")
    hub.publish({"frame_0": block})
    node.step()
    return np.asarray(hub.latest("roi_frame")[0, 0], dtype=float)


def test_small_hist_roi_frame_renders_in_band_pixels_not_blank():
    roi_frame = _value_mask_roi_frame(_sparse_in_band_source(), _LO, _HI)
    in_band = np.isfinite(roi_frame)
    assert 0 < int(in_band.sum()) < roi_frame.size                 # non-empty, but a sparse minority
    assert np.all((roi_frame[in_band] >= _LO) & (roi_frame[in_band] <= _HI))

    plot = Live2DDis(RegularRaster(shape=roi_frame.shape), roi_frame.reshape(-1, 1),
                     labels=["x", "y", "counts"]).show(display=False)
    try:
        # the even-stride sample IS all-NaN here -- the fallback returns the FULL frame's in-band pixels
        stride = roi_frame.size // Live2DDis._DIST_SAMPLE_CAP + 1
        strided = roi_frame.reshape(-1)[::stride]
        assert not np.isfinite(strided).any(), "test setup: the stride must miss all in-band pixels"
        assert np.isfinite(plot._distribution_values()).any(), "fallback must recover the in-band pixels"

        lo_c, hi_c = plot.image.get_clim()
        # FINITE clim with a real span (NOT NaN = blank), and its TOP reaches the in-band values (~600) --
        # NOT the collapsed 0..1 an all-NaN sample would leave, which renders every in-band pixel white.
        assert np.isfinite(lo_c) and np.isfinite(hi_c) and hi_c > lo_c
        assert hi_c > 100.0, f"clim top {hi_c} collapsed -- the in-band pixels would render blank"
        # the displayed image carries finite in-band pixels; the rest is NaN -> the cmap's bad colour
        assert int(np.isfinite(np.asarray(plot.image.get_array(), dtype=float)).sum()) > 0
    finally:
        plt.close(plot.fig)


def test_full_frame_value_mask_survives_panel_decimation_not_blank():
    """The REAL bug the user saw: a FULL-frame (1920x1200) value-mask roi_frame on a 2x2 panel is
    block-mean DECIMATED to the axes budget before imshow.  A plain block mean returns NaN for ANY
    block touching one out-of-band NaN, so a sparse in-band frame decimates to ALL-NaN and the panel
    renders 100% blank -- even with millions of in-band pixels.  NaN-aware decimation keeps every block
    that holds >=1 finite pixel, so the in-band silhouette survives.  Driven through the REAL Live2DDis
    display-refresh / decimation path at a realistic panel budget."""
    H, W = 1200, 1920
    rng = np.random.default_rng(0)
    frame = np.where(rng.random((H, W)) < 0.5, rng.normal(600.0, 20.0, (H, W)), rng.normal(200.0, 12.0, (H, W)))
    lo, hi = 590.0, 610.0                                      # a narrow band: a minority of pixels in-band
    roi_frame = np.where((frame >= lo) & (frame <= hi), frame, np.nan)
    in_band = int(np.isfinite(roi_frame).sum())
    assert 0 < in_band < roi_frame.size

    plot = Live2DDis(RegularRaster(shape=(H, W)), roi_frame.reshape(-1, 1),
                     labels=["x", "y", "counts"], spec=panel_plot_spec("2x2")).show(display=False)
    try:
        bx, by = _image_axes_px_budget(plot.ax)
        assert bx < W and by < H, "test setup: the 2x2 panel budget must force real decimation"
        shown = np.asarray(plot.image.get_array(), dtype=float)   # what imshow actually rasterizes
        assert shown.size < roi_frame.size                        # it WAS decimated (not the full frame)
        finite = int(np.isfinite(shown).sum())
        # WITHOUT the fix this is 0 (all-NaN -> blank).  With NaN-aware decimation a substantial fraction
        # of cells carry an in-band mean -- the user's masked frame lights up instead of white-on-white.
        assert finite > 0, "decimated roi_frame is all-NaN -> the panel renders blank (#4)"
        assert finite > 0.3 * shown.size, f"only {finite}/{shown.size} cells finite -- too sparse to read"
    finally:
        plt.close(plot.fig)


def test_empty_band_roi_frame_falls_back_to_a_finite_clim():
    """A genuinely EMPTY band (no pixels at all) is all-NaN; the panel must still not raise -- it falls
    back to a unit clim rather than a NaN one (a defensive floor, distinct from the sparse case above)."""
    roi_frame = _value_mask_roi_frame(_sparse_in_band_source(), 5000.0, 6000.0)   # nothing in this band
    assert not np.isfinite(roi_frame).any()
    plot = Live2DDis(RegularRaster(shape=roi_frame.shape), roi_frame.reshape(-1, 1),
                     labels=["x", "y", "counts"]).show(display=False)
    try:
        lo_c, hi_c = plot.image.get_clim()
        assert np.isfinite(lo_c) and np.isfinite(hi_c) and hi_c > lo_c
    finally:
        plt.close(plot.fig)
