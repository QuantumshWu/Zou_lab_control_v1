"""The embedded panel canvas must have a DPR-INVARIANT fixed size and a synchronously-rendered buffer.

Root cause of the "rebuild occasionally makes the figure vanish or grow" report:

* GROW -- the canvas size was pinned from ``sizeHint() == figure.bbox / device_pixel_ratio``.  The DPR
  cancels in that ratio ONLY when figure.dpi and device_pixel_ratio are read at the SAME instant; mid
  screen-ratio-change (a rebuild burst, a monitor move) they desync and sizeHint balloons, so the pinned
  ``setMinimumSize`` ballooned the card.  The fix sizes the widget from the DPR-FREE constant
  ``inches x design_dpi x display_scale`` (set once, min==max) -- invariant under any DPR.

* VANISH -- the first paint was ``draw_idle()`` (deferred): a paint hitting the freshly-swapped canvas
  before the deferred draw ran showed an unrendered (blank) Agg buffer.  The fix draws SYNCHRONOUSLY at
  construction / resync / swap, so the buffer is never empty.

Guarded mechanically so the racy ``sizeHint``-pinning / ``draw_idle``-first-paint design cannot return.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("PyQt5")
pytest.importorskip("matplotlib")

from Zou_lab_control.frontend.qt_fluent import ensure_qt_app  # noqa: E402
from Zou_lab_control.frontend import plot as _plot  # noqa: E402
from Zou_lab_control.frontend.qt_canvas import panel_canvas  # noqa: E402
from Zou_lab_control.frontend.style import DESIGN_DPI, PANEL_DISPLAY_SCALE  # noqa: E402


def _hist_canvas():
    fig = _plot(np.random.default_rng(0).normal(300.0, 20.0, 400), kind="hist").fig
    return panel_canvas(fig, isolate_wheel=False)


def _buffer_blank(canvas) -> bool:
    b = np.asarray(canvas.buffer_rgba())
    return bool((b[..., :3] == b[0, 0, :3]).all())


def test_canvas_is_fixed_to_the_dpr_free_design_size():
    ensure_qt_app()
    c = _hist_canvas()
    design = c._zlc_design_size()
    assert (c.minimumWidth(), c.minimumHeight()) == (c.maximumWidth(), c.maximumHeight()), \
        "the canvas must be FIXED (min==max), not a floor a resync can raise"
    assert (c.width(), c.height()) == (design.width(), design.height())
    # the design size is the DPR-FREE product, NOT the stock sizeHint (bbox / device_pixel_ratio)
    w_in, h_in = c._zlc_inches
    assert design.width() == max(1, round(w_in * float(DESIGN_DPI) * PANEL_DISPLAY_SCALE))
    assert design.height() == max(1, round(h_in * float(DESIGN_DPI) * PANEL_DISPLAY_SCALE))


def test_resync_re_asserts_the_same_fixed_size():
    # A deferred _zlc_resync / showEvent must correct the render BUFFER but never re-pin a different
    # size (the old setMinimumSize(sizeHint()) re-pin is what ballooned the figure across rebuilds).
    ensure_qt_app()
    c = _hist_canvas()
    size0 = (c.width(), c.height())
    c._zlc_resync()
    assert (c.width(), c.height()) == size0


def test_widget_size_does_not_follow_the_render_buffer_resolution():
    # The core bug: the widget size was figure.bbox / device_pixel_ratio (sizeHint), so anything that
    # inflated the render buffer dpi (a hi-DPI screen) grew the widget.  The fix sizes the widget from
    # the DPR-free _zlc_design_size (inches x design_dpi x display_scale -- it NEVER reads figure.dpi),
    # so the buffer dpi can change freely while the widget size stays put.  Inflating figure.dpi
    # directly models a hi-DPI screen's larger render buffer, without monkeypatching the C++
    # devicePixelRatioF (which corrupts the sip type).
    ensure_qt_app()
    c = _hist_canvas()
    size0 = (c.width(), c.height())
    buf0 = int(c.renderer.width)
    c.figure._set_dpi(float(c.figure.dpi) * 2.0, forward=False)   # a higher-resolution render buffer
    c.draw()
    assert int(c.renderer.width) > buf0, "the render BUFFER dpi must follow the resolution change"
    assert (c.width(), c.height()) == size0, "the WIDGET size must NOT follow the buffer dpi (DPR-free)"
    assert (c.width(), c.height()) == (c._zlc_design_size().width(), c._zlc_design_size().height())


def test_buffer_is_rendered_synchronously_never_blank():
    ensure_qt_app()
    c = _hist_canvas()
    assert not _buffer_blank(c), "construction must render the Agg buffer synchronously (no blank frame)"
    c._zlc_resync()
    assert not _buffer_blank(c), "a resync must leave a rendered buffer, not a deferred-blank one"


def test_rebuild_burst_keeps_a_constant_size_and_a_painted_buffer():
    # The exact user path: rebuild the same panel many times fast (scroll a rebuild-class param).  Every
    # rebuilt canvas must be the SAME fixed size and carry a painted buffer -- no grow, no vanish.
    ensure_qt_app()
    sizes, blanks = set(), []
    for _ in range(8):
        c = _hist_canvas()
        sizes.add((c.width(), c.height()))
        blanks.append(_buffer_blank(c))
    assert len(sizes) == 1, f"every rebuild must produce the SAME canvas size, got {sizes}"
    assert not any(blanks), "no rebuilt canvas may be blank"
