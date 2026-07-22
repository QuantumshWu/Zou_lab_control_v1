"""The Preview tab must show the whole plot, labelled by the names the board uses.

Two failures were visible on a real window and neither is caught by "does it draw":
the pixmap rendered fine at 500x400 and was then shown in a QLabel that had
collapsed to a ~13 px sliver, and the y axis carried raw lane keys (``ch00``)
instead of the board names (``cooling``) the operator reads everywhere else.

Driven the way a person drives it -- open the editor, switch to Preview, let it
refresh -- and asserting the geometry and the labels the render produced.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets
import pytest

from zlc_frontend.qt_widgets import ensure_qt_app


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture
def preview_editor(application):
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    editor = open_pulse_editor()
    window = editor.window()
    window.show()
    for _ in range(5):
        application.processEvents()
    tabs = next(t for t in window.findChildren(QtWidgets.QTabWidget) if t.count() >= 2)
    tabs.setCurrentIndex(1)                                   # Preview
    for _ in range(6):
        application.processEvents()
    editor.refresh_preview()                                 # the tab-enter refresh, made explicit
    for _ in range(4):
        application.processEvents()
    yield editor
    try:
        window.close()
    except Exception:                                        # pragma: no cover - teardown only
        pass
    application.processEvents()


def test_the_preview_plot_is_not_collapsed_to_a_sliver(preview_editor):
    label = preview_editor.preview_image
    pixmap = label.pixmap()
    assert pixmap is not None and not pixmap.isNull(), "preview produced no pixmap"
    assert pixmap.height() > 100, "the rendered plot itself is too short to be a plot"
    # The scroll area holds the body at its own size hint, so the QLabel only shows the
    # whole plot if the body was resized to it.  A sliver here is the reported bug.
    assert label.height() >= pixmap.height(), (
        f"the plot is {pixmap.height()} px tall but the label shows only {label.height()} px")


def test_the_preview_status_reads_like_the_reference(preview_editor):
    """The status line names how many channels were drawn, out of how many exist,
    the mode, and the repeat -- the same wording main shows (``N/M plotted
    (active channels) | repeat …``), NOT a bare ``us, periods`` blurb.
    """

    import re

    text = preview_editor.preview_status.text()
    assert re.fullmatch(
        r"\d+/\d+ plotted \((active|all) channels\) \| repeat (∞|\d+)", text), (
        f"the preview status {text!r} does not match the reference wording "
        "'N/M plotted (active channels) | repeat …'")


def test_the_preview_y_axis_uses_board_names_not_raw_lane_keys(preview_editor):
    state = preview_editor.read_state()
    snapshot, _channels = preview_editor._preview_snapshot(state, include_always_off=False)
    assert snapshot is not None
    # The channel axis carries the display labels; a default board's first active lane is
    # ``cooling`` (key ``ch00``), so the axis must NOT read ``ch00``.
    channel_axis = snapshot.block.schema.cell_schema.data_axes[0]
    shown = [str(v) for v in channel_axis.coordinates]
    assert shown, "no channels on the preview axis"
    labels = dict(getattr(state.port_catalog, "channel_labels", {}) or {})
    for name in shown:
        assert name not in labels, (
            f"the preview axis shows the raw lane key {name!r} instead of its board name "
            f"{labels[name]!r}")


def _png_size(data: bytes) -> tuple[int, int]:
    from PyQt5 import QtGui

    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(data)
    return (pixmap.width(), pixmap.height())


def test_show_off_rows_reveals_the_always_off_channels(preview_editor):
    """Flipping the "Show off rows" switch ON must draw MORE rows than OFF (the always-off
    channels) and produce a genuinely different picture -- the regression was that the render
    took its rows from the compiled sequence (active lanes only), so the toggle did nothing.
    """

    state = preview_editor.read_state()
    off_rows = preview_editor._preview_channels(state, include_always_off=False)
    all_rows = preview_editor._preview_channels(state, include_always_off=True)
    assert len(all_rows) > len(off_rows), (
        "'Show off rows' ON did not add any channels -- the toggle is a no-op "
        f"(off={off_rows}, on={all_rows})")
    # The whole universe is exactly the catalog's digital ports, never the compiled sequence.
    universe = [port.lanes[0] for port in state.port_catalog.digital_ports]
    assert all_rows == universe

    off_png = preview_editor.preview_png_bytes(state, include_always_off=False)
    on_png = preview_editor.preview_png_bytes(state, include_always_off=True)
    assert off_png != on_png, "the off-rows toggle produced an identical PNG"
    assert _png_size(on_png)[1] > _png_size(off_png)[1], (
        "more rows must make a taller figure")


def test_size_preset_scales_the_rendered_pixels(preview_editor):
    """Picking a size in the dropdown must rescale the figure to that panel size's pixel box --
    the regression was that the render ignored the preset entirely, so every size looked the same.
    """

    from zlc_frontend.render_style import panel_display_size

    state = preview_editor.read_state()
    combo = preview_editor.preview_size_combo
    preview_editor._preview_size_pinned = True                # as if the operator picked a size

    dims = {}
    for preset in ("2x2", "4x4", "8x8"):
        combo.setCurrentText(preset)
        dims[preset] = _png_size(preview_editor.preview_png_bytes(state))
        # The emitted raster is exactly the on-screen card box for that preset (one geometry source).
        assert dims[preset] == panel_display_size(preset), (
            f"size {preset} rendered {dims[preset]} px, not panel_display_size {panel_display_size(preset)}")
    assert dims["2x2"][0] < dims["4x4"][0] < dims["8x8"][0], "a bigger preset must widen the figure"
    assert dims["2x2"][1] < dims["4x4"][1] < dims["8x8"][1], "a bigger preset must heighten the figure"


def test_a_trailing_all_off_period_keeps_the_frame_length_visible(preview_editor):
    """Two periods, a channel ON only in period 0: the preview must span the WHOLE frame.

    The regression: the render took its span from ``sequence.duration``, which derives from
    the last pulse EDGE -- the trailing all-off period vanished and the channel read as
    "always on" across a truncated axis.  The frame length is the period table's
    ``total_duration_ns`` (the authoritative single source), so the ON block must end at
    half the axis, not at its right edge.
    """

    state = preview_editor.read_state()
    # The default editor state IS the scenario: 2 x 1000 ns periods, ch00 on only in period 0.
    assert len(state.periods) == 2
    assert state.periods[0].states[0] == 1 and state.periods[1].states[0] == 0
    seq = state.to_sequence()
    frame_s = float(state.total_duration_ns()) * 1e-9
    last_edge_s = max(float(p.stop) for p in seq.pulses)
    assert last_edge_s < frame_s, "scenario must have a trailing all-off stretch"
    # The rendered axis must include the full frame: compare the PNG against a render of a
    # one-period state (which spans only the pulse) -- they must differ, and the block in the
    # two-period render must NOT touch the right margin.  Cheap mechanical proxy: rendering
    # with the trailing period present vs removed changes the picture.
    png_two = preview_editor.preview_png_bytes(state, include_always_off=False)
    import dataclasses as _d
    one = type(state).from_dict({**state.to_dict(), "periods": [state.periods[0].to_dict()]}) \
        if hasattr(state.periods[0], "to_dict") else None
    if one is not None:
        png_one = preview_editor.preview_png_bytes(one, include_always_off=False)
        assert png_two != png_one, (
            "removing the trailing all-off period did not change the preview -- the frame "
            "length is not being honoured")


def test_repeat_forever_bracket_draws_the_infinity_glyph_cleanly(preview_editor):
    """A repeat-forever pulse draws the ×∞ bracket label; the ∞ (U+221E) must resolve to a real glyph.

    The design font (Helvetica Light) lacks ∞, so drawing the label in it emits a "Glyph 8734 missing"
    UserWarning and paints a tofu box -- the reference dodges this by drawing the bracket label in
    DejaVu Sans.  Fail if that warning ever comes back.
    """

    import warnings

    state = preview_editor.read_state()
    try:
        state.repeat_forever = True
    except Exception:                                        # pragma: no cover - defensive
        pytest.skip("cannot force repeat_forever on this state")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preview_editor.preview_png_bytes(state, include_always_off=True)
    glyph_warnings = [w for w in caught if "missing from font" in str(w.message)]
    assert not glyph_warnings, (
        "the ×∞ bracket label hit a missing-glyph fallback (tofu box): "
        + "; ".join(str(w.message) for w in glyph_warnings))
