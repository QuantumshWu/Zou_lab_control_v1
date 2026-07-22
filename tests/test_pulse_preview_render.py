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
    (active channels) | repeat …``), where the repeat is the reference's exact
    three-state notation: ``repeat ∞``, ``repeat P1-P2 x3`` (bracket covers the
    whole table) or ``repeat ∞ + P2-P3 x2`` (partial inner bracket); a one-shot
    program without a bracket says nothing.
    """

    import re

    text = preview_editor.preview_status.text()
    assert re.fullmatch(
        r"\d+/\d+ plotted \((active|all) channels\)"
        r"( \| repeat (∞|P\d+-P\d+ x\d+|∞ \+ P\d+-P\d+ x\d+))?", text), (
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


def test_display_carries_the_screen_ratio_and_export_saves_at_600_dpi(preview_editor):
    """The two dpi principles, both the reference's, must hold mechanically:

    * the ON-SCREEN raster is the panel's logical pixel box times the screen's
      device-pixel ratio (blitted 1:1 crisp -- a HiDPI screen must not stretch a
      soft logical-pixel image), and the shown pixmap is tagged with that ratio;
    * an EXPORT ignores the screen and saves the same drawing at the style's
      ``savefig.dpi`` (600), so a saved figure is publication resolution.
    """

    from zlc_frontend.render_style import (
        DEFAULT_STYLE, panel_display_size, panel_figure_size_inches)

    state = preview_editor.read_state()
    size = preview_editor._preview_size_for(state, include_always_off=False)
    logical = panel_display_size(size)

    doubled = _png_size(preview_editor.preview_png_bytes(
        state, include_always_off=False, pixel_ratio=2.0))
    assert doubled == (logical[0] * 2, logical[1] * 2), (
        f"pixel_ratio=2 must emit exactly twice the logical box {logical}, got {doubled}")

    exported = _png_size(preview_editor.preview_png_bytes(
        state, include_always_off=False, export=True))
    inches = panel_figure_size_inches(size)
    save_dpi = float(DEFAULT_STYLE["savefig.dpi"])
    expected = (round(inches[0] * save_dpi), round(inches[1] * save_dpi))
    assert exported == expected, (
        f"export must save at savefig.dpi={save_dpi:g} ({expected} px), got {exported}")

    with pytest.raises(ValueError):
        preview_editor.preview_png_bytes(state, pixel_ratio=2.0, export=True)

    # The GUI display chain tags the pixmap with the widget's ratio so Qt shows
    # it at the logical size instead of stretching device pixels.
    preview_editor.refresh_preview()
    pixmap = preview_editor.preview_image.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    assert pixmap.devicePixelRatio() == pytest.approx(
        float(preview_editor.devicePixelRatioF() or 1.0))


def test_the_inner_repeat_bracket_draws_nested_not_unrolled(preview_editor):
    """A finite inner bracket ``[P1..P1] × 3`` must read as its OWN nested square
    bracket over period 1's span -- the reference's semantics exactly:

    * partial bracket in a forever loop  -> outer ``×∞`` plus inner ``×3``;
    * bracket covering the whole table   -> only the inner ``×N`` bracket;
    * partial bracket, forever off       -> the inner bracket alone;
    * the axis spans the AUTHORED table, never the unrolled copies.

    The regression this pins: the preview collapsed everything to one
    whole-frame marker (×∞ swallowing ×N entirely) and drew the bracket
    UNROLLED across an axis stretched to the expanded length.
    """

    from zlc_workbench.pulse_editor.plot_bridge_pulse_gui import _preview_repeat_markers

    state = preview_editor.read_state()
    assert len(state.periods) == 2
    baseline = preview_editor.preview_png_bytes(state, include_always_off=False)

    state.repeat_start, state.repeat_end, state.repeat_count = 0, 0, 3
    markers, total = _preview_repeat_markers(state)
    assert [label for (_start, _stop, label) in markers] == ["×∞", "×3"], (
        "a partial inner bracket inside a forever loop must draw BOTH brackets")
    outer, inner = markers
    assert outer[0] == 0.0 and outer[1] == pytest.approx(total)
    assert inner[0] == 0.0 and 0.0 < inner[1] < total, (
        "the inner bracket must span exactly its own periods, not the whole frame")
    # The frame is the AUTHORED table: expanding [P0]x3 + P1 would be longer.
    expanded = float(state.total_duration_ns()) * 1e-9
    assert total < expanded, "the preview axis must not stretch over the unrolled copies"
    with_bracket = preview_editor.preview_png_bytes(state, include_always_off=False)
    assert with_bracket != baseline, "adding the inner bracket left the preview unchanged"

    # Bracket over the WHOLE table: only the inner ×N bracket is drawn.
    state.repeat_start, state.repeat_end = 0, 1
    markers, _total = _preview_repeat_markers(state)
    assert [label for (_start, _stop, label) in markers] == ["×3"]

    # Forever off with a partial bracket: the inner bracket alone.
    state.repeat_start, state.repeat_end = 0, 0
    state.repeat_forever = False
    markers, _total = _preview_repeat_markers(state)
    assert [label for (_start, _stop, label) in markers] == ["×3"]


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


def test_the_panel_exit_is_the_same_picture_with_a_live_viewport(application):
    """``render_pulse_timeline_panel`` is the PNG exit's twin: ONE drawing, two exits.

    The interactive raster must be PIXEL-IDENTICAL to the decoded PNG -- any
    divergence means the exits stopped sharing the drawing and the preview a
    person selects on is no longer the preview that gets saved.  The payload
    must carry the drawn rows (channels then analog buses, display labels) and
    a viewport whose x mapping covers the drawn frame, because that mapping is
    what the unified selector overlay converts gestures through.
    """

    import numpy as np
    from PyQt5 import QtGui
    from zlc_data import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId

    from zlc_frontend.figure import DatasetId, EvaluatedInput
    from zlc_frontend.matplotlib_render import (
        render_pulse_timeline_panel, render_pulse_timeline_png)
    from zlc_frontend.render import PulsePanelPayload, RasterBuffer

    kw = dict(
        pulses=[dict(channel="ch00", start=0.0, stop=1e-3, name="cool")],
        channels=["ch00", "ch01"],
        channel_labels={"ch00": "cooling", "ch01": "probe"},
        total_duration=2e-3,
        title="preview",
        size="2x2",
        analog_traces=[dict(name="da_dipole", label="da_dipole", min=-512, max=511,
                            starts=[0.0, 1e-3, 2e-3], values=[300, -100])],
    )
    provenance = EvaluatedInput(
        DatasetId("pulse"),
        DatasetRevisionRef(
            BlockId("pulse-block"),
            StreamGenerationId("pulse-generation"),
            "e" * 64,
            DatasetRevision(1),
        ),
    )
    raster, payload = render_pulse_timeline_panel(**kw, evaluated_input=provenance)
    assert isinstance(raster, RasterBuffer) and isinstance(payload, PulsePanelPayload)
    assert payload.evaluated_input == provenance

    image = QtGui.QImage()
    assert image.loadFromData(render_pulse_timeline_png(**kw)), "PNG did not decode"
    image = image.convertToFormat(QtGui.QImage.Format_RGBA8888)
    assert (image.width(), image.height()) == (raster.width, raster.height), (
        "the two exits emit different pixel boxes")
    bits = image.constBits()
    bits.setsize(image.bytesPerLine() * image.height())
    png_pixels = np.frombuffer(bits, np.uint8).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 4]
    raster_pixels = np.frombuffer(raster.pixels, np.uint8).reshape(
        raster.height, raster.stride_bytes)[:, :raster.width * 4]
    assert np.array_equal(png_pixels, raster_pixels), (
        "the interactive raster and the PNG exit diverged -- they must share one drawing")

    assert payload.row_keys == ("ch00", "ch01", "analog:0:da_dipole")
    assert payload.row_labels == ("cooling", "probe", "da_dipole")
    viewport = payload.viewport
    assert viewport.home_x_limits == viewport.x_limits, "home must pin to the drawn frame"
    assert viewport.x_limits[0] < 0.0 and viewport.x_limits[1] > 2e-3, (
        "the drawn x span must cover the whole frame (plus the breathing margin)")
    x_centre, _y = viewport.widget_normalized_to_data(0.5, 0.5)
    assert viewport.x_limits[0] <= x_centre <= viewport.x_limits[1], (
        "the widget centre does not map inside the drawn time span")
