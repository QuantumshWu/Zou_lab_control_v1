"""The legacy worker/GUI hand-off is the documented raster, not a stored QImage.

``zlc_frontend/render.py`` states the boundary rule in its own words: "No live
Figure, Artist, mutable or aliased ndarray view, or QImage storage crosses this
module's boundary."  The legacy canvas's front buffer - the value the render
thread hands the GUI thread - was a stored ``QImage``, i.e. the one alternative
that rule names.  That is not a vocabulary quibble: a QImage is a Qt object owned
by whichever thread built it, so it could never remain the hand-off once the
renderer stops being this very widget.  Owned bytes can, which is why
``RasterBuffer`` is what section 12.5's ``WORKER_RASTER_LIVE`` presenters accept.

There is no other active coverage of the front-buffer protocol:
``test_render_loop_contract.py`` reads ``_zlc_front`` directly and sits outside
``tests/migration_active_tests.txt``, so it is frozen - not run, not modified, and
not evidence for this branch.  This file is the oracle.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from zlc_frontend.render import PixelFormat, RasterBuffer


def _canvas():
    from zlc_frontend.qt_widgets import ensure_qt_app

    ensure_qt_app()
    from matplotlib.figure import Figure

    from Zou_lab_control.frontend.qt_canvas import panel_canvas

    figure = Figure(figsize=(2.0, 1.5))
    figure.subplots().plot([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    canvas = panel_canvas(figure)
    canvas._zlc_draw_if_needed()
    return canvas


def test_the_front_is_an_owned_raster_and_never_a_stored_qimage():
    """The type of the hand-off, asserted on the built object."""

    from PyQt5 import QtGui

    canvas = _canvas()
    canvas._zlc_present()

    front = canvas._zlc_front
    assert isinstance(front, RasterBuffer)
    assert not isinstance(front, QtGui.QImage)
    assert front.pixel_format is PixelFormat.RGBA8888
    # ``RasterBuffer`` already rejects a non-bytes payload, so this asserts the
    # thing that check cannot: the bytes are a COPY, not the live Agg buffer.
    assert front.pixels == bytes(canvas.buffer_rgba())
    assert front.pixels is not canvas.buffer_rgba()


def test_the_front_still_blits_the_same_pixels_as_the_agg_buffer():
    """The behaviour that must not regress while the type changes underneath it.

    A stride or format slip would still produce a plausible image - sheared, or
    with the channels rotated - so compare the actual bits rather than the shape.
    """

    from PyQt5 import QtGui

    canvas = _canvas()
    canvas._zlc_present()

    renderer = canvas.renderer
    width, height = int(renderer.width), int(renderer.height)
    direct = QtGui.QImage(canvas.buffer_rgba(), width, height,
                          QtGui.QImage.Format_RGBA8888)
    imaged = canvas._zlc_front_image(canvas._zlc_front)

    assert (imaged.width(), imaged.height()) == (direct.width(), direct.height())
    assert imaged.format() == direct.format()
    assert imaged.bits().asstring(imaged.byteCount()) == \
        direct.bits().asstring(direct.byteCount())


def test_dropping_the_front_still_falls_back_to_the_live_buffer():
    """Interaction drags drop the front so the live Agg blits show through."""

    canvas = _canvas()
    canvas._zlc_present()
    assert canvas._zlc_front is not None
    canvas._zlc_drop_front()
    assert canvas._zlc_front is None


def test_owning_the_copy_is_what_from_agg_rgba_is_for():
    """A length check passes for an aliasing buffer; only a copy survives a mutation.

    ``memoryview`` and ``bytearray`` both satisfy every other invariant on
    ``RasterBuffer`` while still viewing the buffer the worker is about to
    overwrite - the failure this constructor exists to make impossible.
    """

    source = bytearray(b"\x01\x02\x03\x04" * 6)          # 3 x 2 RGBA
    raster = RasterBuffer.from_agg_rgba(3, 2, memoryview(source))
    before = raster.pixels
    source[0] = 0xFF

    assert raster.pixels == before
    assert raster.pixels[0] == 0x01
    assert raster.stride_bytes == 3 * 4


def test_both_agg_surfaces_construct_the_front_through_the_one_source():
    """No caller may re-type the stride/format/copy triple.

    Asserted structurally rather than by value: two callers that happen to agree
    today are exactly what drifts, and a value comparison would not notice a
    second spelling appearing beside the shared one.
    """

    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    sources = {
        "SinglePanelAggRenderer._draw_raster":
            inspect.getsource(SinglePanelAggRenderer._draw_raster),
        "EmbeddedFigureCanvas._zlc_snapshot_front": _snapshot_front_source(),
    }

    for where, text in sources.items():
        tree = ast.parse(_dedent(text))
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert "from_agg_rgba" in calls, f"{where} must build the raster through the one source"
        assert "RasterBuffer(" not in text, f"{where} re-types the raster construction"


def _snapshot_front_source() -> str:
    """Read the legacy method from the FILE, not from an import.

    ``EmbeddedFigureCanvas`` is defined inside an ``else:`` branch that only runs
    when matplotlib's Qt backend imports, so reading the source off the class
    would make this guard silently skip wherever that import fails.
    """

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "Zou_lab_control" / "frontend" / "qt_canvas.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_zlc_snapshot_front":
            return ast.get_source_segment(path.read_text(encoding="utf-8"), node)
    raise AssertionError("_zlc_snapshot_front is gone; the front protocol moved")


def _dedent(text: str) -> str:
    import textwrap

    return textwrap.dedent(text)
