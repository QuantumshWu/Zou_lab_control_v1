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
