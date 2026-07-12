"""Device-manager ART invariants (#5), offscreen-assertable.

The confocal-style group box carries a GREY title pill (``background: BG``); that pill is
only legible on a WHITE surface.  A ``FluentGroupBox`` placed straight on the grey window /
a transparent scroll host loses its title (pill grey-on-grey).  And the OUTERMOST region of
a GUI must not carry its own card border (the window frame IS the boundary).  These pin:

1. every config-editor / hardware section ``FluentGroupBox`` is nested inside a WHITE
   ``FluentFrame`` (so its title pill reads), never straight on the panel/scroll host;
2. no OUTERMOST ``FluentFrame`` in the panel is bordered (the tab pane / window is the edge);
3. the config body lives in a ``FluentTabWidget`` whose first tab is the permanent "Config"
   (the white card the group boxes read against).
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from conftest import raw_device_set

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.frontend.device_manager import DeviceManagerPanel
from Zou_lab_control.neutral_atom._gui import _session_device_binding
from Zou_lab_control.frontend.qt_fluent import FluentFrame, FluentGroupBox, FluentTabWidget


@pytest.fixture(scope="module", autouse=True)
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def exp():
    session = na.connect("virtual")
    yield session
    session.close()


def _all_group_boxes(root):
    return root.findChildren(FluentGroupBox)


def _ancestor_frame(widget):
    """The nearest FluentFrame ANCESTOR of ``widget`` (its white host), or None."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, FluentFrame):
            return parent
        parent = parent.parentWidget()
    return None


def test_config_body_lives_in_a_tab_widget_with_a_permanent_config_tab(exp):
    panel = DeviceManagerPanel(raw_device_set(exp), session_binding=_session_device_binding(exp))
    tabs = panel.findChildren(FluentTabWidget)
    assert tabs, "the config body must live in a FluentTabWidget"
    top = tabs[0]
    assert top.tabText(0) == "Config", "the first (permanent) tab is Config"


def test_every_section_group_box_is_nested_in_a_white_frame(exp):
    """Each section FluentGroupBox sits inside a FluentFrame (white host) so its grey title
    pill is legible -- never straight on the grey window / a transparent scroll host (#5)."""
    panel = DeviceManagerPanel(raw_device_set(exp), session_binding=_session_device_binding(exp))
    boxes = _all_group_boxes(panel)
    assert boxes, "the editor must render section group boxes"
    for box in boxes:
        host = _ancestor_frame(box)
        assert host is not None, (
            f"group box {box.title()!r} is not nested in a FluentFrame -- its grey title pill "
            "would sit on the grey window and vanish (#5)")


def test_no_outermost_frame_is_bordered(exp):
    """The panel's structural FluentFrames (the column hosts / button strip) are borderless --
    the tab pane / window edge is the boundary, so no inner card border doubles it (#5).  A
    nested per-entry card (``_DeviceEntryCard``) may still be bordered; this pins that the
    COLUMN-LEVEL hosts are not."""
    panel = DeviceManagerPanel(raw_device_set(exp), session_binding=_session_device_binding(exp))
    # the column hosts + button strip are the frames that are DIRECT layout children of the
    # config page (not nested inside a group box): none of them may be bordered.
    for frame in panel.findChildren(FluentFrame):
        if _ancestor_frame(frame) is None:                    # a top-level structural frame
            assert not getattr(frame, "_bordered", False), (
                "an outermost/structural FluentFrame must not be bordered (#5): the window edge "
                "is the boundary")
