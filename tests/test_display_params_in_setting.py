"""#4: a plot panel's pure DISPLAY knobs (bins / history length / fit / log axis / colormap) render in
the lightweight Setting popup, NOT the Edit tab's "Parameters" (acquisition) section.

``ParamDecl.display`` is the SINGLE classification axis: True = a display knob (Setting), False = an
acquisition / measurement-API param (Edit "Parameters").  ``bins`` (hist) and ``length`` (monitor) are
pure display knobs, so flipping their flag to True is the only edit needed -- both the Setting build and
the Edit "Parameters" build derive from ``s.display``, so the param moves surfaces with no extra wiring,
and an empty acquisition set leaves NO empty "Parameters" header.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _card(kind):
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    return PanelCard(PanelConfig(kind=kind, inputs=["frame_0"], source="value = signal"))


def test_bins_and_history_are_setting_display_params():
    """``bins`` (hist) and ``length`` (monitor) are declared display=True, so they appear as live
    controls in the Setting popup's param widgets and are ABSENT from the Edit "Parameters"
    (display=False) set -- the SINGLE-source classification, asserted from PANEL_PARAMS directly."""
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS

    for kind, key in (("hist", "bins"), ("monitor", "length")):
        decls = {s.key: s for s in PANEL_PARAMS.get(kind, ())}
        assert key in decls, f"{kind} must declare a {key!r} param"
        assert decls[key].display is True, f"{key!r} is a pure display knob -> Setting (display=True)"
        # the Edit "Parameters" (acquisition) set derives from display=False; it must NOT contain it
        acquisition = [s.key for s in PANEL_PARAMS.get(kind, ()) if not s.display]
        assert key not in acquisition, f"{key!r} must not be an Edit-tab acquisition param"


def test_setting_popup_actually_builds_the_bins_history_control():
    """End-to-end: the Setting popup of a hist / monitor panel builds a live widget for bins / length
    (so the operator can edit them there), wired through the same PARAM_WIDGETS path as every knob."""
    for kind, key in (("hist", "bins"), ("monitor", "length")):
        card = _card(kind)
        assert key in card.param_widgets, \
            f"{kind} Setting popup must build a control for the display param {key!r}"
