"""#H3v-4b: a plot's tunable DISPLAY params are DECLARED once and auto-injected into BOTH the Setting
popup AND the Edit tab -- no hand-wiring per surface.

The pain this pins: when a new plot display knob (e.g. ``fixed_lo``) was added, it had to be hand-coded
in two separate builders (the Setting popup + the Edit tab).  Now every declarative ``display=True``
ParamDecl for a kind -- the per-kind ``PANEL_PARAMS`` entries PLUS the shared ``relim`` chooser -- is
rendered through ONE shared ``_emit_param_rows`` in both surfaces, so a single declaration shows up in
both.  This test asserts that single-source coverage and that a relim edit drives the live plotter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_task_console_measurement_gui import _camera_console  # noqa: E402
from Zou_lab_control.frontend.task_console import PANEL_PARAMS, _RELIM_PARAM  # noqa: E402


def test_every_declared_display_param_appears_in_both_setting_and_edit():
    """For a plot panel, the set of declarative display knobs (per-kind display=True ParamDecls + the
    shared relim chooser) is rendered in BOTH the Setting popup (``card.param_widgets``) and the Edit
    tab (``editor.ed_params``).  Adding a ParamDecl is the ONLY edit needed to expose it in both."""
    console, node, cam, card, editor = _camera_console()
    try:
        expected = {s.key for s in PANEL_PARAMS.get(card.config.kind, ()) if s.display}
        expected.add(_RELIM_PARAM.key)                     # the shared plot-display knob
        # a 2D image kind declares a colormap, so the set is non-trivial (not a vacuous pass)
        assert "cmap" in expected and "relim" in expected
        # Setting popup: every declared display knob is a live widget
        for key in expected:
            assert key in card.param_widgets, f"Setting popup missing declared display param {key!r}"
        # Edit tab: the SAME declared knobs, rendered by the SAME shared helper
        for key in expected:
            assert key in editor.ed_params, f"Edit tab missing declared display param {key!r}"
    finally:
        console.shutdown()


def test_relim_edit_through_one_path_drives_the_live_plotter_and_reveals_fixed_row():
    """The declarative relim chooser routes through ONE ``_set_param`` path: it persists
    ``config.params['relim']``, pushes the mode onto the LIVE plotter, and reveals the fixed lo/hi row
    only in 'fixed' mode -- the side-effects the old hand-wired ``_on_relim_mode`` carried, now reached
    declaratively.  The Setting popup and the Edit tab share the same config.params (never drift)."""
    console, node, cam, card, editor = _camera_console()
    try:
        # default = tight; the fixed lo/hi row is hidden.  (isHidden() = the explicit hidden FLAG, not
        # isVisible() which is False here merely because the un-shown Setting popup isn't on screen.)
        assert card.config.params.get("relim", "tight") in ("tight",)
        assert card.fixed_lim_row.isHidden()

        # a declarative relim edit pushes the live plotter + reveals the fixed row
        card._set_param("relim", "fixed")
        assert card.config.params["relim"] == "fixed"
        assert card.plotter is None or card.plotter.relim_mode == "fixed"
        assert not card.fixed_lim_row.isHidden()           # fixed lo/hi revealed

        # back to tight hides it again (single source -> the Edit tab reads the same key)
        card._set_param("relim", "tight")
        assert card.fixed_lim_row.isHidden()
        assert card.config.params["relim"] == "tight"
    finally:
        console.shutdown()
