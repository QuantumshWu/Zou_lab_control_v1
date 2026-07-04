"""#6: a panel's param controls are a VIEW of ``config.params`` (the single source of truth), refreshed
on show -- so a value changed in the Setting popup shows in the Edit tab when it next opens, and vice
versa.  Neither surface keeps a private copy that drifts.

The Setting popup (``PanelCard.refresh_on_show``) and the Edit tab (``PanelEditor.refresh_on_show`` ->
``_refresh_display_params``) both re-seed every display control from ``config.params`` through the SAME
``PARAM_WIDGETS.write`` entry point (one rule per kind, no per-key handwiring).  ``_on_tab_changed``
already calls ``refresh_on_show`` when a tab becomes visible; ``_open_settings`` calls the card's.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _console():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
    console._timer.stop()
    console.refresh_once = lambda: None
    return exp, console


def _hist_card(console):
    from Zou_lab_control.frontend import panel_plot
    from Zou_lab_control.frontend.task_console import PanelConfig, GAP
    vals = np.concatenate([np.random.default_rng(0).normal(300.0, 20.0, 400),
                           np.random.default_rng(1).normal(460.0, 20.0, 300)])
    card = console._new_panel_card(PanelConfig(kind="hist", title="dis", source="counts",
                                               row=GAP, col=GAP, size="2x4"))
    console._attach_card(card)
    card.plotter = panel_plot(vals, kind="hist", size="2x4", bins=60, title="dis")
    return card


def _bins_widget_value(widget) -> int:
    # bins is an int param -> a FluentDoubleSpinBox (decimals 0); read its current value
    return int(widget.value())


def test_setting_change_shows_in_edit_on_show():
    """Change bins via the Setting popup's path (card._set_param); the Edit tab's bins control reflects
    the new value after its refresh_on_show (the hook _on_tab_changed fires on tab switch)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelEditor
    ensure_qt_app()
    exp, console = _console()
    try:
        card = _hist_card(console)
        editor = PanelEditor(card, console)
        assert "bins" in card.param_widgets and "bins" in editor.ed_params
        # change bins through the live card (what the Setting spin does)
        card._set_param("bins", 123)
        assert card.config.params["bins"] == 123
        # the Edit tab, opened/shown, re-seeds its control from config.params
        editor.refresh_on_show()
        assert _bins_widget_value(editor.ed_params["bins"]) == 123, \
            "Edit tab must show the bins value changed in the Setting popup"
    finally:
        console.shutdown()


def test_edit_change_shows_in_setting_on_show():
    """Change bins via the Edit tab's path (editor._edit_param); the Setting popup's bins control
    reflects the new value after the card's refresh_on_show (called when the popup opens)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelEditor
    ensure_qt_app()
    exp, console = _console()
    try:
        card = _hist_card(console)
        editor = PanelEditor(card, console)
        editor._edit_param("bins", 77)
        assert card.config.params["bins"] == 77
        card.refresh_on_show()
        assert _bins_widget_value(card.param_widgets["bins"]) == 77, \
            "Setting popup must show the bins value changed in the Edit tab"
    finally:
        console.shutdown()


def test_refresh_on_show_does_not_refire_apply():
    """Re-seeding a control on show must NOT re-fire its apply callback (signals blocked) -- otherwise a
    refresh would needlessly mark dirty / rebuild.  Assert config.params is unchanged by a refresh."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()
    exp, console = _console()
    try:
        card = _hist_card(console)
        card.config.params["bins"] = 88
        before = dict(card.config.params)
        card.refresh_on_show()
        assert card.config.params == before, "refresh_on_show must not mutate config.params"
        assert _bins_widget_value(card.param_widgets["bins"]) == 88
    finally:
        console.shutdown()
