"""MECHANICAL guard for the P4 task-console redesign.

P4 slimmed the per-panel Setting popup down to BASIC display controls only
(Source, size, colormap, colorbar, unit, x/y limits) and moved the heavier
processing (curve fit, relim) into the Control tab.  Two regressions this test
pins so they cannot creep back into the lightweight popup:

1. structural -- the Setting popup contains NO Fit button and NO relim combo,
   and relim is NOT a Setting Display param;
2. behavioural -- the new basic display toggles (colorbar / unit / x-y limits)
   persist through ``PanelConfig.to_dict``/``from_dict`` and are RE-APPLIED on
   the rebuilt plotter (the panel rebuilds whenever its data shape changes, so a
   toggle that was not re-applied would silently revert).

These run on the offscreen Qt platform and build PanelCards directly, so they do
NOT pull in the flaky demo_console GUI fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _card(kind, *, params=None, source=None, size="2x2"):
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    cfg = PanelConfig(kind=kind, title=f"{kind} test", row=0, col=0, size=size,
                      source=source, params=params or {})
    return PanelCard(cfg)


def _settings_widgets(card):
    """Every widget the Setting popup actually contains (recursively)."""
    return card.settings_popup.findChildren(__import__("PyQt5.QtWidgets", fromlist=["QWidget"]).QWidget)


def _button_texts(card):
    from PyQt5.QtWidgets import QPushButton
    return [b.text() for b in card.settings_popup.findChildren(QPushButton)]


def _combo_item_sets(card):
    from PyQt5.QtWidgets import QComboBox
    return [tuple(c.itemText(i) for i in range(c.count()))
            for c in card.settings_popup.findChildren(QComboBox)]


# ---------------------------------------------------------------- structural
def test_setting_popup_has_no_fit_or_relim_controls():
    """The Setting popup is BASIC display only: no Fit/Clear buttons, no relim
    combo, and the (tight, normal) relim choices appear in NO Setting combo."""
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS
    for kind in ("1d", "monitor", "2d", "sites", "hist"):
        card = _card(kind)
        try:
            texts = _button_texts(card)
            assert "Fit" not in texts, (kind, texts)
            assert "Clear" not in texts, (kind, texts)
            # relim is not a Display param anymore (it lives in the Control tab)
            param_keys = {spec.key for spec in PANEL_PARAMS.get(kind, ())}
            assert "relim" not in param_keys, (kind, param_keys)
            # and no Setting combo offers the relim (tight/normal) choice
            assert ("tight", "normal") not in _combo_item_sets(card), kind
            # the card no longer carries the popup-fit handles
            assert not hasattr(card, "fit_combo"), kind
            assert not hasattr(card, "_do_fit"), kind
        finally:
            card.shutdown()


def test_setting_popup_basic_controls_present():
    """The basic controls ARE present: Source (Apply), size, colormap (image
    kinds), the unit cycle button, the lim auto/manual combo + 4 edits, and the
    colorbar checkbox on image kinds only."""
    from PyQt5.QtWidgets import QCheckBox

    # an image kind owns the colorbar toggle + a colormap combo
    img = _card("2d")
    try:
        assert img.cbar_check is not None
        assert isinstance(img.cbar_check, QCheckBox)
        assert "Apply" in _button_texts(img)        # Source apply
        assert "Unit" in _button_texts(img)         # unit cycle
        item_sets = _combo_item_sets(img)
        assert ("auto", "manual") in item_sets      # lim mode combo
        assert any("inferno" in s for s in item_sets)  # colormap combo
        # the four manual-limit edits exist
        assert all(hasattr(img, n) for n in ("xmin_edit", "xmax_edit", "ymin_edit", "ymax_edit"))
    finally:
        img.shutdown()

    # a line kind has unit + lim but NO colorbar toggle
    line = _card("1d")
    try:
        assert line.cbar_check is None
        assert "Unit" in _button_texts(line)
        assert ("auto", "manual") in _combo_item_sets(line)
    finally:
        line.shutdown()


# ------------------------------------------------------------- persistence
def _feed_2d(card):
    arr = np.add.outer(np.linspace(0, 1, 12), np.linspace(0, 2, 12))
    card.refresh({"frame": arr, "shot": 1})


def test_colorbar_toggle_persists_and_reapplies_on_rebuild():
    """colorbar=False round-trips through to_dict/from_dict AND is re-applied to
    the freshly rebuilt plotter (cax hidden), proving it survives a rebuild."""
    from Zou_lab_control.frontend.task_console import PanelConfig

    card = _card("2d", source="value = frame")
    try:
        _feed_2d(card)
        assert card.plotter is not None and card.plotter.cax.get_visible()
        # toggle off via the real popup checkbox -> persisted + applied live
        card.cbar_check.setChecked(False)
        assert card.config.params["colorbar"] is False
        assert not card.plotter.cax.get_visible()

        # round-trip the config, rebuild a fresh card, feed data -> still hidden
        payload = card.config.to_dict()
        cfg2 = PanelConfig.from_dict(payload)
        assert cfg2.params["colorbar"] is False
    finally:
        card.shutdown()

    from Zou_lab_control.frontend.task_console import PanelCard
    card2 = PanelCard(cfg2)
    try:
        _feed_2d(card2)
        assert card2.plotter is not None
        assert not card2.plotter.cax.get_visible()   # re-applied on rebuild
    finally:
        card2.shutdown()


def test_manual_limits_persist_and_reapply_on_rebuild():
    """Manual x/y limits round-trip and are re-applied on rebuild; for a live
    line panel the per-tick relim is frozen (relim_mode='off') so they stick."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = _card("1d", source="value = vec")
    x = np.linspace(0, 10, 40)
    try:
        card.refresh({"vec": np.sin(x), "shot": 1})
        # set manual limits through the popup widgets
        card.lim_combo.setCurrentText("manual")
        card.xmin_edit.setText("2"); card.xmax_edit.setText("8")
        card.ymin_edit.setText("-0.5"); card.ymax_edit.setText("0.5")
        card._on_lim_values()
        assert card.config.params["xlim"] == (2.0, 8.0)
        assert card.config.params["ylim"] == (-0.5, 0.5)
        assert card.plotter.ax.get_xlim() == (2.0, 8.0)
        assert card.plotter.relim_mode == "off"

        # another shot must NOT re-autoscale y away from the manual range
        card.refresh({"vec": 5.0 * np.sin(x), "shot": 2})
        assert card.plotter.ax.get_ylim() == (-0.5, 0.5)

        payload = card.config.to_dict()
    finally:
        card.shutdown()

    cfg2 = PanelConfig.from_dict(payload)
    # JSON-style round-trip yields lists; the values must match
    assert list(cfg2.params["xlim"]) == [2.0, 8.0]
    card2 = PanelCard(cfg2)
    try:
        card2.refresh({"vec": np.sin(x), "shot": 1})
        assert tuple(card2.plotter.ax.get_xlim()) == (2.0, 8.0)
        assert tuple(card2.plotter.ax.get_ylim()) == (-0.5, 0.5)
    finally:
        card2.shutdown()


def test_unit_index_persists_and_reapplies():
    """The unit cycle is stored as unit_index and re-applied on rebuild; an axis
    with a convertible unit (GHz) cycles, a plain axis stays at index 0."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    # a panel whose x-axis label declares a convertible unit -> unit cycles
    card = _card("1d", source="value = vec")
    x = np.linspace(0.1, 1.0, 30)
    try:
        card.refresh({"vec": np.cos(x), "shot": 1})
        # give the axis a convertible unit label, then cycle
        card.plotter.ax.set_xlabel("Detuning (GHz)")
        before = card.plotter.ax.get_xlabel()
        card._on_unit_cycle()
        assert card.config.params["unit_index"] >= 1
        assert card.plotter.ax.get_xlabel() != before   # unit changed on the label
        payload = card.config.to_dict()
        assert payload["params"]["unit_index"] >= 1
    finally:
        card.shutdown()

    # a plain axis (no unit) -> the cycle is a no-op and the index stays 0
    plain = _card("1d", source="value = vec")
    try:
        plain.refresh({"vec": np.cos(x), "shot": 1})
        plain._on_unit_cycle()
        assert int(plain.config.params.get("unit_index", 0)) == 0
    finally:
        plain.shutdown()


def test_control_tab_has_measurement_placeholder_and_processing():
    """The Control tab carries a (disabled) Measurement placeholder reserved for
    P5 plus the processing controls (fit combo, relim combo, Save Fig)."""
    from Zou_lab_control.frontend import devtools as dt

    console = dt.demo_console(shots=4)
    try:
        # the Control tab is the second tab, labelled "Control"
        assert console.tabs.tabText(0) == "Monitor"
        assert console.tabs.tabText(1) == "Control"
        # Measurement placeholder exists, is disabled, and is empty of real controls
        assert console.measurement_group is not None
        assert not console.measurement_group.isEnabled()
        assert console.measurement_group.title() == "Measurement"
        # processing controls live here (fit + relim), not in the Setting popup
        assert console.ed_fit_combo is not None
        assert tuple(console.ed_relim.itemText(i) for i in range(console.ed_relim.count())) == ("tight", "normal")
    finally:
        console.shutdown()
