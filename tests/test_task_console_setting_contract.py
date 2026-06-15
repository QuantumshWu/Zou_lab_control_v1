"""MECHANICAL guard for the per-panel Setting popup.

The popup follows confocal_gui's vertical idiom: bold section headers
(Source / Display / Unit / Limits / Panel) with one control per row underneath
as a fixed-width-label + control cell.  It exposes the BASIC display controls
that an experimenter touches per shot:

* signal picker + expression (Source),
* size + colormap (the colorbar COLORSET chooser) + each kind's declarative
  ParamSpec widget (Display),
* Unit cycle + current unit text (Unit),
* auto / manual + 4 x/y limit edits (Limits),
* title, Remove, Edit…, Save Fig (Panel).

What it deliberately does NOT have (locked here so regressions cannot creep
back in):

* NO Fit / Clear / relim controls -- those live in the Control tab (Edit…
  opens it).
* NO colorbar show/hide checkbox -- the colormap chooser IS the colorbar
  colorset chooser; visibility is not a user toggle.

Unit / xlim / ylim ALSO persist into ``PanelConfig.params`` and re-apply on
rebuild (the panel rebuilds whenever its data shape changes, so a toggle
that was not re-applied would silently revert).  Run on the offscreen Qt
platform and build PanelCards directly, so the flaky demo_console GUI
fixture is not pulled in.
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


def _button_texts(card):
    from PyQt5.QtWidgets import QPushButton
    return [b.text() for b in card.settings_popup.findChildren(QPushButton)]


def _combo_item_sets(card):
    from PyQt5.QtWidgets import QComboBox
    return [tuple(c.itemText(i) for i in range(c.count()))
            for c in card.settings_popup.findChildren(QComboBox)]


# ---------------------------------------------------------------- structural
def test_setting_popup_has_no_fit_controls():
    """The Setting popup is BASIC display only: no Fit/Clear buttons.

    The relim combo (``tight`` / ``normal``, confocal naming) IS present as the
    Limits "mode" row -- that's the user-requested confocal layout, and it
    persists into ``config.params["relim"]`` (the same key Control's ed_relim
    writes to, so they stay in sync).  See ``test_setting_relim_combo_writes
    _config_params_relim`` for the relim semantic, and ``test_setting_popup
    _has_unit_and_limit_controls`` for the structural presence check.
    """
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS
    for kind in ("1d", "monitor", "2d", "sites", "hist"):
        card = _card(kind)
        try:
            texts = _button_texts(card)
            assert "Fit" not in texts, (kind, texts)
            assert "Clear" not in texts, (kind, texts)
            # relim is not a declarative PANEL_PARAMS spec -- it has its OWN
            # row in the Limits section (the Limits combo).  Keep the spec set
            # clean of any duplicate "relim" key.
            param_keys = {spec.key for spec in PANEL_PARAMS.get(kind, ())}
            assert "relim" not in param_keys, (kind, param_keys)
            # the card no longer carries the popup-fit handles
            assert not hasattr(card, "fit_combo"), kind
            assert not hasattr(card, "_do_fit"), kind
        finally:
            card.shutdown()


def test_setting_popup_has_unit_and_relim_controls():
    """Confocal vertical layout: Source + Display (size + colormap + relim +
    unit + per-kind ParamSpec rows) + Panel.

    Unit cycle button and relim combo (tight/normal -- confocal_gui naming) are
    PRESENT.  The popup carries NO manual x/y limit edits (no auto/manual gate
    either), NO cbar show/hide checkbox -- the colormap chooser IS the colorset
    chooser, and interactive ranging is handled by zoom/pan or by Edit… into
    the Control tab."""
    from PyQt5.QtWidgets import QCheckBox

    # an image kind owns the colormap chooser (= colorbar colorset chooser)
    img = _card("2d")
    try:
        texts = _button_texts(img)
        assert "Apply" in texts                                # Source apply
        assert "Unit" in texts                                 # Unit cycle button
        assert {"Remove", "Edit…", "Save Fig"} <= set(texts), texts

        item_sets = _combo_item_sets(img)
        assert any("inferno" in s for s in item_sets), item_sets  # cmap colorset
        assert any("2x2" in s for s in item_sets), item_sets      # size combo
        # relim mode combo uses confocal_gui's NAMING (tight/normal)
        assert ("tight", "normal") in item_sets, item_sets
        # the popup must NOT carry any auto/manual lim mode
        assert ("auto", "manual") not in item_sets, item_sets

        # the present handles (Unit + relim combo)
        for attr in ("unit_button", "unit_label", "lim_combo"):
            assert hasattr(img, attr), attr

        # the restored handlers
        for attr in ("_on_unit_cycle", "_on_relim_mode",
                     "_apply_unit", "_apply_display_params",
                     "_current_unit_text", "_unit_df", "_unit_cycle_len"):
            assert hasattr(img, attr), attr

        # REMOVED clutter that must stay gone (NO manual x/y limits, NO cbar):
        for gone in ("cbar_check", "_on_colorbar_toggle",
                     "_on_lim_values", "_apply_limits",
                     "xmin_edit", "xmax_edit", "ymin_edit", "ymax_edit",
                     "lim_edits"):
            assert not hasattr(img, gone), gone

        # no colorbar show/hide checkbox lives in the popup
        assert img.settings_popup.findChildren(QCheckBox) == []
    finally:
        img.shutdown()

    # a line kind has the same Display section, just no cmap row
    line = _card("1d")
    try:
        texts = _button_texts(line)
        assert "Unit" in texts
        # SAME confocal relim naming on a line kind (tight / normal)
        assert ("tight", "normal") in _combo_item_sets(line)
        assert ("auto", "manual") not in _combo_item_sets(line)
        from PyQt5.QtWidgets import QCheckBox
        assert line.settings_popup.findChildren(QCheckBox) == []
    finally:
        line.shutdown()


# ------------------------------------------------------------- persistence
def test_unit_index_persists_and_reapplies():
    """The unit cycle stores ``unit_index`` and re-applies on rebuild; an axis
    with a convertible unit (GHz) cycles, a plain axis stays at index 0."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    # a panel whose x-axis label declares a convertible unit -> unit cycles
    card = _card("1d", source="value = vec")
    x = np.linspace(0.1, 1.0, 30)
    try:
        card.refresh({"vec": np.cos(x), "shot": 1})
        card.plotter.ax.set_xlabel("Detuning (GHz)")
        before = card.plotter.ax.get_xlabel()
        card._on_unit_cycle()
        assert card.config.params["unit_index"] >= 1
        assert card.plotter.ax.get_xlabel() != before
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


def test_setting_relim_combo_writes_config_params_relim():
    """The Setting popup's mode combo (``tight`` / ``normal`` -- confocal_gui
    naming) writes to ``config.params["relim"]`` -- the SAME key the Control
    tab's ed_relim writes to, so the two are kept in sync via a single source
    of truth.  The persisted value is re-applied to the plotter every rebuild
    via ``_apply_display_params``.  No manual xlim/ylim path exists any more."""
    card = _card("1d", source="value = vec")
    try:
        x = np.linspace(0, 1, 10)
        card.refresh({"vec": np.cos(x), "shot": 1})
        # default is "tight" (the panel_plot default) -- combo reflects it
        assert card.lim_combo.currentText() == "tight"
        # picking "normal" persists onto the same key the Control tab uses
        card.lim_combo.setCurrentText("normal")
        card._on_relim_mode("normal")
        assert card.config.params["relim"] == "normal"
        assert card.plotter.relim_mode == "normal"
        # data-shape change rebuilds the plotter -> relim must reapply
        card.refresh({"vec": 5.0 * np.cos(x[:8]), "shot": 2})
        assert card.plotter.relim_mode == "normal"
    finally:
        card.shutdown()


def test_panel_title_edit_goes_through_frontend_apply_title():
    """Setting's title edit must update the title via ``BaseLivePlot._apply_title``
    (which routes through ``style.apply_title`` and the design-token
    ``title_fontsize()``), NOT raw ``ax.set_title(text)``.  The raw call would
    inherit matplotlib's ``axes.titlesize`` rcParam and visibly shrink/grow the
    title after every edit; the sealed API pins it at the frontend's chosen
    size."""
    from Zou_lab_control.frontend.style import title_fontsize
    card = _card("1d", source="value = vec")
    try:
        x = np.linspace(0, 1, 10)
        card.refresh({"vec": np.cos(x), "shot": 1})
        expected = float(title_fontsize())
        # the title text the panel was BUILT with already sits at expected size
        # (BaseLivePlot.init -> _apply_title -> style.apply_title).  Verify.
        title_artist = card.plotter.ax.title
        assert abs(title_artist.get_fontsize() - expected) < 1e-6
        # NOW edit it via the popup handler; size MUST stay at expected
        card._on_title("renamed via Setting popup")
        assert card.plotter.title == "renamed via Setting popup"
        assert card.plotter.ax.get_title() == "renamed via Setting popup"
        assert abs(card.plotter.ax.title.get_fontsize() - expected) < 1e-6
    finally:
        card.shutdown()


def test_panel_config_roundtrip_persists_setting_keys():
    """The Setting toggles are stored on PanelConfig.params and survive a
    JSON-style round-trip (to_dict + from_dict): unit_index, relim, cmap."""
    from Zou_lab_control.frontend.task_console import PanelConfig
    cfg = PanelConfig(
        kind="2d", title="t", row=0, col=0, size="2x2",
        params={"unit_index": 2, "relim": "normal", "cmap": "inferno"},
    )
    cfg2 = PanelConfig.from_dict(cfg.to_dict())
    assert cfg2.params["unit_index"] == 2
    assert cfg2.params["relim"] == "normal"
    assert cfg2.params["cmap"] == "inferno"


# --------------------------------------------------- no global Control tab
def test_no_control_tab_monitor_only():
    """There is NO global Control tab any more (it was an empty placeholder when
    no measurements were wired).  Monitor is the only permanent tab; every panel
    gets its OWN closable Edit tab on demand, and the measurement form lives in
    each measurement panel's Edit (not a shared launcher), so the console no
    longer carries a global measurement_group / measurement_panel / launcher."""
    from Zou_lab_control.frontend import devtools as dt

    console = dt.demo_console(shots=4)
    try:
        # Monitor is the ONLY permanent tab; no "Control" anywhere.
        assert console.tabs.tabText(0) == "Monitor"
        titles = [console.tabs.tabText(i) for i in range(console.tabs.count())]
        assert "Control" not in titles, titles
        # the empty global launcher is gone (kept as None for stop/poll fallback)
        assert console.measurement_group is None
        assert console.measurement_panel is None
        # the old single-editor handles never came back
        for gone in ("ed_fit_combo", "ed_relim", "ed_xmin", "_editor_card",
                     "_editor_plotter", "_build_editor_tab", "_build_control_tab"):
            assert not hasattr(console, gone), gone
        # per-panel editors are tracked in a registry, opened on demand
        assert console._panel_editors == {}
    finally:
        console.shutdown()


def test_setting_keeps_only_colormap_functional_params_go_to_edit():
    """Setting/Edit never DUPLICATE a parameter: the Setting popup renders only
    DISPLAY params (the colormap / colorset chooser), while FUNCTIONAL plot
    params (length / bins / centers / image) are rendered in the panel's Edit
    tab instead -- so a 2d panel's Setting shows cmap but NOT length, and a
    monitor panel's Setting shows NO functional spec widget."""
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS

    # cmap is the only display=True spec; everything else is functional (Edit).
    for kind, specs in PANEL_PARAMS.items():
        for spec in specs:
            if spec.key == "cmap":
                assert spec.display is True, kind
            else:
                assert spec.display is False, (kind, spec.key)

    # a 2d card's Setting popup renders the colormap chooser (display) but no
    # functional param widget; a monitor card's Setting renders no param widget.
    img = _card("2d")
    try:
        assert set(img.param_widgets) == {"cmap"}
    finally:
        img.shutdown()
    mon = _card("monitor")
    try:
        assert mon.param_widgets == {}      # 'length' moved to the Edit tab
    finally:
        mon.shutdown()
