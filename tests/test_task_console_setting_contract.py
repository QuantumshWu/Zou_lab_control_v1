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
        assert {"Remove", "Edit…"} <= set(texts), texts
        # Save moved ENTIRELY to the Edit tab (it owns the folder picker + the full
        # DataFigure controls) -- the lightweight Setting popup no longer saves.
        assert "Save Fig" not in texts, texts

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


def test_2d_relim_normal_anchors_colorbar_at_zero_tight_brackets():
    """G3: the 2D colorbar (clim) is the image analogue of the 1D y-axis and MUST
    obey relim_mode.  ``normal`` anchors clim at 0 (counts read against zero);
    ``tight`` brackets the data away from 0.  Previously the 2D clim was hard-coded
    tight, so the Setting's normal did nothing -- this pins both modes."""
    normal = _card("2d", source="value = frame", params={"relim": "normal"})
    try:
        frame = np.full((8, 8), 50.0)
        frame[3:5, 3:5] = 250.0                       # data spans 50..250
        normal.refresh({"frame": frame, "shot": 1})
        assert normal.plotter.ylim_min == 0.0         # normal: clim anchored at 0
        assert normal.plotter.ylim_max > 250.0
    finally:
        normal.shutdown()

    tight = _card("2d", source="value = frame", params={"relim": "tight"})
    try:
        frame = np.full((8, 8), 50.0)
        frame[3:5, 3:5] = 250.0
        tight.refresh({"frame": frame, "shot": 1})
        assert tight.plotter.ylim_min > 0.0           # tight: brackets data, not 0
    finally:
        tight.shutdown()


def test_2d_relim_toggle_remaps_colorbar_both_ways():
    """The runtime half of G3: switching the relim combo on a LIVE 2D panel
    re-maps the clim immediately (apply_relim_now), in BOTH directions -- a
    normal<->tight toggle visibly re-colours the image and colorbar."""
    card = _card("2d", source="value = frame", params={"relim": "tight"})
    try:
        frame = np.full((8, 8), 50.0)
        frame[3:5, 3:5] = 250.0
        card.refresh({"frame": frame, "shot": 1})
        assert card.plotter.ylim_min > 0.0            # built tight
        card._on_relim_mode("normal")
        assert card.plotter.ylim_min == 0.0           # re-mapped to normal live
        card._on_relim_mode("tight")
        assert card.plotter.ylim_min > 0.0            # back to tight live
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


def test_setting_keeps_display_params_functional_scalars_go_to_edit():
    """Setting/Edit never DUPLICATE a parameter, and a plot's signal INPUTS are SLOTS,
    not params: the Setting popup renders the colormap chooser (the only display param)
    plus ONE signal picker per :func:`panel_input_slots` slot (signal[0], signal[1], ...),
    while FUNCTIONAL scalar params (length / bins) live in the panel's Edit tab.  So a 2d
    panel's Setting shows cmap but NOT length; a monitor panel shows no param widget; and
    the site map shows its ONE input-slot picker (the occupancy signal; its centres + frame
    underlay auto-resolve from the producing node, not extra slots)."""
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS, panel_input_slots

    # the ONLY display=True param is the colormap chooser; every other param is a
    # FUNCTIONAL scalar that stays in the Edit tab (signal inputs are slots, not params).
    for kind, specs in PANEL_PARAMS.items():
        for spec in specs:
            assert spec.display is (spec.key == "cmap"), (kind, spec.key)

    # a 2d card's Setting popup renders the colormap chooser (display) + its ONE signal
    # slot; a monitor card renders no param widget and one slot.
    img = _card("2d")
    try:
        assert set(img.param_widgets) == {"cmap"}
        assert len(img.slot_combos) == len(panel_input_slots("2d")) == 1
    finally:
        img.shutdown()
    mon = _card("monitor")
    try:
        assert mon.param_widgets == {}      # 'length' moved to the Edit tab
        assert len(mon.slot_combos) == 1
    finally:
        mon.shutdown()
    # the site map's Setting: cmap (display param) + ONE input-slot picker (the occupancy
    # signal); its centres + frame underlay are auto-resolved from the SAME producing node,
    # so the user picks just the one signal (centers/image are NOT slots or display params).
    sites = _card("sites")
    try:
        assert set(sites.param_widgets) == {"cmap"}
        assert len(sites.slot_combos) == len(panel_input_slots("sites")) == 1
    finally:
        sites.shutdown()


def test_multi_signal_slots_let_an_expression_index_signal_i():
    """#6: a non-site panel can ADD signal slots so the source expression combines several
    signals.  ONE slot -> ``signal`` is the scalar value (so ``value = signal`` plots it);
    TWO+ slots -> ``signal`` is a list, so ``value = signal[0] - signal[1]`` reads each slot
    by index.  Add seeds the canonical two-signal expression; remove restores ``value = signal``."""
    import numpy as np
    from Zou_lab_control.frontend.task_console import PanelCard

    card = _card("1d", source="value = signal")
    try:
        assert len(card.slot_combos) == 1               # one slot by default
        card.config.inputs = ["a"]
        ns = {"a": np.arange(4.0), "b": np.ones(4)}
        scalar = card._signal_expr().signal_for(ns, resolve=card._frame_resolve)
        assert np.allclose(scalar, np.arange(4.0))       # single slot -> scalar `signal`

        card._add_signal_slot()                          # + signal
        assert len(card.slot_combos) == 2
        assert card.config.source == "value = signal[0] - signal[1]"   # seeded default
        card.config.inputs = ["a", "b"]
        sig = card._signal_expr().signal_for(ns, resolve=card._frame_resolve)
        assert isinstance(sig, list) and len(sig) == 2   # >1 slot -> list `signal`
        assert np.allclose(sig[0], ns["a"]) and np.allclose(sig[1], ns["b"])
        # the difference expression the user would type evaluates to signal[0]-signal[1]
        assert np.allclose(np.asarray(sig[0]) - np.asarray(sig[1]), np.arange(4.0) - 1.0)

        card._remove_signal_slot()                       # − signal
        assert len(card.slot_combos) == 1
        assert card.config.source == "value = signal"    # restored
    finally:
        card.shutdown()


def test_signal_combo_groups_signals_under_their_producer():
    """#5b: the signal picker is a TWO-LEVEL list -- a non-selectable header per producing node,
    its signals indented beneath -- so the producer is named ONCE (the group) instead of being
    repeated in every signal label (which made the labels too long)."""
    from Zou_lab_control.frontend.task_console import grouped_signal_items

    # the SINGLE shared builder every signal picker uses (plot panels AND the logic-node source)
    items = grouped_signal_items(
        ["occupied", "rate"],
        {"occupied": ["occupancy"], "rate": ["loading"]},
        {"occupied": "(N,)", "rate": "(N,)"})
    # producer headers carry a None bare-name; signals carry their bare name and are indented
    headers = [disp for disp, bare in items if bare is None]
    signals = [(disp, bare) for disp, bare in items if bare is not None]
    assert "occupancy" in headers and "loading" in headers       # grouped by producer
    assert {bare for _disp, bare in signals} == {"occupied", "rate"}
    # the signal labels do NOT repeat the producer name (that was the "too long" bug)
    assert all("occupancy" not in disp and "loading" not in disp for disp, _b in signals)


def test_display_param_change_rerenders_when_source_is_stopped():
    """White-screen regression: a display-only change (e.g. the colormap) must take effect
    IMMEDIATELY.  ``_set_param`` tears the plotter down and normally waits for the next hub
    tick to rebuild -- but a STOPPED measurement freezes ``hub.version``, so ``_tick`` would
    never call ``refresh`` again and the panel would stay blank (white).  Here NO tick runs
    after the change: the plot must still be rebuilt from the last namespace, not None."""
    card = _card("2d", source="value = frame")
    try:
        card.refresh({"frame": np.random.rand(8, 8), "shot": 1})   # one render -> caches the namespace
        assert card.plotter is not None
        card._set_param("cmap", "plasma")                           # different value; NO tick follows
        assert card.config.params["cmap"] == "plasma"
        assert card.plotter is not None, "colormap change while stopped blanked the panel (white screen)"
    finally:
        card.shutdown()


def test_setting_has_per_panel_update_combo():
    """Each panel's Setting carries its OWN refresh-rate combo (UPDATE_INTERVALS); picking a
    rate persists to config.params['update_ms'] and signals the console to re-base its timer."""
    from Zou_lab_control.frontend.task_console import UPDATE_INTERVALS, DEFAULT_UPDATE_MS
    card = _card("1d")
    try:
        assert hasattr(card, "update_combo")
        assert [card.update_combo.itemData(i) for i in range(card.update_combo.count())] == list(UPDATE_INTERVALS)
        assert card.config.update_ms == DEFAULT_UPDATE_MS
        fired = []
        card.update_interval_changed.connect(lambda: fired.append(1))
        card.update_combo.setCurrentText("100 ms")
        assert card.config.update_ms == 100      # persisted (round-trips with the layout)
        assert fired                              # console asked to re-base the shared timer
    finally:
        card.shutdown()


def test_console_timer_base_is_the_fastest_panel():
    """The shared timer ticks at the SMALLEST panel update_ms (the harmonic base that divides
    every other rate, so panels stay phase-aligned); adding a fast 100 ms panel speeds it up."""
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, default_console_state, PanelConfig, PanelCard, DEFAULT_UPDATE_MS)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    console = TaskConsole(hub=SignalHub(), state=default_console_state())
    console._timer.stop()
    try:
        assert console._base_interval_ms == DEFAULT_UPDATE_MS          # default board: 400 ms
        card = PanelCard(PanelConfig(kind="1d", size="1x2", params={"update_ms": 100}),
                         names_provider=console.hub.names)
        console._attach_card(card)
        assert console._base_interval_ms == 100 and console._timer.interval() == 100
    finally:
        console.shutdown()
