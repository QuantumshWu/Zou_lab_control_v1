"""MECHANICAL guard for the ONE plot-kind table (#H3r-F7).

Each plot kind used to be described in SEVERAL parallel places: a ``_normalize_kind`` /
``plot()`` dispatch ladder in ``frontend/live.py`` and the separate ``PANEL_KINDS`` /
``PANEL_INPUT_FORMAT`` / ``PANEL_INPUT_SLOTS`` / ``PANEL_SINGLE_SLOT_KINDS`` literals in
``frontend/task_console.py``.  They are now collapsed into ONE declarative table --
``live.PLOT_KINDS`` (a tuple of frozen ``PlotKind`` records) -- that both ``live.plot()``
and the task_console PANEL_* lookups READ.

This test encodes the single-source contract so it cannot regress to parallel literals:

1. every kind in the table has a real ``BaseLivePlot`` subclass ``cls`` and a valid
   ``render_family`` ("1D" / "2D") that MATCHES its class attribute;
2. the task_console PANEL_* dicts EQUAL exactly what the table derives (no drift) -- so a
   hand edit that desyncs them from the table fails the build;
3. every panel kind appears in the console Add-Panel menu;
4. a built plot of each kind installs a PlotState whose family is in the known set, and the
   declared ``render_family`` agrees with the legacy "an image axes => 2D" artist heuristic
   (the behaviour-preserving guarantee of the data_figure switch).

Run with ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import Zou_lab_control.frontend as zf
from Zou_lab_control.frontend import live as live_mod
from Zou_lab_control.frontend.live import BaseLivePlot, PLOT_KINDS, PLOT_KIND_BY_KEY, PlotKind


_FIT_FAMILIES = {"1D", "2D"}                 # the resolved DataFigure fitting families
_DECLARED_FAMILIES = {"1D", "2D", "auto"}    # what a PlotKind may DECLARE ("auto" = artist-derived)


def test_table_is_the_single_source_every_kind_has_a_baseliveplot_class():
    """Each PlotKind record carries a real BaseLivePlot subclass + a valid render_family that
    matches the class' own attribute (the class is the authority; the table mirrors it)."""
    assert PLOT_KINDS, "PLOT_KINDS table must not be empty"
    assert {pk.key for pk in PLOT_KINDS} == set(PLOT_KIND_BY_KEY)
    for pk in PLOT_KINDS:
        assert isinstance(pk, PlotKind)
        assert inspect.isclass(pk.cls) and issubclass(pk.cls, BaseLivePlot), (
            f"PLOT_KINDS[{pk.key!r}].cls must be a BaseLivePlot subclass, got {pk.cls!r}")
        assert pk.render_family in _DECLARED_FAMILIES, (pk.key, pk.render_family)
        # the table mirrors the class attribute -- they cannot disagree
        assert pk.cls.render_family == pk.render_family, (
            f"{pk.key}: table render_family {pk.render_family!r} != class "
            f"{pk.cls.__name__}.render_family {pk.cls.render_family!r}")


def test_panel_dicts_are_derived_from_the_table_no_drift():
    """The task_console PANEL_* lookups must EQUAL exactly what the table derives -- they are
    derived, not hand-maintained, so any drift (a stale literal) fails here.

    EVERY plot kind is a console PANEL kind (it renders through the SAME PanelCard, so a saved figure of
    any kind -- incl. ``pulse`` -- seeds a normal panel): ``PANEL_KINDS`` / ``PANEL_INPUT_FORMAT`` /
    ``PANEL_INPUT_SLOTS`` / ``PANEL_SINGLE_SLOT_KINDS`` derive from the WHOLE table.  The ``panel`` flag
    ONLY gates the live Add-Panel dropdown, so ``ADDABLE_PANEL_KINDS`` is the ``panel=True`` subset."""
    from Zou_lab_control.frontend import task_console as tc

    all_kinds = list(PLOT_KINDS)
    addable_kinds = [pk for pk in PLOT_KINDS if pk.panel]

    # PANEL_KINDS (label + PanelConfig validation) covers EVERY kind, in table order
    assert list(tc.PANEL_KINDS) == [pk.key for pk in all_kinds]
    assert tc.PANEL_KINDS == {pk.key: pk.label for pk in all_kinds}
    assert tc.PANEL_INPUT_FORMAT == {pk.key: pk.input_format for pk in all_kinds}
    assert tc.PANEL_INPUT_SLOTS == {pk.key: pk.input_slots for pk in all_kinds if pk.input_slots}
    assert tc.PANEL_SINGLE_SLOT_KINDS == frozenset(pk.key for pk in all_kinds if pk.single_slot)

    # ADDABLE_PANEL_KINDS (the live Add-Panel dropdown) is the panel=True subset, in menu order
    assert list(tc.ADDABLE_PANEL_KINDS) == [pk.key for pk in addable_kinds]
    assert tc.ADDABLE_PANEL_KINDS == {pk.key: pk.label for pk in addable_kinds}
    assert "pulse" in tc.PANEL_KINDS, "pulse is a real panel kind (seedable)"
    assert "pulse" not in tc.ADDABLE_PANEL_KINDS, "pulse is NOT offered in the live Add-Panel dropdown"

    # the helper accessors agree with the table too (every kind)
    for pk in all_kinds:
        assert tc.panel_allows_multi_slot(pk.key) is (not pk.single_slot)
        expected_slots = pk.input_slots if pk.input_slots else (("signal", "", "the hub signal to plot"),)
        assert tc.panel_input_slots(pk.key) == expected_slots


def test_every_panel_kind_appears_in_add_panel_menu():
    """Every kind flagged ``panel=True`` in the table is offered in the console Add-Panel combo
    (and the notebook-only ``pulse`` diagram is NOT) -- the menu reads the table, no parallel list."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    exp = na.connect("virtual")
    try:
        console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                              tasks=exp.readout.task_specs(), window_px=(900, 600))
        console._timer.stop()
        # the Plot: entries carry the bare kind key as itemData
        menu_keys = {console.kind_combo.itemData(i)
                     for i in range(console.kind_combo.count())}
        for pk in PLOT_KINDS:
            if pk.panel:
                assert pk.key in menu_keys, f"panel kind {pk.key!r} missing from Add-Panel menu"
            else:
                assert pk.key not in menu_keys, f"non-panel kind {pk.key!r} leaked into Add-Panel menu"
        console.shutdown()
    finally:
        exp.close()


def _build_one(pk: PlotKind):
    """Build ONE plotter of kind ``pk`` (display off) with kind-appropriate data."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 50).reshape(-1, 1)
    y = np.sin(np.linspace(0, 6, 50)).reshape(-1, 1)
    if pk.key == "2d":
        xy = np.column_stack([np.tile(np.arange(7), 7), np.repeat(np.arange(7), 7)])
        return zf.plot(xy, rng.normal(size=49), kind="2d", display=False, update="once")
    if pk.key == "sites":
        centers = np.column_stack([(np.arange(9) % 3) * 5.0, (np.arange(9) // 3) * 5.0])
        return zf.plot(centers, rng.random(9), kind="sites", display=False, update="once")
    if pk.key == "hist":
        vals = np.concatenate([rng.normal(3, 1, 150), rng.normal(40, 4, 160)])
        return zf.plot(vals, kind="hist", display=False, update="once")
    if pk.key == "pulse":
        return zf.plot([{"channel": "a", "start": 0.0, "duration": 1.0, "level": 1}],
                       kind="pulse", display=False)
    # 1d / monitor
    return zf.plot(x, y, kind=pk.key, display=False, update="once")


def test_built_plot_family_matches_declared_and_legacy_artist_heuristic():
    """For every kind the built plot's DataFigure fitting family is in the known set and EQUALS
    the legacy "an image axes => 2D" artist derivation -- so reading the declared family (instead
    of re-deriving from artists) preserves behaviour.  A kind that declares a concrete "1D"/"2D"
    must additionally match it; an ``"auto"`` kind (the site map) is intentionally artist-derived
    per figure, so it only has to agree with the heuristic, not a fixed string."""
    for pk in PLOT_KINDS:
        p = _build_one(pk)
        try:
            assert isinstance(p, pk.cls), f"{pk.key}: plot() built {type(p).__name__}, not {pk.cls.__name__}"
            df = p.to_data_figure()
            assert df.plot_type in _FIT_FAMILIES, (pk.key, df.plot_type)
            legacy = "2D" if df._ax.images else "1D"
            assert df.plot_type == legacy, (
                f"{pk.key}: resolved family {df.plot_type!r} disagrees with the legacy artist "
                f"heuristic {legacy!r} -- the data_figure switch is NOT behaviour-preserving")
            if pk.render_family in _FIT_FAMILIES:        # a concretely-declared kind must match
                assert df.plot_type == pk.render_family, (
                    f"{pk.key}: DataFigure family {df.plot_type!r} != declared {pk.render_family!r}")
        finally:
            plt.close(p.fig)
