# -*- coding: utf-8 -*-
"""Contract (#6): a console fit NEVER solves on the Qt event loop.

Two mechanically-enforced rules:

* (#6d) the UNBOUNDED curve fit (``core.fitting.fit_selected`` -> ``_solve_candidates``) is armed by the
  opt-in ``fit_thread_guard`` fixture to RAISE if it runs on the Qt application thread.  Each plot kind's
  fit is driven through the REAL console path and must never trip it: a general fit is a per-panel worker
  node (2-D image, 1-D curve, faceted grid), and a display overlay reconstructs from published params via
  a ``solve=False`` path (no solver) that is exempt.  A notebook main-thread solve stays legal (no
  QApplication guard fire is scoped to a live GUI session).

* (#6c) the BOUNDED domain fits (a hist bimodal, a monitor side-gaussian, a grid hist cell) stay IN-PLOT
  by design: they call ``fit_histogram`` (NOT ``fit_selected``), and ``counts.size`` is bounded by the
  ``bins`` ParamDecl cap declared ONCE in ``PANEL_PARAMS`` -- an O(bins) micro-solve any thread may run.
"""
import threading

import numpy as np
import pytest

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.frontend.task_console import PANEL_PARAMS

from conftest import add_logic_row, make_console, tick


# --------------------------------------------------------------------------- helpers
def _hist_bins_cap() -> int:
    """The single-source cap on a hist fit's bin count -- the ``bins`` ParamDecl ``hi`` in PANEL_PARAMS.
    Derived, never re-typed, so this contract tracks the declaration automatically."""
    bins = next(d for d in PANEL_PARAMS["hist"] if d.key == "bins")
    return int(bins.hi)


def _live_camera_signal(con):
    row = add_logic_row(con, ("camera", "live"))
    con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
    con._start_logic_node(row)
    node = con._logic_nodes[id(row)]
    node.step()
    return row, sorted(node.published_signals())[0]


def _add_panel(con, kind_label, signal, *, source="value = signal", title=None):
    kc = con.kind_combo
    kc.setCurrentIndex(next(i for i in range(kc.count()) if kc.itemText(i) == kind_label))
    con._add_panel()
    card = con.cards[-1]
    if title:
        card.config.title = title
    card.config.inputs = [signal]
    card.source_edit.setText(source)
    card._apply_source()
    con.refresh_once()
    return card


# --------------------------------------------------------------------------- (#6c) bounded exemption
def test_hist_fit_bin_count_is_bounded_by_the_panel_params_cap():
    """The hist domain fit's ``counts.size`` is bounded by the ``bins`` ParamDecl cap (single source in
    PANEL_PARAMS), so the in-plot bimodal solve is a principled O(bins) micro-solve -- never unbounded."""
    from Zou_lab_control.frontend import live as live_mod
    cap = _hist_bins_cap()
    seen = []
    real = live_mod.fit_histogram

    def spy(edges, counts, mode):
        seen.append(int(np.asarray(counts).size))
        return real(edges, counts, mode)

    live_mod.fit_histogram = spy
    try:
        samples = np.concatenate([np.random.normal(0.0, 1.0, 4000),
                                  np.random.normal(8.0, 1.0, 4000)])
        pl = live_mod.plot(samples, kind="hist", bins=cap, fit="double", display=False)
        pl.update(samples)
    finally:
        live_mod.fit_histogram = real
    assert seen, "the hist bimodal fit_histogram was never called"
    assert max(seen) == cap                       # exactly the declared cap -> the bound is real + single-sourced
    assert all(n <= cap for n in seen)


def test_hist_domain_fit_does_not_route_through_the_unbounded_guard(fit_thread_guard):
    """With the guard ARMED on the main thread, a hist bimodal fit still runs (it uses ``fit_histogram``,
    not ``fit_selected``) -- the mechanical proof that the bounded exemption is exempt by construction."""
    from Zou_lab_control.frontend import live as live_mod
    samples = np.concatenate([np.random.normal(0.0, 1.0, 3000),
                              np.random.normal(7.0, 1.0, 3000)])
    pl = live_mod.plot(samples, kind="hist", bins=_hist_bins_cap(), fit="double", display=False)
    pl.update(samples)                            # main-thread bimodal solve, guard armed -> must NOT raise
    assert pl._histogram_fit_result is not None


# --------------------------------------------------------------------------- (#6d) per-kind: off the GUI thread
def _await_fit(con, node, *, timeout=8.0):
    """Drive the console until the per-panel fit node has published a result on its WORKER thread.  Step
    the camera just ONCE (a fresh frame the reactive node wakes on) then poll -- flooding the bounded
    signal journal with frames the slow fit falls behind would raise a benign SignalHistoryGap."""
    import time
    cam = con._logic_nodes[id(con.logic_nodes[0])]
    key = node.prefix + "fit_valid"
    cam.step()
    tick(con)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if key in con.hub.names():
            return True
        tick(con)
        time.sleep(0.02)
    return False


_GUARD_MARK = "Qt application thread"


def _no_guard_violation(node) -> bool:
    """True unless the node's worker latched a THREAD-GUARD violation.  A benign SignalHistoryGap (the
    fast live camera outpacing a slow fit) is unrelated and does not count -- the guard only ever fires
    ON the Qt thread, so the worker never trips it; this is a belt-and-suspenders check."""
    err = getattr(node, "last_error", None)
    return not (err and _GUARD_MARK in str(err))


def test_2d_image_fit_solves_on_the_worker_not_the_gui_thread(fit_thread_guard):
    """A 2-D image centre fit is a per-panel FitProcessor node: with the guard armed, arming + driving the
    fit must never solve on the Qt thread (the node solves on ``zlc-node-*``; the overlay only draws)."""
    from Zou_lab_control.neutral_atom.core.fitting import FitRequest
    from Zou_lab_control.neutral_atom.core.selection import Selection
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam_row, sig = _live_camera_signal(con)
        card = _add_panel(con, "Plot: 2D image", sig, title="fit2d")
        card.set_fit_request(FitRequest("center", selection=Selection.rectangle(0, 320, 0, 240)))
        node = con._logic_nodes.get(id(con._panel_analysis_row(card)))
        assert node is not None
        assert _await_fit(con, node), "the 2-D fit node never published"
        assert _no_guard_violation(node)          # the worker solve never trips the guard
        assert bool(np.asarray(con.hub.latest(node.prefix + "fit_valid")).reshape(-1)[0])
        con._update_fit_overlays()                # overlay push (draw-only) on the main thread -> no solve
    finally:
        con.shutdown()
        exp.close()


def test_1d_curve_fit_solves_on_the_worker_not_the_gui_thread(fit_thread_guard):
    """A 1-D curve fit is likewise a worker node; the guard must never fire on the GUI thread for it."""
    from Zou_lab_control.neutral_atom.core.fitting import FitRequest
    from Zou_lab_control.neutral_atom.core.selection import Selection
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam_row, sig = _live_camera_signal(con)
        # a 1-D panel reading a per-column reduction of the frame (a curve over the frame's columns)
        card = _add_panel(con, "Plot: 1D vector", sig, source="value = signal.mean(axis=0)", title="curve")
        card.set_fit_request(FitRequest("gaussian", selection=Selection()))
        node = con._logic_nodes.get(id(con._panel_analysis_row(card)))
        assert node is not None
        assert _await_fit(con, node), "the 1-D fit node never published"
        assert _no_guard_violation(node)
        con._update_fit_overlays()
    finally:
        con.shutdown()
        exp.close()


def test_monitor_side_gaussian_stays_in_plot_and_off_the_guard(fit_thread_guard):
    """A rolling-monitor side distribution's single-gaussian fit is a BOUNDED domain fit (``fit_histogram``):
    with the guard armed, driving the panel on the main thread must never trip it."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        _cam_row, sig = _live_camera_signal(con)
        card = _add_panel(con, "Plot: Rolling trace", sig, source="value = signal.mean()", title="mon")
        cam = con._logic_nodes[id(con.logic_nodes[0])]
        for _ in range(4):
            cam.step()
            tick(con)                             # side-gaussian fit runs in-plot -> must not raise
        assert card.plotter is not None
    finally:
        con.shutdown()
        exp.close()
