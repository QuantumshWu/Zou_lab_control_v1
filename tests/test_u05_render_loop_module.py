"""The render worker leaves the legacy tree -- and the reason it could is worth pinning.

For twenty-five slices the ledger treated ``render_loop`` and ``qt_canvas`` as one thing: "这对
Qt+mpl 联姻件", judged jointly at S5-shell(a) and quoted ever since as the reason the whole shell
was stuck behind the §12.5 rework.  Re-derived from source, only ONE of the two is married.
``qt_canvas`` really is -- ``EmbeddedFigureCanvas`` subclasses ``FigureCanvasQTAgg``, so it is a Qt
widget and a Matplotlib canvas at once, which L4650 places nowhere inside ``zlc_frontend``.
``render_loop`` never was: it mentions Matplotlib only in prose, and its actual import closure is
``threading``, ``time`` and ``PyQt5``.  Against L4650 that reads: imports PyQt5, so it must be in
``qt_widgets``; imports no Matplotlib, so it may be.  It was movable the whole time.

The lesson is the test below, not the move: a judgement about two files is not evidence about
either one.  ``test_the_render_worker_is_not_married_to_matplotlib`` re-derives the import closure
on every run, so the claim this move rests on cannot rot into folklore the way the joint one did.

Moving it also fixed a smaller thing on the line being changed.  The Qt-thread fit guard was
registered through ``..neutral_atom.core.fitting`` -- itself a "MOVED to zlc_data.curve_fitting"
shim -- inside a bare ``try/except Exception: pass``.  Naming a shim as the owner is how a second
apparent source is born, and swallowing the failure means a silently UNARMED thread guard: nothing
raises, and a curve fit is free to run on the GUI event loop, which is the exact bug the guard
exists to catch.  It now imports the real owner and does not swallow.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).
DIES_WITH_PARTIAL = {
    "test_the_legacy_path_resolves_to_the_same_objects": 'Zou_lab_control/frontend/render_loop.py',
    "test_the_shim_forwards_live_rather_than_copying_bindings_once": 'Zou_lab_control/frontend/render_loop.py',
    "test_the_two_real_consumers_import_the_owner_not_the_shim": ('Zou_lab_control/frontend/qt_canvas.py', 'Zou_lab_control/frontend/task_console.py'),
    "test_its_neighbour_really_is_married_which_is_why_it_stayed": 'Zou_lab_control/frontend/qt_canvas.py',
}

import ast
import pathlib
import threading

import pytest

import zlc_frontend.qt_widgets as _qt_widgets

_render_loop = _qt_widgets.render_loop

RENDER_THREAD_NAME = _render_loop.RENDER_THREAD_NAME
RenderLoop = _render_loop.RenderLoop

REPO = pathlib.Path(__file__).resolve().parents[1]
MOVED = REPO / "zlc_frontend" / "qt_widgets" / "render_loop.py"
SHIM = REPO / "Zou_lab_control" / "frontend" / "render_loop.py"


def _import_roots(path: pathlib.Path) -> set[str]:
    """Top-level packages the module actually reaches for, by AST, at ANY nesting depth --
    a deferred import inside a function counts, which is how a dependency usually hides."""

    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add(("." * node.level) + (node.module or "").split(".")[0])
    return roots


def test_the_render_worker_is_not_married_to_matplotlib():
    """The claim the move rests on, re-derived every run rather than remembered.

    Prose mentions Matplotlib all over this module -- it describes what the worker composes.
    Only the import closure decides where it may live."""

    roots = _import_roots(MOVED)
    assert "matplotlib" not in roots, roots
    assert "PyQt5" in roots, roots
    assert roots <= {"__future__", "threading", "time", "PyQt5", "zlc_data"}, roots
    assert "matplotlib" in MOVED.read_text(encoding="utf-8"), (
        "if the prose stopped mentioning matplotlib this test would pass vacuously")


def test_its_neighbour_really_is_married_which_is_why_it_stayed():
    """The other half of the same judgement, so 'render_loop moved' is never read as
    'the render pair is done'.  ``qt_canvas`` needs BOTH toolkits and L4650 places it nowhere
    in zlc_frontend; it leaves only via the §12.5 worker-raster rework."""

    # The canvas moved INTO the sanctioned Qt x matplotlib zone -- the exact destiny this
    # judgement predicted for it; its legacy path is a PEP 562 shim.  The proof that the
    # marriage is REAL (both toolkits, a QTAgg base) follows the code.
    canvas = REPO / "zlc_workbench" / "task_console" / "plot_bridge_canvas.py"
    roots = _import_roots(canvas)
    assert "matplotlib" in roots, roots
    tree = ast.parse(canvas.read_text(encoding="utf-8"))
    bases = [base.id for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
             for base in node.bases if isinstance(base, ast.Name)]
    assert any("QTAgg" in name for name in bases), bases


def test_the_placement_axiom_holds_for_the_new_home():
    """L4650 in both directions, checked against the package it landed in."""

    qt_widgets = REPO / "zlc_frontend" / "qt_widgets"
    for path in qt_widgets.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        assert "matplotlib" not in _import_roots(path), path.name

    for path in (REPO / "zlc_frontend").rglob("*.py"):
        if "__pycache__" in path.parts or "qt_widgets" in path.parts:
            continue
        assert "PyQt5" not in _import_roots(path), (
            f"{path.relative_to(REPO).as_posix()} imports PyQt5 outside qt_widgets")


# --- the seam ------------------------------------------------------------------------------

def test_the_thread_guard_is_registered_with_its_real_owner_not_a_shim():
    """``register_solve_thread_guard`` lives in ``zlc_data.curve_fitting``; the legacy
    ``neutral_atom.core.fitting`` path is a forwarding shim to it.  Importing through the shim
    worked, which is exactly why it survived -- and it is how a shim starts looking like an owner."""

    import zlc_data.curve_fitting as owner

    moved = _render_loop

    assert owner._solve_thread_guard is moved._fit_solve_on_gui_thread_guard
    modules = {node.module for node in ast.walk(ast.parse(MOVED.read_text(encoding="utf-8")))
               if isinstance(node, ast.ImportFrom) and node.module}
    assert "zlc_data.curve_fitting" in modules
    assert not any("fitting" in m and m.startswith("Zou_lab_control") for m in modules), modules


def test_the_registration_is_not_wrapped_in_a_swallow():
    """It used to sit in ``try: ... except Exception: pass``.  A swallowed failure there does not
    crash -- it leaves the guard UNARMED, so a curve fit runs on the GUI thread and nothing says
    so.  A lifecycle sentinel must never have a silent fallback."""

    tree = ast.parse(MOVED.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            handlers = ast.dump(node)
            assert "_register_guard" not in handlers, "the guard registration is swallowed again"


def test_the_guard_fires_on_the_qt_thread_and_stays_quiet_off_it():
    """Behaviour, not wiring: the scope is what makes it usable -- a worker-node solve and a
    headless notebook solve must both pass."""

    from PyQt5 import QtWidgets

    moved = _render_loop

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    with pytest.raises(RuntimeError, match="ZLC_FIT_THREAD_GUARD"):
        moved._fit_solve_on_gui_thread_guard()

    off = []
    worker = threading.Thread(target=lambda: off.append(
        moved._fit_solve_on_gui_thread_guard() or "returned"))
    worker.start()
    worker.join()
    assert off == ["returned"]


# --- the ownership protocol, unchanged by the move -------------------------------------------

def test_a_finished_but_undelivered_batch_still_counts_as_busy():
    """The load-bearing one.  Idle means the GUI has CONSUMED the result, not that the worker
    finished computing it -- marking idle any earlier lets the next tick re-submit the same
    figures while the present pass is still queued (two threads, one Agg buffer)."""

    import time

    from PyQt5 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    seen: list = []
    loop = RenderLoop(consumer=seen.append)
    try:
        gate = threading.Event()
        assert loop.submit(lambda: (gate.wait(5.0), {"k": 1})[1])
        assert loop.busy
        assert not loop.submit(lambda: {"k": 2}), "a second job must be refused while busy"
        gate.set()
        time.sleep(0.2)
        assert loop.busy, "finished computing is NOT delivered"
        assert loop.deliver_pending() and seen == [{"k": 1}]
        assert not loop.busy
        assert not loop.deliver_pending(), "delivery is single-consumption"
    finally:
        loop.stop(timeout=5.0)


def test_a_barrier_delivers_rather_than_merely_waits():
    from PyQt5 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    seen: list = []
    loop = RenderLoop(consumer=seen.append)
    try:
        assert loop.submit(lambda: {"k": 3})
        assert loop.barrier(timeout=5.0)
        assert seen == [{"k": 3}], "the caller must return to a settled board"
        assert not loop.busy
        assert loop.barrier(timeout=5.0), "a barrier on an idle loop is a no-op, not a failure"
    finally:
        loop.stop(timeout=5.0)


def test_the_worker_thread_is_named_from_the_one_source_and_goes_away_on_stop():
    """The name is how a compose-phase re-entry recognises it is already ON the worker and must
    not wait on its own in-flight job -- so it is a contract, not a debugging label."""

    from PyQt5 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    loop = RenderLoop(consumer=lambda batch: None)
    # Scoped to THIS loop's own thread object.  Asserting that no thread in the process carries
    # the name is a claim about the whole process, and in a shared pytest process another test's
    # live loop makes it false -- which says nothing about whether THIS one shut down.
    mine = loop._thread
    assert mine.name == RENDER_THREAD_NAME, mine.name
    assert mine.is_alive()
    assert loop.stop(timeout=5.0)
    assert loop.stop(timeout=5.0), "stop is idempotent"
    assert not loop.submit(lambda: {"k": 4}), "a stopped loop accepts nothing"
    assert not mine.is_alive(), "stop must join its own worker, not merely signal it"


# --- the shim ---------------------------------------------------------------------------------

def test_the_legacy_path_resolves_to_the_same_objects():
    from Zou_lab_control.frontend import render_loop as legacy

    assert legacy.RenderLoop is RenderLoop
    assert legacy.RENDER_THREAD_NAME == RENDER_THREAD_NAME
    moved = _render_loop
    assert legacy._fit_solve_on_gui_thread_guard is moved._fit_solve_on_gui_thread_guard


def test_the_shim_forwards_live_rather_than_copying_bindings_once():
    """``globals().update(vars(_moved))`` copies BINDINGS at import, so a global the moved module
    later rebinds leaves the shim frozen -- which is exactly how ``_solve_thread_guard`` stayed
    ``None`` after the frontend armed the real guard."""

    source = SHIM.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"__getattr__", "__dir__"} <= names, names
    assert source.lstrip().startswith('"""MOVED to'), "the Z3 shim marker must open the file"

    # By AST, not by substring: the shim's own docstring SAYS "globals().update(vars(_moved))"
    # while explaining why it must not do that, so a text search matches the prose that exists
    # precisely because the code is right.  Anchor on a real Call node instead.
    eager = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "update"
             and isinstance(node.func.value, ast.Call)
             and isinstance(node.func.value.func, ast.Name)
             and node.func.value.func.id == "globals"]
    assert not eager, "the shim copies bindings once instead of forwarding live"


def test_the_two_real_consumers_import_the_owner_not_the_shim():
    """A shim exists for code nobody has revisited.  Anything this move touched should already
    name the new home, or the shim quietly becomes the path of record again.

    Through the package FACADE, not the submodule path: ``qt_widgets`` is the package's one Qt
    owner and its own guard forbids anyone outside it importing ``zlc_frontend.qt_widgets.<sub>``
    directly, exactly as ``param_widgets`` is reached.  So the assertion is "names the new
    package", and separately that nothing still names the shim."""

    # qt_canvas moved wholesale into the plot_bridge zone (its legacy path is a PEP 562
    # shim); the consumer proof follows the code.
    for relative in ("zlc_workbench/task_console/plot_bridge_canvas.py",
                     "Zou_lab_control/frontend/task_console.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        facade = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Import)
                  and any(alias.name == "zlc_frontend.qt_widgets" for alias in node.names)]
        assert len(facade) == 1, (relative, len(facade))
        assert "_qt_widgets.render_loop." in source, relative
        deep = [node for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
                and node.module.startswith("zlc_frontend.qt_widgets.")]
        assert not deep, (relative, [n.module for n in deep])
        assert "from .render_loop import" not in source, relative
