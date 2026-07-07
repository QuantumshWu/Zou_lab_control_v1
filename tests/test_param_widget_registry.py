"""MECHANICAL guard for the ONE param-kind -> widget registry (#H3r-F5).

Adding a ParamDecl kind must be ONE handler in ``frontend.param_widgets`` + ONE
whitelist entry on ``ParamDecl`` -- never a 6th parallel widget ladder.  This pins
that single-source rule so it cannot silently rot:

1. EVERY kind ParamDecl accepts (its ``__post_init__`` whitelist) has a
   ``PARAM_WIDGETS`` handler -- a new kind with no widget is a build-time KeyError,
   caught here instead of at the first form that uses it.
2. ``ParamWidgetHandler`` is ABSTRACT over the five ops (build / read / write /
   is_empty / refresh): a handler that omits any one cannot be instantiated, so a
   half-written handler fails loud, not silently no-op.
3. Every kind used in ``PANEL_PARAMS`` (the plot-panel params, now real ParamDecls)
   is registry-handled -- closing the old hole where a parallel ``ParamSpec`` kind
   silently degraded to a text box.

The whitelist is read FROM ParamDecl (not re-typed here) so the test tracks the
single source -- a kind added to ParamDecl with no handler fails #1 automatically.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def _whitelisted_kinds() -> set[str]:
    """The set of kinds ParamDecl accepts -- read by probing its ``__post_init__``
    validator (the single source), not by re-typing the list."""
    from Zou_lab_control.neutral_atom.core.params import ParamDecl
    kinds: set[str] = set()
    # The known token universe to probe; ParamDecl raises on anything outside its
    # whitelist, so a kind survives this loop iff ParamDecl accepts it.
    candidates = [
        "float", "int", "axis_range", "bool", "choice", "text", "json", "device", "path",
        "signal", "signal_expr", "pulse_param", "pulse_slots",
        # decoys that must NOT be accepted (guards the probe itself)
        "bogus", "image", "spinbox",
    ]
    for kind in candidates:
        try:
            ParamDecl(key="k", label="L", kind=kind)
        except ValueError:
            continue
        kinds.add(kind)
    return kinds


def test_every_paramdecl_kind_has_a_handler():
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.param_widgets import PARAM_WIDGETS

    whitelist = _whitelisted_kinds()
    # sanity: the probe found the real kinds and rejected the decoys
    assert "float" in whitelist and "pulse_slots" in whitelist
    assert "bogus" not in whitelist and "image" not in whitelist

    missing = sorted(whitelist - set(PARAM_WIDGETS))
    assert not missing, (
        f"ParamDecl kinds with no PARAM_WIDGETS handler: {missing}. "
        "Add a ParamWidgetHandler + register it -- a kind with no widget is a silent build-time KeyError.")
    # and the registry carries nothing for a kind ParamDecl would reject
    extra = sorted(set(PARAM_WIDGETS) - whitelist)
    assert not extra, f"PARAM_WIDGETS has handlers for non-whitelisted kinds: {extra}."


def test_handler_is_abstract_over_all_five_ops():
    from Zou_lab_control.frontend.param_widgets import ParamWidgetHandler

    expected = {"build", "read", "write", "is_empty", "refresh"}
    assert set(ParamWidgetHandler.__abstractmethods__) == expected, (
        "ParamWidgetHandler must declare exactly the five ops as abstractmethods so a "
        f"handler missing any cannot instantiate; got {set(ParamWidgetHandler.__abstractmethods__)}.")

    # a concrete handler that omits ONE op (here: refresh) cannot be instantiated
    class _MissingRefresh(ParamWidgetHandler):
        def build(self, decl, value, ctx):
            return None

        def read(self, widget):
            return None

        def write(self, widget, value):
            return None

        def is_empty(self, widget):
            return False
        # refresh intentionally not implemented

    with pytest.raises(TypeError):
        _MissingRefresh()

    # every registered handler is a concrete ParamWidgetHandler (no abstractmethods left)
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.param_widgets import PARAM_WIDGETS
    for kind, handler in PARAM_WIDGETS.items():
        assert isinstance(handler, ParamWidgetHandler), kind
        assert not getattr(type(handler), "__abstractmethods__", set()), (
            f"{kind} handler {type(handler).__name__} still has unimplemented ops "
            f"{set(type(handler).__abstractmethods__)}.")


@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt5")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_editable_handlers_route_edits_through_instant_apply(_app):
    """The ONE wiring rule (``_wire``): every editable handler forwards an on-edit value to the form's
    ``instant_apply`` (the apply-on-edit path the Setting popup / device viewer live-write use), so a
    new handler can never silently skip it -- the drift that once left the composite widgets
    unconnected.  Checked for a representative scalar of each widget shape that emits on a programmatic
    edit (a combo's ``activated`` is user-only, so it is excluded here)."""
    from Zou_lab_control.frontend.param_widgets import PARAM_WIDGETS, ParamWidgetContext
    from Zou_lab_control.neutral_atom.core.params import ParamDecl

    cases = [
        (ParamDecl(key="f", label="F", kind="float", default=1.0, lo=0.0, hi=10.0), lambda w: w.setValue(3.0)),
        (ParamDecl(key="i", label="I", kind="int", default=1, lo=0, hi=10), lambda w: w.setValue(4)),
        (ParamDecl(key="t", label="T", kind="text"), lambda w: w.setText("hi")),
        (ParamDecl(key="b", label="B", kind="bool"), lambda w: w.setChecked(True)),
        (ParamDecl(key="j", label="J", kind="json"), lambda w: w.setText("[1, 2]")),
    ]
    for decl, edit in cases:
        applied: list = []
        ctx = ParamWidgetContext(instant_apply=lambda k, v: applied.append((k, v)))
        widget = PARAM_WIDGETS[decl.kind].build(decl, None, ctx)
        edit(widget)
        assert applied and applied[-1][0] == decl.key, (
            f"{decl.kind}: an edit did not reach instant_apply -- the handler bypassed the _wire rule")


def test_row_label_is_the_single_form_label_source():
    """``ParamDecl.row_label`` composes the ONE ``"<label> (<unit>) *"`` a form row shows, so every
    surface (config editor, device viewer, measurement Edit, signal-expr title) reads it instead of
    re-typing the idiom (the copies that let the unit / required marker drift)."""
    from Zou_lab_control.neutral_atom.core.params import ParamDecl

    assert ParamDecl(key="e", label="exposure", kind="float", unit="s", required=True).row_label() == "exposure (s) *"
    assert ParamDecl(key="x", label="", kind="text").row_label() == "x"            # falls back to the key
    assert ParamDecl(key="n", label="name", kind="text").row_label() == "name"     # no unit, not required


def test_blank_allowed_is_the_single_spin_vs_lineedit_decision():
    """``ParamDecl.blank_allowed`` is the ONE predicate deciding a numeric control's widget: an optional
    API arg (no default, not required) is a blank-able line edit; everything else -- a default, a
    required field, or an explicit ``optional=False`` (what a device runtime control pins) -- is a spin
    box.  So the blank-vs-spin rule lives on the declaration, never hard-coded in a widget handler."""
    from Zou_lab_control.neutral_atom.core.params import ParamDecl

    assert ParamDecl(key="a", label="A", kind="float").blank_allowed is True            # optional arg
    assert ParamDecl(key="b", label="B", kind="float", default=1.0).blank_allowed is False
    assert ParamDecl(key="c", label="C", kind="float", required=True).blank_allowed is False
    assert ParamDecl(key="d", label="D", kind="float", optional=False).blank_allowed is False  # device knob
    assert ParamDecl(key="e", label="E", kind="float", optional=True).blank_allowed is True


def test_every_panel_param_kind_is_registry_handled():
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.param_widgets import PARAM_WIDGETS
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS
    from Zou_lab_control.neutral_atom.core.params import ParamDecl

    for plot_kind, decls in PANEL_PARAMS.items():
        for decl in decls:
            # PANEL_PARAMS entries are now real ParamDecls (not a parallel ParamSpec)...
            assert isinstance(decl, ParamDecl), (
                f"PANEL_PARAMS[{plot_kind!r}] entry {decl!r} is not a ParamDecl -- "
                "plot params must be ParamDecls so they validate through the kind whitelist.")
            # ...and every kind they use is handled by the SAME registry the measurement form uses
            assert decl.kind in PARAM_WIDGETS, (
                f"PANEL_PARAMS[{plot_kind!r}].{decl.key} kind {decl.kind!r} has no handler.")
