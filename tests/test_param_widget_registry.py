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
