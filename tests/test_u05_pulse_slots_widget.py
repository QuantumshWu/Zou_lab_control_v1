"""The last composite moves, and ParamWidgetContext stops reaching into the console.

``ParamWidgetContext`` carried two zero-arg factories whose only purpose was to build
widgets that happened to live in ``task_console``.  ``signal_expr_factory`` went in the
previous slice; ``pulse_slots_factory`` goes here.  The context now holds no callback
that reaches back into the legacy tree at all - the handlers construct what they need.

Seeding is DEFERRED by design and that is the part worth pinning: ``seed_value`` stores
a payload against its ``program_id``, and only a later ``rebuild`` with a MATCHING id
restores it.  A saved workspace arrives before the template rows are known, so an eager
apply would write into fields that do not exist yet.  The golden below was captured from
the legacy class immediately before the move and covers that, plus the case that is easy
to get wrong: a non-numeric seed drops the key entirely rather than writing text into a
numeric field.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).

import ast
import pathlib

import pytest

from zlc_data.scan_template import ScanColumnSpec
from zlc_frontend.qt_widgets import PulseSlotsWidget, ensure_qt_app

COLUMNS = (ScanColumnSpec(name="s0", lo=-512.0, hi=512.0, is_dac=True, unit=""),)
API_ROWS = (("h0", "c0", "duration", "t_probe", "ns", "100"),
            ("h1", "c1", "duration", "t_hold", "ns", "50"))
SCAN_ROWS = (("c0", "dac", "s0", "", "s0"),)

TEMPLATE = {"api": {"h0": 100.0, "h1": 50.0}, "sweep_kind": "scan_slot", "program": "col"}


def _built(seed=None, *, api=API_ROWS, scan=SCAN_ROWS, rebuilds=1):
    ensure_qt_app()
    widget = PulseSlotsWidget()
    if seed is not None:
        widget.seed_value(seed)
    for _ in range(rebuilds):
        widget.rebuild(api, scan, scan_columns=COLUMNS if scan else (),
                       hardware_program="col" if scan else "", program_id="tpl")
    return widget


def test_a_fresh_rebuild_shows_the_template_values():
    assert _built().values_dict() == {"program_id": "tpl", **TEMPLATE}


def test_a_seed_for_this_program_is_restored_on_the_next_rebuild():
    """The deferred contract: stored at seed time, applied at rebuild time."""

    widget = _built({"program_id": "tpl", "api": {"h0": 250.0, "h1": 7.0},
                     "sweep_kind": "api_slot", "program": "grid"})
    assert widget.values_dict() == {"program_id": "tpl", "api": {"h0": 250.0, "h1": 7.0},
                                    "sweep_kind": "api_slot", "program": "grid"}


def test_a_seed_for_a_different_program_is_ignored():
    """Loading a workspace saved against another template must not poison this one."""

    widget = _built({"program_id": "OTHER", "api": {"h0": 999.0}})
    assert widget.values_dict() == {"program_id": "tpl", **TEMPLATE}


def test_a_non_numeric_seed_drops_that_key_rather_than_writing_text():
    """The easy-to-break case: h0 disappears; the untouched h1 keeps its template value."""

    widget = _built({"program_id": "tpl", "api": {"h0": "abc"}})
    assert widget.values_dict() == {"program_id": "tpl", "api": {"h1": 50.0},
                                    "sweep_kind": "scan_slot", "program": "col"}


def test_rebuilding_the_same_program_twice_changes_nothing():
    assert _built(rebuilds=2).values_dict() == {"program_id": "tpl", **TEMPLATE}


def test_a_template_with_no_slots_reads_back_empty():
    widget = _built(api=(), scan=())
    assert widget.values_dict() == {"program_id": "tpl", "api": {},
                                    "sweep_kind": "", "program": ""}


def test_the_context_carries_no_factory_into_the_legacy_tree_any_more():
    """Both fields retired, not merely unused - this is the point of the two slices."""

    from zlc_frontend.qt_widgets.param_widgets import ParamWidgetContext

    fields = set(ParamWidgetContext.__dataclass_fields__)
    assert not [name for name in fields if name.endswith("_factory")], fields


@pytest.mark.parametrize("handler, built", [("PulseSlotsHandler", "PulseSlotsWidget"),
                                            ("SignalExprHandler", "SignalExprWidget")])
def test_each_handler_builds_its_own_widget(handler, built):
    """Structural: a handler still calling a factory would pass every behaviour test."""

    from zlc_frontend.qt_widgets import param_widgets

    tree = ast.parse(pathlib.Path(param_widgets.__file__).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == handler)
    build = next(n for n in node.body
                 if isinstance(n, ast.FunctionDef) and n.name == "build")
    constructed = {c.func.id for c in ast.walk(build)
                   if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert built in constructed
    assert "_factory" not in ast.dump(build)


def test_the_widget_module_pulls_in_no_renderer():
    from zlc_frontend.qt_widgets import pulse_slots_widget

    tree = ast.parse(pathlib.Path(pulse_slots_widget.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(m.split(".")[0] == "matplotlib" for m in modules), modules


