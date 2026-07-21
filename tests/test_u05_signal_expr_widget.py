"""The signal-expression composite moved, and the factory port that reached for it is gone.

``ParamWidgetContext`` used to carry a ``signal_expr_factory``.  Its own docstring said
why: the composite widgets "live in ``task_console``… injected so this module needn't
import them".  That is an inversion whose only cause was placement.  With the widget in
``zlc_frontend.qt_widgets`` the handler builds it directly, and the port is retired -
one fewer callback the console has to remember to pass.

Every primitive the widget uses was already migrated (the grouped picker, the editable
combo reader, the fluent leaves, the seed rule, the help literal), so this is a
consumer joining its dependencies rather than a new copy of anything.

Round-trips below are a GOLDEN captured from the legacy class immediately before the
move, including the empty and ``None`` cases that fall back to a one-slot default.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).
DIES_WITH_PARTIAL = {
    "test_the_legacy_name_is_the_same_class": 'Zou_lab_control/frontend/task_console.py',
}

import ast
import pathlib

import pytest

from zlc_frontend.qt_widgets import SignalExprWidget, ensure_qt_app

NAMES = ["cam/frame", "scan/points", "fit/x0"]

GOLDEN = [
    ({}, {"inputs": ["frame_0"], "source": "value = signal"}, 1),
    (None, {"inputs": ["frame_0"], "source": "value = signal"}, 1),
    ({"inputs": [], "source": ""}, {"inputs": ["frame_0"], "source": "value = signal"}, 1),
    ({"inputs": ["cam/frame"], "source": "value = signal"},
     {"inputs": ["cam/frame"], "source": "value = signal"}, 1),
    ({"inputs": ["cam/frame", "scan/points"], "source": "value = signal[0] - signal[1]"},
     {"inputs": ["cam/frame", "scan/points"], "source": "value = signal[0] - signal[1]"}, 2),
    ({"inputs": ["fit/x0"], "source": "value = signal * 2"},
     {"inputs": ["fit/x0"], "source": "value = signal * 2"}, 1),
    ({"inputs": ["cam/frame", "scan/points", "fit/x0"], "source": "value = signal[2]"},
     {"inputs": ["cam/frame", "scan/points", "fit/x0"], "source": "value = signal[2]"}, 3),
]


def _widget():
    ensure_qt_app()
    return SignalExprWidget(signals_provider=lambda: NAMES, sources_provider=lambda: {},
                            formats_provider=lambda: {}, labels_provider=lambda: {},
                            title="Source")


@pytest.mark.parametrize("seed, expected, slots", GOLDEN)
def test_every_captured_round_trip_is_reproduced(seed, expected, slots):
    """The move must be invisible, including the empty/None fallback."""

    widget = _widget()
    widget.set_value(seed)
    assert widget.values_dict() == expected
    assert len(widget.slot_combos) == slots


def test_the_context_no_longer_carries_a_factory_for_it():
    """The port is retired, not merely unused - the field is gone.

    A field left behind would keep the console passing a callback nobody reads, which
    is exactly the kind of residue that outlives the reason for it.
    """

    from zlc_frontend.qt_widgets.param_widgets import ParamWidgetContext

    fields = set(ParamWidgetContext.__dataclass_fields__)
    assert "signal_expr_factory" not in fields
    # This test once also asserted that pulse_slots_factory was STILL here, to stay honest
    # that only one composite had moved.  It has since moved too (S5-shell(p)), so that
    # half is now the newer file's job - keeping it would have been a stale claim, which
    # is precisely what the two-way assertion existed to surface.


def test_the_handler_builds_the_widget_itself():
    """Structural: a handler still calling a factory would pass every behaviour test."""

    from zlc_frontend.qt_widgets import param_widgets

    tree = ast.parse(pathlib.Path(param_widgets.__file__).read_text(encoding="utf-8"))
    handler = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef) and node.name == "SignalExprHandler")
    build = next(node for node in handler.body
                 if isinstance(node, ast.FunctionDef) and node.name == "build")
    constructed = {node.func.id for node in ast.walk(build)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "SignalExprWidget" in constructed
    assert "signal_expr_factory" not in ast.dump(build)


def test_the_widget_module_pulls_in_no_renderer():
    """Qt is allowed here; Matplotlib is not - qt_widgets may never import it."""

    from zlc_frontend.qt_widgets import signal_expr_widget

    tree = ast.parse(pathlib.Path(signal_expr_widget.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(m.split(".")[0] == "matplotlib" for m in modules), modules


def test_the_legacy_name_is_the_same_class():
    """The shell alias must BE the moved class, not a lookalike left behind."""

    from Zou_lab_control.frontend.task_console import _SignalExprWidget

    assert _SignalExprWidget is SignalExprWidget
