"""The tree signal picker's data layer moved in with the widget it fills.

`FluentTreeComboBox` was migrated into `zlc_frontend.qt_widgets`, but the helpers
that turn a set of signal names into its collapsible groups - and that read a
selection back out - stayed behind in the old monolith's
`Zou_lab_control/frontend/param_widgets.py`.  A new consumer could therefore get
the widget from the new package and then had to reach into the old tree to fill
it, which is the reverse of this migration's package direction.  That is the
concrete reason no window under `Zou_lab_control/workbench/` uses the picker at
all, and it blocks the TaskConsole's per-panel source picker.

The cluster is pure Qt-combo + plain data (no domain types, no hub, no
ParamDecl), so it moved as a unit.  The old module re-exports it rather than
keeping a copy: one owner, no drift.
"""

from __future__ import annotations

from pathlib import Path

from zlc_frontend.qt_widgets import (
    coerce_short_labels,
    fill_grouped_signal_combo,
    grouped_signal_items,
    read_editable_combo,
    signal_state,
    signal_tree_groups,
)
from zlc_frontend.qt_widgets import signal_picker


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (
    "coerce_short_labels",
    "fill_grouped_signal_combo",
    "grouped_signal_items",
    "read_editable_combo",
    "signal_state",
    "signal_tree_groups",
)


def test_the_helpers_are_owned_by_the_widget_package():
    for name in PUBLIC:
        assert getattr(signal_picker, name).__module__ == (
            "zlc_frontend.qt_widgets.signal_picker"
        ), name


def test_the_old_module_re_exports_rather_than_copying():
    """One owner: the old leaf must hand back the very same objects."""

    from Zou_lab_control.frontend import param_widgets

    for name in PUBLIC:
        assert getattr(param_widgets, name) is getattr(signal_picker, name), name

    source = (
        ROOT / "Zou_lab_control" / "frontend" / "param_widgets.py"
    ).read_text(encoding="utf-8")
    # Since the shell salvage the whole legacy module is a forwarding shim onto
    # zlc_frontend.qt_widgets.param_widgets - a STRONGER form of the original
    # "re-export, never copy" requirement this test was written to pin.
    assert "zlc_frontend.qt_widgets.param_widgets" in source
    assert "MOVED to" in source
    for name in PUBLIC:
        assert f"def {name}(" not in source, f"{name} still has a second definition"


def test_the_moved_module_stayed_dependency_closed():
    """It may import this package's widgets and nothing from the old tree."""

    import ast

    path = ROOT / "zlc_frontend" / "qt_widgets" / "signal_picker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("Zou_lab_control", "zlc_neutral_atom", "zlc_workbench")
    for module in imported:
        assert not any(module.startswith(bad) for bad in forbidden), module


def test_grouping_still_reads_the_same_two_signal_states():
    names = ["cam_frame", "roi_sum"]
    formats = {"cam_frame": "1920x1200 uint8"}

    assert signal_state("cam_frame", formats) == "ready"
    assert signal_state("roi_sum", formats) == "waiting"

    # `sources` maps a signal to the LIST of nodes that produce it.
    groups = signal_tree_groups(
        names,
        {"cam_frame": ["Camera"], "roi_sum": ["Camera"]},
        formats,
    )
    assert groups, "one producing node must yield one group"
    producers = [producer for producer, _leaves in groups]
    assert producers == ["Camera"]
    leaves = groups[0][1]
    assert {bare for _label, bare, _full in leaves} == set(names)


def test_the_flat_projection_shares_the_same_grouping():
    items = grouped_signal_items(
        ["a_x", "a_y"],
        {"a_x": ["Node"], "a_y": ["Node"]},
        {"a_x": "(4,)"},
    )
    # One disabled group header plus one row per signal.
    assert items[0] == ("Node", None)
    assert [name for _label, name in items[1:]] == ["a_x", "a_y"]


def test_coerce_short_labels_normalises_one_provider_callback():
    """It takes a CALLBACK, not a mapping, and swallows a failing provider."""

    assert coerce_short_labels(lambda: {"b": "B", "c": ""}) == {"b": "B"}
    assert coerce_short_labels(None) == {}
    assert coerce_short_labels({"a": "A"}) == {}, "a bare mapping is not callable"

    def broken():
        raise RuntimeError("provider blew up")

    assert coerce_short_labels(broken) == {}


def test_fill_grouped_signal_combo_is_re_exported_for_widget_callers():
    """The filler ships with the widget; constructing Qt here is left to the
    existing GUI suites, which already drive the real picker end to end."""

    assert callable(fill_grouped_signal_combo)
    assert fill_grouped_signal_combo.__module__ == (
        "zlc_frontend.qt_widgets.signal_picker"
    )
    assert callable(read_editable_combo)
