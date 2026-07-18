from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]


def _editor_types():
    from zlc_frontend.image_display import (
        ImageColormap,
        ImageDisplayState,
        image_display_form_spec,
    )
    from zlc_frontend.qt_widgets.fluent import ensure_qt_app
    from zlc_frontend.qt_widgets.image_display import FluentImageDisplayEditor

    ensure_qt_app()
    return (
        FluentImageDisplayEditor,
        ImageDisplayState,
        ImageColormap,
        image_display_form_spec,
    )


def test_editor_leaf_keeps_headless_root_qt_free_and_has_no_runtime_owner() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import zlc_frontend\n"
                "assert 'PyQt5' not in sys.modules\n"
                "assert 'zlc_frontend.qt_widgets.image_display' not in sys.modules\n"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    tree = ast.parse(
        (ROOT / "zlc_frontend/qt_widgets/image_display.py").read_text(
            encoding="utf-8"
        )
    )
    roots = {
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module.split(".", 1)[0]]
            if isinstance(node, ast.ImportFrom) and node.module
            else []
        )
    }
    assert roots.isdisjoint(
        {"Zou_lab_control", "zlc_neutral_atom", "zlc_workbench", "zlc_pulse"}
    )


def test_instances_share_the_one_form_spec_and_exact_keys() -> None:
    Editor, _State, _Colormap, image_display_form_spec = _editor_types()

    first = Editor()
    second = Editor()
    spec = image_display_form_spec()

    assert first.form.spec is spec
    assert second.form.spec is spec
    assert first.form.keys == second.form.keys == spec.keys
    assert first.form.keys == (
        "relim_mode",
        "colormap",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "color_min",
        "color_max",
    )


def test_dirty_newer_load_preserves_exact_draft_and_marks_stale() -> None:
    Editor, State, Colormap, _spec = _editor_types()
    editor = Editor()
    editor.load(State(revision=3), runtime_color_limits=(10.0, 20.0))
    editor.form.widget_for("x_min").setText("2.25")
    draft = editor.read_all()

    editor.load(
        State(revision=4, colormap=Colormap.VIRIDIS),
        runtime_color_limits=(11.0, 21.0),
    )

    assert editor.read_all() == draft
    assert editor.base_revision == 3
    assert editor.dirty
    assert editor.stale
    assert editor.status_severity == "warning"
    assert "stale" in editor.status_text.lower()


def test_runtime_limits_are_only_empty_field_placeholders() -> None:
    Editor, State, _Colormap, _spec = _editor_types()
    editor = Editor()
    editor.load(State(), runtime_color_limits=(10.0, 20.0))
    low = editor.form.widget_for("color_min")
    high = editor.form.widget_for("color_max")

    assert low.text() == high.text() == ""
    assert low.placeholderText() == "10.0"
    assert high.placeholderText() == "20.0"

    low.setText("12.5")
    editor.mark_runtime_color_limits((30.0, 40.0))
    assert low.text() == "12.5"
    assert high.text() == ""
    assert high.placeholderText() == "40.0"

    low.setText("")
    assert low.text() == ""
    assert low.placeholderText() == "30.0"


def test_apply_validates_then_emits_base_revision_and_exact_raw_values() -> None:
    Editor, State, _Colormap, spec = _editor_types()
    editor = Editor()
    editor.load(State(revision=7))
    emitted: list[tuple[int, object]] = []
    editor.applyRequested.connect(lambda revision, values: emitted.append((revision, values)))

    editor.form.widget_for("x_min").setText("2**3")
    editor.apply_button.click()
    assert emitted == []
    assert editor.status_severity == "error"
    assert "finite decimal" in editor.status_text

    editor.form.widget_for("x_min").setText("1.25")
    editor.form.widget_for("x_max").setText("3.5")
    editor.apply_button.click()

    assert len(emitted) == 1
    revision, values = emitted[0]
    assert revision == 7
    assert tuple(values) == spec().keys
    assert values["x_min"] == 1.25
    assert values["x_max"] == 3.5


def test_cancel_only_requests_owner_reload_and_that_reload_replaces_draft() -> None:
    Editor, State, Colormap, _spec = _editor_types()
    editor = Editor()
    initial = State(revision=1)
    latest = State(revision=2, colormap=Colormap.MAGMA)
    editor.load(initial)
    editor.form.widget_for("y_min").setText("4")
    editor.load(latest)
    assert editor.stale

    requests: list[None] = []

    def reload_from_owner() -> None:
        requests.append(None)
        editor.load(latest, runtime_color_limits=(2.0, 8.0))

    editor.cancelRequested.connect(reload_from_owner)
    editor.cancel_button.click()

    assert requests == [None]
    assert editor.base_revision == 2
    assert not editor.dirty
    assert not editor.stale
    assert editor.read_all()["colormap"] is Colormap.MAGMA
    assert editor.read_all()["y_min"] is None


def test_owner_accept_cleans_only_submitter_and_supports_exact_no_op() -> None:
    Editor, State, Colormap, _spec = _editor_types()
    submitter = Editor()
    concurrent = Editor()
    base = State(revision=5)
    committed = State(revision=6, colormap=Colormap.VIRIDIS)
    submitter.load(base)
    concurrent.load(base)
    submitter.form.widget_for("colormap").setCurrentIndex(
        submitter.form.widget_for("colormap").findData(Colormap.VIRIDIS)
    )
    concurrent.form.widget_for("x_min").setText("1.25")

    submitter.accept_commit(5, committed, runtime_color_limits=(2.0, 9.0))
    concurrent.load(committed, runtime_color_limits=(2.0, 9.0))

    assert submitter.base_revision == 6
    assert not submitter.dirty
    assert not submitter.stale
    assert submitter.read_all()["colormap"] is Colormap.VIRIDIS
    assert concurrent.base_revision == 5
    assert concurrent.dirty
    assert concurrent.stale
    assert concurrent.read_all()["x_min"] == 1.25

    submitter.form.widget_for("x_min").setText("3")
    assert submitter.dirty
    submitter.accept_commit(6, committed)
    assert submitter.base_revision == 6
    assert not submitter.dirty
    assert not submitter.stale
    assert submitter.read_all()["x_min"] is None
