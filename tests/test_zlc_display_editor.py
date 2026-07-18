from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _types():
    from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
    from zlc_frontend.qt_widgets.display_editor import FluentRevisionedFormEditor
    from zlc_frontend.qt_widgets.fluent import ensure_qt_app

    ensure_qt_app()
    image = FormSpec(
        (
            FormFieldProps(
                "mode",
                "choice",
                "Mode",
                default="tight",
                choices=(FormChoice("Tight", "tight"), FormChoice("Fixed", "fixed")),
            ),
            FormFieldProps("color_min", "float", "Color minimum", default=None),
            FormFieldProps("color_max", "float", "Color maximum", default=None),
        )
    )
    curve = FormSpec(
        (
            FormFieldProps(
                "mode",
                "choice",
                "Mode",
                default="normal",
                choices=(FormChoice("Normal", "normal"), FormChoice("Fixed", "fixed")),
            ),
            FormFieldProps("x_min", "float", "X minimum", default=None),
            FormFieldProps("x_max", "float", "X maximum", default=None),
            FormFieldProps("enabled", "bool", "Enabled", default=True),
        )
    )
    return FluentRevisionedFormEditor, image, curve


def test_two_specs_share_one_parameterized_editor_without_state_semantics() -> None:
    Editor, image_spec, curve_spec = _types()
    image = Editor(
        image_spec,
        "image display",
        runtime_placeholder_fields=("color_min", "color_max"),
    )
    curve = Editor(
        curve_spec,
        "curve display",
        runtime_placeholder_fields=("x_min", "x_max"),
    )

    image.load(
        revision=2,
        semantic_identity=("image", 2, "gray"),
        values={"mode": "tight", "color_min": None, "color_max": None},
        runtime_placeholders={"color_min": "10.0", "color_max": "20.0"},
    )
    curve.load(
        revision=5,
        semantic_identity=("curve", 5, "normal"),
        values={"mode": "normal", "x_min": None, "x_max": None, "enabled": True},
    )

    assert image._form.spec is image_spec
    assert curve._form.spec is curve_spec
    assert image.base_revision == 2
    assert curve.base_revision == 5
    assert image._form.widget_for("color_min").placeholderText() == "10.0"
    assert curve._form.widget_for("x_min").placeholderText() == "(optional)"


def test_clean_reload_dirty_stale_and_observed_identity_guards() -> None:
    Editor, _image_spec, curve_spec = _types()
    clean = Editor(curve_spec, "curve display")
    dirty = Editor(curve_spec, "curve display")
    base_values = {"mode": "normal", "x_min": None, "x_max": None, "enabled": True}
    newer_values = {"mode": "fixed", "x_min": 1.0, "x_max": 3.0, "enabled": True}
    for editor in (clean, dirty):
        editor.load(revision=4, semantic_identity=("curve", 4), values=base_values)
    dirty._form.widget_for("x_min").setText("2.25")
    draft = dirty._form.read_all()

    clean.load(revision=5, semantic_identity=("curve", 5), values=newer_values)
    dirty.load(revision=5, semantic_identity=("curve", 5), values=newer_values)

    assert clean.base_revision == 5
    assert clean._form.read_all() == newer_values
    assert not clean._dirty and not clean._stale
    assert dirty.base_revision == 4
    assert dirty._form.read_all() == draft
    assert dirty._dirty and dirty._stale
    assert dirty._status.severity == "warning"
    with pytest.raises(ValueError, match="conflicting semantic state"):
        dirty.load(
            revision=5,
            semantic_identity=("curve", 5, "conflict"),
            values=newer_values,
        )
    with pytest.raises(ValueError, match="cannot move backwards"):
        dirty.load(revision=4, semantic_identity=("curve", 4), values=base_values)


def test_apply_accept_noop_and_one_step_preserve_two_surface_cas() -> None:
    Editor, image_spec, _curve_spec = _types()
    first = Editor(image_spec, "image display")
    second = Editor(image_spec, "image display")
    base = {"mode": "tight", "color_min": None, "color_max": None}
    committed = {"mode": "fixed", "color_min": 2.0, "color_max": 9.0}
    for editor in (first, second):
        editor.load(revision=7, semantic_identity=("image", 7), values=base)

    emitted: list[tuple[int, object]] = []
    first.applyRequested.connect(
        lambda revision, values: emitted.append((revision, values))
    )
    first._form.widget_for("color_min").setText("2**3")
    first._apply_button.click()
    assert emitted == []
    assert first._status.severity == "error"

    first._form.widget_for("color_min").setText("2")
    first._form.widget_for("color_max").setText("9")
    mode = first._form.widget_for("mode")
    mode.setCurrentIndex(mode.findData("fixed"))
    first._apply_button.click()
    assert emitted == [(7, committed)]

    second._form.widget_for("color_min").setText("4")
    first.accept_commit(
        base_revision=7,
        revision=8,
        semantic_identity=("image", 8),
        values=committed,
    )
    second.load(revision=8, semantic_identity=("image", 8), values=committed)
    assert first.base_revision == 8 and not first._dirty and not first._stale
    assert second.base_revision == 7 and second._dirty and second._stale

    first._form.widget_for("color_min").setText("5")
    first.accept_commit(
        base_revision=8,
        revision=8,
        semantic_identity=("image", 8),
        values=committed,
    )
    assert first._form.read_all() == committed
    assert not first._dirty
    with pytest.raises(ValueError, match="no-op conflicts"):
        first.accept_commit(
            base_revision=8,
            revision=8,
            semantic_identity=("image", 8, "different"),
            values=committed,
        )
    with pytest.raises(ValueError, match="advance once"):
        first.accept_commit(
            base_revision=8,
            revision=10,
            semantic_identity=("image", 10),
            values=committed,
        )


def test_cancel_reload_gate_and_runtime_placeholder_never_overwrite_draft() -> None:
    Editor, image_spec, _curve_spec = _types()
    editor = Editor(
        image_spec,
        "image display",
        runtime_placeholder_fields=("color_min", "color_max"),
    )
    base = {"mode": "tight", "color_min": None, "color_max": None}
    latest = {"mode": "fixed", "color_min": 3.0, "color_max": 8.0}
    editor.load(
        revision=1,
        semantic_identity=("image", 1),
        values=base,
        runtime_placeholders={"color_min": "1.0", "color_max": "9.0"},
    )
    low = editor._form.widget_for("color_min")
    low.setText("4.5")
    editor.load(
        revision=2,
        semantic_identity=("image", 2),
        values=latest,
        runtime_placeholders={"color_min": "2.0", "color_max": "10.0"},
    )
    assert low.text() == "4.5"
    low.setText("")
    assert low.placeholderText() == "2.0"

    editor._form.widget_for("color_min").setText("4.5")
    assert editor._stale

    requests: list[None] = []

    def reload() -> None:
        requests.append(None)
        editor.load(revision=2, semantic_identity=("image", 2), values=latest)

    editor.cancelRequested.connect(reload)
    editor._cancel_button.click()
    assert requests == [None]
    assert editor.base_revision == 2
    assert editor._form.read_all() == latest
    assert not editor._dirty and not editor._stale


def test_explicit_owner_replacement_discards_old_revision_and_dirty_units() -> None:
    Editor, _image_spec, curve_spec = _types()
    editor = Editor(
        curve_spec,
        "curve display",
        runtime_placeholder_fields=("x_min", "x_max"),
    )
    old_values = {
        "mode": "fixed",
        "x_min": 10.0,
        "x_max": 20.0,
        "enabled": True,
    }
    reset_values = {
        "mode": "normal",
        "x_min": None,
        "x_max": None,
        "enabled": True,
    }
    editor.load(
        revision=9,
        semantic_identity=("MHz scan", 9),
        values=old_values,
    )
    editor._form.widget_for("x_min").setText("12")
    assert editor._dirty

    editor.replace_owner_state(
        revision=0,
        semantic_identity=("ms scan", 0),
        values=reset_values,
        runtime_placeholders={"x_min": "0", "x_max": "5"},
    )
    assert editor.base_revision == 0
    assert editor._form.read_all() == reset_values
    assert not editor._dirty and not editor._stale
    assert editor._form.widget_for("x_min").placeholderText() == "0"


def test_runtime_placeholder_contract_is_closed_and_line_edit_only() -> None:
    Editor, _image_spec, curve_spec = _types()
    with pytest.raises(TypeError, match="must project to QLineEdit"):
        Editor(
            curve_spec,
            "curve display",
            runtime_placeholder_fields=("enabled",),
        )
    editor = Editor(
        curve_spec,
        "curve display",
        runtime_placeholder_fields=("x_min",),
    )
    values = {"mode": "normal", "x_min": None, "x_max": None, "enabled": True}
    with pytest.raises(ValueError, match="not admitted"):
        editor.load(
            revision=0,
            semantic_identity=("curve", 0),
            values=values,
            runtime_placeholders={"x_max": "3.0"},
        )
    with pytest.raises(TypeError, match="must be str"):
        editor.load(
            revision=0,
            semantic_identity=("curve", 0),
            values=values,
            runtime_placeholders={"x_min": 3.0},  # type: ignore[dict-item]
        )
