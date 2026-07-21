from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _spec():
    from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec

    return FormSpec(
        (
            FormFieldProps("text", "text", "Text", default=""),
            FormFieldProps(
                "budget",
                "int",
                "Memory budget",
                default=2**40,
                minimum=1,
                unit="bytes",
            ),
            FormFieldProps(
                "count", "int", "Count", default=3, minimum=1, maximum=9
            ),
            FormFieldProps(
                "timeout", "float", "Timeout", default=1.25, minimum=0.0
            ),
            FormFieldProps(
                "gain",
                "float",
                "Gain",
                default=0.12345678901234566,
                minimum=-1.0,
                maximum=1.0,
            ),
            FormFieldProps("authored", "number", "Authored number", default=1),
            FormFieldProps(
                "mode",
                "choice",
                "Mode",
                default=1,
                choices=(FormChoice("integer", 1), FormChoice("text", "1")),
            ),
            FormFieldProps("enabled", "bool", "Enabled", default=True),
        )
    )


def test_form_owners_have_no_domain_or_legacy_dependency() -> None:
    forbidden = {"Zou_lab_control", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"}
    for relative in ("zlc_frontend/form.py", "zlc_frontend/qt_widgets/form.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
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
        assert roots.isdisjoint(forbidden), relative


def test_headless_form_contract_is_immutable_exact_and_does_not_load_qt() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import zlc_frontend\n"
                "from zlc_frontend import FormChoice, FormFieldProps, FormSpec\n"
                "assert 'PyQt5' not in sys.modules\n"
                "spec = FormSpec((FormFieldProps('key', 'choice', 'Key', "
                "default=1, choices=(FormChoice('one', 1),)),))\n"
                "assert spec.keys == ('key',)\n"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    spec = _spec()
    assert spec.keys == (
        "text",
        "budget",
        "count",
        "timeout",
        "gain",
        "authored",
        "mode",
        "enabled",
    )
    assert spec.fields[1].row_label == "Memory budget (bytes)"
    with pytest.raises(FrozenInstanceError):
        spec.fields[0].label = "changed"


def test_choice_values_are_typed_and_invalid_specs_fail_closed() -> None:
    from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec

    field = FormFieldProps(
        "mode",
        "choice",
        "Mode",
        default=1,
        choices=(FormChoice("integer", 1), FormChoice("text", "1")),
    )
    assert field.choice_for(1).label == "integer"
    assert field.choice_for("1").label == "text"
    assert field.choice_for(True) is None

    with pytest.raises(ValueError, match="unique"):
        FormSpec((field, field))
    with pytest.raises(ValueError, match="typed choice"):
        FormFieldProps(
            "bad",
            "choice",
            "Bad",
            default=True,
            choices=(FormChoice("integer", 1),),
        )
    with pytest.raises(TypeError, match="immutable scalar"):
        FormChoice("mutable", [])


def test_shared_number_leaf_preserves_authored_type_and_rejects_expressions() -> None:
    from zlc_frontend import parse_number_text

    assert type(parse_number_text("1", "cell")) is int
    assert type(parse_number_text("1.0", "cell")) is float
    assert parse_number_text(" -2.5e3 ", "cell") == -2500.0
    with pytest.raises(ValueError, match="finite decimal"):
        parse_number_text("2**8", "cell")
    with pytest.raises(ValueError, match="finite"):
        parse_number_text("1e9999", "cell")


def test_closed_registry_has_one_complete_atomic_handler_per_kind() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import FORM_WIDGET_HANDLERS, FormWidgetHandler

    assert set(FORM_WIDGET_HANDLERS) == {
        "text",
        "int",
        "float",
        "number",
        "choice",
        "bool",
    }
    assert FormWidgetHandler.__abstractmethods__ == {
        "normalize",
        "build",
        "read",
        "write",
        "is_empty",
        "refresh",
    }
    for handler in FORM_WIDGET_HANDLERS.values():
        assert isinstance(handler, FormWidgetHandler)
        assert all(
            callable(getattr(handler, operation))
            for operation in FormWidgetHandler.__abstractmethods__
        )
    with pytest.raises(TypeError):
        FORM_WIDGET_HANDLERS["new"] = object()


def test_form_preserves_unbounded_numbers_lossless_float_and_typed_choices() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import (
        FluentDoubleSpinBox,
        FluentLineEdit,
        FluentParameterForm,
        FluentSpinBox,
        ensure_qt_app,
    )

    ensure_qt_app()
    form = FluentParameterForm(_spec())
    assert isinstance(form.widget_for("budget"), FluentLineEdit)
    assert isinstance(form.widget_for("count"), FluentSpinBox)
    assert isinstance(form.widget_for("timeout"), FluentLineEdit)
    assert isinstance(form.widget_for("gain"), FluentDoubleSpinBox)

    values = form.read_all()
    assert values["budget"] == 2**40
    assert values["gain"] == 0.12345678901234566
    assert type(values["authored"]) is int
    assert type(values["mode"]) is int

    huge = 2**80 + 123
    values["budget"] = huge
    values["timeout"] = 1.2345678901234567e100
    values["authored"] = 1.0
    values["mode"] = "1"
    form.write_all(values)
    written = form.read_all()
    assert written["budget"] == huge
    assert written["timeout"] == 1.2345678901234567e100
    assert type(written["authored"]) is float
    assert type(written["mode"]) is str


def test_full_state_populate_is_exact_prevalidated_and_signal_blocked() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import FluentParameterForm, ensure_qt_app

    ensure_qt_app()
    form = FluentParameterForm(_spec())
    changes: list[str] = []
    form.changed.connect(changes.append)
    initial = form.read_all()

    with pytest.raises(ValueError, match="exact keys"):
        form.populate({key: value for key, value in initial.items() if key != "gain"})
    assert form.read_all() == initial

    invalid = dict(initial)
    invalid["text"] = "would be a partial write"
    invalid["count"] = 10
    with pytest.raises(ValueError, match="above"):
        form.populate(invalid)
    assert form.read_all() == initial

    updated = dict(initial)
    updated.update(text="new", budget=2**70, enabled=False)
    form.widget_for("budget").setDisabled(True)
    form.widget_for("enabled").hide()
    form.populate(updated)
    assert form.read_all() == updated
    assert changes == []

    form.widget_for("text").setText("user edit")
    assert changes == ["text"]


def test_text_is_never_evaluated_and_invalid_numeric_text_fails_closed() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import FluentParameterForm, ensure_qt_app

    ensure_qt_app()
    form = FluentParameterForm(_spec())
    literal = "__import__('pathlib').Path('should_not_exist').touch()"
    form.widget_for("text").setText(literal)
    assert form.read_all()["text"] == literal

    form.widget_for("budget").setText("2**63")
    with pytest.raises(ValueError, match="base-10 integer"):
        form.read_all()


def test_refresh_preserves_typed_choice_and_emits_no_change() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import FluentParameterForm, ensure_qt_app

    ensure_qt_app()
    form = FluentParameterForm(_spec())
    values = form.read_all()
    values["mode"] = "1"
    form.populate(values)
    changes: list[str] = []
    form.changed.connect(changes.append)

    form.refresh()

    current = form.read_all()["mode"]
    assert type(current) is str
    assert current == "1"
    assert changes == []


def test_required_empty_field_is_reported_by_key() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.form import FormFieldProps, FormSpec
    from zlc_frontend.qt_widgets import FluentParameterForm, ensure_qt_app

    ensure_qt_app()
    form = FluentParameterForm(
        FormSpec((FormFieldProps("name", "text", "Name", required=True),))
    )
    assert form.is_empty("name")
    with pytest.raises(ValueError, match="'name'.*required"):
        form.read_all()
    with pytest.raises(KeyError, match="unknown form field"):
        form.widget_for("missing")


