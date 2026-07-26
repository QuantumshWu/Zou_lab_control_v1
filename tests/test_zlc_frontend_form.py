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
                "iterations",
                "int",
                "Iterations",
                default=40,
                minimum=1,
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
            FormFieldProps(
                "site",
                "int",
                "Site",
                default=None,
                minimum=0,
                allow_blank=True,
            ),
            FormFieldProps(
                "exposure",
                "float",
                "Exposure",
                default=None,
                minimum=float.fromhex("0x0.0000000000001p-1022"),
                allow_blank=True,
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
        "iterations",
        "count",
        "timeout",
        "gain",
        "site",
        "exposure",
        "authored",
        "mode",
        "enabled",
    )
    assert spec.fields[1].row_label == "Iterations"
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
        "axis_range",
        "path",
        "signal",
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


def test_form_keeps_spins_numeric_and_projects_optional_numbers_as_typed_blanks() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtGui

    from zlc_frontend.qt_widgets import (
        FluentDoubleSpinBox,
        FluentLineEdit,
        FluentParameterForm,
        FluentSpinBox,
        ensure_qt_app,
    )

    ensure_qt_app()
    form = FluentParameterForm(_spec())
    assert isinstance(form.widget_for("iterations"), FluentSpinBox)
    assert isinstance(form.widget_for("count"), FluentSpinBox)
    assert isinstance(form.widget_for("timeout"), FluentDoubleSpinBox)
    assert isinstance(form.widget_for("gain"), FluentDoubleSpinBox)
    site = form.widget_for("site")
    exposure = form.widget_for("exposure")
    assert isinstance(site, FluentLineEdit)
    assert isinstance(exposure, FluentLineEdit)
    assert form.widget_for("iterations").specialValueText() == ""
    assert form.widget_for("timeout").specialValueText() == ""
    for optional in (site, exposure):
        validator = optional.validator()
        assert validator is not None
        assert validator.validate("", 0)[0] == QtGui.QValidator.Intermediate
        assert validator.validate("Auto", 4)[0] == QtGui.QValidator.Invalid

    values = form.read_all()
    assert values["iterations"] == 40
    assert values["gain"] == 0.12345678901234566
    assert values["site"] is None
    assert values["exposure"] is None
    assert type(values["authored"]) is int
    assert type(values["mode"]) is int

    values["iterations"] = 100_000
    values["timeout"] = 1.2345678901234567e100
    values["site"] = 7
    values["exposure"] = 0.013
    values["authored"] = 1.0
    values["mode"] = "1"
    form.write_all(values)
    written = form.read_all()
    assert written["iterations"] == 100_000
    assert written["timeout"] == 1.2345678901234567e100
    assert written["site"] == 7
    assert written["exposure"] == 0.013
    assert type(written["authored"]) is float
    assert type(written["mode"]) is str

    values["site"] = None
    values["exposure"] = None
    form.write_all(values)
    assert form.read_all()["site"] is None
    assert form.read_all()["exposure"] is None


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
    updated.update(text="new", iterations=2**20, enabled=False)
    form.widget_for("iterations").setDisabled(True)
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

    form.widget_for("iterations").lineEdit().setText("2**63")
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


