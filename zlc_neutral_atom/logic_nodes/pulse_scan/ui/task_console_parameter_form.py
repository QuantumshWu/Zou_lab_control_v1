"""Scan Workbench composite editor for one PulseScan request."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.form import FormFieldProps, FormSpec
from zlc_frontend.qt_widgets import (
    FluentParameterForm,
    FormRuntimeContext,
    scaled_px,
    signals_blocked,
)
from zlc_pulse import (
    PulseTemplateDescription,
    describe_pulse_template,
)
from zlc_storage.paths import resolve_under

from .pulse_slots_widget import PulseSlotsWidget

__all__ = ["task_console_editor"]


@dataclass(frozen=True, slots=True)
class PulseScanFormSpec:
    """Ordinary pulse fields plus the one structured slot/program value."""

    program_fields: tuple[FormFieldProps, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.program_fields)
        if tuple(field.key for field in fields) != (
            "pulse",
            "scan_sweep_count",
        ):
            raise ValueError(
                "PulseScan program fields must be pulse and scan_sweep_count"
            )
        if fields[0].kind != "path" or fields[1].kind != "int":
            raise ValueError("PulseScan program fields have unexpected kinds")
        object.__setattr__(self, "program_fields", fields)

    @property
    def fields(self) -> tuple[FormFieldProps, ...]:
        return self.program_fields

    @property
    def keys(self) -> tuple[str, str, str]:
        return "pulse", "scan_sweep_count", "pulse_slots"

    def default_values(self) -> dict[str, object]:
        return {
            **{field.key: field.default for field in self.program_fields},
            "pulse_slots": {},
        }


@dataclass(frozen=True, slots=True)
class _PulseScanEditorSpec:
    program: PulseScanFormSpec
    input_fields: tuple[FormFieldProps, ...]

    @property
    def fields(self) -> tuple[FormFieldProps, ...]:
        return self.program.fields + self.input_fields

    @property
    def keys(self) -> tuple[str, ...]:
        return self.program.keys + tuple(field.key for field in self.input_fields)

    def default_values(self) -> dict[str, object]:
        return {
            **self.program.default_values(),
            **{field.key: field.default for field in self.input_fields},
        }


class PulseScanParameterForm(QtWidgets.QWidget):
    """Compose generic leaves around the explicit slot/program presenter.

    Loading a pulse document happens only when the path is committed (browse or
    ``editingFinished``), never for every keystroke.  Until that commit the old
    rows remain visible but the form is invalid, so no stale template can start
    a run and no QWidget tree churn occurs while typing.
    """

    changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        spec: PulseScanFormSpec,
        *,
        input_fields: tuple[FormFieldProps, ...],
        runtime: FormRuntimeContext,
        pulse_template_reader=None,
        parent=None,
    ) -> None:
        if not isinstance(spec, PulseScanFormSpec):
            raise TypeError("spec must be PulseScanFormSpec")
        if not isinstance(runtime, FormRuntimeContext):
            raise TypeError("runtime must be FormRuntimeContext")
        inputs = tuple(input_fields)
        if not inputs or any(not isinstance(field, FormFieldProps) for field in inputs):
            raise TypeError("PulseScan requires projected input FormFieldProps")
        if pulse_template_reader is not None and not callable(pulse_template_reader):
            raise TypeError("pulse_template_reader must be callable or None")
        super().__init__(parent)
        self._spec = spec
        self._editor_spec = _PulseScanEditorSpec(spec, inputs)
        self._reader = pulse_template_reader
        self._loaded_path: str | None = None
        self._path_dirty = False
        self._template_error = ""

        self._program_form = FluentParameterForm(
            FormSpec(spec.program_fields),
            runtime=runtime,
            parent=self,
        )
        self._slots = PulseSlotsWidget(self)
        self._input_form = FluentParameterForm(
            FormSpec(inputs),
            runtime=runtime,
            parent=self,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(scaled_px(6, minimum=4))
        layout.addWidget(self._program_form)
        layout.addWidget(self._slots)
        layout.addWidget(self._input_form)

        self._program_form.changed.connect(self._on_program_field_edited)
        self._slots.changed.connect(lambda: self.changed.emit("pulse_slots"))
        self._input_form.changed.connect(lambda key: self.changed.emit(str(key)))

        path_widget = self._program_form.widget_for("pulse")
        path_widget.selected.connect(lambda _path: self._commit_template())
        path_widget.edit.editingFinished.connect(self._commit_template)
        self._commit_template(force=True)

    @property
    def spec(self) -> _PulseScanEditorSpec:
        return self._editor_spec

    def widget_for(self, key: str) -> QtWidgets.QWidget:
        if key == "pulse":
            return self._program_form.widget_for(key)
        if key == "scan_sweep_count":
            return self._program_form.widget_for(key)
        if key == "pulse_slots":
            return self._slots
        if key in self._input_form.spec.keys:
            return self._input_form.widget_for(key)
        raise KeyError(f"unknown PulseScan field key: {key!r}")

    def is_empty(self, key: str) -> bool:
        if key in self._program_form.spec.keys:
            return self._program_form.is_empty(key)
        if key in self._input_form.spec.keys:
            return self._input_form.is_empty(key)
        if key == "pulse_slots":
            value = self._slots.values_dict()
            return not (
                value.get("program_id")
                and value.get("sweep_kind")
                and str(value.get("program") or "").strip()
            )
        raise KeyError(f"unknown PulseScan field key: {key!r}")

    def missing_required_labels(self) -> tuple[str, ...]:
        missing = []
        missing.extend(
            field.label
            for field in self._program_form.spec.fields
            if field.required and self._program_form.is_empty(field.key)
        )
        if self.is_empty("pulse_slots"):
            missing.append("Slots")
        missing.extend(
            field.label
            for field in self._input_form.spec.fields
            if field.required and self._input_form.is_empty(field.key)
        )
        return tuple(missing)

    def unavailable_reasons(self) -> tuple[str, ...]:
        if self._path_dirty:
            return ("finish editing the pulse template path",)
        return (self._template_error,) if self._template_error else ()

    def read_all(self) -> dict[str, object]:
        reasons = self.unavailable_reasons()
        if reasons:
            raise ValueError(reasons[0])
        missing = self.missing_required_labels()
        if missing:
            raise ValueError("set required: " + ", ".join(missing))
        return {
            **self._program_form.read_all(),
            "pulse_slots": self._slots.values_dict(),
            **self._input_form.read_all(),
        }

    def populate(self, values: Mapping[str, object]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("PulseScan form values must be a mapping")
        supplied = set(values)
        expected = set(self._editor_spec.keys)
        if supplied != expected:
            raise ValueError(
                "PulseScan form values must have exact keys; "
                f"missing={sorted(expected - supplied)!r}, "
                f"extra={sorted(supplied - expected)!r}"
            )
        pulse = values["pulse"]
        slots = values["pulse_slots"]
        if pulse is not None and not isinstance(pulse, str):
            raise TypeError("PulseScan pulse must be a path string")
        if not isinstance(slots, Mapping):
            raise TypeError("PulseScan pulse_slots must be a mapping")

        input_values = {
            key: values[key] for key in self._input_form.spec.keys
        }
        program_values = {
            key: values[key] for key in self._program_form.spec.keys
        }
        with signals_blocked(self, self._program_form, self._input_form, self._slots):
            self._program_form.populate(program_values)
            self._input_form.populate(input_values)
            self._slots.seed_value(slots)
            self._path_dirty = False
            self._loaded_path = None
            self._commit_template(force=True)

    def refresh(self) -> None:
        """Refresh dynamic signal choices without rereading a pulse file."""

        self._input_form.refresh()

    def _on_program_field_edited(self, key: str) -> None:
        if key == "pulse":
            self._path_dirty = True
        self.changed.emit(str(key))

    def _commit_template(self, *, force: bool = False) -> None:
        path = str(self._program_form.widget_for("pulse").text()).strip()
        if not force and not self._path_dirty and path == self._loaded_path:
            return
        self._path_dirty = False
        self._loaded_path = path
        if not path:
            self._template_error = ""
            self._slots.reconcile((), (), program_id="")
            self.changed.emit("pulse")
            return
        if self._reader is None:
            self._template_error = "pulse template reader is unavailable"
            self._slots.reconcile((), (), program_id="")
            self.changed.emit("pulse")
            return
        try:
            description = self._reader(path)
            if not isinstance(description, PulseTemplateDescription):
                raise TypeError(
                    "pulse template reader must return PulseTemplateDescription"
                )
        except Exception as error:
            self._template_error = str(error) or type(error).__name__
            self._slots.reconcile((), (), program_id="")
            self.changed.emit("pulse")
            return

        self._template_error = ""
        self._slots.reconcile(
            description.api_rows,
            description.scan_rows,
            api_columns=description.api_columns,
            scan_columns=description.scan_columns,
            hardware_program=description.program,
            program_id=description.program_id,
        )
        self.changed.emit("pulse")


def task_console_editor(base_form: FormSpec, *, pulses_root: str | Path):
    """Augment the declared path field with PulseScan's structured editor."""

    if not isinstance(base_form, FormSpec):
        raise TypeError("base_form must be FormSpec")
    if tuple(base_form.keys) != ("pulse", "scan_sweep_count"):
        raise ValueError(
            "PulseScan base form must contain pulse and scan_sweep_count"
        )
    root = Path(pulses_root).expanduser()
    if not root.is_absolute():
        raise ValueError("pulses_root must be absolute")
    root = root.resolve()
    spec = PulseScanFormSpec(base_form.fields)

    def read_template(value):
        return describe_pulse_template(resolve_under(root, value))

    def editor_factory(*, runtime, input_fields, parent=None):
        return PulseScanParameterForm(
            spec,
            input_fields=input_fields,
            runtime=runtime,
            pulse_template_reader=read_template,
            parent=parent,
        )

    return spec, editor_factory
