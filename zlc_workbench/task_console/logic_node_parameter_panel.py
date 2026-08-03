"""Descriptor-driven parameter form shared by every TaskConsole Logic node."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.form import (
    FormChoice,
    FormFieldProps,
    FormSpec,
    project_authoring_form,
)
from zlc_frontend.qt_widgets import (
    ElidedLabel,
    FluentButton,
    FluentParameterForm,
    FormRuntimeContext,
    GREEN,
    GREY,
    ORANGE,
    RED,
    scaled_px,
)
from zlc_neutral_atom.input_spec import ArtifactInputSpec, DatasetInputSpec
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.logic_node import LogicNodeDescriptor
from zlc_storage import resolve_under

__all__ = ["LogicNodeParameterPanel"]


def _device_choice_projection(
    descriptor: LogicNodeDescriptor,
    catalog: DeviceCatalogView,
) -> dict[str, object]:
    """Project grouped capabilities to stable instance-id choices once."""

    requirements = dict(descriptor.device_requirements)
    dynamic_keys = tuple(
        field.key
        for field in descriptor.authoring_schema.fields
        if field.dynamic_choices
    )
    undeclared = set(dynamic_keys).difference(requirements)
    if undeclared:
        raise ValueError(
            "dynamic authoring fields need descriptor device requirements; "
            f"missing={tuple(sorted(undeclared))}"
        )
    projected: dict[str, object] = {}
    for key in dynamic_keys:
        capabilities = requirements[key]
        devices = tuple(
            info
            for info in catalog.values()
            if all(value in info.capabilities for value in capabilities)
        )
        choices = tuple(
            FormChoice(
                f"{info.role} · {info.type_id}  [{info.instance_id}]",
                info.instance_id,
            )
            for info in devices
        )
        projected[key] = SimpleNamespace(
            field_key=key,
            choices=choices,
            default=devices[0].instance_id if devices else None,
            unavailable_reason=(
                ""
                if devices
                else "no installed device provides " + ", ".join(capabilities)
            ),
        )
    return projected


def _input_fields(descriptor: LogicNodeDescriptor) -> tuple[FormFieldProps, ...]:
    """Mechanically expose the two owner-declared input kinds."""

    fields: list[FormFieldProps] = []
    for spec in descriptor.input_specs:
        if isinstance(spec, DatasetInputSpec):
            fields.append(
                FormFieldProps(
                    spec.key,
                    "signal",
                    spec.label,
                    required=True,
                    description=spec.description,
                )
            )
            continue
        if not isinstance(spec, ArtifactInputSpec):
            raise TypeError("descriptor contains another input kind")
        if not spec.allow_saved_reference:
            fields.append(
                FormFieldProps(
                    spec.producer_key,
                    "signal",
                    spec.label,
                    required=True,
                    description=spec.description,
                )
            )
            continue
        fields.extend(
            (
                FormFieldProps(
                    spec.source_key,
                    "choice",
                    f"{spec.label} source",
                    default="current",
                    required=True,
                    choices=(
                        FormChoice("Current artifact", "current"),
                        FormChoice("Task output", "producer"),
                        FormChoice("Saved record", "saved"),
                    ),
                    description=spec.description,
                ),
                FormFieldProps(
                    spec.producer_key,
                    "signal",
                    f"{spec.label} Task output",
                    description="Used only when source is Task output.",
                ),
                FormFieldProps(
                    spec.reference_path_key,
                    "path",
                    f"Saved {spec.label.lower()}",
                    default=spec.default_reference_path,
                    base_dir="",
                    description=(
                        "Project-relative record path; used only when source is Saved record."
                    ),
                ),
            )
        )
    return tuple(fields)


def _load_custom_form(
    descriptor: LogicNodeDescriptor,
    base_form: FormSpec,
    *,
    pulses_root: Path,
):
    structured = tuple(
        field for field in descriptor.authoring_schema.fields
        if field.kind == "structured"
    )
    contribution = next(
        (
            value
            for value in descriptor.ui_contributions
            if value.purpose == "task_console_editor"
        ),
        None,
    )
    if structured and contribution is None:
        raise RuntimeError(
            f"{descriptor.api_name!r} declares structured authoring without "
            "one task_console_editor contribution"
        )
    if contribution is None:
        return base_form, None
    symbol = getattr(import_module(contribution.module), contribution.symbol)
    if not callable(symbol):
        raise TypeError("task_console_editor contribution must be callable")
    built = symbol(base_form, pulses_root=pulses_root)
    if not isinstance(built, tuple) or len(built) != 2:
        raise TypeError("task_console_editor must return (form_spec, editor_factory)")
    spec, factory = built
    if not callable(factory):
        raise TypeError("task_console_editor factory must be callable")
    return spec, factory


class LogicNodeParameterPanel(QtWidgets.QWidget):
    """Render one descriptor and return separate authored values/input refs."""

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    draft_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        descriptor: LogicNodeDescriptor,
        *,
        device_catalog: DeviceCatalogView,
        project_root: Path,
        pulses_root: Path,
        runtime: FormRuntimeContext,
        parent=None,
        controls: bool = True,
    ) -> None:
        if not isinstance(descriptor, LogicNodeDescriptor):
            raise TypeError("descriptor must be LogicNodeDescriptor")
        if not isinstance(device_catalog, DeviceCatalogView):
            raise TypeError("device_catalog must be DeviceCatalogView")
        if not isinstance(runtime, FormRuntimeContext):
            raise TypeError("runtime must be FormRuntimeContext")
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._descriptor = descriptor
        self._project_root = Path(project_root).resolve()
        self._running = False
        inputs = _input_fields(descriptor)
        base = project_authoring_form(
            descriptor.authoring_schema,
            dynamic_choices=_device_choice_projection(descriptor, device_catalog),
        )
        form_spec, editor_factory = _load_custom_form(
            descriptor,
            base,
            pulses_root=Path(pulses_root).resolve(),
        )
        if editor_factory is None:
            form = FluentParameterForm(
                FormSpec((*form_spec.fields, *inputs)),
                parent=self,
                runtime=runtime,
            )
        else:
            form = editor_factory(runtime=runtime, input_fields=inputs, parent=self)
            if not isinstance(form, QtWidgets.QWidget):
                raise TypeError("task_console_editor must build a QWidget")
            for name in ("changed", "read_all", "populate", "refresh", "spec"):
                if not hasattr(form, name):
                    raise TypeError(f"custom Logic form lacks {name}")
        expected = set(descriptor.authoring_schema.keys)
        expected.update(key for spec in descriptor.input_specs for key in spec.field_keys)
        actual = set(form.spec.keys)
        if actual != expected:
            raise RuntimeError(
                "Logic form differs from descriptor-owned fields; "
                f"missing={tuple(sorted(expected - actual))}, "
                f"extra={tuple(sorted(actual - expected))}"
            )
        self._parameter_form = form

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(scaled_px(6, minimum=4))
        root.addWidget(form)
        actions = QtWidgets.QHBoxLayout()
        self.start_button = FluentButton("Start", color=GREEN)
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setEnabled(False)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        root.addLayout(actions)
        if not controls:
            self.start_button.hide()
            self.stop_button.hide()
        self.status = ElidedLabel("")
        root.addWidget(self.status)
        form.changed.connect(self._on_changed)
        self._refresh_start_enabled()

    def _on_changed(self, *_args) -> None:
        self._refresh_start_enabled()
        self.draft_changed.emit()

    def _missing_required(self) -> tuple[str, ...]:
        specialized = getattr(self._parameter_form, "missing_required_labels", None)
        if callable(specialized):
            return tuple(specialized())
        return tuple(
            field.label
            for field in self._parameter_form.spec.fields
            if field.required
            and self._parameter_form.is_empty(field.key)
            and not field.required_choice_unavailable
        )

    def _unavailable_reasons(self) -> tuple[str, ...]:
        specialized = getattr(self._parameter_form, "unavailable_reasons", None)
        if callable(specialized):
            return tuple(str(value) for value in specialized() if str(value))
        return tuple(
            field.unavailable_reason
            for field in self._parameter_form.spec.fields
            if field.required_choice_unavailable
        )

    def _refresh_start_enabled(self) -> None:
        if self._running:
            return
        missing = self._missing_required()
        unavailable = self._unavailable_reasons()
        self.start_button.setEnabled(not missing and not unavailable)
        if missing:
            self.set_status("set required: " + ", ".join(missing), error=True)
        elif unavailable:
            self.set_status("unavailable: " + "; ".join(unavailable), error=True)
        else:
            self.set_status("ready", error=False)

    def _on_start(self) -> None:
        if not self._missing_required() and not self._unavailable_reasons():
            self.start_requested.emit()

    def collect_binding(self) -> tuple[dict[str, object], dict[str, object]]:
        values = self._parameter_form.read_all()
        authored = self._descriptor.authoring_schema.freeze(
            {key: values[key] for key in self._descriptor.authoring_schema.keys}
        )
        inputs: dict[str, object] = {}
        for spec in self._descriptor.input_specs:
            if isinstance(spec, DatasetInputSpec):
                selected = values[spec.key]
                if not isinstance(selected, str) or not selected.strip():
                    raise ValueError(f"select {spec.label}")
                inputs[spec.key] = selected.strip()
                continue
            if not spec.allow_saved_reference:
                selected = values[spec.producer_key]
                if not isinstance(selected, str) or not selected.strip():
                    raise ValueError(f"select {spec.label}")
                inputs[spec.key] = selected.strip()
                continue
            source = values[spec.source_key]
            if source == "current":
                continue
            if source == "producer":
                selected = values[spec.producer_key]
                if not isinstance(selected, str) or not selected.strip():
                    raise ValueError(f"select a {spec.label} Task output")
                inputs[spec.key] = selected.strip()
                continue
            if source != "saved":
                raise ValueError(f"unknown {spec.label} source {source!r}")
            raw = values[spec.reference_path_key]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"select a saved {spec.label.lower()}")
            path = Path(raw).expanduser()
            relative = (
                path.resolve().relative_to(self._project_root)
                if path.is_absolute()
                else path
            )
            inputs[spec.key] = relative.as_posix()
            resolve_under(self._project_root, relative)
        return authored, inputs

    def seed_binding(
        self,
        authored: Mapping[str, object],
        inputs: Mapping[str, object],
    ) -> None:
        if not isinstance(authored, Mapping) or not isinstance(inputs, Mapping):
            raise TypeError("authored values and input refs must be mappings")
        values = self._parameter_form.spec.default_values()
        values.update(authored)
        for spec in self._descriptor.input_specs:
            selected = inputs.get(spec.key)
            if isinstance(spec, DatasetInputSpec):
                if selected is not None:
                    values[spec.key] = selected
                continue
            if not spec.allow_saved_reference:
                if selected is not None:
                    values[spec.producer_key] = selected
                continue
            if selected is None:
                values[spec.source_key] = "current"
            elif isinstance(selected, str) and selected.startswith("@logic/"):
                values[spec.source_key] = "producer"
                values[spec.producer_key] = selected
            else:
                values[spec.source_key] = "saved"
                values[spec.reference_path_key] = str(selected)
        self._parameter_form.populate(values)
        self._refresh_start_enabled()

    def refresh_on_show(self) -> None:
        self._parameter_form.refresh()
        self._refresh_start_enabled()

    def set_axis_range(self, lo: float, hi: float) -> bool:
        for field in self._parameter_form.spec.fields:
            if field.kind != "axis_range":
                continue
            widget = self._parameter_form.widget_for(field.key)
            widget.min_spin.setValue(float(min(lo, hi)))
            widget.max_spin.setValue(float(max(lo, hi)))
            return True
        return False

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.start_button.setEnabled(
            not running and not self._missing_required() and not self._unavailable_reasons()
        )
        self.stop_button.setEnabled(bool(running))
        self._parameter_form.setEnabled(not running)

    def set_mutation_enabled(self, enabled: bool) -> None:
        """Enable/disable authored edits without changing lifecycle state."""

        allowed = bool(enabled) and not self._running
        self._parameter_form.setEnabled(allowed)
        if not enabled:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        else:
            self.set_running(self._running)

    def set_status(self, text: str, *, error: bool) -> None:
        self.status.setText(str(text))
        self.status.setStyleSheet(
            f"color: {RED if error else GREY}; background: transparent; border: none;"
        )
