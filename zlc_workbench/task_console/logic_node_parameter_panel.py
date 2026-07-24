"""One shared editor for TaskConsole measurement, processor, and task forms."""

from __future__ import annotations

from typing import Mapping, Sequence

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.form import FormSpec
from zlc_frontend.qt_widgets import (
    ElidedLabel,
    FluentButton,
    FluentComboBox,
    FluentLabel,
    GREEN,
    GREY,
    ORANGE,
    RED,
    scaled_px,
)
from zlc_frontend.qt_widgets import FluentParameterForm, FormRuntimeContext

__all__ = ["LogicNodeParameterPanel"]


class LogicNodeParameterPanel(QtWidgets.QWidget):
    """Render a catalog entry's one :class:`zlc_frontend.form.FormSpec`.

    Catalog entries own declarations and request construction.  This widget
    owns only stable Qt controls, validation feedback, and Start/Stop gestures.
    """

    start_requested = QtCore.pyqtSignal(object)
    stop_requested = QtCore.pyqtSignal()
    draft_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        measurements: Sequence[object],
        parent=None,
        *,
        single: bool = False,
        controls: bool = True,
        signals_provider=None,
        signal_providers=None,
        sources_provider=None,
        formats_provider=None,
        short_names_provider=None,
        runtime: FormRuntimeContext | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._specs = list(measurements)
        self._single = bool(single)
        self._controls = bool(controls)
        self._signals_provider = signals_provider
        self._signal_providers = dict(signal_providers or {})
        self._sources_provider = sources_provider
        self._formats_provider = formats_provider
        self._short_names_provider = short_names_provider
        if runtime is not None and not isinstance(runtime, FormRuntimeContext):
            raise TypeError("runtime must be FormRuntimeContext or None")
        self._explicit_runtime = runtime
        self._running = False
        self._parameter_form: QtWidgets.QWidget | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(scaled_px(6, minimum=4))

        picker = QtWidgets.QHBoxLayout()
        picker.setSpacing(scaled_px(6, minimum=4))
        self._pick_label = FluentLabel("measurement")
        self._pick_label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        self.type_combo = FluentComboBox()
        self.type_combo.setMinimumWidth(scaled_px(220, minimum=160))
        for index, spec in enumerate(self._specs):
            self.type_combo.addItem(str(spec.name), index)
        self.type_combo.activated.connect(lambda *_: self._rebuild_form())
        picker.addWidget(self._pick_label)
        picker.addWidget(self.type_combo, 1)
        root.addLayout(picker)
        if self._single:
            self._pick_label.hide()
            self.type_combo.hide()

        self._form_host = QtWidgets.QVBoxLayout()
        self._form_host.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self._form_host)

        actions = QtWidgets.QHBoxLayout()
        self.start_button = FluentButton("Start", color=GREEN)
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit())
        self.stop_button.setEnabled(False)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        root.addLayout(actions)
        if not self._controls:
            self.start_button.hide()
            self.stop_button.hide()

        self.status = ElidedLabel("")
        self.status.setMinimumWidth(0)
        self.status.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        root.addWidget(self.status)

        if self._specs:
            self._rebuild_form()

    def current_spec(self):
        index = self.type_combo.currentData()
        if index is None or not self._specs:
            return None
        return self._specs[int(index)]

    def _runtime(self) -> FormRuntimeContext:
        if self._explicit_runtime is not None:
            return self._explicit_runtime
        def names(key: str):
            provider = self._signal_providers.get(key, self._signals_provider)
            return provider() if callable(provider) else ()

        return FormRuntimeContext(
            signal_names=names,
            signal_sources=self._sources_provider,
            signal_formats=self._formats_provider,
            signal_labels=self._short_names_provider,
        )

    def _clear_form(self) -> None:
        while self._form_host.count():
            item = self._form_host.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # The replacement is performed on a potentially visible Edit
                # page.  Keep the old form parented until DeferredDelete so a
                # dynamic catalog refresh cannot flash it as a top-level.
                widget.hide()
                widget.deleteLater()
        self._parameter_form = None

    def _rebuild_form(self) -> None:
        self._clear_form()
        spec = self.current_spec()
        if spec is None:
            return
        editor_factory = spec.editor_factory
        if editor_factory is None:
            form = FluentParameterForm(
                FormSpec(tuple(spec.editor_fields)),
                parent=self,
                runtime=self._runtime(),
            )
        else:
            form = editor_factory(runtime=self._runtime(), parent=self)
            if not isinstance(form, QtWidgets.QWidget):
                raise TypeError("console editor_factory must return QWidget")
            for name in ("changed", "read_all", "refresh"):
                if not hasattr(form, name):
                    raise TypeError(
                        f"console editor_factory result lacks {name}"
                    )
        form.changed.connect(self._on_form_changed)
        self._parameter_form = form
        self._form_host.addWidget(form)
        self._refresh_start_enabled()

    def _on_form_changed(self, *_args) -> None:
        """Report an edited draft without materialising a config snapshot.

        The containing Workbench decides when that draft crosses an authored
        commit boundary (Apply, Start, Save, or tab close).  Keeping this signal
        value-free is deliberate: ordinary typing must never rebuild a Logic
        node or manufacture a cross-thread snapshot.
        """

        self._refresh_start_enabled()
        self.draft_changed.emit()

    def collect_values(self) -> dict[str, object]:
        if self._parameter_form is None:
            return {}
        return self._parameter_form.read_all()

    def refresh_on_show(self) -> None:
        if self._parameter_form is not None:
            self._parameter_form.refresh()
        self._refresh_start_enabled()

    def set_axis_range(self, lo: float, hi: float) -> bool:
        form = self._parameter_form
        if form is None:
            return False
        for field in form.spec.fields:
            if field.kind != "axis_range":
                continue
            widget = form.widget_for(field.key)
            widget.min_spin.setValue(float(min(lo, hi)))
            widget.max_spin.setValue(float(max(lo, hi)))
            return True
        return False

    def _missing_required(self) -> list[str]:
        form = self._parameter_form
        if form is None:
            return []
        specialized = getattr(form, "missing_required_labels", None)
        if callable(specialized):
            return list(specialized())
        return [
            field.label
            for field in form.spec.fields
            if (
                field.required
                and form.is_empty(field.key)
                and not field.required_choice_unavailable
            )
        ]

    def _unavailable_reasons(self) -> tuple[str, ...]:
        form = self._parameter_form
        if form is None:
            return ()
        specialized = getattr(form, "unavailable_reasons", None)
        if callable(specialized):
            return tuple(str(reason) for reason in specialized() if str(reason))
        return tuple(
            field.unavailable_reason
            for field in form.spec.fields
            if field.required_choice_unavailable
        )

    def _refresh_start_enabled(self, *_args) -> None:
        if self._running:
            return
        missing = self._missing_required()
        unavailable = self._unavailable_reasons()
        self.start_button.setEnabled(not missing and not unavailable)
        if missing:
            self.set_status("set required: " + ", ".join(missing), error=True)
        elif unavailable:
            self.set_status("unavailable: " + "; ".join(unavailable), error=True)
        elif not self.status.text().startswith(("running", "done", "fit", "failed", "T")):
            self.set_status("ready", error=False)

    def _on_start(self) -> None:
        if (
            self.current_spec() is not None
            and not self._missing_required()
            and not self._unavailable_reasons()
        ):
            self.start_requested.emit(self)

    def seed_values(self, values: Mapping[str, object]) -> None:
        form = self._parameter_form
        if form is None:
            return
        if not isinstance(values, Mapping):
            raise TypeError("logic-node values must be a mapping")
        unknown = set(values) - set(form.spec.keys)
        if unknown:
            raise ValueError(
                "logic-node values contain fields absent from the selected "
                f"definition: {tuple(sorted(map(str, unknown)))}"
            )
        merged = form.spec.default_values()
        for key in form.spec.keys:
            if key in values:
                merged[key] = values[key]
        # FluentParameterForm.populate() validates the complete value before it
        # touches any QWidget.  A saved physical request is therefore restored
        # exactly or rejected as a whole; no field may silently fall back.
        form.populate(merged)
        self._refresh_start_enabled()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.start_button.setEnabled(
            not running
            and not self._missing_required()
            and not self._unavailable_reasons()
        )
        self.stop_button.setEnabled(bool(running))
        self.type_combo.setEnabled(not running)
        if self._parameter_form is not None:
            self._parameter_form.setEnabled(not running)

    def set_status(self, text: str, *, error: bool) -> None:
        self.status.setText(str(text))
        self.status.setStyleSheet(
            f"color: {RED if error else GREY}; background: transparent; border: none;"
        )
