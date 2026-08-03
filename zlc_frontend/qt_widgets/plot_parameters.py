"""Fluent editors for the parameters owned by one raster plot host.

This module is deliberately an adapter, not a display-state owner.  Every edit
is submitted directly to :class:`zlc_plot.RasterPlotHost`; immutable worker
descriptions are projected onto stable widgets on the Qt owner thread.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from threading import Lock
from typing import Callable
from weakref import ref

from PyQt5 import QtCore, QtWidgets

from zlc_plot import (
    ControlKind,
    DisplayDescription,
    ParameterControl,
    RasterPlotHost,
    parameter_controls,
)

from .fluent import (
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentLineEdit,
    FluentSettingRow,
    FluentSpinBox,
    FluentSwitch,
    setting_label_width,
    signals_blocked,
)
from .owner_wake import QtOwnerWake


_INTEGER_MINIMUM = -2_147_483_648
_INTEGER_MAXIMUM = 2_147_483_647
_NUMBER_MINIMUM = -1.0e100
_NUMBER_MAXIMUM = 1.0e100


class _OptionalValueEditor(QtWidgets.QWidget):
    """Keep ``None`` outside the value widget itself.

    In particular, a numeric spin box never uses ``specialValueText`` to encode
    ``None``/``AUTO``.  The adjacent switch is the complete optional-value UI.
    """

    valueChanged = QtCore.pyqtSignal(object)

    def __init__(
        self,
        editor: QtWidgets.QWidget,
        current_value: Callable[[], object],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.editor = editor
        self.editor.setParent(self)
        self.automatic = FluentSwitch("Automatic", self)
        self._current_value = current_value
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.automatic, 0)
        self.automatic.toggled.connect(self._automatic_toggled)

    def edited(self) -> None:
        if not self.automatic.isChecked():
            self.valueChanged.emit(self._current_value())

    def _automatic_toggled(self, automatic: bool) -> None:
        self.editor.setEnabled(not automatic)
        self.valueChanged.emit(None if automatic else self._current_value())

    def set_value(
        self,
        value: object,
        setter: Callable[[object], None],
    ) -> None:
        automatic = value is None
        with signals_blocked(self.editor, self.automatic):
            self.automatic.setChecked(automatic)
            self.editor.setEnabled(not automatic)
            if not automatic:
                setter(value)


@dataclass(slots=True)
class _Binding:
    signature: tuple[ControlKind, bool]
    row: FluentSettingRow
    row_control: QtWidgets.QWidget
    editor: QtWidgets.QWidget


class FluentParameterControlSurface(QtWidgets.QWidget):
    """The sole Fluent projection of toolkit-neutral ``ParameterControl`` values."""

    valueEdited = QtCore.pyqtSignal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._bindings: dict[str, _Binding] = {}
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._bindings)

    def editor(self, name: str) -> QtWidgets.QWidget:
        if not isinstance(name, str):
            raise TypeError("parameter name must be a string")
        try:
            return self._bindings[name].editor
        except KeyError as error:
            raise KeyError(f"unknown parameter: {name!r}") from error

    def reconcile_controls(self, controls: tuple[ParameterControl, ...]) -> None:
        names = tuple(control.name for control in controls)
        if len(names) != len(set(names)):
            raise ValueError("parameter controls must have unique names")

        wanted = set(names)
        for name in tuple(self._bindings):
            if name not in wanted:
                binding = self._bindings.pop(name)
                self._layout.removeWidget(binding.row)
                binding.row.hide()
                binding.row.deleteLater()

        label_width = setting_label_width(control.label for control in controls)
        for control in controls:
            signature = (control.kind, control.allow_none)
            binding = self._bindings.get(control.name)
            if binding is None or binding.signature != signature:
                if binding is not None:
                    self._layout.removeWidget(binding.row)
                    binding.row.hide()
                    binding.row.deleteLater()
                binding = self._make_binding(control, label_width)
                self._bindings[control.name] = binding
            else:
                binding.row.set_label(control.label, width=label_width)
            self._sync_binding(binding, control)

        while self._layout.count():
            self._layout.takeAt(0)
        for name in names:
            self._layout.addWidget(self._bindings[name].row)
        self._layout.addStretch(1)
        self._bindings = {name: self._bindings[name] for name in names}

    def _make_binding(
        self,
        control: ParameterControl,
        label_width: int,
    ) -> _Binding:
        editor, row_control = self._make_editor(control)
        editor.setObjectName(f"parameter-{control.name}")
        row = FluentSettingRow(
            control.label,
            row_control,
            label_width=label_width,
            parent=self,
        )
        return _Binding(
            (control.kind, control.allow_none),
            row,
            row_control,
            editor,
        )

    def _emit_value(self, name: str, value: object) -> None:
        self.valueEdited.emit(name, value)

    def _make_editor(
        self,
        control: ParameterControl,
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QWidget]:
        name = control.name
        if control.kind is ControlKind.BOOLEAN:
            editor = FluentSwitch(parent=self)
            if control.allow_none:
                optional = _OptionalValueEditor(editor, editor.isChecked, self)
                editor.toggled.connect(lambda _value: optional.edited())
                optional.valueChanged.connect(
                    lambda value, parameter=name: self._emit_value(parameter, value)
                )
                return editor, optional
            editor.toggled.connect(
                lambda value, parameter=name: self._emit_value(parameter, bool(value))
            )
            return editor, editor

        if control.kind is ControlKind.CHOICE:
            editor = FluentComboBox(self)
            editor.currentIndexChanged.connect(
                lambda index, widget=editor, parameter=name,
                optional=control.allow_none: self._emit_value(
                    parameter,
                    None if optional and index == 0 else widget.currentData(),
                )
            )
            return editor, editor

        if control.kind is ControlKind.INTEGER:
            editor = FluentSpinBox(self)
            if control.allow_none:
                optional = _OptionalValueEditor(editor, editor.value, self)
                editor.valueChanged.connect(lambda _value: optional.edited())
                optional.valueChanged.connect(
                    lambda value, parameter=name: self._emit_value(parameter, value)
                )
                return editor, optional
            editor.valueChanged.connect(
                lambda value, parameter=name: self._emit_value(parameter, int(value))
            )
            return editor, editor

        if control.kind is ControlKind.NUMBER:
            editor = FluentDoubleSpinBox(allow_minus=True, parent=self)
            if control.allow_none:
                optional = _OptionalValueEditor(editor, editor.value, self)
                editor.valueChanged.connect(lambda _value: optional.edited())
                optional.valueChanged.connect(
                    lambda value, parameter=name: self._emit_value(parameter, value)
                )
                return editor, optional
            editor.valueChanged.connect(
                lambda value, parameter=name: self._emit_value(parameter, float(value))
            )
            return editor, editor

        if control.kind is ControlKind.TEXT:
            editor = FluentLineEdit(parent=self)
            if control.allow_none:
                optional = _OptionalValueEditor(editor, editor.text, self)
                editor.editingFinished.connect(optional.edited)
                optional.valueChanged.connect(
                    lambda value, parameter=name: self._emit_value(parameter, value)
                )
                return editor, optional
            editor.editingFinished.connect(
                lambda widget=editor, parameter=name: self._emit_value(
                    parameter, widget.text()
                )
            )
            return editor, editor
        raise TypeError(f"unsupported parameter control: {control.kind!r}")

    def _sync_binding(
        self,
        binding: _Binding,
        control: ParameterControl,
    ) -> None:
        editor = binding.editor
        if control.kind is ControlKind.BOOLEAN:
            self._sync_value(binding, control.value, editor.setChecked)
            return
        if control.kind is ControlKind.CHOICE:
            self._sync_choice(editor, control)
            return
        if control.kind is ControlKind.INTEGER:
            self._configure_integer(editor, control)
            self._sync_value(binding, control.value, editor.setValue)
            return
        if control.kind is ControlKind.NUMBER:
            self._configure_number(editor, control)
            self._sync_value(binding, control.value, editor.setValue)
            return
        if control.kind is ControlKind.TEXT:
            self._sync_value(binding, control.value, editor.setText)
            return
        raise TypeError(f"unsupported parameter control: {control.kind!r}")

    @staticmethod
    def _sync_value(
        binding: _Binding,
        value: object,
        setter: Callable[[object], None],
    ) -> None:
        if isinstance(binding.row_control, _OptionalValueEditor):
            binding.row_control.set_value(value, setter)
            return
        if value is None:
            raise ValueError("non-optional parameter cannot be None")
        with signals_blocked(binding.editor):
            setter(value)

    @staticmethod
    def _sync_choice(
        editor: QtWidgets.QComboBox,
        control: ParameterControl,
    ) -> None:
        entries: list[tuple[str, object]] = []
        if control.allow_none:
            entries.append(("Automatic", None))
        entries.extend((str(choice), choice) for choice in control.choices)
        if control.value is not None and all(
            value != control.value for _label, value in entries
        ):
            entries.append((str(control.value), control.value))
        with signals_blocked(editor):
            editor.clear()
            for label, value in entries:
                editor.addItem(label, value)
            # ``None`` is represented by an invalid QVariant and Qt cannot
            # recover it with findData().  The sole optional entry is always
            # first, and the signal mapper above translates that index back.
            if control.allow_none and control.value is None:
                index = 0
            else:
                index = next(
                    (
                        position
                        for position in range(editor.count())
                        if editor.itemData(position) == control.value
                    ),
                    -1,
                )
            if index < 0:
                raise ValueError(
                    f"choice parameter {control.name!r} has no current value"
                )
            editor.setCurrentIndex(index)

    @staticmethod
    def _configure_integer(
        editor: QtWidgets.QSpinBox,
        control: ParameterControl,
    ) -> None:
        minimum = _INTEGER_MINIMUM if control.minimum is None else int(control.minimum)
        maximum = _INTEGER_MAXIMUM if control.maximum is None else int(control.maximum)
        step = 1 if control.step is None else max(1, int(control.step))
        with signals_blocked(editor):
            editor.setRange(minimum, maximum)
            editor.setSingleStep(step)
            editor.setSpecialValueText("")

    @staticmethod
    def _configure_number(
        editor: QtWidgets.QDoubleSpinBox,
        control: ParameterControl,
    ) -> None:
        minimum = _NUMBER_MINIMUM if control.minimum is None else float(control.minimum)
        maximum = _NUMBER_MAXIMUM if control.maximum is None else float(control.maximum)
        step = 0.1 if control.step is None else float(control.step)
        with signals_blocked(editor):
            editor.setRange(minimum, maximum)
            editor.setSingleStep(step)
            editor.setSpecialValueText("")


class FluentPlotParameterPanel(QtWidgets.QWidget):
    """Reconcile one worker-owned plot parameter surface into Fluent widgets.

    The panel retains only widget bindings and the supplied host reference. It
    neither exposes worker internals nor mirrors plot or display state.
    Unchanged controls keep their widget identities while rows that appeared
    or disappeared are changed in place.
    """

    parameterRejected = QtCore.pyqtSignal(str, object)
    frontReady = QtCore.pyqtSignal(object)

    def __init__(
        self,
        host: RasterPlotHost,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(host, RasterPlotHost):
            raise TypeError("host must be RasterPlotHost")
        super().__init__(parent)
        self._host = host
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.controls = FluentParameterControlSurface(self)
        self.controls.valueEdited.connect(self._submit)
        layout.addWidget(self.controls)
        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._event_lock = Lock()
        self._display_dirty = False
        self._closing = False
        self._subscription_future: Future | None = None
        self._unsubscribe_display: Callable[[], Future] | None = None
        self._description_future: Future | None = None
        self._mutations: dict[Future, str] = {}
        self.destroyed.connect(self._release_subscription)
        self._install_subscription()
        self.reconcile()

    @property
    def host(self) -> RasterPlotHost:
        return self._host

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.controls.parameter_names

    def editor(self, name: str) -> QtWidgets.QWidget:
        return self.controls.editor(name)

    def reconcile(self) -> None:
        """Request the current worker description for owner-thread projection."""

        self._require_owner()
        if self._description_future is None and not self._closing:
            self._request_description()
        else:
            with self._event_lock:
                self._display_dirty = True

    @staticmethod
    def _description_controls(
        description: DisplayDescription,
    ) -> tuple[ParameterControl, ...]:
        if not isinstance(description, DisplayDescription):
            raise TypeError("RasterPlotHost.describe_display() returned another type")
        return parameter_controls(
            description.parameter_schema,
            description.display_state.values,
            choice_overrides=description.parameter_choices,
        )

    def _submit(self, name: str, value: object) -> None:
        self._require_owner()
        try:
            future = self._host.set_parameter(name, value)
        except Exception as error:
            self.reconcile()
            self.parameterRejected.emit(name, error)
            return
        if not isinstance(future, Future):
            error = TypeError("RasterPlotHost.set_parameter() must return Future")
            self.reconcile()
            self.parameterRejected.emit(name, error)
            return
        self._mutations[future] = name
        self._wake_on_completion(future)

    def _install_subscription(self) -> None:
        panel_ref = ref(self)

        def display_changed(_state: object) -> None:
            panel = panel_ref()
            if panel is not None:
                panel._display_changed_from_worker()

        try:
            future = self._host.subscribe_display(display_changed)
        except Exception as error:
            self.parameterRejected.emit("", error)
            return
        if not isinstance(future, Future):
            self.parameterRejected.emit(
                "",
                TypeError("RasterPlotHost.subscribe_display() must return Future"),
            )
            return
        self._subscription_future = future
        self._wake_on_completion(future)

    def _request_description(self) -> None:
        try:
            future = self._host.describe_display()
        except Exception as error:
            self.parameterRejected.emit("", error)
            return
        if not isinstance(future, Future):
            self.parameterRejected.emit(
                "",
                TypeError("RasterPlotHost.describe_display() must return Future"),
            )
            return
        self._description_future = future
        self._wake_on_completion(future)

    def _wake_on_completion(self, future: Future) -> None:
        panel_ref = ref(self)

        def completed(_future: Future) -> None:
            panel = panel_ref()
            if panel is not None and not panel._closing:
                panel._wake.request_owner_wake()

        future.add_done_callback(completed)

    def _display_changed_from_worker(self) -> None:
        with self._event_lock:
            if self._closing:
                return
            self._display_dirty = True
        self._wake.request_owner_wake()

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        self._require_owner()
        if self._closing:
            return
        self._accept_subscription()
        self._accept_description()
        self._accept_mutations()
        if self._description_future is not None:
            return
        with self._event_lock:
            dirty = self._display_dirty
            self._display_dirty = False
        if dirty:
            self._request_description()

    def _accept_subscription(self) -> None:
        future = self._subscription_future
        if future is None or not future.done():
            return
        self._subscription_future = None
        try:
            unsubscribe = future.result().value
            if not callable(unsubscribe):
                raise TypeError("display subscription did not return an unsubscribe")
        except BaseException as error:
            self.parameterRejected.emit("", error)
        else:
            self._unsubscribe_display = unsubscribe

    def _accept_description(self) -> None:
        future = self._description_future
        if future is None or not future.done():
            return
        self._description_future = None
        try:
            description = future.result().value
            controls = self._description_controls(description)
            self.controls.reconcile_controls(controls)
        except CancelledError:
            with self._event_lock:
                self._display_dirty = True
        except BaseException as error:
            self.parameterRejected.emit("", error)

    def _accept_mutations(self) -> None:
        completed = tuple(future for future in self._mutations if future.done())
        for future in completed:
            name = self._mutations.pop(future)
            try:
                operation = future.result()
            except CancelledError:
                continue
            except BaseException as error:
                self.parameterRejected.emit(name, error)
            else:
                self.frontReady.emit(operation.front)
            with self._event_lock:
                self._display_dirty = True

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("plot parameter work must run on its Qt owner")

    def _release_subscription(self, *_args: object) -> None:
        self._closing = True
        self._wake.detach()
        unsubscribe = self._unsubscribe_display
        self._unsubscribe_display = None
        if unsubscribe is not None:
            unsubscribe()
        setup = self._subscription_future
        self._subscription_future = None
        if setup is not None:
            setup.add_done_callback(self._release_completed_subscription)

    @staticmethod
    def _release_completed_subscription(future: Future) -> None:
        try:
            unsubscribe = future.result().value
            if callable(unsubscribe):
                unsubscribe()
        except BaseException:
            pass


__all__ = ["FluentParameterControlSurface", "FluentPlotParameterPanel"]
