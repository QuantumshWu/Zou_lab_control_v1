"""Qt projection of the headless simple-form contract.

Only this closed form registry knows how a field kind maps to a widget.  The
form owns no Apply workflow, revision, repository, run, domain object, or
hardware access; it only reads and writes an exact keyed draft.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import re
import sys
from types import MappingProxyType
from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_frontend.form import (
    FormChoice,
    FormFieldProps,
    FormSpec,
    parse_number_text,
)

from .fluent import (
    GREY,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentSectionLabel,
    FluentSettingRow,
    FluentSpinBox,
    FluentSwitch,
    FluentTreeComboBox,
    scaled_px,
    setting_label_width,
    signals_blocked,
)
from .signal_picker import (
    coerce_short_labels,
    fill_grouped_signal_combo,
    read_editable_combo,
)


_INT_TEXT = re.compile(r"[+-]?\d+")
_FLOAT_TEXT = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)
_QT_INT_MIN = -(2**31)
_QT_INT_MAX = 2**31 - 1


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class FormRuntimeContext:
    """Qt-only live providers used by dynamic form fields."""

    signal_names: Callable[[str], object] | None = None
    signal_sources: Callable[[], object] | None = None
    signal_formats: Callable[[], object] | None = None
    signal_labels: Callable[[], object] | None = None

    def names_for(self, key: str) -> tuple[str, ...]:
        if not callable(self.signal_names):
            return ()
        return tuple(str(value) for value in self.signal_names(str(key)))

    @staticmethod
    def _mapping(provider) -> Mapping[str, object]:
        if not callable(provider):
            return _empty_mapping()
        value = provider()
        if not isinstance(value, Mapping):
            raise TypeError("dynamic form provider must return a mapping")
        return value

    def sources(self) -> Mapping[str, object]:
        return self._mapping(self.signal_sources)

    def formats(self) -> Mapping[str, object]:
        return self._mapping(self.signal_formats)

    def labels(self) -> Mapping[str, object]:
        return coerce_short_labels(self.signal_labels)

def _value_error(field: FormFieldProps, message: str) -> ValueError:
    return ValueError(f"field {field.key!r}: {message}")


def _connect_change(signal, on_change: Callable[[], None]) -> None:
    signal.connect(lambda *_args: on_change())


class FormWidgetHandler(ABC):
    """The complete typed lifecycle for one field kind."""

    @abstractmethod
    def normalize(self, field: FormFieldProps, value: object) -> object:
        """Validate/coerce without touching a widget, for atomic population."""

    @abstractmethod
    def build(
        self,
        field: FormFieldProps,
        value: object,
        on_change: Callable[[], None],
        context: FormRuntimeContext | None = None,
    ) -> QtWidgets.QWidget:
        """Construct, seed, and wire one widget."""

    @abstractmethod
    def read(self, field: FormFieldProps, widget: QtWidgets.QWidget) -> object:
        """Read one typed value without evaluating free text."""

    @abstractmethod
    def write(
        self,
        field: FormFieldProps,
        widget: QtWidgets.QWidget,
        value: object,
    ) -> None:
        """Write one already validated value."""

    @abstractmethod
    def is_empty(self, field: FormFieldProps, widget: QtWidgets.QWidget) -> bool:
        """Report whether a required value is absent."""

    @abstractmethod
    def refresh(
        self,
        field: FormFieldProps,
        widget: QtWidgets.QWidget,
        context: FormRuntimeContext | None = None,
    ) -> None:
        """Refresh presentation options while preserving a legal selection."""

class _StaticHandler(FormWidgetHandler):
    def refresh(self, field, widget, context=None) -> None:
        del field, widget, context


class _TextHandler(_StaticHandler):
    def normalize(self, field: FormFieldProps, value: object) -> str:
        if value is None and field.default is None:
            return ""
        if not isinstance(value, str):
            raise _value_error(field, "value must be str")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        edit = FluentLineEdit()
        edit.setMinimumWidth(scaled_px(160, minimum=120))
        edit.setPlaceholderText(field.description[:48])
        edit.setToolTip(field.description)
        self.write(field, edit, value)
        _connect_change(edit.textChanged, on_change)
        return edit

    def read(self, field, widget):
        del field
        return widget.text()

    def write(self, field, widget, value):
        widget.setText(self.normalize(field, value))

    def is_empty(self, field, widget):
        del field
        return not widget.text().strip()


class _IntHandler(_StaticHandler):
    @staticmethod
    def _spin_range(field: FormFieldProps) -> tuple[int, int] | None:
        # QSpinBox is a numeric value editor, not a tagged ``None | int``
        # control.  Optional numbers use the validated blank-or-number editor
        # below; never steal an out-of-domain integer and paint it as text.
        if field.blank_allowed:
            return None
        minimum = _QT_INT_MIN if field.minimum is None else int(field.minimum)
        maximum = _QT_INT_MAX if field.maximum is None else int(field.maximum)
        if not (_QT_INT_MIN <= minimum <= maximum <= _QT_INT_MAX):
            return None
        return minimum, maximum

    @classmethod
    def _configure_spin(cls, field: FormFieldProps, widget: FluentSpinBox) -> None:
        spin_range = cls._spin_range(field)
        if spin_range is None:
            raise ValueError("field cannot be represented by FluentSpinBox")
        widget.setRange(*spin_range)

    def normalize(self, field: FormFieldProps, value: object) -> int | None:
        if value is None:
            if field.blank_allowed:
                return None
            raise _value_error(field, "value cannot be None")
        if not isinstance(value, int) or isinstance(value, bool):
            raise _value_error(field, "value must be int")
        if field.minimum is not None and value < field.minimum:
            raise _value_error(field, f"value is below {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise _value_error(field, f"value is above {field.maximum}")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        spin_range = self._spin_range(field)
        if spin_range is not None:
            widget = FluentSpinBox()
            self._configure_spin(field, widget)
            self.write(field, widget, value)
            _connect_change(widget.valueChanged, on_change)
        else:
            widget = FluentLineEdit()
            widget.setMinimumWidth(scaled_px(120, minimum=96))
            widget.setPlaceholderText("(optional)" if field.blank_allowed else "")
            widget.set_numeric_validator(
                "int",
                bottom=field.minimum,
                top=field.maximum,
            )
            self.write(field, widget, value)
            _connect_change(widget.textChanged, on_change)
        widget.setToolTip(field.description)
        return widget

    def read(self, field, widget):
        if isinstance(widget, FluentSpinBox):
            if not widget.hasAcceptableInput():
                raise _value_error(field, "value is not a base-10 integer")
            return self.normalize(field, int(widget.value()))
        text = widget.text().strip()
        if not text:
            return self.normalize(field, None)
        if _INT_TEXT.fullmatch(text) is None:
            raise _value_error(field, "value is not a base-10 integer")
        return self.normalize(field, int(text, 10))

    def write(self, field, widget, value):
        prepared = self.normalize(field, value)
        if isinstance(widget, FluentSpinBox):
            if prepared is None:
                raise _value_error(field, "numeric spin cannot represent None")
            widget.setValue(prepared)
        else:
            widget.setText("" if prepared is None else str(prepared))

    def is_empty(self, field, widget):
        if isinstance(widget, FluentSpinBox):
            return False
        return not widget.text().strip()


class _NumberHandler(_StaticHandler):
    """Lossless ``int | float`` editor for owner contracts that accept both.

    Unlike a float spin box, entering ``1`` remains the integer ``1`` and an
    existing ``1.0`` remains a float on an untouched round trip.  Pulse API
    values need this distinction because their canonical lineage retains the
    authored numeric token even though hardware binding later validates it.
    """

    def normalize(
        self,
        field: FormFieldProps,
        value: object,
    ) -> int | float | None:
        if value is None:
            if field.blank_allowed:
                return None
            raise _value_error(field, "value cannot be None")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _value_error(field, "value must be an int or float")
        if isinstance(value, float) and not math.isfinite(value):
            raise _value_error(field, "value must be finite")
        if field.minimum is not None and value < field.minimum:
            raise _value_error(field, f"value is below {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise _value_error(field, f"value is above {field.maximum}")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        widget = FluentLineEdit()
        widget.setMinimumWidth(scaled_px(120, minimum=96))
        widget.setPlaceholderText("(optional)" if field.blank_allowed else "")
        widget.set_numeric_validator(
            "float",
            bottom=field.minimum,
            top=field.maximum,
        )
        widget.setToolTip(field.description)
        self.write(field, widget, value)
        _connect_change(widget.textChanged, on_change)
        return widget

    def read(self, field, widget):
        if not widget.text().strip():
            return self.normalize(field, None)
        try:
            value = parse_number_text(widget.text(), field.key)
        except (TypeError, ValueError) as exc:
            raise _value_error(field, str(exc)) from exc
        return self.normalize(field, value)

    def write(self, field, widget, value):
        prepared = self.normalize(field, value)
        widget.setText(
            ""
            if prepared is None
            else str(prepared)
            if isinstance(prepared, int)
            else repr(prepared)
        )

    def is_empty(self, field, widget):
        del field
        return not widget.text().strip()


class _LosslessFloatSpinBox(FluentDoubleSpinBox):
    """A bounded Fluent spin whose display round-trips the stored IEEE float."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # QDoubleSpinBox otherwise rounds its stored value to its display decimals.
        # The repr formatter below keeps the visible text compact despite this limit.
        self.setDecimals(323)

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        return repr(float(value))

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt API
        return float(text.strip())

    def validate(self, text: str, position: int):
        stripped = text.strip()
        if stripped in {"", "+", "-", ".", "+.", "-.", "e", "E"} or re.fullmatch(
            r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?)?",
            stripped,
        ):
            if _FLOAT_TEXT.fullmatch(stripped) is None:
                return QtGui.QValidator.Intermediate, text, position
        if _FLOAT_TEXT.fullmatch(stripped) is None:
            return QtGui.QValidator.Invalid, text, position
        value = float(stripped)
        if not math.isfinite(value):
            return QtGui.QValidator.Invalid, text, position
        if not self.minimum() <= value <= self.maximum():
            return QtGui.QValidator.Intermediate, text, position
        return QtGui.QValidator.Acceptable, text, position


class _FloatHandler(_StaticHandler):
    @staticmethod
    def _spin_range(field: FormFieldProps) -> tuple[float, float] | None:
        # QDoubleSpinBox owns one real number only.  Optional numbers remain a
        # typed blank-or-number edit instead of smuggling None through an
        # out-of-domain floating-point value and a textual numeric sentinel.
        if field.blank_allowed:
            return None
        minimum = (
            -sys.float_info.max
            if field.minimum is None
            else float(field.minimum)
        )
        maximum = (
            sys.float_info.max
            if field.maximum is None
            else float(field.maximum)
        )
        return minimum, maximum

    @classmethod
    def _configure_spin(
        cls,
        field: FormFieldProps,
        widget: FluentDoubleSpinBox,
    ) -> None:
        spin_range = cls._spin_range(field)
        if spin_range is None:
            raise ValueError("field cannot be represented by FluentDoubleSpinBox")
        widget.setRange(*spin_range)

    def normalize(self, field: FormFieldProps, value: object) -> float | None:
        if value is None:
            if field.blank_allowed:
                return None
            raise _value_error(field, "value cannot be None")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _value_error(field, "value must be a real number")
        result = float(value)
        if not math.isfinite(result):
            raise _value_error(field, "value must be finite")
        if field.minimum is not None and result < field.minimum:
            raise _value_error(field, f"value is below {field.minimum}")
        if field.maximum is not None and result > field.maximum:
            raise _value_error(field, f"value is above {field.maximum}")
        return result

    def build(self, field, value, on_change, context=None):
        del context
        spin_range = self._spin_range(field)
        if spin_range is not None:
            widget = _LosslessFloatSpinBox()
            self._configure_spin(field, widget)
            self.write(field, widget, value)
            _connect_change(widget.valueChanged, on_change)
        else:
            widget = FluentLineEdit()
            widget.setMinimumWidth(scaled_px(120, minimum=96))
            widget.setPlaceholderText("(optional)" if field.blank_allowed else "")
            widget.set_numeric_validator(
                "float",
                bottom=field.minimum,
                top=field.maximum,
            )
            self.write(field, widget, value)
            _connect_change(widget.textChanged, on_change)
        widget.setToolTip(field.description)
        return widget

    def read(self, field, widget):
        if isinstance(widget, FluentDoubleSpinBox):
            if not widget.hasAcceptableInput():
                raise _value_error(field, "value is not a finite decimal number")
            return self.normalize(field, float(widget.value()))
        text = widget.text().strip()
        if not text:
            return self.normalize(field, None)
        if _FLOAT_TEXT.fullmatch(text) is None:
            raise _value_error(field, "value is not a finite decimal number")
        return self.normalize(field, float(text))

    def write(self, field, widget, value):
        prepared = self.normalize(field, value)
        if isinstance(widget, FluentDoubleSpinBox):
            if prepared is None:
                raise _value_error(field, "numeric spin cannot represent None")
            widget.setValue(prepared)
        else:
            widget.setText("" if prepared is None else repr(prepared))

    def is_empty(self, field, widget):
        if isinstance(widget, FluentDoubleSpinBox):
            return False
        return not widget.text().strip()


class _BoolHandler(_StaticHandler):
    def normalize(self, field: FormFieldProps, value: object) -> bool:
        if not isinstance(value, bool):
            raise _value_error(field, "value must be bool")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        widget = FluentSwitch("")
        widget.setToolTip(field.description)
        self.write(field, widget, value)
        _connect_change(widget.toggled, on_change)
        return widget

    def read(self, field, widget):
        return self.normalize(field, bool(widget.isChecked()))

    def write(self, field, widget, value):
        widget.setChecked(self.normalize(field, value))

    def is_empty(self, field, widget):
        del field, widget
        return False


class _ChoiceHandler(FormWidgetHandler):
    def normalize(self, field: FormFieldProps, value: object) -> object:
        if value is None:
            return None
        choice = field.choice_for(value)
        if choice is None:
            raise _value_error(field, "value is not one of the typed choices")
        return choice.value

    @staticmethod
    def _fill(widget: FluentComboBox, choices: tuple[FormChoice, ...]) -> None:
        widget.clear()
        for choice in choices:
            widget.addItem(choice.label, choice.value)

    def build(self, field, value, on_change, context=None):
        del context
        widget = FluentComboBox()
        self._fill(widget, field.choices)
        self.write(field, widget, value)
        widget.setEnabled(not field.required_choice_unavailable)
        widget.setToolTip(field.unavailable_reason or field.description)
        _connect_change(widget.activated, on_change)
        return widget

    def read(self, field, widget):
        if widget.currentIndex() < 0:
            return None
        return self.normalize(field, widget.currentData())

    def write(self, field, widget, value):
        prepared = self.normalize(field, value)
        if prepared is None:
            widget.setCurrentIndex(-1)
            return
        choice = field.choice_for(prepared)
        assert choice is not None
        index = next(index for index, item in enumerate(field.choices) if item is choice)
        widget.setCurrentIndex(index)

    def is_empty(self, field, widget):
        del field
        return widget.currentIndex() < 0

    def refresh(self, field, widget, context=None):
        del context
        current = self.read(field, widget)
        self._fill(widget, field.choices)
        self.write(field, widget, current)
        widget.setEnabled(not field.required_choice_unavailable)
        widget.setToolTip(field.unavailable_reason or field.description)


def _grey_label(text: str) -> FluentLabel:
    label = FluentLabel(text)
    label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
    return label


class _AxisRangeHandler(_StaticHandler):
    def normalize(self, field, value):
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise _value_error(field, "value must be (minimum, maximum, points)")
        lo, hi, points = value
        if (
            isinstance(lo, bool)
            or isinstance(hi, bool)
            or not isinstance(lo, (int, float))
            or not isinstance(hi, (int, float))
            or not isinstance(points, int)
            or isinstance(points, bool)
        ):
            raise _value_error(field, "range endpoints must be numeric and points int")
        lo_value, hi_value = float(lo), float(hi)
        if not math.isfinite(lo_value) or not math.isfinite(hi_value):
            raise _value_error(field, "range endpoints must be finite")
        if field.minimum is not None and min(lo_value, hi_value) < field.minimum:
            raise _value_error(field, f"range is below {field.minimum}")
        if field.maximum is not None and max(lo_value, hi_value) > field.maximum:
            raise _value_error(field, f"range is above {field.maximum}")
        if points < 2 or points > 100_000:
            raise _value_error(field, "points must be between 2 and 100000")
        return lo_value, hi_value, points

    def build(self, field, value, on_change, context=None):
        del context
        lo, hi, points = self.normalize(field, value)
        minimum = -1.0e12 if field.minimum is None else float(field.minimum)
        maximum = 1.0e12 if field.maximum is None else float(field.maximum)
        digits = max(5, len(str(int(abs(maximum) + 1))) + 4)
        lo_spin = FluentDoubleSpinBox(
            length=digits,
            allow_minus=minimum < 0,
        )
        hi_spin = FluentDoubleSpinBox(
            length=digits,
            allow_minus=minimum < 0,
        )
        for spin, seed in ((lo_spin, lo), (hi_spin, hi)):
            spin.setRange(minimum, maximum)
            spin.setValue(seed)
            spin.setToolTip(field.description)
        points_spin = FluentSpinBox()
        points_spin.setRange(2, 100_000)
        points_spin.setValue(points)
        points_spin.setToolTip("Number of scan points (>= 2).")

        host = QtWidgets.QWidget()
        host.setStyleSheet("background: transparent;")
        row = QtWidgets.QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(scaled_px(4, minimum=3))
        row.addWidget(lo_spin)
        row.addWidget(_grey_label("to"))
        row.addWidget(hi_spin)
        row.addWidget(_grey_label("/"))
        row.addWidget(points_spin)
        row.addWidget(_grey_label("pts"))
        host.min_spin = lo_spin
        host.max_spin = hi_spin
        host.pts_spin = points_spin
        for spin in (lo_spin, hi_spin, points_spin):
            _connect_change(spin.valueChanged, on_change)
        return host

    def read(self, field, widget):
        return self.normalize(
            field,
            (
                float(widget.min_spin.value()),
                float(widget.max_spin.value()),
                int(widget.pts_spin.value()),
            ),
        )

    def write(self, field, widget, value):
        lo, hi, points = self.normalize(field, value)
        widget.min_spin.setValue(lo)
        widget.max_spin.setValue(hi)
        widget.pts_spin.setValue(points)

    def is_empty(self, field, widget):
        del field, widget
        return False


class _PathHandler(_StaticHandler):
    def normalize(self, field, value):
        if value is None:
            return ""
        if not isinstance(value, str):
            raise _value_error(field, "value must be a path string")
        return value

    def build(self, field, value, on_change, context=None):
        del context
        picker = FluentPathEdit(
            self.normalize(field, value),
            mode=field.path_mode,
            caption=f"Choose {field.label}",
            file_filter=field.file_filter,
            base_dir=field.base_dir,
        )
        picker.setToolTip(field.description)
        _connect_change(picker.changed, on_change)
        return picker

    def read(self, field, widget):
        return self.normalize(field, widget.text())

    def write(self, field, widget, value):
        widget.setText(self.normalize(field, value))

    def is_empty(self, field, widget):
        del field
        return not widget.text().strip()


class _SignalHandler(FormWidgetHandler):
    def normalize(self, field, value):
        if value is None:
            return ""
        if not isinstance(value, str):
            raise _value_error(field, "value must be a signal key string")
        return value

    @staticmethod
    def _fill(field, widget, current, context):
        runtime = context or FormRuntimeContext()
        fill_grouped_signal_combo(
            widget,
            names=runtime.names_for(field.key),
            sources=runtime.sources(),
            formats=runtime.formats(),
            labels=runtime.labels(),
            current=current,
        )

    def build(self, field, value, on_change, context=None):
        combo = FluentTreeComboBox()
        self._fill(field, combo, self.normalize(field, value), context)
        combo.setToolTip(field.description)
        _connect_change(combo.activated, on_change)
        return combo

    def read(self, field, widget):
        return self.normalize(field, read_editable_combo(widget))

    def write(self, field, widget, value):
        del field
        widget.select_signal("" if value is None else str(value))

    def is_empty(self, field, widget):
        del field
        return not read_editable_combo(widget)

    def refresh(self, field, widget, context=None):
        current = read_editable_combo(widget)
        self._fill(field, widget, current, context)


FORM_WIDGET_HANDLERS: Mapping[str, FormWidgetHandler] = MappingProxyType(
    {
        "text": _TextHandler(),
        "int": _IntHandler(),
        "float": _FloatHandler(),
        "number": _NumberHandler(),
        "choice": _ChoiceHandler(),
        "bool": _BoolHandler(),
        "axis_range": _AxisRangeHandler(),
        "path": _PathHandler(),
        "signal": _SignalHandler(),
    }
)


def _widget_family(field: FormFieldProps) -> str:
    """Concrete control family required by one declaration."""

    if field.kind == "int" and _IntHandler._spin_range(field) is not None:
        return "int-spin"
    if field.kind == "float" and _FloatHandler._spin_range(field) is not None:
        return "float-spin"
    if field.kind in {"text", "int", "float", "number"}:
        return "line-edit"
    if field.kind == "choice":
        return "choice"
    if field.kind == "bool":
        return "bool"
    if field.kind == "axis_range":
        return f"axis-range:{field.minimum}:{field.maximum}"
    if field.kind == "path":
        return f"path:{field.path_mode}:{field.file_filter}:{field.base_dir}"
    if field.kind == "signal":
        return "signal"
    raise ValueError(f"unsupported form field kind: {field.kind!r}")


def _same_typed_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _widget_has_value(
    handler: FormWidgetHandler,
    field: FormFieldProps,
    widget: QtWidgets.QWidget,
    value: object,
) -> bool:
    try:
        current = handler.read(field, widget)
    except (TypeError, ValueError):
        return False
    return _same_typed_value(current, value)


def _reconfigure_widget(
    old_field: FormFieldProps,
    field: FormFieldProps,
    widget: QtWidgets.QWidget,
) -> None:
    """Apply changed presentation constraints to one compatible control."""

    widget.setToolTip(field.unavailable_reason or field.description)
    if isinstance(widget, FluentLineEdit):
        if field.kind == "text":
            widget.setPlaceholderText(field.description[:48])
        elif field.kind in {"int", "float", "number"}:
            widget.setPlaceholderText("(optional)" if field.blank_allowed else "")
    elif isinstance(widget, FluentSpinBox):
        _IntHandler._configure_spin(field, widget)
    elif isinstance(widget, _LosslessFloatSpinBox):
        _FloatHandler._configure_spin(field, widget)
    elif isinstance(widget, FluentComboBox) and old_field.choices != field.choices:
        _ChoiceHandler._fill(widget, field.choices)
        widget.setEnabled(not field.required_choice_unavailable)


class FluentParameterForm(QtWidgets.QWidget):
    """Thin exact-key form built from one ordered :class:`FormSpec`."""

    changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        spec: FormSpec,
        values: Mapping[str, object] | None = None,
        parent=None,
        *,
        runtime: FormRuntimeContext | None = None,
        label_width: int | None = None,
    ) -> None:
        if not isinstance(spec, FormSpec):
            raise TypeError("spec must be FormSpec")
        if label_width is not None and (
            isinstance(label_width, bool)
            or not isinstance(label_width, int)
            or label_width <= 0
        ):
            raise ValueError("label_width must be a positive integer or None")
        super().__init__(parent)
        self._spec = spec
        self._runtime = runtime or FormRuntimeContext()
        self._label_width = label_width
        self._fields = {field.key: field for field in spec.fields}
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._handlers: dict[str, FormWidgetHandler] = {}
        self._rows: dict[str, QtWidgets.QWidget] = {}

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(scaled_px(6, minimum=4))
        label_width = self._label_width or setting_label_width(
            field.row_label for field in spec.fields
        )
        for field in spec.fields:
            handler = FORM_WIDGET_HANDLERS[field.kind]
            widget = handler.build(
                field,
                field.default,
                lambda key=field.key: self.changed.emit(key),
                self._runtime,
            )
            self._widgets[field.key] = widget
            self._handlers[field.key] = handler
            row = self._make_row(field, widget, label_width)
            self._rows[field.key] = row
            self._layout.addWidget(row)

        if values is not None:
            self.populate(values)

    def _make_row(self, field, widget, label_width):
        return FluentSettingRow(
            field.row_label,
            widget,
            label_width=label_width,
            parent=self,
        )

    @property
    def spec(self) -> FormSpec:
        return self._spec

    @property
    def keys(self) -> tuple[str, ...]:
        return self._spec.keys

    def widget_for(self, key: str) -> QtWidgets.QWidget:
        try:
            return self._widgets[key]
        except KeyError as exc:
            raise KeyError(f"unknown form field key: {key!r}") from exc

    def is_empty(self, key: str) -> bool:
        field = self._field_for(key)
        return self._handlers[key].is_empty(field, self._widgets[key])

    def read_value(self, key: str) -> object:
        """Read one edited leaf without promoting it to a whole-form snapshot."""
        field = self._field_for(key)
        handler, widget = self._handlers[key], self._widgets[key]
        if (
            field.required
            and handler.is_empty(field, widget)
            and not field.required_choice_unavailable
        ):
            raise _value_error(field, "required value is empty")
        try:
            return handler.read(field, widget)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("field "):
                raise
            raise _value_error(field, str(exc)) from exc

    def read_all(self) -> dict[str, object]:
        return {field.key: self.read_value(field.key) for field in self._spec.fields}

    def populate(self, values: Mapping[str, object]) -> None:
        """Atomically populate every exact key without emitting edit signals.

        All coercion and bound/choice checks finish before the first widget is
        changed, so an invalid full state cannot leave a partially updated form.
        """

        prepared = self._prepare_population(values)
        widgets = tuple(self._widgets[key] for key in self._spec.keys)
        with signals_blocked(*widgets):
            for field in self._spec.fields:
                self._handlers[field.key].write(
                    field, self._widgets[field.key], prepared[field.key]
                )

    def validate_population(self, values: Mapping[str, object]) -> None:
        """Validate one exact owner projection without mutating any widget."""

        self._prepare_population(values)

    def _prepare_population(
        self,
        values: Mapping[str, object],
    ) -> dict[str, object]:
        exact = self._require_exact_values(values)
        return {
            field.key: self._handlers[field.key].normalize(
                field, exact[field.key]
            )
            for field in self._spec.fields
        }

    def write_all(self, values: Mapping[str, object]) -> None:
        self.populate(values)

    def reconcile(
        self,
        spec: FormSpec,
        values: Mapping[str, object],
    ) -> None:
        """Keyed-diff a new declaration into this stable form owner.

        Same-key controls with a compatible concrete widget family are updated
        in place.  New keys create one row, removed keys destroy one row,
        reordering only moves existing rows, and a true widget-family change
        replaces only that key's row.  All values are normalized before the
        first QWidget mutation.
        """

        if not isinstance(spec, FormSpec):
            raise TypeError("spec must be FormSpec")
        if not isinstance(values, Mapping):
            raise TypeError("form values must be a mapping")
        incoming = {key: values[key] for key in values}
        supplied = set(incoming)
        expected = set(spec.keys)
        if supplied != expected:
            missing = sorted(repr(key) for key in expected - supplied)
            extra = sorted(repr(key) for key in supplied - expected)
            raise ValueError(
                f"form values must have exact keys; missing={missing}, extra={extra}"
            )
        new_handlers = {
            field.key: FORM_WIDGET_HANDLERS[field.kind]
            for field in spec.fields
        }
        prepared = {
            field.key: new_handlers[field.key].normalize(
                field, incoming[field.key]
            )
            for field in spec.fields
        }

        old_fields = self._fields
        replacements: dict[str, tuple[QtWidgets.QWidget, QtWidgets.QWidget]] = {}
        label_width = self._label_width or setting_label_width(
            field.row_label for field in spec.fields
        )
        for field in spec.fields:
            old_field = old_fields.get(field.key)
            if (
                old_field is not None
                and _widget_family(old_field) == _widget_family(field)
            ):
                continue
            handler = new_handlers[field.key]
            widget = handler.build(
                field,
                prepared[field.key],
                lambda key=field.key: self.changed.emit(key),
                self._runtime,
            )
            row = self._make_row(field, widget, label_width)
            replacements[field.key] = widget, row

        retained_widgets = tuple(
            self._widgets[field.key]
            for field in spec.fields
            if field.key not in replacements and field.key in self._widgets
        )
        self.setUpdatesEnabled(False)
        try:
            with signals_blocked(*retained_widgets):
                for field in spec.fields:
                    if field.key in replacements:
                        continue
                    old_field = old_fields[field.key]
                    widget = self._widgets[field.key]
                    handler = new_handlers[field.key]
                    _reconfigure_widget(old_field, field, widget)
                    if not _widget_has_value(
                        handler, field, widget, prepared[field.key]
                    ):
                        handler.write(field, widget, prepared[field.key])

            desired_keys = set(spec.keys)
            replaced_keys = set(replacements)
            for key, row in tuple(self._rows.items()):
                if key in desired_keys and key not in replaced_keys:
                    continue
                self._layout.removeWidget(row)
                # Reconcile may run while an Edit page is visible.  An
                # unparented QWidget becomes a transient native window; hide
                # the retired row and retain this form as QObject owner until
                # DeferredDelete instead.
                row.hide()
                row.deleteLater()
                self._rows.pop(key, None)
                self._widgets.pop(key, None)

            for key, (widget, row) in replacements.items():
                self._widgets[key] = widget
                self._rows[key] = row

            for index, field in enumerate(spec.fields):
                row = self._rows[field.key]
                if isinstance(row, FluentSettingRow):
                    row.set_label(field.row_label, width=label_width)
                self._layout.removeWidget(row)
                self._layout.insertWidget(index, row)

            self._spec = spec
            self._fields = {field.key: field for field in spec.fields}
            self._handlers = new_handlers
        finally:
            self.setUpdatesEnabled(True)

    def refresh(self) -> None:
        """Refresh every handler, preserving legal selections and edit silence."""

        widgets = tuple(self._widgets[key] for key in self._spec.keys)
        with signals_blocked(*widgets):
            for field in self._spec.fields:
                self._handlers[field.key].refresh(
                    field,
                    self._widgets[field.key],
                    self._runtime,
                )

    def _field_for(self, key: str) -> FormFieldProps:
        try:
            return self._fields[key]
        except KeyError as exc:
            raise KeyError(f"unknown form field key: {key!r}") from exc

    def _require_exact_values(
        self, values: Mapping[str, object]
    ) -> dict[str, object]:
        if not isinstance(values, Mapping):
            raise TypeError("form values must be a mapping")
        supplied = set(values.keys())
        expected = set(self._spec.keys)
        if supplied != expected:
            missing = sorted(repr(key) for key in expected - supplied)
            extra = sorted(repr(key) for key in supplied - expected)
            raise ValueError(
                f"form values must have exact keys; missing={missing}, extra={extra}"
            )
        return {key: values[key] for key in self._spec.keys}


__all__ = [
    "FORM_WIDGET_HANDLERS",
    "FluentParameterForm",
    "FormRuntimeContext",
    "FormWidgetHandler",
]
