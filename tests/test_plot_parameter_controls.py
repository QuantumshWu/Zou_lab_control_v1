from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
import os
from threading import Thread

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets  # noqa: E402

from zlc_frontend.qt_widgets import (  # noqa: E402
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentLineEdit,
    FluentPlotParameterPanel,
    FluentSpinBox,
    FluentSwitch,
    ensure_qt_app,
)
from zlc_plot import (  # noqa: E402
    ControlKind,
    DisplayDescription,
    NumericRange,
    ParameterControl,
    PlotKind,
    RasterOperation,
    RasterPlotHost,
    RectangleRange,
)
from zlc_plot.parameters import (  # noqa: E402
    ParameterSchema,
    ParameterSpec,
    RenderEffect,
)
from zlc_plot.state import DisplayState  # noqa: E402


def _control(
    name: str,
    kind: ControlKind,
    value: object,
    *,
    allow_none: bool = False,
    choices: tuple[object, ...] = (),
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
) -> ParameterControl:
    return ParameterControl(
        name=name,
        label=name.replace("_", " ").title(),
        kind=kind,
        value=value,
        allow_none=allow_none,
        choices=choices,
        minimum=minimum,
        maximum=maximum,
        step=step,
        effects=RenderEffect.TEXT,
    )


def _finished(value: object, front: object = None) -> Future:
    future = Future()
    future.set_result(RasterOperation(value, front))
    return future


class _ControlledHost(RasterPlotHost):
    """Raster host test double retaining the public worker boundary."""

    def __init__(self, controls: tuple[ParameterControl, ...]) -> None:
        self.controls = controls
        self.submissions: list[tuple[str, object]] = []
        self.callbacks = []
        self.revision = 0
        self.published_fronts: list[object] = []

    def subscribe_display(self, callback) -> Future:
        self.callbacks.append(callback)

        def unsubscribe() -> Future:
            if callback in self.callbacks:
                self.callbacks.remove(callback)
            return _finished(None)

        return _finished(unsubscribe)

    def describe_display(self) -> Future:
        return _finished(self._description())

    def set_parameter(self, name: str, value: object) -> Future:
        self.submissions.append((name, value))
        self.controls = tuple(
            replace(control, value=value) if control.name == name else control
            for control in self.controls
        )
        self.revision += 1
        state = self._state()
        for callback in tuple(self.callbacks):
            callback(state)
        front = object()
        self.published_fronts.append(front)
        return _finished(state, front)

    def replace_controls(self, controls: tuple[ParameterControl, ...]) -> None:
        self.controls = controls
        self.revision += 1
        state = self._state()
        for callback in tuple(self.callbacks):
            callback(state)

    def _schema(self) -> ParameterSchema:
        value_types = {
            ControlKind.BOOLEAN: bool,
            ControlKind.INTEGER: int,
            ControlKind.NUMBER: (int, float),
            ControlKind.TEXT: str,
            ControlKind.CHOICE: str,
        }
        return ParameterSchema(
            ParameterSpec(
                name=control.name,
                label=control.label,
                value_type=value_types[control.kind],
                effects=control.effects,
                default=control.value,
                allow_none=control.allow_none,
                choices=control.choices,
                minimum=control.minimum,
                maximum=control.maximum,
                step=control.step,
            )
            for control in self.controls
        )

    def _state(self, schema: ParameterSchema | None = None) -> DisplayState:
        selected = self._schema() if schema is None else schema
        return DisplayState(
            revision=self.revision,
            values=selected.initial_values(),
            changed_names=frozenset(),
            effects=RenderEffect.TEXT,
        )

    def _description(self) -> DisplayDescription:
        schema = self._schema()
        return DisplayDescription(
            kind=PlotKind.CURVE,
            size="default",
            size_choices=("default",),
            parameter_schema=schema,
            display_state=self._state(schema),
            parameter_choices={
                control.name: control.choices for control in self.controls
            },
            limits=RectangleRange(NumericRange(0.0, 1.0), NumericRange(0.0, 1.0)),
            viewport=None,
        )


def _host() -> _ControlledHost:
    return _ControlledHost(
        (
            _control("visible", ControlKind.BOOLEAN, True),
            _control(
                "count",
                ControlKind.INTEGER,
                2,
                minimum=1,
                maximum=9,
                step=2,
            ),
            _control(
                "threshold",
                ControlKind.NUMBER,
                None,
                allow_none=True,
                minimum=-5.0,
                maximum=5.0,
                step=0.25,
            ),
            _control("title", ControlKind.TEXT, "Initial"),
            _control(
                "mode",
                ControlKind.CHOICE,
                "a",
                choices=("a", "b"),
            ),
        )
    )


def _settle(application: QtWidgets.QApplication) -> None:
    for _ in range(8):
        application.processEvents()


def test_mapper_uses_fluent_widgets_and_mutates_the_worker_host() -> None:
    application = ensure_qt_app()
    host = _host()
    panel = FluentPlotParameterPanel(host)
    ready_fronts = []
    panel.frontReady.connect(ready_fronts.append)
    _settle(application)

    visible = panel.editor("visible")
    count = panel.editor("count")
    threshold = panel.editor("threshold")
    title = panel.editor("title")
    mode = panel.editor("mode")

    assert isinstance(visible, FluentSwitch)
    assert isinstance(count, FluentSpinBox)
    assert isinstance(threshold, FluentDoubleSpinBox)
    assert isinstance(title, FluentLineEdit)
    assert isinstance(mode, FluentComboBox)
    assert threshold.specialValueText() == ""
    assert "auto" not in threshold.lineEdit().text().lower()
    assert "none" not in threshold.lineEdit().text().lower()
    assert not threshold.isEnabled()

    visible.setChecked(False)
    count.setValue(4)
    title.setText("Changed")
    title.editingFinished.emit()
    mode.setCurrentIndex(mode.findData("b"))
    automatic = next(
        switch
        for switch in panel.findChildren(FluentSwitch)
        if switch.text() == "Automatic"
    )
    automatic.setChecked(False)
    threshold.setValue(2.5)
    _settle(application)

    assert ("visible", False) in host.submissions
    assert ("count", 4) in host.submissions
    assert ("title", "Changed") in host.submissions
    assert ("mode", "b") in host.submissions
    assert ("threshold", 2.5) in host.submissions
    assert ready_fronts == host.published_fronts

    panel.deleteLater()
    application.processEvents()


def test_reconcile_keeps_widgets_and_refreshes_dynamic_choices() -> None:
    application = ensure_qt_app()
    host = _host()
    panel = FluentPlotParameterPanel(host)
    _settle(application)
    original = {name: panel.editor(name) for name in panel.parameter_names}

    controls = tuple(
        replace(
            control,
            label="Acquisition mode",
            value="c",
            choices=("b", "c", "d"),
        )
        if control.name == "mode"
        else replace(control, value=7, maximum=12)
        if control.name == "count"
        else control
        for control in host.controls
    )
    worker = Thread(target=host.replace_controls, args=(controls,))
    worker.start()
    worker.join()
    _settle(application)

    for name, editor in original.items():
        assert panel.editor(name) is editor
    mode = panel.editor("mode")
    assert isinstance(mode, QtWidgets.QComboBox)
    assert [mode.itemData(index) for index in range(mode.count())] == ["b", "c", "d"]
    assert mode.currentData() == "c"
    count = panel.editor("count")
    assert isinstance(count, QtWidgets.QSpinBox)
    assert count.value() == 7
    assert count.maximum() == 12

    panel.deleteLater()
    application.processEvents()
