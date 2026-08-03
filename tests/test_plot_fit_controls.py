from __future__ import annotations

from concurrent.futures import Future
import os
from threading import Thread
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets  # noqa: E402

from zlc_frontend.qt_widgets import (  # noqa: E402
    FluentComboBox,
    FluentLineEdit,
    FluentPlotFitPanel,
    ensure_qt_app,
)
from zlc_plot import PlotKind, RasterOperation, RasterPlotHost  # noqa: E402
from zlc_plot.fit import builtin_fit_models  # noqa: E402


def _finished(value: object, front: object) -> Future:
    future = Future()
    future.set_result(RasterOperation(value, front))
    return future


class _FitHost(RasterPlotHost):
    def __init__(self) -> None:
        self.models = (builtin_fit_models()[0],)
        self.catalog_front = object()
        self.fit_future: Future = Future()
        self.clear_future: Future = Future()
        self.fit_calls: list[tuple[object, object, bool]] = []
        self.clear_calls = 0

    def fit_models(self) -> Future:
        return _finished(self.models, self.catalog_front)

    def configuration(self) -> Future:
        return _finished(
            SimpleNamespace(spec=SimpleNamespace(kind=PlotKind.CURVE)),
            self.catalog_front,
        )

    def fit(self, model, *, initial=None, live=True, **_kwargs) -> Future:
        self.fit_calls.append((model, initial, live))
        return self.fit_future

    def clear_fit(self) -> Future:
        self.clear_calls += 1
        return self.clear_future


def _settle(application: QtWidgets.QApplication) -> None:
    for _ in range(12):
        application.processEvents()


def test_fit_panel_uses_one_command_field_and_forwards_only_result_and_front() -> None:
    application = ensure_qt_app()
    host = _FitHost()
    panel = FluentPlotFitPanel(host, live=True)
    accepted: list[object] = []
    fronts: list[object] = []
    rejected: list[tuple[str, object]] = []
    panel.fitAccepted.connect(accepted.append)
    panel.frontReady.connect(fronts.append)
    panel.fitRejected.connect(lambda action, error: rejected.append((action, error)))
    _settle(application)

    assert isinstance(panel.model_combo, FluentComboBox)
    assert isinstance(panel.initials_edit, FluentLineEdit)
    assert panel.model_combo.count() == 1
    stable = (
        panel.model_combo,
        panel.initials_edit,
        panel.fit_button,
        panel.clear_button,
    )

    panel.initials_edit.setText("center=50 amplitude=2")
    panel.fit_button.click()
    assert host.fit_calls == [
        (host.models[0], {"center": 50.0, "amplitude": 2.0}, True)
    ]
    assert panel.initials_edit.text() == "center=50, amplitude=2"

    result = object()
    fit_front = object()
    worker = Thread(
        target=host.fit_future.set_result,
        args=(RasterOperation(result, fit_front),),
    )
    worker.start()
    worker.join()
    _settle(application)

    assert accepted == [result]
    assert fronts == [fit_front]
    assert rejected == []
    assert stable == (
        panel.model_combo,
        panel.initials_edit,
        panel.fit_button,
        panel.clear_button,
    )

    clear_front = object()
    panel.clear_button.click()
    host.clear_future.set_result(RasterOperation(None, clear_front))
    _settle(application)
    assert host.clear_calls == 1
    assert accepted == [result]
    assert fronts == [fit_front, clear_front]

    panel.deleteLater()
    application.processEvents()


def test_fit_panel_reports_parser_failure_as_typed_error_without_submitting() -> None:
    application = ensure_qt_app()
    host = _FitHost()
    panel = FluentPlotFitPanel(host, live=False)
    rejected: list[tuple[str, object]] = []
    panel.fitRejected.connect(lambda action, error: rejected.append((action, error)))
    _settle(application)

    panel.initials_edit.setText("unknown=3")
    panel.fit_button.click()
    _settle(application)

    assert host.fit_calls == []
    assert len(rejected) == 1
    assert rejected[0][0] == "fit"
    assert isinstance(rejected[0][1], KeyError)

    panel.deleteLater()
    application.processEvents()
