"""Minimal Fluent fit controls for one worker-owned raster plot."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from weakref import ref

from PyQt5 import QtCore, QtWidgets

from zlc_plot import (
    FitModelSpec,
    PlotKind,
    RasterOperation,
    RasterPlotHost,
    format_fit_initials,
    parse_fit_initials,
)

from .fluent import (
    FluentButton,
    FluentComboBox,
    FluentLineEdit,
    FluentSettingRow,
    FluentSwitch,
    setting_label_width,
)
from .owner_wake import QtOwnerWake


class FluentPlotFitPanel(QtWidgets.QWidget):
    """Adapt a :class:`RasterPlotHost` fit surface onto stable Qt controls.

    The host remains the sole owner of models, data, fitting, overlays, and
    raster state.  This widget retains only its controls and in-flight Futures;
    accepted operations are reduced to their immutable front and FitResult.
    """

    frontReady = QtCore.pyqtSignal(object)
    fitAccepted = QtCore.pyqtSignal(object)
    fitRejected = QtCore.pyqtSignal(str, object)

    def __init__(
        self,
        host: RasterPlotHost,
        live: bool,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(host, RasterPlotHost):
            raise TypeError("host must be RasterPlotHost")
        if not isinstance(live, bool):
            raise TypeError("live must be bool")
        super().__init__(parent)
        self._host = host
        self._live = live
        self._closing = False
        self._model_future: Future | None = None
        self._configuration_future: Future | None = None
        self._operations: dict[Future, str] = {}
        self._deferred_errors: list[tuple[str, BaseException]] = []

        self.model_combo = FluentComboBox(self)
        self.model_combo.setObjectName("plotFitModel")
        self.model_combo.setEnabled(False)
        self.initials_edit = FluentLineEdit("", self)
        self.initials_edit.setObjectName("plotFitInitials")
        self.initials_edit.setPlaceholderText("center=50 amplitude=2")
        self.fit_button = FluentButton("Fit", self)
        self.fit_button.setObjectName("plotFitApply")
        self.fit_button.setEnabled(False)
        self.clear_button = FluentButton("Clear", self)
        self.clear_button.setObjectName("plotFitClear")
        self.all_facets_switch = FluentSwitch("All facets", self)
        self.all_facets_switch.setObjectName("plotFitAllFacets")
        self.all_facets_switch.setVisible(False)

        width = setting_label_width(("Model", "Initial"))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(
            FluentSettingRow("Model", self.model_combo, label_width=width, parent=self)
        )
        layout.addWidget(
            FluentSettingRow(
                "Initial",
                self.initials_edit,
                label_width=width,
                parent=self,
            )
        )
        self.all_facets_row = FluentSettingRow(
            "Scope",
            self.all_facets_switch,
            label_width=width,
            parent=self,
        )
        self.all_facets_row.setVisible(False)
        layout.addWidget(self.all_facets_row)
        actions = QtWidgets.QWidget(self)
        action_layout = QtWidgets.QHBoxLayout(actions)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        action_layout.addWidget(self.fit_button)
        action_layout.addWidget(self.clear_button)
        action_layout.addStretch(1)
        layout.addWidget(actions)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self.fit_button.clicked.connect(self._submit_fit)
        self.initials_edit.returnPressed.connect(self._submit_fit)
        self.clear_button.clicked.connect(self._submit_clear)
        self.destroyed.connect(self._release)
        self._request_models()

    @property
    def host(self) -> RasterPlotHost:
        return self._host

    @property
    def live(self) -> bool:
        return self._live

    @QtCore.pyqtSlot()
    def _submit_fit(self) -> None:
        self._require_owner()
        model = self.model_combo.currentData()
        if not isinstance(model, FitModelSpec):
            self._defer_error("fit", RuntimeError("no fit model is selected"))
            return
        try:
            initial = parse_fit_initials(model, self.initials_edit.text())
            canonical = format_fit_initials(model, initial)
            future = self._host.fit(
                model,
                initial=initial or None,
                live=self._live,
                fit_all_facets=self.all_facets_switch.isChecked(),
            )
        except BaseException as error:
            self._defer_error("fit", error)
            return
        self.initials_edit.setText(canonical)
        self._track_operation(future, "fit")

    @QtCore.pyqtSlot()
    def _submit_clear(self) -> None:
        self._require_owner()
        try:
            future = self._host.clear_fit()
        except BaseException as error:
            self._defer_error("clear", error)
            return
        self._track_operation(future, "clear")

    def _request_models(self) -> None:
        try:
            future = self._host.fit_models()
            if not isinstance(future, Future):
                raise TypeError("RasterPlotHost.fit_models() must return Future")
        except BaseException as error:
            self._defer_error("models", error)
            return
        self._model_future = future
        self._wake_on_completion(future)
        try:
            configuration = self._host.configuration()
            if not isinstance(configuration, Future):
                raise TypeError("RasterPlotHost.configuration() must return Future")
        except BaseException as error:
            self._defer_error("configuration", error)
        else:
            self._configuration_future = configuration
            self._wake_on_completion(configuration)

    def _track_operation(self, future: object, action: str) -> None:
        if not isinstance(future, Future):
            self._defer_error(
                action,
                TypeError(f"RasterPlotHost.{action} operation must return Future"),
            )
            return
        self._operations[future] = action
        self._wake_on_completion(future)

    def _wake_on_completion(self, future: Future) -> None:
        panel_ref = ref(self)

        def completed(_future: Future) -> None:
            panel = panel_ref()
            if panel is not None and not panel._closing:
                panel._wake.request_owner_wake()

        future.add_done_callback(completed)

    def _defer_error(self, action: str, error: BaseException) -> None:
        self._deferred_errors.append((action, error))
        self._wake.request_owner_wake()

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        self._require_owner()
        if self._closing:
            return
        self._accept_models()
        self._accept_configuration()
        self._accept_operations()
        errors, self._deferred_errors = self._deferred_errors, []
        for action, error in errors:
            self.fitRejected.emit(action, error)

    def _accept_models(self) -> None:
        future = self._model_future
        if future is None or not future.done():
            return
        self._model_future = None
        try:
            operation = future.result()
            if not isinstance(operation, RasterOperation):
                raise TypeError("fit_models completion must be RasterOperation")
            models = operation.value
            if not isinstance(models, tuple) or any(
                not isinstance(model, FitModelSpec) for model in models
            ):
                raise TypeError("fit_models value must be tuple[FitModelSpec, ...]")
            ids = tuple(model.model_id for model in models)
            if len(ids) != len(set(ids)):
                raise ValueError("fit model ids must be unique")
        except CancelledError:
            return
        except BaseException as error:
            self._deferred_errors.append(("models", error))
            return
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(model.display_name, model)
        available = bool(models)
        self.model_combo.setEnabled(available)
        self.fit_button.setEnabled(available)

    def _accept_configuration(self) -> None:
        future = self._configuration_future
        if future is None or not future.done():
            return
        self._configuration_future = None
        try:
            operation = future.result()
            if not isinstance(operation, RasterOperation):
                raise TypeError("configuration completion must be RasterOperation")
            config = operation.value
            kind = config.spec.kind
        except CancelledError:
            return
        except BaseException as error:
            self._deferred_errors.append(("configuration", error))
            return
        facets = kind is PlotKind.FACET_GRID
        self.all_facets_switch.setVisible(facets)
        self.all_facets_row.setVisible(facets)
        if not facets:
            self.all_facets_switch.setChecked(False)

    def _accept_operations(self) -> None:
        completed = tuple(future for future in self._operations if future.done())
        for future in completed:
            action = self._operations.pop(future)
            try:
                operation = future.result()
                if not isinstance(operation, RasterOperation):
                    raise TypeError(f"{action} completion must be RasterOperation")
            except CancelledError:
                continue
            except BaseException as error:
                self._deferred_errors.append((action, error))
                continue
            self.frontReady.emit(operation.front)
            if action == "fit":
                self.fitAccepted.emit(operation.value)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("plot fit controls must run on their Qt owner")

    def _release(self, *_args: object) -> None:
        self._closing = True
        self._wake.detach()
        self._model_future = None
        self._configuration_future = None
        self._operations.clear()
        self._deferred_errors.clear()


__all__ = ["FluentPlotFitPanel"]
