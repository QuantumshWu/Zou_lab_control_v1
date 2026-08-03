"""Fluent projection of the PlotSpec authoring contract owned by ``zlc_plot``."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from weakref import ref

from PyQt5 import QtCore, QtWidgets

from zlc_data import DatasetSchema
from zlc_plot import (
    PlotKind,
    PlotLabels,
    PlotSessionConfig,
    PlotSpec,
    RasterOperation,
    RasterPlotHost,
    plot_spec_controls,
    resolve_plot_spec,
)

from .owner_wake import QtOwnerWake
from .plot_parameters import FluentParameterControlSurface


class FluentPlotSpecPanel(QtWidgets.QWidget):
    """Edit one zlc_plot-owned PlotSpec before or after host creation.

    With a host, accepted drafts are submitted through ``replace_spec``.  With
    no host, the same controls resolve an initial PlotSpec and emit it to the
    composition owner.  The widget never constructs kind-specific specs.
    """

    frontReady = QtCore.pyqtSignal(object)
    specAccepted = QtCore.pyqtSignal(object)
    specRejected = QtCore.pyqtSignal(object)

    def __init__(
        self,
        host: RasterPlotHost | None,
        schema: DatasetSchema,
        parent: QtWidgets.QWidget | None = None,
        *,
        kind: PlotKind | None = None,
        spec: PlotSpec | None = None,
    ) -> None:
        if host is not None and not isinstance(host, RasterPlotHost):
            raise TypeError("host must be RasterPlotHost or None")
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if host is None and not isinstance(kind, PlotKind):
            raise TypeError("host-free PlotSpec authoring requires PlotKind")
        if kind is not None and not isinstance(kind, PlotKind):
            raise TypeError("kind must be PlotKind or None")
        if spec is not None and not isinstance(spec, PlotSpec):
            raise TypeError("spec must be a zlc_plot PlotSpec or None")
        if spec is not None and kind is not None and spec.kind is not kind:
            raise ValueError("initial PlotSpec kind differs from authoring kind")
        super().__init__(parent)
        self._host = host
        self._schema = schema
        self._kind = spec.kind if spec is not None else kind
        self._spec = spec
        self._draft: dict[str, object] = {}
        self._controls = ()
        self._configuration: Future | None = None
        self._replacements: dict[Future, int] = {}
        self._submit_serial = 0
        self._closing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.controls = FluentParameterControlSurface(self)
        self.controls.valueEdited.connect(self._edit_value)
        layout.addWidget(self.controls)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self.destroyed.connect(self._release)
        if host is None:
            self._reconcile_controls()
        else:
            self.reconcile()

    @property
    def host(self) -> RasterPlotHost | None:
        return self._host

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.controls.parameter_names

    def editor(self, name: str) -> QtWidgets.QWidget:
        return self.controls.editor(name)

    def reconcile(self) -> None:
        self._require_owner()
        host = self._host
        if host is None:
            self._reconcile_controls()
            return
        if self._closing or self._configuration is not None:
            return
        try:
            future = host.configuration()
            if not isinstance(future, Future):
                raise TypeError("RasterPlotHost.configuration() must return Future")
        except BaseException as error:
            self.specRejected.emit(error)
            return
        self._configuration = future
        self._wake_on_completion(future)

    def _reconcile_controls(self) -> None:
        kind = self._kind
        if kind is None:
            return
        self._controls = plot_spec_controls(
            self._schema,
            kind,
            spec=self._spec,
            values=self._draft or None,
        )
        self.controls.reconcile_controls(self._controls)

    @QtCore.pyqtSlot(str, object)
    def _edit_value(self, name: str, value: object) -> None:
        self._require_owner()
        if self._closing or self._kind is None:
            return
        self._draft[str(name)] = value
        try:
            self._reconcile_controls()
            values = {control.name: control.value for control in self._controls}
            labels = PlotLabels() if self._spec is None else self._spec.labels
            candidate = resolve_plot_spec(
                self._schema,
                self._kind,
                values,
                labels=labels,
            )
        except (KeyError, TypeError, ValueError) as error:
            self.specRejected.emit(error)
            return
        host = self._host
        if host is None:
            self._spec = candidate
            self._draft.clear()
            self._reconcile_controls()
            self.specAccepted.emit(candidate)
            return
        try:
            future = host.replace_spec(candidate)
            if not isinstance(future, Future):
                raise TypeError("RasterPlotHost.replace_spec() must return Future")
        except BaseException as error:
            self.specRejected.emit(error)
            self.reconcile()
            return
        self._submit_serial += 1
        self._replacements[future] = self._submit_serial
        self._wake_on_completion(future)

    def _wake_on_completion(self, future: Future) -> None:
        panel_ref = ref(self)

        def completed(_future: Future) -> None:
            panel = panel_ref()
            if panel is not None and not panel._closing:
                panel._wake.request_owner_wake()

        future.add_done_callback(completed)

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        self._require_owner()
        if self._closing:
            return
        self._accept_configuration()
        self._accept_replacements()

    def _accept_configuration(self) -> None:
        future = self._configuration
        if future is None or not future.done():
            return
        self._configuration = None
        try:
            operation = future.result()
            if not isinstance(operation, RasterOperation) or not isinstance(
                operation.value, PlotSessionConfig
            ):
                raise TypeError("plot configuration returned another value")
        except CancelledError:
            return
        except BaseException as error:
            self.specRejected.emit(error)
            return
        config = operation.value
        self._kind = config.spec.kind
        self._spec = config.spec
        self._draft.clear()
        self._reconcile_controls()

    def _accept_replacements(self) -> None:
        completed = tuple(future for future in self._replacements if future.done())
        for future in completed:
            serial = self._replacements.pop(future)
            latest = serial == self._submit_serial
            try:
                operation = future.result()
                if not isinstance(operation, RasterOperation) or not isinstance(
                    operation.value, PlotSessionConfig
                ):
                    raise TypeError("plot replacement returned another value")
            except CancelledError:
                continue
            except BaseException as error:
                if latest:
                    self.specRejected.emit(error)
                    self.reconcile()
                continue
            if not latest:
                continue
            config = operation.value
            self._kind = config.spec.kind
            self._spec = config.spec
            self._draft.clear()
            self._reconcile_controls()
            self.frontReady.emit(operation.front)
            self.specAccepted.emit(config)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("PlotSpec controls must run on their Qt owner")

    def _release(self, *_args: object) -> None:
        self._closing = True
        self._wake.detach()
        self._configuration = None
        self._replacements.clear()


__all__ = ["FluentPlotSpecPanel"]
