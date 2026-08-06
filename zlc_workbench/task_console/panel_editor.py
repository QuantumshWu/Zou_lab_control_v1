"""Snapshot-scoped Plot Edit surface backed by the sole ``zlc_plot`` host."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from datetime import datetime
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    ACCENT,
    GREY,
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentPlotFitPanel,
    FluentPlotParameterPanel,
    FluentPlotSpecPanel,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentSwitch,
    QtOwnerWake,
    scaled_px,
    setting_label_width,
)
from zlc_plot import Qt5PlotWidget, RasterFront, RasterOperation, RasterPlotHost
from zlc_workbench.retiring_hosts import RetiringPlotHosts

from .logic_node_parameter_panel import LogicNodeParameterPanel


_IMAGE_FORMATS = ("png", "pdf", "svg")


class PanelEditor(QtWidgets.QWidget):
    """A frozen surface that cannot pause or mutate its live Monitor sibling."""

    shutdownFinished = QtCore.pyqtSignal()

    def __init__(self, card: "PanelCard", console: "TaskConsole", parent) -> None:
        if parent is None:
            raise TypeError("PanelEditor requires its tab-stack parent")
        super().__init__(parent)
        self.card = card
        self.console = console
        self.render_surface_id = f"{card.panel_id}::edit::{id(self):x}"
        self._host: RasterPlotHost | None = None
        self._plot_widget = None
        self._initial_future: Future | None = None
        self._front_futures: set[Future] = set()
        self._save_futures: dict[Future, Path] = {}
        self._retiring_hosts = RetiringPlotHosts()
        self._closing = False
        self._closed = False
        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._shutdown_timer = QtCore.QTimer(self)
        self._shutdown_timer.setInterval(25)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        column = QtWidgets.QVBoxLayout(page)
        margin = scaled_px(10, minimum=6)
        column.setContentsMargins(margin, margin, margin, margin)
        column.setSpacing(scaled_px(6, minimum=4))
        width = setting_label_width(("Title", "Path", "Name"))

        column.addWidget(FluentSectionLabel("Panel"))
        self.title_edit = FluentLineEdit(card.config.title)
        self.title_edit.editingFinished.connect(self._commit_title)
        column.addWidget(
            FluentSettingRow("Title", self.title_edit, label_width=width, parent=page)
        )

        column.addWidget(FluentSectionLabel("Snapshot"))
        refresh_row = QtWidgets.QWidget(page)
        refresh_layout = QtWidgets.QHBoxLayout(refresh_row)
        refresh_layout.setContentsMargins(0, 0, 0, 0)
        refresh_layout.addWidget(FluentLabel("Frozen current panel front"), 1)
        self.refresh_button = FluentButton("Refresh", color=GREY)
        self.refresh_button.clicked.connect(self.refresh_snapshot)
        refresh_layout.addWidget(self.refresh_button)
        column.addWidget(refresh_row)
        self.canvas_holder = QtWidgets.QVBoxLayout()
        self.canvas_holder.setContentsMargins(0, 0, 0, 0)
        column.addLayout(self.canvas_holder)

        column.addWidget(FluentSectionLabel("Display"))
        self.controls_holder = QtWidgets.QVBoxLayout()
        self.controls_holder.setContentsMargins(0, 0, 0, 0)
        self.controls_holder.setSpacing(scaled_px(6, minimum=4))
        column.addLayout(self.controls_holder)

        column.addWidget(FluentSectionLabel("Fit"))
        self.fit_holder = QtWidgets.QVBoxLayout()
        self.fit_holder.setContentsMargins(0, 0, 0, 0)
        column.addLayout(self.fit_holder)

        # If this panel is bound to a Logic-node output, expose the producer's
        # same descriptor form as a second Qt projection.  The row draft remains
        # the sole parameter owner; an unbound/pure external signal gets no
        # invented producer form.
        self._producer_row = None
        self._producer_form: LogicNodeParameterPanel | None = None
        producer = console.producer_parameter_panel_for_signal(
            card.config.signal,
            parent=page,
        )
        if producer is not None:
            row, form = producer
            self._producer_row = row
            self._producer_form = form
            column.addWidget(FluentSectionLabel("Producer"))
            producer_holder = QtWidgets.QVBoxLayout()
            producer_holder.setContentsMargins(0, 0, 0, 0)
            producer_holder.addWidget(form)
            column.addLayout(producer_holder)

        column.addWidget(FluentSectionLabel("Save"))
        # Figure output is a visible project-root workspace path.  TaskConsole
        # exposes the same figures_root used by all other figure exporters.
        default_dir = console._figures_root / "task-console"
        self.save_dir_edit = FluentPathEdit(
            console._last_save_dir or str(default_dir),
            mode="dir",
            caption="Choose where to save",
            base_dir=str(default_dir),
        )
        column.addWidget(
            FluentSettingRow("Path", self.save_dir_edit, label_width=width, parent=page)
        )
        options = QtWidgets.QWidget(page)
        options_layout = QtWidgets.QHBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        self.save_autoname = FluentSwitch("auto-name")
        self.save_autoname.setChecked(True)
        self.save_format_combo = FluentComboBox()
        for extension in _IMAGE_FORMATS:
            self.save_format_combo.addItem(extension.upper(), extension)
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.clicked.connect(self.save)
        options_layout.addWidget(self.save_autoname)
        options_layout.addWidget(self.save_format_combo)
        options_layout.addWidget(self.save_button)
        options_layout.addStretch(1)
        column.addWidget(
            FluentSettingRow("Name", options, label_width=width, parent=page)
        )
        self.save_preview = FluentReadoutEdit("")
        column.addWidget(
            FluentSettingRow("File", self.save_preview, label_width=width, parent=page)
        )
        self.status = FluentLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        column.addWidget(self.status)
        column.addStretch(1)
        scroll.set_width_bounded_widget(page)

        self.save_dir_edit.changed.connect(self._update_save_preview)
        self.save_autoname.toggled.connect(self._update_save_preview)
        self.save_format_combo.currentIndexChanged.connect(self._update_save_preview)
        card.selectors_enabled_changed.connect(self._sync_selectors)
        self._update_save_preview()
        self.refresh_snapshot()

    @property
    def host(self) -> RasterPlotHost | None:
        return self._host

    @property
    def plot_widget(self):
        return self._plot_widget

    def refresh_snapshot(self) -> None:
        if self._closing:
            return
        snapshot = self.card.presented_snapshot
        config = self.card.current_plot_config()
        if snapshot is None or config is None:
            self.status.setText("The live panel has no complete front yet.")
            return
        spec, parameters = config
        self._retire_surface()
        host = RasterPlotHost.from_plot(
            snapshot,
            spec,
            size=self.card.config.size,
            parameters=parameters,
        )
        self._host = host
        future = host.dispatch(lambda: None)
        self._initial_future = future
        future.add_done_callback(lambda _completed: self._wake.request_owner_wake())
        self.status.setText("Preparing snapshot…")

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        self._poll_retiring_hosts()
        if self._closing:
            return
        future = self._initial_future
        if future is not None and future.done():
            self._initial_future = None
            try:
                operation = future.result()
                if not isinstance(operation, RasterOperation):
                    raise TypeError("snapshot host returned another operation type")
            except CancelledError:
                pass
            except BaseException as error:
                self.status.setText(f"Snapshot failed: {error}")
            else:
                self._install_surface(operation.front)
        for future in tuple(self._save_futures):
            if not future.done():
                continue
            path = self._save_futures.pop(future)
            try:
                future.result()
            except CancelledError:
                continue
            except BaseException as error:
                self.status.setText(f"Save failed: {error}")
            else:
                self.console._last_save_dir = str(path.parent)
                self.status.setText(f"Saved {path}")
        for future in tuple(self._front_futures):
            if not future.done():
                continue
            self._front_futures.discard(future)
            try:
                operation = future.result()
                if not isinstance(operation, RasterOperation):
                    raise TypeError("snapshot update returned another operation type")
            except CancelledError:
                continue
            except BaseException as error:
                self.status.setText(f"Display: {error}")
            else:
                self._present_front(operation.front)

    def _install_surface(self, front: RasterFront) -> None:
        host = self._host
        if host is None:
            return
        widget = Qt5PlotWidget(host, self, auto_present=False)
        widget.set_interaction_enabled(self.card.selectors_enabled)
        widget.present_front(front)
        self._plot_widget = widget
        self.canvas_holder.addWidget(widget)
        spec_panel = FluentPlotSpecPanel(
            host,
            self.card.presented_snapshot.block.schema,
            self,
        )
        parameter_panel = FluentPlotParameterPanel(host, self)
        fit_panel = FluentPlotFitPanel(host, live=False, parent=self)
        self.controls_holder.addWidget(spec_panel)
        self.controls_holder.addWidget(parameter_panel)
        self.fit_holder.addWidget(fit_panel)
        spec_panel.frontReady.connect(self._present_front)
        parameter_panel.frontReady.connect(self._present_front)
        fit_panel.frontReady.connect(self._present_front)
        spec_panel.specRejected.connect(
            lambda error: self.status.setText(f"Display: {error}")
        )
        parameter_panel.parameterRejected.connect(
            lambda _name, error: self.status.setText(f"Display: {error}")
        )
        fit_panel.fitRejected.connect(
            lambda _action, error: self.status.setText(f"Fit: {error}")
        )
        self.status.setText("Snapshot ready")

    @QtCore.pyqtSlot(object)
    def _present_front(self, front: object) -> None:
        widget = self._plot_widget
        if widget is not None and isinstance(front, RasterFront):
            try:
                widget.present_front(front)
            except (RuntimeError, TypeError, ValueError) as error:
                self.status.setText(str(error))

    def _sync_selectors(self, enabled: bool) -> None:
        widget = self._plot_widget
        if widget is not None:
            widget.set_interaction_enabled(bool(enabled))

    def _commit_title(self) -> None:
        title = self.title_edit.text().strip()
        if title == self.card.config.title:
            return
        self.card.config.title = title
        self.card._refresh_title()
        self.card.changed.emit()

    def _save_path(self) -> Path:
        base = Path(self.save_dir_edit.text()).expanduser()
        extension = str(self.save_format_combo.currentData() or "png")
        if self.save_autoname.isChecked():
            stem = self.card.config.title.strip() or self.card.config.kind.value
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return base / f"{stem}_{stamp}.{extension}"
        return base if base.suffix else base.with_suffix(f".{extension}")

    def _update_save_preview(self, *_args: object) -> None:
        self.save_preview.setText(str(self._save_path()))

    def save(self) -> None:
        host = self._host
        if host is None:
            self.status.setText("No snapshot to save")
            return
        path = self._save_path().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        future = host.save(path)
        self._save_futures[future] = path
        future.add_done_callback(lambda _completed: self._wake.request_owner_wake())

    def refresh_on_show(self) -> None:
        return

    def set_producer_editing_enabled(self, enabled: bool) -> None:
        """Backward-named narrow hook used by the console command gate."""

        self.set_mutation_enabled(enabled)

    def set_mutation_enabled(self, enabled: bool) -> None:
        """Disable snapshot/config mutations while a Task owns the console."""

        enabled = bool(enabled)
        for widget in (
            self.title_edit,
            self.refresh_button,
            self.save_dir_edit,
            self.save_autoname,
            self.save_format_combo,
            self.save_button,
        ):
            widget.setEnabled(enabled)
        if self._producer_form is not None:
            self._producer_form.set_mutation_enabled(enabled)
        for layout in (self.controls_holder, self.fit_holder):
            for index in range(layout.count()):
                child = layout.itemAt(index).widget()
                if child is not None:
                    child.setEnabled(enabled)

    def _retire_surface(self) -> None:
        # The host goes first: ``close_adapter`` below re-raises its first
        # cleanup failure by design, and a host still held when that happens
        # would never be asked to stop at all.
        host = self._host
        self._host = None
        if host is not None:
            self._retiring_hosts.retire(host)
            self._arm_retirement_wake()
        future = self._initial_future
        self._initial_future = None
        if future is not None:
            future.cancel()
        for future in tuple(self._front_futures):
            future.cancel()
        self._front_futures.clear()
        widget = self._plot_widget
        self._plot_widget = None
        if widget is not None:
            self.canvas_holder.removeWidget(widget)
            widget.hide()
            widget.close_adapter()
            widget.deleteLater()
        while self.controls_holder.count():
            item = self.controls_holder.takeAt(0)
            child = item.widget()
            if child is not None:
                child.hide()
                child.deleteLater()
        while self.fit_holder.count():
            item = self.fit_holder.takeAt(0)
            child = item.widget()
            if child is not None:
                child.hide()
                child.deleteLater()

    def _poll_retiring_hosts(self) -> bool:
        drained = self._retiring_hosts.poll()
        self._arm_retirement_wake()
        return drained

    def _arm_retirement_wake(self) -> None:
        # The retirement set is the level; this timer is only an edge that
        # re-reads it, so it is armed off the set and never off a flag.  While
        # closing the console drives the polling and no wake owner remains.
        if self._retiring_hosts and not self._closing:
            QtCore.QTimer.singleShot(25, self._wake.request_owner_wake)

    def teardown(self) -> bool:
        if not self._closing:
            self._closing = True
            self._wake.detach()
        # Every entrance retires whatever is still held; an entrance that died
        # on the widget cleanup edge must not leave an empty set reading as
        # "drained" while the host is alive and never asked to stop.
        self._retire_surface()
        closed = self._poll_retiring_hosts()
        if closed:
            self._finish_shutdown()
        elif not self._shutdown_timer.isActive():
            self._shutdown_timer.start()
        return closed

    @QtCore.pyqtSlot()
    def _poll_shutdown(self) -> None:
        if self._closed or not self._closing:
            self._shutdown_timer.stop()
            return
        if self._poll_retiring_hosts():
            self._finish_shutdown()

    def _finish_shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown_timer.stop()
        self.shutdownFinished.emit()


__all__ = ["PanelEditor"]
