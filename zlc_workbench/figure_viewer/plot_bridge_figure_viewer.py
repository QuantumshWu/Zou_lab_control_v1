"""Formal saved-figure viewer shell.

The left pane preserves the established File/Plot/Measurement/Device/Flow/Raw
layout.  The right pane embeds the same DataFigure interaction owner used by
typed notebook figures; this module owns neither a second renderer nor a fake
live-data graph.

A file open is the one whole-generation replacement boundary in this window.
It is decoded and validated on the shared raster worker, then a complete
candidate pane is built on the Qt owner before the previous valid pane and Info
projection are replaced.  Failed candidates therefore leave the visible figure
untouched.  Ordinary selector, Setting/Edit, viewport, and label changes remain
inside the stable DataFigure pane and never reload the archive or rebuild this
window.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import CancelledError, Future
from enum import Enum
from pathlib import Path
from pprint import pformat

from PyQt5 import QtCore, QtWidgets

from zlc_storage.paths import display_path

from zlc_frontend.qt_widgets import (
    CARD_PAD,
    FlowGraphView,
    FluentCodeEdit,
    FluentFrame,
    FluentLabel,
    FluentPathEdit,
    FluentReadoutMultiline,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusStrip,
    FluentTabWidget,
    QtOwnerWake,
    WINDOW_SCREEN_FRACTION,
    ensure_qt_app,
    launch_fluent_window,
    scaled_px,
    screen_fit_window_size,
    set_fluent_scale,
    setting_label_width,
    window_pad,
)
from Zou_lab_control.workbench._window_runtime import RASTER_WORK_EXECUTOR


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_ARCHIVE_SUFFIX = ".npz"


def _archive_path(path: Path) -> Path:
    """Resolve a committed user choice to the exact data archive path."""

    suffix = path.suffix.lower()
    if suffix == _ARCHIVE_SUFFIX:
        return path
    if suffix in _IMAGE_SUFFIXES:
        return path.with_suffix(_ARCHIVE_SUFFIX)
    raise ValueError(
        f"saved figure must be .npz or an image with a same-stem .npz, got {path}"
    )


def _load_archive(path: Path):
    """Lazy worker entry so importing the window does no archive work."""

    from zlc_frontend.figure_archive import load_figure_archive

    return load_figure_archive(path)


def _enum_text(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _axis_text(axis) -> str:
    """Describe one declared axis without inferring anything from array rank."""

    unit = "" if axis.unit is None else f" [{axis.unit}]"
    return (
        f"{axis.name} ({axis.axis_id}; role={axis.role}; "
        f"size={axis.size}{unit})"
    )


def _view_text(view) -> str:
    bindings = ", ".join(
        f"{binding.axis_id}={binding.role.value}"
        for binding in view.axis_bindings
    )
    selections = len(view.display_selections)
    suffix = "" if selections == 0 else f"; selections={selections}"
    return f"intent={view.intent.value}; {bindings or 'no axis bindings'}{suffix}"


def _dataset_projection(figure) -> tuple[tuple[str, object], ...]:
    """Project typed source schemas into human-readable, array-free rows."""

    rows: list[tuple[str, object]] = []
    document = figure.document
    datasets = figure.datasets
    for descriptor in document.datasets:
        snapshot = datasets.resolve(descriptor.dataset_id)
        schema = snapshot.block.schema
        cell = schema.cell_schema
        prefix = descriptor.label
        rows.extend(
            (
                (f"{prefix} id", descriptor.dataset_id),
                (f"{prefix} revision", snapshot.ref),
                (f"{prefix} shape", schema.physical_shape),
                (f"{prefix} repeat", _axis_text(schema.repeat_axis)),
                (
                    f"{prefix} points",
                    ", ".join(_axis_text(axis) for axis in schema.point_axes)
                    or "(none)",
                ),
                (
                    f"{prefix} data",
                    ", ".join(_axis_text(axis) for axis in cell.data_axes)
                    or "scalar",
                ),
                (f"{prefix} dtype", cell.dtype),
                (f"{prefix} unit", cell.value_unit or "(none)"),
                (
                    f"{prefix} validity",
                    cell.validity_contract.mode.value,
                ),
            )
        )
    return tuple(rows)


def _flow_graph(metadata: Mapping[str, object]) -> object:
    direct = metadata.get("flow_graph")
    if direct is not None:
        return direct
    provenance = metadata.get("provenance")
    if isinstance(provenance, Mapping):
        return provenance.get("flow_graph")
    return None


def _raw_projection(archive) -> str:
    """Show the complete typed descriptive record, excluding source array bytes."""

    figure = archive.figure
    datasets = []
    for descriptor in figure.document.datasets:
        snapshot = figure.datasets.resolve(descriptor.dataset_id)
        datasets.append(
            {
                "descriptor": descriptor,
                "reference": snapshot.ref,
                "schema": snapshot.block.schema,
                "validity": snapshot.block.validity,
            }
        )
    return pformat(
        {
            "path": str(archive.path),
            "payload_digest": archive.payload_digest,
            "document": figure.document,
            "datasets": tuple(datasets),
            "fit_results": dict(figure.fit_results),
            "display": archive.display,
            "metadata": dict(archive.metadata),
        },
        sort_dicts=False,
        width=100,
    )


class FigureViewer(QtWidgets.QWidget):
    """Saved-figure browser with one current DataFigure pane."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        scale: float | None = None,
        window_ratio: float = WINDOW_SCREEN_FRACTION,
        parent=None,
    ) -> None:
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__(parent)
        self.window_ratio = float(window_ratio)
        self._current_path: Path | None = None
        self.archive = None
        self.figure_pane: QtWidgets.QWidget | None = None
        self._retiring_panes: list[QtWidgets.QWidget] = []
        self._load_revision = 0
        self._active_load: tuple[int, Path, Future] | None = None
        self._pending_load: tuple[int, Path] | None = None
        self._closing = False
        self._closed = False

        self._label_w = setting_label_width(
            ("payload_digest", "schema_fingerprint", "coordinate_frame")
        )
        self._info_col_w = self._label_w + scaled_px(320, minimum=240)
        self.setFixedSize(screen_fit_window_size(self.window_ratio))

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, window_pad(1), 0, window_pad(1))
        root.setSpacing(window_pad(0.5))
        root.addWidget(self._build_info_column(), 0)

        holder = QtWidgets.QWidget(self)
        holder.setStyleSheet("background: transparent;")
        holder.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self._pane_holder = QtWidgets.QVBoxLayout(holder)
        self._pane_holder.setContentsMargins(0, 0, window_pad(1), 0)
        self._pane_holder.setSpacing(0)
        self._placeholder = self._build_placeholder(holder)
        self._pane_holder.addWidget(self._placeholder)
        root.addWidget(holder, 1)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._retirement_timer = QtCore.QTimer(self)
        self._retirement_timer.setInterval(50)
        self._retirement_timer.timeout.connect(self._reap_retiring_panes)

        if path is not None:
            self.open_path(path)

    # ---------------------------------------------------------------- layout
    def _build_placeholder(self, parent) -> QtWidgets.QWidget:
        frame = FluentFrame(parent=parent)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.addStretch(1)
        label = FluentLabel("Open a saved figure to begin", frame)
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        layout.addStretch(1)
        return frame

    def _build_info_column(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget(self)
        col.setStyleSheet("background: transparent;")
        col.setFixedWidth(self._info_col_w)
        layout = QtWidgets.QVBoxLayout(col)
        layout.setContentsMargins(window_pad(1), 0, 0, 0)
        layout.setSpacing(window_pad(0.5))

        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(
            scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6)
        )
        header.setSpacing(scaled_px(8, minimum=5))
        header.addWidget(FluentSectionLabel("File"))
        self.path_edit = FluentPathEdit(
            "",
            mode="file",
            caption="Open a saved figure (image or .npz)",
            file_filter="Saved figures (*.png *.jpg *.jpeg *.npz);;All files (*)",
        )
        self.path_edit.setToolTip(
            "Choose a current saved Figure .npz, or its same-stem PNG/JPEG image."
        )
        # File I/O is a committed action, never a textChanged side effect.
        self.path_edit.selected.connect(self._commit_path)
        self.path_edit.edit.editingFinished.connect(
            lambda: self._commit_path(self.path_edit.text())
        )
        header.addWidget(self.path_edit, 1)
        layout.addWidget(header_frame)

        self.info_tabs = FluentTabWidget()
        self.info_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.plot_layout = self._add_rows_tab("Plot")
        self.meas_layout = self._add_rows_tab("Measurement")
        self.info_layout = self._add_rows_tab("Device")
        self.flow_view = self._add_flow_tab("Flow")
        self.raw_info = self._add_raw_tab("Raw")
        layout.addWidget(self.info_tabs, 1)

        self.status = FluentStatusStrip()
        self.status.show_message("Open a current saved Figure (.npz).")
        layout.addWidget(self.status)
        return col

    def _add_rows_tab(self, title: str) -> QtWidgets.QVBoxLayout:
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(CARD_PAD, minimum=6)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(scaled_px(3, minimum=2))
        layout.setAlignment(QtCore.Qt.AlignTop)
        scroll.setWidget(body)
        self.info_tabs.add_permanent_tab(scroll, title)
        return layout

    def _add_flow_tab(self, title: str) -> FlowGraphView:
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        view = FlowGraphView()
        scroll.setWidget(view)
        self.info_tabs.add_permanent_tab(scroll, title)
        return view

    def _add_raw_tab(self, title: str) -> FluentCodeEdit:
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(CARD_PAD, minimum=6)
        layout.setContentsMargins(margin, margin, margin, margin)
        raw = FluentCodeEdit("", read_only=True)
        raw.setToolTip(
            "The complete typed archive description; source array bytes are represented by schema."
        )
        layout.addWidget(raw)
        self.info_tabs.add_permanent_tab(body, title)
        return raw

    # -------------------------------------------------------------- public API
    def window(self):
        return getattr(self, "_zlc_window", None)

    @property
    def worker_idle(self) -> bool:
        return self._active_load is None and self._pending_load is None

    def open_path(self, path: str | Path) -> None:
        """Commit one programmatic path exactly like Browse/editingFinished."""

        text = str(path).strip()
        if not text:
            return
        self.path_edit.setText(text)
        self._commit_path(text)

    @QtCore.pyqtSlot(str)
    def _commit_path(self, text: str) -> None:
        if self._closing:
            return
        raw = str(text).strip()
        if not raw:
            return
        try:
            path = _archive_path(Path(raw))
        except BaseException as error:
            self.status.show_message(
                f"{type(error).__name__}: {error}", severity="error"
            )
            return
        if (
            path == self._current_path
            and self._active_load is None
            and self._pending_load is None
        ):
            return
        if self._active_load is not None and path == self._active_load[1]:
            if self._pending_load is None:
                return
        if self._pending_load is not None and path == self._pending_load[1]:
            return
        self._load_revision += 1
        self._pending_load = (self._load_revision, path)
        self.status.show_message(
            f"Loading {display_path(str(path))}", severity="task"
        )
        self._start_pending_load()

    def _start_pending_load(self) -> None:
        if self._active_load is not None or self._pending_load is None or self._closing:
            return
        revision, path = self._pending_load
        self._pending_load = None
        try:
            future = RASTER_WORK_EXECUTOR.submit(_load_archive, path)
        except BaseException as error:
            self.status.show_message(
                f"{type(error).__name__}: {error}", severity="error"
            )
            return
        self._active_load = (revision, path, future)
        future.add_done_callback(lambda _done: self._wake.request_owner_wake())

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        active = self._active_load
        if active is not None and active[2].done():
            revision, path, future = active
            self._active_load = None
            if not self._closing and revision == self._load_revision:
                try:
                    archive = future.result()
                    self._accept_archive(archive)
                except CancelledError:
                    self.status.show_message(
                        f"Load cancelled: {display_path(str(path))}",
                        severity="warning",
                    )
                except BaseException as error:
                    # Full error text is retained by FluentStatusStrip's tooltip.
                    self.status.show_message(
                        f"{type(error).__name__}: {error}", severity="error"
                    )
            else:
                # Always observe a completed future, including stale generations.
                try:
                    future.result()
                except BaseException:
                    pass
            self._start_pending_load()
        self._finish_close_if_ready()

    def _accept_archive(self, archive) -> None:
        """Construct a complete candidate, then atomically replace the visible generation."""

        from Zou_lab_control.workbench import create_data_figure_pane

        metadata = archive.metadata
        if not isinstance(metadata, Mapping):
            raise TypeError("FigureArchive metadata must be a mapping")
        info = self._info_projection(archive)
        candidate = create_data_figure_pane(
            archive.figure,
            initial_display=archive.display,
            initial_fit_result_identity=(
                archive.payload_digest
                if archive.figure.has_fit_overlays
                else None
            ),
            embedded=True,
        )
        if not isinstance(candidate, QtWidgets.QWidget):
            raise TypeError("create_data_figure_pane must return a QWidget")

        previous = self.figure_pane
        if self._placeholder is not None:
            self._pane_holder.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None
        self._pane_holder.addWidget(candidate)
        self.figure_pane = candidate
        if previous is not None:
            self._retire_pane(previous)

        self.archive = archive
        self._current_path = Path(archive.path)
        self.path_edit.setText(str(self._current_path))
        self._apply_info_projection(*info)
        self.status.show_message(
            f"Loaded {display_path(str(self._current_path))}"
        )

    # --------------------------------------------------------------- Info data
    def _info_projection(self, archive):
        figure = archive.figure
        document = figure.document
        plot_rows: list[tuple[str, object]] = [
            ("document", document.document_id),
            ("revision", document.revision),
            ("payload_digest", archive.payload_digest),
        ]
        for layer in document.layers:
            descriptor = document.descriptor(layer.dataset_id)
            plot_rows.append(
                (
                    f"layer {layer.layer_id}",
                    f"{descriptor.label} ({descriptor.dataset_id}); {_view_text(layer.view)}",
                )
            )
        if document.selections:
            plot_rows.append(("selections", len(document.selections)))
        if archive.display is not None:
            plot_rows.append(("display", archive.display))

        measurement_rows = list(_dataset_projection(figure))
        measurement_rows.append(("path", display_path(str(archive.path))))

        device_rows: list[tuple[str, object]] = []
        for key, value in archive.metadata.items():
            if key not in {"flow_graph"}:
                device_rows.append((str(key), value))
        if not device_rows:
            device_rows.append(("metadata", "(none recorded)"))
        return (
            tuple(plot_rows),
            tuple(measurement_rows),
            tuple(device_rows),
            _flow_graph(archive.metadata),
            _raw_projection(archive),
        )

    def _apply_info_projection(
        self,
        plot_rows,
        measurement_rows,
        device_rows,
        flow_graph,
        raw_text,
    ) -> None:
        for layout in (self.plot_layout, self.meas_layout, self.info_layout):
            self._clear_layout(layout)
        self._fill_rows(self.plot_layout, plot_rows)
        self._fill_rows(self.meas_layout, measurement_rows)
        self._fill_rows(self.info_layout, device_rows)
        self.flow_view.set_graph(flow_graph)
        self.raw_info.setPlainText(raw_text)

    @staticmethod
    def _clear_layout(layout: QtWidgets.QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fill_rows(self, layout, rows) -> None:
        for key, value in rows:
            field = FluentReadoutMultiline(self._readout_text(value))
            field.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Fixed,
            )
            layout.addWidget(
                FluentSettingRow(str(key), field, label_width=self._label_w)
            )

    @staticmethod
    def _readout_text(value: object) -> str:
        value = _enum_text(value)
        if isinstance(value, Mapping):
            return pformat(dict(value), sort_dicts=False, width=80)
        if isinstance(value, (tuple, list)):
            return ", ".join(str(_enum_text(item)) for item in value)
        return str(value)

    # ------------------------------------------------------------ pane lifetime
    def _retire_pane(self, pane: QtWidgets.QWidget) -> None:
        self._pane_holder.removeWidget(pane)
        pane.hide()
        shutdown = getattr(pane, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._retiring_panes.append(pane)
        if not self._retirement_timer.isActive():
            self._retirement_timer.start()

    def _reap_retiring_panes(self) -> None:
        pending = []
        for pane in self._retiring_panes:
            if bool(getattr(pane, "closed", True)):
                pane.deleteLater()
            else:
                pending.append(pane)
        self._retiring_panes = pending
        if not pending:
            self._retirement_timer.stop()
        self._finish_close_if_ready()

    def teardown(self) -> bool:
        if self._closed:
            return True
        if not self._closing:
            self._closing = True
            self._pending_load = None
            active = self._active_load
            if active is not None:
                active[2].cancel()
            pane = self.figure_pane
            self.figure_pane = None
            if pane is not None:
                self._retire_pane(pane)
        self._finish_close_if_ready()
        return self._closed

    def _finish_close_if_ready(self) -> None:
        if (
            not self._closing
            or self._active_load is not None
            or self._retiring_panes
            or self._closed
        ):
            return
        self._retirement_timer.stop()
        self._wake.detach()
        self.archive = None
        self._closed = True
        window = self.window()
        if window is not None:
            QtCore.QTimer.singleShot(0, window.close)

    # ---------------------------------------------------------------- sizing
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        try:
            return screen_fit_window_size(self.window_ratio)
        except Exception:
            return super().sizeHint()


def show_figure_viewer(
    path: str | Path | None = None,
    *,
    scale: float | None = None,
    window_ratio: float = WINDOW_SCREEN_FRACTION,
    hide_on_close: bool = False,
) -> FigureViewer:
    """Open the session-independent saved-figure viewer."""

    ensure_qt_app()
    viewer = FigureViewer(path, scale=scale, window_ratio=window_ratio)

    def _wire(window):
        if not hide_on_close:
            window.set_close_guard(viewer.teardown)

    window = launch_fluent_window(
        viewer,
        title="FigureViewer@Zou lab",
        hide_on_close=hide_on_close,
        wire=_wire,
    )
    viewer._zlc_window = window
    return viewer


__all__ = ["FigureViewer", "show_figure_viewer"]
