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
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from zlc_storage.paths import display_path

from zlc_frontend.qt_widgets import (
    FigureInfoPane,
    FluentFrame,
    FluentLabel,
    QtOwnerWake,
    WINDOW_SCREEN_FRACTION,
    ensure_qt_app,
    screen_fit_window_size,
    set_fluent_scale,
    window_pad,
)
from zlc_workbench.window_runtime import RASTER_WORK_EXECUTOR

from .info_projection import project_figure_info


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
        self._candidate_load: tuple[int, object, tuple, QtWidgets.QWidget] | None = None
        self._retiring_panes: list[QtWidgets.QWidget] = []
        self._load_revision = 0
        self._active_load: tuple[int, Path, Future] | None = None
        self._pending_load: tuple[int, Path] | None = None
        self._closing = False
        self._closed = False

        self.setStyleSheet("background: transparent;")
        self.setFixedSize(screen_fit_window_size(self.window_ratio))

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, window_pad(1), 0, window_pad(1))
        root.setSpacing(window_pad(0.5))
        self.info_pane = FigureInfoPane(
            label_names=(
                "payload_digest",
                "schema_fingerprint",
                "coordinate_frame",
            ),
            parent=self,
        )
        self.info_pane.pathCommitted.connect(self._commit_path)
        root.addWidget(self.info_pane, 0)

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

    # -------------------------------------------------------------- public API
    def window(self):
        return getattr(self, "_zlc_window", None)

    @property
    def worker_idle(self) -> bool:
        return (
            self._active_load is None
            and self._pending_load is None
            and self._candidate_load is None
        )

    def open_path(self, path: str | Path) -> None:
        """Commit one programmatic path exactly like Browse/editingFinished."""

        text = str(path).strip()
        if not text:
            return
        self.info_pane.path_edit.setText(text)
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
            self.info_pane.status.show_message(
                f"{type(error).__name__}: {error}", severity="error"
            )
            return
        candidate = self._candidate_load
        if candidate is not None:
            candidate_path = Path(candidate[1].path)
            if path == candidate_path:
                return
            self._candidate_load = None
            self._retire_pane(candidate[3])
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
        self.info_pane.status.show_message(
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
            self.info_pane.status.show_message(
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
                    self._accept_archive(revision, archive)
                except CancelledError:
                    self.info_pane.status.show_message(
                        f"Load cancelled: {display_path(str(path))}",
                        severity="warning",
                    )
                except BaseException as error:
                    # Full error text is retained by FluentStatusStrip's tooltip.
                    self.info_pane.status.show_message(
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

    def _accept_archive(self, revision: int, archive) -> None:
        """Build a hidden candidate; its admitted first front commits the generation."""

        from zlc_workbench.data_figure.app import create_data_figure_pane

        metadata = archive.metadata
        if not isinstance(metadata, Mapping):
            raise TypeError("FigureArchive metadata must be a mapping")
        info = project_figure_info(archive)
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
        ready = getattr(candidate, "initialReady", None)
        failed = getattr(candidate, "initialFailed", None)
        if ready is None or failed is None:
            candidate.shutdown()
            raise TypeError(
                "DataFigure pane must expose initialReady and initialFailed"
            )

        self._candidate_load = (revision, archive, info, candidate)
        ready.connect(
            lambda pane=candidate: self._commit_candidate(pane),
            QtCore.Qt.QueuedConnection,
        )
        failed.connect(
            lambda detail, pane=candidate: self._reject_candidate(pane, detail),
            QtCore.Qt.QueuedConnection,
        )
        self.info_pane.status.show_message(
            f"Rendering {display_path(str(archive.path))}",
            severity="task",
        )

    def _commit_candidate(self, candidate: QtWidgets.QWidget) -> None:
        pending = self._candidate_load
        if pending is None or pending[3] is not candidate:
            return
        revision, archive, info, _pane = pending
        self._candidate_load = None
        if self._closing or revision != self._load_revision:
            self._retire_pane(candidate)
            return

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
        self.info_pane.path_edit.setText(str(self._current_path))
        (
            plot_rows,
            measurement_rows,
            device_rows,
            flow_graph,
            raw_text,
        ) = info
        self.info_pane.replace_info(
            plot_rows=plot_rows,
            measurement_rows=measurement_rows,
            device_rows=device_rows,
            flow_graph=flow_graph,
            raw_text=raw_text,
        )
        self.info_pane.status.show_message(
            f"Loaded {display_path(str(self._current_path))}"
        )

    def _reject_candidate(
        self,
        candidate: QtWidgets.QWidget,
        detail: str,
    ) -> None:
        pending = self._candidate_load
        if pending is None or pending[3] is not candidate:
            return
        revision, _archive, _info, _pane = pending
        self._candidate_load = None
        self._retire_pane(candidate)
        if not self._closing and revision == self._load_revision:
            self.info_pane.status.show_message(
                f"Figure render failed: {detail}",
                severity="error",
            )

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
            candidate = self._candidate_load
            self._candidate_load = None
            if candidate is not None:
                self._retire_pane(candidate[3])
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
__all__ = ["FigureViewer"]
