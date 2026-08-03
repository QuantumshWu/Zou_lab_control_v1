"""Saved-Figure browser that embeds the sole DataFigure/zlc_plot surface."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from pathlib import Path
from weakref import ref

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    FigureInfoPane,
    FluentFrame,
    FluentLabel,
    WINDOW_SCREEN_FRACTION,
    ensure_qt_app,
    screen_fit_window_size,
    set_fluent_scale,
    window_pad,
)
from zlc_workbench.data_figure.archive_io import (
    LoadedFigureArchive,
    load_figure_archive,
)
from zlc_workbench.window_runtime import submit_compute

from .info_projection import FigureInfoProjection, project_figure_info


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def _archive_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return path
    if suffix in _IMAGE_SUFFIXES:
        return path.with_suffix(".npz")
    raise ValueError(
        f"saved figure must be .npz or an image with a same-stem .npz, got {path}"
    )


def _load_and_project(path: Path) -> tuple[LoadedFigureArchive, FigureInfoProjection]:
    archive = load_figure_archive(path)
    return archive, project_figure_info(archive)


class FigureViewer(QtWidgets.QWidget):
    """File/Info shell; rendering, interaction and Fit stay in DataFigure."""

    _loadFinished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        output_root: Path,
        scale: float | None = None,
        window_ratio: float = WINDOW_SCREEN_FRACTION,
        parent=None,
    ) -> None:
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__(parent)
        self._output_root = Path(output_root).expanduser()
        if not self._output_root.is_absolute():
            raise ValueError("FigureViewer output_root must be absolute")
        self._output_root = self._output_root.resolve()
        if not self._output_root.is_dir():
            raise FileNotFoundError(f"FigureViewer output_root does not exist: {self._output_root}")
        self.window_ratio = float(window_ratio)
        self._current_path: Path | None = None
        self.archive: LoadedFigureArchive | None = None
        self.figure_pane: QtWidgets.QWidget | None = None
        self._retiring_panes: set[QtWidgets.QWidget] = set()
        self._candidate: tuple[
            int,
            LoadedFigureArchive,
            FigureInfoProjection,
            QtWidgets.QWidget,
        ] | None = None
        self._load_revision = 0
        self._active_load: tuple[int, Path, Future] | None = None
        self._pending_load: tuple[int, Path] | None = None
        self._closing = False
        self._closed = False
        self._retirement_timer = QtCore.QTimer(self)
        self._retirement_timer.setInterval(25)
        self._retirement_timer.timeout.connect(self._reap_retiring_panes)

        self.setObjectName("figureViewer")
        self.setStyleSheet("background: transparent;")
        self.setFixedSize(screen_fit_window_size(self.window_ratio))
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, window_pad(1), 0, window_pad(1))
        root.setSpacing(window_pad(0.5))
        self.info_pane = FigureInfoPane(
            label_names=("point columns", "grid topology"),
            parent=self,
        )
        self.info_pane.pathCommitted.connect(self._commit_path)
        root.addWidget(self.info_pane, 0)

        holder = FluentFrame(self, bordered=False)
        holder.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._pane_layout = QtWidgets.QVBoxLayout(holder)
        self._pane_layout.setContentsMargins(0, 0, window_pad(1), 0)
        self._pane_layout.setSpacing(0)
        self._placeholder = FluentLabel("Open a saved figure to begin", holder)
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._pane_layout.addWidget(self._placeholder, 1)
        root.addWidget(holder, 1)

        self._loadFinished.connect(self._accept_load_completion, QtCore.Qt.QueuedConnection)
        if path is not None:
            self.open_path(path)

    @property
    def worker_idle(self) -> bool:
        return (
            self._active_load is None
            and self._pending_load is None
            and self._candidate is None
            and not self._retiring_panes
        )

    def open_path(self, path: str | Path) -> None:
        text = str(path).strip()
        if text:
            self.info_pane.path_edit.setText(text)
            self._commit_path(text)

    @QtCore.pyqtSlot(str)
    def _commit_path(self, text: str) -> None:
        if self._closing:
            return
        try:
            path = _archive_path(Path(str(text).strip())).expanduser().resolve()
        except BaseException as error:
            self.info_pane.status.show_message(
                f"{type(error).__name__}: {error}", severity="error"
            )
            return
        if path == self._current_path and self._active_load is None and self._pending_load is None:
            return
        if self._candidate is not None:
            self._retire_pane(self._candidate[3])
            self._candidate = None
        self._load_revision += 1
        self._pending_load = (self._load_revision, path)
        self.info_pane.status.show_message(f"Loading {path}", severity="task")
        self._start_pending_load()

    def _start_pending_load(self) -> None:
        if self._closing or self._active_load is not None or self._pending_load is None:
            return
        revision, path = self._pending_load
        self._pending_load = None
        future = submit_compute(_load_and_project, path)
        self._active_load = (revision, path, future)
        viewer_ref = ref(self)

        def completed(done: Future, rev: int = revision, selected: Path = path) -> None:
            viewer = viewer_ref()
            if viewer is None:
                return
            try:
                viewer._loadFinished.emit((rev, selected, done))
            except RuntimeError:
                return

        future.add_done_callback(completed)

    @QtCore.pyqtSlot(object)
    def _accept_load_completion(self, completion: object) -> None:
        revision, path, future = completion
        active = self._active_load
        if active is not None and active[2] is future:
            self._active_load = None
        try:
            result = future.result()
            if self._closing or revision != self._load_revision:
                return
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("Figure archive loader returned another result")
            archive, info = result
            if not isinstance(archive, LoadedFigureArchive):
                raise TypeError("Figure archive loader returned another archive type")
            self._accept_archive(revision, archive, info)
        except CancelledError:
            if not self._closing:
                self.info_pane.status.show_message(f"Load cancelled: {path}", severity="warning")
        except BaseException as error:
            if not self._closing and revision == self._load_revision:
                self.info_pane.status.show_message(
                    f"{type(error).__name__}: {error}", severity="error"
                )
        finally:
            self._start_pending_load()
            self._finish_close_if_ready()

    def _accept_archive(
        self,
        revision: int,
        archive: LoadedFigureArchive,
        info: FigureInfoProjection,
    ) -> None:
        from zlc_workbench.data_figure.app import create_data_figure_pane

        value = archive.archive
        candidate = create_data_figure_pane(
            value.snapshot,
            value.spec,
            output_root=self._output_root,
            size=value.size,
            parameters=value.parameters,
            archive_path=archive.path,
            metadata=value.metadata,
            embedded=True,
            parent=self,
        )
        candidate.hide()
        self._candidate = (revision, archive, info, candidate)
        candidate.initialReady.connect(
            lambda pane=candidate: self._commit_candidate(pane),
            QtCore.Qt.QueuedConnection,
        )
        candidate.initialFailed.connect(
            lambda detail, pane=candidate: self._reject_candidate(pane, detail),
            QtCore.Qt.QueuedConnection,
        )
        candidate.archiveSaved.connect(
            lambda loaded, pane=candidate: self._accept_archive_save(pane, loaded),
            QtCore.Qt.QueuedConnection,
        )
        self.info_pane.status.show_message(f"Rendering {archive.path}", severity="task")

    def _commit_candidate(self, candidate: QtWidgets.QWidget) -> None:
        pending = self._candidate
        if pending is None or pending[3] is not candidate:
            return
        revision, archive, info, _pane = pending
        self._candidate = None
        if self._closing or revision != self._load_revision:
            self._retire_pane(candidate)
            return
        previous = self.figure_pane
        if self._placeholder is not None:
            self._pane_layout.removeWidget(self._placeholder)
            self._placeholder.hide()
            self._placeholder.deleteLater()
            self._placeholder = None
        self._pane_layout.addWidget(candidate, 1)
        candidate.show()
        self.figure_pane = candidate
        if previous is not None:
            self._retire_pane(previous)
        self.archive = archive
        self._current_path = archive.path
        self.info_pane.path_edit.setText(str(archive.path))
        self._install_info(info)
        self.info_pane.status.show_message(f"Loaded {archive.path}")

    def _reject_candidate(self, candidate: QtWidgets.QWidget, detail: str) -> None:
        if self._candidate is None or self._candidate[3] is not candidate:
            return
        self._candidate = None
        self._retire_pane(candidate)
        if not self._closing:
            self.info_pane.status.show_message(
                f"Figure render failed: {detail}", severity="error"
            )

    def _accept_archive_save(self, pane: QtWidgets.QWidget, loaded: object) -> None:
        if self._closing or self.figure_pane is not pane:
            return
        if not isinstance(loaded, LoadedFigureArchive):
            self.info_pane.status.show_message("Figure save returned another archive type", severity="error")
            return
        self.archive = loaded
        self._current_path = loaded.path
        self.info_pane.path_edit.setText(str(loaded.path))
        self._install_info(project_figure_info(loaded))
        self.info_pane.status.show_message(f"Saved {loaded.path}")

    def _install_info(self, info: FigureInfoProjection) -> None:
        if not isinstance(info, tuple) or len(info) != 5:
            raise TypeError("Figure info projection has another type")
        plot_rows, measurement_rows, device_rows, flow_graph, raw_text = info
        self.info_pane.replace_info(
            plot_rows=plot_rows,
            measurement_rows=measurement_rows,
            device_rows=device_rows,
            flow_graph=flow_graph,
            raw_text=raw_text,
        )

    def _retire_pane(self, pane: QtWidgets.QWidget) -> bool:
        self._pane_layout.removeWidget(pane)
        pane.hide()
        teardown = getattr(pane, "teardown", None)
        stopped = True if not callable(teardown) else bool(teardown())
        if stopped:
            self._retiring_panes.discard(pane)
            pane.deleteLater()
        else:
            self._retiring_panes.add(pane)
            if not self._retirement_timer.isActive():
                self._retirement_timer.start()
        return stopped

    @QtCore.pyqtSlot()
    def _reap_retiring_panes(self) -> None:
        for pane in tuple(self._retiring_panes):
            if not bool(getattr(pane, "closed", False)):
                continue
            self._retiring_panes.discard(pane)
            pane.deleteLater()
        if not self._retiring_panes:
            self._retirement_timer.stop()
        self._finish_close_if_ready()

    def teardown(self) -> bool:
        if self._closed:
            return True
        if not self._closing:
            self._closing = True
            self._pending_load = None
            if self._active_load is not None:
                self._active_load[2].cancel()
            if self._candidate is not None:
                self._retire_pane(self._candidate[3])
                self._candidate = None
            if self.figure_pane is not None:
                self._retire_pane(self.figure_pane)
                self.figure_pane = None
        self._reap_retiring_panes()
        return self._closed

    def _finish_close_if_ready(self) -> None:
        if (
            not self._closing
            or self._closed
            or self._active_load is not None
            or self._retiring_panes
        ):
            return
        self._retirement_timer.stop()
        self.archive = None
        self._closed = True
        window = getattr(self, "_zlc_window", None)
        if window is not None:
            QtCore.QTimer.singleShot(0, window.close)


__all__ = ["FigureViewer"]
