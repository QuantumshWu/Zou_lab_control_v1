"""Read-only Workbench navigation for exact committed occupancy/camera cells."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading

from PyQt5 import QtCore

from zlc_data import Selection
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.occupancy_render import OccupancyCellNavigation
from zlc_frontend.qt_widgets import (
    AxisLayoutNavigator,
    signals_blocked,
)
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef

from ._frozen_raster import FrozenRasterWindow, _error_summary, _open_frozen_window


_DEFAULT_OCCUPANCY_CELL_GUI_MEMORY_LIMIT_BYTES = 512 << 20


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _load_navigation(
    loader,
    reference: OccupancyArtifactRef,
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> OccupancyCellNavigation:
    _require_not_cancelled(cancelled)
    navigation = loader(reference, memory_limit_bytes=memory_limit_bytes)
    if not isinstance(navigation, OccupancyCellNavigation):
        raise TypeError("navigation loader must return OccupancyCellNavigation")
    if navigation.artifact_identity != reference.target_ref:
        raise ValueError("occupancy navigation names a different artifact")
    if navigation.retained_upper_bound_bytes >= memory_limit_bytes:
        raise MemoryError("occupancy navigation leaves no exact-cell display budget")
    _require_not_cancelled(cancelled)
    return navigation


def _render_occupancy_cell(
    loader,
    reference: OccupancyArtifactRef,
    selection: Selection,
    navigation: OccupancyCellNavigation,
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    _require_not_cancelled(cancelled)
    view, retained_upper_bound = loader(
        reference,
        selection,
        memory_limit_bytes=memory_limit_bytes,
        expected_navigation=navigation,
    )
    _require_not_cancelled(cancelled)
    from zlc_frontend.occupancy_render import render_occupancy_cell

    result = render_occupancy_cell(
        view,
        memory_limit_bytes=memory_limit_bytes,
        source_retained_upper_bound_bytes=retained_upper_bound,
        checkpoint=lambda: _require_not_cancelled(cancelled),
    )
    if not isinstance(result, EncodedRasterDocument):
        raise TypeError("occupancy renderer must return EncodedRasterDocument")
    if len(result.pages) != 1:
        raise ValueError("exact occupancy cell renderer must return one page")
    if result.source_front_peak_nbytes > memory_limit_bytes:
        raise MemoryError(
            "encoded occupancy front requires "
            f"{result.source_front_peak_nbytes} bytes; limit is {memory_limit_bytes}"
        )
    _require_not_cancelled(cancelled)
    return result


def _cell_job(
    loader,
    reference: OccupancyArtifactRef,
    selection: Selection,
    navigation: OccupancyCellNavigation,
    memory_limit_bytes: int,
    request_revision: int,
    cancelled: threading.Event,
):
    return (
        request_revision,
        navigation.identity,
        selection,
        _render_occupancy_cell(
            loader,
            reference,
            selection,
            navigation,
            memory_limit_bytes,
            cancelled,
        ),
    )


class OccupancyCellWindow(FrozenRasterWindow):
    """Navigate exact repeat/PointLayout cells without becoming a live viewer."""

    def __init__(
        self,
        navigation_loader,
        cell_loader,
        reference: OccupancyArtifactRef,
        *,
        selection: Selection | None,
        memory_limit_bytes: int,
    ) -> None:
        if not callable(navigation_loader):
            raise TypeError("navigation_loader must be callable")
        if not callable(cell_loader):
            raise TypeError("cell_loader must be callable")
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        super().__init__(
            None,
            window_title="Occupancy Cell",
            mode_text="EXACT OCCUPANCY CELL · SAME-SHOT FRAME · DISPLAY ONLY",
            loading_summary=f"Resolving {reference.target_ref}…",
            object_prefix="occupancyCell",
            subject="OCCUPANCY CELL",
            memory_limit_bytes=memory_limit_bytes,
        )
        self._status.setText("READING FINAL CELL INDEX")
        self._navigation_loader = navigation_loader
        self._cell_loader = cell_loader
        self._reference = reference
        self._initial_selection = selection
        self._cell_memory_limit_bytes = 0
        self._navigation: OccupancyCellNavigation | None = None
        self._navigator: AxisLayoutNavigator | None = None
        self._active_kind: str | None = None
        self._active_revision = 0
        self._pending: tuple[int, Selection] | None = None
        self._pending_start_scheduled = False
        self._request_revision = 0
        self._requested_selection: Selection | None = None
        self._presented_selection: Selection | None = None
        self._start_navigation()

    @property
    def worker_idle(self) -> bool:
        return (
            self._future is None
            and self._pending is None
            and not self._pending_start_scheduled
        )

    @property
    def navigation_ready(self) -> bool:
        return self._navigation is not None

    @property
    def raster_ready(self) -> bool:
        return (
            super().raster_ready
            and self._presented_selection is not None
            and self._presented_selection == self._requested_selection
        )

    def _submit(self, kind: str, revision: int, function, *args) -> None:
        self._active_kind = kind
        self._active_revision = revision
        if not self._submit_future(function, *args):
            self._active_kind = None

    def _start_navigation(self) -> None:
        self._submit(
            "navigation",
            0,
            _load_navigation,
            self._navigation_loader,
            self._reference,
            self._memory_limit_bytes,
            self._cancelled,
        )

    def _start_cell(self, revision: int, selection: Selection) -> None:
        navigation = self._navigation
        if navigation is None:
            raise RuntimeError("occupancy navigation is not ready")
        self._submit(
            "cell",
            revision,
            _cell_job,
            self._cell_loader,
            self._reference,
            selection,
            navigation,
            self._cell_memory_limit_bytes,
            revision,
            self._cancelled,
        )

    def _build_navigation_controls(self) -> None:
        navigation = self._navigation
        if navigation is None:
            return
        navigator = AxisLayoutNavigator(
            navigation.axes,
            navigation.cell_layout,
            object_prefix="occupancyCell",
            action_text="Load exact cell",
            parent=self,
        )
        navigator.candidateChanged.connect(self._axis_value_changed)
        navigator.activated.connect(self._activate_indices)
        # Preserve the existing automation/accessibility identity while the
        # mechanical controls move into the shared widget owner.
        navigator.action_button.setObjectName("occupancyCellLoad")
        self._navigator = navigator
        self._layout.insertWidget(3, navigator)

    def _selection_from_controls(self) -> Selection | None:
        navigation = self._navigation
        navigator = self._navigator
        if navigation is None or navigator is None:
            return None
        indices = navigator.indices
        if indices is None:
            return None
        try:
            return navigation.selection_for_indices(indices[0], tuple(indices[1:]))
        except KeyError as error:
            raise ValueError("selected logical point was not acquired") from error

    def _set_controls(self, selection: Selection) -> None:
        navigation = self._navigation
        navigator = self._navigator
        if navigation is None or navigator is None:
            return
        repeat_index, _storage, logical, _label = navigation.resolve_selection(selection)
        values = (repeat_index, *logical)
        with signals_blocked(navigator):
            navigator.set_indices(values)

    def _clear_front(self) -> None:
        self._bundle = None
        self._presented_selection = None
        for board in self._boards:
            board.clear()

    def _build_boards(self, bundle: EncodedRasterDocument):
        if len(bundle.pages) != 1:
            raise ValueError("exact occupancy cell renderer must return one page")
        if len(self._boards) == 1:
            return self._boards
        return super()._build_boards(bundle)

    def _axis_value_changed(self) -> None:
        if self._closing:
            return
        self._request_revision += 1
        self._pending = None
        self._requested_selection = None
        self._clear_front()
        self._refresh_candidate_state()

    def _refresh_candidate_state(self) -> None:
        try:
            selection = self._selection_from_controls()
        except BaseException as error:
            self._status.setText("POINT NOT ACQUIRED")
            self._summary.setText("Choose a logical point present in the frozen PointLayout")
            self._diagnostic.setText(_error_summary(error))
            return
        if selection is None:
            self._status.setText("NEEDS CELL SELECTION")
            self._summary.setText(
                "Choose an exact index for every non-singleton repeat / point axis"
            )
            self._diagnostic.setText("")
            return
        navigation = self._navigation
        assert navigation is not None
        _repeat, _storage, _logical, label = navigation.resolve_selection(selection)
        self._status.setText("EXACT CELL READY TO LOAD")
        self._summary.setText(label)
        self._diagnostic.setText("")

    def _activate_indices(self, indices: object) -> None:
        navigation = self._navigation
        if navigation is None:
            return
        multi = tuple(indices)
        self._queue_cell(
            navigation.selection_for_indices(multi[0], tuple(multi[1:]))
        )

    def _queue_cell(self, selection: Selection) -> None:
        navigation = self._navigation
        if navigation is None:
            return
        repeat_index, point_storage, logical, label = navigation.resolve_selection(selection)
        canonical = navigation.selection_for_indices(repeat_index, logical)
        self._request_revision += 1
        revision = self._request_revision
        self._requested_selection = canonical
        self._set_controls(canonical)
        self._clear_front()
        self._status.setText("BUILDING OCCUPANCY CELL")
        self._summary.setText(
            f"{label} | physical point row {point_storage} | request {revision}"
        )
        self._diagnostic.setText("")
        if self._future is None and not self._pending_start_scheduled:
            self._start_cell(revision, canonical)
        else:
            self._pending = (revision, canonical)

    def _accept_navigation(self, navigation: OccupancyCellNavigation) -> None:
        if navigation.artifact_identity != self._reference.target_ref:
            raise ValueError("navigation result names a different occupancy artifact")
        self._navigation = navigation
        self._cell_memory_limit_bytes = (
            self._memory_limit_bytes - navigation.retained_upper_bound_bytes
        )
        if self._cell_memory_limit_bytes <= 0:
            raise MemoryError("occupancy navigation leaves no exact-cell display budget")
        self._build_navigation_controls()
        initial = self._initial_selection
        self._initial_selection = None
        if initial is None and all(axis.size == 1 for axis in navigation.axes):
            initial = navigation.selection_at_linear(0)
        if initial is None:
            self._refresh_candidate_state()
            return
        try:
            repeat_index, _storage, logical, _label = navigation.resolve_selection(initial)
            canonical = navigation.selection_for_indices(repeat_index, logical)
        except BaseException as error:
            self._status.setText("INITIAL CELL SELECTION INVALID")
            self._summary.setText(
                "Choose a valid exact repeat / point cell from the frozen metadata"
            )
            self._diagnostic.setText(_error_summary(error))
            return
        self._queue_cell(canonical)

    def _presentation_memory_limit(self) -> int:
        return self._cell_memory_limit_bytes

    def _accept_cell_result(self, result) -> None:
        revision, identity, selection, bundle = result
        navigation = self._navigation
        if (
            navigation is None
            or revision != self._request_revision
            or identity != navigation.identity
            or selection != self._requested_selection
        ):
            return
        if not isinstance(bundle, EncodedRasterDocument) or len(bundle.pages) != 1:
            raise TypeError("occupancy cell job returned an invalid raster document")
        if self._present_bundle(bundle):
            self._presented_selection = selection

    def _schedule_pending_start(self) -> None:
        if self._pending_start_scheduled:
            return
        self._pending_start_scheduled = True
        QtCore.QTimer.singleShot(0, self._start_pending_after_reap)

    def _start_pending_after_reap(self) -> None:
        self._pending_start_scheduled = False
        if self._closing or self._future is not None or self._pending is None:
            self._finish_close_if_ready()
            return
        pending = self._pending
        self._pending = None
        self._start_cell(*pending)

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        future = self._future
        if future is not None and future.done():
            kind = self._active_kind
            revision = self._active_revision
            self._future = None
            self._active_kind = None
            try:
                result = future.result()
            except CancelledError:
                if not self._closing and revision == self._request_revision:
                    self._status.setText("OCCUPANCY CELL CANCELLED")
            except BaseException as error:
                if not self._closing and (
                    kind == "navigation" or revision == self._request_revision
                ):
                    self._status.setText("OCCUPANCY CELL FAILED")
                    self._summary.setText("No exact-cell raster was admitted")
                    self._diagnostic.setText(_error_summary(error))
            else:
                if not self._closing:
                    try:
                        if kind == "navigation":
                            self._accept_navigation(result)
                        elif kind == "cell":
                            self._accept_cell_result(result)
                        else:
                            raise RuntimeError("unknown occupancy worker job")
                    except BaseException as error:
                        self._status.setText("OCCUPANCY CELL FAILED")
                        self._summary.setText("The committed artifact remains unchanged")
                        self._diagnostic.setText(_error_summary(error))
            if not self._closing and self._future is None and self._pending is not None:
                self._schedule_pending_start()
        self._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        self._pending = None
        self._pending_start_scheduled = False
        self._requested_selection = None
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(False)
        super().shutdown()

    def _finish_close_if_ready(self) -> None:
        if self._closing and self._future is None and not self._closed:
            self._navigation_loader = None
            self._cell_loader = None
            self._navigation = None
            self._navigator = None
        super()._finish_close_if_ready()


def open_occupancy_cell_workbench(
    navigation_loader,
    cell_loader,
    reference: OccupancyArtifactRef,
    *,
    selection: Selection | None = None,
    memory_limit_bytes: int = _DEFAULT_OCCUPANCY_CELL_GUI_MEMORY_LIMIT_BYTES,
) -> OccupancyCellWindow:
    """Open one FINAL-metadata navigator on the shared frozen-raster lane."""

    return _open_frozen_window(
        lambda: OccupancyCellWindow(
            navigation_loader,
            cell_loader,
            reference,
            selection=selection,
            memory_limit_bytes=memory_limit_bytes,
        )
    )


__all__ = ["OccupancyCellWindow", "open_occupancy_cell_workbench"]
