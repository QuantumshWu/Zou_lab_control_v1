"""Exact saved-fit GridPlot explorer over immutable current artifacts."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from pathlib import Path
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import Selection
from zlc_frontend import (
    DataFigure,
    FigurePanelRegion,
    FitGridCellSummary,
    FitGridModel,
    FitGridPage,
)
from zlc_frontend.encoded_raster import EncodedRasterDocument, EncodedRasterPage
from zlc_frontend.qt_widgets import (
    AxisLayoutNavigator,
    FluentButton,
    FluentLabel,
    GREY,
    ORANGE,
    signals_blocked,
)
from zlc_neutral_atom.capture_fit_reference import CaptureFitResultArtifactRef
from zlc_storage import positive_integer

from ._frozen_raster import (
    FrozenRasterWindow,
)
from ._window_runtime import (
    cancel_export_commits,
    error_summary,
    open_workbench_window,
    stage_and_replace_export,
)


_DEFAULT_FIT_GRID_MEMORY_LIMIT_BYTES = 512 << 20


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _validated_regions(
    model: FitGridModel,
    regions: tuple[FigurePanelRegion, ...],
) -> tuple[FigurePanelRegion, ...]:
    prepared = tuple(regions)
    if not prepared or any(
        not isinstance(region, FigurePanelRegion) for region in prepared
    ):
        raise TypeError("saved-fit grid renderer must return panel regions")
    if len(prepared) > 36:
        raise ValueError("saved-fit grid page exceeded 36 display panels")
    if len({region.key for region in prepared}) != len(prepared):
        raise ValueError("saved-fit grid panel keys must be unique")
    for region in prepared:
        expected = model.storage_index_or_none(region.selection)
        if region.fit_storage_index != expected:
            raise ValueError(
                "saved-fit panel hit map disagrees with the exact batch layout"
            )
    return prepared


def _load_grid_view(
    view_loader,
    reference: CaptureFitResultArtifactRef,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    memory_limit_bytes: int,
    revision: int,
    return_model: bool,
    cancelled: threading.Event,
):
    _require_not_cancelled(cancelled)
    loaded = view_loader(
        reference,
        page_address=page_address,
        cell_selection=cell_selection,
        memory_limit_bytes=memory_limit_bytes,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 5:
        raise TypeError(
            "saved-fit loader must return figure/model/page/cell summary/session budget"
        )
    figure, model, page, cell_summary, session_retained_bytes = loaded
    if not isinstance(figure, DataFigure):
        raise TypeError("saved-fit loader must return DataFigure")
    if not isinstance(model, FitGridModel):
        raise TypeError("saved-fit loader must return FitGridModel")
    if model.artifact_identity != reference.target_ref:
        raise ValueError("saved-fit loader names a different artifact")
    session_retained_bytes = positive_integer(
        session_retained_bytes,
        "saved-fit session retained bytes",
    )
    if cell_selection is None:
        if not isinstance(page, FitGridPage) or cell_summary is not None:
            raise TypeError("saved-fit page load returned invalid compact metadata")
        resolved_selection = page.selection
    else:
        if page is not None or not isinstance(cell_summary, FitGridCellSummary):
            raise TypeError("saved-fit cell load returned invalid compact metadata")
        if cell_summary.selection != cell_selection:
            raise ValueError("saved-fit cell summary belongs to another selection")
        resolved_selection = cell_selection
    payload, regions = figure.to_png_bytes_with_panel_regions(
        memory_limit_bytes=figure.render_memory_limit_bytes,
    )
    summary = (
        f"{model.summary} · {page.label}"
        if page is not None
        else model.summary
    )
    bundle = EncodedRasterDocument(
        summary,
        (EncodedRasterPage("figure", "Fit grid", payload),),
    )
    prepared_regions = _validated_regions(model, regions)
    if cell_selection is not None:
        if len(prepared_regions) != 1:
            raise ValueError("exact saved-fit focus must render one panel")
        if prepared_regions[0].fit_storage_index != cell_summary.storage_index:
            raise ValueError("focused panel and stored cell summary disagree")
    required = bundle.source_front_peak_nbytes
    if return_model:
        required += session_retained_bytes + model.retained_upper_bound_bytes
    if required > memory_limit_bytes:
        raise MemoryError("saved-fit session and raster exceed worker budget")
    _require_not_cancelled(cancelled)
    model_identity = model.identity
    returned_model = model if return_model else None
    del loaded, figure
    if not return_model:
        del model
    return (
        revision,
        returned_model,
        model_identity,
        session_retained_bytes,
        page,
        cell_summary,
        resolved_selection,
        bundle,
        prepared_regions,
    )


def _export_grid_view(
    view_loader,
    reference: CaptureFitResultArtifactRef,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    destination: Path,
    memory_limit_bytes: int,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
):
    _require_not_cancelled(cancelled)
    loaded = view_loader(
        reference,
        page_address=page_address,
        cell_selection=cell_selection,
        memory_limit_bytes=memory_limit_bytes,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 5:
        raise TypeError("saved-fit export loader returned invalid values")
    figure, model, page, cell_summary, _session_retained_bytes = loaded
    if not isinstance(figure, DataFigure) or not isinstance(model, FitGridModel):
        raise TypeError("saved-fit export loader returned invalid values")
    if model.artifact_identity != reference.target_ref:
        raise ValueError("saved-fit export loader names another artifact")
    if cell_selection is None:
        if not isinstance(page, FitGridPage) or cell_summary is not None:
            raise TypeError("saved-fit page export metadata is invalid")
    elif page is not None or not isinstance(cell_summary, FitGridCellSummary):
        raise TypeError("saved-fit cell export metadata is invalid")
    _require_not_cancelled(cancelled)
    target = Path(destination)
    image_format = target.suffix.lstrip(".") or "png"
    if not target.suffix:
        target = target.with_suffix(f".{image_format}")

    def write_staged(temporary: Path) -> None:
        exported = figure.export(
            temporary,
            image_format=image_format,
            memory_limit_bytes=figure.render_memory_limit_bytes,
        )
        if Path(exported) != temporary:
            raise RuntimeError("saved-fit export changed its staged destination")

    committed = stage_and_replace_export(
        target,
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, committed


class SavedFitGridWindow(FrozenRasterWindow):
    """Browse one exact saved ``FitResultBatch`` without ever re-solving it."""

    def __init__(
        self,
        view_loader,
        reference: CaptureFitResultArtifactRef,
        *,
        memory_limit_bytes: int,
    ) -> None:
        if not callable(view_loader):
            raise TypeError("saved-fit view_loader must be callable")
        if not isinstance(reference, CaptureFitResultArtifactRef):
            raise TypeError("reference must be CaptureFitResultArtifactRef")
        self._view_loader = view_loader
        self._reference = reference
        self._model: FitGridModel | None = None
        self._session_retained_bytes = 0
        self._navigator: AxisLayoutNavigator | None = None
        self._page: FitGridPage | None = None
        self._page_bundle: EncodedRasterDocument | None = None
        self._page_regions: tuple[FigurePanelRegion, ...] = ()
        self._regions: tuple[FigurePanelRegion, ...] = ()
        self._current_selection: Selection | None = None
        self._requested_selection: Selection | None = None
        self._requested_page_address: tuple[int, ...] | None = None
        self._request_revision = 0
        self._active_revision = 0
        self._active_kind: str | None = "page"
        self._front_memory_limit = memory_limit_bytes
        self._export_commit_lock = threading.Lock()

        super().__init__(
            None,
            window_title="Saved Fit Grid",
            mode_text="EXACT SAVED FIT · GRID EXPLORER · DISPLAY ONLY",
            loading_summary=f"Resolving {reference.target_ref}…",
            object_prefix="savedFitGrid",
            subject="SAVED FIT GRID",
            memory_limit_bytes=memory_limit_bytes,
        )

        self._previous_page_button = FluentButton(
            "Previous page",
            self,
            color=GREY,
        )
        self._previous_page_button.setObjectName("savedFitGridPreviousPage")
        self._overview_button = FluentButton("Overview", self, color=GREY)
        self._overview_button.setObjectName("savedFitGridOverview")
        self._next_page_button = FluentButton("Next page", self, color=GREY)
        self._next_page_button.setObjectName("savedFitGridNextPage")
        self._export_button = FluentButton("Export image…", self, color=ORANGE)
        self._export_button.setObjectName("savedFitGridExport")
        actions = QtWidgets.QHBoxLayout()
        for button in (
            self._previous_page_button,
            self._overview_button,
            self._next_page_button,
            self._export_button,
        ):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch(1)

        self._navigation_host = QtWidgets.QWidget(self)
        host_layout = QtWidgets.QVBoxLayout(self._navigation_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(6)
        host_layout.addLayout(actions)
        self._cell_detail = FluentLabel("", self._navigation_host)
        self._cell_detail.setObjectName("savedFitGridCellDetail")
        self._cell_detail.setWordWrap(True)
        self._cell_detail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        host_layout.addWidget(self._cell_detail)
        self._layout.insertWidget(3, self._navigation_host)

        self._previous_page_button.clicked.connect(
            lambda: self._move_page(-1)
        )
        self._overview_button.clicked.connect(self._show_page)
        self._next_page_button.clicked.connect(lambda: self._move_page(1))
        self._export_button.clicked.connect(self._choose_export)
        self._submit_view(
            "page",
            page_address=None,
            cell_selection=None,
            memory_limit_bytes=self._memory_limit_bytes,
        )

    def _build_boards(self, bundle: EncodedRasterDocument):
        if len(bundle.pages) != 1:
            raise ValueError("saved-fit GridPlot requires one atomic board raster")
        boards = super()._build_boards(bundle)
        boards[0].normalizedDoubleClicked.connect(self._focus_at)
        return boards

    def _presentation_memory_limit(self) -> int:
        return self._front_memory_limit

    def _install_model(self, model: FitGridModel) -> None:
        if self._model is not None:
            raise RuntimeError("saved-fit compact metadata was installed twice")
        self._model = model
        if not model.axes:
            return
        navigator = AxisLayoutNavigator(
            model.axes,
            model.layout,
            object_prefix="savedFitGrid",
            action_text="Focus exact cell",
            parent=self._navigation_host,
        )
        navigator.candidateChanged.connect(self._candidate_changed)
        navigator.activated.connect(self._focus_indices)
        self._navigator = navigator
        self._navigation_host.layout().insertWidget(1, navigator)
        self._candidate_changed()

    def _retained_state_bytes(
        self,
        *,
        keep_page: bool,
        keep_current: bool,
    ) -> int:
        retained = (
            self._session_retained_bytes
            + (
                0
                if self._model is None
                else self._model.retained_upper_bound_bytes
            )
        )
        page_bundle = self._page_bundle
        if keep_page and page_bundle is not None:
            retained += page_bundle.source_front_peak_nbytes
        if (
            keep_current
            and self._bundle is not None
            and self._bundle is not page_bundle
        ):
            retained += self._bundle.source_front_peak_nbytes
        return retained

    def _available_worker_limit(
        self,
        *,
        keep_page: bool,
        keep_current: bool,
    ) -> int:
        available = self._memory_limit_bytes - self._retained_state_bytes(
            keep_page=keep_page,
            keep_current=keep_current,
        )
        if available <= 0:
            raise MemoryError("saved-fit retained fronts leave no worker budget")
        return available

    def _set_controls_enabled(self, enabled: bool) -> None:
        page = self._page
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(enabled)
        self._previous_page_button.setEnabled(
            enabled and page is not None and page.previous_address is not None
        )
        self._next_page_button.setEnabled(
            enabled and page is not None and page.next_address is not None
        )
        self._overview_button.setEnabled(
            enabled
            and self._current_selection is not None
            and self._page_bundle is not None
        )
        self._export_button.setEnabled(enabled and self._bundle is not None)

    def _candidate_changed(self) -> None:
        navigator = self._navigator
        model = self._model
        if navigator is None or model is None:
            return
        indices = navigator.indices
        if indices is None:
            self._cell_detail.setText(
                "Choose every non-singleton batch axis to inspect one exact saved fit cell."
            )
            return
        try:
            selection = model.selection_for_indices(indices)
            storage, _multi, address = model.resolve_selection(selection)
        except BaseException as error:
            self._cell_detail.setText(error_summary(error))
            return
        self._cell_detail.setText(
            f"{address}\nstorage row {storage} · activate to load stored fit facts"
        )

    def _submit_view(
        self,
        kind: str,
        *,
        page_address: tuple[int, ...] | None,
        cell_selection: Selection | None,
        memory_limit_bytes: int,
    ) -> None:
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = kind
        self._requested_page_address = page_address
        self._requested_selection = cell_selection
        if not self._submit_future(
            _load_grid_view,
            self._view_loader,
            self._reference,
            page_address,
            cell_selection,
            memory_limit_bytes,
            self._active_revision,
            self._model is None,
            self._cancelled,
        ):
            self._active_kind = None
            self._set_controls_enabled(True)

    def _start_page(self, address: tuple[int, ...]) -> None:
        if self._future is not None or self._closing:
            return
        try:
            worker_limit = self._available_worker_limit(
                keep_page=False,
                keep_current=False,
            )
        except BaseException as error:
            self._diagnostic.setText(error_summary(error))
            return
        self._clear_bundle()
        self._page_bundle = None
        self._page_regions = ()
        self._regions = ()
        self._current_selection = None
        self._status.setText("BUILDING FIT GRID PAGE")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        self._submit_view(
            "page",
            page_address=tuple(address),
            cell_selection=None,
            memory_limit_bytes=worker_limit,
        )

    def _move_page(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("fit grid page direction must be -1 or 1")
        page = self._page
        if page is None:
            return
        address = (
            page.previous_address if direction < 0 else page.next_address
        )
        if address is not None:
            self._start_page(address)

    def _focus_at(self, x: float, y: float) -> None:
        if self._future is not None or self._closing:
            return
        for region in self._regions:
            if not region.contains(x, y):
                continue
            if region.fit_storage_index is None:
                self._status.setText("FIT CELL NOT PRESENT")
                self._cell_detail.setText(
                    "This logical gallery position is a sparse-layout hole; "
                    "no neighbouring stored fit row was substituted."
                )
                return
            if region.selection is not None:
                self._start_focus(region.selection)
            return

    def _focus_indices(self, indices: object) -> None:
        model = self._model
        if model is None:
            return
        try:
            selection = model.selection_for_indices(tuple(indices))
            if selection is None:
                return
            self._start_focus(selection)
        except BaseException as error:
            self._status.setText("FIT CELL INVALID")
            self._diagnostic.setText(error_summary(error))

    def _start_focus(self, selection: Selection) -> None:
        if self._future is not None or self._closing:
            return
        model = self._model
        if model is None or self._page_bundle is None:
            return
        try:
            model.resolve_selection(selection)
            worker_limit = self._available_worker_limit(
                keep_page=True,
                keep_current=False,
            )
        except BaseException as error:
            self._status.setText("FIT CELL INVALID")
            self._diagnostic.setText(error_summary(error))
            return
        self._clear_bundle()
        self._regions = ()
        self._current_selection = None
        self._status.setText("BUILDING FIT CELL")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        self._submit_view(
            "focus",
            page_address=None,
            cell_selection=selection,
            memory_limit_bytes=worker_limit,
        )

    def _show_page(self) -> None:
        model = self._model
        page = self._page
        bundle = self._page_bundle
        if self._future is not None or model is None or page is None or bundle is None:
            return
        self._front_memory_limit = (
            self._memory_limit_bytes
            - self._session_retained_bytes
            - model.retained_upper_bound_bytes
        )
        if self._present_bundle(bundle):
            self._regions = self._page_regions
            self._current_selection = None
            self._requested_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
            self._set_controls_enabled(True)
        else:
            self._set_controls_enabled(True)

    def _choose_export(self) -> None:
        if self._future is not None or self._closing or self._bundle is None:
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export saved fit view",
            "saved_fit_grid.png",
            "Images (*.png *.pdf *.svg *.jpg *.jpeg)",
        )
        if path:
            self._start_export(Path(path))

    def _start_export(self, destination: Path) -> None:
        if self._future is not None or self._closing:
            return
        page = self._page
        try:
            export_limit = self._available_worker_limit(
                keep_page=True,
                keep_current=True,
            )
        except BaseException as error:
            self._status.setText("FIT VIEW EXPORT FAILED")
            self._diagnostic.setText(error_summary(error))
            return
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = "export"
        self._status.setText("EXPORTING FIT VIEW")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        if not self._submit_future(
            _export_grid_view,
            self._view_loader,
            self._reference,
            (
                None
                if self._current_selection is not None or page is None
                else page.address
            ),
            self._current_selection,
            Path(destination),
            export_limit,
            self._active_revision,
            self._cancelled,
            self._export_commit_lock,
        ):
            self._active_kind = None
            self._set_controls_enabled(True)

    def _accept_view_result(self, result, kind: str, revision: int) -> None:
        (
            result_revision,
            model,
            model_identity,
            session_retained_bytes,
            page,
            cell_summary,
            resolved_selection,
            bundle,
            regions,
        ) = result
        if result_revision != revision or revision != self._request_revision:
            self._set_controls_enabled(True)
            return
        if self._model is None:
            if not isinstance(model, FitGridModel):
                raise TypeError("initial saved-fit page omitted compact metadata")
            if model.identity != model_identity:
                raise ValueError("saved-fit worker metadata identity changed")
            self._session_retained_bytes = positive_integer(
                session_retained_bytes,
                "saved-fit session retained bytes",
            )
            self._install_model(model)
        else:
            if (
                model is not None
                or model_identity != self._model.identity
                or session_retained_bytes != self._session_retained_bytes
            ):
                raise ValueError("saved-fit metadata changed during one exact-ref session")
        current_model = self._model
        assert current_model is not None
        if kind == "page":
            if not isinstance(page, FitGridPage) or cell_summary is not None:
                raise TypeError("saved-fit page result is invalid")
            self._page = page
            self._page_bundle = bundle
            self._page_regions = regions
            self._front_memory_limit = (
                self._memory_limit_bytes
                - self._session_retained_bytes
                - current_model.retained_upper_bound_bytes
            )
            if not self._present_bundle(bundle):
                self._set_controls_enabled(True)
                return
            self._regions = regions
            self._current_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{current_model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
        elif kind == "focus":
            if page is not None or not isinstance(cell_summary, FitGridCellSummary):
                raise TypeError("saved-fit focus result is invalid")
            retained_page = (
                0
                if self._page_bundle is None
                else self._page_bundle.source_front_peak_nbytes
            )
            self._front_memory_limit = (
                self._memory_limit_bytes
                - self._session_retained_bytes
                - current_model.retained_upper_bound_bytes
                - retained_page
            )
            if not self._present_bundle(bundle):
                self._set_controls_enabled(True)
                return
            self._regions = regions
            self._current_selection = resolved_selection
            if self._navigator is not None:
                _storage, multi, _label = current_model.resolve_selection(
                    resolved_selection
                )
                with signals_blocked(self._navigator):
                    self._navigator.set_indices(multi)
            self._status.setText("FIT CELL FOCUSED")
            self._summary.setText(current_model.summary)
            self._cell_detail.setText(cell_summary.text)
        else:
            raise RuntimeError("unknown saved-fit view result")
        self._requested_selection = resolved_selection
        self._diagnostic.setText("")
        self._set_controls_enabled(True)

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        revision = self._active_revision
        self._active_kind = None
        try:
            result = future.result()
        except CancelledError:
            if not self._closing:
                self._status.setText("SAVED FIT GRID CANCELLED")
                self._set_controls_enabled(True)
        except BaseException as error:
            if not self._closing:
                self._status.setText("SAVED FIT GRID FAILED")
                self._summary.setText("The exact saved artifact remains unchanged")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)
        else:
            if self._closing:
                return
            try:
                if kind in ("page", "focus"):
                    self._accept_view_result(result, kind, revision)
                elif kind == "export":
                    result_revision, destination = result
                    if (
                        result_revision == revision == self._request_revision
                    ):
                        self._status.setText("FIT VIEW EXPORTED")
                        self._diagnostic.setText(str(destination))
                    self._set_controls_enabled(True)
                else:
                    raise RuntimeError("unknown saved-fit worker result")
            except BaseException as error:
                self._status.setText("SAVED FIT GRID FAILED")
                self._summary.setText("The exact saved artifact remains unchanged")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and self._current_selection is not None:
            self._show_page()
            event.accept()
            return
        super().keyPressEvent(event)

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        cancel_export_commits(
            cancelled=self._cancelled,
            commit_lock=self._export_commit_lock,
        )
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(False)
        for button in (
            self._previous_page_button,
            self._overview_button,
            self._next_page_button,
            self._export_button,
        ):
            button.setEnabled(False)
        super().shutdown()

    def _finish_close_if_ready(self) -> None:
        if self._closing and self._future is None and not self._closed:
            self._view_loader = None
            self._model = None
            self._session_retained_bytes = 0
            self._navigator = None
            self._page = None
            self._page_bundle = None
            self._page_regions = ()
            self._regions = ()
        super()._finish_close_if_ready()


def open_saved_fit_grid_workbench(
    view_loader,
    reference: CaptureFitResultArtifactRef,
    *,
    memory_limit_bytes: int = _DEFAULT_FIT_GRID_MEMORY_LIMIT_BYTES,
) -> SavedFitGridWindow:
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    return open_workbench_window(
        lambda: SavedFitGridWindow(
            view_loader,
            reference,
            memory_limit_bytes=limit,
        )
    )


__all__ = ["SavedFitGridWindow", "open_saved_fit_grid_workbench"]
