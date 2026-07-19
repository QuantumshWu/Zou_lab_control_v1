"""Nonblocking Qt viewer for one frozen current :class:`DataFigure`.

The generic fallback remains an immutable encoded board.  A closed, already
earned product slice -- one logical HISTOGRAM panel with one or more series --
uses the shared typed board so display interaction never has to reverse-map a
whole-board PNG.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from pathlib import Path
import threading

from PyQt5 import QtCore, QtWidgets

from zlc_data import dataset_revision_ref_to_tree
from zlc_frontend import (
    BoardFrame,
    CoherenceStamp,
    DataFigure,
    HistogramPanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    SourceIdentity,
)
from zlc_frontend.encoded_raster import EncodedRasterDocument, EncodedRasterPage
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import EvaluatedHistogram, ViewIntent
from zlc_frontend.histogram_display import (
    HistogramCountScale,
    HistogramDisplayState,
    histogram_display_form_spec,
    histogram_display_form_values,
    histogram_display_from_form,
    histogram_display_with_x_view,
)
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentPopup,
    FluentRevisionedFormEditor,
    FluentSwitch,
    GREY,
    ORANGE,
    QtRasterBoard,
    runtime_range_placeholders,
    show_fluent_popup_for_anchor,
    sync_revisioned_form_editors,
)
from zlc_frontend.selector import (
    HistogramInteractionIntent,
    HistogramRangeGesture,
    HistogramViewportCommit,
    PanelInteractionOrigin,
)
from zlc_storage import canonical_digest, nonnegative_integer, positive_integer

from ._frozen_raster import FrozenRasterWindow
from ._window_runtime import (
    cancel_export_commits,
    error_summary,
    open_workbench_window,
    stage_and_replace_export,
)


_DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_HISTOGRAM_BOARD_ID = "generic-histogram-figure"
_HISTOGRAM_PANEL_ID = "generic-histogram"
_HISTOGRAM_RASTER_SIZE = (800, 520)
_HISTOGRAM_JOIN_SCHEMA_DIGEST = canonical_digest(
    {
        "schema": "zlc_frontend.FrozenHistogramFigureJoin",
        "fields": ("document", "input"),
    }
)


def _require_not_cancelled(cancelled: threading.Event | None) -> None:
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()


def _figure_summary(figure: DataFigure) -> str:
    document = figure.document
    intents = tuple(dict.fromkeys(layer.view.intent.value for layer in document.layers))
    panel_count = sum(len(layer.cells) for layer in figure.evaluated.layers)
    return (
        f"{'/'.join(value.lower() for value in intents)} · {panel_count} panel(s) · "
        f"document revision {document.revision}"
    )


def _figure_render_limit(figure: DataFigure, memory_limit_bytes: int) -> int:
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    frozen_limit = figure.render_memory_limit_bytes
    return limit if frozen_limit is None else min(limit, frozen_limit)


def _single_histogram_panel(figure: DataFigure) -> bool:
    """Recognize only the current typed product boundary; never guess rank."""

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    document = figure.document
    evaluated = figure.evaluated
    if (
        len(document.layers) != 1
        or document.layers[0].view.intent is not ViewIntent.HISTOGRAM
        or len(evaluated.layers) != 1
        or len(evaluated.layers[0].cells) != 1
        or len(evaluated.inputs) != 1
    ):
        return False
    series = evaluated.layers[0].cells[0].series
    return bool(series) and all(
        isinstance(item.data, EvaluatedHistogram) for item in series
    )


def _encoded_figure(
    figure: DataFigure,
    memory_limit_bytes: int,
    cancelled: threading.Event | None,
) -> EncodedRasterDocument:
    _require_not_cancelled(cancelled)
    render_limit = _figure_render_limit(figure, memory_limit_bytes)
    payload = figure.to_png_bytes(memory_limit_bytes=render_limit)
    _require_not_cancelled(cancelled)
    document = EncodedRasterDocument(
        _figure_summary(figure),
        (EncodedRasterPage("figure", "Figure", payload),),
    )
    if document.source_front_peak_nbytes > memory_limit_bytes:
        raise MemoryError(
            "encoded raster fronts require "
            f"{document.source_front_peak_nbytes} bytes; limit is {memory_limit_bytes}"
        )
    return document


def _render_figure(
    loader,
    memory_limit_bytes: int,
    cancelled: threading.Event | None = None,
) -> EncodedRasterDocument:
    """Retain the exact encoded fallback used by every non-typed figure."""

    _require_not_cancelled(cancelled)
    figure = loader()
    if not isinstance(figure, DataFigure):
        raise TypeError("figure loader must return DataFigure")
    return _encoded_figure(figure, memory_limit_bytes, cancelled)


@dataclass(frozen=True, slots=True)
class _HistogramFigureFront:
    summary: str
    frame: BoardFrame
    required_peak_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary:
            raise ValueError("histogram figure summary must be non-empty")
        if not isinstance(self.frame, BoardFrame) or len(self.frame.panels) != 1:
            raise TypeError("histogram figure front requires one BoardFrame panel")
        panel = self.frame.panels[0]
        if panel.panel_id != _HISTOGRAM_PANEL_ID or not isinstance(
            panel.display_payload,
            HistogramPanelPayload,
        ):
            raise ValueError("histogram figure front has another typed payload")
        raster = panel.raster
        if (raster.width, raster.height) != _HISTOGRAM_RASTER_SIZE:
            raise ValueError(
                "histogram figure front has another raster geometry"
            )
        if raster.stride_bytes != raster.width * 4:
            raise ValueError("histogram figure front requires packed RGBA")
        object.__setattr__(
            self,
            "required_peak_bytes",
            positive_integer(self.required_peak_bytes, "required_peak_bytes"),
        )


def _histogram_join_digest(figure: DataFigure) -> str:
    evaluated = figure.evaluated
    source = evaluated.inputs[0]
    return canonical_digest(
        {
            "schema": "zlc_frontend.FrozenHistogramFigureJoin",
            "document": {
                "id": figure.document.document_id,
                "revision": figure.document.revision,
            },
            "input": {
                "dataset_id": source.dataset_id.value,
                "ref": dataset_revision_ref_to_tree(source.ref),
            },
        }
    )


def _histogram_front_contract(
    front: _HistogramFigureFront,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Freeze data/provenance identity while excluding display revision."""

    if not isinstance(front, _HistogramFigureFront):
        raise TypeError("front must be _HistogramFigureFront")
    frame = front.frame
    panel = frame.panels[0]
    payload = panel.display_payload
    assert isinstance(payload, HistogramPanelPayload)
    stamp = panel.coherence_stamp
    if len(stamp.presentations) != 1:
        raise ValueError("generic histogram front requires one presentation identity")
    presentation = stamp.presentations[0]
    if presentation.panel_id != panel.panel_id:
        raise ValueError("histogram presentation names another panel")
    identity = (
        frame.board_id,
        frame.layout_generation,
        panel.panel_id,
        panel.coherence_group,
        panel.source_identity,
        stamp.run_id,
        stamp.provenance_epoch_id,
        stamp.join_key_type,
        stamp.join_key_schema_fingerprint,
        stamp.join_key_digest,
        stamp.inputs,
        presentation.panel_id,
        presentation.document_id,
        presentation.document_revision,
        presentation.selection_revision,
        payload.evaluated_input,
        payload.series_labels,
        payload.value_unit,
    )
    return identity, tuple(payload.series)


def _histogram_front_required_peak_bytes(
    figure: DataFigure,
    state: HistogramDisplayState,
) -> int:
    if not _single_histogram_panel(figure):
        raise ValueError("typed histogram budget requires one logical panel")
    if not isinstance(state, HistogramDisplayState):
        raise TypeError("state must be HistogramDisplayState")
    from zlc_frontend.matplotlib_render import (
        estimate_live_panel_raster_peak_nbytes,
        evaluated_figure_array_nbytes,
    )

    width, height = _HISTOGRAM_RASTER_SIZE
    series_count = len(figure.evaluated.layers[0].cells[0].series)
    return estimate_live_panel_raster_peak_nbytes(
        width,
        height,
        evaluated_data_upper_bound_bytes=evaluated_figure_array_nbytes(
            figure.evaluated
        ),
        histogram_bins=state.bin_count,
        histogram_series_count=series_count,
        # Admission covers the currently painted Qt/held front while the next
        # immutable worker front is composed and admitted.
        extra_retained_fronts=1,
    )


def _render_histogram_front(
    figure: DataFigure,
    state: HistogramDisplayState,
    *,
    current_count_limits: tuple[float, float] | None,
    previous_relim_mode,
    previous_count_scale: HistogramCountScale | None,
    sequence: int,
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> _HistogramFigureFront:
    if not _single_histogram_panel(figure):
        raise ValueError("typed histogram render requires one logical HISTOGRAM panel")
    if not isinstance(state, HistogramDisplayState):
        raise TypeError("state must be HistogramDisplayState")
    _require_not_cancelled(cancelled)

    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    width, height = _HISTOGRAM_RASTER_SIZE
    required = _histogram_front_required_peak_bytes(figure, state)
    effective_limit = _figure_render_limit(figure, memory_limit_bytes)
    if required > effective_limit:
        raise MemoryError(
            f"interactive histogram requires {required} bytes; "
            f"limit is {effective_limit}"
        )

    renderer = SinglePanelAggRenderer(
        figure.document,
        width=width,
        height=height,
    )
    try:
        raster, payload = renderer.render_interactive_histogram(
            figure.evaluated,
            state,
            current_count_limits=current_count_limits,
            previous_relim_mode=previous_relim_mode,
            previous_count_scale=previous_count_scale,
        )
    finally:
        renderer.close()
    _require_not_cancelled(cancelled)

    evaluated_input = payload.evaluated_input
    presentation = PanelPresentationIdentity(
        _HISTOGRAM_PANEL_ID,
        figure.document.document_id,
        figure.document.revision,
        0,
        state.revision,
    )
    ref = evaluated_input.ref
    stamp = CoherenceStamp(
        f"figure:{ref.block_id.value}",
        ref.stream_generation.value,
        "FrozenHistogramFigureJoin",
        _HISTOGRAM_JOIN_SCHEMA_DIGEST,
        _histogram_join_digest(figure),
        (evaluated_input,),
        (presentation,),
    )
    source = SourceIdentity(
        evaluated_input.dataset_id,
        ref.block_id,
        ref.stream_generation,
        ref.schema_fingerprint,
    )
    frame = BoardFrame(
        _HISTOGRAM_BOARD_ID,
        0,
        sequence,
        (
            PanelFrame(
                _HISTOGRAM_PANEL_ID,
                "frozen-histogram",
                source,
                stamp,
                raster,
                payload,
            ),
        ),
    )
    return _HistogramFigureFront(_figure_summary(figure), frame, required)


def _export_histogram_png(
    frame: BoardFrame,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> tuple[int, Path]:
    if not isinstance(frame, BoardFrame) or len(frame.panels) != 1:
        raise TypeError("histogram export requires one exact BoardFrame")
    panel = frame.panels[0]
    if panel.panel_id != _HISTOGRAM_PANEL_ID or not isinstance(
        panel.display_payload,
        HistogramPanelPayload,
    ):
        raise ValueError("histogram export frame has another presentation")
    raster = panel.raster
    if raster.stride_bytes != raster.width * 4:
        raise ValueError("histogram export requires a packed RGBA raster")

    def write_staged(path: Path) -> None:
        from PIL import Image

        image = Image.frombytes(
            "RGBA",
            (raster.width, raster.height),
            raster.pixels,
        )
        try:
            image.save(path, format="PNG")
        finally:
            image.close()

    result = stage_and_replace_export(
        Path(destination),
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, result


class DataFigureWindow(FrozenRasterWindow):
    """Hybrid generic viewer with one closed typed HISTOGRAM product path."""

    def __init__(
        self,
        initial_loader,
        histogram_renderer,
        *,
        memory_limit_bytes: int,
    ) -> None:
        if not callable(initial_loader) or not callable(histogram_renderer):
            raise TypeError("figure worker callables must be callable")
        self._histogram_renderer = histogram_renderer
        self._view_family: str | None = None
        self._display = HistogramDisplayState()
        self._histogram_contract: (
            tuple[tuple[object, ...], tuple[object, ...]] | None
        ) = None
        self._typed_pages_admitted = False
        self._request_revision = 0
        self._active_kind: str | None = "initial"
        self._pending_state: HistogramDisplayState | None = None
        self._pending_origin: PanelInteractionOrigin | None = None
        self._pending_editor: FluentRevisionedFormEditor | None = None
        self._pending_editor_revision: int | None = None
        self._export_commit_lock = threading.Lock()

        super().__init__(
            None,
            window_title="Data Figure",
            mode_text="FROZEN DATA FIGURE · DISPLAY ONLY",
            loading_summary="Resolving immutable input…",
            object_prefix="figureViewer",
            subject="figure",
            memory_limit_bytes=memory_limit_bytes,
        )

        self._histogram_page = QtWidgets.QWidget(self._tabs)
        self._histogram_page.hide()
        page_layout = QtWidgets.QVBoxLayout(self._histogram_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self._board_widget = QtRasterBoard(
            (_HISTOGRAM_PANEL_ID,),
            self._histogram_page,
            columns=1,
            empty_text="Resolving exact histogram…",
        )
        self._board_widget.setObjectName("figureViewerHistogramBoard")
        self._board_widget.setMinimumSize(480, 320)
        self._board_widget.bind_histogram_interaction(
            _HISTOGRAM_PANEL_ID,
            self._accept_histogram_interaction,
            enabled=True,
        )
        page_layout.addWidget(self._board_widget, 1)
        self._edit_display = FluentRevisionedFormEditor(
            histogram_display_form_spec(),
            "histogram display",
            runtime_placeholder_fields=("count_min", "count_max"),
            parent=self._tabs,
        )
        self._edit_display.setObjectName("figureViewerHistogramEditEditor")
        self._edit_display.hide()

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("figureViewerHistogramSettingsPopup")
        popup_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        self._setting_display = FluentRevisionedFormEditor(
            histogram_display_form_spec(),
            "histogram display",
            runtime_placeholder_fields=("count_min", "count_max"),
            parent=self._settings_popup,
        )
        self._setting_display.setObjectName(
            "figureViewerHistogramSettingEditor"
        )
        popup_layout.addWidget(self._setting_display)

        self._interaction_switch = FluentSwitch("Interact", self)
        self._interaction_switch.setObjectName("figureViewerHistogramInteractSwitch")
        self._interaction_switch.setChecked(True)
        self._settings_button = FluentButton("Setting…", self, color=GREY)
        self._settings_button.setObjectName("figureViewerHistogramSettingButton")
        self._export_button = FluentButton("Export PNG…", self, color=ORANGE)
        self._export_button.setObjectName("figureViewerHistogramExportButton")
        self._controls.insertWidget(0, self._interaction_switch)
        self._controls.insertWidget(1, self._settings_button)
        self._controls.insertWidget(2, self._export_button)
        for widget in (
            self._interaction_switch,
            self._settings_button,
            self._export_button,
        ):
            widget.hide()

        self._settings_button.clicked.connect(
            lambda: show_fluent_popup_for_anchor(
                self._settings_popup,
                self._settings_button,
            )
        )
        self._export_button.clicked.connect(self._choose_export)
        self._interaction_switch.toggled.connect(self._toggle_interaction)
        self._edit_display.applyRequested.connect(
            lambda revision, values: self._apply_display_form(
                self._edit_display,
                revision,
                values,
            )
        )
        self._setting_display.applyRequested.connect(
            lambda revision, values: self._apply_display_form(
                self._setting_display,
                revision,
                values,
            )
        )
        self._edit_display.cancelRequested.connect(
            lambda: self._reload_editor(self._edit_display)
        )
        self._setting_display.cancelRequested.connect(
            lambda: self._reload_editor(self._setting_display)
        )
        self._set_typed_controls_enabled(False)

        self._submit_future(
            initial_loader,
            self._display,
            self._memory_limit_bytes,
            self._request_revision,
            self._cancelled,
        )

    @property
    def raster_ready(self) -> bool:
        if self._view_family == "histogram":
            frame = self._board_widget.front_frame
            payload = self._board_widget.visible_histogram_payload(
                _HISTOGRAM_PANEL_ID
            )
            return bool(
                frame is not None
                and self._pending_state is None
                and payload is not None
                and payload.viewport.display_revision == self._display.revision
            )
        return super().raster_ready

    def _visible_count_limits(self) -> tuple[float, float] | None:
        payload = self._board_widget.visible_histogram_payload(
            _HISTOGRAM_PANEL_ID
        )
        return None if payload is None else payload.viewport.count_limits

    def _runtime_placeholders(self):
        return runtime_range_placeholders(
            self._visible_count_limits(),
            "count_min",
            "count_max",
        )

    def _sync_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        sync_revisioned_form_editors(
            (self._edit_display, self._setting_display),
            revision=self._display.revision,
            semantic_identity=self._display,
            values=histogram_display_form_values(self._display),
            runtime_placeholders=self._runtime_placeholders(),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    def _sync_committed_histogram_controls(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        """Finish ancillary Qt state without rolling back an admitted front."""

        try:
            self._sync_editors(
                accepted_editor=accepted_editor,
                accepted_base_revision=accepted_base_revision,
            )
            self._set_typed_controls_enabled(True)
        except BaseException as error:
            try:
                self._set_typed_controls_enabled(False)
            except BaseException:
                pass
            self._status.setText("HISTOGRAM CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _reload_editor(self, editor: FluentRevisionedFormEditor) -> None:
        if editor not in (self._edit_display, self._setting_display):
            raise ValueError("histogram editor does not belong to this window")
        editor.load(
            revision=self._display.revision,
            semantic_identity=self._display,
            values=histogram_display_form_values(self._display),
            runtime_placeholders=self._runtime_placeholders(),
        )

    def _set_typed_controls_enabled(self, enabled: bool) -> None:
        active = bool(enabled and self._view_family == "histogram")
        self._board_widget.set_interaction_readiness(
            image=False,
            curve=False,
            histogram=active,
        )
        self._settings_button.setEnabled(active)
        self._export_button.setEnabled(
            active and self._board_widget.front_frame is not None
        )
        self._interaction_switch.setEnabled(active)
        self._edit_display.setEnabled(active)
        self._setting_display.setEnabled(active)

    def _toggle_interaction(self, enabled: bool) -> None:
        if self._view_family != "histogram":
            return
        try:
            self._board_widget.set_selectors_enabled(bool(enabled))
        except BaseException as error:
            self._diagnostic.setText(error_summary(error))

    def _apply_display_form(
        self,
        editor: FluentRevisionedFormEditor,
        base_revision: int,
        values: object,
    ) -> None:
        if editor not in (self._edit_display, self._setting_display):
            raise ValueError("histogram editor does not belong to this window")
        try:
            if self._future is not None or self._closing:
                raise RuntimeError("histogram display work is already active")
            if base_revision != self._display.revision:
                raise RuntimeError(
                    f"histogram draft r{base_revision} is stale; "
                    f"current revision is r{self._display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("histogram display form must emit one exact mapping")
            candidate = histogram_display_from_form(
                self._display,
                values,
                current_count_limits=self._visible_count_limits(),
            )
            self._start_histogram_render(
                candidate,
                editor=editor,
                editor_revision=base_revision,
            )
        except BaseException as error:
            self._diagnostic.setText(
                f"Histogram display edit rejected: {error_summary(error)}"
            )

    def _accept_histogram_interaction(
        self,
        command: HistogramInteractionIntent,
    ) -> None:
        if not isinstance(
            command,
            (HistogramViewportCommit, HistogramRangeGesture),
        ):
            raise TypeError("unknown histogram interaction command")
        origin = command.origin
        if (
            origin.panel_id != _HISTOGRAM_PANEL_ID
            or self._board_widget.visible_histogram_origin(_HISTOGRAM_PANEL_ID)
            != origin
            or origin.presentation.panel_revision != self._display.revision
        ):
            raise RuntimeError("histogram interaction origin is stale")
        if isinstance(command, HistogramRangeGesture):
            self._board_widget.set_histogram_range_candidate(
                command.x_span,
                panel_id=_HISTOGRAM_PANEL_ID,
            )
            self._diagnostic.setText(
                ""
                if command.x_span is None
                else (
                    "DISPLAY ONLY value span "
                    f"{command.x_span[0]:.6g}..{command.x_span[1]:.6g}"
                )
            )
            return
        if command.viewport.display_revision != self._display.revision + 1:
            raise RuntimeError("histogram viewport commit must advance once")
        candidate = histogram_display_with_x_view(
            self._display,
            command.viewport.x_limits,
        )
        self._start_histogram_render(candidate, origin=origin)

    def _start_histogram_render(
        self,
        candidate: HistogramDisplayState,
        *,
        editor: FluentRevisionedFormEditor | None = None,
        editor_revision: int | None = None,
        origin: PanelInteractionOrigin | None = None,
    ) -> None:
        payload = self._board_widget.visible_histogram_payload(
            _HISTOGRAM_PANEL_ID
        )
        if self._view_family != "histogram" or payload is None:
            raise RuntimeError("typed histogram is not ready")
        if self._future is not None or self._closing:
            raise RuntimeError("histogram render is already active")
        if not isinstance(candidate, HistogramDisplayState):
            raise TypeError("candidate must be HistogramDisplayState")
        if candidate == self._display:
            if origin is not None:
                raise ValueError("histogram interaction cannot commit a no-op")
            self._sync_editors(
                accepted_editor=editor,
                accepted_base_revision=editor_revision,
            )
            return
        if candidate.revision != self._display.revision + 1:
            raise ValueError("histogram display revision must advance once")
        self._request_revision += 1
        self._active_kind = "histogram"
        self._pending_state = candidate
        self._pending_origin = origin
        self._pending_editor = editor
        self._pending_editor_revision = editor_revision
        self._status.setText("RENDERING HISTOGRAM")
        self._diagnostic.setText("")
        self._set_typed_controls_enabled(False)
        submitted = self._submit_future(
            self._histogram_renderer,
            candidate,
            payload.viewport.count_limits,
            payload.viewport.relim_mode,
            payload.viewport.count_scale,
            self._request_revision,
            self._memory_limit_bytes,
            self._cancelled,
        )
        if not submitted:
            self._discard_pending_histogram()

    def _discard_pending_histogram(self) -> None:
        origin = self._pending_origin
        self._pending_state = None
        self._pending_origin = None
        self._pending_editor = None
        self._pending_editor_revision = None
        self._active_kind = None
        cleanup_errors = []
        if origin is not None:
            try:
                self._board_widget.discard_pending_histogram_interaction(origin)
            except BaseException as error:
                cleanup_errors.append(error_summary(error))
        if self._view_family == "histogram":
            try:
                self._sync_editors()
                self._set_typed_controls_enabled(True)
            except BaseException as error:
                cleanup_errors.append(error_summary(error))
        if cleanup_errors:
            existing = self._diagnostic.text()
            suffix = "cleanup: " + " | ".join(cleanup_errors)
            self._diagnostic.setText(suffix if not existing else f"{existing} | {suffix}")

    def _present_histogram_front(
        self,
        front: _HistogramFigureFront,
        *,
        expected_state: HistogramDisplayState,
        request_revision: int,
    ) -> None:
        if front.required_peak_bytes > self._memory_limit_bytes:
            raise MemoryError("typed histogram front exceeds the window budget")
        request_revision = nonnegative_integer(
            request_revision,
            "histogram request revision",
        )
        if front.frame.sequence != request_revision:
            raise ValueError("histogram worker returned another request sequence")
        payload = front.frame.panels[0].display_payload
        assert isinstance(payload, HistogramPanelPayload)
        viewport = payload.viewport
        if (
            viewport.display_revision != expected_state.revision
            or viewport.count_scale is not expected_state.count_scale
            or viewport.relim_mode is not expected_state.relim_mode
            or viewport.bin_count != expected_state.bin_count
            or viewport.x_limits_are_auto != (expected_state.x_view is None)
            or (
                expected_state.x_view is not None
                and viewport.x_limits != expected_state.x_view
            )
            or (
                expected_state.relim_mode is RelimMode.FIXED
                and viewport.count_limits != expected_state.fixed_count_limits
            )
        ):
            raise ValueError("histogram worker returned conflicting authored state")

        contract = _histogram_front_contract(front)
        expected_contract = self._histogram_contract
        if expected_contract is not None:
            expected_identity, expected_series = expected_contract
            identity, series = contract
            if identity != expected_identity:
                raise ValueError("histogram worker changed frozen source provenance")
            if len(series) != len(expected_series) or any(
                actual is not expected
                for actual, expected in zip(series, expected_series, strict=True)
            ):
                raise ValueError("histogram worker changed frozen evaluated series")

        self._board_widget.present(front.frame)
        if expected_contract is None:
            self._histogram_contract = contract
        if not self._typed_pages_admitted:
            self._retire_tab_pages()
            self._tabs.addTab(self._histogram_page, "Histogram")
            self._tabs.addTab(self._edit_display, "Edit")
            self._tabs.tabBar().setVisible(True)
            self._histogram_page.show()
            self._edit_display.show()
            for widget in (
                self._interaction_switch,
                self._settings_button,
                self._export_button,
            ):
                widget.show()
            self._typed_pages_admitted = True
        self._view_family = "histogram"
        self._mode.setText("EXACT HISTOGRAM · INTERACTIVE · DISPLAY ONLY")
        self._status.setText("READY")
        self._summary.setText(front.summary)
        self._diagnostic.setText("")

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        try:
            result = future.result()
        except CancelledError:
            if not self._closing:
                self._status.setText("FIGURE CANCELLED")
                if kind == "histogram":
                    self._discard_pending_histogram()
                else:
                    self._active_kind = None
        except BaseException as error:
            if not self._closing:
                self._reject_completed_work(kind, error)
        else:
            if self._closing:
                return
            try:
                self._accept_completed_work(kind, result)
            except BaseException as error:
                self._reject_completed_work(kind, error)

    def _accept_completed_work(self, kind: str | None, result: object) -> None:
        if kind == "initial":
            if isinstance(result, EncodedRasterDocument):
                self._view_family = "encoded"
                self._set_typed_controls_enabled(False)
                self._mode.setText("FROZEN DATA FIGURE · DISPLAY ONLY")
                self._present_bundle(result)
            elif isinstance(result, _HistogramFigureFront):
                self._present_histogram_front(
                    result,
                    expected_state=self._display,
                    request_revision=self._request_revision,
                )
                self._sync_committed_histogram_controls()
            else:
                raise TypeError("initial figure worker returned another result")
            self._active_kind = None
            return
        if kind == "histogram":
            if not isinstance(result, _HistogramFigureFront):
                raise TypeError("histogram worker returned another result")
            pending = self._pending_state
            editor = self._pending_editor
            editor_revision = self._pending_editor_revision
            if pending is None:
                raise RuntimeError("histogram worker completed without pending state")
            self._present_histogram_front(
                result,
                expected_state=pending,
                request_revision=self._request_revision,
            )
            self._display = pending
            self._pending_state = None
            self._pending_origin = None
            self._pending_editor = None
            self._pending_editor_revision = None
            self._active_kind = None
            self._sync_committed_histogram_controls(
                accepted_editor=editor,
                accepted_base_revision=editor_revision,
            )
            return
        if kind == "export":
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("histogram export returned another result")
            revision, destination = result
            if revision != self._request_revision:
                raise ValueError("histogram export revision is stale")
            self._active_kind = None
            self._status.setText("READY")
            self._diagnostic.setText(f"Exported {destination}")
            try:
                self._set_typed_controls_enabled(True)
            except BaseException as error:
                self._status.setText("HISTOGRAM CONTROLS FAILED")
                self._diagnostic.setText(
                    f"Exported {destination} | {error_summary(error)}"
                )
            return
        raise RuntimeError("figure window completed unknown work")

    def _reject_completed_work(
        self,
        kind: str | None,
        error: BaseException,
    ) -> None:
        if kind == "histogram":
            self._status.setText("HISTOGRAM DISPLAY FAILED")
            self._diagnostic.setText(error_summary(error))
            self._discard_pending_histogram()
        elif kind == "export":
            self._status.setText("HISTOGRAM EXPORT FAILED")
            self._diagnostic.setText(error_summary(error))
            self._active_kind = None
            self._set_typed_controls_enabled(True)
        else:
            self._status.setText("FIGURE FAILED")
            self._summary.setText("No raster was admitted")
            self._diagnostic.setText(error_summary(error))
            self._active_kind = None

    def _choose_export(self) -> None:
        if (
            self._future is not None
            or self._closing
            or self._view_family != "histogram"
            or self._board_widget.front_frame is None
        ):
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export current histogram view",
            "histogram.png",
            "PNG image (*.png)",
        )
        if path:
            destination = Path(path)
            if destination.suffix.lower() != ".png":
                destination = destination.with_suffix(".png")
            self._start_export(destination)

    def _start_export(self, destination: Path) -> None:
        frame = self._board_widget.front_frame
        if self._future is not None or self._closing or frame is None:
            return
        self._request_revision += 1
        self._active_kind = "export"
        self._status.setText("EXPORTING HISTOGRAM")
        self._diagnostic.setText("")
        self._set_typed_controls_enabled(False)
        if not self._submit_future(
            _export_histogram_png,
            frame,
            Path(destination),
            self._request_revision,
            self._cancelled,
            self._export_commit_lock,
        ):
            self._active_kind = None
            self._set_typed_controls_enabled(True)

    def _clear_bundle(self) -> None:
        super()._clear_bundle()
        if self._view_family == "histogram":
            self._board_widget.clear()

    def _finish_close_if_ready(self) -> None:
        if self._closing and self._future is None and not self._closed:
            self._histogram_renderer = None
            self._histogram_contract = None
        super()._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        cancel_export_commits(
            cancelled=self._cancelled,
            commit_lock=self._export_commit_lock,
        )
        super().shutdown()


def _figure_window_factory(loader, *, memory_limit_bytes: int):
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    worker_thread_id: int | None = None
    cached_histogram: DataFigure | None = None

    def require_worker_owner() -> None:
        nonlocal worker_thread_id
        current = threading.get_ident()
        if worker_thread_id is None:
            worker_thread_id = current
        elif worker_thread_id != current:
            raise RuntimeError("figure view session changed worker thread")

    def initial(
        state: HistogramDisplayState,
        memory_limit: int,
        sequence: int,
        cancelled: threading.Event,
    ):
        nonlocal cached_histogram
        require_worker_owner()
        _require_not_cancelled(cancelled)
        figure = loader()
        if not isinstance(figure, DataFigure):
            raise TypeError("figure loader must return DataFigure")
        if _single_histogram_panel(figure):
            if _histogram_front_required_peak_bytes(
                figure,
                state,
            ) > _figure_render_limit(figure, memory_limit):
                return _encoded_figure(figure, memory_limit, cancelled)
            cached_histogram = figure
            return _render_histogram_front(
                figure,
                state,
                current_count_limits=None,
                previous_relim_mode=None,
                previous_count_scale=None,
                sequence=sequence,
                memory_limit_bytes=memory_limit,
                cancelled=cancelled,
            )
        return _encoded_figure(figure, memory_limit, cancelled)

    def rerender(
        state: HistogramDisplayState,
        current_count_limits,
        previous_relim_mode,
        previous_count_scale,
        sequence: int,
        memory_limit: int,
        cancelled: threading.Event,
    ) -> _HistogramFigureFront:
        require_worker_owner()
        figure = cached_histogram
        if figure is None:
            raise RuntimeError("typed histogram session has no frozen DataFigure")
        return _render_histogram_front(
            figure,
            state,
            current_count_limits=current_count_limits,
            previous_relim_mode=previous_relim_mode,
            previous_count_scale=previous_count_scale,
            sequence=sequence,
            memory_limit_bytes=memory_limit,
            cancelled=cancelled,
        )

    return lambda: DataFigureWindow(
            initial,
            rerender,
            memory_limit_bytes=limit,
        )


def open_data_figure_workbench(
    figure: DataFigure,
    *,
    memory_limit_bytes: int = _DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES,
) -> DataFigureWindow:
    """Open an already-resolved DataFigure on the shared raster lane."""

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    return open_workbench_window(
        _figure_window_factory(
            lambda: figure,
            memory_limit_bytes=memory_limit_bytes,
        )
    )


def open_figure_workbench(
    figure_factory,
    source,
    *,
    intent=None,
    selection=None,
    preferences=None,
    occupancy_output=None,
    memory_limit_bytes: int = _DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES,
) -> DataFigureWindow:
    """Resolve and display a current artifact entirely on the bounded worker."""

    if not callable(figure_factory):
        raise TypeError("figure_factory must be callable")
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    options = {
        "intent": intent,
        "selection": selection,
        "preferences": preferences,
        "memory_limit_bytes": limit,
    }
    if occupancy_output is not None:
        options["occupancy_output"] = occupancy_output
    return open_workbench_window(
        _figure_window_factory(
            lambda: figure_factory(source, **options),
            memory_limit_bytes=limit,
        )
    )


__all__ = [
    "DataFigureWindow",
    "open_data_figure_workbench",
    "open_figure_workbench",
]
