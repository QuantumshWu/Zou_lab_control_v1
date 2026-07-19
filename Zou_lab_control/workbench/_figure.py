"""Nonblocking Qt viewer for one frozen current :class:`DataFigure`.

The generic fallback remains an immutable encoded board.  The two earned
numeric products -- one logical CURVE or HISTOGRAM panel with one or more
series -- share one typed board, one Setting/Edit projection, and one
render/export lifecycle.  No whole-board PNG is reverse-mapped into data.
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
    CurvePanelPayload,
    DataFigure,
    HistogramPanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    SourceIdentity,
)
from zlc_frontend.encoded_raster import EncodedRasterDocument, EncodedRasterPage
from zlc_frontend.curve_display import (
    CurveDisplayState,
    curve_display_form_spec,
    curve_display_form_values,
    curve_display_from_form,
    curve_home_x_limits,
    curve_display_with_x_view,
    numeric_curve_coordinates,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import EvaluatedCurve, EvaluatedHistogram, ViewIntent
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
    CurveInteractionIntent,
    CurveRangeGesture,
    CurveViewportCommit,
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
_NUMERIC_BOARD_ID = "generic-numeric-figure"
_NUMERIC_PANEL_ID = "generic-numeric"
_NUMERIC_RASTER_SIZE = (800, 520)
_NUMERIC_JOIN_SCHEMA_DIGEST = canonical_digest(
    {
        "schema": "zlc_frontend.FrozenNumericFigureJoin",
        "fields": ("document", "input", "intent"),
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


def _classify_single_numeric(
    figure: DataFigure,
) -> tuple[ViewIntent | None, str | None]:
    """Return the typed intent or one explicit encoded-fallback reason."""

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    document = figure.document
    evaluated = figure.evaluated
    if (
        len(document.layers) != 1
        or len(evaluated.layers) != 1
        or len(evaluated.layers[0].cells) != 1
        or len(evaluated.inputs) != 1
    ):
        return None, "typed interaction requires exactly one layer, cell, and input"
    intent = document.layers[0].view.intent
    if figure.has_fit_overlays:
        return None, "authoritative fit overlays require whole-figure rendering"
    series = evaluated.layers[0].cells[0].series
    if not series:
        return None, "typed interaction requires at least one evaluated series"
    if intent is ViewIntent.HISTOGRAM and all(
        isinstance(item.data, EvaluatedHistogram) for item in series
    ):
        return intent, None
    if intent is not ViewIntent.CURVE:
        return None, f"{intent.value} is outside the current numeric interaction slice"
    first = series[0].data
    if not isinstance(first, EvaluatedCurve):
        return None, "CURVE intent did not evaluate to homogeneous curve series"
    try:
        numeric_curve_coordinates(first.x_axis)
    except (TypeError, ValueError) as error:
        return None, str(error)
    for index in range(1, len(series)):
        curve = series[index].data
        if not isinstance(curve, EvaluatedCurve):
            return None, "CURVE intent did not evaluate to homogeneous curve series"
        if curve.x_axis != first.x_axis or curve.value_unit != first.value_unit:
            return None, "curve series do not share one exact x axis and value unit"
    return intent, None


def _encoded_figure(
    figure: DataFigure,
    memory_limit_bytes: int,
    cancelled: threading.Event | None,
    *,
    unavailable_reason: str | None = None,
) -> EncodedRasterDocument:
    _require_not_cancelled(cancelled)
    render_limit = _figure_render_limit(figure, memory_limit_bytes)
    payload = figure.to_png_bytes(memory_limit_bytes=render_limit)
    _require_not_cancelled(cancelled)
    summary = _figure_summary(figure)
    if unavailable_reason is not None:
        if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
            raise ValueError("unavailable_reason must be non-empty text or None")
        summary = f"{summary} · interaction unavailable: {unavailable_reason.strip()}"
    document = EncodedRasterDocument(
        summary,
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
    """Retain the exact encoded fallback used by current fit and figure views."""

    _require_not_cancelled(cancelled)
    figure = loader()
    if not isinstance(figure, DataFigure):
        raise TypeError("figure loader must return DataFigure")
    return _encoded_figure(figure, memory_limit_bytes, cancelled)


_NumericDisplayState = CurveDisplayState | HistogramDisplayState
_NumericPanelPayload = CurvePanelPayload | HistogramPanelPayload


def _state_intent(state: _NumericDisplayState) -> ViewIntent:
    if isinstance(state, CurveDisplayState):
        return ViewIntent.CURVE
    if isinstance(state, HistogramDisplayState):
        return ViewIntent.HISTOGRAM
    raise TypeError("numeric display state must be CURVE or HISTOGRAM")


def _default_numeric_state(intent: ViewIntent) -> _NumericDisplayState:
    if intent is ViewIntent.CURVE:
        return CurveDisplayState()
    if intent is ViewIntent.HISTOGRAM:
        return HistogramDisplayState()
    raise ValueError("typed numeric intent must be CURVE or HISTOGRAM")


def _numeric_form_spec(state: _NumericDisplayState):
    return (
        curve_display_form_spec()
        if isinstance(state, CurveDisplayState)
        else histogram_display_form_spec()
    )


def _numeric_form_values(state: _NumericDisplayState) -> dict[str, object]:
    if isinstance(state, CurveDisplayState):
        return curve_display_form_values(state)
    if isinstance(state, HistogramDisplayState):
        return histogram_display_form_values(state)
    raise TypeError("numeric display state must be CURVE or HISTOGRAM")


def _numeric_state_from_form(
    state: _NumericDisplayState,
    values: dict[str, object],
    *,
    current_value_limits: tuple[float, float] | None,
) -> _NumericDisplayState:
    if isinstance(state, CurveDisplayState):
        return curve_display_from_form(
            state,
            values,
            current_y_limits=current_value_limits,
        )
    if isinstance(state, HistogramDisplayState):
        return histogram_display_from_form(
            state,
            values,
            current_count_limits=current_value_limits,
        )
    raise TypeError("numeric display state must be CURVE or HISTOGRAM")


def _numeric_state_with_x_view(
    state: _NumericDisplayState,
    x_view: tuple[float, float] | None,
) -> _NumericDisplayState:
    if isinstance(state, CurveDisplayState):
        return curve_display_with_x_view(state, x_view)
    if isinstance(state, HistogramDisplayState):
        return histogram_display_with_x_view(state, x_view)
    raise TypeError("numeric display state must be CURVE or HISTOGRAM")


def _payload_intent(payload: _NumericPanelPayload) -> ViewIntent:
    if isinstance(payload, CurvePanelPayload):
        return ViewIntent.CURVE
    if isinstance(payload, HistogramPanelPayload):
        return ViewIntent.HISTOGRAM
    raise TypeError("numeric payload must be CURVE or HISTOGRAM")


@dataclass(frozen=True, slots=True)
class _NumericFigureFront:
    intent: ViewIntent
    state: _NumericDisplayState
    summary: str
    frame: BoardFrame
    required_peak_bytes: int
    effective_limit_bytes: int

    def __post_init__(self) -> None:
        if self.intent not in (ViewIntent.CURVE, ViewIntent.HISTOGRAM):
            raise ValueError("numeric figure front has another intent")
        if _state_intent(self.state) is not self.intent:
            raise ValueError("numeric figure front state belongs to another intent")
        if not isinstance(self.summary, str) or not self.summary:
            raise ValueError("numeric figure summary must be non-empty")
        if not isinstance(self.frame, BoardFrame) or len(self.frame.panels) != 1:
            raise TypeError("numeric figure front requires one BoardFrame panel")
        panel = self.frame.panels[0]
        payload = panel.display_payload
        if (
            panel.panel_id != _NUMERIC_PANEL_ID
            or not isinstance(payload, (CurvePanelPayload, HistogramPanelPayload))
            or _payload_intent(payload) is not self.intent
        ):
            raise ValueError("numeric figure front has another typed payload")
        raster = panel.raster
        if (raster.width, raster.height) != _NUMERIC_RASTER_SIZE:
            raise ValueError("numeric figure front has another raster geometry")
        if raster.stride_bytes != raster.width * 4:
            raise ValueError("numeric figure front requires packed RGBA")
        required = positive_integer(
            self.required_peak_bytes,
            "required_peak_bytes",
        )
        effective = positive_integer(
            self.effective_limit_bytes,
            "effective_limit_bytes",
        )
        if required > effective:
            raise MemoryError("numeric figure front exceeds its frozen budget")
        object.__setattr__(self, "required_peak_bytes", required)
        object.__setattr__(self, "effective_limit_bytes", effective)


def _numeric_join_digest(figure: DataFigure, intent: ViewIntent) -> str:
    evaluated = figure.evaluated
    source = evaluated.inputs[0]
    return canonical_digest(
        {
            "schema": "zlc_frontend.FrozenNumericFigureJoin",
            "document": {
                "id": figure.document.document_id,
                "revision": figure.document.revision,
            },
            "intent": intent.value,
            "input": {
                "dataset_id": source.dataset_id.value,
                "ref": dataset_revision_ref_to_tree(source.ref),
            },
        }
    )


def _numeric_front_contract(
    front: _NumericFigureFront,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Freeze data/provenance identity while excluding display revision."""

    if not isinstance(front, _NumericFigureFront):
        raise TypeError("front must be _NumericFigureFront")
    frame = front.frame
    panel = frame.panels[0]
    payload = panel.display_payload
    assert isinstance(payload, (CurvePanelPayload, HistogramPanelPayload))
    stamp = panel.coherence_stamp
    if len(stamp.presentations) != 1:
        raise ValueError("generic numeric front requires one presentation identity")
    presentation = stamp.presentations[0]
    if presentation.panel_id != panel.panel_id:
        raise ValueError("numeric presentation names another panel")
    identity = (
        front.intent,
        front.effective_limit_bytes,
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


def _numeric_front_required_peak_bytes(
    figure: DataFigure,
    state: _NumericDisplayState,
) -> int:
    intent, unavailable_reason = _classify_single_numeric(figure)
    if intent is None or intent is not _state_intent(state):
        raise ValueError(
            "typed numeric budget requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    from zlc_frontend.matplotlib_render import (
        estimate_live_panel_raster_peak_nbytes,
        evaluated_figure_array_nbytes,
    )

    width, height = _NUMERIC_RASTER_SIZE
    series_count = len(figure.evaluated.layers[0].cells[0].series)
    options = {
        "evaluated_data_upper_bound_bytes": evaluated_figure_array_nbytes(
            figure.evaluated
        ),
        # Admission covers the currently painted Qt/held front while the next
        # immutable worker front is composed and admitted.
        "extra_retained_fronts": 1,
    }
    if isinstance(state, HistogramDisplayState):
        options.update(
            histogram_bins=state.bin_count,
            histogram_series_count=series_count,
        )
    return estimate_live_panel_raster_peak_nbytes(width, height, **options)


def _render_numeric_front(
    figure: DataFigure,
    state: _NumericDisplayState,
    *,
    current_value_limits: tuple[float, float] | None,
    previous_relim_mode,
    previous_count_scale: HistogramCountScale | None,
    sequence: int,
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> _NumericFigureFront:
    intent, unavailable_reason = _classify_single_numeric(figure)
    if intent is None or intent is not _state_intent(state):
        raise ValueError(
            "typed numeric render requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    _require_not_cancelled(cancelled)

    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    width, height = _NUMERIC_RASTER_SIZE
    required = _numeric_front_required_peak_bytes(figure, state)
    effective_limit = _figure_render_limit(figure, memory_limit_bytes)
    if required > effective_limit:
        raise MemoryError(
            f"interactive {intent.value.lower()} requires {required} bytes; "
            f"limit is {effective_limit}"
        )

    renderer = SinglePanelAggRenderer(
        figure.document,
        width=width,
        height=height,
    )
    try:
        if isinstance(state, CurveDisplayState):
            raster, payload = renderer.render_interactive_curve(
                figure.evaluated,
                state,
                current_y_limits=current_value_limits,
                previous_relim_mode=previous_relim_mode,
            )
        else:
            raster, payload = renderer.render_interactive_histogram(
                figure.evaluated,
                state,
                current_count_limits=current_value_limits,
                previous_relim_mode=previous_relim_mode,
                previous_count_scale=previous_count_scale,
            )
    finally:
        renderer.close()
    _require_not_cancelled(cancelled)

    evaluated_input = payload.evaluated_input
    presentation = PanelPresentationIdentity(
        _NUMERIC_PANEL_ID,
        figure.document.document_id,
        figure.document.revision,
        0,
        state.revision,
    )
    ref = evaluated_input.ref
    stamp = CoherenceStamp(
        f"figure:{ref.block_id.value}",
        ref.stream_generation.value,
        "FrozenNumericFigureJoin",
        _NUMERIC_JOIN_SCHEMA_DIGEST,
        _numeric_join_digest(figure, intent),
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
        _NUMERIC_BOARD_ID,
        0,
        sequence,
        (
            PanelFrame(
                _NUMERIC_PANEL_ID,
                f"frozen-{intent.value.lower()}",
                source,
                stamp,
                raster,
                payload,
            ),
        ),
    )
    return _NumericFigureFront(
        intent,
        state,
        _figure_summary(figure),
        frame,
        required,
        effective_limit,
    )


def _export_numeric_png(
    frame: BoardFrame,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> tuple[int, Path]:
    if not isinstance(frame, BoardFrame) or len(frame.panels) != 1:
        raise TypeError("numeric export requires one exact BoardFrame")
    panel = frame.panels[0]
    if panel.panel_id != _NUMERIC_PANEL_ID or not isinstance(
        panel.display_payload,
        (CurvePanelPayload, HistogramPanelPayload),
    ):
        raise ValueError("numeric export frame has another presentation")
    raster = panel.raster
    if raster.stride_bytes != raster.width * 4:
        raise ValueError("numeric export requires a packed RGBA raster")

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
    """Frozen generic viewer with one closed CURVE/HISTOGRAM front."""

    def __init__(
        self,
        initial_loader,
        numeric_renderer,
        *,
        memory_limit_bytes: int,
    ) -> None:
        if not callable(initial_loader) or not callable(numeric_renderer):
            raise TypeError("figure worker callables must be callable")
        self._numeric_renderer = numeric_renderer
        self._view_family: str | None = None
        self._display: _NumericDisplayState | None = None
        self._numeric_contract: (
            tuple[tuple[object, ...], tuple[object, ...]] | None
        ) = None
        self._typed_pages_admitted = False
        self._numeric_ui_faulted = False
        self._request_revision = 0
        self._active_kind: str | None = "initial"
        self._pending_state: _NumericDisplayState | None = None
        self._pending_origin: PanelInteractionOrigin | None = None
        self._pending_editor: FluentRevisionedFormEditor | None = None
        self._pending_editor_revision: int | None = None
        self._edit_display: FluentRevisionedFormEditor | None = None
        self._setting_display: FluentRevisionedFormEditor | None = None
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

        self._numeric_page = QtWidgets.QWidget(self._tabs)
        self._numeric_page.hide()
        page_layout = QtWidgets.QVBoxLayout(self._numeric_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self._board_widget = QtRasterBoard(
            (_NUMERIC_PANEL_ID,),
            self._numeric_page,
            columns=1,
            empty_text="Resolving exact numeric figure…",
        )
        self._board_widget.setObjectName("figureViewerNumericBoard")
        self._board_widget.setMinimumSize(480, 320)
        page_layout.addWidget(self._board_widget, 1)

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("figureViewerNumericSettingsPopup")
        self._settings_popup_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        self._interaction_switch = FluentSwitch("Interact", self)
        self._interaction_switch.setObjectName("figureViewerNumericInteractSwitch")
        self._interaction_switch.setChecked(True)
        self._settings_button = FluentButton("Setting…", self, color=GREY)
        self._settings_button.setObjectName("figureViewerNumericSettingButton")
        self._export_button = FluentButton("Export PNG…", self, color=ORANGE)
        self._export_button.setObjectName("figureViewerNumericExportButton")
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
        self._set_typed_controls_enabled(False)
        self._submit_future(
            initial_loader,
            self._memory_limit_bytes,
            self._request_revision,
            self._cancelled,
        )

    @property
    def raster_ready(self) -> bool:
        if self._view_family in ("curve", "histogram"):
            display = self._display
            payload = self._visible_numeric_payload()
            return bool(
                display is not None
                and self._board_widget.front_frame is not None
                and self._pending_state is None
                and payload is not None
                and payload.viewport.display_revision == display.revision
            )
        return super().raster_ready

    def _visible_numeric_payload(self) -> _NumericPanelPayload | None:
        if self._view_family == "curve":
            payload = self._board_widget.visible_curve_payload(_NUMERIC_PANEL_ID)
        elif self._view_family == "histogram":
            payload = self._board_widget.visible_histogram_payload(_NUMERIC_PANEL_ID)
        else:
            return None
        if payload is not None:
            return payload
        # A valid front is admitted before optional interaction controls.  If
        # their construction fails there is deliberately no binding, but the
        # exact current raster/payload remains visible and ready.
        frame = self._board_widget.front_frame
        if frame is None or len(frame.panels) != 1:
            return None
        candidate = frame.panels[0].display_payload
        expected_type = (
            CurvePanelPayload
            if self._view_family == "curve"
            else HistogramPanelPayload
        )
        return candidate if isinstance(candidate, expected_type) else None

    def _visible_numeric_origin(self) -> PanelInteractionOrigin | None:
        if self._view_family == "curve":
            return self._board_widget.visible_curve_origin(_NUMERIC_PANEL_ID)
        if self._view_family == "histogram":
            return self._board_widget.visible_histogram_origin(_NUMERIC_PANEL_ID)
        return None

    def _visible_value_limits(self) -> tuple[float, float] | None:
        payload = self._visible_numeric_payload()
        if isinstance(payload, CurvePanelPayload):
            return payload.viewport.y_limits
        if isinstance(payload, HistogramPanelPayload):
            return payload.viewport.count_limits
        return None

    def _runtime_placeholders(self):
        display = self._display
        if isinstance(display, CurveDisplayState):
            fields = ("y_min", "y_max")
        elif isinstance(display, HistogramDisplayState):
            fields = ("count_min", "count_max")
        else:
            return {}
        return runtime_range_placeholders(self._visible_value_limits(), *fields)

    def _ensure_numeric_controls(self, state: _NumericDisplayState) -> None:
        if self._edit_display is not None or self._setting_display is not None:
            if self._display is not None and _state_intent(self._display) is not _state_intent(state):
                raise RuntimeError("numeric window cannot change typed family")
            return
        if isinstance(state, CurveDisplayState):
            runtime_fields = ("y_min", "y_max")
            subject = "curve display"
            bind = self._board_widget.bind_curve_interaction
        else:
            runtime_fields = ("count_min", "count_max")
            subject = "histogram display"
            bind = self._board_widget.bind_histogram_interaction
        spec = _numeric_form_spec(state)
        edit = None
        setting = None
        try:
            edit = FluentRevisionedFormEditor(
                spec,
                subject,
                runtime_placeholder_fields=runtime_fields,
                parent=self._tabs,
            )
            setting = FluentRevisionedFormEditor(
                spec,
                subject,
                runtime_placeholder_fields=runtime_fields,
                parent=self._settings_popup,
            )
            edit.setObjectName("figureViewerNumericEditEditor")
            setting.setObjectName("figureViewerNumericSettingEditor")
            edit.hide()
            edit.applyRequested.connect(
                lambda revision, values: self._apply_display_form(
                    edit,
                    revision,
                    values,
                )
            )
            setting.applyRequested.connect(
                lambda revision, values: self._apply_display_form(
                    setting,
                    revision,
                    values,
                )
            )
            edit.cancelRequested.connect(lambda: self._reload_editor(edit))
            setting.cancelRequested.connect(lambda: self._reload_editor(setting))
            self._settings_popup_layout.addWidget(setting)
            bind(_NUMERIC_PANEL_ID, self._accept_numeric_interaction, enabled=True)
        except BaseException:
            if setting is not None:
                self._settings_popup_layout.removeWidget(setting)
                setting.setParent(None)
                setting.deleteLater()
            if edit is not None:
                edit.setParent(None)
                edit.deleteLater()
            raise
        self._edit_display = edit
        self._setting_display = setting

    def _editors(self) -> tuple[FluentRevisionedFormEditor, FluentRevisionedFormEditor]:
        if self._edit_display is None or self._setting_display is None:
            raise RuntimeError("numeric controls are not admitted")
        return self._edit_display, self._setting_display

    def _sync_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        display = self._display
        if display is None:
            raise RuntimeError("numeric display state is not admitted")
        sync_revisioned_form_editors(
            self._editors(),
            revision=display.revision,
            semantic_identity=display,
            values=_numeric_form_values(display),
            runtime_placeholders=self._runtime_placeholders(),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    def _sync_committed_numeric_controls(
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
            self._numeric_ui_faulted = True
            try:
                self._set_typed_controls_enabled(False)
            except BaseException:
                pass
            self._status.setText("NUMERIC CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _reload_editor(self, editor: FluentRevisionedFormEditor) -> None:
        if editor not in self._editors():
            raise ValueError("numeric editor does not belong to this window")
        display = self._display
        if display is None:
            raise RuntimeError("numeric display state is not admitted")
        editor.load(
            revision=display.revision,
            semantic_identity=display,
            values=_numeric_form_values(display),
            runtime_placeholders=self._runtime_placeholders(),
        )

    def _set_typed_controls_enabled(self, enabled: bool) -> None:
        active = bool(
            enabled
            and not self._numeric_ui_faulted
            and self._view_family in ("curve", "histogram")
        )
        self._board_widget.set_interaction_readiness(
            image=False,
            curve=active and self._view_family == "curve",
            histogram=active and self._view_family == "histogram",
        )
        self._settings_button.setEnabled(active)
        self._export_button.setEnabled(
            active and self._board_widget.front_frame is not None
        )
        self._interaction_switch.setEnabled(active)
        for editor in (self._edit_display, self._setting_display):
            if editor is not None:
                editor.setEnabled(active)

    def _toggle_interaction(self, enabled: bool) -> None:
        if self._view_family not in ("curve", "histogram"):
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
        if editor not in self._editors():
            raise ValueError("numeric editor does not belong to this window")
        try:
            display = self._display
            if display is None:
                raise RuntimeError("numeric display state is not admitted")
            if self._future is not None or self._closing:
                raise RuntimeError("numeric display work is already active")
            if base_revision != display.revision:
                raise RuntimeError(
                    f"numeric draft r{base_revision} is stale; "
                    f"current revision is r{display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("numeric display form must emit one exact mapping")
            candidate = _numeric_state_from_form(
                display,
                values,
                current_value_limits=self._visible_value_limits(),
            )
            self._start_numeric_render(
                candidate,
                editor=editor,
                editor_revision=base_revision,
            )
        except BaseException as error:
            self._diagnostic.setText(
                f"Numeric display edit rejected: {error_summary(error)}"
            )

    def _accept_numeric_interaction(
        self,
        command: CurveInteractionIntent | HistogramInteractionIntent,
    ) -> None:
        display = self._display
        is_curve = isinstance(command, (CurveViewportCommit, CurveRangeGesture))
        is_histogram = isinstance(
            command,
            (HistogramViewportCommit, HistogramRangeGesture),
        )
        if not (is_curve or is_histogram):
            raise TypeError("unknown numeric interaction command")
        if (
            display is None
            or is_curve != isinstance(display, CurveDisplayState)
            or is_histogram != isinstance(display, HistogramDisplayState)
        ):
            raise RuntimeError("numeric interaction belongs to another family")
        origin = command.origin
        if (
            origin.panel_id != _NUMERIC_PANEL_ID
            or self._visible_numeric_origin() != origin
            or origin.presentation.panel_revision != display.revision
        ):
            raise RuntimeError("numeric interaction origin is stale")
        if isinstance(command, (CurveRangeGesture, HistogramRangeGesture)):
            setter = (
                self._board_widget.set_curve_range_candidate
                if is_curve
                else self._board_widget.set_histogram_range_candidate
            )
            setter(command.x_span, panel_id=_NUMERIC_PANEL_ID)
            self._diagnostic.setText(
                ""
                if command.x_span is None
                else (
                    "DISPLAY ONLY x span "
                    f"{command.x_span[0]:.6g}..{command.x_span[1]:.6g}"
                )
            )
            return
        if command.viewport.display_revision != display.revision + 1:
            raise RuntimeError("numeric viewport commit must advance once")
        self._start_numeric_render(
            _numeric_state_with_x_view(display, command.viewport.x_limits),
            origin=origin,
        )

    def _start_numeric_render(
        self,
        candidate: _NumericDisplayState,
        *,
        editor: FluentRevisionedFormEditor | None = None,
        editor_revision: int | None = None,
        origin: PanelInteractionOrigin | None = None,
    ) -> None:
        display = self._display
        payload = self._visible_numeric_payload()
        if display is None or payload is None:
            raise RuntimeError("typed numeric figure is not ready")
        if self._future is not None or self._closing:
            raise RuntimeError("numeric render is already active")
        if _state_intent(candidate) is not _state_intent(display):
            raise TypeError("candidate belongs to another numeric family")
        if candidate == display:
            if origin is not None:
                raise ValueError("numeric interaction cannot commit a no-op")
            self._sync_editors(
                accepted_editor=editor,
                accepted_base_revision=editor_revision,
            )
            return
        if candidate.revision != display.revision + 1:
            raise ValueError("numeric display revision must advance once")
        self._request_revision += 1
        self._active_kind = "numeric"
        self._pending_state = candidate
        self._pending_origin = origin
        self._pending_editor = editor
        self._pending_editor_revision = editor_revision
        self._status.setText(f"RENDERING {self._view_family.upper()}")
        self._diagnostic.setText("")
        self._set_typed_controls_enabled(False)
        previous_scale = (
            display.count_scale
            if isinstance(display, HistogramDisplayState)
            else None
        )
        submitted = self._submit_future(
            self._numeric_renderer,
            candidate,
            self._visible_value_limits(),
            display.relim_mode,
            previous_scale,
            self._request_revision,
            self._memory_limit_bytes,
            self._cancelled,
        )
        if not submitted:
            self._discard_pending_numeric()

    def _discard_pending_numeric(self) -> None:
        origin = self._pending_origin
        family = self._view_family
        self._pending_state = None
        self._pending_origin = None
        self._pending_editor = None
        self._pending_editor_revision = None
        self._active_kind = None
        cleanup_errors = []
        if origin is not None:
            try:
                discard = (
                    self._board_widget.discard_pending_curve_interaction
                    if family == "curve"
                    else self._board_widget.discard_pending_histogram_interaction
                )
                discard(origin)
            except BaseException as error:
                cleanup_errors.append(error_summary(error))
        if family in ("curve", "histogram"):
            try:
                self._sync_editors()
                self._set_typed_controls_enabled(True)
            except BaseException as error:
                cleanup_errors.append(error_summary(error))
        if cleanup_errors:
            existing = self._diagnostic.text()
            suffix = "cleanup: " + " | ".join(cleanup_errors)
            self._diagnostic.setText(suffix if not existing else f"{existing} | {suffix}")

    @staticmethod
    def _validate_authored_front(
        front: _NumericFigureFront,
        expected_state: _NumericDisplayState,
    ) -> None:
        if front.state != expected_state or front.intent is not _state_intent(expected_state):
            raise ValueError("numeric worker returned conflicting authored state")
        payload = front.frame.panels[0].display_payload
        assert isinstance(payload, (CurvePanelPayload, HistogramPanelPayload))
        viewport = payload.viewport
        if viewport.display_revision != expected_state.revision:
            raise ValueError("numeric worker returned another display revision")
        if isinstance(expected_state, CurveDisplayState):
            assert isinstance(payload, CurvePanelPayload)
            expected_home = curve_home_x_limits(viewport.x_axis)
            expected_x = expected_state.x_view or expected_home
            if (
                viewport.home_x_limits != expected_home
                or viewport.x_limits != expected_x
                or (
                    expected_state.relim_mode is RelimMode.FIXED
                    and viewport.y_limits != expected_state.fixed_y_limits
                )
            ):
                raise ValueError("curve worker returned conflicting authored state")
            return
        assert isinstance(payload, HistogramPanelPayload)
        if (
            viewport.count_scale is not expected_state.count_scale
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

    def _present_numeric_front(
        self,
        front: _NumericFigureFront,
        *,
        expected_state: _NumericDisplayState,
        request_revision: int,
    ) -> None:
        if front.required_peak_bytes > self._memory_limit_bytes:
            raise MemoryError("typed numeric front exceeds the window budget")
        request_revision = nonnegative_integer(
            request_revision,
            "numeric request revision",
        )
        if front.frame.sequence != request_revision:
            raise ValueError("numeric worker returned another request sequence")
        self._validate_authored_front(front, expected_state)
        contract = _numeric_front_contract(front)
        expected_contract = self._numeric_contract
        if expected_contract is not None:
            expected_identity, expected_series = expected_contract
            identity, series = contract
            if identity != expected_identity:
                raise ValueError("numeric worker changed frozen source provenance")
            if len(series) != len(expected_series) or any(
                actual is not expected
                for actual, expected in zip(series, expected_series, strict=True)
            ):
                raise ValueError("numeric worker changed frozen evaluated series")

        self._board_widget.present(front.frame)
        # The admitted board front is the transaction boundary.  Commit the
        # exact authored state/contract before any optional Qt chrome work.
        if expected_contract is None:
            self._numeric_contract = contract
        self._display = expected_state
        self._view_family = front.intent.value.lower()
        # Page/chrome and controls are ancillary to the already-admitted
        # immutable data front.  Their faults can disable UI, never roll it back.
        try:
            if not self._typed_pages_admitted:
                self._retire_tab_pages()
                self._tabs.addTab(self._numeric_page, front.intent.value.title())
                self._tabs.tabBar().setVisible(False)
                self._numeric_page.show()
                self._typed_pages_admitted = True
            self._mode.setText(
                f"EXACT {front.intent.value} · INTERACTIVE · DISPLAY ONLY"
            )
            self._status.setText("READY")
            self._summary.setText(front.summary)
            self._diagnostic.setText("")
        except BaseException as error:
            self._numeric_ui_faulted = True
            self._set_typed_controls_enabled(False)
            self._status.setText("NUMERIC CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))
            return
        try:
            self._ensure_numeric_controls(expected_state)
            edit, _setting = self._editors()
            if self._tabs.indexOf(edit) < 0:
                self._tabs.addTab(edit, "Edit")
            self._tabs.tabBar().setVisible(True)
            edit.show()
            for widget in (
                self._interaction_switch,
                self._settings_button,
                self._export_button,
            ):
                widget.show()
        except BaseException as error:
            self._numeric_ui_faulted = True
            self._set_typed_controls_enabled(False)
            self._status.setText("NUMERIC CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        try:
            result = future.result()
        except CancelledError:
            if not self._closing:
                self._status.setText("FIGURE CANCELLED")
                if kind == "numeric":
                    self._discard_pending_numeric()
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
            elif isinstance(result, _NumericFigureFront):
                self._present_numeric_front(
                    result,
                    expected_state=_default_numeric_state(result.intent),
                    request_revision=self._request_revision,
                )
                if not self._numeric_ui_faulted:
                    self._sync_committed_numeric_controls()
            else:
                raise TypeError("initial figure worker returned another result")
            self._active_kind = None
            return
        if kind == "numeric":
            if not isinstance(result, _NumericFigureFront):
                raise TypeError("numeric worker returned another result")
            pending = self._pending_state
            editor = self._pending_editor
            editor_revision = self._pending_editor_revision
            if pending is None:
                raise RuntimeError("numeric worker completed without pending state")
            self._present_numeric_front(
                result,
                expected_state=pending,
                request_revision=self._request_revision,
            )
            self._pending_state = None
            self._pending_origin = None
            self._pending_editor = None
            self._pending_editor_revision = None
            self._active_kind = None
            if not self._numeric_ui_faulted:
                self._sync_committed_numeric_controls(
                    accepted_editor=editor,
                    accepted_base_revision=editor_revision,
                )
            return
        if kind == "export":
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("numeric export returned another result")
            revision, destination = result
            if revision != self._request_revision:
                raise ValueError("numeric export revision is stale")
            self._active_kind = None
            self._status.setText("READY")
            self._diagnostic.setText(f"Exported {destination}")
            try:
                self._set_typed_controls_enabled(True)
            except BaseException as error:
                self._numeric_ui_faulted = True
                self._status.setText("NUMERIC CONTROLS FAILED")
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
        if kind == "numeric":
            family = (self._view_family or "numeric").upper()
            self._status.setText(f"{family} DISPLAY FAILED")
            self._diagnostic.setText(error_summary(error))
            self._discard_pending_numeric()
        elif kind == "export":
            self._status.setText("NUMERIC EXPORT FAILED")
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
            or self._view_family not in ("curve", "histogram")
            or self._board_widget.front_frame is None
        ):
            return
        family = self._view_family
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Export current {family} view",
            f"{family}.png",
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
        self._status.setText(f"EXPORTING {self._view_family.upper()}")
        self._diagnostic.setText("")
        self._set_typed_controls_enabled(False)
        if not self._submit_future(
            _export_numeric_png,
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
        if self._view_family in ("curve", "histogram"):
            self._board_widget.clear()

    def _finish_close_if_ready(self) -> None:
        if self._closing and self._future is None and not self._closed:
            self._numeric_renderer = None
            self._numeric_contract = None
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
    cached_numeric: DataFigure | None = None

    def require_worker_owner() -> None:
        nonlocal worker_thread_id
        current = threading.get_ident()
        if worker_thread_id is None:
            worker_thread_id = current
        elif worker_thread_id != current:
            raise RuntimeError("figure view session changed worker thread")

    def initial(
        memory_limit: int,
        sequence: int,
        cancelled: threading.Event,
    ):
        nonlocal cached_numeric
        require_worker_owner()
        _require_not_cancelled(cancelled)
        figure = loader()
        if not isinstance(figure, DataFigure):
            raise TypeError("figure loader must return DataFigure")
        intent, unavailable_reason = _classify_single_numeric(figure)
        if intent is not None:
            state = _default_numeric_state(intent)
            if _numeric_front_required_peak_bytes(figure, state) > _figure_render_limit(
                figure,
                memory_limit,
            ):
                return _encoded_figure(
                    figure,
                    memory_limit,
                    cancelled,
                    unavailable_reason=(
                        f"interactive {intent.value} front exceeds the frozen memory budget"
                    ),
                )
            cached_numeric = figure
            return _render_numeric_front(
                figure,
                state,
                current_value_limits=None,
                previous_relim_mode=None,
                previous_count_scale=None,
                sequence=sequence,
                memory_limit_bytes=memory_limit,
                cancelled=cancelled,
            )
        return _encoded_figure(
            figure,
            memory_limit,
            cancelled,
            unavailable_reason=unavailable_reason,
        )

    def rerender(
        state: _NumericDisplayState,
        current_value_limits,
        previous_relim_mode,
        previous_count_scale,
        sequence: int,
        memory_limit: int,
        cancelled: threading.Event,
    ) -> _NumericFigureFront:
        require_worker_owner()
        figure = cached_numeric
        if figure is None:
            raise RuntimeError("typed numeric session has no frozen DataFigure")
        return _render_numeric_front(
            figure,
            state,
            current_value_limits=current_value_limits,
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
