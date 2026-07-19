"""Nonblocking Qt viewer for one frozen current :class:`DataFigure`.

The generic fallback remains an immutable encoded board.  The three earned
products -- one logical IMAGE, CURVE, or HISTOGRAM panel -- share one typed
board, one Setting/Edit projection, and one render/export lifecycle.  No
whole-board PNG is reverse-mapped into data.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, replace
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
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
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
from zlc_frontend.figure import (
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    ViewIntent,
)
from zlc_frontend.histogram_display import (
    HistogramCountScale,
    HistogramDisplayState,
    histogram_display_form_spec,
    histogram_display_form_values,
    histogram_display_from_form,
    histogram_display_with_x_view,
)
from zlc_frontend.image_display import (
    ImageDisplayState,
    image_display_form_spec,
    image_display_form_values,
    image_display_for_viewport,
    image_display_from_form,
    image_viewport_for_display_state,
)
from zlc_frontend.image_raster import (
    estimate_indexed8_raster_peak_nbytes,
    rasterize_image_indexed8,
)
from zlc_frontend.image_view import image_viewport_for_evaluated_image
from zlc_frontend.render_style import indexed_colormap
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
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportCommit,
    PanelInteractionOrigin,
    RectangleGesture,
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
_TYPED_BOARD_ID = "generic-typed-figure"
_TYPED_PANEL_ID = "generic-typed"
_NUMERIC_RASTER_SIZE = (800, 520)
_TYPED_JOIN_SCHEMA_DIGEST = canonical_digest(
    {
        "schema": "zlc_frontend.FrozenTypedFigureJoin",
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


def _classify_single_typed(
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
    if intent is ViewIntent.IMAGE:
        if len(series) != 1 or not isinstance(series[0].data, EvaluatedImage):
            return None, "IMAGE interaction requires exactly one evaluated image"
        if series[0].data.values.dtype.kind not in "biuf":
            return None, "IMAGE interaction requires real numeric evaluated values"
        try:
            image_viewport_for_evaluated_image(series[0].data)
        except (TypeError, ValueError) as error:
            return None, str(error)
        return intent, None
    if intent is ViewIntent.HISTOGRAM and all(
        isinstance(item.data, EvaluatedHistogram) for item in series
    ):
        return intent, None
    if intent is not ViewIntent.CURVE:
        return None, f"{intent.value} is outside the current typed interaction slice"
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


_TypedDisplayState = ImageDisplayState | CurveDisplayState | HistogramDisplayState
_TypedPanelPayload = ImagePanelPayload | CurvePanelPayload | HistogramPanelPayload


def _state_intent(state: _TypedDisplayState) -> ViewIntent:
    if isinstance(state, ImageDisplayState):
        return ViewIntent.IMAGE
    if isinstance(state, CurveDisplayState):
        return ViewIntent.CURVE
    if isinstance(state, HistogramDisplayState):
        return ViewIntent.HISTOGRAM
    raise TypeError("typed display state must be IMAGE, CURVE, or HISTOGRAM")


def _default_typed_state(intent: ViewIntent) -> _TypedDisplayState:
    if intent is ViewIntent.IMAGE:
        return ImageDisplayState()
    if intent is ViewIntent.CURVE:
        return CurveDisplayState()
    if intent is ViewIntent.HISTOGRAM:
        return HistogramDisplayState()
    raise ValueError("typed intent must be IMAGE, CURVE, or HISTOGRAM")


def _typed_form_spec(state: _TypedDisplayState):
    if isinstance(state, ImageDisplayState):
        return image_display_form_spec()
    if isinstance(state, CurveDisplayState):
        return curve_display_form_spec()
    if isinstance(state, HistogramDisplayState):
        return histogram_display_form_spec()
    raise TypeError("unknown typed display state")


def _typed_form_values(state: _TypedDisplayState) -> dict[str, object]:
    if isinstance(state, ImageDisplayState):
        return image_display_form_values(state)
    if isinstance(state, CurveDisplayState):
        return curve_display_form_values(state)
    if isinstance(state, HistogramDisplayState):
        return histogram_display_form_values(state)
    raise TypeError("unknown typed display state")


def _typed_state_from_form(
    state: _TypedDisplayState,
    values: dict[str, object],
    *,
    current_value_limits: tuple[float, float] | None,
) -> _TypedDisplayState:
    if isinstance(state, ImageDisplayState):
        return image_display_from_form(
            state,
            values,
            current_color_limits=current_value_limits,
        )
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
    raise TypeError("unknown typed display state")


def _typed_state_with_x_view(
    state: _TypedDisplayState,
    x_view: tuple[float, float] | None,
) -> _TypedDisplayState:
    if isinstance(state, CurveDisplayState):
        return curve_display_with_x_view(state, x_view)
    if isinstance(state, HistogramDisplayState):
        return histogram_display_with_x_view(state, x_view)
    raise TypeError("x-view commits require CURVE or HISTOGRAM state")


def _payload_intent(payload: _TypedPanelPayload) -> ViewIntent:
    if isinstance(payload, ImagePanelPayload):
        return ViewIntent.IMAGE
    if isinstance(payload, CurvePanelPayload):
        return ViewIntent.CURVE
    if isinstance(payload, HistogramPanelPayload):
        return ViewIntent.HISTOGRAM
    raise TypeError("unknown typed payload")


@dataclass(frozen=True, slots=True)
class _TypedFigureFront:
    intent: ViewIntent
    state: _TypedDisplayState
    summary: str
    frame: BoardFrame
    required_peak_bytes: int
    effective_limit_bytes: int

    def __post_init__(self) -> None:
        if self.intent not in (
            ViewIntent.IMAGE,
            ViewIntent.CURVE,
            ViewIntent.HISTOGRAM,
        ):
            raise ValueError("typed figure front has another intent")
        if _state_intent(self.state) is not self.intent:
            raise ValueError("typed figure front state belongs to another intent")
        if not isinstance(self.summary, str) or not self.summary:
            raise ValueError("typed figure summary must be non-empty")
        if not isinstance(self.frame, BoardFrame) or len(self.frame.panels) != 1:
            raise TypeError("typed figure front requires one BoardFrame panel")
        panel = self.frame.panels[0]
        payload = panel.display_payload
        if (
            panel.panel_id != _TYPED_PANEL_ID
            or not isinstance(
                payload,
                (ImagePanelPayload, CurvePanelPayload, HistogramPanelPayload),
            )
            or _payload_intent(payload) is not self.intent
        ):
            raise ValueError("typed figure front has another payload")
        raster = panel.raster
        if isinstance(payload, ImagePanelPayload):
            expected_height, expected_width = payload.image.values.shape
            if (raster.width, raster.height) != (expected_width, expected_height):
                raise ValueError("IMAGE front differs from its exact raster geometry")
            if (
                raster.pixel_format is not PixelFormat.INDEXED8
                or raster.stride_bytes != raster.width
            ):
                raise ValueError("IMAGE front requires packed INDEXED8")
        else:
            if (raster.width, raster.height) != _NUMERIC_RASTER_SIZE:
                raise ValueError("numeric front has another raster geometry")
            if (
                raster.pixel_format is not PixelFormat.RGBA8888
                or raster.stride_bytes != raster.width * 4
            ):
                raise ValueError("numeric front requires packed RGBA")
        required = positive_integer(
            self.required_peak_bytes,
            "required_peak_bytes",
        )
        effective = positive_integer(
            self.effective_limit_bytes,
            "effective_limit_bytes",
        )
        if required > effective:
            raise MemoryError("typed figure front exceeds its frozen budget")
        object.__setattr__(self, "required_peak_bytes", required)
        object.__setattr__(self, "effective_limit_bytes", effective)


def _typed_join_digest(figure: DataFigure, intent: ViewIntent) -> str:
    evaluated = figure.evaluated
    source = evaluated.inputs[0]
    return canonical_digest(
        {
            "schema": "zlc_frontend.FrozenTypedFigureJoin",
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


def _typed_front_contract(
    front: _TypedFigureFront,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Freeze data/provenance identity while excluding display revision."""

    if not isinstance(front, _TypedFigureFront):
        raise TypeError("front must be _TypedFigureFront")
    frame = front.frame
    panel = frame.panels[0]
    payload = panel.display_payload
    assert isinstance(
        payload,
        (ImagePanelPayload, CurvePanelPayload, HistogramPanelPayload),
    )
    stamp = panel.coherence_stamp
    if len(stamp.presentations) != 1:
        raise ValueError("generic typed front requires one presentation identity")
    presentation = stamp.presentations[0]
    if presentation.panel_id != panel.panel_id:
        raise ValueError("typed presentation names another panel")
    if isinstance(payload, ImagePanelPayload):
        family_identity = (
            payload.image.x_axis,
            payload.image.y_axis,
            payload.viewport.axes,
            payload.viewport.coordinate_frame,
            payload.value_unit,
        )
        exact_data = (payload.image,)
    else:
        family_identity = (payload.series_labels, payload.value_unit)
        exact_data = tuple(payload.series)
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
        family_identity,
    )
    return identity, exact_data


def _typed_front_required_peak_bytes(
    figure: DataFigure,
    state: _TypedDisplayState,
) -> int:
    intent, unavailable_reason = _classify_single_typed(figure)
    if intent is None or intent is not _state_intent(state):
        raise ValueError(
            "typed budget requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    from zlc_frontend.matplotlib_render import (
        estimate_live_panel_raster_peak_nbytes,
        evaluated_figure_array_nbytes,
    )

    evaluated_bytes = evaluated_figure_array_nbytes(figure.evaluated)
    if isinstance(state, ImageDisplayState):
        from zlc_frontend.matplotlib_render import (
            estimate_image_png_export_peak_nbytes,
        )

        series = figure.evaluated.layers[0].cells[0].series
        image = series[0].data
        assert isinstance(image, EvaluatedImage)
        height, width = image.values.shape
        # The frozen session retains exact evaluated arrays and one current
        # INDEXED8/Qt-detached front.  A rerasterization candidate and a PNG
        # export are mutually exclusive on the one worker lane, so admission
        # adds the larger incremental peak instead of summing both operations.
        retained_baseline = evaluated_bytes + 2 * height * width
        raster_incremental = estimate_indexed8_raster_peak_nbytes(
            height,
            width,
            value_itemsize=image.values.dtype.itemsize,
            retained_fronts=0,
        )
        export_incremental = estimate_image_png_export_peak_nbytes(image)
        return retained_baseline + max(raster_incremental, export_incremental)

    width, height = _NUMERIC_RASTER_SIZE
    series_count = len(figure.evaluated.layers[0].cells[0].series)
    options = {
        "evaluated_data_upper_bound_bytes": evaluated_bytes,
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


def _render_typed_front(
    figure: DataFigure,
    state: _TypedDisplayState,
    *,
    current_value_limits: tuple[float, float] | None,
    previous_relim_mode,
    previous_count_scale: HistogramCountScale | None,
    sequence: int,
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> _TypedFigureFront:
    intent, unavailable_reason = _classify_single_typed(figure)
    if intent is None or intent is not _state_intent(state):
        raise ValueError(
            "typed render requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    _require_not_cancelled(cancelled)

    required = _typed_front_required_peak_bytes(figure, state)
    effective_limit = _figure_render_limit(figure, memory_limit_bytes)
    if required > effective_limit:
        raise MemoryError(
            f"interactive {intent.value.lower()} requires {required} bytes; "
            f"limit is {effective_limit}"
        )

    if isinstance(state, ImageDisplayState):
        evaluated_input = figure.evaluated.inputs[0]
        image = figure.evaluated.layers[0].cells[0].series[0].data
        assert isinstance(image, EvaluatedImage)
        home_viewport = image_viewport_for_evaluated_image(image)
        viewport = image_viewport_for_display_state(state, home_viewport)
        raster, data_range, histogram_counts, color_limits = (
            rasterize_image_indexed8(
                image,
                state,
                current_color_limits=current_value_limits,
                previous_relim_mode=previous_relim_mode,
            )
        )
        payload: _TypedPanelPayload = ImagePanelPayload(
            image,
            evaluated_input,
            viewport,
            data_range,
            histogram_counts,
            indexed_colormap(state.colormap.value),
            color_limits,
        )
    else:
        from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

        width, height = _NUMERIC_RASTER_SIZE
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
        _TYPED_PANEL_ID,
        figure.document.document_id,
        figure.document.revision,
        0,
        state.revision,
    )
    ref = evaluated_input.ref
    stamp = CoherenceStamp(
        f"figure:{ref.block_id.value}",
        ref.stream_generation.value,
        "FrozenTypedFigureJoin",
        _TYPED_JOIN_SCHEMA_DIGEST,
        _typed_join_digest(figure, intent),
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
        _TYPED_BOARD_ID,
        0,
        sequence,
        (
            PanelFrame(
                _TYPED_PANEL_ID,
                f"frozen-{intent.value.lower()}",
                source,
                stamp,
                raster,
                payload,
            ),
        ),
    )
    return _TypedFigureFront(
        intent,
        state,
        _figure_summary(figure),
        frame,
        required,
        effective_limit,
    )


def _export_typed_png(
    frame: BoardFrame,
    state: _TypedDisplayState,
    destination: Path,
    memory_limit_bytes: int,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> tuple[int, Path]:
    if not isinstance(frame, BoardFrame) or len(frame.panels) != 1:
        raise TypeError("typed export requires one exact BoardFrame")
    panel = frame.panels[0]
    payload = panel.display_payload
    if panel.panel_id != _TYPED_PANEL_ID or _payload_intent(payload) is not _state_intent(state):
        raise ValueError("typed export frame has another presentation")
    if isinstance(payload, ImagePanelPayload):
        def write_staged(path: Path) -> None:
            _require_not_cancelled(cancelled)
            from zlc_frontend.matplotlib_render import save_image_panel_png

            save_image_panel_png(
                payload,
                state,
                path,
                memory_limit_bytes=positive_integer(
                    memory_limit_bytes,
                    "image export memory limit",
                ),
            )
            _require_not_cancelled(cancelled)

        result = stage_and_replace_export(
            Path(destination),
            write_staged=write_staged,
            cancelled=cancelled,
            commit_lock=commit_lock,
        )
        return revision, result
    if not isinstance(payload, (CurvePanelPayload, HistogramPanelPayload)):
        raise ValueError("typed export payload is unsupported")
    raster = panel.raster
    if (
        raster.pixel_format is not PixelFormat.RGBA8888
        or raster.stride_bytes != raster.width * 4
    ):
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
    """Frozen generic viewer with one closed IMAGE/CURVE/HISTOGRAM front."""

    def __init__(
        self,
        initial_loader,
        typed_renderer,
        *,
        memory_limit_bytes: int,
    ) -> None:
        if not callable(initial_loader) or not callable(typed_renderer):
            raise TypeError("figure worker callables must be callable")
        self._typed_renderer = typed_renderer
        self._view_family: str | None = None
        self._display: _TypedDisplayState | None = None
        self._typed_contract: (
            tuple[tuple[object, ...], tuple[object, ...]] | None
        ) = None
        self._typed_pages_admitted = False
        self._typed_ui_faulted = False
        self._request_revision = 0
        self._active_kind: str | None = "initial"
        self._pending_state: _TypedDisplayState | None = None
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

        self._typed_page = QtWidgets.QWidget(self._tabs)
        self._typed_page.hide()
        page_layout = QtWidgets.QVBoxLayout(self._typed_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self._board_widget = QtRasterBoard(
            (_TYPED_PANEL_ID,),
            self._typed_page,
            columns=1,
            empty_text="Resolving exact typed figure…",
        )
        self._board_widget.setObjectName("figureViewerTypedBoard")
        self._board_widget.setMinimumSize(480, 320)
        page_layout.addWidget(self._board_widget, 1)

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("figureViewerTypedSettingsPopup")
        self._settings_popup_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        self._interaction_switch = FluentSwitch("Interact", self)
        self._interaction_switch.setObjectName("figureViewerTypedInteractSwitch")
        self._interaction_switch.setChecked(True)
        self._settings_button = FluentButton("Setting…", self, color=GREY)
        self._settings_button.setObjectName("figureViewerTypedSettingButton")
        self._export_button = FluentButton("Export PNG…", self, color=ORANGE)
        self._export_button.setObjectName("figureViewerTypedExportButton")
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
        if self._view_family in ("image", "curve", "histogram"):
            display = self._display
            payload = self._visible_typed_payload()
            visible_revision = (
                None
                if payload is None
                else (
                    payload.viewport.viewport_revision
                    if isinstance(payload, ImagePanelPayload)
                    else payload.viewport.display_revision
                )
            )
            return bool(
                display is not None
                and self._board_widget.front_frame is not None
                and self._pending_state is None
                and payload is not None
                and visible_revision == display.revision
            )
        return super().raster_ready

    def _visible_typed_payload(self) -> _TypedPanelPayload | None:
        if self._view_family == "image":
            payload = self._board_widget.visible_image_payload(_TYPED_PANEL_ID)
        elif self._view_family == "curve":
            payload = self._board_widget.visible_curve_payload(_TYPED_PANEL_ID)
        elif self._view_family == "histogram":
            payload = self._board_widget.visible_histogram_payload(_TYPED_PANEL_ID)
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
            ImagePanelPayload
            if self._view_family == "image"
            else (
                CurvePanelPayload
                if self._view_family == "curve"
                else HistogramPanelPayload
            )
        )
        return candidate if isinstance(candidate, expected_type) else None

    def _visible_typed_origin(self) -> PanelInteractionOrigin | None:
        if self._view_family == "image":
            return self._board_widget.visible_image_origin(_TYPED_PANEL_ID)
        if self._view_family == "curve":
            return self._board_widget.visible_curve_origin(_TYPED_PANEL_ID)
        if self._view_family == "histogram":
            return self._board_widget.visible_histogram_origin(_TYPED_PANEL_ID)
        return None

    def _visible_value_limits(self) -> tuple[float, float] | None:
        payload = self._visible_typed_payload()
        if isinstance(payload, ImagePanelPayload):
            return payload.color_limits
        if isinstance(payload, CurvePanelPayload):
            return payload.viewport.y_limits
        if isinstance(payload, HistogramPanelPayload):
            return payload.viewport.count_limits
        return None

    def _runtime_placeholders(self):
        display = self._display
        if isinstance(display, ImageDisplayState):
            payload = self._visible_typed_payload()
            if not isinstance(payload, ImagePanelPayload):
                return {}
            x_view, y_view = (
                payload.viewport.optional_coordinate_views_for_normalized_bounds()
            )
            placeholders: dict[str, str] = {}
            for limits, low, high in (
                (x_view, "x_min", "x_max"),
                (y_view, "y_min", "y_max"),
                (payload.color_limits, "color_min", "color_max"),
            ):
                resolved = runtime_range_placeholders(limits, low, high)
                if resolved is not None:
                    placeholders.update(resolved)
            return placeholders
        if isinstance(display, CurveDisplayState):
            fields = ("y_min", "y_max")
        elif isinstance(display, HistogramDisplayState):
            fields = ("count_min", "count_max")
        else:
            return {}
        return runtime_range_placeholders(self._visible_value_limits(), *fields)

    def _ensure_typed_controls(self, state: _TypedDisplayState) -> None:
        if self._edit_display is not None or self._setting_display is not None:
            if (
                self._display is not None
                and _state_intent(self._display) is not _state_intent(state)
            ):
                raise RuntimeError("typed window cannot change family")
            return
        if isinstance(state, ImageDisplayState):
            runtime_fields = (
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "color_min",
                "color_max",
            )
            subject = "image display"
            bind = None
        elif isinstance(state, CurveDisplayState):
            runtime_fields = ("y_min", "y_max")
            subject = "curve display"
            bind = self._board_widget.bind_curve_interaction
        else:
            runtime_fields = ("count_min", "count_max")
            subject = "histogram display"
            bind = self._board_widget.bind_histogram_interaction
        spec = _typed_form_spec(state)
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
            edit.setObjectName("figureViewerTypedEditEditor")
            setting.setObjectName("figureViewerTypedSettingEditor")
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
            if isinstance(state, ImageDisplayState):
                payload = self._visible_typed_payload()
                if not isinstance(payload, ImagePanelPayload):
                    raise RuntimeError("IMAGE controls require one exact payload")
                self._board_widget.bind_rectangle_selector(
                    _TYPED_PANEL_ID,
                    payload.viewport,
                    self._accept_image_rectangle,
                    enabled=True,
                    interaction_callback=self._accept_image_interaction,
                )
            else:
                assert bind is not None
                bind(_TYPED_PANEL_ID, self._accept_numeric_interaction, enabled=True)
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
            raise RuntimeError("typed controls are not admitted")
        return self._edit_display, self._setting_display

    def _sync_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        display = self._display
        if display is None:
            raise RuntimeError("typed display state is not admitted")
        sync_revisioned_form_editors(
            self._editors(),
            revision=display.revision,
            semantic_identity=display,
            values=_typed_form_values(display),
            runtime_placeholders=self._runtime_placeholders(),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    def _sync_committed_typed_controls(
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
            self._typed_ui_faulted = True
            try:
                self._set_typed_controls_enabled(False)
            except BaseException:
                pass
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _reload_editor(self, editor: FluentRevisionedFormEditor) -> None:
        if editor not in self._editors():
            raise ValueError("typed editor does not belong to this window")
        display = self._display
        if display is None:
            raise RuntimeError("typed display state is not admitted")
        editor.load(
            revision=display.revision,
            semantic_identity=display,
            values=_typed_form_values(display),
            runtime_placeholders=self._runtime_placeholders(),
        )

    def _set_typed_controls_enabled(self, enabled: bool) -> None:
        active = bool(
            enabled
            and not self._typed_ui_faulted
            and self._view_family in ("image", "curve", "histogram")
        )
        self._board_widget.set_interaction_readiness(
            image=active and self._view_family == "image",
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
        if self._view_family not in ("image", "curve", "histogram"):
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
            raise ValueError("typed editor does not belong to this window")
        try:
            display = self._display
            if display is None:
                raise RuntimeError("typed display state is not admitted")
            if self._future is not None or self._closing:
                raise RuntimeError("typed display work is already active")
            if base_revision != display.revision:
                raise RuntimeError(
                    f"typed draft r{base_revision} is stale; "
                    f"current revision is r{display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("typed display form must emit one exact mapping")
            candidate = _typed_state_from_form(
                display,
                values,
                current_value_limits=self._visible_value_limits(),
            )
            self._start_typed_render(
                candidate,
                editor=editor,
                editor_revision=base_revision,
            )
        except BaseException as error:
            self._diagnostic.setText(
                f"Typed display edit rejected: {error_summary(error)}"
            )

    def _accept_image_rectangle(self, gesture: RectangleGesture) -> None:
        display = self._display
        origin = self._visible_typed_origin()
        if not isinstance(display, ImageDisplayState) or origin is None:
            raise RuntimeError("IMAGE rectangle has no current exact front")
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("IMAGE rectangle must be RectangleGesture")
        if (
            gesture.panel_id != _TYPED_PANEL_ID
            or (
                gesture.board_id,
                gesture.layout_generation,
                gesture.sequence,
                gesture.source_identity,
                gesture.viewport_revision,
            )
            != (
                origin.board_id,
                origin.layout_generation,
                origin.sequence,
                origin.source_identity,
                display.revision,
            )
        ):
            raise RuntimeError("IMAGE rectangle origin is stale")
        self._board_widget.set_image_rectangle_candidate(
            gesture.normalized_bounds,
            panel_id=_TYPED_PANEL_ID,
        )
        left, top, right, bottom = gesture.normalized_bounds
        self._diagnostic.setText(
            "DISPLAY ONLY rectangle "
            f"({left:.6g}, {top:.6g})..({right:.6g}, {bottom:.6g})"
        )

    def _accept_image_interaction(self, command: ImageInteractionCommit) -> None:
        display = self._display
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("unknown IMAGE interaction command")
        if not isinstance(display, ImageDisplayState):
            raise RuntimeError("IMAGE interaction belongs to another family")
        origin = command.origin
        if (
            origin.panel_id != _TYPED_PANEL_ID
            or self._visible_typed_origin() != origin
            or origin.presentation.panel_revision != display.revision
        ):
            raise RuntimeError("IMAGE interaction origin is stale")
        if isinstance(command, ImageViewportCommit):
            candidate = image_display_for_viewport(display, command.viewport)
        else:
            candidate = replace(
                display,
                revision=display.revision + 1,
                relim_mode=RelimMode.FIXED,
                fixed_color_limits=command.color_limits,
            )
        self._start_typed_render(candidate, origin=origin)

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
            origin.panel_id != _TYPED_PANEL_ID
            or self._visible_typed_origin() != origin
            or origin.presentation.panel_revision != display.revision
        ):
            raise RuntimeError("numeric interaction origin is stale")
        if isinstance(command, (CurveRangeGesture, HistogramRangeGesture)):
            setter = (
                self._board_widget.set_curve_range_candidate
                if is_curve
                else self._board_widget.set_histogram_range_candidate
            )
            setter(command.x_span, panel_id=_TYPED_PANEL_ID)
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
        self._start_typed_render(
            _typed_state_with_x_view(display, command.viewport.x_limits),
            origin=origin,
        )

    def _start_typed_render(
        self,
        candidate: _TypedDisplayState,
        *,
        editor: FluentRevisionedFormEditor | None = None,
        editor_revision: int | None = None,
        origin: PanelInteractionOrigin | None = None,
    ) -> None:
        display = self._display
        payload = self._visible_typed_payload()
        if display is None or payload is None:
            raise RuntimeError("typed figure is not ready")
        if self._future is not None or self._closing:
            raise RuntimeError("typed render is already active")
        if _state_intent(candidate) is not _state_intent(display):
            raise TypeError("candidate belongs to another typed family")
        if candidate == display:
            if origin is not None:
                raise ValueError("typed interaction cannot commit a no-op")
            self._sync_editors(
                accepted_editor=editor,
                accepted_base_revision=editor_revision,
            )
            return
        if candidate.revision != display.revision + 1:
            raise ValueError("typed display revision must advance once")
        self._request_revision += 1
        self._active_kind = "typed"
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
            self._typed_renderer,
            candidate,
            self._visible_value_limits(),
            display.relim_mode,
            previous_scale,
            self._request_revision,
            self._memory_limit_bytes,
            self._cancelled,
        )
        if not submitted:
            self._discard_pending_typed()

    def _discard_pending_typed(self) -> None:
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
                discard = {
                    "image": self._board_widget.discard_pending_image_interaction,
                    "curve": self._board_widget.discard_pending_curve_interaction,
                    "histogram": (
                        self._board_widget.discard_pending_histogram_interaction
                    ),
                }.get(family)
                if discard is None:
                    raise RuntimeError("pending interaction has no typed family")
                discard(origin)
            except BaseException as error:
                cleanup_errors.append(error_summary(error))
        if family in ("image", "curve", "histogram"):
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
        front: _TypedFigureFront,
        expected_state: _TypedDisplayState,
    ) -> None:
        if front.state != expected_state or front.intent is not _state_intent(expected_state):
            raise ValueError("typed worker returned conflicting authored state")
        payload = front.frame.panels[0].display_payload
        assert isinstance(
            payload,
            (ImagePanelPayload, CurvePanelPayload, HistogramPanelPayload),
        )
        viewport = payload.viewport
        revision = (
            viewport.viewport_revision
            if isinstance(payload, ImagePanelPayload)
            else viewport.display_revision
        )
        if revision != expected_state.revision:
            raise ValueError("typed worker returned another display revision")
        if isinstance(expected_state, ImageDisplayState):
            assert isinstance(payload, ImagePanelPayload)
            home = image_viewport_for_evaluated_image(payload.image)
            expected_viewport = image_viewport_for_display_state(
                expected_state,
                home,
            )
            if (
                payload.viewport != expected_viewport
                or payload.fit_overlay is not None
                or payload.base_palette
                != indexed_colormap(expected_state.colormap.value)
                or (
                    expected_state.relim_mode is RelimMode.FIXED
                    and payload.color_limits
                    != expected_state.fixed_color_limits
                )
            ):
                raise ValueError("IMAGE worker returned conflicting authored state")
            return
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

    def _present_typed_front(
        self,
        front: _TypedFigureFront,
        *,
        expected_state: _TypedDisplayState,
        request_revision: int,
    ) -> None:
        if front.required_peak_bytes > self._memory_limit_bytes:
            raise MemoryError("typed front exceeds the window budget")
        request_revision = nonnegative_integer(
            request_revision,
            "typed request revision",
        )
        if front.frame.sequence != request_revision:
            raise ValueError("typed worker returned another request sequence")
        self._validate_authored_front(front, expected_state)
        contract = _typed_front_contract(front)
        expected_contract = self._typed_contract
        if expected_contract is not None:
            expected_identity, expected_series = expected_contract
            identity, series = contract
            if identity != expected_identity:
                raise ValueError("typed worker changed frozen source provenance")
            if len(series) != len(expected_series) or any(
                actual is not expected
                for actual, expected in zip(series, expected_series, strict=True)
            ):
                raise ValueError("typed worker changed frozen evaluated data")

        self._board_widget.present(front.frame)
        # The admitted board front is the transaction boundary.  Commit the
        # exact authored state/contract before any optional Qt chrome work.
        if expected_contract is None:
            self._typed_contract = contract
        self._display = expected_state
        self._view_family = front.intent.value.lower()
        # Page/chrome and controls are ancillary to the already-admitted
        # immutable data front.  Their faults can disable UI, never roll it back.
        try:
            if not self._typed_pages_admitted:
                self._retire_tab_pages()
                self._tabs.addTab(self._typed_page, front.intent.value.title())
                self._tabs.tabBar().setVisible(False)
                self._typed_page.show()
                self._typed_pages_admitted = True
            self._mode.setText(
                f"EXACT {front.intent.value} · INTERACTIVE · DISPLAY ONLY"
            )
            self._status.setText("READY")
            self._summary.setText(front.summary)
            self._diagnostic.setText("")
        except BaseException as error:
            self._typed_ui_faulted = True
            self._set_typed_controls_enabled(False)
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))
            return
        try:
            self._ensure_typed_controls(expected_state)
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
            self._typed_ui_faulted = True
            self._set_typed_controls_enabled(False)
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        try:
            result = future.result()
        except CancelledError:
            if not self._closing:
                self._status.setText("FIGURE CANCELLED")
                if kind == "typed":
                    self._discard_pending_typed()
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
            elif isinstance(result, _TypedFigureFront):
                self._present_typed_front(
                    result,
                    expected_state=_default_typed_state(result.intent),
                    request_revision=self._request_revision,
                )
                if not self._typed_ui_faulted:
                    self._sync_committed_typed_controls()
            else:
                raise TypeError("initial figure worker returned another result")
            self._active_kind = None
            return
        if kind == "typed":
            if not isinstance(result, _TypedFigureFront):
                raise TypeError("typed worker returned another result")
            pending = self._pending_state
            editor = self._pending_editor
            editor_revision = self._pending_editor_revision
            if pending is None:
                raise RuntimeError("typed worker completed without pending state")
            self._present_typed_front(
                result,
                expected_state=pending,
                request_revision=self._request_revision,
            )
            self._pending_state = None
            self._pending_origin = None
            self._pending_editor = None
            self._pending_editor_revision = None
            self._active_kind = None
            if not self._typed_ui_faulted:
                self._sync_committed_typed_controls(
                    accepted_editor=editor,
                    accepted_base_revision=editor_revision,
                )
            return
        if kind == "export":
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("typed export returned another result")
            revision, destination = result
            if revision != self._request_revision:
                raise ValueError("typed export revision is stale")
            self._active_kind = None
            self._status.setText("READY")
            self._diagnostic.setText(f"Exported {destination}")
            try:
                self._set_typed_controls_enabled(True)
            except BaseException as error:
                self._typed_ui_faulted = True
                self._status.setText("TYPED CONTROLS FAILED")
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
        if kind == "typed":
            family = (self._view_family or "typed").upper()
            self._status.setText(f"{family} DISPLAY FAILED")
            self._diagnostic.setText(error_summary(error))
            self._discard_pending_typed()
        elif kind == "export":
            self._status.setText("TYPED EXPORT FAILED")
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
            or self._view_family not in ("image", "curve", "histogram")
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
        display = self._display
        if display is None:
            self._active_kind = None
            self._set_typed_controls_enabled(True)
            return
        if not self._submit_future(
            _export_typed_png,
            frame,
            display,
            Path(destination),
            self._memory_limit_bytes,
            self._request_revision,
            self._cancelled,
            self._export_commit_lock,
        ):
            self._active_kind = None
            self._set_typed_controls_enabled(True)

    def _clear_bundle(self) -> None:
        super()._clear_bundle()
        if self._view_family in ("image", "curve", "histogram"):
            self._board_widget.clear()

    def _finish_close_if_ready(self) -> None:
        if self._closing and self._future is None and not self._closed:
            self._typed_renderer = None
            self._typed_contract = None
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
    cached_typed: DataFigure | None = None

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
        nonlocal cached_typed
        require_worker_owner()
        _require_not_cancelled(cancelled)
        figure = loader()
        if not isinstance(figure, DataFigure):
            raise TypeError("figure loader must return DataFigure")
        intent, unavailable_reason = _classify_single_typed(figure)
        if intent is not None:
            state = _default_typed_state(intent)
            if _typed_front_required_peak_bytes(figure, state) > _figure_render_limit(
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
            cached_typed = figure
            return _render_typed_front(
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
        state: _TypedDisplayState,
        current_value_limits,
        previous_relim_mode,
        previous_count_scale,
        sequence: int,
        memory_limit: int,
        cancelled: threading.Event,
    ) -> _TypedFigureFront:
        require_worker_owner()
        figure = cached_typed
        if figure is None:
            raise RuntimeError("typed session has no frozen DataFigure")
        return _render_typed_front(
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
