"""Nonblocking Qt viewer for one frozen current :class:`DataFigure`.

The generic fallback remains an immutable encoded board.  The three earned
products -- one logical IMAGE, CURVE, or HISTOGRAM panel -- share one typed
board, one Setting/Edit projection, and one render/export lifecycle.  No
whole-board PNG is reverse-mapped into data.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
import time

from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    AxisId,
    CoordinateRangeSelection,
    FitBatchStatus,
    FitCancelled,
    FitDeadlineExceeded,
    FitResultBatch,
    FitSpec,
    IndexRangeSelection,
    Selection,
    dataset_revision_ref_to_tree,
    fit_result_retained_upper_bound_nbytes,
)
from zlc_frontend import (
    BoardFrame,
    CoherenceStamp,
    CurveFitOverlay,
    CurvePanelPayload,
    DataFigure,
    FitAuthoringOption,
    HistogramPanelPayload,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
    RadialGaussianImageFitOverlay,
    SourceIdentity,
)
from zlc_frontend.fit_curve_projection import (
    CurveFitOverlayPlan,
    estimate_curve_fit_overlay_plan_nbytes,
    materialize_curve_fit_overlay_plan,
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
    AxisViewRole,
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
    FitAuthoringPane,
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
from zlc_neutral_atom.artifacts import FitExecution, FitResultArtifactRef
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
from zlc_workbench.fit import FitDraftAuthority, FitDraftResult


_DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_DEFAULT_FIT_TIMEOUT_SECONDS = 30.0
_FIT_WORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-data-figure-fit",
)
_TYPED_BOARD_ID = "generic-typed-figure"
_TYPED_PANEL_ID = "generic-typed"
_NUMERIC_RASTER_SIZE = (800, 520)
_TYPED_JOIN_SCHEMA_DIGEST = canonical_digest(
    {
        "schema": "zlc_frontend.FrozenTypedFigureJoin",
        "fields": ("document", "input", "intent", "fit_result_identity"),
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


@dataclass(frozen=True, slots=True)
class _FitWorkbenchBindings:
    """Composition-owned capabilities for the optional Figure Fit surface.

    The viewer receives only four narrow calls.  It never receives a repository,
    Experiment, source resolver, or generic analysis registry.  ``prepare``
    turns exact named display axes plus an optional authority candidate into
    already-bound data-owned Fit requests; ``execute`` is the only materializing
    operation; ``save`` publishes the process-local execution hidden behind
    :class:`FitDraftAuthority`; and ``reload`` proves the saved reference before
    the UI labels an overlay as durable.
    """

    prepare: object
    execute: object
    save: object
    reload: object
    selected_model: str | None = None
    initial_selection: Selection | None = None
    open_analysis: bool = False
    timeout_seconds: float = _DEFAULT_FIT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        for name in ("prepare", "execute", "save", "reload"):
            if not callable(getattr(self, name)):
                raise TypeError(f"fit {name} capability must be callable")
        selected = self.selected_model
        if selected is not None and (
            not isinstance(selected, str) or not selected.strip()
        ):
            raise ValueError("selected_model must be non-empty text or None")
        if self.initial_selection is not None and not isinstance(
            self.initial_selection,
            Selection,
        ):
            raise TypeError("initial_selection must be Selection or None")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("fit timeout_seconds must be finite and positive")
        object.__setattr__(self, "timeout_seconds", timeout)


@dataclass(frozen=True, slots=True)
class _FitSelectionCandidate:
    origin: PanelInteractionOrigin
    selection: Selection

    def __post_init__(self) -> None:
        if not isinstance(self.origin, PanelInteractionOrigin):
            raise TypeError("fit selection origin must be PanelInteractionOrigin")
        if not isinstance(self.selection, Selection):
            raise TypeError("fit selection candidate must be Selection")


@dataclass(frozen=True, slots=True)
class _FitOverlayRequest:
    analysis_revision: int
    result: FitResultBatch | None
    result_identity: str | None
    result_retained_bytes: int = 0

    def __post_init__(self) -> None:
        revision = nonnegative_integer(
            self.analysis_revision,
            "fit overlay analysis revision",
        )
        object.__setattr__(self, "analysis_revision", revision)
        if self.result is not None and not isinstance(self.result, FitResultBatch):
            raise TypeError("fit overlay result must be FitResultBatch or None")
        if self.result is not None and self.result_identity is None:
            raise ValueError("transient fit overlay result requires an identity")
        if self.result_identity is not None and (
            not isinstance(self.result_identity, str)
            or not self.result_identity.strip()
        ):
            raise ValueError("fit overlay identity must be non-empty text or None")
        retained = nonnegative_integer(
            self.result_retained_bytes,
            "fit overlay result_retained_bytes",
        )
        if self.result is None and retained:
            raise ValueError("an overlay without a transient result retains no result bytes")
        if self.result is not None and retained <= 0:
            raise ValueError("a transient Fit overlay requires a retained-byte bound")
        object.__setattr__(self, "result_retained_bytes", retained)


def _same_fit_overlay_request(
    left: _FitOverlayRequest | None,
    right: _FitOverlayRequest | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        left.analysis_revision == right.analysis_revision
        and left.result_identity == right.result_identity
        and left.result is right.result
        and left.result_retained_bytes == right.result_retained_bytes
    )


def _fit_summary(
    draft: FitDraftResult,
    *,
    cancelled=None,
) -> str:
    result = draft.result
    counts = Counter(status.value for status in result.statuses)
    status_text = ", ".join(
        f"{name.lower()}={count}" for name, count in sorted(counts.items())
    )
    quality_min = math.inf
    quality_max = -math.inf
    for index, (status, rss, used) in enumerate(
        zip(
            result.statuses,
            result.residual_sum_squares,
            result.used_observation_counts,
            strict=True,
        )
    ):
        if cancelled is not None and index % 1024 == 0 and cancelled():
            raise CancelledError()
        if status is not FitBatchStatus.CONVERGED or int(used) <= 0:
            continue
        value = math.sqrt(float(rss) / int(used))
        if math.isfinite(value):
            quality_min = min(quality_min, value)
            quality_max = max(quality_max, value)
    quality_text = (
        "no converged RMSE"
        if not math.isfinite(quality_min)
        else f"RMSE {quality_min:.4g}–{quality_max:.4g}"
    )
    return (
        f"{result.spec.model_id} · {len(result.statuses)} named batch cell(s) · "
        f"{status_text} · {quality_text} · draft is not saved"
    )


def _prepare_fit_options(
    prepare,
    fit_axis_ids: tuple[AxisId, ...],
    axis_roles: tuple[tuple[AxisId, AxisViewRole], ...],
    selection: Selection | None,
    operation_memory_limit_bytes: int,
) -> tuple[FitAuthoringOption, ...]:
    options = tuple(
        prepare(
            fit_axis_ids,
            selection,
            positive_integer(
                operation_memory_limit_bytes,
                "operation_memory_limit_bytes",
            ),
        )
    )
    if not options or any(
        not isinstance(option, FitAuthoringOption) for option in options
    ):
        raise ValueError("Fit preparation produced no FitAuthoringOption")
    schemas = {option.spec.input_schema_fingerprint for option in options}
    models = tuple(option.spec.model_id for option in options)
    if len(schemas) != 1 or len(models) != len(set(models)):
        raise ValueError("Fit options require one source schema and unique models")
    if any(option.spec.fit_axis_ids != fit_axis_ids for option in options):
        raise ValueError("Fit option axes differ from the exact displayed axes")
    return _validate_fit_replay_options(
        options,
        fit_axis_ids=fit_axis_ids,
        axis_roles=axis_roles,
        selection=selection,
    )


def _execute_fit_draft(
    authority: FitDraftAuthority,
    spec: FitSpec,
    deadline_monotonic: float,
    window_cancelled: threading.Event,
    analysis_cancelled: threading.Event,
) -> tuple[FitDraftResult, str, int]:
    def cancelled() -> bool:
        return window_cancelled.is_set() or analysis_cancelled.is_set()

    if cancelled():
        raise CancelledError()
    if time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit expired while waiting for its worker lane")
    draft = authority.execute(spec, cancelled, deadline_monotonic)
    try:
        return (
            draft,
            _fit_summary(draft, cancelled=cancelled),
            fit_result_retained_upper_bound_nbytes(draft.result),
        )
    except BaseException:
        # ``authority.execute`` has already installed the one live draft.  Any
        # failure in worker-only presentation/accounting must release that exact
        # generation or all later Fit submissions deadlock behind a hidden draft.
        authority.discard(draft)
        raise


def _reload_fit_result_with_retained(
    reload_result,
    reference: FitResultArtifactRef,
    memory_limit_bytes: int,
) -> tuple[FitResultBatch, int]:
    result = reload_result(reference, memory_limit_bytes)
    if not isinstance(result, FitResultBatch):
        raise TypeError("saved Fit reload returned another result type")
    return result, fit_result_retained_upper_bound_nbytes(result)


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


def _fit_projection_metadata(
    figure: DataFigure,
    intent: ViewIntent,
) -> tuple[tuple[AxisId, ...], tuple[tuple[AxisId, AxisViewRole], ...]]:
    layer = figure.document.layers[0]
    roles = tuple(
        sorted(
            ((binding.axis_id, binding.role) for binding in layer.view.axis_bindings),
            key=lambda item: item[0].value,
        )
    )
    if intent is ViewIntent.CURVE:
        fit_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.X
        )
    elif intent is ViewIntent.IMAGE:
        x_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.IMAGE_X
        )
        y_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.IMAGE_Y
        )
        fit_axes = (*x_axes, *y_axes)
    else:
        fit_axes = ()
    expected = 1 if intent is ViewIntent.CURVE else 2 if intent is ViewIntent.IMAGE else 0
    if len(fit_axes) != expected:
        raise ValueError("typed figure has ambiguous fitted display axes")
    return fit_axes, roles


def _validate_fit_replay_options(
    options: tuple[FitAuthoringOption, ...],
    *,
    fit_axis_ids: tuple[AxisId, ...],
    axis_roles: tuple[tuple[AxisId, AxisViewRole], ...],
    selection: Selection | None,
) -> tuple[FitAuthoringOption, ...]:
    """Reject a solve whose named result rows cannot map to this exact panel."""

    role_by_axis = dict(axis_roles)
    accepted_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
        AxisViewRole.SLIDER,
    }
    prepared = []
    for option in options:
        if option.spec.fit_axis_ids != fit_axis_ids:
            continue
        batch_sizes = dict(option.batch_axis_sizes)
        def batch_axis_is_replayable(axis_id: AxisId) -> bool:
            role = role_by_axis.get(axis_id)
            if role in accepted_batch_roles:
                return True
            return bool(
                role is AxisViewRole.REDUCED
                and batch_sizes[axis_id] == 1
            )

        if any(
            not batch_axis_is_replayable(axis_id)
            for axis_id in option.spec.batch_axis_ids
        ):
            continue
        transform = option.spec.committed_transform
        if selection is None:
            if transform is not None:
                continue
        else:
            if transform is None:
                continue
            if tuple(transform.spec.operations) != (selection,):
                continue
        prepared.append(option)
    if not prepared:
        raise ValueError(
            "the visible panel cannot map an authoritative Fit result without "
            "reducing or guessing a named batch axis"
        )
    return tuple(prepared)


@dataclass(frozen=True, slots=True)
class _TypedFigureFront:
    intent: ViewIntent
    state: _TypedDisplayState
    summary: str
    frame: BoardFrame
    data_contract: tuple[tuple[object, ...], tuple[object, ...]]
    fit_axis_ids: tuple[AxisId, ...]
    axis_roles: tuple[tuple[AxisId, AxisViewRole], ...]
    fit_result_identity: str | None
    transient_fit_result_owner: FitResultBatch | None
    transient_fit_result_retained_bytes: int
    release_initial_canonical_on_commit: bool
    retained_figure_upper_bound_bytes: int
    fit_overlay_retained_bytes: int
    session_peak_bytes: int
    concurrent_reservation_bytes: int
    required_peak_bytes: int
    effective_limit_bytes: int
    aggregate_limit_bytes: int

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
        if (
            not isinstance(self.data_contract, tuple)
            or len(self.data_contract) != 2
            or not isinstance(self.data_contract[0], tuple)
            or not isinstance(self.data_contract[1], tuple)
        ):
            raise TypeError(
                "typed figure data_contract must be identity/exact-owner tuples"
            )
        fit_axis_ids = tuple(self.fit_axis_ids)
        if any(not isinstance(axis_id, AxisId) for axis_id in fit_axis_ids):
            raise TypeError("typed fit_axis_ids must contain AxisId values")
        if len(set(fit_axis_ids)) != len(fit_axis_ids):
            raise ValueError("typed fit_axis_ids must be unique")
        roles = tuple(self.axis_roles)
        if any(
            not isinstance(axis_id, AxisId) or not isinstance(role, AxisViewRole)
            for axis_id, role in roles
        ):
            raise TypeError("typed axis_roles must contain AxisId/AxisViewRole pairs")
        if len({axis_id for axis_id, _role in roles}) != len(roles):
            raise ValueError("typed axis_roles repeat an axis")
        if self.fit_result_identity is not None and (
            not isinstance(self.fit_result_identity, str)
            or not self.fit_result_identity.strip()
        ):
            raise ValueError("typed fit result identity must be non-empty text or None")
        if self.transient_fit_result_owner is not None and not isinstance(
            self.transient_fit_result_owner,
            FitResultBatch,
        ):
            raise TypeError("transient_fit_result_owner must be FitResultBatch or None")
        transient_retained = nonnegative_integer(
            self.transient_fit_result_retained_bytes,
            "transient_fit_result_retained_bytes",
        )
        if (self.transient_fit_result_owner is None) != (transient_retained == 0):
            raise ValueError(
                "transient Fit result owner and retained bound must be present together"
            )
        object.__setattr__(
            self,
            "transient_fit_result_retained_bytes",
            transient_retained,
        )
        if not isinstance(self.release_initial_canonical_on_commit, bool):
            raise TypeError("release_initial_canonical_on_commit must be bool")
        retained = positive_integer(
            self.retained_figure_upper_bound_bytes,
            "retained_figure_upper_bound_bytes",
        )
        object.__setattr__(self, "fit_axis_ids", fit_axis_ids)
        object.__setattr__(self, "axis_roles", roles)
        object.__setattr__(self, "retained_figure_upper_bound_bytes", retained)
        overlay_retained = nonnegative_integer(
            self.fit_overlay_retained_bytes,
            "fit_overlay_retained_bytes",
        )
        object.__setattr__(self, "fit_overlay_retained_bytes", overlay_retained)
        session_peak = positive_integer(self.session_peak_bytes, "session_peak_bytes")
        if session_peak < retained:
            raise ValueError("typed session peak is smaller than retained DataFigure")
        object.__setattr__(self, "session_peak_bytes", session_peak)
        concurrent = positive_integer(
            self.concurrent_reservation_bytes,
            "concurrent_reservation_bytes",
        )
        if concurrent < retained:
            raise ValueError("typed concurrent reservation is smaller than retained DataFigure")
        object.__setattr__(self, "concurrent_reservation_bytes", concurrent)
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
        aggregate = positive_integer(
            self.aggregate_limit_bytes,
            "aggregate_limit_bytes",
        )
        if session_peak > aggregate:
            raise MemoryError("typed Figure session exceeds its frozen budget")
        if concurrent > aggregate:
            raise MemoryError("typed Figure rerender reservation exceeds its frozen budget")
        object.__setattr__(self, "required_peak_bytes", required)
        object.__setattr__(self, "effective_limit_bytes", effective)
        object.__setattr__(self, "aggregate_limit_bytes", aggregate)


def _typed_join_digest(
    figure: DataFigure,
    intent: ViewIntent,
    fit_result_identity: str | None,
) -> str:
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
            "fit_result_identity": fit_result_identity,
            "input": {
                "dataset_id": source.dataset_id.value,
                "ref": dataset_revision_ref_to_tree(source.ref),
            },
        }
    )


def _build_typed_front_contract(
    intent: ViewIntent,
    effective_limit_bytes: int,
    frame: BoardFrame,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Build a bounded source token plus exact immutable data owners."""

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
        # Object ids are safe here because ``exact_data`` keeps both immutable
        # axes/images alive while the token is compared.  They avoid rescanning
        # a large coordinate vector on the Qt owner while still detecting a
        # replacement axis/value object returned by a faulty worker.
        def axis_identity(axis) -> tuple[object, ...]:
            return (
                axis.axis_id,
                axis.role,
                axis.unit,
                axis.coordinate_frame,
                len(axis.indices),
                id(axis.indices),
                len(axis.coordinates),
                id(axis.coordinates),
            )

        family_identity = (
            axis_identity(payload.image.x_axis),
            axis_identity(payload.image.y_axis),
            payload.viewport.coordinate_frame,
            payload.value_unit,
        )
        exact_data = (payload.image,)
    else:
        family_identity = (payload.series_labels, payload.value_unit)
        exact_data = payload.series
    stable_identity = (
        intent,
        effective_limit_bytes,
        frame.board_id,
        frame.layout_generation,
        panel.panel_id,
        panel.coherence_group,
        panel.source_identity,
        stamp.run_id,
        stamp.provenance_epoch_id,
        stamp.join_key_type,
        stamp.join_key_schema_fingerprint,
        stamp.inputs,
        presentation.panel_id,
        presentation.document_id,
        presentation.document_revision,
        presentation.selection_revision,
        payload.evaluated_input,
        family_identity,
    )
    # Fit-result identity is intentionally part of the per-front join digest
    # and therefore may change between legitimate overlay commits.  Nest it
    # outside the stable source identity: self-validation compares both;
    # cross-front CAS compares only the stable component.
    identity = (stable_identity, stamp.join_key_digest)
    return identity, exact_data


def _typed_front_contract(
    front: _TypedFigureFront,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Rebuild the bounded token from the payload actually being presented."""

    if not isinstance(front, _TypedFigureFront):
        raise TypeError("front must be _TypedFigureFront")
    return _build_typed_front_contract(
        front.intent,
        front.effective_limit_bytes,
        front.frame,
    )


def _same_exact_data_owners(
    left: tuple[object, ...],
    right: tuple[object, ...],
) -> bool:
    return len(left) == len(right) and all(
        actual is expected
        for actual, expected in zip(left, right, strict=True)
    )


def _validate_rendered_authored_payload(
    payload: _TypedPanelPayload,
    expected_state: _TypedDisplayState,
    fit_result_identity: str | None,
) -> None:
    """Perform data-sized authored-state proof on the render worker."""

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
        if (
            payload.viewport.optional_coordinate_views_for_normalized_bounds()
            != (expected_state.x_view, expected_state.y_view)
            or ((payload.fit_overlay is None) != (fit_result_identity is None))
            or (
                payload.fit_overlay is not None
                and payload.fit_overlay.result_identity != fit_result_identity
            )
            or payload.base_palette != indexed_colormap(expected_state.colormap.value)
            or (
                expected_state.relim_mode is RelimMode.FIXED
                and payload.color_limits != expected_state.fixed_color_limits
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
            or ((not payload.fit_overlays) != (fit_result_identity is None))
            or any(
                overlay.result_identity != fit_result_identity
                for overlay in payload.fit_overlays
            )
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


def _typed_front_required_peak_bytes(
    figure: DataFigure,
    state: _TypedDisplayState,
    *,
    curve_fit_overlay_nbytes: tuple[int, int] | None = None,
    image_fit_overlay_retained_bytes: int = 0,
    fit_result_retained_bytes: int = 0,
    previous_fit_overlay_retained_bytes: int = 0,
    external_session_retained_bytes: int = 0,
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
        retained_baseline = (
            evaluated_bytes
            + external_session_retained_bytes
            + fit_result_retained_bytes
            + previous_fit_overlay_retained_bytes
            + image_fit_overlay_retained_bytes
            + 2 * height * width
        )
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
        # FitResult parameters/covariance are held once; they are not copied
        # into Matplotlib artists.  The visible old overlay remains reachable
        # through Qt until the new board front is atomically presented.
        "extra_retained_evaluated_data_bytes": (
            fit_result_retained_bytes
            + previous_fit_overlay_retained_bytes
            + external_session_retained_bytes
        ),
    }
    if curve_fit_overlay_nbytes is not None:
        overlay_retained, prediction_bytes = curve_fit_overlay_nbytes
        options.update(
            fit_overlay_retained_upper_bound_bytes=overlay_retained,
            fit_prediction_upper_bound_bytes=prediction_bytes,
        )
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
    fit_result: FitResultBatch | None = None,
    fit_result_identity: str | None = None,
    previous_fit_overlay_retained_bytes: int = 0,
    external_session_retained_bytes: int = 0,
    release_initial_canonical_on_commit: bool = False,
) -> _TypedFigureFront:
    intent, unavailable_reason = _classify_single_typed(figure)
    if intent is None or intent is not _state_intent(state):
        raise ValueError(
            "typed render requires one matching logical panel"
            + ("" if unavailable_reason is None else f": {unavailable_reason}")
        )
    _require_not_cancelled(cancelled)
    external_session_retained_bytes = nonnegative_integer(
        external_session_retained_bytes,
        "external_session_retained_bytes",
    )
    if not isinstance(release_initial_canonical_on_commit, bool):
        raise TypeError("release_initial_canonical_on_commit must be bool")

    if figure.has_fit_overlays:
        if fit_result is not None or fit_result_identity is None:
            raise ValueError(
                "canonical typed Fit replay requires one caller-supplied result identity"
            )
    elif (fit_result is None) != (fit_result_identity is None):
        raise ValueError("transient typed Fit result and identity must be present together")
    fit_result_retained_bytes = (
        0
        if fit_result is None
        else fit_result_retained_upper_bound_nbytes(fit_result)
    )
    validation_peak = 0
    overlay_retained_preflight = 0
    prediction_bytes_preflight = 0
    projection_peak_preflight = 0
    if figure.has_fit_overlays or fit_result is not None:
        assert fit_result_identity is not None
        (
            validation_peak,
            overlay_retained_preflight,
            prediction_bytes_preflight,
            projection_peak_preflight,
        ) = figure.single_panel_fit_overlay_preflight_nbytes(
            fit_result,
            result_identity=fit_result_identity,
        )
    curve_overlay_nbytes = (
        (overlay_retained_preflight, prediction_bytes_preflight)
        if intent is ViewIntent.CURVE and overlay_retained_preflight
        else None
    )
    image_overlay_retained = (
        overlay_retained_preflight
        if intent is ViewIntent.IMAGE
        else 0
    )
    required = _typed_front_required_peak_bytes(
        figure,
        state,
        curve_fit_overlay_nbytes=curve_overlay_nbytes,
        image_fit_overlay_retained_bytes=image_overlay_retained,
        fit_result_retained_bytes=fit_result_retained_bytes,
        previous_fit_overlay_retained_bytes=(
            previous_fit_overlay_retained_bytes
        ),
        external_session_retained_bytes=external_session_retained_bytes,
    )
    effective_limit = _figure_render_limit(figure, memory_limit_bytes)
    from zlc_frontend.matplotlib_render import evaluated_figure_array_nbytes

    figure_retained = figure.retained_upper_bound_nbytes
    evaluated_bytes = evaluated_figure_array_nbytes(figure.evaluated)
    render_session_peak = required + max(0, figure_retained - evaluated_bytes)
    if isinstance(state, ImageDisplayState):
        image = figure.evaluated.layers[0].cells[0].series[0].data
        assert isinstance(image, EvaluatedImage)
        current_front_bytes = 2 * int(image.values.size)
    else:
        width, height = _NUMERIC_RASTER_SIZE
        current_front_bytes = width * height * 4
    planning_session_peak = (
        figure_retained
        + external_session_retained_bytes
        + fit_result_retained_bytes
        + previous_fit_overlay_retained_bytes
        + current_front_bytes
        + max(validation_peak, projection_peak_preflight)
    )
    # Pre-admit the next steady-state rerender as well.  Its "old" overlay is
    # the candidate being built now; deferring this calculation until after
    # prediction materialization would turn required-1 into a fail-late path.
    post_commit_external = 0
    next_required = _typed_front_required_peak_bytes(
        figure,
        state,
        curve_fit_overlay_nbytes=curve_overlay_nbytes,
        image_fit_overlay_retained_bytes=image_overlay_retained,
        fit_result_retained_bytes=fit_result_retained_bytes,
        previous_fit_overlay_retained_bytes=overlay_retained_preflight,
        external_session_retained_bytes=post_commit_external,
    )
    next_render_session_peak = next_required + max(
        0,
        figure_retained - evaluated_bytes,
    )
    next_planning_session_peak = (
        figure_retained
        + post_commit_external
        + fit_result_retained_bytes
        + overlay_retained_preflight
        + current_front_bytes
        + max(validation_peak, projection_peak_preflight)
    )
    concurrent_reservation = max(
        next_render_session_peak,
        next_planning_session_peak,
    )
    session_peak = max(
        render_session_peak,
        planning_session_peak,
        concurrent_reservation,
    )
    if required > effective_limit:
        raise MemoryError(
            f"interactive {intent.value.lower()} requires {required} bytes; "
            f"limit is {effective_limit}"
        )
    if session_peak > memory_limit_bytes:
        raise MemoryError(
            f"interactive {intent.value.lower()} aggregate peak {session_peak} "
            f"exceeds limit {memory_limit_bytes}"
        )
    _require_not_cancelled(cancelled)
    curve_fit_overlay_plan: CurveFitOverlayPlan | None = None
    image_fit_overlay: RadialGaussianImageFitOverlay | None = None
    if figure.has_fit_overlays:
        if intent is ViewIntent.CURVE:
            curve_fit_overlay_plan = figure.single_panel_curve_fit_overlay_plan(
                result_identity=fit_result_identity,
            )
        else:
            image_fit_overlay = figure.single_panel_radial_fit_overlay(
                result_identity=fit_result_identity,
            )
    elif fit_result is not None:
        if intent is ViewIntent.CURVE:
            curve_fit_overlay_plan = (
                figure.transient_single_panel_curve_fit_overlay_plan(
                    fit_result,
                    result_identity=fit_result_identity,
                )
            )
        else:
            image_fit_overlay = figure.transient_single_panel_radial_fit_overlay(
                fit_result,
                result_identity=fit_result_identity,
                check_cancelled=lambda: _require_not_cancelled(cancelled),
            )
    if curve_fit_overlay_plan is not None:
        exact_overlay_nbytes = estimate_curve_fit_overlay_plan_nbytes(
            curve_fit_overlay_plan
        )
        if (
            exact_overlay_nbytes[0] > overlay_retained_preflight
            or exact_overlay_nbytes[1] > prediction_bytes_preflight
        ):
            raise RuntimeError("curve Fit overlay exceeded its preflight bound")
    curve_fit_overlays: tuple[CurveFitOverlay, ...] = (
        ()
        if curve_fit_overlay_plan is None
        else materialize_curve_fit_overlay_plan(
            curve_fit_overlay_plan,
            check_cancelled=lambda: _require_not_cancelled(cancelled),
        )
    )
    _require_not_cancelled(cancelled)

    # External bytes belong to the window, not the immutable front.  They are
    # required while this candidate is built (old result/canonical cache and
    # compact authoring options), then the window either releases the old owner
    # or continues accounting the still-live options independently.
    if concurrent_reservation > memory_limit_bytes:
        raise MemoryError(
            f"interactive {intent.value.lower()} rerender reservation "
            f"{concurrent_reservation} exceeds limit {memory_limit_bytes}"
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
            image_fit_overlay,
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
                    fit_overlays=curve_fit_overlays,
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
        _typed_join_digest(figure, intent, fit_result_identity),
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
    _validate_rendered_authored_payload(payload, state, fit_result_identity)
    fit_axis_ids, axis_roles = _fit_projection_metadata(figure, intent)
    data_contract = _build_typed_front_contract(
        intent,
        effective_limit,
        frame,
    )
    return _TypedFigureFront(
        intent=intent,
        state=state,
        summary=_figure_summary(figure),
        frame=frame,
        data_contract=data_contract,
        fit_axis_ids=fit_axis_ids,
        axis_roles=axis_roles,
        fit_result_identity=fit_result_identity,
        transient_fit_result_owner=fit_result,
        transient_fit_result_retained_bytes=fit_result_retained_bytes,
        release_initial_canonical_on_commit=(
            release_initial_canonical_on_commit
        ),
        retained_figure_upper_bound_bytes=figure_retained,
        fit_overlay_retained_bytes=overlay_retained_preflight,
        session_peak_bytes=session_peak,
        concurrent_reservation_bytes=concurrent_reservation,
        required_peak_bytes=required,
        effective_limit_bytes=effective_limit,
        aggregate_limit_bytes=memory_limit_bytes,
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
        fit_overlay_renderer=None,
        *,
        memory_limit_bytes: int,
        fit_bindings: _FitWorkbenchBindings | None = None,
        typed_front_committed=None,
    ) -> None:
        if not callable(initial_loader) or not callable(typed_renderer):
            raise TypeError("figure worker callables must be callable")
        if fit_bindings is not None and not isinstance(
            fit_bindings,
            _FitWorkbenchBindings,
        ):
            raise TypeError("fit_bindings must be _FitWorkbenchBindings or None")
        if (fit_bindings is None) != (fit_overlay_renderer is None):
            raise ValueError("Fit bindings and overlay renderer must be supplied together")
        if fit_overlay_renderer is not None and not callable(fit_overlay_renderer):
            raise TypeError("fit_overlay_renderer must be callable or None")
        if typed_front_committed is not None and not callable(typed_front_committed):
            raise TypeError("typed_front_committed must be callable or None")
        self._typed_renderer = typed_renderer
        self._fit_overlay_renderer = fit_overlay_renderer
        self._fit_bindings = fit_bindings
        self._typed_front_committed = typed_front_committed
        self._view_family: str | None = None
        self._display: _TypedDisplayState | None = None
        self._typed_contract: (
            tuple[tuple[object, ...], object] | None
        ) = None
        self._typed_pages_admitted = False
        self._typed_ui_faulted = False
        self._request_revision = 0
        self._active_kind: str | None = "initial"
        self._pending_state: _TypedDisplayState | None = None
        self._pending_origin: PanelInteractionOrigin | None = None
        self._pending_editor: FluentRevisionedFormEditor | None = None
        self._pending_editor_revision: int | None = None
        self._completion_handoff_active = False
        self._deferred_typed_retry: tuple[object, ...] | None = None
        self._edit_display: FluentRevisionedFormEditor | None = None
        self._setting_display: FluentRevisionedFormEditor | None = None
        self._export_commit_lock = threading.Lock()
        self._current_front_peak_bytes = 0
        self._visible_fit_overlay_retained_bytes = 0
        self._visible_transient_fit_result_owner: FitResultBatch | None = None
        self._visible_transient_fit_result_retained_bytes = 0
        self._fit_axis_ids: tuple[AxisId, ...] = ()
        self._fit_axis_roles: tuple[tuple[AxisId, AxisViewRole], ...] = ()
        self._visible_fit_result_identity: str | None = None

        self._fit_future: Future | None = None
        self._fit_job_kind: str | None = None
        self._fit_job_revision: int | None = None
        self._fit_analysis_revision = 0
        self._fit_prepare_pending = False
        self._fit_overlay_desired: _FitOverlayRequest | None = None
        self._fit_overlay_pending: _FitOverlayRequest | None = None
        self._fit_overlay_inflight: _FitOverlayRequest | None = None
        self._fit_candidate: _FitSelectionCandidate | None = None
        self._fit_initial_selection_consumed = False
        self._fit_auto_open_consumed = False
        self._fit_options: dict[str, FitAuthoringOption] = {}
        self._fit_options_retained_bytes = 0
        self._fit_options_release_pending = False
        self._fit_cancelled: threading.Event | None = None
        self._fit_execution_limit_bytes = 0
        self._fit_save_limit_bytes = 0
        self._fit_draft: FitDraftResult | None = None
        self._fit_draft_summary: str | None = None
        self._fit_save_inflight: FitDraftResult | None = None
        self._deferred_fit_reload: tuple[FitResultArtifactRef, int] | None = None
        self._saved_fit_reference: FitResultArtifactRef | None = None
        self._close_deferred_during_fit_save = False
        self._fit_pane: FitAuthoringPane | None = None
        self._fit_authority: FitDraftAuthority | None = None
        if fit_bindings is not None:
            self._fit_authority = FitDraftAuthority(
                lambda spec, cancel_check, deadline: fit_bindings.execute(
                    spec,
                    cancel_check,
                    deadline,
                    self._fit_execution_limit_bytes,
                ),
                lambda execution: fit_bindings.save(
                    execution,
                    self._fit_save_limit_bytes,
                ),
            )

        super().__init__(
            None,
            window_title="Data Figure",
            mode_text="FROZEN DATA FIGURE · INTERACTIVE",
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
        self._analyze_button = FluentButton("Analyze → Fit", self, color=ORANGE)
        self._analyze_button.setObjectName("figureViewerAnalyzeFitButton")
        self._controls.insertWidget(0, self._interaction_switch)
        self._controls.insertWidget(1, self._settings_button)
        self._controls.insertWidget(2, self._analyze_button)
        self._controls.insertWidget(3, self._export_button)
        for widget in (
            self._interaction_switch,
            self._settings_button,
            self._analyze_button,
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
        self._analyze_button.clicked.connect(self._open_fit_analysis)
        self._interaction_switch.toggled.connect(self._toggle_interaction)
        if fit_bindings is not None:
            pane = FitAuthoringPane(self._tabs)
            pane.setObjectName("figureViewerFitAuthoring")
            pane.fitRequested.connect(self._start_fit)
            pane.fitRequestRejected.connect(self._reject_fit_request)
            pane.cancelRequested.connect(self._cancel_fit)
            pane.saveRequested.connect(self._save_fit)
            pane.clearRequested.connect(self._clear_fit)
            pane.clearSelectionRequested.connect(self._clear_fit_selection)
            pane.editorChanged.connect(self._fit_editor_changed)
            pane.optionsReleased.connect(self._fit_option_widgets_released)
            pane.hide()
            self._fit_pane = pane
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

    @property
    def worker_idle(self) -> bool:
        return bool(
            self._future is None
            and self._fit_future is None
            and self._fit_overlay_pending is None
            and not self._fit_prepare_pending
            and self._deferred_typed_retry is None
            and self._deferred_fit_reload is None
            and not self._completion_handoff_active
            and not self._fit_options_release_pending
        )

    @property
    def draft_ready(self) -> bool:
        return self._fit_draft is not None

    @property
    def saved_reference(self) -> FitResultArtifactRef | None:
        return self._saved_fit_reference

    @property
    def fit_models(self) -> tuple[str, ...]:
        return tuple(self._fit_options)

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
        self._analyze_button.setEnabled(
            active
            and self._fit_bindings is not None
            and self._view_family in ("curve", "image")
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

    def _fit_analysis_is_open(self) -> bool:
        pane = self._fit_pane
        return pane is not None and self._tabs.indexOf(pane) >= 0

    def _fit_authoring_busy_kind(self) -> str | None:
        kind = self._fit_job_kind
        if kind in ("prepare", "fit", "save"):
            return kind
        if kind == "reload_saved":
            return "render"
        if (
            self._future is not None
            or self._fit_overlay_pending is not None
            or self._fit_overlay_inflight is not None
            or self._deferred_fit_reload is not None
        ):
            return "render"
        if self._fit_prepare_pending:
            return "prepare"
        return None

    def _sync_fit_authoring_busy(self) -> None:
        pane = self._fit_pane
        if pane is not None and not self._closing:
            pane.set_busy(
                self._fit_authoring_busy_kind(),
                draft_ready=self._fit_draft is not None,
            )

    def _open_fit_analysis(self) -> None:
        pane = self._fit_pane
        if (
            pane is None
            or self._closing
            or self._view_family not in ("curve", "image")
        ):
            return
        if self._tabs.indexOf(pane) < 0:
            self._tabs.addTab(pane, "Analysis")
            pane.show()
        self._tabs.setCurrentWidget(pane)
        self._start_fit_prepare()

    def _submit_fit_future(self, kind: str, function, *args) -> bool:
        if self._fit_future is not None:
            raise RuntimeError("Fit worker already has active work")
        try:
            future = _FIT_WORK_EXECUTOR.submit(function, *args)
        except BaseException as error:
            self._status.setText("FIT SUBMISSION FAILED")
            self._diagnostic.setText(error_summary(error))
            return False
        self._fit_job_kind = kind
        self._fit_future = future
        future.add_done_callback(lambda _done: self._wake.request_owner_wake())
        return True

    def _current_fit_selection(self) -> Selection | None:
        candidate = self._fit_candidate
        return None if candidate is None else candidate.selection

    def _start_fit_prepare(self) -> None:
        bindings = self._fit_bindings
        pane = self._fit_pane
        if (
            bindings is None
            or pane is None
            or self._closing
            or not self._fit_axis_ids
            or not self._fit_analysis_is_open()
        ):
            return
        if self._fit_options_release_pending:
            self._fit_prepare_pending = True
            return
        if self._completion_handoff_active:
            self._fit_prepare_pending = True
            return
        if self._fit_future is not None:
            self._fit_prepare_pending = True
            return
        self._fit_prepare_pending = False
        residual = self._fit_operation_residual_bytes()
        if residual <= 0:
            pane.set_busy(None, draft_ready=self._fit_draft is not None)
            self._status.setText("FIT PREPARATION REJECTED")
            self._diagnostic.setText(
                "visible Figure front leaves no Fit preparation budget"
            )
            return
        pane.set_busy("prepare", draft_ready=self._fit_draft is not None)
        self._status.setText("PREPARING FIT")
        self._diagnostic.setText("")
        self._fit_job_revision = self._fit_analysis_revision
        if not self._submit_fit_future(
            "prepare",
            _prepare_fit_options,
            bindings.prepare,
            self._fit_axis_ids,
            self._fit_axis_roles,
            self._current_fit_selection(),
            residual,
        ):
            self._fit_job_revision = None
            pane.set_busy(None, draft_ready=self._fit_draft is not None)

    def _discard_fit_draft(self) -> None:
        draft, self._fit_draft = self._fit_draft, None
        self._fit_draft_summary = None
        authority = self._fit_authority
        if draft is not None and authority is not None:
            authority.discard(draft)

    def _advance_fit_analysis(self, *, prepare: bool) -> None:
        self._fit_analysis_revision += 1
        # A saved-result reload belongs to the exact editor revision that
        # requested the save; never occupy the lane with stale authority.
        self._deferred_fit_reload = None
        self._discard_fit_draft()
        self._queue_fit_overlay(None, None)
        if self._fit_job_kind == "fit" and self._fit_cancelled is not None:
            self._fit_cancelled.set()
        if prepare:
            self._fit_options = {}
            pane = self._fit_pane
            release_pending = self._fit_options_release_pending
            if pane is not None:
                released_later = pane.clear_options()
                release_pending = release_pending or released_later
            # Remove the old selected-axis QString while its conservative
            # option charge is still held.
            self._summary.setText("")
            # Qt destroys the detached form through DeferredDelete.  Keep its
            # conservative option charge until a fresh queued owner turn; a
            # same-callback prepare could otherwise overlap the old widgets
            # with a full new option set outside the aggregate budget.
            self._fit_options_release_pending = release_pending
            if not release_pending:
                self._fit_options_retained_bytes = 0
            self._fit_prepare_pending = True
            self._wake.request_owner_wake()

    @QtCore.pyqtSlot()
    def _fit_option_widgets_released(self) -> None:
        """Release old option charge only after Qt confirms widget destruction."""

        if not self._fit_options_release_pending:
            return
        self._fit_options_release_pending = False
        self._fit_options_retained_bytes = 0
        if self._fit_prepare_pending and not self._closing:
            self._wake.request_owner_wake()

    def _fit_editor_changed(self, _pane_revision: int) -> None:
        if self._closing:
            return
        self._advance_fit_analysis(prepare=False)
        self._status.setText("FIT DRAFT CHANGED")
        self._diagnostic.setText(
            "The visible model or constraints changed; press Fit to submit the "
            "new authoritative draft."
        )

    def _reject_fit_request(self, diagnostic: str) -> None:
        self._status.setText("FIT REQUEST INVALID")
        self._diagnostic.setText(str(diagnostic))

    def _fit_operation_limit(self) -> int:
        residual = self._fit_operation_residual_bytes()
        if residual <= 0:
            raise MemoryError("visible Figure front leaves no Fit operation budget")
        return residual

    def _unpresented_fit_retained_bytes(self) -> int:
        retained = 0
        seen: list[FitResultBatch] = []
        for request in (
            self._fit_overlay_desired,
            self._fit_overlay_pending,
            self._fit_overlay_inflight,
        ):
            if request is None or request.result is None:
                continue
            if request.result is self._visible_transient_fit_result_owner:
                continue
            if any(request.result is item for item in seen):
                continue
            seen.append(request.result)
            retained += request.result_retained_bytes
        return retained

    def _fit_operation_residual_bytes(self) -> int:
        return int(
            self._memory_limit_bytes
            - self._current_front_peak_bytes
            - self._unpresented_fit_retained_bytes()
            - self._fit_options_retained_bytes
        )

    def _render_external_retained_bytes(
        self,
        incoming_result: FitResultBatch | None,
    ) -> int:
        old_result_bytes = (
            self._visible_transient_fit_result_retained_bytes
            if self._visible_transient_fit_result_owner is not None
            and self._visible_transient_fit_result_owner is not incoming_result
            else 0
        )
        return int(self._fit_options_retained_bytes + old_result_bytes)

    def _start_fit(self, _pane_revision: int, spec: FitSpec) -> None:
        pane = self._fit_pane
        authority = self._fit_authority
        bindings = self._fit_bindings
        if (
            pane is None
            or authority is None
            or bindings is None
            or self._closing
            or self._fit_future is not None
            or self._deferred_fit_reload is not None
        ):
            return
        try:
            current = pane.current_option()
            if not isinstance(spec, FitSpec):
                raise TypeError("Fit pane emitted another request type")
            if (
                spec.model_id != current.spec.model_id
                or spec.input_schema_fingerprint
                != current.spec.input_schema_fingerprint
                or spec.committed_transform != current.spec.committed_transform
                or spec.fit_axis_ids != current.spec.fit_axis_ids
                or spec.batch_axis_ids != current.spec.batch_axis_ids
                or spec.numeric_policy != current.spec.numeric_policy
            ):
                raise ValueError("Fit request differs from the prepared authority draft")
            operation_limit = self._fit_operation_limit()
        except BaseException as error:
            self._status.setText("FIT REQUEST INVALID")
            self._diagnostic.setText(error_summary(error))
            return

        self._discard_fit_draft()
        self._fit_execution_limit_bytes = operation_limit
        self._fit_cancelled = threading.Event()
        self._fit_job_revision = self._fit_analysis_revision
        deadline = time.monotonic() + bindings.timeout_seconds
        pane.set_busy("fit", draft_ready=False)
        self._status.setText("FITTING")
        self._summary.setText(pane.axis_summary.text())
        self._diagnostic.setText("")
        self._queue_fit_overlay(None, None, result_retained_bytes=0)
        if not self._submit_fit_future(
            "fit",
            _execute_fit_draft,
            authority,
            spec,
            deadline,
            self._cancelled,
            self._fit_cancelled,
        ):
            self._fit_job_revision = None
            self._fit_cancelled = None
            pane.set_busy(None, draft_ready=False)

    def _cancel_fit(self) -> None:
        if self._fit_job_kind != "fit" or self._fit_cancelled is None:
            return
        self._fit_cancelled.set()
        pane = self._fit_pane
        if pane is not None:
            pane.cancel_button.setEnabled(False)
        self._status.setText("CANCELLING FIT")

    def _save_fit(self) -> None:
        pane = self._fit_pane
        authority = self._fit_authority
        bindings = self._fit_bindings
        draft = self._fit_draft
        if (
            pane is None
            or authority is None
            or bindings is None
            or draft is None
            or self._fit_future is not None
            or self._closing
        ):
            return
        residual = self._fit_operation_residual_bytes()
        if residual <= 0:
            self._status.setText("FIT SAVE REJECTED")
            self._diagnostic.setText("visible Figure front leaves no reload budget")
            return
        self._fit_save_inflight = draft
        self._fit_save_limit_bytes = residual
        self._fit_job_revision = self._fit_analysis_revision
        pane.set_busy("save", draft_ready=True)
        self._status.setText("SAVING FIT")
        self._summary.setText("Publishing and reopening the exact Fit artifact…")
        self._diagnostic.setText("")
        if not self._submit_fit_future(
            "save",
            authority.save,
            draft,
        ):
            self._fit_save_inflight = None
            self._fit_save_limit_bytes = 0
            self._fit_job_revision = None
            pane.set_busy(None, draft_ready=True)

    def _clear_fit(self) -> None:
        if self._closing or self._fit_future is not None:
            return
        self._fit_analysis_revision += 1
        self._discard_fit_draft()
        self._queue_fit_overlay(None, None)
        pane = self._fit_pane
        if pane is not None:
            pane.set_busy(None, draft_ready=False)
        self._status.setText("FIT CLEARED")
        self._summary.setText("Source view retained; selection remains a draft candidate")
        self._diagnostic.setText("")

    def _queue_fit_overlay(
        self,
        result: FitResultBatch | None,
        result_identity: str | None,
        *,
        result_retained_bytes: int = 0,
    ) -> None:
        if self._fit_overlay_renderer is None or self._display is None:
            return
        request = _FitOverlayRequest(
            self._fit_analysis_revision,
            result,
            result_identity,
            result_retained_bytes,
        )
        self._fit_overlay_desired = request
        if (
            self._fit_overlay_inflight is None
            and self._fit_overlay_pending is None
            and self._visible_fit_result_identity == result_identity
        ):
            return
        self._fit_overlay_pending = request
        self._start_pending_fit_overlay()

    def _start_pending_fit_overlay(self) -> None:
        request = self._fit_overlay_pending
        display = self._display
        renderer = self._fit_overlay_renderer
        if (
            request is None
            or display is None
            or renderer is None
            or self._future is not None
            or self._closing
            or self._completion_handoff_active
        ):
            return
        candidate = replace(display, revision=display.revision + 1)
        self._request_revision += 1
        self._active_kind = "fit_overlay"
        self._pending_state = candidate
        self._fit_overlay_inflight = request
        self._fit_overlay_pending = None
        self._status.setText("RENDERING FIT OVERLAY")
        self._set_typed_controls_enabled(False)
        previous_scale = (
            display.count_scale
            if isinstance(display, HistogramDisplayState)
            else None
        )
        if not self._submit_future(
            renderer,
            request.result,
            request.result_identity,
            candidate,
            self._visible_value_limits(),
            display.relim_mode,
            previous_scale,
            self._request_revision,
            self._memory_limit_bytes,
            self._cancelled,
            self._visible_fit_overlay_retained_bytes,
            self._render_external_retained_bytes(request.result),
        ):
            self._fit_overlay_inflight = None
            self._pending_state = None
            self._active_kind = None
            self._set_typed_controls_enabled(True)
        else:
            self._sync_fit_authoring_busy()

    @staticmethod
    def _curve_span_for_selection(
        selection: Selection,
        payload: CurvePanelPayload,
    ) -> tuple[float, float]:
        axis = payload.viewport.x_axis
        matches = tuple(term for term in selection.terms if term.axis_id == axis.axis_id)
        if len(matches) != 1:
            raise ValueError("curve Fit selection does not name the displayed x axis")
        term = matches[0]
        if isinstance(term, CoordinateRangeSelection):
            return float(term.lower), float(term.upper)
        if isinstance(term, IndexRangeSelection):
            coordinates = axis.coordinates
            if term.stop > len(coordinates):
                raise IndexError("curve Fit index range exceeds displayed coordinates")
            low = float(coordinates[term.start])
            high = float(coordinates[term.stop - 1])
            return (min(low, high), max(low, high))
        raise ValueError("curve Fit selection must preserve a non-empty range")

    def _reapply_fit_candidate(self) -> None:
        candidate = self._fit_candidate
        pane = self._fit_pane
        if pane is not None:
            pane.set_selection_present(candidate is not None)
        if candidate is None:
            return
        origin = self._visible_typed_origin()
        payload = self._visible_typed_payload()
        if origin is None or payload is None:
            return
        if isinstance(payload, CurvePanelPayload):
            self._board_widget.set_curve_range_candidate(
                self._curve_span_for_selection(candidate.selection, payload),
                panel_id=_TYPED_PANEL_ID,
            )
        elif isinstance(payload, ImagePanelPayload):
            self._board_widget.set_selector_applied_selection(
                candidate.selection,
                panel_id=_TYPED_PANEL_ID,
            )
        else:
            return
        self._fit_candidate = _FitSelectionCandidate(origin, candidate.selection)

    def _accept_fit_selection_candidate(
        self,
        origin: PanelInteractionOrigin,
        selection: Selection | None,
    ) -> None:
        if self._fit_bindings is None:
            return
        self._fit_candidate = (
            None
            if selection is None
            else _FitSelectionCandidate(origin, selection)
        )
        pane = self._fit_pane
        if pane is not None:
            pane.set_selection_present(selection is not None)
        self._advance_fit_analysis(prepare=self._fit_analysis_is_open())
        self._status.setText("FIT SELECTION READY" if selection is not None else "FIT SELECTION CLEARED")
        self._diagnostic.setText(
            "Press Fit to submit this named-axis selection as authority."
            if selection is not None
            else "Fit will use the full named dataset; current zoom is not authority."
        )

    def _clear_fit_selection(self) -> None:
        if self._closing or self._fit_candidate is None:
            return
        origin = self._visible_typed_origin()
        if origin is None:
            raise RuntimeError("Fit selection has no current exact panel")
        if self._view_family == "image":
            self._board_widget.set_image_rectangle_candidate(
                None,
                panel_id=_TYPED_PANEL_ID,
            )
        elif self._view_family == "curve":
            self._board_widget.set_curve_range_candidate(
                None,
                panel_id=_TYPED_PANEL_ID,
            )
        else:
            raise RuntimeError("Fit selection belongs to another view family")
        self._accept_fit_selection_candidate(origin, None)

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
        selection = None
        if self._fit_bindings is not None:
            # Resolve authority while QtRasterBoard still holds the exact front
            # on which this gesture was completed.  Painting the candidate first
            # would release that proof and make a later conversion racy.
            selection = self._board_widget.selection_for_rectangle_gesture(gesture)
        self._board_widget.set_image_rectangle_candidate(
            gesture.normalized_bounds,
            panel_id=_TYPED_PANEL_ID,
        )
        if selection is not None:
            self._accept_fit_selection_candidate(origin, selection)
            return
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
            selection = None
            if is_curve and self._fit_bindings is not None and command.x_span is not None:
                # As with IMAGE, the exact held origin must be consumed before
                # set_curve_range_candidate finalizes the display-only gesture.
                selection = self._board_widget.selection_for_curve_range_gesture(
                    command
                )
            setter = (
                self._board_widget.set_curve_range_candidate
                if is_curve
                else self._board_widget.set_histogram_range_candidate
            )
            setter(command.x_span, panel_id=_TYPED_PANEL_ID)
            if is_curve and self._fit_bindings is not None:
                self._accept_fit_selection_candidate(origin, selection)
                return
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
        if self._completion_handoff_active:
            self._deferred_typed_retry = (
                candidate,
                editor,
                editor_revision,
                origin,
            )
            return
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
        overlay = self._fit_overlay_desired
        overlay_result = None if overlay is None else overlay.result
        overlay_identity = None if overlay is None else overlay.result_identity
        self._fit_overlay_inflight = overlay
        submitted = self._submit_future(
            self._typed_renderer,
            overlay_result,
            overlay_identity,
            candidate,
            self._visible_value_limits(),
            display.relim_mode,
            previous_scale,
            self._request_revision,
            self._memory_limit_bytes,
            self._cancelled,
            self._visible_fit_overlay_retained_bytes,
            self._render_external_retained_bytes(overlay_result),
        )
        if not submitted:
            self._fit_overlay_inflight = None
            self._discard_pending_typed()
        else:
            self._sync_fit_authoring_busy()

    def _discard_pending_typed(self) -> None:
        origin = self._pending_origin
        family = self._view_family
        self._fit_overlay_inflight = None
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
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        # The worker performs the data-sized proof.  Qt repeats only bounded
        # display fields and immutable object identities; it never rescans a
        # coordinate vector or trusts a token detached from the current frame.
        if (
            front.state != expected_state
            or front.intent is not _state_intent(expected_state)
        ):
            raise ValueError("typed worker returned conflicting authored state")
        payload = front.frame.panels[0].display_payload
        assert isinstance(
            payload,
            (ImagePanelPayload, CurvePanelPayload, HistogramPanelPayload),
        )
        _validate_rendered_authored_payload(
            payload,
            expected_state,
            front.fit_result_identity,
        )
        current = _typed_front_contract(front)
        frozen_identity, frozen_data = front.data_contract
        identity, exact_data = current
        if identity != frozen_identity:
            raise ValueError("typed worker changed frozen source provenance")
        if not _same_exact_data_owners(exact_data, frozen_data):
            raise ValueError("typed worker changed frozen evaluated data")
        return current

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
        contract = self._validate_authored_front(front, expected_state)
        expected_contract = self._typed_contract
        if expected_contract is not None:
            expected_identity, expected_data = expected_contract
            identity, exact_data = contract
            if identity[0] != expected_identity[0]:
                raise ValueError("typed worker changed frozen source provenance")
            if not _same_exact_data_owners(exact_data, expected_data):
                raise ValueError("typed worker changed frozen evaluated data")

        self._board_widget.present(front.frame)
        if self._typed_front_committed is not None:
            self._typed_front_committed(
                front.release_initial_canonical_on_commit
            )
        # The admitted board front is the transaction boundary.  Commit the
        # exact authored state/contract before any optional Qt chrome work.
        if expected_contract is None:
            self._typed_contract = contract
        self._display = expected_state
        self._view_family = front.intent.value.lower()
        self._current_front_peak_bytes = front.concurrent_reservation_bytes
        self._visible_fit_overlay_retained_bytes = front.fit_overlay_retained_bytes
        self._fit_axis_ids = front.fit_axis_ids
        self._fit_axis_roles = front.axis_roles
        self._visible_fit_result_identity = front.fit_result_identity
        self._visible_transient_fit_result_owner = (
            front.transient_fit_result_owner
        )
        self._visible_transient_fit_result_retained_bytes = (
            front.transient_fit_result_retained_bytes
        )
        if self._fit_overlay_desired is None:
            self._fit_overlay_desired = _FitOverlayRequest(
                self._fit_analysis_revision,
                None,
                front.fit_result_identity,
            )
        # Page/chrome and controls are ancillary to the already-admitted
        # immutable data front.  Their faults can disable UI, never roll it back.
        try:
            if not self._typed_pages_admitted:
                self._retire_tab_pages()
                self._tabs.addTab(self._typed_page, front.intent.value.title())
                self._tabs.tabBar().setVisible(False)
                self._typed_page.show()
                self._typed_pages_admitted = True
            fit_capable = (
                self._fit_bindings is not None
                and front.intent in (ViewIntent.CURVE, ViewIntent.IMAGE)
            )
            self._mode.setText(
                f"EXACT {front.intent.value} · INTERACTIVE"
                + ("" if fit_capable else " · DISPLAY ONLY")
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
            if (
                self._fit_bindings is not None
                and front.intent in (ViewIntent.CURVE, ViewIntent.IMAGE)
            ):
                self._analyze_button.show()
                if not self._fit_initial_selection_consumed:
                    initial = self._fit_bindings.initial_selection
                    self._fit_initial_selection_consumed = True
                    origin = self._visible_typed_origin()
                    if initial is not None and origin is not None:
                        self._fit_candidate = _FitSelectionCandidate(origin, initial)
                self._reapply_fit_candidate()
                if (
                    self._fit_bindings.open_analysis
                    and not self._fit_auto_open_consumed
                ):
                    self._fit_auto_open_consumed = True
                    self._open_fit_analysis()
            else:
                self._analyze_button.hide()
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

    def _accept_finished_fit_future(self, future: Future) -> None:
        kind, self._fit_job_kind = self._fit_job_kind, None
        job_revision, self._fit_job_revision = self._fit_job_revision, None
        fit_cancelled, self._fit_cancelled = self._fit_cancelled, None
        save_inflight, self._fit_save_inflight = self._fit_save_inflight, None
        close_after_save = bool(
            kind == "save" and self._close_deferred_during_fit_save
        )
        if kind == "save":
            self._close_deferred_during_fit_save = False
            self._fit_save_limit_bytes = 0
        pane = self._fit_pane
        completed_draft: FitDraftResult | None = None
        completed_summary: str | None = None
        completed_result_retained_bytes = 0
        try:
            result = future.result()
            if kind == "prepare":
                options = (
                    () if job_revision != self._fit_analysis_revision else tuple(result)
                )
            elif kind == "fit":
                if (
                    not isinstance(result, tuple)
                    or len(result) != 3
                    or not isinstance(result[0], FitDraftResult)
                    or not isinstance(result[1], str)
                    or isinstance(result[2], bool)
                    or not isinstance(result[2], int)
                    or result[2] <= 0
                ):
                    raise TypeError("Fit worker returned another draft type")
                (
                    completed_draft,
                    completed_summary,
                    completed_result_retained_bytes,
                ) = result
                if fit_cancelled is not None and fit_cancelled.is_set():
                    raise CancelledError()
            elif kind == "save":
                if not isinstance(result, FitResultArtifactRef):
                    raise TypeError("Fit save returned another reference type")
                reference = result
            elif kind == "reload_saved":
                if (
                    not isinstance(result, tuple)
                    or len(result) != 2
                    or not isinstance(result[0], FitResultBatch)
                    or isinstance(result[1], bool)
                    or not isinstance(result[1], int)
                    or result[1] <= 0
                ):
                    raise TypeError("saved Fit reload returned another result type")
                reloaded_result, reloaded_result_retained_bytes = result
            else:
                raise RuntimeError(f"unknown Fit worker completion {kind!r}")
        except (CancelledError, FitCancelled):
            if completed_draft is not None and self._fit_authority is not None:
                self._fit_authority.discard(completed_draft)
            if not self._closing:
                self._status.setText("FIT CANCELLED")
                self._diagnostic.setText("")
        except FitDeadlineExceeded as error:
            if not self._closing:
                self._status.setText("FIT DEADLINE EXCEEDED")
                self._diagnostic.setText(error_summary(error))
        except BaseException as error:
            if completed_draft is not None and self._fit_authority is not None:
                self._fit_authority.discard(completed_draft)
            if not self._closing:
                label = {
                    "prepare": "FIT PREPARATION FAILED",
                    "fit": "FIT FAILED",
                    "save": "FIT SAVE FAILED",
                    "reload_saved": "FIT SAVED · REOPEN FAILED",
                }.get(kind, "FIT WORK FAILED")
                self._status.setText(label)
                self._diagnostic.setText(error_summary(error))
                if kind == "save" and save_inflight is not None:
                    if job_revision == self._fit_analysis_revision:
                        self._fit_draft = save_inflight
                    else:
                        if self._fit_authority is not None:
                            self._fit_authority.discard(save_inflight)
                        self._status.setText("FIT SAVE FAILED · EDITOR CHANGED")
        else:
            if self._closing:
                if completed_draft is not None and self._fit_authority is not None:
                    self._fit_authority.discard(completed_draft)
            elif kind == "prepare":
                if job_revision != self._fit_analysis_revision:
                    self._fit_prepare_pending = True
                else:
                    assert pane is not None
                    preferred = pane.model_combo.currentData()
                    model_ids = {option.spec.model_id for option in options}
                    if preferred not in model_ids:
                        preferred = (
                            None
                            if self._fit_bindings is None
                            else self._fit_bindings.selected_model
                        )
                    if preferred not in model_ids:
                        preferred = None
                    pane.install_options(options, selected_model=preferred)
                    self._fit_options = {
                        option.spec.model_id: option for option in options
                    }
                    self._fit_options_retained_bytes = sum(
                        option.retained_upper_bound_bytes for option in options
                    )
                    self._status.setText("FIT READY")
                    self._summary.setText(pane.axis_summary.text())
                    self._diagnostic.setText(
                        "Fit submits the displayed named axes; zoom is presentation only."
                    )
            elif kind == "fit":
                assert completed_draft is not None
                assert completed_summary is not None
                if job_revision != self._fit_analysis_revision:
                    assert self._fit_authority is not None
                    self._fit_authority.discard(completed_draft)
                    self._status.setText("STALE FIT DISCARDED")
                else:
                    self._fit_draft = completed_draft
                    self._fit_draft_summary = completed_summary
                    identity = (
                        f"draft-fit:r{job_revision}:g{completed_draft.generation}"
                    )
                    self._queue_fit_overlay(
                        completed_draft.result,
                        identity,
                        result_retained_bytes=completed_result_retained_bytes,
                    )
                    self._status.setText("DRAFT FIT READY")
                    self._summary.setText(completed_summary)
                    self._diagnostic.setText("")
            elif kind == "save":
                # The durable ref is accepted before any decode or rendering.
                # Nothing after this assignment may erase it.
                self._saved_fit_reference = reference
                self._fit_draft = None
                self._fit_draft_summary = None
                if job_revision != self._fit_analysis_revision:
                    self._status.setText("FIT SAVED · EDITOR CHANGED")
                else:
                    self._status.setText("FIT SAVED · REOPENING")
                self._summary.setText(
                    f"{reference.repository_id}:{reference.manifest_digest}"
                )
                bindings = self._fit_bindings
                residual = self._fit_operation_residual_bytes()
                if close_after_save:
                    self._diagnostic.setText(
                        "Artifact reference accepted; completing the deferred close."
                    )
                elif job_revision != self._fit_analysis_revision:
                    self._diagnostic.setText(
                        "Artifact is saved; stale editor authority was not reloaded."
                    )
                elif bindings is None or residual <= 0:
                    self._diagnostic.setText(
                        "Artifact is saved; the visible Figure leaves no reopen budget."
                    )
                else:
                    self._deferred_fit_reload = (reference, job_revision)
                    if pane is not None:
                        pane.set_busy("render", draft_ready=False)
            elif kind == "reload_saved":
                reference = self._saved_fit_reference
                if reference is None:
                    raise RuntimeError("saved Fit reload lost its durable reference")
                if job_revision == self._fit_analysis_revision:
                    identity = f"{reference.repository_id}:{reference.manifest_digest}"
                    self._queue_fit_overlay(
                        reloaded_result,
                        identity,
                        result_retained_bytes=reloaded_result_retained_bytes,
                    )
                    self._status.setText("FIT SAVED")
                    self._diagnostic.setText("")
                else:
                    self._status.setText("FIT SAVED · EDITOR CHANGED")
                    self._diagnostic.setText(
                        "Saved artifact retained; its overlay was not applied to a newer draft."
                    )
        if close_after_save and not self._closing:
            self.shutdown()
            return
        if pane is not None and not self._closing:
            self._sync_fit_authoring_busy()

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
            rendered_overlay = self._fit_overlay_inflight
            self._fit_overlay_inflight = None
            if not _same_fit_overlay_request(
                rendered_overlay,
                self._fit_overlay_desired,
            ):
                if _same_fit_overlay_request(
                    self._fit_overlay_pending,
                    self._fit_overlay_desired,
                ):
                    self._fit_overlay_pending = None
                editor = self._pending_editor
                editor_revision = self._pending_editor_revision
                origin = self._pending_origin
                self._active_kind = None
                self._start_typed_render(
                    pending,
                    editor=editor,
                    editor_revision=editor_revision,
                    origin=origin,
                )
                return
            expected_overlay_identity = (
                None
                if rendered_overlay is None
                else rendered_overlay.result_identity
            )
            if result.fit_result_identity != expected_overlay_identity:
                raise ValueError(
                    "typed worker returned another Fit result identity"
                )
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
        if kind == "fit_overlay":
            if not isinstance(result, _TypedFigureFront):
                raise TypeError("Fit overlay worker returned another result")
            pending = self._pending_state
            request = self._fit_overlay_inflight
            if pending is None or request is None:
                raise RuntimeError("Fit overlay completed without an admitted request")
            self._pending_state = None
            self._fit_overlay_inflight = None
            self._active_kind = None
            if not _same_fit_overlay_request(
                request,
                self._fit_overlay_desired,
            ):
                self._set_typed_controls_enabled(True)
                return
            if result.fit_result_identity != request.result_identity:
                raise ValueError("Fit overlay worker returned another result identity")
            self._present_typed_front(
                result,
                expected_state=pending,
                request_revision=self._request_revision,
            )
            if self._fit_job_kind == "fit":
                self._status.setText("FITTING")
            elif self._fit_draft is not None:
                self._status.setText("DRAFT FIT READY")
                if self._fit_draft_summary is None:
                    raise RuntimeError("Fit draft summary was not retained from its worker")
                self._summary.setText(self._fit_draft_summary)
            elif self._saved_fit_reference is not None and request.result is not None:
                self._status.setText("FIT SAVED")
            else:
                self._status.setText("READY")
            self._set_typed_controls_enabled(True)
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
            self._fit_overlay_inflight = None
            self._discard_pending_typed()
        elif kind == "fit_overlay":
            self._status.setText("FIT OVERLAY FAILED")
            self._diagnostic.setText(error_summary(error))
            self._pending_state = None
            self._fit_overlay_inflight = None
            self._active_kind = None
            self._set_typed_controls_enabled(True)
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
        export_limit = self._memory_limit_bytes - self._fit_options_retained_bytes
        if export_limit <= 0:
            self._active_kind = None
            self._set_typed_controls_enabled(True)
            self._status.setText("TYPED EXPORT REJECTED")
            self._diagnostic.setText(
                "Fit authoring options leave no aggregate export budget."
            )
            return
        if not self._submit_future(
            _export_typed_png,
            frame,
            display,
            Path(destination),
            export_limit,
            self._request_revision,
            self._cancelled,
            self._export_commit_lock,
        ):
            self._active_kind = None
            self._set_typed_controls_enabled(True)
        else:
            self._sync_fit_authoring_busy()

    def _clear_bundle(self) -> None:
        super()._clear_bundle()
        if self._view_family in ("image", "curve", "histogram"):
            self._board_widget.clear()

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        # Raster owns the currently visible front, so accept it first.  Fit may
        # then enqueue the newest overlay against that accepted viewport in the
        # same Qt-owner turn without either future touching a QWidget.
        consumed_completion = False
        self._completion_handoff_active = True
        try:
            raster_future = self._future
            if raster_future is not None and raster_future.done():
                self._future = None
                self._accept_finished_future(raster_future)
                consumed_completion = True
            fit_future = self._fit_future
            if fit_future is not None and fit_future.done():
                self._fit_future = None
                self._accept_finished_fit_future(fit_future)
                consumed_completion = True
        finally:
            self._completion_handoff_active = False
        if consumed_completion:
            # The completed Future/traceback can retain a whole rejected front
            # until this callback unwinds.  Resume only on a fresh queued turn.
            self._wake.request_owner_wake()
        elif not self._closing:
            retry = self._deferred_typed_retry
            if retry is not None and self._future is None:
                self._deferred_typed_retry = None
                candidate, editor, editor_revision, origin = retry
                self._start_typed_render(
                    candidate,
                    editor=editor,
                    editor_revision=editor_revision,
                    origin=origin,
                )
            if self._future is None:
                self._start_pending_fit_overlay()
            deferred_reload = self._deferred_fit_reload
            if deferred_reload is not None and self._fit_future is None:
                reference, revision = deferred_reload
                self._deferred_fit_reload = None
                bindings = self._fit_bindings
                residual = self._fit_operation_residual_bytes()
                if (
                    revision == self._fit_analysis_revision
                    and bindings is not None
                    and residual > 0
                ):
                    self._fit_job_revision = revision
                    if not self._submit_fit_future(
                        "reload_saved",
                        _reload_fit_result_with_retained,
                        bindings.reload,
                        reference,
                        residual,
                    ):
                        self._fit_job_revision = None
            if self._fit_prepare_pending and self._fit_future is None:
                self._start_fit_prepare()
            self._sync_fit_authoring_busy()
        self._finish_close_if_ready()

    def _finish_close_if_ready(self) -> None:
        if self._fit_future is not None:
            return
        if self._closing and self._future is None and not self._closed:
            self._fit_draft = None
            self._fit_draft_summary = None
            self._fit_save_inflight = None
            self._fit_candidate = None
            self._fit_options.clear()
            self._fit_options_retained_bytes = 0
            self._fit_options_release_pending = False
            pane = self._fit_pane
            if pane is not None:
                pane.clear_options()
            self._fit_authority = None
            self._typed_renderer = None
            self._typed_front_committed = None
            self._typed_contract = None
            self._fit_overlay_renderer = None
            self._fit_bindings = None
            self._fit_overlay_pending = None
            self._fit_overlay_inflight = None
            self._fit_overlay_desired = None
            self._deferred_fit_reload = None
            self._deferred_typed_retry = None
            self._visible_transient_fit_result_owner = None
            self._visible_transient_fit_result_retained_bytes = 0
        super()._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        if self._fit_job_kind == "save" and self._fit_future is not None:
            self._close_deferred_during_fit_save = True
            self._status.setText("FIT SAVE IN PROGRESS · CLOSE DEFERRED")
            self._diagnostic.setText(
                "The immutable reference will be accepted before close can continue."
            )
            return
        cancel_export_commits(
            cancelled=self._cancelled,
            commit_lock=self._export_commit_lock,
        )
        authority = self._fit_authority
        if authority is not None:
            authority.close()
        if self._fit_cancelled is not None:
            self._fit_cancelled.set()
        self._fit_prepare_pending = False
        self._fit_options_release_pending = False
        self._fit_overlay_pending = None
        super().shutdown()
        fit_future = self._fit_future
        if fit_future is not None:
            fit_future.cancel()


def _figure_window_factory(
    loader,
    *,
    memory_limit_bytes: int,
    fit_bindings: _FitWorkbenchBindings | None = None,
    initial_fit_result_identity: str | None = None,
):
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    worker_thread_id: int | None = None
    cached_typed: DataFigure | None = None
    cached_base: DataFigure | None = None
    cached_canonical_result_retained_bytes = 0

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
        nonlocal cached_typed, cached_base, cached_canonical_result_retained_bytes
        require_worker_owner()
        _require_not_cancelled(cancelled)
        figure = loader()
        if not isinstance(figure, DataFigure):
            raise TypeError("figure loader must return DataFigure")
        intent, unavailable_reason = _classify_single_typed(figure)
        if intent is not None:
            if figure.has_fit_overlays and initial_fit_result_identity is None:
                return _encoded_figure(
                    figure,
                    memory_limit,
                    cancelled,
                    unavailable_reason=(
                        "typed Fit replay requires an exact caller-supplied result identity"
                    ),
                )
            if not figure.has_fit_overlays and initial_fit_result_identity is not None:
                raise ValueError("Fit result identity was supplied for a source-only Figure")
            state = _default_typed_state(intent)
            try:
                front = _render_typed_front(
                    figure,
                    state,
                    current_value_limits=None,
                    previous_relim_mode=None,
                    previous_count_scale=None,
                    sequence=sequence,
                    memory_limit_bytes=memory_limit,
                    cancelled=cancelled,
                    fit_result_identity=initial_fit_result_identity,
                )
            except MemoryError:
                return _encoded_figure(
                    figure,
                    memory_limit,
                    cancelled,
                    unavailable_reason=(
                        f"interactive {intent.value} front exceeds the frozen memory budget"
                    ),
                )
            cached_typed = figure
            cached_canonical_result_retained_bytes = (
                figure.fit_results_retained_upper_bound_nbytes
            )
            cached_base = (
                figure.with_fit_results(None)
                if figure.has_fit_overlays
                else figure
            )
            return front
        return _encoded_figure(
            figure,
            memory_limit,
            cancelled,
            unavailable_reason=unavailable_reason,
        )

    def rerender(
        fit_result: FitResultBatch | None,
        fit_result_identity: str | None,
        state: _TypedDisplayState,
        current_value_limits,
        previous_relim_mode,
        previous_count_scale,
        sequence: int,
        memory_limit: int,
        cancelled: threading.Event,
        previous_fit_overlay_retained_bytes: int,
        window_external_retained_bytes: int,
    ) -> _TypedFigureFront:
        require_worker_owner()
        figure = cached_typed
        base = cached_base
        if base is None:
            raise RuntimeError("typed session has no frozen DataFigure")
        if fit_result is not None:
            render_figure = base
        elif fit_result_identity is None:
            render_figure = base
        elif (
            figure is not None
            and figure.has_fit_overlays
            and fit_result_identity == initial_fit_result_identity
        ):
            render_figure = figure
        else:
            raise ValueError("typed renderer has no exact result for this identity")
        releases_canonical = bool(
            render_figure is base
            and cached_canonical_result_retained_bytes
        )
        return _render_typed_front(
            render_figure,
            state,
            current_value_limits=current_value_limits,
            previous_relim_mode=previous_relim_mode,
            previous_count_scale=previous_count_scale,
            sequence=sequence,
            memory_limit_bytes=memory_limit,
            cancelled=cancelled,
            fit_result=fit_result,
            fit_result_identity=fit_result_identity,
            previous_fit_overlay_retained_bytes=(
                previous_fit_overlay_retained_bytes
            ),
            external_session_retained_bytes=(
                cached_canonical_result_retained_bytes
                if releases_canonical
                else 0
            )
            + nonnegative_integer(
                window_external_retained_bytes,
                "window_external_retained_bytes",
            ),
            release_initial_canonical_on_commit=releases_canonical,
        )

    def commit_front(release_initial_canonical: bool) -> None:
        nonlocal cached_typed, cached_canonical_result_retained_bytes
        if release_initial_canonical:
            cached_typed = None
            cached_canonical_result_retained_bytes = 0

    return lambda: DataFigureWindow(
        initial,
        rerender,
        rerender if fit_bindings is not None else None,
        memory_limit_bytes=limit,
        fit_bindings=fit_bindings,
        typed_front_committed=commit_front,
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
    fit_preparer=None,
    fit_executor=None,
    fit_saver=None,
    fit_reloader=None,
    fit_selected_model: str | None = None,
    fit_initial_selection: Selection | None = None,
    open_fit_analysis: bool = False,
    fit_timeout_seconds: float = _DEFAULT_FIT_TIMEOUT_SECONDS,
    initial_fit_result_identity: str | None = None,
) -> DataFigureWindow:
    """Resolve and display a current artifact entirely on the bounded worker."""

    if not callable(figure_factory):
        raise TypeError("figure_factory must be callable")
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    fit_calls = (fit_preparer, fit_executor, fit_saver, fit_reloader)
    if any(item is not None for item in fit_calls) and not all(
        callable(item) for item in fit_calls
    ):
        raise ValueError("all four Figure Fit capabilities must be supplied together")
    fit_bindings = (
        None
        if not any(item is not None for item in fit_calls)
        else _FitWorkbenchBindings(
            fit_preparer,
            fit_executor,
            fit_saver,
            fit_reloader,
            selected_model=fit_selected_model,
            initial_selection=fit_initial_selection,
            open_analysis=bool(open_fit_analysis),
            timeout_seconds=fit_timeout_seconds,
        )
    )
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
            fit_bindings=fit_bindings,
            initial_fit_result_identity=initial_fit_result_identity,
        )
    )


__all__ = [
    "DataFigureWindow",
    "open_data_figure_workbench",
    "open_figure_workbench",
]
