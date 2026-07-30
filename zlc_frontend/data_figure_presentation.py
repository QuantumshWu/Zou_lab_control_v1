"""Headless DataFigure classification, display authoring, and front contracts."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import AxisSourceRef
from zlc_data.codec import dataset_revision_ref_to_tree
from .data_figure import DataFigure, FigurePanelRegion
from .meter_display import MeterDisplayState
from .render import (
    BoardFrame,
    CurvePanelPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
)
from .curve_display import (
    CurveDisplayState,
    curve_home_x_limits,
    numeric_curve_coordinates,
)
from .display_range import RelimMode, validated_display_range
from .encoded_raster import EncodedRasterDocument
from .figure import (
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    ViewIntent,
)
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramDisplayState,
)
from .fit_projection import canonical_panel_focus_address
from .image_display import (
    ImageDisplayState,
)
from .image_view import image_viewport_for_evaluated_image
from .panel_size import DEFAULT_PANEL_SIZE
from .panel_params import panel_display_state_intent
from .plot_layout import optimal_grid_size_for_view
from zlc_storage import canonical_digest

DATA_FIGURE_PANEL_ID = "generic-typed"

def data_figure_summary(figure: DataFigure) -> str:
    document = figure.document
    intents = tuple(dict.fromkeys(layer.view.intent.value for layer in document.layers))
    panel_count = sum(len(layer.cells) for layer in figure.evaluated.layers)
    return (
        f"{'/'.join(value.lower() for value in intents)} · {panel_count} panel(s) · "
        f"document revision {document.revision}"
    )


def classify_single_data_figure(
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
    if intent is ViewIntent.METER:
        if figure.has_fit_overlays:
            return None, "METER display does not accept fit overlays"
        if all(isinstance(item.data, EvaluatedMeter) for item in series):
            units = {item.data.value_unit for item in series}
            if len(units) != 1:
                return None, "METER series do not share one exact value unit"
            return intent, None
        return None, "METER intent did not evaluate to homogeneous meter series"
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

def classify_faceted_data_figure(
    figure: DataFigure,
) -> tuple[ViewIntent | None, int | None, str | None]:
    """Return one supported single-layer typed grid without selecting a cell."""

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    if (
        len(figure.document.layers) != 1
        or len(figure.evaluated.layers) != 1
        or len(figure.evaluated.inputs) != 1
    ):
        return None, None, "typed grid requires exactly one layer and input"
    intent = figure.document.layers[0].view.intent
    data_type = {
        ViewIntent.IMAGE: EvaluatedImage,
        ViewIntent.CURVE: EvaluatedCurve,
        ViewIntent.METER: EvaluatedMeter,
        ViewIntent.HISTOGRAM: EvaluatedHistogram,
    }.get(intent)
    if data_type is None:
        return None, None, "figure intent has no current typed grid consumer"
    if figure.has_fit_overlays:
        if intent not in (
            ViewIntent.IMAGE,
            ViewIntent.CURVE,
            ViewIntent.HISTOGRAM,
        ):
            return None, None, (
                "typed fit grids require IMAGE, CURVE, or HISTOGRAM intent"
            )
        fit_results = tuple(figure.fit_results.values())
        if len(fit_results) != 1:
            return None, None, "typed fit grid requires one exact layer result"
        if (
            intent is ViewIntent.IMAGE
            and fit_results[0].spec.model_id != "radial_gaussian_center"
        ):
            return None, None, (
                "typed IMAGE fit grid requires radial_gaussian_center"
            )
    cells = figure.evaluated.layers[0].cells
    if len(cells) <= 1:
        return None, None, "typed grid requires more than one logical panel"
    grid_curve_axis = None
    grid_value_unit = None
    grid_image_axes = None
    for cell in cells:
        series_group = cell.series
        if not series_group or any(
            not isinstance(series.data, data_type)
            for series in series_group
        ):
            return None, None, "typed grid contains another evaluated data kind"
        if len({series.data.value_unit for series in series_group}) != 1:
            return None, None, "typed grid panel mixes value units"
        if intent is ViewIntent.IMAGE:
            if len(series_group) != 1:
                return None, None, "typed IMAGE grid panel must contain one image"
            image = series_group[0].data
            assert isinstance(image, EvaluatedImage)
            if image.values.dtype.kind not in "biuf":
                return None, None, "typed IMAGE grid requires real numeric values"
            try:
                image_viewport_for_evaluated_image(image)
            except (TypeError, ValueError) as error:
                return None, None, str(error)
            axes = (image.x_axis, image.y_axis, image.value_unit)
            if grid_image_axes is None:
                grid_image_axes = axes
            elif axes != grid_image_axes:
                return None, None, (
                    "typed IMAGE grid cells do not share axes and value unit"
                )
        if intent is ViewIntent.CURVE:
            first_curve = series_group[0].data
            assert isinstance(first_curve, EvaluatedCurve)
            try:
                numeric_curve_coordinates(first_curve.x_axis)
            except (TypeError, ValueError) as error:
                return None, None, str(error)
            if any(
                series.data.x_axis != first_curve.x_axis
                for series in series_group[1:]
            ):
                return None, None, "typed CURVE grid panel mixes exact x axes"
            if grid_curve_axis is None:
                grid_curve_axis = first_curve.x_axis
                grid_value_unit = first_curve.value_unit
            elif (
                first_curve.x_axis != grid_curve_axis
                or first_curve.value_unit != grid_value_unit
            ):
                return None, None, "typed CURVE grid cells do not share x axis and unit"
    return intent, len(cells), None


def data_figure_initial_size_name(figure: DataFigure) -> str:
    """Apply the product's one ordinary/Grid initial-size policy to a Figure."""

    _intent, panel_count, _reason = classify_faceted_data_figure(figure)
    if panel_count is None:
        return DEFAULT_PANEL_SIZE
    layer = figure.document.layers[0]
    snapshot = figure.datasets.resolve(layer.dataset_id)
    return optimal_grid_size_for_view(snapshot.block.schema, layer.view)

@dataclass(frozen=True, slots=True)
class DataFigureGridOverview:
    intent: ViewIntent
    figure: DataFigure
    bundle: EncodedRasterDocument
    regions: tuple[FigurePanelRegion, ...]
    histogram_home_x_limits: tuple[float, float] | None
    display_state: DataFigureGridDisplayState | None

    def __post_init__(self) -> None:
        if self.intent not in (
            ViewIntent.IMAGE,
            ViewIntent.CURVE,
            ViewIntent.METER,
            ViewIntent.HISTOGRAM,
        ):
            raise ValueError(
                "typed grid overview requires IMAGE, CURVE, METER, or HISTOGRAM"
            )
        if not isinstance(self.figure, DataFigure):
            raise TypeError("typed grid overview requires one exact DataFigure")
        figure_intent, panel_count, reason = classify_faceted_data_figure(self.figure)
        if figure_intent is not self.intent or panel_count is None:
            raise ValueError(
                "typed grid overview figure does not match its intent"
                + ("" if reason is None else f": {reason}")
            )
        if not isinstance(self.bundle, EncodedRasterDocument):
            raise TypeError("typed grid overview requires EncodedRasterDocument")
        if len(self.bundle.pages) != 1:
            raise ValueError("typed grid overview requires one encoded page")
        regions = tuple(self.regions)
        if len(regions) <= 1 or any(
            not isinstance(region, FigurePanelRegion) for region in regions
        ):
            raise ValueError(
                "typed grid overview requires multiple FigurePanelRegion values"
            )
        if len({region.key for region in regions}) != len(regions):
            raise ValueError("typed grid overview region keys must be unique")
        if len(regions) != panel_count:
            raise ValueError("typed grid overview regions do not cover its figure")
        addresses = tuple(region.focus_address for region in regions)
        if len(set(addresses)) != len(addresses):
            raise ValueError("typed grid addresses must identify unique panels")
        display = self.display_state
        if (
            display is not None
            and panel_display_state_intent(display) is not self.intent
        ):
            raise ValueError("typed grid display state does not match its intent")
        object.__setattr__(self, "regions", regions)
        home = self.histogram_home_x_limits
        if self.intent is ViewIntent.HISTOGRAM:
            if home is None:
                raise ValueError("HISTOGRAM grid requires one shared home x range")
            object.__setattr__(
                self,
                "histogram_home_x_limits",
                validated_display_range(home, "histogram grid home x limits"),
            )
        elif home is not None:
            raise ValueError("non-HISTOGRAM grid cannot carry a histogram home x range")

DataFigureDisplayState = (
    ImageDisplayState
    | CurveDisplayState
    | HistogramDisplayState
    | MeterDisplayState
)

DataFigureGridDisplayState = DataFigureDisplayState | FacetedHistogramDisplayState

DataFigurePanelPayload = (
    ImagePanelPayload
    | CurvePanelPayload
    | HistogramPanelPayload
    | MeterPanelPayload
)

@dataclass(frozen=True, slots=True)
class DataFigureGridFocusRequest:
    panel_index: int
    expected_address: tuple[tuple[AxisSourceRef, int], ...]
    display: DataFigureDisplayState
    histogram_home_x_limits: tuple[float, float] | None

    def __post_init__(self) -> None:
        if isinstance(self.panel_index, bool) or not isinstance(self.panel_index, int):
            raise TypeError("grid focus panel_index must be a non-negative integer")
        if self.panel_index < 0:
            raise ValueError("grid focus panel_index must be a non-negative integer")
        object.__setattr__(
            self,
            "expected_address",
            canonical_panel_focus_address(self.expected_address),
        )
        if panel_display_state_intent(self.display) not in (
            ViewIntent.IMAGE,
            ViewIntent.CURVE,
            ViewIntent.METER,
            ViewIntent.HISTOGRAM,
        ):
            raise ValueError(
                "grid focus supports IMAGE, CURVE, METER, or HISTOGRAM display state"
            )
        home = self.histogram_home_x_limits
        if isinstance(self.display, HistogramDisplayState):
            if home is None:
                raise ValueError("HISTOGRAM focus requires the grid home x range")
            object.__setattr__(
                self,
                "histogram_home_x_limits",
                validated_display_range(home, "histogram grid focus home x limits"),
            )
        elif home is not None:
            raise ValueError("METER focus cannot carry a histogram home x range")

def default_data_figure_display_state(intent: ViewIntent) -> DataFigureDisplayState:
    if intent is ViewIntent.IMAGE:
        return ImageDisplayState()
    if intent is ViewIntent.CURVE:
        return CurveDisplayState()
    if intent is ViewIntent.HISTOGRAM:
        return HistogramDisplayState()
    if intent is ViewIntent.METER:
        return MeterDisplayState(0, None)
    raise ValueError("typed intent must be IMAGE, CURVE, HISTOGRAM, or METER")

def data_figure_payload_intent(payload: DataFigurePanelPayload) -> ViewIntent:
    if isinstance(payload, ImagePanelPayload):
        return ViewIntent.IMAGE
    if isinstance(payload, CurvePanelPayload):
        return ViewIntent.CURVE
    if isinstance(payload, HistogramPanelPayload):
        return ViewIntent.HISTOGRAM
    if isinstance(payload, MeterPanelPayload):
        return ViewIntent.METER
    raise TypeError("unknown typed payload")

def data_figure_join_digest(
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

def data_figure_frame_contract(
    intent: ViewIntent,
    frame: BoardFrame,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Build a compact source token plus exact immutable data owners."""

    panel = frame.panels[0]
    payload = panel.display_payload
    assert isinstance(
        payload,
        (
            ImagePanelPayload,
            CurvePanelPayload,
            HistogramPanelPayload,
            MeterPanelPayload,
        ),
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

def same_exact_data_owners(
    left: tuple[object, ...],
    right: tuple[object, ...],
) -> bool:
    return len(left) == len(right) and all(
        actual is expected
        for actual, expected in zip(left, right, strict=True)
    )

def validate_rendered_data_figure_payload(
    payload: DataFigurePanelPayload,
    expected_state: DataFigureDisplayState,
    fit_result_identity: str | None,
) -> None:
    """Perform data-sized authored-state proof on the render worker."""

    if isinstance(expected_state, MeterDisplayState):
        if not isinstance(payload, MeterPanelPayload):
            raise ValueError("METER worker returned another payload kind")
        if fit_result_identity is not None:
            raise ValueError("METER display cannot carry a Fit result identity")
        if payload.display_revision != expected_state.revision:
            raise ValueError("METER worker returned another display revision")
        return
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
            or payload.colormap is not expected_state.colormap
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
        viewport.log_count_axis is not expected_state.log_count_axis
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
        or ((not payload.fit_overlays) != (fit_result_identity is None))
        or any(
            overlay.result_identity != fit_result_identity
            for overlay in payload.fit_overlays
        )
    ):
        raise ValueError("histogram worker returned conflicting authored state")


__all__ = [
    "DATA_FIGURE_PANEL_ID",
    "DataFigureDisplayState",
    "DataFigureGridDisplayState",
    "DataFigureGridFocusRequest",
    "DataFigureGridOverview",
    "DataFigurePanelPayload",
    "classify_faceted_data_figure",
    "classify_single_data_figure",
    "data_figure_initial_size_name",
    "data_figure_frame_contract",
    "data_figure_join_digest",
    "data_figure_payload_intent",
    "data_figure_summary",
    "default_data_figure_display_state",
    "same_exact_data_owners",
    "validate_rendered_data_figure_payload",
]
