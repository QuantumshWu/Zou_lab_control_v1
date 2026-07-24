"""Pure frontend projection for one progressive occupancy scan curve."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import DatasetSchema, IndexSelection, Selection, SITE
from zlc_storage import canonical_text

from .curve_display import numeric_curve_coordinates
from .figure import (
    AxisViewRole,
    CURVE_CONTRACT,
    DatasetDescriptor,
    DatasetId,
    EvaluatedAxis,
    FigureDocument,
    FigureLayer,
    RepeatViewMode,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    suggest_view,
    validate_view_spec,
)


SCAN_CURVE_PANEL_ID = "scan-curve"


@dataclass(frozen=True, slots=True)
class ScanDisplayIntent:
    """Visible, non-authoritative site presentation choice."""

    site_mode: str = "auto"
    site_index: int = 0

    def __post_init__(self) -> None:
        mode = canonical_text(self.site_mode, "site_mode")
        if mode not in {"auto", "batch", "select"}:
            raise ValueError("site_mode must be 'auto', 'batch', or 'select'")
        if (
            isinstance(self.site_index, bool)
            or not isinstance(self.site_index, int)
            or self.site_index < 0
        ):
            raise ValueError("site_index must be a nonnegative integer")
        if mode != "select" and self.site_index != 0:
            raise ValueError("site_index is meaningful only when site_mode='select'")


@dataclass(frozen=True, slots=True)
class ScanCurvePresentation:
    """Frozen display-only projection derived solely from a declared schema."""

    document: FigureDocument
    projection_summary: str
    display_selection: Selection | None
    display_preferences: ViewPreferences
    interactive_curve: bool
    interaction_unavailable_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if self.display_selection is not None and not isinstance(
            self.display_selection,
            Selection,
        ):
            raise TypeError("display_selection must be Selection or None")
        if not isinstance(self.display_preferences, ViewPreferences):
            raise TypeError("display_preferences must be ViewPreferences")
        if not isinstance(self.interactive_curve, bool):
            raise TypeError("interactive_curve must be bool")
        if self.interactive_curve:
            if self.interaction_unavailable_reason is not None:
                raise ValueError(
                    "interactive curve cannot have an unavailable reason"
                )
        else:
            object.__setattr__(
                self,
                "interaction_unavailable_reason",
                canonical_text(
                    self.interaction_unavailable_reason,
                    "interaction_unavailable_reason",
                ),
            )
        dataset_id = self.document.datasets[0].dataset_id if self.document.datasets else None
        if (
            len(self.document.datasets) != 1
            or len(self.document.layers) != 1
            or self.document.layers[0].dataset_id != dataset_id
            or self.document.layers[0].view.intent is not ViewIntent.CURVE
            or self.document.layers[0].layer_id != SCAN_CURVE_PANEL_ID
            or any(
                binding.role is AxisViewRole.FACET
                for binding in self.document.layers[0].view.axis_bindings
            )
        ):
            raise ValueError(
                "scan curve presentation must be one non-faceted CURVE layer"
            )
        object.__setattr__(
            self,
            "projection_summary",
            canonical_text(self.projection_summary, "projection_summary"),
        )

    @property
    def dataset_id(self) -> DatasetId:
        return self.document.datasets[0].dataset_id


def build_occupancy_scan_curve(
    schema: DatasetSchema,
    *,
    identity: str,
    display_intent: ScanDisplayIntent = ScanDisplayIntent(),
) -> ScanCurvePresentation:
    """Derive a visible curve without interpreting a Measurement or Run."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(display_intent, ScanDisplayIntent):
        raise TypeError("display_intent must be ScanDisplayIntent")
    identity = canonical_text(identity, "identity")
    if not schema.point_axes:
        raise ValueError("progressive scan curve requires a declared point axis")
    x_axis = schema.point_axes[0]
    try:
        numeric_curve_coordinates(
            EvaluatedAxis(
                x_axis.axis_id,
                x_axis.name,
                x_axis.role,
                x_axis.unit,
                tuple(range(x_axis.size)),
                tuple(x_axis.coordinates),
            )
        )
    except (TypeError, ValueError) as error:
        interactive_curve = False
        interaction_unavailable_reason = f"{type(error).__name__}: {error}"
    else:
        interactive_curve = True
        interaction_unavailable_reason = None

    first_point = schema.point_layout.multi_index(0)
    terms = [
        IndexSelection(axis.axis_id, first_point[index])
        for index, axis in enumerate(schema.point_axes)
        if axis.axis_id != x_axis.axis_id
    ]
    data_axes = schema.cell_schema.data_axes
    site_axes = tuple(axis for axis in data_axes if axis.role == SITE)
    if len(site_axes) != 1:
        raise ValueError("occupancy output must declare exactly one SITE axis")
    site_axis = site_axes[0]
    if display_intent.site_mode == "batch":
        if site_axis.size <= 1:
            raise ValueError("site batch display requires at least 2 sites")
        batch_axis = site_axis
    elif display_intent.site_mode == "select":
        if display_intent.site_index >= site_axis.size:
            raise ValueError("selected site index exceeds the declared SITE axis")
        batch_axis = None
    else:
        batch_axis = site_axis if site_axis.size > 1 else None
    terms.extend(
        IndexSelection(
            axis.axis_id,
            display_intent.site_index
            if axis is site_axis and display_intent.site_mode == "select"
            else 0,
        )
        for axis in data_axes
        if axis is not batch_axis
    )
    selection = None if not terms else Selection(tuple(terms))
    preferences = ViewPreferences(
        repeat_mode=RepeatViewMode.MEAN,
        x_axis_id=x_axis.axis_id,
        batch_axis_ids=(() if batch_axis is None else (batch_axis.axis_id,)),
    )
    suggestion = suggest_view(
        schema,
        ViewIntent.CURVE,
        selection,
        preferences,
    )
    if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
        detail = " · ".join(reason.message for reason in suggestion.reasons)
        raise ValueError(
            "occupancy progressive curve needs an explicit display choice"
            + ("" if not detail else f": {detail}")
        )
    view = suggestion.spec
    validate_view_spec(schema, view, CURVE_CONTRACT)
    dataset_id = DatasetId(f"scan-preview-{identity}")
    document = FigureDocument(
        f"scan-preview-{identity}",
        0,
        (
            DatasetDescriptor(
                dataset_id,
                "Occupancy counts · PROVISIONAL",
                schema.fingerprint,
            ),
        ),
        (FigureLayer(SCAN_CURVE_PANEL_ID, dataset_id, view),),
    )
    axes_by_id = {
        axis.axis_id: axis
        for axis in (
            schema.repeat_axis,
            *schema.point_axes,
            *schema.cell_schema.data_axes,
        )
    }
    selections = [
        f"{axes_by_id[term.axis_id].name}="
        f"{axes_by_id[term.axis_id].coordinate_at(term.index)}"
        for term in terms
    ]
    summary = f"x={x_axis.name} · repeat=mean/{schema.repeat_axis.size}"
    if batch_axis is not None:
        summary += f" · {batch_axis.name}=batch/{batch_axis.size}"
    if selections:
        summary += " · " + " · ".join(selections)
    if not interactive_curve:
        assert interaction_unavailable_reason is not None
        summary += (
            " · static curve (interactive selector unavailable: "
            f"{interaction_unavailable_reason})"
        )
    return ScanCurvePresentation(
        document,
        summary,
        selection,
        preferences,
        interactive_curve,
        interaction_unavailable_reason,
    )


def describe_scan_figure(document: FigureDocument) -> str:
    """Describe the first scan layer from its frozen frontend view contract."""

    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    if len(document.layers) != 1:
        raise ValueError("scan Figure must contain exactly one layer")
    view = document.layers[0].view
    bindings = " · ".join(
        f"{binding.axis_id.value}={binding.role.value.lower()}"
        for binding in view.axis_bindings
    )
    summary = view.intent.value.lower()
    if bindings:
        summary = f"{summary} · {bindings}"
    if view.display_selections:
        summary += f" · selections={len(view.display_selections)}"
    return summary


__all__ = [
    "build_occupancy_scan_curve",
    "describe_scan_figure",
    "SCAN_CURVE_PANEL_ID",
    "ScanCurvePresentation",
    "ScanDisplayIntent",
]
