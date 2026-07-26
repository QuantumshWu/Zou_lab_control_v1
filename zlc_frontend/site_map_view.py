"""Generic, headless SiteMap view projection for every frontend host."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zlc_data import (
    COMPONENT,
    AxisId,
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    CoordinateRangeSelection,
    DatasetRevisionRef,
    OwnedSnapshot,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
    Value,
    ValueSchema,
    dataset_cell_value,
    dataset_revision_ref_to_tree,
    expand_value_validity,
    materialize_component_dataset,
    selection_for_outer_cell,
    selection_to_tree,
)
from zlc_storage import canonical_digest, canonical_text, positive_real

from zlc_frontend.figure import DatasetId, EvaluatedImage, EvaluatedInput, evaluate_axis
from zlc_frontend.figure_outputs import (
    AREA_DATA_OUTPUT,
    FigureDerivedSignal,
    area_data_output_presentation,
    figure_derived_signal,
    figure_output_revision_ref,
    materialize_area_range_output,
)
from zlc_frontend.figure_source import FigureSource
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.site_map import immutable_site_state, site_ring_radius


def _site_map_input_tree(view) -> dict[str, object]:
    """Canonical lineage for one source-owned, already coherent SiteMap."""

    def evaluated_input_tree(value: EvaluatedInput) -> dict[str, object]:
        return {
            "dataset_id": value.dataset_id.value,
            "ref": dataset_revision_ref_to_tree(value.ref),
        }

    return {
        "background": evaluated_input_tree(view.background_input),
        "site_state": evaluated_input_tree(view.site_state_input),
        "site_geometry_identity": view.site_geometry_identity,
        "view_identity": view.view_identity,
    }


def _site_map_data_snapshot(
    source_ref: DatasetRevisionRef,
    site_axis: AxisSpec,
    values: np.ndarray,
    validity: np.ndarray,
    *,
    data_axes: tuple[AxisSpec, ...],
    unit: str | None,
    semantic_identity: dict[str, object],
) -> OwnedSnapshot:
    """Materialise selected site components without reducing SITE validity."""

    axes = tuple(data_axes)
    if not axes or axes[0] != site_axis:
        raise ValueError("SiteMap Area data must begin with its selected SITE axis")
    return materialize_component_dataset(
        source_ref,
        values,
        data_axes=axes,
        validity_axis_ids=(site_axis.axis_id,),
        validity=validity,
        unit=unit,
        reference_for=lambda schema: figure_output_revision_ref(
            AREA_DATA_OUTPUT,
            source_ref,
            schema,
            semantic_identity,
        ),
    )


def _site_map_area_outputs(
    source: FigureSource,
    selection: Selection,
    view,
) -> dict[str, FigureDerivedSignal]:
    """Select sites by their declared coordinates."""

    snapshot = source.snapshot
    if snapshot.ref != view.site_state_input.ref:
        raise ValueError("SiteMap Area source differs from its exact site-state input")
    x_axis = view.home_viewport.x_axis
    y_axis = view.home_viewport.y_axis
    terms = {term.axis_id: term for term in selection.terms}
    if set(terms) != {x_axis.axis_id, y_axis.axis_id}:
        raise ValueError("SiteMap Area must select its painted x and y axes")
    x_term = terms[x_axis.axis_id]
    y_term = terms[y_axis.axis_id]
    if not isinstance(x_term, CoordinateRangeSelection) or not isinstance(
        y_term, CoordinateRangeSelection
    ):
        raise TypeError("SiteMap Area requires coordinate-range x and y terms")
    if any(
        term.coordinate_frame != view.coordinate_frame
        for term in (x_term, y_term)
    ):
        raise ValueError("SiteMap Area coordinate frame differs from its sites")

    centers = np.asarray(view.centers_xy, dtype="<f8")
    selected = np.flatnonzero(
        (centers[:, 0] >= float(x_term.lower))
        & (centers[:, 0] <= float(x_term.upper))
        & (centers[:, 1] >= float(y_term.lower))
        & (centers[:, 1] <= float(y_term.upper))
    )
    lineage = _site_map_input_tree(view)
    selection_tree = selection_to_tree(selection)
    derivation_digest = canonical_digest(
        {
            "owner": "zlc_frontend.site-map-area",
            "inputs": lineage,
            "selection": selection_tree,
        }
    )
    outputs: dict[str, FigureDerivedSignal] = {}
    for axis, term in ((x_axis, x_term), (y_axis, y_term)):
        key, value = materialize_area_range_output(
            source,
            snapshot.ref,
            axis,
            (float(term.lower), float(term.upper)),
            ("lower", "upper"),
            {
                "inputs": lineage,
                "selection": selection_tree,
                "axis_id": axis.axis_id.value,
            },
            unit=axis.unit,
            derivation_digest=derivation_digest,
        )
        outputs[key] = value

    # An empty box still publishes its physical bounds.  Inventing a sentinel
    # site would turn a valid empty selection into false data.
    if not selected.size:
        return outputs

    selected_indices = tuple(int(index) for index in selected)
    source_site_axis = view.site_axis
    site_axis = AxisSpec(
        source_site_axis.axis_id,
        source_site_axis.name,
        source_site_axis.role,
        len(selected_indices),
        tuple(source_site_axis.coordinate_at(index) for index in selected_indices),
        source_site_axis.unit,
        source_site_axis.coordinate_frame,
    )
    validity = np.asarray(view.site_validity, dtype=np.bool_)[selected]
    site_state = view.site_state
    if site_state is not None:
        values = np.asarray(site_state, dtype=np.bool_)[selected]
        data_axes = (site_axis,)
        unit = None
        quantity = "site-state"
    else:
        if x_axis.unit != y_axis.unit:
            raise ValueError(
                "state-free SiteMap Area cannot combine x/y coordinates with "
                "different units into one area.data signal"
            )
        identity = canonical_digest(
            {
                "owner": "zlc_frontend.site-map-area-coordinate",
                "source_block_id": snapshot.ref.block_id.value,
                "selection": selection_tree,
            }
        )
        coordinate_axis = AxisSpec(
            AxisId(f"figure-output-{identity[:24]}-coordinate"),
            "coordinate",
            COMPONENT,
            2,
            ("x", "y"),
        )
        values = np.asarray(centers[selected], dtype="<f8")
        data_axes = (site_axis, coordinate_axis)
        unit = x_axis.unit
        quantity = "calibrated-centers"

    result = _site_map_data_snapshot(
        snapshot.ref,
        site_axis,
        values,
        validity,
        data_axes=data_axes,
        unit=unit,
        semantic_identity={
            "inputs": lineage,
            "selection": selection_tree,
            "quantity": quantity,
        },
    )
    outputs[AREA_DATA_OUTPUT] = figure_derived_signal(
        AREA_DATA_OUTPUT,
        result,
        source,
        preserve_source_coverage=False,
        presentation=area_data_output_presentation(
            source.source_contract_id,
        ),
        derivation_digest=derivation_digest,
    )
    return outputs

def _evaluated_image_cell(
    values: np.ndarray,
    validity: np.ndarray,
    schema: ValueSchema,
) -> tuple[EvaluatedImage, ImageViewportTransform, AxisSpec, AxisSpec]:
    if not isinstance(schema, ValueSchema):
        raise TypeError("schema must be ValueSchema")
    axes = schema.data_axes
    x_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == SPATIAL_X
    )
    y_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == SPATIAL_Y
    )
    if len(axes) != 2 or len(x_positions) != 1 or len(y_positions) != 1:
        raise ValueError(
            "site-map background requires exactly one SPATIAL_X and "
            "SPATIAL_Y data axis"
        )
    x_position, y_position = x_positions[0], y_positions[0]
    x_axis, y_axis = axes[x_position], axes[y_position]
    order_yx = (y_position, x_position)
    image = EvaluatedImage(
        evaluate_axis(x_axis, tuple(range(x_axis.size))),
        evaluate_axis(y_axis, tuple(range(y_axis.size))),
        np.transpose(values, order_yx),
        np.transpose(validity, order_yx),
        schema.value_unit,
    )
    return image, ImageViewportTransform((y_axis, x_axis)), x_axis, y_axis


def _image_cell(
    snapshot: OwnedSnapshot,
    repeat_index: int,
    point_storage_index: int,
) -> tuple[EvaluatedImage, ImageViewportTransform, AxisSpec, AxisSpec]:
    return _image_value(
        dataset_cell_value(
            snapshot.block,
            repeat_index,
            point_storage_index,
        )
    )


def _image_value(
    value: Value,
) -> tuple[EvaluatedImage, ImageViewportTransform, AxisSpec, AxisSpec]:
    """Project one already selected image value without a dataset materialization."""

    if not isinstance(value, Value):
        raise TypeError("image cell must be Value")
    return _evaluated_image_cell(
        value.values,
        expand_value_validity(value.validity, value.schema),
        value.schema,
    )


def build_site_map_snapshot_view(
    snapshot: OwnedSnapshot,
    *,
    site_axis: AxisSpec,
    coordinate_frame: CoordinateFrameId,
    centers_xy: np.ndarray,
    site_validity: np.ndarray,
    site_geometry_identity: str,
    coherence_identity: str,
    run_id: str,
    provenance_epoch_id: str,
    summary: str,
    presentation_kind: str = "site-map",
) -> "SiteMapView":
    """Build one state-free SiteMap from an already-selected Dataset cell."""

    schema = snapshot.block.schema
    if schema.repeat_axis.size != 1 or schema.point_layout.storage_size != 1:
        raise ValueError("snapshot SiteMap background must contain one cell")
    background, viewport, x_axis, y_axis = _image_cell(snapshot, 0, 0)
    if (
        x_axis.coordinate_frame != coordinate_frame
        or y_axis.coordinate_frame != coordinate_frame
    ):
        raise ValueError(
            "SiteMap background and site geometry use different coordinate frames"
        )
    identity = canonical_digest(
        {
            "owner": "zlc_frontend.site-map-snapshot-view",
            "source": dataset_revision_ref_to_tree(snapshot.ref),
            "site_geometry_identity": site_geometry_identity,
            "coherence_identity": coherence_identity,
        }
    )
    selection = selection_for_outer_cell(
        schema.repeat_axis,
        schema.point_axes,
        schema.point_layout,
        0,
        tuple(0 for _axis in schema.point_axes),
    )
    return SiteMapView(
        background=background,
        background_input=EvaluatedInput(
            DatasetId(f"site-map-background-{identity}"),
            snapshot.ref,
        ),
        site_state_input=EvaluatedInput(
            DatasetId(f"site-map-state-{identity}"),
            snapshot.ref,
        ),
        cell_selection=selection,
        home_viewport=viewport,
        site_axis=site_axis,
        coordinate_frame=coordinate_frame,
        centers_xy=centers_xy,
        site_radius=site_ring_radius(centers_xy),
        site_validity=site_validity,
        site_state=None,
        site_geometry_identity=site_geometry_identity,
        view_identity=identity,
        coherence_identity=coherence_identity,
        presentation_kind=presentation_kind,
        run_id=run_id,
        provenance_epoch_id=provenance_epoch_id,
        summary=summary,
    )


def build_site_map_cell_view(
    background_value: Value,
    background_ref: DatasetRevisionRef,
    state_value: Value,
    state_ref: DatasetRevisionRef,
    selection: Selection,
    *,
    site_axis: AxisSpec,
    coordinate_frame: CoordinateFrameId,
    centers_xy: np.ndarray,
    site_geometry_identity: str,
    coherence_identity: str,
    run_id: str,
    provenance_epoch_id: str,
    summary: str,
    presentation_kind: str = "site-map-cell",
) -> "SiteMapView":
    """Project one already-coherent background/state cell for SiteMap display."""

    if not isinstance(background_value, Value) or not isinstance(
        state_value,
        Value,
    ):
        raise TypeError("background_value and state_value must be Value")
    if not isinstance(background_ref, DatasetRevisionRef) or not isinstance(
        state_ref,
        DatasetRevisionRef,
    ):
        raise TypeError("background_ref and state_ref must be DatasetRevisionRef")
    if not isinstance(selection, Selection):
        raise TypeError("selection must be Selection")
    background, viewport, x_axis, y_axis = _image_value(background_value)
    if (
        x_axis.coordinate_frame != coordinate_frame
        or y_axis.coordinate_frame != coordinate_frame
    ):
        raise ValueError(
            "SiteMap background and geometry use different coordinate frames"
        )
    data_axes = state_value.schema.data_axes
    if len(data_axes) != 1 or data_axes[0] != site_axis or site_axis.role != SITE:
        raise ValueError("SiteMap state must follow its declared SITE axis")
    if state_value.schema.dtype != np.dtype(bool):
        raise TypeError("SiteMap state values must be boolean")
    if not isinstance(state_value.validity, ComponentValidity) or (
        state_value.validity.axis_ids != (site_axis.axis_id,)
    ):
        raise ValueError("SiteMap state validity must name exactly the SITE axis")
    state_values = np.asarray(state_value.values, dtype=np.bool_)
    site_validity = np.asarray(state_value.validity.mask, dtype=np.bool_)
    identity = canonical_digest(
        {
            "owner": "zlc_frontend.site-map-cell-view",
            "source": dataset_revision_ref_to_tree(background_ref),
            "state": dataset_revision_ref_to_tree(state_ref),
            "site_geometry_identity": site_geometry_identity,
            "coherence_identity": coherence_identity,
            "selection": selection_to_tree(selection),
        }
    )
    return SiteMapView(
        background=background,
        background_input=EvaluatedInput(
            DatasetId(f"site-map-background-{identity}"),
            background_ref,
        ),
        site_state_input=EvaluatedInput(
            DatasetId(f"site-map-state-{identity}"),
            state_ref,
        ),
        cell_selection=selection,
        home_viewport=viewport,
        site_axis=site_axis,
        coordinate_frame=coordinate_frame,
        centers_xy=centers_xy,
        site_radius=site_ring_radius(centers_xy),
        site_state=state_values,
        site_validity=site_validity,
        site_geometry_identity=site_geometry_identity,
        view_identity=identity,
        coherence_identity=coherence_identity,
        presentation_kind=presentation_kind,
        run_id=run_id,
        provenance_epoch_id=provenance_epoch_id,
        summary=summary,
    )

@dataclass(frozen=True, eq=False)
class SiteMapView:
    """Frontend-owned immutable view of one already-coherent SiteMap value."""

    background: EvaluatedImage
    background_input: EvaluatedInput
    site_state_input: EvaluatedInput
    cell_selection: Selection
    home_viewport: ImageViewportTransform
    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    centers_xy: np.ndarray
    site_radius: float
    site_validity: np.ndarray
    site_state: np.ndarray | None
    site_geometry_identity: str
    view_identity: str
    coherence_identity: str
    presentation_kind: str
    run_id: str
    provenance_epoch_id: str
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.background, EvaluatedImage):
            raise TypeError("background must be EvaluatedImage")
        if not isinstance(self.background_input, EvaluatedInput):
            raise TypeError("background_input must be EvaluatedInput")
        if not isinstance(self.site_state_input, EvaluatedInput):
            raise TypeError("site_state_input must be EvaluatedInput")
        if not isinstance(self.cell_selection, Selection):
            raise TypeError("cell_selection must be Selection")
        if self.background_input.dataset_id == self.site_state_input.dataset_id:
            raise ValueError("background and state inputs require distinct DatasetIds")
        if not isinstance(self.home_viewport, ImageViewportTransform):
            raise TypeError("home_viewport must be ImageViewportTransform")
        if self.home_viewport.viewport_revision != 0:
            raise ValueError("home_viewport must begin in authored revision zero")
        for evaluated, declared, role, name in (
            (self.background.x_axis, self.home_viewport.x_axis, SPATIAL_X, "x"),
            (self.background.y_axis, self.home_viewport.y_axis, SPATIAL_Y, "y"),
        ):
            if declared.role != role or evaluated.role != role:
                raise ValueError(f"SiteMap background {name} axis has the wrong role")
            if (
                evaluated.axis_id != declared.axis_id
                or evaluated.name != declared.name
                or evaluated.unit != declared.unit
                or evaluated.indices != tuple(range(declared.size))
                or evaluated.coordinates
                != tuple(declared.coordinate_at(index) for index in range(declared.size))
            ):
                raise ValueError(
                    f"SiteMap background {name} axis differs from its declared viewport"
                )
        if self.background.values.dtype.kind == "c":
            raise TypeError("SiteMap background must contain real values")
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must be an AxisSpec with role SITE")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        if self.home_viewport.coordinate_frame != self.coordinate_frame:
            raise ValueError("background and site geometry coordinate frames differ")
        sites = self.site_axis.size
        state_present = self.site_state is not None
        centers, site_state, site_validity = immutable_site_state(
            self.centers_xy,
            (
                self.site_state
                if state_present
                else np.zeros(sites, dtype=np.bool_)
            ),
            self.site_validity,
            site_count=sites,
        )
        radius = positive_real(self.site_radius, "site_radius")
        for value, name in (
            (self.site_geometry_identity, "site_geometry_identity"),
            (self.view_identity, "view_identity"),
            (self.coherence_identity, "coherence_identity"),
            (self.presentation_kind, "presentation_kind"),
            (self.run_id, "run_id"),
            (self.provenance_epoch_id, "provenance_epoch_id"),
        ):
            canonical_text(value, name)
        canonical_text(self.summary, "SiteMap summary")
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "site_radius", radius)
        object.__setattr__(self, "site_state", site_state if state_present else None)
        object.__setattr__(self, "site_validity", site_validity)

    def materialize_area_outputs(
        self,
        source: FigureSource,
        selection: Selection,
    ) -> dict[str, FigureDerivedSignal]:
        return _site_map_area_outputs(source, selection, self)

    @property
    def valid_site_count(self) -> int:
        """Number of physically admitted SITE components in this view."""

        return int(np.count_nonzero(self.site_validity))

    @property
    def occupied_site_count(self) -> int:
        """Number of occupied components among the admitted SITE components."""

        if self.site_state is None:
            return 0
        return int(np.count_nonzero(self.site_state & self.site_validity))

    @property
    def invalid_site_count(self) -> int:
        return self.site_axis.size - self.valid_site_count

    @property
    def site_count_summary(self) -> str:
        """Canonical display text for component validity and boolean state."""

        return (
            f"occupied={self.occupied_site_count}/{self.valid_site_count} valid sites | "
            f"invalid={self.invalid_site_count}"
        )

__all__ = [
    "SiteMapView",
    "build_site_map_cell_view",
    "build_site_map_snapshot_view",
]
