"""Headless exact occupancy-cell values for interactive SiteMap display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zlc_data import (
    AxisLayout,
    AxisSpec,
    CoordinateFrameId,
    IndexSelection,
    PointLayout,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
    StreamGenerationId,
    resolve_selection_indices,
)
from zlc_storage import canonical_text, sha256_text

from .figure import EvaluatedImage, EvaluatedInput
from .image_view import ImageViewportTransform
from .site_map import immutable_site_state


@dataclass(frozen=True)
class OccupancyCellNavigation:
    """Frozen outer-axis identity for navigating one committed occupancy artifact."""

    artifact_identity: str
    schema_fingerprint: str
    generation: StreamGenerationId
    repeat_axis: AxisSpec
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    cell_layout: AxisLayout

    def __post_init__(self) -> None:
        canonical_text(self.artifact_identity, "artifact_identity")
        sha256_text(self.schema_fingerprint, "schema_fingerprint")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        if not isinstance(self.repeat_axis, AxisSpec):
            raise TypeError("repeat_axis must be AxisSpec")
        point_axes = tuple(self.point_axes)
        if any(not isinstance(axis, AxisSpec) for axis in point_axes):
            raise TypeError("point_axes must contain AxisSpec values")
        axes = (self.repeat_axis, *point_axes)
        if len({axis.axis_id for axis in axes}) != len(axes):
            raise ValueError("occupancy navigation axes must have unique AxisId values")
        if not isinstance(self.point_layout, PointLayout):
            raise TypeError("point_layout must be PointLayout")
        expected_shape = tuple(axis.size for axis in point_axes)
        if self.point_layout.logical_shape != expected_shape:
            raise ValueError("point_layout logical shape differs from point axes")
        if not isinstance(self.cell_layout, AxisLayout):
            raise TypeError("cell_layout must be AxisLayout")
        expected_cell_shape = (self.repeat_axis.size, *expected_shape)
        if (
            self.cell_layout.logical_shape != expected_cell_shape
            or self.cell_layout.storage_size
            != self.repeat_axis.size * self.point_layout.storage_size
        ):
            raise ValueError("cell_layout differs from repeat and point layout")
        object.__setattr__(self, "point_axes", point_axes)

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        return (self.repeat_axis, *self.point_axes)

    @property
    def identity(self) -> tuple[str, str, StreamGenerationId]:
        return (
            self.artifact_identity,
            self.schema_fingerprint,
            self.generation,
        )

    @property
    def linear_cell_count(self) -> int:
        return self.cell_layout.storage_size

    def resolve_selection(
        self,
        selection: Selection | None,
    ) -> tuple[int, int, tuple[int, ...], str]:
        """Resolve one exact named cell without first/latest/reduce fallbacks."""

        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        by_axis = {} if selection is None else {
            term.axis_id: term for term in selection.terms
        }
        known = {axis.axis_id for axis in self.axes}
        if any(axis_id not in known for axis_id in by_axis):
            raise ValueError(
                "occupancy cell selection may name only repeat and point axes"
            )
        indices = []
        labels = []
        for axis in self.axes:
            term = by_axis.get(axis.axis_id)
            if term is None:
                if axis.size != 1:
                    raise ValueError(
                        f"occupancy cell requires an explicit index for axis {axis.axis_id}"
                    )
                index = 0
            else:
                if not isinstance(term, IndexSelection):
                    raise TypeError(
                        "occupancy cell selection accepts only exact IndexSelection terms"
                    )
                resolved, drop = resolve_selection_indices(axis, term)
                if not drop or len(resolved) != 1:
                    raise ValueError("occupancy cell selection must resolve one exact index")
                index = resolved.start
            coordinate = axis.coordinate_at(index)
            unit = "" if axis.unit is None else f" {axis.unit}"
            labels.append(f"{axis.name}={coordinate}{unit} [index {index}]")
            indices.append(index)
        logical_point = tuple(indices[1:])
        try:
            point_storage_index = self.point_layout.storage_index(logical_point)
        except KeyError as error:
            raise ValueError(
                f"selected logical point {logical_point} is absent from PointLayout"
            ) from error
        return (
            indices[0],
            point_storage_index,
            logical_point,
            " | ".join(labels),
        )

    def selection_for_indices(
        self,
        repeat_index: int,
        logical_point: tuple[int, ...],
    ) -> Selection:
        """Build the canonical all-axis exact selection for one logical cell."""

        logical = tuple(logical_point)
        self.point_layout.storage_index(logical)
        terms = [IndexSelection(self.repeat_axis.axis_id, repeat_index)]
        terms.extend(
            IndexSelection(axis.axis_id, index)
            for axis, index in zip(self.point_axes, logical, strict=True)
        )
        selection = Selection(tuple(terms))
        self.resolve_selection(selection)
        return selection

    def selection_at_linear(self, linear_index: int) -> Selection:
        """Map repeat-major physical order to a canonical exact selection."""

        if (
            isinstance(linear_index, bool)
            or not isinstance(linear_index, int)
            or not 0 <= linear_index < self.linear_cell_count
        ):
            raise IndexError("occupancy navigation index is out of range")
        multi = self.cell_layout.multi_index(linear_index)
        return self.selection_for_indices(multi[0], tuple(multi[1:]))

    def linear_index(self, selection: Selection) -> int:
        repeat_index, _point_storage_index, logical, _label = self.resolve_selection(
            selection
        )
        return self.cell_layout.storage_index((repeat_index, *logical))


@dataclass(frozen=True, eq=False)
class OccupancyCellView:
    """Self-contained physical facts for one exact ``(repeat, point)`` cell."""

    background: EvaluatedImage
    background_input: EvaluatedInput
    occupancy_input: EvaluatedInput
    home_viewport: ImageViewportTransform
    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    centers_xy: np.ndarray
    occupied: np.ndarray
    site_validity: np.ndarray
    calibration_identity: str
    cell_identity: str
    cell_selection: Selection
    run_id: str
    provenance_epoch_id: str
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.background, EvaluatedImage):
            raise TypeError("background must be EvaluatedImage")
        if not isinstance(self.background_input, EvaluatedInput):
            raise TypeError("background_input must be EvaluatedInput")
        if not isinstance(self.occupancy_input, EvaluatedInput):
            raise TypeError("occupancy_input must be EvaluatedInput")
        if self.background_input.dataset_id == self.occupancy_input.dataset_id:
            raise ValueError("background and occupancy inputs require distinct DatasetIds")
        if self.background_input.ref.revision != self.occupancy_input.ref.revision:
            raise ValueError("background and occupancy inputs must name one exact revision")
        if not isinstance(self.home_viewport, ImageViewportTransform):
            raise TypeError("home_viewport must be ImageViewportTransform")
        if self.home_viewport.viewport_revision != 0:
            raise ValueError("home_viewport must begin in authored revision zero")
        for evaluated, declared, role, name in (
            (self.background.x_axis, self.home_viewport.x_axis, SPATIAL_X, "x"),
            (self.background.y_axis, self.home_viewport.y_axis, SPATIAL_Y, "y"),
        ):
            if declared.role != role or evaluated.role != role:
                raise ValueError(f"occupancy background {name} axis has the wrong role")
            if (
                evaluated.axis_id != declared.axis_id
                or evaluated.name != declared.name
                or evaluated.unit != declared.unit
                or evaluated.indices != tuple(range(declared.size))
                or evaluated.coordinates
                != tuple(declared.coordinate_at(index) for index in range(declared.size))
            ):
                raise ValueError(
                    f"occupancy background {name} axis differs from its declared viewport"
                )
        if self.background.values.dtype.kind == "c":
            raise TypeError("occupancy background must contain real values")
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site_axis must be an AxisSpec with role SITE")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        if self.home_viewport.coordinate_frame != self.coordinate_frame:
            raise ValueError("background and site geometry coordinate frames differ")
        sites = self.site_axis.size
        centers, occupied, site_validity = immutable_site_state(
            self.centers_xy,
            self.occupied,
            self.site_validity,
            site_count=sites,
        )
        if not isinstance(self.cell_selection, Selection):
            raise TypeError("cell_selection must be Selection")
        if any(
            not isinstance(term, IndexSelection)
            for term in self.cell_selection.terms
        ):
            raise TypeError("cell_selection must contain only exact IndexSelection terms")
        for value, name in (
            (self.calibration_identity, "calibration_identity"),
            (self.cell_identity, "cell_identity"),
            (self.run_id, "run_id"),
            (self.provenance_epoch_id, "provenance_epoch_id"),
        ):
            canonical_text(value, name)
        canonical_text(self.summary, "occupancy cell summary")
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "site_validity", site_validity)

__all__ = [
    "OccupancyCellNavigation",
    "OccupancyCellView",
]
