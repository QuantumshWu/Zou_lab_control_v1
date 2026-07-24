"""Committed exact occupancy-cell facts for presentation composition.

This application boundary owns admission of the occupancy artifact, its source
capture, and its calibration.  It reads exactly one chunk-backed Camera cell;
frontend code receives typed values and never asks the capture repository to
materialize the complete frame dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zlc_data import (
    AxisLayout,
    AxisSpec,
    ComponentValidity,
    DatasetRevisionRef,
    DatasetSchema,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
    StreamGenerationId,
    Value,
    dataset_cell_value,
    resolve_outer_cell_selection,
    selection_for_outer_cell,
)
from zlc_storage import canonical_text

from zlc_neutral_atom.devices.camera.contract import CameraFrameMetadata
from zlc_neutral_atom.logic_nodes.camera_capture.artifact import (
    CaptureArtifact,
    CaptureRepository,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

from zlc_neutral_atom.logic_nodes.calibration.calibration import (
    ReadoutModelKind,
    SiteMap,
)
from zlc_neutral_atom.logic_nodes.calibration.repository import CalibrationRepository
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from .processor import OccupancyArtifact
from .reference import OccupancyArtifactRef
from .repository import OccupancyRepository


@dataclass(frozen=True, slots=True)
class OccupancyCellDomain:
    """Admitted immutable identities and schemas shared by every cell."""

    artifact_identity: str
    source_capture_identity: str
    calibration_identity: str
    source_schema: DatasetSchema
    occupancy_schema: DatasetSchema
    source_ref: DatasetRevisionRef
    occupancy_ref: DatasetRevisionRef
    generation: StreamGenerationId
    site_map: SiteMap
    model_kind: ReadoutModelKind
    readout_binding: ReadoutBindingKey
    run_id: str
    provenance_epoch_id: str

    def __post_init__(self) -> None:
        for field in (
            "artifact_identity",
            "source_capture_identity",
            "calibration_identity",
            "run_id",
            "provenance_epoch_id",
        ):
            canonical_text(getattr(self, field), field)
        if not isinstance(self.source_schema, DatasetSchema) or not isinstance(
            self.occupancy_schema,
            DatasetSchema,
        ):
            raise TypeError("source_schema and occupancy_schema must be DatasetSchema")
        if not isinstance(self.source_ref, DatasetRevisionRef) or not isinstance(
            self.occupancy_ref,
            DatasetRevisionRef,
        ):
            raise TypeError("source_ref and occupancy_ref must be DatasetRevisionRef")
        if self.source_ref.schema_fingerprint != self.source_schema.fingerprint or (
            self.occupancy_ref.schema_fingerprint
            != self.occupancy_schema.fingerprint
        ):
            raise ValueError("cell-domain refs differ from their declared schemas")
        if self.source_ref.revision != self.occupancy_ref.revision:
            raise ValueError("occupancy revision differs from its Camera source")
        if not isinstance(self.generation, StreamGenerationId) or (
            self.occupancy_ref.stream_generation != self.generation
        ):
            raise ValueError("occupancy generation differs from its dataset ref")
        if not isinstance(self.site_map, SiteMap):
            raise TypeError("site_map must be SiteMap")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")

    @property
    def identity(self) -> tuple[str, str, StreamGenerationId]:
        return (
            self.artifact_identity,
            self.occupancy_schema.fingerprint,
            self.generation,
        )

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        schema = self.occupancy_schema
        return (schema.repeat_axis, *schema.point_axes)

    @property
    def cell_layout(self) -> AxisLayout:
        return self.occupancy_schema.cell_layout

    @property
    def linear_cell_count(self) -> int:
        return self.cell_layout.storage_size

    def resolve_selection(
        self,
        selection: Selection | None,
    ) -> tuple[int, int, tuple[int, ...]]:
        schema = self.occupancy_schema
        return resolve_outer_cell_selection(
            schema.repeat_axis,
            schema.point_axes,
            schema.point_layout,
            selection,
        )

    def selection_for_indices(
        self,
        repeat_index: int,
        logical_point: tuple[int, ...],
    ) -> Selection:
        schema = self.occupancy_schema
        selection = selection_for_outer_cell(
            schema.repeat_axis,
            schema.point_axes,
            schema.point_layout,
            repeat_index,
            tuple(logical_point),
        )
        self.resolve_selection(selection)
        return selection

    def selection_at_linear(self, linear_index: int) -> Selection:
        if (
            isinstance(linear_index, bool)
            or not isinstance(linear_index, int)
            or not 0 <= linear_index < self.linear_cell_count
        ):
            raise IndexError("occupancy cell index is out of range")
        multi = self.cell_layout.multi_index(linear_index)
        return self.selection_for_indices(multi[0], tuple(multi[1:]))

    def linear_index(self, selection: Selection) -> int:
        repeat_index, _point_storage, logical = self.resolve_selection(selection)
        return self.cell_layout.storage_index((repeat_index, *logical))


@dataclass(frozen=True, slots=True)
class ExactOccupancyCellSource:
    """One exact same-shot Camera value and classified SITE value."""

    domain: OccupancyCellDomain
    selection: Selection
    address: DatasetCellAddress
    logical_point: tuple[int, ...]
    image: Value
    occupied: Value
    frame_metadata: CameraFrameMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.domain, OccupancyCellDomain):
            raise TypeError("domain must be OccupancyCellDomain")
        if not isinstance(self.selection, Selection):
            raise TypeError("selection must be Selection")
        if not isinstance(self.address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress")
        logical = tuple(self.logical_point)
        repeat, point_storage, resolved_logical = resolve_outer_cell_selection(
            self.domain.occupancy_schema.repeat_axis,
            self.domain.occupancy_schema.point_axes,
            self.domain.occupancy_schema.point_layout,
            self.selection,
        )
        if (
            repeat != self.address.repeat_index
            or point_storage != self.address.point_storage_index
            or resolved_logical != logical
        ):
            raise ValueError("cell selection, logical point, and address differ")
        if not isinstance(self.image, Value) or (
            self.image.schema != self.domain.source_schema.cell_schema
        ):
            raise ValueError("image must match the admitted Camera cell schema")
        if not isinstance(self.occupied, Value) or (
            self.occupied.schema != self.domain.occupancy_schema.cell_schema
        ):
            raise ValueError("occupied must match the admitted SITE cell schema")
        if not isinstance(self.frame_metadata, CameraFrameMetadata):
            raise TypeError("frame_metadata must be CameraFrameMetadata")
        object.__setattr__(self, "logical_point", logical)


def _admit_cell_domain(
    reference: OccupancyArtifactRef,
    occupancy_repository: OccupancyRepository,
    capture_repository: CaptureRepository,
    calibration_repository: CalibrationRepository,
) -> tuple[OccupancyCellDomain, CaptureArtifact, OccupancyArtifact]:
    if not isinstance(reference, OccupancyArtifactRef):
        raise TypeError("reference must be OccupancyArtifactRef")
    if type(occupancy_repository) is not OccupancyRepository:
        raise TypeError("occupancy_repository must be OccupancyRepository")
    if type(capture_repository) is not CaptureRepository:
        raise TypeError("capture_repository must be CaptureRepository")
    if type(calibration_repository) is not CalibrationRepository:
        raise TypeError("calibration_repository must be CalibrationRepository")

    resolved = occupancy_repository.admit(
        reference,
        capture_repository,
        calibration_repository,
    )
    artifact = resolved.artifact
    source_admission = capture_repository.admit(artifact.source_capture_ref)
    source = source_admission.artifact
    calibration = calibration_repository.load(artifact.calibration_reference)
    if calibration.frame_contract.binding != resolved.readout_binding:
        raise ValueError("occupancy source and calibration bindings differ")

    source_schema = source.frame_source.schema
    occupancy_schema = artifact.occupied.schema
    if (
        source_schema.repeat_axis != occupancy_schema.repeat_axis
        or source_schema.point_axes != occupancy_schema.point_axes
        or source_schema.point_layout != occupancy_schema.point_layout
    ):
        raise ValueError("occupancy outer axes differ from the source capture")
    if source.frame_source.revision != artifact.occupied.revision:
        raise ValueError("occupancy revision differs from its source frame revision")

    frame_axes = source_schema.cell_schema.data_axes
    x_axes = tuple(axis for axis in frame_axes if axis.role == SPATIAL_X)
    y_axes = tuple(axis for axis in frame_axes if axis.role == SPATIAL_Y)
    if len(frame_axes) != 2 or len(x_axes) != 1 or len(y_axes) != 1:
        raise ValueError(
            "physical occupancy map requires exactly one SPATIAL_X and "
            "SPATIAL_Y frame axis"
        )
    site_axes = occupancy_schema.cell_schema.data_axes
    if len(site_axes) != 1 or site_axes[0] != calibration.site_map.site_axis:
        raise ValueError("occupancy SITE axis differs from its calibration")
    coordinate_frame = calibration.site_map.coordinate_frame
    if (
        x_axes[0].coordinate_frame != coordinate_frame
        or y_axes[0].coordinate_frame != coordinate_frame
    ):
        raise ValueError(
            "Camera spatial axes and calibration centers use different coordinate frames"
        )

    source_generation = source.provenance.generation
    source_ref = source.frame_source.ref(source_generation)
    occupancy_ref = artifact.occupied.ref(artifact.generation)
    domain = OccupancyCellDomain(
        artifact_identity=reference.target_ref,
        source_capture_identity=artifact.source_capture_ref.target_ref,
        calibration_identity=artifact.calibration_reference.target_ref,
        source_schema=source_schema,
        occupancy_schema=occupancy_schema,
        source_ref=source_ref,
        occupancy_ref=occupancy_ref,
        generation=artifact.generation,
        site_map=calibration.site_map,
        model_kind=artifact.model_kind,
        readout_binding=resolved.readout_binding,
        run_id=source.run_id,
        provenance_epoch_id=source_generation.value,
    )
    del resolved, source_admission, calibration
    return domain, source, artifact


def inspect_occupancy_cell_domain(
    reference: OccupancyArtifactRef,
    occupancy_repository: OccupancyRepository,
    capture_repository: CaptureRepository,
    calibration_repository: CalibrationRepository,
) -> OccupancyCellDomain:
    """Admit and describe a committed occupancy target without reading a frame."""

    domain, _source, _artifact = _admit_cell_domain(
        reference,
        occupancy_repository,
        capture_repository,
        calibration_repository,
    )
    return domain


def load_exact_occupancy_cell_source(
    reference: OccupancyArtifactRef,
    occupancy_repository: OccupancyRepository,
    capture_repository: CaptureRepository,
    calibration_repository: CalibrationRepository,
    selection: Selection | None,
    *,
    expected_domain_identity: tuple[str, str, StreamGenerationId] | None = None,
) -> ExactOccupancyCellSource:
    """Read one exact cell after re-admitting all persisted domain facts."""

    domain, source, artifact = _admit_cell_domain(
        reference,
        occupancy_repository,
        capture_repository,
        calibration_repository,
    )
    if expected_domain_identity is not None:
        expected = tuple(expected_domain_identity)
        if expected != domain.identity:
            raise ValueError("occupancy artifact changed after navigation inspection")
    repeat_index, point_storage_index, logical_point = (
        resolve_outer_cell_selection(
            domain.occupancy_schema.repeat_axis,
            domain.occupancy_schema.point_axes,
            domain.occupancy_schema.point_layout,
            selection,
        )
    )
    canonical_selection = selection_for_outer_cell(
        domain.occupancy_schema.repeat_axis,
        domain.occupancy_schema.point_axes,
        domain.occupancy_schema.point_layout,
        repeat_index,
        logical_point,
    )
    address = DatasetCellAddress(repeat_index, point_storage_index)
    sample = source.frame_source.read(address)
    if sample.image.schema != domain.source_schema.cell_schema:
        raise ValueError("exact source frame differs from the admitted frame schema")

    site_axis = domain.site_map.site_axis
    occupied = dataset_cell_value(
        artifact.occupied,
        repeat_index,
        point_storage_index,
    )
    if not isinstance(occupied.validity, ComponentValidity) or (
        occupied.validity.axis_ids != (site_axis.axis_id,)
    ):
        raise ValueError("occupancy validity must name exactly the SITE axis")
    if np.any(occupied.validity.mask & ~domain.site_map.validity.mask):
        raise ValueError("occupancy marks a calibration-invalid site as valid")
    return ExactOccupancyCellSource(
        domain,
        canonical_selection,
        address,
        logical_point,
        sample.image,
        occupied,
        sample.metadata,
    )


__all__ = [
    "ExactOccupancyCellSource",
    "OccupancyCellDomain",
    "inspect_occupancy_cell_domain",
    "load_exact_occupancy_cell_source",
]
