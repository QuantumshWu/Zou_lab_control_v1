"""Loaded exact Occupancy-cell facts.

This application boundary owns loading of the occupancy artifact, its source
capture, and its calibration.  It reads exactly one chunk-backed Camera cell;
callers never join artifacts or materialize the complete frame dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zlc_data import (
    ComponentValidity,
    DatasetRevisionRef,
    DatasetSchema,
    SPATIAL_X,
    SPATIAL_Y,
    Value,
)
from zlc_data.value import dataset_cell_value
from zlc_storage import canonical_text

from zlc_neutral_atom.devices.camera.contract import CameraFrameMetadata
from zlc_neutral_atom.capture.artifact import (
    CaptureArtifact,
    load_capture_artifact,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    SiteMap,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
    load_calibration_artifact,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from .processor import OccupancyArtifact, _occupancy_generation_for_run
from .reference import OccupancyArtifactRef
from .artifact import load_occupancy_artifact


@dataclass(frozen=True, slots=True)
class OccupancyCellDomain:
    """Validated immutable identities and schemas shared by every cell."""

    artifact_identity: str
    source_capture_identity: str
    calibration_identity: str
    source_schema: DatasetSchema
    occupancy_schema: DatasetSchema
    source_ref: DatasetRevisionRef
    occupancy_ref: DatasetRevisionRef
    site_map: SiteMap
    model_kind: ReadoutModelKind
    readout_binding: ReadoutBindingKey

    def __post_init__(self) -> None:
        for field in (
            "artifact_identity",
            "source_capture_identity",
            "calibration_identity",
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
        if not isinstance(self.site_map, SiteMap):
            raise TypeError("site_map must be SiteMap")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")

    @property
    def identity(
        self,
    ) -> tuple[str, DatasetRevisionRef, DatasetRevisionRef]:
        return (
            self.artifact_identity,
            self.source_ref,
            self.occupancy_ref,
        )

    @property
    def repeat_count(self) -> int:
        return self.occupancy_schema.repeat_axis.size

    @property
    def point_count(self) -> int:
        return self.occupancy_schema.point_table.row_count

    @property
    def linear_cell_count(self) -> int:
        return self.repeat_count * self.point_count

    def logical_point(self, point_ordinal: int) -> tuple[int, ...]:
        if not 0 <= point_ordinal < self.point_count:
            raise IndexError("occupancy point ordinal is out of range")
        topology = self.occupancy_schema.grid_topology
        return (
            topology.row_to_cell[point_ordinal]
            if topology is not None
            else (point_ordinal,)
        )

    def resolve_address(
        self,
        address: DatasetCellAddress,
    ) -> tuple[int, int, tuple[int, ...]]:
        if not isinstance(address, DatasetCellAddress):
            raise TypeError("occupancy cell address must be DatasetCellAddress")
        repeat_index = address.repeat_index
        point_ordinal = address.point_ordinal
        if not 0 <= repeat_index < self.repeat_count:
            raise IndexError("occupancy repeat index is out of range")
        return repeat_index, point_ordinal, self.logical_point(point_ordinal)

    def address_at_linear(self, linear_index: int) -> DatasetCellAddress:
        if (
            isinstance(linear_index, bool)
            or not isinstance(linear_index, int)
            or not 0 <= linear_index < self.linear_cell_count
        ):
            raise IndexError("occupancy cell index is out of range")
        repeat_index, point_ordinal = divmod(linear_index, self.point_count)
        return DatasetCellAddress(repeat_index, point_ordinal)

    def linear_index(self, address: DatasetCellAddress) -> int:
        repeat_index, point_ordinal, _logical = self.resolve_address(address)
        return repeat_index * self.point_count + point_ordinal


@dataclass(frozen=True, slots=True)
class ExactOccupancyCellSource:
    """One exact same-shot Camera value and classified SITE value."""

    domain: OccupancyCellDomain
    address: DatasetCellAddress
    logical_point: tuple[int, ...]
    image: Value
    occupied: Value
    frame_metadata: CameraFrameMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.domain, OccupancyCellDomain):
            raise TypeError("domain must be OccupancyCellDomain")
        if not isinstance(self.address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress")
        logical = tuple(self.logical_point)
        repeat, point_ordinal, resolved_logical = self.domain.resolve_address(
            self.address
        )
        if (
            repeat != self.address.repeat_index
            or point_ordinal != self.address.point_ordinal
            or resolved_logical != logical
        ):
            raise ValueError("cell logical point and address differ")
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
    occupancy_root: Path,
    captures_root: Path,
    calibrations_root: Path,
) -> tuple[OccupancyCellDomain, CaptureArtifact, OccupancyArtifact]:
    if not isinstance(reference, OccupancyArtifactRef):
        raise TypeError("reference must be OccupancyArtifactRef")
    for field, value in (
        ("occupancy_root", occupancy_root),
        ("captures_root", captures_root),
        ("calibrations_root", calibrations_root),
    ):
        if not isinstance(value, Path) or not value.is_absolute():
            raise TypeError(f"{field} must be an absolute Path")

    resolved = load_occupancy_artifact(
        occupancy_root,
        captures_root,
        calibrations_root,
        reference,
    )
    artifact = resolved.artifact
    source = load_capture_artifact(captures_root, artifact.source_capture_ref)
    calibration = load_calibration_artifact(
        calibrations_root,
        captures_root,
        artifact.calibration_reference,
    )
    calibration_artifact = calibration.artifact
    if calibration_artifact.frame_contract.binding != resolved.readout_binding:
        raise ValueError("occupancy source and calibration bindings differ")

    source_schema = source.frame_source.schema
    occupancy_schema = artifact.occupied.schema
    if (
        source_schema.repeat_axis != occupancy_schema.repeat_axis
        or source_schema.point_table != occupancy_schema.point_table
        or source_schema.grid_topology != occupancy_schema.grid_topology
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
    if len(site_axes) != 1 or site_axes[0] != calibration_artifact.site_map.site_axis:
        raise ValueError("occupancy SITE axis differs from its calibration")
    coordinate_frame = calibration_artifact.site_map.coordinate_frame
    if (
        x_axes[0].coordinate_frame != coordinate_frame
        or y_axes[0].coordinate_frame != coordinate_frame
    ):
        raise ValueError(
            "Camera spatial axes and calibration centers use different coordinate frames"
        )

    source_generation = source.provenance.generation
    source_ref = source.frame_source.ref(source_generation)
    occupancy_ref = artifact.occupied.ref(
        _occupancy_generation_for_run(artifact.run_id)
    )
    domain = OccupancyCellDomain(
        artifact_identity=reference.target_ref,
        source_capture_identity=artifact.source_capture_ref.target_ref,
        calibration_identity=artifact.calibration_reference.target_ref,
        source_schema=source_schema,
        occupancy_schema=occupancy_schema,
        source_ref=source_ref,
        occupancy_ref=occupancy_ref,
        site_map=calibration_artifact.site_map,
        model_kind=artifact.model_kind,
        readout_binding=resolved.readout_binding,
    )
    del resolved, calibration
    return domain, source, artifact


def inspect_occupancy_cell_domain(
    reference: OccupancyArtifactRef,
    occupancy_root: Path,
    captures_root: Path,
    calibrations_root: Path,
) -> OccupancyCellDomain:
    """Validate and describe an Occupancy target without reading a frame."""

    domain, _source, _artifact = _admit_cell_domain(
        reference,
        occupancy_root,
        captures_root,
        calibrations_root,
    )
    return domain


def load_exact_occupancy_cell_source(
    reference: OccupancyArtifactRef,
    occupancy_root: Path,
    captures_root: Path,
    calibrations_root: Path,
    address: DatasetCellAddress | None,
    *,
    expected_domain_identity: (
        tuple[str, DatasetRevisionRef, DatasetRevisionRef] | None
    ) = None,
) -> ExactOccupancyCellSource:
    """Read one exact cell after revalidating all persisted domain facts."""

    domain, source, artifact = _admit_cell_domain(
        reference,
        occupancy_root,
        captures_root,
        calibrations_root,
    )
    if expected_domain_identity is not None:
        expected = tuple(expected_domain_identity)
        if expected != domain.identity:
            raise ValueError("occupancy artifact changed after navigation inspection")
    if address is None:
        if domain.linear_cell_count != 1:
            raise ValueError("occupancy cell address is required for a multi-cell Dataset")
        address = domain.address_at_linear(0)
    repeat_index, point_ordinal, logical_point = domain.resolve_address(address)
    sample = source.frame_source.read(address)
    if sample.image.schema != domain.source_schema.cell_schema:
        raise ValueError("exact source frame differs from the admitted frame schema")

    site_axis = domain.site_map.site_axis
    occupied = dataset_cell_value(
        artifact.occupied,
        repeat_index,
        point_ordinal,
    )
    if not isinstance(occupied.validity, ComponentValidity) or (
        occupied.validity.axis_ids != (site_axis.axis_id,)
    ):
        raise ValueError("occupancy validity must name exactly the SITE axis")
    if np.any(occupied.validity.mask & ~domain.site_map.validity.mask):
        raise ValueError("occupancy marks a calibration-invalid site as valid")
    return ExactOccupancyCellSource(
        domain,
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
