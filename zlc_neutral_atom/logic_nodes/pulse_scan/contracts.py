"""PulseScan definition metadata and authoritative scan-output binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_data import (
    SCAN_POINT,
    CommittedTransform,
    DatasetSchema,
    PointTable,
    committed_transform_from_tree,
    committed_transform_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    resolve_transformed_schema,
)
from zlc_storage import canonical_digest, exact_mapping
from zlc_neutral_atom.catalog import DefinitionKey, MeasurementDefinition
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA,
)


SCAN_OUTPUT_CONTRACT_SCHEMA = "zlc_neutral_atom.ScanOutputContract"
PULSE_SCAN_MEASUREMENT_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.pulse_scan",
    "pulse-scan",
)
PULSE_SCAN_MEASUREMENT_DEFINITION = MeasurementDefinition(
    PULSE_SCAN_MEASUREMENT_KEY,
    "Pulse scan",
    PULSE_PARAMETER_SCAN_PROGRAM_SCHEMA,
    "zlc.pulse-scan-binding",
)


@dataclass(frozen=True)
class ScanOutputContract:
    """Lossless authoritative output over one exact R/P Dataset carrier."""

    committed_transform: CommittedTransform | None
    output_dataset_schema: DatasetSchema

    def __post_init__(self) -> None:
        if self.committed_transform is not None and not isinstance(
            self.committed_transform, CommittedTransform
        ):
            raise TypeError("committed_transform must be CommittedTransform or None")
        if not isinstance(self.output_dataset_schema, DatasetSchema):
            raise TypeError("output_dataset_schema must be DatasetSchema")
        if (
            self.committed_transform is not None
            and self.output_dataset_schema
            != self.committed_transform.effective_output_schema
        ):
            raise ValueError("output schema differs from the committed transform")

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_dataset_schema.fingerprint

    @property
    def fingerprint(self) -> str:
        return canonical_digest(scan_output_contract_to_tree(self))


def bind_scan_output_contract(
    input_schema: DatasetSchema,
    scan_points: PointTable,
    committed_transform: CommittedTransform | None = None,
) -> ScanOutputContract:
    """Preserve authored scan rows and repeat while allowing explicit data transforms."""

    if not isinstance(input_schema, DatasetSchema):
        raise TypeError("input_schema must be DatasetSchema")
    if not isinstance(scan_points, PointTable):
        raise TypeError("scan_points must be PointTable")
    if input_schema.point_table != scan_points:
        raise ValueError("DatasetSchema PointTable differs from the frozen pulse rows")
    if any(column.role != SCAN_POINT for column in scan_points.columns):
        raise ValueError("PulseScan point columns must use the SCAN_POINT role")
    if any(axis.role == SCAN_POINT for axis in input_schema.cell_schema.data_axes):
        raise ValueError("SCAN_POINT metadata belongs only to PointTable")

    if committed_transform is None:
        resolved = input_schema
    elif isinstance(committed_transform, CommittedTransform):
        resolved = resolve_transformed_schema(input_schema, committed_transform)
    else:
        raise TypeError("committed_transform must be CommittedTransform or None")
    if resolved.repeat_axis != input_schema.repeat_axis:
        raise ValueError("scan output must preserve the repeat axis exactly")
    if (
        resolved.point_table != input_schema.point_table
        or resolved.grid_topology != input_schema.grid_topology
    ):
        raise ValueError("scan output must preserve authored point rows exactly")
    source_data_ids = tuple(
        axis.axis_id for axis in input_schema.cell_schema.data_axes
    )
    try:
        positions = tuple(
            source_data_ids.index(axis.axis_id)
            for axis in resolved.cell_schema.data_axes
        )
    except ValueError as exc:
        raise ValueError("scan output introduced a trailing data axis") from exc
    if positions != tuple(sorted(positions)):
        raise ValueError("scan output reordered trailing data axes")
    return ScanOutputContract(committed_transform, resolved)


def scan_output_contract_to_tree(value: ScanOutputContract) -> dict[str, Any]:
    if not isinstance(value, ScanOutputContract):
        raise TypeError("value must be ScanOutputContract")
    return {
        "schema": SCAN_OUTPUT_CONTRACT_SCHEMA,
        "committed_transform": (
            None
            if value.committed_transform is None
            else committed_transform_to_tree(value.committed_transform)
        ),
        "output_dataset_schema": dataset_schema_to_tree(
            value.output_dataset_schema
        ),
    }


def scan_output_contract_from_tree(tree: Any) -> ScanOutputContract:
    data = exact_mapping(
        tree,
        {"schema", "committed_transform", "output_dataset_schema"},
        SCAN_OUTPUT_CONTRACT_SCHEMA,
    )
    value = ScanOutputContract(
        (
            None
            if data["committed_transform"] is None
            else committed_transform_from_tree(data["committed_transform"])
        ),
        dataset_schema_from_tree(data["output_dataset_schema"]),
    )
    if scan_output_contract_to_tree(value) != tree:
        raise ValueError("ScanOutputContract tree is typed but non-canonical")
    return value


__all__ = [
    "PULSE_SCAN_MEASUREMENT_DEFINITION",
    "PULSE_SCAN_MEASUREMENT_KEY",
    "ScanOutputContract",
    "bind_scan_output_contract",
    "scan_output_contract_from_tree",
    "scan_output_contract_to_tree",
]
