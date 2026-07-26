"""PulseScan definition metadata and authoritative scan-output binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_data import (
    SCAN_POINT,
    AxisLayout,
    CommittedTransform,
    DatasetSchema,
    TransformedSchema,
    ValidityContract,
    ValueSchema,
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
    ScanPointTable,
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
    """Lossless authoritative scan-output semantics over one exact source.

    The current slice preserves every surviving trailing axis as a batch axis
    and preserves declared component validity.  Additional acceptance policies
    are intentionally not modelled until a second real policy is consumed.
    """

    committed_transform: CommittedTransform | None
    output_dataset_schema: DatasetSchema

    def __post_init__(self) -> None:
        if self.committed_transform is not None and not isinstance(
            self.committed_transform,
            CommittedTransform,
        ):
            raise TypeError("committed_transform must be CommittedTransform or None")
        if not isinstance(self.output_dataset_schema, DatasetSchema):
            raise TypeError("output_dataset_schema must be DatasetSchema")
        schema = self.output_dataset_schema
        transformed_schema = TransformedSchema(
            (schema.repeat_axis, *schema.point_axes),
            schema.cell_layout,
            schema.cell_schema.data_axes,
            schema.cell_schema.validity_contract.component_axis_ids,
            schema.cell_schema.dtype,
            schema.cell_schema.value_unit,
        )
        if (
            self.committed_transform is not None
            and transformed_schema.fingerprint
            != self.committed_transform.output_schema_fingerprint
        ):
            raise ValueError(
                "output_dataset_schema differs from the committed transform"
            )

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_dataset_schema.fingerprint

    @property
    def fingerprint(self) -> str:
        return canonical_digest(scan_output_contract_to_tree(self))


def bind_scan_output_contract(
    input_schema: DatasetSchema,
    scan_points: ScanPointTable,
    committed_transform: CommittedTransform | None = None,
) -> ScanOutputContract:
    """Bind one axis-total output while preserving repeat and physical scan rows."""

    if not isinstance(input_schema, DatasetSchema):
        raise TypeError("input_schema must be DatasetSchema")
    if not isinstance(scan_points, ScanPointTable):
        raise TypeError("scan_points must be ScanPointTable")
    source_scan_axes = tuple(
        axis for axis in input_schema.point_axes if axis.role == SCAN_POINT
    )
    if source_scan_axes != scan_points.point_axes:
        raise ValueError("DatasetSchema scan axes differ from the frozen scan table")
    if any(axis.role == SCAN_POINT for axis in input_schema.cell_schema.data_axes):
        raise ValueError("SCAN_POINT axes must be DatasetSchema point axes")

    if committed_transform is None:
        resolved = TransformedSchema(
            (input_schema.repeat_axis, *input_schema.point_axes),
            input_schema.cell_layout,
            input_schema.cell_schema.data_axes,
            input_schema.cell_schema.validity_contract.component_axis_ids,
            input_schema.cell_schema.dtype,
            input_schema.cell_schema.value_unit,
        )
    elif isinstance(committed_transform, CommittedTransform):
        resolved = resolve_transformed_schema(input_schema, committed_transform)
    else:
        raise TypeError("committed_transform must be CommittedTransform or None")
    output_cell_axes = resolved.cell_axes
    output_cell_layout = resolved.cell_layout
    output_data_axes = resolved.data_axes
    validity_contract = (
        ValidityContract.components(*resolved.validity_axis_ids)
        if resolved.validity_axis_ids
        else ValidityContract.value()
    )
    output_value_schema = ValueSchema(
        resolved.data_axes,
        validity_contract,
        resolved.dtype,
        resolved.value_unit,
    )

    expected_cell_axes = (input_schema.repeat_axis, *scan_points.point_axes)
    if output_cell_axes != expected_cell_axes:
        raise ValueError(
            "scan output must preserve repeat and every scan axis exactly while "
            "explicitly eliminating all other cell axes"
        )
    expected_layout = AxisLayout.product(
        AxisLayout.rect_c((input_schema.repeat_axis.size,)),
        scan_points.point_layout,
    )
    if output_cell_layout != expected_layout:
        raise ValueError("scan output changed the frozen repeat/scan point layout")

    source_data_ids = tuple(
        axis.axis_id for axis in input_schema.cell_schema.data_axes
    )
    try:
        output_positions = tuple(
            source_data_ids.index(axis.axis_id) for axis in output_data_axes
        )
    except ValueError as exc:
        raise ValueError("scan output introduced a trailing data axis") from exc
    if output_positions != tuple(sorted(output_positions)):
        raise ValueError("scan output reordered trailing data axes")

    output_schema = DatasetSchema(
        input_schema.repeat_axis,
        scan_points.point_axes,
        scan_points.point_layout,
        output_value_schema,
    )

    return ScanOutputContract(
        committed_transform,
        output_schema,
    )


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


def scan_output_contract_from_tree(
    tree: Any,
) -> ScanOutputContract:
    data = exact_mapping(
        tree,
        {
            "schema",
            "committed_transform",
            "output_dataset_schema",
        },
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
