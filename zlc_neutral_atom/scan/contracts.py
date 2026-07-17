"""Physical scan coordinates and authoritative scan-output binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_data import (
    SCAN_POINT,
    AxisId,
    AxisLayout,
    AxisSpec,
    CommittedTransform,
    DatasetSchema,
    PointLayout,
    TransformedSchema,
    ValidityContract,
    ValueSchema,
    committed_transform_from_tree,
    committed_transform_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    resolve_transformed_schema,
)
from zlc_pulse import PulseDocument
from zlc_storage import canonical_digest, exact_mapping
from zlc_neutral_atom.catalog import DefinitionKey, TaskDefinition


SCAN_OUTPUT_CONTRACT_SCHEMA = "zlc_neutral_atom.ScanOutputContract"
AUTONOMOUS_SCAN_SLOT_TASK_KEY = DefinitionKey(
    "zlc_neutral_atom.scan",
    "autonomous-scan-slot",
)
AUTONOMOUS_SCAN_SLOT_DEFINITION = TaskDefinition(
    AUTONOMOUS_SCAN_SLOT_TASK_KEY,
    "Autonomous SCAN_SLOT",
    "zlc_neutral_atom.autonomous-scan-slot-request",
)
SCAN_TASK_DEFINITIONS = (AUTONOMOUS_SCAN_SLOT_DEFINITION,)


def _scan_axis_id(parameter_id: str) -> AxisId:
    return AxisId(f"scan.parameter.{parameter_id}")


@dataclass(frozen=True)
class ScanPointTable:
    """One lossless mapping from frozen pulse rows to named scan axes."""

    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout

    def __post_init__(self) -> None:
        axes = tuple(self.point_axes)
        if not axes or any(
            not isinstance(axis, AxisSpec) or axis.role != SCAN_POINT
            for axis in axes
        ):
            raise ValueError("point_axes must contain SCAN_POINT AxisSpec values")
        if len({axis.axis_id for axis in axes}) != len(axes):
            raise ValueError("scan point axis ids must be unique")
        if any(axis.coordinates is None for axis in axes):
            raise ValueError("scan point axes must freeze explicit physical coordinates")
        if any(
            not isinstance(value, (int, float))
            for axis in axes
            for value in axis.coordinates or ()
        ):
            raise TypeError("scan point coordinates must be numeric pulse values")
        if not isinstance(self.point_layout, PointLayout):
            raise TypeError("point_layout must be PointLayout")
        if self.point_layout.logical_shape != tuple(axis.size for axis in axes):
            raise ValueError("scan point layout shape differs from its named axes")
        if len(set(self.rows)) != self.point_layout.storage_size:
            raise ValueError("scan point rows must identify unique physical coordinates")
        object.__setattr__(self, "point_axes", axes)

    @property
    def rows(self) -> tuple[tuple[int | float, ...], ...]:
        """Reconstruct physical rows from the single axes/layout authority."""

        return tuple(
            tuple(
                axis.coordinate_at(index)
                for axis, index in zip(
                    self.point_axes,
                    self.point_layout.multi_index(storage_index),
                )
            )
            for storage_index in range(self.point_layout.storage_size)
        )

    @classmethod
    def from_pulse_document(cls, document: PulseDocument) -> "ScanPointTable":
        """Derive axes and sparse/rectangular layout solely from declared rows."""

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        table = document.scan_table
        if table is None or not document.scan_parameters:
            raise ValueError("a scan point table requires declared parameters and rows")
        parameters = document.scan_parameter_by_id
        coordinates: list[tuple[int | float, ...]] = []
        coordinate_indices: list[dict[int | float, int]] = []
        for column_index, parameter_id in enumerate(table.columns):
            ordered: dict[int | float, int] = {}
            for row in table.rows:
                value = row[column_index]
                if value not in ordered:
                    ordered[value] = len(ordered)
            coordinate_indices.append(ordered)
            coordinates.append(tuple(ordered))

        rows = tuple(
            tuple(
                coordinate_indices[column_index][value]
                for column_index, value in enumerate(row)
            )
            for row in table.rows
        )
        axes = tuple(
            AxisSpec(
                axis_id=_scan_axis_id(parameter_id),
                name=parameters[parameter_id].label or parameter_id,
                role=SCAN_POINT,
                size=len(coordinates[column_index]),
                coordinates=coordinates[column_index],
                unit=parameters[parameter_id].unit,
            )
            for column_index, parameter_id in enumerate(table.columns)
        )
        result = cls(
            axes,
            PointLayout.from_mapping(
                tuple(axis.size for axis in axes),
                rows,
            ),
        )
        if result.rows != table.rows:
            raise ValueError("named scan axes/layout do not reproduce frozen pulse rows")
        return result


@dataclass(frozen=True)
class ScanOutputContract:
    """Lossless authoritative scan-output semantics over one exact source.

    The current slice preserves every surviving trailing axis as a batch axis
    and preserves declared component validity.  Additional acceptance policies
    are intentionally not modelled until a second real policy is consumed.
    """

    committed_transform: CommittedTransform
    output_dataset_schema: DatasetSchema

    def __post_init__(self) -> None:
        if not isinstance(self.committed_transform, CommittedTransform):
            raise TypeError("committed_transform must be CommittedTransform")
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
            transformed_schema.fingerprint
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
    committed_transform: CommittedTransform,
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

    if not isinstance(committed_transform, CommittedTransform):
        raise TypeError("committed_transform must be CommittedTransform")
    resolved = resolve_transformed_schema(input_schema, committed_transform)
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
        "committed_transform": committed_transform_to_tree(
            value.committed_transform
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
        committed_transform_from_tree(data["committed_transform"]),
        dataset_schema_from_tree(data["output_dataset_schema"]),
    )
    if scan_output_contract_to_tree(value) != tree:
        raise ValueError("ScanOutputContract tree is typed but non-canonical")
    return value


__all__ = [
    "AUTONOMOUS_SCAN_SLOT_DEFINITION",
    "AUTONOMOUS_SCAN_SLOT_TASK_KEY",
    "SCAN_TASK_DEFINITIONS",
    "ScanOutputContract",
    "ScanPointTable",
    "bind_scan_output_contract",
]
