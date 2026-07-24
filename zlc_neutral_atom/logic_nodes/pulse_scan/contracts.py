"""Physical scan coordinates and authoritative scan-output binding."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing import Any, Mapping

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
from zlc_pulse import (
    PulseDocument,
    pulse_document_from_tree,
    pulse_document_to_tree,
    resolve_api_parameters,
    resolve_api_segment_document,
)
from zlc_storage import canonical_digest, canonical_text, exact_mapping
from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
)
from zlc_neutral_atom.catalog import DefinitionKey, MeasurementDefinition


SCAN_OUTPUT_CONTRACT_SCHEMA = "zlc_neutral_atom.ScanOutputContract"
PULSE_SCAN_PROGRAM_SCHEMA = "zlc_neutral_atom.PulseScanProgram"
PULSE_SCAN_MEASUREMENT_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.pulse_scan",
    "pulse-scan",
)
PULSE_SCAN_MEASUREMENT_DEFINITION = MeasurementDefinition(
    PULSE_SCAN_MEASUREMENT_KEY,
    "Pulse scan",
    PULSE_SCAN_PROGRAM_SCHEMA,
    "zlc.pulse-scan-binding",
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
)

_AUTONOMOUS_SCAN_SLOT_KIND = "AUTONOMOUS_SCAN_SLOT"
_API_SLOT_SEGMENTED_KIND = "API_SLOT_SEGMENTED_EXISTING"


def _scan_axis_id(parameter_id: str) -> AxisId:
    return AxisId(f"scan.parameter.{parameter_id}")


def _whole_document_repeat_count(
    document: PulseDocument,
    *,
    execution_name: str,
) -> int:
    repeat = document.repeat
    if repeat is None:
        return 1
    if (
        repeat.start_period_id != document.periods[0].period_id
        or repeat.end_period_id != document.periods[-1].period_id
    ):
        raise ValueError(
            f"{execution_name} repeat axis requires a whole-document RepeatRegion"
        )
    return repeat.count


@dataclass(frozen=True)
class ApiSegmentTable:
    """Ordered, complete API rows; values are never rounded or inferred."""

    columns: tuple[str, ...]
    rows: tuple[tuple[int | float, ...], ...]

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("API segment table columns must be non-empty")
        for column in columns:
            canonical_text(column, "API segment table column")
        if len(set(columns)) != len(columns):
            raise ValueError("API segment table columns must be unique")
        rows = tuple(tuple(row) for row in self.rows)
        if not rows:
            raise ValueError("API segment table must contain at least one row")
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("every API segment row must match its named columns")
        for row in rows:
            for value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                ):
                    raise TypeError("API segment values must be numeric")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("API segment values must be finite")
        if len(set(rows)) != len(rows):
            raise ValueError("API segment rows must identify unique coordinates")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)


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
        return cls._from_physical_rows(
            columns=table.columns,
            rows=table.rows,
            labels={
                parameter_id: parameters[parameter_id].label or parameter_id
                for parameter_id in table.columns
            },
            units={
                parameter_id: parameters[parameter_id].unit
                for parameter_id in table.columns
            },
        )

    @classmethod
    def from_api_segment_table(
        cls,
        document: PulseDocument,
        table: ApiSegmentTable,
    ) -> "ScanPointTable":
        """Derive named axes from declared API roles and explicit physical rows."""

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(table, ApiSegmentTable):
            raise TypeError("table must be ApiSegmentTable")
        expected = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        if table.columns != expected:
            raise ValueError(
                "API segment columns must match declared API parameter order"
            )
        parameters = document.api_parameter_by_id
        return cls._from_physical_rows(
            columns=table.columns,
            rows=table.rows,
            labels={parameter_id: parameter_id for parameter_id in table.columns},
            units={
                parameter_id: parameters[parameter_id].unit
                for parameter_id in table.columns
            },
        )

    @classmethod
    def _from_physical_rows(
        cls,
        *,
        columns: tuple[str, ...],
        rows: tuple[tuple[int | float, ...], ...],
        labels: dict[str, str],
        units: dict[str, str],
    ) -> "ScanPointTable":
        physical_rows = rows
        coordinates: list[tuple[int | float, ...]] = []
        coordinate_indices: list[dict[int | float, int]] = []
        for column_index, _parameter_id in enumerate(columns):
            ordered: dict[int | float, int] = {}
            for row in physical_rows:
                value = row[column_index]
                if value not in ordered:
                    ordered[value] = len(ordered)
            coordinate_indices.append(ordered)
            coordinates.append(tuple(ordered))

        logical_rows = tuple(
            tuple(
                coordinate_indices[column_index][value]
                for column_index, value in enumerate(row)
            )
            for row in physical_rows
        )
        axes = tuple(
            AxisSpec(
                axis_id=_scan_axis_id(parameter_id),
                name=labels[parameter_id],
                role=SCAN_POINT,
                size=len(coordinates[column_index]),
                coordinates=coordinates[column_index],
                unit=units[parameter_id],
            )
            for column_index, parameter_id in enumerate(columns)
        )
        result = cls(
            axes,
            PointLayout.from_mapping(
                tuple(axis.size for axis in axes),
                logical_rows,
            ),
        )
        if result.rows != physical_rows:
            raise ValueError("named scan axes/layout do not reproduce frozen pulse rows")
        return result


@dataclass(frozen=True)
class AutonomousScanSlotProgram:
    """Editable SCAN_SLOT document plus frozen whole-run API constants."""

    document: PulseDocument
    api_values: tuple[tuple[str, int | float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        ScanPointTable.from_pulse_document(self.document)
        _whole_document_repeat_count(
            self.document,
            execution_name="autonomous SCAN_SLOT",
        )
        api_values = tuple(self.api_values)
        if any(
            not isinstance(item, tuple) or len(item) != 2
            for item in api_values
        ):
            raise TypeError("api_values must contain (parameter_id, value) tuples")
        for parameter_id, _value in api_values:
            canonical_text(parameter_id, "API value parameter_id")
        expected = tuple(
            parameter.parameter_id for parameter in self.document.api_parameters
        )
        actual = tuple(parameter_id for parameter_id, _value in api_values)
        if actual != expected:
            raise ValueError(
                "autonomous SCAN_SLOT API values must exactly cover declared "
                "parameters in declaration order"
            )
        execution_document = resolve_api_parameters(
            self.document,
            dict(api_values),
        )
        if execution_document.api_parameters:
            raise RuntimeError("autonomous SCAN_SLOT retained unresolved APIs")
        object.__setattr__(self, "api_values", api_values)

    @classmethod
    def from_api_values(
        cls,
        document: PulseDocument,
        values: Mapping[str, int | float] | None = None,
    ) -> "AutonomousScanSlotProgram":
        """Freeze the document's complete whole-run API assignment.

        API declaration order and completeness are pulse-scan semantics.  A
        notebook or GUI may collect the mapping, but it must not independently
        decide how that mapping becomes an autonomous program.
        """

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        supplied = {} if values is None else dict(values)
        expected = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        if set(supplied) != set(expected):
            missing = tuple(key for key in expected if key not in supplied)
            extra = tuple(key for key in supplied if key not in set(expected))
            raise ValueError(
                "SCAN_SLOT requires explicit whole-run values for every API "
                f"parameter; missing={missing}, extra={extra}"
            )
        return cls(
            document,
            tuple((key, supplied[key]) for key in expected),
        )

    @property
    def execution_document(self) -> PulseDocument:
        return resolve_api_parameters(self.document, dict(self.api_values))

    @property
    def point_table(self) -> ScanPointTable:
        return ScanPointTable.from_pulse_document(self.document)

    @property
    def repeat_count(self) -> int:
        return _whole_document_repeat_count(
            self.document,
            execution_name="autonomous SCAN_SLOT",
        )

    @property
    def fingerprint(self) -> str:
        return canonical_digest(pulse_scan_program_to_tree(self))


@dataclass(frozen=True)
class ApiSlotSegmentedProgram:
    """An explicitly segmented API_SLOT sweep over finite static pulses."""

    document: PulseDocument
    table: ApiSegmentTable
    segmentation_rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(self.table, ApiSegmentTable):
            raise TypeError("table must be ApiSegmentTable")
        object.__setattr__(
            self,
            "segmentation_rationale",
            canonical_text(
                self.segmentation_rationale,
                "segmentation_rationale",
            ),
        )
        expected = tuple(
            parameter.parameter_id for parameter in self.document.api_parameters
        )
        if not expected:
            raise ValueError("API_SLOT segmented program requires API parameters")
        if self.table.columns != expected:
            raise ValueError(
                "API segment columns must exactly match declared API parameter order"
            )
        _whole_document_repeat_count(
            self.document,
            execution_name="API_SLOT segmented",
        )
        # Resolve every row while the intent/request is still being built.  A
        # sub-tick duration or out-of-range DAC is an authoring error, not a Run
        # failure after camera/sequencer resources have already been admitted.
        for row in self.table.rows:
            resolve_api_segment_document(
                self.document,
                dict(zip(self.table.columns, row)),
            )

    @cached_property
    def point_table(self) -> ScanPointTable:
        return ScanPointTable.from_api_segment_table(self.document, self.table)

    @property
    def point_count(self) -> int:
        """Number of unique frozen API rows without deriving point objects."""

        return len(self.table.rows)

    @property
    def repeat_count(self) -> int:
        return _whole_document_repeat_count(
            self.document,
            execution_name="API_SLOT segmented",
        )

    @property
    def segment_count(self) -> int:
        """Exact R by P execution cardinality without materializing a schedule."""

        return self.repeat_count * self.point_count

    @cached_property
    def resolved_point_documents(self) -> tuple[PulseDocument, ...]:
        """The P finite point documents, resolved once and retained in row order."""

        documents = tuple(
            resolve_api_segment_document(
                self.document,
                dict(zip(self.table.columns, row)),
            )
            for row in self.table.rows
        )
        if len(documents) != self.point_count:
            raise RuntimeError("API segment resolution changed the point count")
        return documents

    @property
    def fingerprint(self) -> str:
        return canonical_digest(pulse_scan_program_to_tree(self))


PulseScanProgram = AutonomousScanSlotProgram | ApiSlotSegmentedProgram


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


def _api_segment_table_to_tree(value: ApiSegmentTable) -> dict[str, Any]:
    if not isinstance(value, ApiSegmentTable):
        raise TypeError("value must be ApiSegmentTable")
    return {
        "columns": list(value.columns),
        "rows": [list(row) for row in value.rows],
    }


def _api_segment_table_from_tree(tree: Any) -> ApiSegmentTable:
    data = exact_mapping(
        tree,
        {"columns", "rows"},
        "ApiSegmentTable",
        discriminator=None,
    )
    if not isinstance(data["columns"], list) or not isinstance(data["rows"], list):
        raise TypeError("ApiSegmentTable columns and rows must be lists")
    if any(not isinstance(row, list) for row in data["rows"]):
        raise TypeError("ApiSegmentTable rows must contain lists")
    return ApiSegmentTable(
        tuple(data["columns"]),
        tuple(tuple(row) for row in data["rows"]),
    )


def pulse_scan_program_to_tree(value: PulseScanProgram) -> dict[str, Any]:
    """Encode the sole current discriminated pulse-scan program schema."""

    if isinstance(value, AutonomousScanSlotProgram):
        return {
            "schema": PULSE_SCAN_PROGRAM_SCHEMA,
            "kind": _AUTONOMOUS_SCAN_SLOT_KIND,
            "document": pulse_document_to_tree(value.document),
            "api_values": [
                [parameter_id, parameter_value]
                for parameter_id, parameter_value in value.api_values
            ],
        }
    if isinstance(value, ApiSlotSegmentedProgram):
        return {
            "schema": PULSE_SCAN_PROGRAM_SCHEMA,
            "kind": _API_SLOT_SEGMENTED_KIND,
            "document": pulse_document_to_tree(value.document),
            "table": _api_segment_table_to_tree(value.table),
            "segmentation_rationale": value.segmentation_rationale,
        }
    raise TypeError("value must be a PulseScanProgram")


def pulse_scan_program_from_tree(tree: Any) -> PulseScanProgram:
    """Decode only the current pulse-scan program variants and field sets."""

    if not isinstance(tree, dict):
        raise TypeError("PulseScanProgram must be a mapping")
    if tree.get("schema") != PULSE_SCAN_PROGRAM_SCHEMA:
        raise ValueError("PulseScanProgram schema differs")
    kind = tree.get("kind")
    if kind == _AUTONOMOUS_SCAN_SLOT_KIND:
        data = exact_mapping(
            tree,
            {"schema", "kind", "document", "api_values"},
            PULSE_SCAN_PROGRAM_SCHEMA,
        )
        raw_api_values = data["api_values"]
        if not isinstance(raw_api_values, list) or any(
            not isinstance(item, list) or len(item) != 2
            for item in raw_api_values
        ):
            raise TypeError("PulseScanProgram api_values must be pair lists")
        value: PulseScanProgram = AutonomousScanSlotProgram(
            pulse_document_from_tree(data["document"]),
            tuple((item[0], item[1]) for item in raw_api_values),
        )
    elif kind == _API_SLOT_SEGMENTED_KIND:
        data = exact_mapping(
            tree,
            {
                "schema",
                "kind",
                "document",
                "table",
                "segmentation_rationale",
            },
            PULSE_SCAN_PROGRAM_SCHEMA,
        )
        value = ApiSlotSegmentedProgram(
            pulse_document_from_tree(data["document"]),
            _api_segment_table_from_tree(data["table"]),
            data["segmentation_rationale"],
        )
    else:
        raise ValueError("PulseScanProgram kind differs")
    if pulse_scan_program_to_tree(value) != tree:
        raise ValueError("PulseScanProgram tree is typed but non-canonical")
    return value


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
    "ApiSegmentTable",
    "ApiSlotSegmentedProgram",
    "AutonomousScanSlotProgram",
    "PULSE_SCAN_MEASUREMENT_DEFINITION",
    "PULSE_SCAN_PROGRAM_SCHEMA",
    "PULSE_SCAN_MEASUREMENT_KEY",
    "PulseScanProgram",
    "ScanOutputContract",
    "ScanPointTable",
    "bind_scan_output_contract",
    "pulse_scan_program_from_tree",
    "pulse_scan_program_to_tree",
]
