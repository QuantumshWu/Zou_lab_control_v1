"""Camera-frame to neutral-atom occupancy product.

The physical operation is deliberately small: bind one admitted calibration,
apply its selected readout model to every ``(R, P)`` camera cell, and preserve
the model's SITE axis and component validity.  Exact/live execution policies
are generic hosts around the same ``CameraSample -> OccupancySample`` record.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from zlc_data import (
    READOUT_EVENT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    Selection,
    StreamGenerationId,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
    dataset_cell_value,
    dataset_revision_ref_to_tree,
    expand_dataset_validity,
)
from zlc_neutral_atom.catalog import DefinitionKey, ProcessorDefinition
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraDatasetEventAdapter,
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    CameraSampleContract,
    ReadoutBindingKey,
)
from zlc_neutral_atom.input_spec import ArtifactInputSpec, DatasetInputSpec
from zlc_neutral_atom.logic_nodes.calibration.calibration import (
    ReadoutModel,
    ReadoutModelKind,
    ResolvedCalibration,
    SiteMap,
    _apply_readout_model,
    readout_model_authoring_schema,
    readout_model_kind_from_authoring,
)
from zlc_neutral_atom.logic_nodes.calibration.reference import (
    CALIBRATION_ARTIFACT_REF_FORMAT,
    CalibrationArtifactRef,
    calibration_artifact_input_ref,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.logic_nodes.camera_capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.camera_capture.session import (
    CaptureProcessorInputBinding,
)
from zlc_neutral_atom.logic_nodes.camera_measurement import (
    CAMERA_FRAME_OUTPUT_CONTRACT_ID,
    current_camera_monitor_selection,
)
from zlc_neutral_atom.logic_nodes.readout_common.contracts import FrameContract
from zlc_neutral_atom.logic_nodes.readout_common.physical_context import (
    _derive_readout_physical_context_from_evidence,
)
from zlc_neutral_atom.processing.stream import (
    BoundStreamProcessor,
    ExactStreamProcessorWorker,
)
from zlc_neutral_atom.runtime.cancellation import CancellationToken
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    FrozenDatasetEdge,
    MonitorCoverage,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    ExactReservation,
    StreamId,
)
from zlc_storage import canonical_digest, canonical_text, sha256_text

from .reference import OccupancyArtifactRef

if TYPE_CHECKING:
    from zlc_neutral_atom.logic_nodes.camera_capture.artifact import AdmittedCapture


OCCUPANCY_PROCESSOR_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.occupancy",
    "occupancy-processor",
)
_COUNTS_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "counts",
    "zlc_neutral_atom.occupancy.counts",
)
_OCCUPIED_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "occupied",
    "zlc_neutral_atom.occupancy.occupied",
)
_RATE_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "rate",
    "zlc_neutral_atom.occupancy.rate",
)
OCCUPANCY_LIVE_OUTPUT_DECLARATIONS = (
    _COUNTS_OUTPUT_DECLARATION,
    _OCCUPIED_OUTPUT_DECLARATION,
    _RATE_OUTPUT_DECLARATION,
)
OCCUPANCY_EXACT_SCAN_OUTPUT_DECLARATIONS = (
    _COUNTS_OUTPUT_DECLARATION,
    _OCCUPIED_OUTPUT_DECLARATION,
)
OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION = _OCCUPIED_OUTPUT_DECLARATION
DEFAULT_OCCUPANCY_CALIBRATION_POINTER = (
    "_output/calibrations/calibration_ref.json"
)
_OCCUPANCY_CONFIG_FORMAT = (
    "zlc_neutral_atom.logic_nodes.occupancy.processor-config"
)
OCCUPANCY_PROCESSOR_DEFINITION = ProcessorDefinition(
    OCCUPANCY_PROCESSOR_KEY,
    "Classify one camera frame into atomic site counts and occupancy",
    _OCCUPANCY_CONFIG_FORMAT,
)
OCCUPANCY_COUNTS_BLOCK_ID = BlockId("occupancy-counts")
OCCUPANCY_OCCUPIED_BLOCK_ID = BlockId("occupancy-occupied")
OCCUPANCY_RATE_BLOCK_ID = BlockId("occupancy-rate")

OCCUPANCY_CAMERA_INPUT_SPEC = DatasetInputSpec(
    "camera_frame",
    "Frame source",
    (CAMERA_FRAME_OUTPUT_CONTRACT_ID,),
    description=(
        "Current frame output of an already-running Camera Measurement; "
        "Occupancy never starts or reconfigures that Camera"
    ),
)
OCCUPANCY_CALIBRATION_INPUT_SPEC = ArtifactInputSpec(
    "calibration",
    "Calibration",
    CALIBRATION_ARTIFACT_REF_FORMAT,
    description=(
        "Exact FINAL CalibrationArtifactRef or an explicitly selected saved "
        "calibration_ref.json pointer"
    ),
    allow_saved_reference=True,
    default_reference_path=DEFAULT_OCCUPANCY_CALIBRATION_POINTER,
)
_OCCUPANCY_INPUT_SPECS = (
    OCCUPANCY_CAMERA_INPUT_SPEC,
    OCCUPANCY_CALIBRATION_INPUT_SPEC,
)


@dataclass(frozen=True, slots=True)
class OccupancyProcessorConfig:
    """Operator-authored model choice before input binding."""

    model_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")


def occupancy_authoring_schema():
    return readout_model_authoring_schema()


def occupancy_input_specs():
    return _OCCUPANCY_INPUT_SPECS


def build_occupancy_processor_config(
    values: Mapping[str, object],
) -> OccupancyProcessorConfig:
    authored = occupancy_authoring_schema().freeze(values)
    return OccupancyProcessorConfig(
        readout_model_kind_from_authoring(authored["model_kind"])
    )


def _require_output_value_schemas(
    counts: ValueSchema,
    occupied: ValueSchema,
) -> AxisSpec:
    if not isinstance(counts, ValueSchema) or not isinstance(occupied, ValueSchema):
        raise TypeError("occupancy cell schemas must be ValueSchema")
    axes = counts.data_axes
    if len(axes) != 1 or axes[0].role != SITE or occupied.data_axes != axes:
        raise ValueError("occupancy outputs require one shared SITE data axis")
    expected_validity = ValidityContract.components(axes[0].axis_id)
    if counts.validity_contract != expected_validity or (
        occupied.validity_contract != expected_validity
    ):
        raise ValueError("occupancy outputs require SITE component validity")
    if counts.dtype != np.dtype("<f8") or occupied.dtype != np.dtype(bool) or (
        occupied.value_unit != "occupation"
    ):
        raise ValueError("occupancy output dtype/unit contracts are not canonical")
    return axes[0]


def _require_occupancy_output_schemas(
    counts_schema: DatasetSchema,
    occupied_schema: DatasetSchema,
) -> AxisSpec:
    """Validate the one canonical counts/occupied dataset relationship."""

    if not isinstance(counts_schema, DatasetSchema) or not isinstance(
        occupied_schema,
        DatasetSchema,
    ):
        raise TypeError("occupancy output schemas must be DatasetSchema")
    if (
        counts_schema.repeat_axis,
        counts_schema.point_axes,
        counts_schema.point_layout,
    ) != (
        occupied_schema.repeat_axis,
        occupied_schema.point_axes,
        occupied_schema.point_layout,
    ):
        raise ValueError("occupancy outputs do not share one sampling domain")
    return _require_output_value_schemas(
        counts_schema.cell_schema,
        occupied_schema.cell_schema,
    )


def _same_validity(left: ComponentValidity, right: ComponentValidity) -> bool:
    return left.axis_ids == right.axis_ids and np.array_equal(
        left.mask,
        right.mask,
    )


def _require_canonical_fillers(
    counts: np.ndarray,
    occupied: np.ndarray,
    valid: np.ndarray,
) -> None:
    count_values = np.asarray(counts)
    occupied_values = np.asarray(occupied)
    mask = np.asarray(valid, dtype=bool)
    if count_values.shape != occupied_values.shape or mask.shape != count_values.shape:
        raise ValueError("occupancy values and validity must share one shape")
    if not np.all(np.isfinite(count_values)):
        raise ValueError("occupancy counts must be finite")
    invalid_counts = count_values[~mask]
    if np.any(invalid_counts != 0.0) or np.any(np.signbit(invalid_counts)):
        raise ValueError("invalid counts require canonical positive-zero fillers")
    if np.any(occupied_values[~mask]):
        raise ValueError("invalid occupied sites require canonical False fillers")


def _validate_sample_fields(counts: Value, occupied: Value) -> None:
    if not isinstance(counts, Value) or not isinstance(occupied, Value):
        raise TypeError("counts and occupied must be Value")
    _require_output_value_schemas(counts.schema, occupied.schema)
    if not isinstance(counts.validity, ComponentValidity) or not isinstance(
        occupied.validity,
        ComponentValidity,
    ):
        raise TypeError("occupancy fields require ComponentValidity")
    if not _same_validity(counts.validity, occupied.validity):
        raise ValueError("counts and occupied must have identical validity")
    _require_canonical_fillers(
        counts.values,
        occupied.values,
        counts.validity.mask,
    )


def _output_schemas(
    frame_contract: FrameContract,
    site_axis: AxisSpec,
) -> tuple[ValueSchema, ValueSchema]:
    validity = ValidityContract.components(site_axis.axis_id)
    return (
        ValueSchema(
            (site_axis,),
            validity,
            np.dtype("<f8"),
            frame_contract.count_unit,
        ),
        ValueSchema((site_axis,), validity, np.dtype(bool), "occupation"),
    )


@dataclass(frozen=True, slots=True)
class ResolvedOccupancyProcessorSchema:
    """Selected immutable readout model and its complete output schemas."""

    selected_model: ReadoutModel
    counts_schema: DatasetSchema
    occupied_schema: DatasetSchema

    def __post_init__(self) -> None:
        if not isinstance(self.selected_model, ReadoutModel):
            raise TypeError("selected_model must be ReadoutModel")
        site_axis = _require_occupancy_output_schemas(
            self.counts_schema,
            self.occupied_schema,
        )
        if self.selected_model.feature.site_axis != site_axis:
            raise ValueError("occupancy schemas differ from the selected model")

    @property
    def model_kind(self) -> ReadoutModelKind:
        return self.selected_model.kind


def _resolve_occupancy_processor_schema_parts(
    calibration: ResolvedCalibration,
    source_schema: DatasetSchema,
    model_kind: ReadoutModelKind,
) -> ResolvedOccupancyProcessorSchema:
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an exact ResolvedCalibration")
    calibration._require_authority()
    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(model_kind, ReadoutModelKind):
        raise TypeError("model_kind must be a concrete ReadoutModelKind")

    artifact = calibration.artifact
    frame_contract = artifact.frame_contract
    if source_schema.cell_schema != frame_contract.frame_schema:
        raise ValueError("source schema differs from the calibration FrameContract")
    model = artifact.select_model(model_kind)
    site_axis = artifact.site_map.site_axis
    if model.kind is not model_kind or model.feature.site_axis != site_axis:
        raise ValueError("selected model differs from the frozen calibration choice")
    outer_axis_ids = {
        source_schema.repeat_axis.axis_id,
        *(axis.axis_id for axis in source_schema.point_axes),
    }
    if site_axis.axis_id in outer_axis_ids:
        raise ValueError("SITE AxisId collides with a capture outer AxisId")
    counts_value, occupied_value = _output_schemas(frame_contract, site_axis)
    outer = (
        source_schema.repeat_axis,
        source_schema.point_axes,
        source_schema.point_layout,
    )
    return ResolvedOccupancyProcessorSchema(
        model,
        DatasetSchema(*outer, counts_value),
        DatasetSchema(*outer, occupied_value),
    )


def resolve_occupancy_processor_schema(
    spec: "OccupancyProcessorSpec",
    source_schema: DatasetSchema,
) -> ResolvedOccupancyProcessorSchema:
    """Freeze occupancy outputs before opening a camera session."""

    if not isinstance(spec, OccupancyProcessorSpec):
        raise TypeError("spec must be OccupancyProcessorSpec")
    if not isinstance(spec.model_kind, ReadoutModelKind):
        raise TypeError("processor spec has no concrete model_kind")
    return _resolve_occupancy_processor_schema_parts(
        spec.calibration,
        source_schema,
        spec.model_kind,
    )


@dataclass(frozen=True, slots=True)
class _CommittedOccupancyBinding:
    """Named schema authority shared by analysis and repository admission."""

    readout_event_axis_id: AxisId
    resolved_schema: ResolvedOccupancyProcessorSchema

    def __post_init__(self) -> None:
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if not isinstance(
            self.resolved_schema,
            ResolvedOccupancyProcessorSchema,
        ):
            raise TypeError("resolved_schema must be ResolvedOccupancyProcessorSchema")


@dataclass(frozen=True, slots=True)
class _ResolvedCommittedOccupancy:
    """Admitted source, calibration, and schema carried across one flat Run."""

    source: AdmittedCapture
    calibration: ResolvedCalibration
    binding: _CommittedOccupancyBinding

    def __post_init__(self) -> None:
        from zlc_neutral_atom.logic_nodes.camera_capture.artifact import AdmittedCapture

        if type(self.source) is not AdmittedCapture:
            raise PermissionError("occupancy source lacks repository admission")
        self.source._require_authority()  # type: ignore[attr-defined]
        if type(self.calibration) is not ResolvedCalibration:
            raise PermissionError("occupancy calibration lacks repository admission")
        self.calibration._require_authority()
        if not isinstance(self.binding, _CommittedOccupancyBinding):
            raise PermissionError("occupancy binding authority is invalid")


def _resolve_committed_occupancy_structure(
    capture: object,
    calibration: ResolvedCalibration,
    *,
    readout_event_axis_id: AxisId,
    model_kind: ReadoutModelKind,
) -> _CommittedOccupancyBinding:
    """Resolve named-axis, frame, model, and output schema facts."""

    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an exact ResolvedCalibration")
    calibration._require_authority()
    if not isinstance(readout_event_axis_id, AxisId):
        raise TypeError("readout_event_axis_id must be AxisId")
    if not isinstance(model_kind, ReadoutModelKind):
        raise TypeError("model_kind must be a concrete ReadoutModelKind")
    try:
        source = capture.frame_source  # type: ignore[attr-defined]
        provenance = capture.camera_provenance  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("capture must be a raw CaptureArtifact") from error
    schema = source.schema
    event_axes = tuple(
        axis for axis in schema.point_axes if axis.role == READOUT_EVENT
    )
    if len(event_axes) != 1 or event_axes[0].size != 1:
        raise ValueError(
            "committed occupancy requires one singleton READOUT_EVENT axis"
        )
    event_axis = event_axes[0]
    if event_axis.axis_id != readout_event_axis_id or (
        provenance.descriptor.readout_event_axis_id != event_axis.axis_id
    ):
        raise ValueError("capture and request name different READOUT_EVENT axes")
    artifact = calibration.artifact
    if provenance.binding != artifact.frame_contract.binding:
        raise ValueError("capture and calibration name different readout bindings")
    artifact.frame_contract.assert_compatible(
        provenance.binding,
        provenance.descriptor,
        schema,
        readout_event_index=0,
    )
    return _CommittedOccupancyBinding(
        event_axis.axis_id,
        _resolve_occupancy_processor_schema_parts(
            calibration,
            schema,
            model_kind,
        ),
    )


def _require_committed_occupancy_context(
    source: object,
    calibration: ResolvedCalibration,
    binding: _CommittedOccupancyBinding,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> _ResolvedCommittedOccupancy:
    """Compare every selected pulse window before authoritative analysis."""

    from zlc_neutral_atom.logic_nodes.camera_capture.artifact import AdmittedCapture

    if type(source) is not AdmittedCapture:
        raise TypeError("source must be an exact AdmittedCapture")
    source._require_authority()
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an exact ResolvedCalibration")
    calibration._require_authority()
    if not isinstance(binding, _CommittedOccupancyBinding):
        raise TypeError("binding must be resolved committed occupancy")
    event_axis_id = binding.readout_event_axis_id
    resolved_schema = binding.resolved_schema
    capture = source.artifact
    if capture.pulse_evidence is None:
        raise ValueError("authoritative occupancy requires persisted pulse lineage")
    context = _derive_readout_physical_context_from_evidence(
        capture.pulse_evidence,
        capture.frame_source.schema,
        capture.frame_source.iter_cell_schedule(),
        readout_event_index=0,
        integration_start_offset_seconds=(
            capture.camera_capability_evidence.physical_facts
            .external_trigger_integration_start_offset_seconds
        ),
        integration_seconds=calibration.artifact.frame_contract.exposure_seconds,
        checkpoint=checkpoint,
    )
    if context != calibration.artifact.readout_physical_context:
        raise ValueError("capture pulse context differs from the calibration")
    return _ResolvedCommittedOccupancy(source, calibration, binding)


def _occupancy_generation_for_run(run_id: str) -> StreamGenerationId:
    run = canonical_text(run_id, "run_id")
    return StreamGenerationId(
        canonical_digest(
            {
                "owner": "zlc_neutral_atom.logic_nodes.occupancy.committed-run",
                "run_id": run,
            }
        )
    )


@dataclass(frozen=True, slots=True, eq=False)
class OccupancyArtifact:
    """Durable counts/occupation derived from two committed inputs."""

    source_capture_ref: CaptureArtifactRef
    calibration_reference: CalibrationArtifactRef
    readout_event_axis_id: AxisId
    model_kind: ReadoutModelKind
    generation: StreamGenerationId
    counts: DataBlock
    occupied: DataBlock

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.calibration_reference, CalibrationArtifactRef):
            raise TypeError("calibration_reference must be CalibrationArtifactRef")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        if not isinstance(self.counts, DataBlock) or not isinstance(
            self.occupied,
            DataBlock,
        ):
            raise TypeError("counts and occupied must be DataBlock")
        if self.counts.block_id != OCCUPANCY_COUNTS_BLOCK_ID or (
            self.occupied.block_id != OCCUPANCY_OCCUPIED_BLOCK_ID
        ):
            raise ValueError("occupancy blocks use non-canonical BlockId values")
        site_axis = _require_occupancy_output_schemas(
            self.counts.schema,
            self.occupied.schema,
        )
        if self.counts.revision != self.occupied.revision or (
            self.counts.validity is not self.occupied.validity
        ):
            raise ValueError("occupancy blocks must share revision and validity")
        validity = self.counts.validity
        if not isinstance(validity, ComponentValidity) or (
            validity.axis_ids != (site_axis.axis_id,)
        ):
            raise ValueError("occupancy validity must name exactly the SITE axis")
        _require_canonical_fillers(
            self.counts.values,
            self.occupied.values,
            validity.mask,
        )

    @property
    def counts_snapshot(self) -> OwnedSnapshot:
        return OwnedSnapshot(self.counts.ref(self.generation), self.counts)

    @property
    def occupied_snapshot(self) -> OwnedSnapshot:
        return OwnedSnapshot(self.occupied.ref(self.generation), self.occupied)


_RESOLVED_OCCUPANCY_TOKEN = object()


class ResolvedOccupancy:
    """Process-local proof that one exact occupancy target was committed."""

    __slots__ = (
        "_token",
        "_repository_token",
        "_reference",
        "_artifact",
        "_readout_binding",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("ResolvedOccupancy is final")

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError("ResolvedOccupancy is returned by OccupancyRepository.admit")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ResolvedOccupancy is immutable")

    def __reduce__(self):
        raise TypeError("ResolvedOccupancy is process-local")

    @classmethod
    def _from_admission(
        cls,
        token: object,
        *,
        repository_token: object,
        reference: OccupancyArtifactRef,
        artifact: OccupancyArtifact,
        readout_binding: ReadoutBindingKey,
    ) -> "ResolvedOccupancy":
        if token is not _RESOLVED_OCCUPANCY_TOKEN:
            raise PermissionError("occupancy admission token is invalid")
        if repository_token is None:
            raise ValueError("occupancy repository authority is absent")
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if not isinstance(artifact, OccupancyArtifact):
            raise TypeError("artifact must be OccupancyArtifact")
        if not isinstance(readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        resolved = object.__new__(cls)
        object.__setattr__(resolved, "_token", token)
        object.__setattr__(resolved, "_repository_token", repository_token)
        object.__setattr__(resolved, "_reference", reference)
        object.__setattr__(resolved, "_artifact", artifact)
        object.__setattr__(resolved, "_readout_binding", readout_binding)
        return resolved

    def _require_authority(self) -> None:
        if type(self) is not ResolvedOccupancy or (
            self._token is not _RESOLVED_OCCUPANCY_TOKEN
            or self._repository_token is None
        ):
            raise PermissionError("ResolvedOccupancy authority is invalid")

    @property
    def reference(self) -> OccupancyArtifactRef:
        self._require_authority()
        return self._reference

    @property
    def artifact(self) -> OccupancyArtifact:
        self._require_authority()
        return self._artifact

    @property
    def readout_binding(self) -> ReadoutBindingKey:
        self._require_authority()
        return self._readout_binding


def _classify_cells(
    resolved: ResolvedOccupancyProcessorSchema,
    revision,
    cells: Iterable[tuple[int, int, Value]],
    *,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[DataBlock, DataBlock]:
    """Single R/P/SITE materializer shared by snapshot and committed analysis."""

    counts_values = np.zeros(resolved.counts_schema.physical_shape, dtype="<f8")
    occupied_values = np.zeros(resolved.occupied_schema.physical_shape, dtype=bool)
    valid_values = np.zeros(resolved.counts_schema.physical_shape, dtype=bool)
    for repeat_index, point_index, frame in cells:
        if checkpoint is not None:
            checkpoint()
        result = _apply_readout_model(resolved.selected_model, frame)
        validity = result.occupied.validity
        if not isinstance(validity, ComponentValidity):
            raise TypeError("readout result requires ComponentValidity")
        location = (repeat_index, point_index)
        counts_values[location] = result.signals.values
        occupied_values[location] = result.occupied.values
        valid_values[location] = validity.mask
    if checkpoint is not None:
        checkpoint()
    validity = ComponentValidity(
        (resolved.selected_model.feature.site_axis.axis_id,),
        valid_values,
    )
    return (
        DataBlock(
            OCCUPANCY_COUNTS_BLOCK_ID,
            revision,
            counts_values,
            validity,
            resolved.counts_schema,
        ),
        DataBlock(
            OCCUPANCY_OCCUPIED_BLOCK_ID,
            revision,
            occupied_values,
            validity,
            resolved.occupied_schema,
        ),
    )


def _analyze_committed_occupancy_resolved(
    resolved: _ResolvedCommittedOccupancy,
    *,
    run_id: str,
    checkpoint: Callable[[], None],
) -> OccupancyArtifact:
    """Stream raw frames once while preserving the complete R/P/SITE domain."""

    if not isinstance(resolved, _ResolvedCommittedOccupancy):
        raise TypeError("resolved must be admitted committed occupancy")
    source = resolved.source
    calibration = resolved.calibration
    event_axis_id = resolved.binding.readout_event_axis_id
    schema = resolved.binding.resolved_schema
    if not callable(checkpoint):
        raise TypeError("checkpoint must be callable")
    frame_source = source.artifact.frame_source  # type: ignore[attr-defined]
    counts, occupied = _classify_cells(
        schema,
        frame_source.revision,
        (
            (cell.repeat_index, cell.point_storage_index, sample.image)
            for cell, sample in frame_source.iter_event_order()
        ),
        checkpoint=checkpoint,
    )
    run = canonical_text(run_id, "run_id")
    return OccupancyArtifact(
        source.reference,  # type: ignore[attr-defined]
        calibration.reference,
        event_axis_id,
        schema.model_kind,
        _occupancy_generation_for_run(run),
        counts,
        occupied,
    )


@dataclass(frozen=True, slots=True, eq=False)
class OccupancySample:
    """Counts and occupancy atomically derived from one camera frame."""

    counts: Value
    occupied: Value
    metadata: CameraFrameMetadata

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, CameraFrameMetadata):
            raise TypeError("metadata must be CameraFrameMetadata")
        _validate_sample_fields(self.counts, self.occupied)


@dataclass(frozen=True)
class OccupancySampleContract:
    """Payload contract and counts-dataset projection for one atomic sample."""

    counts_schema: ValueSchema
    occupied_schema: ValueSchema
    source_metadata_contract: CameraFrameMetadataContract

    def __post_init__(self) -> None:
        _require_output_value_schemas(self.counts_schema, self.occupied_schema)
        if not isinstance(self.source_metadata_contract, CameraFrameMetadataContract):
            raise TypeError("source_metadata_contract must be CameraFrameMetadataContract")

    @property
    def _counts(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.counts_schema)

    @property
    def _occupied(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.occupied_schema)

    @property
    def payload_contract(self) -> "OccupancySampleContract":
        return self

    @property
    def metadata_contract(self) -> "OccupancySampleContract":
        return self

    @property
    def value_schema(self) -> ValueSchema:
        return self.counts_schema

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.OccupancySample",
                "counts": self._counts.fingerprint,
                "occupied": self._occupied.fingerprint,
                "metadata": self.source_metadata_contract.fingerprint,
            }
        )

    @property
    def operator_fingerprint(self) -> str:
        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.logic_nodes.occupancy.counts-projection",
                "payload": self.fingerprint,
            }
        )

    def snapshot(self, payload: OccupancySample) -> OccupancySample:
        self.validate(payload)
        return payload

    def validate(self, payload: object | None) -> None:
        if not isinstance(payload, OccupancySample):
            raise TypeError("payload must be OccupancySample")
        self._counts.validate(payload.counts)
        self._occupied.validate(payload.occupied)
        self.source_metadata_contract.validate(payload.metadata)

    def digest(self, payload: object | None) -> str:
        self.validate(payload)
        assert isinstance(payload, OccupancySample)
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.OccupancySampleContent",
                "counts": self._counts.digest(payload.counts),
                "occupied": self._occupied.digest(payload.occupied),
                "metadata": self.source_metadata_contract.digest(payload.metadata),
            }
        )

    @staticmethod
    def value(payload: OccupancySample) -> Value:
        return payload.counts

    @staticmethod
    def source_ordinal(payload: OccupancySample) -> int:
        return payload.metadata.source_ordinal

    @staticmethod
    def captured_at(payload: OccupancySample) -> float:
        return payload.metadata.captured_at

    @staticmethod
    def correlation_id(payload: OccupancySample) -> str:
        return payload.metadata.correlation_id


@dataclass(frozen=True)
class OccupancyProcessorSpec:
    """Reusable calibration choice; a CaptureSession supplies the input."""

    calibration: ResolvedCalibration
    output_stream_id: StreamId
    output_source_id: str
    model_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        if type(self.calibration) is not ResolvedCalibration:
            raise TypeError("calibration must be an exact ResolvedCalibration")
        self.calibration._require_authority()
        selected = self.calibration.artifact.select_model(self.model_kind)
        object.__setattr__(self, "model_kind", selected.kind)
        if not isinstance(self.output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        canonical_text(self.output_source_id, "output_source_id")


@dataclass(frozen=True, slots=True, eq=False)
class _OccupancyConfig:
    """Typed cell-level subset of a resolved schema accepted by the worker."""

    model: ReadoutModel
    counts_schema: ValueSchema
    occupied_schema: ValueSchema

    def __post_init__(self) -> None:
        if not isinstance(self.model, ReadoutModel):
            raise TypeError("model must be ReadoutModel")
        site_axis = _require_output_value_schemas(
            self.counts_schema,
            self.occupied_schema,
        )
        if self.model.feature.site_axis != site_axis:
            raise ValueError("worker schemas differ from the selected model")


def _occupancy_operator(payload: object, config: object) -> object:
    """Reviewed synchronous operator used by the generic stream worker."""

    if not isinstance(payload, CameraSample):
        raise TypeError("occupancy processor requires CameraSample")
    if not isinstance(config, _OccupancyConfig):
        raise TypeError("occupancy processor received another config")
    result = _apply_readout_model(config.model, payload.image)
    validity = result.occupied.validity
    if not isinstance(validity, ComponentValidity):
        raise TypeError("readout result requires ComponentValidity")
    return OccupancySample(
        Value(result.signals.values, validity, config.counts_schema),
        Value(result.occupied.values, validity, config.occupied_schema),
        payload.metadata,
    )


@dataclass(frozen=True)
class BoundOccupancyProcessor:
    """Validated calibration/camera binding ready for exact execution."""

    processor: BoundStreamProcessor = field(repr=False)
    capture_input: CaptureProcessorInputBinding = field(repr=False, compare=False)
    output_edge: FrozenDatasetEdge[OccupancySample]
    calibration_reference: CalibrationArtifactRef
    model_kind: ReadoutModelKind

    def __post_init__(self) -> None:
        if not isinstance(self.processor, BoundStreamProcessor):
            raise TypeError("processor must be BoundStreamProcessor")
        if not isinstance(self.capture_input, CaptureProcessorInputBinding):
            raise TypeError("capture_input must be CaptureProcessorInputBinding")
        if not isinstance(self.output_edge, FrozenDatasetEdge):
            raise TypeError("output_edge must be FrozenDatasetEdge")
        if not isinstance(self.calibration_reference, CalibrationArtifactRef):
            raise TypeError("calibration_reference must be CalibrationArtifactRef")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if self.processor.input_payload_contract is not (
            self.capture_input.payload_contract
        ) or self.processor.join_key_contract is not (
            self.capture_input.join_key_contract
        ):
            raise ValueError("processor lost the capture contract owner")
        if self.output_edge.payload_contract is not (
            self.processor.output_payload_contract
        ) or self.output_edge.cell_schedule is not (
            self.capture_input.capture_contract.cell_schedule
        ):
            raise ValueError("processor output edge differs from its bound contracts")

    @property
    def processor_binding_digest(self) -> str:
        return self.processor.fingerprint

    @property
    def output_stream_id(self) -> StreamId:
        return self.processor.output_stream_id

    @property
    def output_source_id(self) -> str:
        return self.processor.output_source_id

    @property
    def output_schema(self) -> DatasetSchema:
        return self.output_edge.schema

    @property
    def output_payload_contract(self) -> OccupancySampleContract:
        contract = self.output_edge.payload_contract
        if not isinstance(contract, OccupancySampleContract):
            raise TypeError("occupancy output edge has another payload contract")
        return contract

    def evaluate(self, sample: CameraSample) -> OccupancySample:
        self.processor.input_payload_contract.validate(sample)
        result = self.processor.operator(sample, self.processor.config)
        if not isinstance(result, OccupancySample):
            raise TypeError("occupancy operator returned another payload type")
        self.output_payload_contract.validate(result)
        return result

    def create_exact_worker(
        self,
        input_reservation: ExactReservation,
        input_cursor: AcquisitionCursor,
        *,
        output_producer: AcquisitionProducer,
        deadline_monotonic: float,
        output_cursor: AcquisitionCursor | None = None,
        output_builder: DatasetBuilder | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ExactStreamProcessorWorker:
        self.capture_input.require_reservation(input_reservation)
        return ExactStreamProcessorWorker(
            self.processor,
            input_reservation,
            input_cursor,
            input_edge=self.capture_input.input_edge,
            output_producer=output_producer,
            deadline_monotonic=deadline_monotonic,
            output_cursor=output_cursor,
            output_builder=output_builder,
            cancellation=cancellation,
        )


def _readout_event_index(capture_input: CaptureProcessorInputBinding) -> int:
    contract = capture_input.capture_contract
    provenance = contract.camera_provenance
    if provenance is None:
        raise ValueError("occupancy requires broker-attested camera provenance")
    declared_id = provenance.descriptor.readout_event_axis_id
    axes = tuple(
        axis for axis in contract.dataset_schema.point_axes if axis.role == READOUT_EVENT
    )
    if declared_id is None:
        if axes:
            raise ValueError("camera descriptor and schema disagree on READOUT_EVENT")
        return 0
    if len(axes) != 1 or axes[0].axis_id != declared_id or axes[0].size != 1:
        raise ValueError("occupancy requires one matching physical READOUT_EVENT")
    return 0


def _validate_camera_binding(
    capture_input: CaptureProcessorInputBinding,
    frame_contract: FrameContract,
) -> CameraSampleContract:
    contract = capture_input.capture_contract
    if type(contract.event_adapter) is not CameraDatasetEventAdapter:
        raise TypeError("occupancy source must use the camera owner adapter")
    payload_contract = capture_input.payload_contract
    if type(payload_contract) is not CameraSampleContract:
        raise TypeError("occupancy source must publish CameraSample")
    provenance = contract.camera_provenance
    if provenance is None or provenance.binding != frame_contract.binding:
        raise ValueError("camera and calibration name different readout bindings")
    frame_contract.assert_compatible(
        frame_contract.binding,
        provenance.descriptor,
        contract.dataset_schema,
        readout_event_index=_readout_event_index(capture_input),
    )
    if payload_contract.value_schema != frame_contract.frame_schema:
        raise ValueError("camera payload schema differs from the FrameContract")
    return payload_contract


def bind_occupancy_processor(
    spec: OccupancyProcessorSpec,
    capture_input: CaptureProcessorInputBinding,
) -> BoundOccupancyProcessor:
    """Validate one physical binding and freeze its exact output edge."""

    if not isinstance(spec, OccupancyProcessorSpec):
        raise TypeError("spec must be OccupancyProcessorSpec")
    if not isinstance(capture_input, CaptureProcessorInputBinding):
        raise TypeError("capture_input must be CaptureProcessorInputBinding")
    source = capture_input.capture_contract
    if spec.output_stream_id == source.stream_id or (
        spec.output_source_id == source.source_id
    ):
        raise ValueError("occupancy output identity must differ from camera input")
    artifact = spec.calibration.artifact
    input_contract = _validate_camera_binding(
        capture_input,
        artifact.frame_contract,
    )
    resolved = resolve_occupancy_processor_schema(spec, source.dataset_schema)
    output_contract = OccupancySampleContract(
        resolved.counts_schema.cell_schema,
        resolved.occupied_schema.cell_schema,
        input_contract.metadata_contract,
    )
    output_edge = FrozenDatasetEdge(
        resolved.counts_schema,
        output_contract,
        source.cell_schedule,
    )
    processor = BoundStreamProcessor(
        OCCUPANCY_PROCESSOR_DEFINITION,
        _OccupancyConfig(
            resolved.selected_model,
            resolved.counts_schema.cell_schema,
            resolved.occupied_schema.cell_schema,
        ),
        input_contract,
        output_contract,
        capture_input.join_key_contract,
        spec.output_stream_id,
        spec.output_source_id,
        _occupancy_operator,
        artifact_inputs=(
            calibration_artifact_input_ref(spec.calibration.reference),
        ),
    )
    return BoundOccupancyProcessor(
        processor,
        capture_input,
        output_edge,
        spec.calibration.reference,
        resolved.model_kind,
    )


def apply_occupancy_snapshot(
    source: OwnedSnapshot,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> tuple[OwnedSnapshot, OwnedSnapshot]:
    """Classify an immutable Camera dataset without reacquiring it."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("source must be OwnedSnapshot")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    model = calibration.artifact.select_model(model_kind)
    resolved = _resolve_occupancy_processor_schema_parts(
        calibration,
        source.block.schema,
        model.kind,
    )
    schema = source.block.schema
    counts, occupied = _classify_cells(
        resolved,
        source.block.revision,
        (
            (repeat_index, point_index, dataset_cell_value(
                source.block,
                repeat_index,
                point_index,
            ))
            for repeat_index in range(schema.repeat_axis.size)
            for point_index in range(schema.point_layout.storage_size)
        ),
    )
    reference = calibration.reference
    generation = StreamGenerationId(
        "occupancy-processor-"
        + canonical_digest(
            {
                "owner": "zlc_neutral_atom.logic_nodes.occupancy.snapshot-application",
                "source_generation": source.ref.stream_generation.value,
                "source_schema": source.ref.schema_fingerprint,
                "calibration_repository": reference.repository_id,
                "calibration_manifest": reference.manifest_digest,
                "model_kind": model.kind.value,
            }
        )
    )
    return (
        OwnedSnapshot(counts.ref(generation), counts),
        OwnedSnapshot(occupied.ref(generation), occupied),
    )


def occupancy_rate_snapshot(occupied: OwnedSnapshot) -> OwnedSnapshot:
    """Reduce the declared SITE axis into a validity-aware occupancy rate."""

    if not isinstance(occupied, OwnedSnapshot):
        raise TypeError("occupied must be an OwnedSnapshot")
    schema = occupied.block.schema
    axes = schema.cell_schema.data_axes
    if len(axes) != 1 or axes[0].role != SITE:
        raise ValueError("occupancy rate requires one declared SITE axis")
    validity = np.asarray(
        expand_dataset_validity(occupied.block.validity, schema),
        dtype=bool,
    )
    values = np.asarray(occupied.block.values, dtype=bool)
    denominator = np.count_nonzero(validity, axis=2)
    cell_validity = denominator > 0
    rate = np.zeros(cell_validity.shape, dtype="<f8")
    np.divide(
        np.count_nonzero(values & validity, axis=2),
        denominator,
        out=rate,
        where=cell_validity,
    )
    rate_schema = DatasetSchema(
        schema.repeat_axis,
        schema.point_axes,
        schema.point_layout,
        ValueSchema.scalar(np.dtype("<f8"), None),
    )
    block = DataBlock(
        OCCUPANCY_RATE_BLOCK_ID,
        occupied.block.revision,
        rate[..., np.newaxis],
        CellValidity(cell_validity),
        rate_schema,
    )
    return OwnedSnapshot(block.ref(occupied.ref.stream_generation), block)


@dataclass(frozen=True, slots=True, eq=False)
class OccupancyProcessorEvaluation:
    """Atomic neutral result for one immutable Camera source revision."""

    outputs: Mapping[str, LiveDatasetOutput]
    background_value: Value
    background_ref: DatasetRevisionRef
    occupied_value: Value
    occupied_ref: DatasetRevisionRef
    selection: Selection
    logical_point: tuple[int, ...]
    site_map: SiteMap
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        outputs = dict(self.outputs)
        expected = tuple(
            declaration.name for declaration in OCCUPANCY_LIVE_OUTPUT_DECLARATIONS
        )
        if tuple(outputs) != expected or any(
            not isinstance(output, LiveDatasetOutput) or output.name != name
            for name, output in outputs.items()
        ):
            raise ValueError("Occupancy outputs must be counts, occupied, and rate")
        if not isinstance(self.background_value, Value) or not isinstance(
            self.occupied_value,
            Value,
        ):
            raise TypeError("background_value and occupied_value must be Value")
        if not isinstance(self.background_ref, DatasetRevisionRef) or not isinstance(
            self.occupied_ref,
            DatasetRevisionRef,
        ):
            raise TypeError("background_ref and occupied_ref must be DatasetRevisionRef")
        if not isinstance(self.selection, Selection):
            raise TypeError("selection must be Selection")
        logical = tuple(self.logical_point)
        if any(type(index) is not int or index < 0 for index in logical):
            raise ValueError("logical_point must contain non-negative integers")
        if not isinstance(self.site_map, SiteMap) or (
            self.occupied_value.schema.data_axes != (self.site_map.site_axis,)
        ):
            raise ValueError("occupied cell differs from the calibration SiteMap")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef) or not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("calibration_ref/model_kind have another type")
        snapshots = tuple(output.snapshot for output in outputs.values())
        if len(
            {
                *(snapshot.ref.revision for snapshot in snapshots),
                self.background_ref.revision,
                self.occupied_ref.revision,
            }
        ) != 1:
            raise ValueError("Occupancy outputs do not share one revision")
        if len({snapshot.ref.stream_generation for snapshot in snapshots}) != 1 or (
            outputs["occupied"].snapshot.ref != self.occupied_ref
        ):
            raise ValueError("Occupancy outputs do not share one generation/cell")
        if len({output.join_digest for output in outputs.values()}) != 1:
            raise ValueError("Occupancy outputs do not share one join")
        object.__setattr__(self, "outputs", MappingProxyType(outputs))
        object.__setattr__(self, "logical_point", logical)


def evaluate_occupancy_processor(
    source: OwnedSnapshot,
    calibration: ResolvedCalibration,
    coverage: MonitorCoverage,
    *,
    model_kind: ReadoutModelKind | None = None,
    source_event_digest: str,
) -> OccupancyProcessorEvaluation:
    """Classify one current Camera revision and select its current display cell."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("source must be OwnedSnapshot")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    if not isinstance(coverage, MonitorCoverage):
        raise TypeError("coverage must be MonitorCoverage")
    source_digest = sha256_text(source_event_digest, "source_event_digest")
    assert source_digest is not None
    model = calibration.artifact.select_model(model_kind)
    counts, occupied = apply_occupancy_snapshot(
        source,
        calibration,
        model_kind=model.kind,
    )
    rate = occupancy_rate_snapshot(occupied)
    point_index, logical_point, selection = current_camera_monitor_selection(
        source.block.schema,
        coverage,
    )
    reference = calibration.reference
    join_digest = canonical_digest(
        {
            "owner": "zlc_neutral_atom.logic_nodes.occupancy.processor-evaluation",
            "source_revision": dataset_revision_ref_to_tree(source.ref),
            "source_event": source_digest,
            "calibration": calibration_artifact_ref_to_tree(reference),
            "model_kind": model.kind.value,
        }
    )
    outputs = {
        declaration.name: LiveDatasetOutput(
            declaration,
            snapshot,
            coverage,
            join_digest,
        )
        for declaration, snapshot in zip(
            OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
            (counts, occupied, rate),
            strict=True,
        )
    }
    return OccupancyProcessorEvaluation(
        outputs,
        dataset_cell_value(source.block, 0, point_index),
        source.ref,
        dataset_cell_value(occupied.block, 0, point_index),
        occupied.ref,
        selection,
        logical_point,
        calibration.artifact.site_map,
        reference,
        model.kind,
    )


__all__ = [
    "apply_occupancy_snapshot",
    "DEFAULT_OCCUPANCY_CALIBRATION_POINTER",
    "OCCUPANCY_COUNTS_BLOCK_ID",
    "OCCUPANCY_OCCUPIED_BLOCK_ID",
    "OCCUPANCY_RATE_BLOCK_ID",
    "OCCUPANCY_PROCESSOR_DEFINITION",
    "OCCUPANCY_PROCESSOR_KEY",
    "OCCUPANCY_EXACT_SCAN_OUTPUT_DECLARATIONS",
    "OCCUPANCY_CAMERA_INPUT_SPEC",
    "OCCUPANCY_CALIBRATION_INPUT_SPEC",
    "OCCUPANCY_LIVE_OUTPUT_DECLARATIONS",
    "OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION",
    "BoundOccupancyProcessor",
    "OccupancyArtifact",
    "OccupancyProcessorConfig",
    "OccupancySample",
    "OccupancySampleContract",
    "OccupancyProcessorSpec",
    "OccupancyProcessorEvaluation",
    "ResolvedOccupancyProcessorSchema",
    "ResolvedOccupancy",
    "bind_occupancy_processor",
    "build_occupancy_processor_config",
    "evaluate_occupancy_processor",
    "occupancy_rate_snapshot",
    "occupancy_authoring_schema",
    "occupancy_input_specs",
    "resolve_occupancy_processor_schema",
]
