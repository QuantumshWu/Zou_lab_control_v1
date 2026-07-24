"""One bound camera-frame to atomic occupancy processor.

Binding joins an admitted calibration to one broker-owned camera generation and
checks the complete physical frame contract once.  The hot path is then only a
synchronous ``CameraSample -> OccupancySample`` call through the calibration
domain's single readout operator.  Exact ordering, gap handling, and full
``(repeat, point, site)`` materialization remain owned by the generic stream and
dataset machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

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
    expand_dataset_validity,
    Invalid,
    OwnedSnapshot,
    Selection,
    StreamGenerationId,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
    Valid,
    INVALID,
    VALID,
    dataset_cell_value,
    dataset_revision_ref_to_tree,
)
from zlc_neutral_atom.camera_measurement import current_camera_monitor_selection
from zlc_neutral_atom.acquisition.camera import (
    CameraDatasetEventAdapter,
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.catalog import DefinitionKey, StreamProcessorDefinition
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from zlc_neutral_atom.processing.stream import (
    BoundStreamProcessor,
    ExactStreamProcessorWorker,
)
from zlc_neutral_atom.runtime.cancellation import CancellationToken
from zlc_neutral_atom.runtime.capture import CaptureProcessorInputBinding
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
from zlc_storage import (
    canonical_digest,
    canonical_text,
    positive_real,
    sha256_text,
)

from .calibration import (
    ReadoutModel,
    ReadoutModelKind,
    ResolvedCalibration,
    SiteMap,
    _apply_readout_model,
)
from .calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_input_ref,
    calibration_artifact_ref_to_tree,
)
from .contracts import FrameContract, ReadoutBindingKey
from .occupancy_reference import OccupancyArtifactRef
from .physical_context import (
    _derive_readout_physical_context_from_evidence,
)


OCCUPANCY_STREAM_PROCESSOR_KEY = DefinitionKey(
    "zlc_neutral_atom.readout",
    "occupancy-stream",
)
OCCUPANCY_LIVE_OUTPUT_NAMES = ("counts", "occupied", "rate")
OCCUPANCY_EXACT_SOURCE_OUTPUT_NAMES = OCCUPANCY_LIVE_OUTPUT_NAMES[:2]
_OCCUPANCY_CONFIG_FORMAT = "zlc_neutral_atom.occupancy-stream-config"
OCCUPANCY_STREAM_PROCESSOR_DEFINITION = StreamProcessorDefinition(
    OCCUPANCY_STREAM_PROCESSOR_KEY,
    "Classify one camera frame into atomic site counts and occupancy",
    _OCCUPANCY_CONFIG_FORMAT,
)
OCCUPANCY_STREAM_PROCESSOR_DEFINITIONS = (
    OCCUPANCY_STREAM_PROCESSOR_DEFINITION,
)
OCCUPANCY_COUNTS_BLOCK_ID = BlockId("occupancy-counts")
OCCUPANCY_OCCUPIED_BLOCK_ID = BlockId("occupancy-occupied")
OCCUPANCY_RATE_BLOCK_ID = BlockId("occupancy-rate")


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
        counts_schema.cell_schema.data_axes,
    ) != (
        occupied_schema.repeat_axis,
        occupied_schema.point_axes,
        occupied_schema.point_layout,
        occupied_schema.cell_schema.data_axes,
    ):
        raise ValueError("occupancy output schemas do not share one axis domain")
    axes = counts_schema.cell_schema.data_axes
    if len(axes) != 1 or axes[0].role != SITE:
        raise ValueError("occupancy output requires exactly one site data axis")
    site_axis = axes[0]
    component_contract = ValidityContract.components(site_axis.axis_id)
    if counts_schema.cell_schema.validity_contract != component_contract or (
        occupied_schema.cell_schema.validity_contract != component_contract
    ):
        raise ValueError("occupancy output requires site component validity")
    if counts_schema.cell_schema.dtype != np.dtype("<f8") or (
        occupied_schema.cell_schema.dtype != np.dtype(bool)
        or occupied_schema.cell_schema.value_unit != "occupation"
    ):
        raise ValueError("occupancy output dtype/unit contracts are not canonical")
    return site_axis


def _same_validity(left: ComponentValidity, right: ComponentValidity) -> bool:
    return left.axis_ids == right.axis_ids and np.array_equal(left.mask, right.mask)


def _validate_sample_fields(counts: Value, occupied: Value) -> None:
    if counts.schema.data_axes != occupied.schema.data_axes:
        raise ValueError("counts and occupied must name the same site axis")
    if not isinstance(counts.validity, ComponentValidity) or not isinstance(
        occupied.validity,
        ComponentValidity,
    ):
        raise TypeError("occupancy fields require ComponentValidity")
    if not _same_validity(counts.validity, occupied.validity):
        raise ValueError("counts and occupied must have identical validity")
    count_values = np.asarray(counts.values)
    occupied_values = np.asarray(occupied.values)
    if not np.all(np.isfinite(count_values)):
        raise ValueError("occupancy counts must be finite")
    invalid = ~counts.validity.mask
    invalid_counts = count_values[invalid]
    if np.any(invalid_counts != 0.0) or np.any(np.signbit(invalid_counts)):
        raise ValueError("invalid counts require canonical positive-zero fillers")
    if np.any(occupied_values[invalid]):
        raise ValueError("invalid occupied sites require canonical False fillers")


@dataclass(frozen=True, slots=True)
class _CommittedOccupancyBinding:
    """Cheap named-axis/model/schema join, before physical-context proof."""

    readout_event_axis_id: AxisId
    model: ReadoutModel
    counts_schema: DatasetSchema
    occupied_schema: DatasetSchema


@dataclass(frozen=True, slots=True)
class ResolvedOccupancyStreamSchema:
    """Pure pre-FIRE occupancy model and output-schema resolution."""

    selected_model: ReadoutModel
    counts_schema: DatasetSchema
    occupied_schema: DatasetSchema

    def __post_init__(self) -> None:
        if not isinstance(self.selected_model, ReadoutModel):
            raise TypeError("selected_model must be ReadoutModel")
        if not isinstance(self.counts_schema, DatasetSchema):
            raise TypeError("counts_schema must be DatasetSchema")
        if not isinstance(self.occupied_schema, DatasetSchema):
            raise TypeError("occupied_schema must be DatasetSchema")
        site_axis = _require_occupancy_output_schemas(
            self.counts_schema,
            self.occupied_schema,
        )
        if self.selected_model.feature.site_axis != site_axis:
            raise ValueError("occupancy output schemas differ from the selected model")

    @property
    def model_kind(self) -> ReadoutModelKind:
        return self.selected_model.kind


_RESOLVED_COMMITTED_OCCUPANCY_TOKEN = object()


@dataclass(frozen=True, init=False, slots=True)
class _ResolvedCommittedOccupancy:
    """Exact admitted inputs after their readout physics has been compared."""

    source: object
    calibration: ResolvedCalibration
    binding: _CommittedOccupancyBinding
    _token: object = field(repr=False, compare=False)

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("_ResolvedCommittedOccupancy is final")

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError(
            "resolved committed occupancy is minted by physical-context admission"
        )

    def __reduce__(self):
        raise TypeError("resolved committed occupancy is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("resolved committed occupancy is process-local")

    @classmethod
    def _from_context(
        cls,
        token: object,
        source: object,
        calibration: ResolvedCalibration,
        binding: _CommittedOccupancyBinding,
    ) -> "_ResolvedCommittedOccupancy":
        from zlc_neutral_atom.artifacts.capture import AdmittedCapture

        if token is not _RESOLVED_COMMITTED_OCCUPANCY_TOKEN:
            raise PermissionError(
                "resolved occupancy inputs require physical-context admission"
            )
        if type(source) is not AdmittedCapture:
            raise TypeError("source must be an exact AdmittedCapture")
        source._require_authority()
        if type(calibration) is not ResolvedCalibration:
            raise TypeError("calibration must be an exact ResolvedCalibration")
        calibration._require_authority()
        if not isinstance(binding, _CommittedOccupancyBinding):
            raise TypeError("binding must be a committed occupancy structure")
        resolved = object.__new__(cls)
        object.__setattr__(resolved, "source", source)
        object.__setattr__(resolved, "calibration", calibration)
        object.__setattr__(resolved, "binding", binding)
        object.__setattr__(resolved, "_token", token)
        return resolved

    def _require_authority(
        self,
    ) -> tuple[object, ResolvedCalibration, _CommittedOccupancyBinding]:
        from zlc_neutral_atom.artifacts.capture import AdmittedCapture

        if type(self) is not _ResolvedCommittedOccupancy or (
            self._token is not _RESOLVED_COMMITTED_OCCUPANCY_TOKEN
        ):
            raise PermissionError("resolved occupancy input authority is invalid")
        if type(self.source) is not AdmittedCapture:
            raise PermissionError("resolved occupancy source authority is invalid")
        self.source._require_authority()
        if type(self.calibration) is not ResolvedCalibration:
            raise PermissionError(
                "resolved occupancy calibration authority is invalid"
            )
        self.calibration._require_authority()
        if not isinstance(self.binding, _CommittedOccupancyBinding):
            raise PermissionError("resolved occupancy binding authority is invalid")
        return self.source, self.calibration, self.binding


def _resolve_committed_occupancy_structure(
    capture: object,
    calibration: ResolvedCalibration,
    *,
    readout_event_axis_id: AxisId,
    model_kind: ReadoutModelKind,
) -> _CommittedOccupancyBinding:
    """Resolve the cheap named-axis, frame, model, and output-schema join."""

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
    except AttributeError as exc:
        raise TypeError("capture must be a resolved raw CaptureArtifact") from exc
    schema = source.schema
    event_axes = tuple(axis for axis in schema.point_axes if axis.role == READOUT_EVENT)
    if len(event_axes) != 1:
        raise ValueError(
            "authoritative occupancy requires exactly one named READOUT_EVENT axis"
        )
    event_axis = event_axes[0]
    if event_axis.axis_id != readout_event_axis_id:
        raise ValueError("capture and request name different READOUT_EVENT axes")
    if event_axis.size != 1:
        raise ValueError(
            "committed occupancy baseline requires a singleton READOUT_EVENT axis"
        )
    descriptor = provenance.descriptor
    if descriptor.readout_event_axis_id != event_axis.axis_id:
        raise ValueError("camera descriptor names another READOUT_EVENT axis")

    artifact = calibration.artifact
    if provenance.binding != artifact.frame_contract.binding:
        raise ValueError("capture and calibration name different readout bindings")
    artifact.frame_contract.assert_compatible(
        provenance.binding,
        descriptor,
        schema,
        readout_event_index=0,
    )
    resolved = _resolve_occupancy_stream_schema_parts(
        calibration,
        schema,
        model_kind,
    )
    return _CommittedOccupancyBinding(
        event_axis.axis_id,
        resolved.selected_model,
        resolved.counts_schema,
        resolved.occupied_schema,
    )


def _require_committed_occupancy_context(
    source: object,
    calibration: ResolvedCalibration,
    binding: _CommittedOccupancyBinding,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> _ResolvedCommittedOccupancy:
    """Stream-compare every selected pulse window."""

    from zlc_neutral_atom.artifacts.capture import AdmittedCapture

    if type(source) is not AdmittedCapture:
        raise TypeError("source must be an exact AdmittedCapture")
    source._require_authority()
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an exact ResolvedCalibration")
    calibration._require_authority()
    if not isinstance(binding, _CommittedOccupancyBinding):
        raise TypeError("binding must be resolved committed occupancy")
    capture = source.artifact
    try:
        frame_source = capture.frame_source  # type: ignore[attr-defined]
        capability = capture.camera_capability_evidence  # type: ignore[attr-defined]
        pulse_evidence = capture.pulse_evidence  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("capture must be a resolved raw CaptureArtifact") from exc
    if pulse_evidence is None:
        raise ValueError("authoritative occupancy requires persisted pulse lineage")
    current_context = _derive_readout_physical_context_from_evidence(
        pulse_evidence,
        frame_source.schema,
        frame_source.iter_cell_schedule(),
        readout_event_index=0,
        integration_start_offset_seconds=(
            capability.physical_facts
            .external_trigger_integration_start_offset_seconds
        ),
        integration_seconds=(
            calibration.artifact.frame_contract.exposure_seconds
        ),
        checkpoint=checkpoint,
    )
    if current_context != calibration.artifact.readout_physical_context:
        raise ValueError("capture pulse context differs from the calibration")
    return _ResolvedCommittedOccupancy._from_context(
        _RESOLVED_COMMITTED_OCCUPANCY_TOKEN,
        source,
        calibration,
        binding,
    )


def _occupancy_generation_for_run(run_id: str) -> StreamGenerationId:
    run = canonical_text(run_id, "run_id")
    return StreamGenerationId(
        canonical_digest(
            {
                "owner": "zlc_neutral_atom.readout.committed-occupancy-run",
                "run_id": run,
            }
        )
    )


@dataclass(frozen=True, eq=False)
class OccupancyArtifact:
    """Durable counts/occupation derived from two explicit committed inputs."""

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
        if self.counts.revision != self.occupied.revision:
            raise ValueError("occupancy blocks must share one revision")
        if self.counts.validity is not self.occupied.validity:
            raise ValueError("occupancy blocks must share one validity authority")
        validity = self.counts.validity
        if not isinstance(validity, ComponentValidity) or (
            validity.axis_ids != (site_axis.axis_id,)
        ):
            raise ValueError("occupancy validity must name exactly the site axis")
        if not np.all(np.isfinite(self.counts.values)):
            raise ValueError("occupancy counts must be finite")
        invalid = ~validity.mask
        invalid_counts = self.counts.values[invalid]
        if np.any(invalid_counts != 0.0) or np.any(np.signbit(invalid_counts)):
            raise ValueError("invalid counts require canonical positive-zero fillers")
        if np.any(self.occupied.values[invalid]):
            raise ValueError("invalid occupied sites require canonical False fillers")
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
        raise TypeError("ResolvedOccupancy is final and cannot be subclassed")

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError("ResolvedOccupancy is returned by OccupancyRepository.admit")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ResolvedOccupancy is immutable")

    def __reduce__(self):
        raise TypeError("ResolvedOccupancy is process-local and cannot be serialized")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("ResolvedOccupancy is process-local and cannot be serialized")

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
            raise PermissionError(
                "ResolvedOccupancy can only be minted by OccupancyRepository.admit"
            )
        if repository_token is None:
            raise ValueError("ResolvedOccupancy repository authority is absent")
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
        if (
            type(self) is not ResolvedOccupancy
            or self._token is not _RESOLVED_OCCUPANCY_TOKEN
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

_OCCUPANCY_ANALYSIS_TOKEN = object()


@dataclass(frozen=True, init=False)
class OccupancyAnalysisResult:
    """Submittable result minted from exact source and calibration admissions."""

    artifact: OccupancyArtifact
    _token: object = field(repr=False, compare=False)
    _resolved_input: _ResolvedCommittedOccupancy = field(
        repr=False,
        compare=False,
    )
    _run_id: str = field(repr=False, compare=False)

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("OccupancyAnalysisResult is final and cannot be subclassed")

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError("OccupancyAnalysisResult is returned by occupancy analysis")

    def __reduce__(self):
        raise TypeError("OccupancyAnalysisResult is process-local and cannot be serialized")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("OccupancyAnalysisResult is process-local and cannot be serialized")

    @classmethod
    def _from_analysis(
        cls,
        token: object,
        artifact: OccupancyArtifact,
        resolved: _ResolvedCommittedOccupancy,
        run_id: str,
    ) -> "OccupancyAnalysisResult":
        if token is not _OCCUPANCY_ANALYSIS_TOKEN:
            raise PermissionError(
                "OccupancyAnalysisResult can only be minted by occupancy analysis"
            )
        if not isinstance(artifact, OccupancyArtifact):
            raise TypeError("artifact must be OccupancyArtifact")
        if type(resolved) is not _ResolvedCommittedOccupancy:
            raise TypeError("resolved must be exact committed occupancy inputs")
        source, calibration, binding = resolved._require_authority()
        if artifact.source_capture_ref != source.reference:
            raise ValueError("occupancy result names another admitted capture")
        if artifact.calibration_reference != calibration.reference:
            raise ValueError("occupancy result names another calibration admission")
        if (
            artifact.readout_event_axis_id != binding.readout_event_axis_id
            or artifact.model_kind is not binding.model.kind
            or artifact.counts.schema != binding.counts_schema
            or artifact.occupied.schema != binding.occupied_schema
        ):
            raise ValueError("occupancy result differs from its resolved input")
        run = canonical_text(run_id, "run_id")
        if artifact.generation != _occupancy_generation_for_run(run):
            raise ValueError("occupancy generation was not minted by this Run")
        result = object.__new__(cls)
        object.__setattr__(result, "artifact", artifact)
        object.__setattr__(result, "_token", token)
        object.__setattr__(result, "_resolved_input", resolved)
        object.__setattr__(result, "_run_id", run)
        return result

    def _admissions_for_commit(
        self,
        run_id: str,
    ) -> _ResolvedCommittedOccupancy:
        """Return the exact retained inputs after validating result authority."""

        if type(self) is not OccupancyAnalysisResult or (
            self._token is not _OCCUPANCY_ANALYSIS_TOKEN
        ):
            raise PermissionError("occupancy result authority is invalid")
        resolved = self._resolved_input
        if type(resolved) is not _ResolvedCommittedOccupancy:
            raise PermissionError("occupancy result input authority is invalid")
        source, calibration, binding = resolved._require_authority()
        if self.artifact.source_capture_ref != source.reference or (
            self.artifact.calibration_reference != calibration.reference
        ):
            raise PermissionError("occupancy result inputs changed after analysis")
        if (
            self.artifact.readout_event_axis_id != binding.readout_event_axis_id
            or self.artifact.model_kind is not binding.model.kind
            or self.artifact.counts.schema != binding.counts_schema
            or self.artifact.occupied.schema != binding.occupied_schema
        ):
            raise PermissionError("occupancy result binding changed after analysis")
        run = canonical_text(run_id, "run_id")
        if self._run_id != run or (
            self.artifact.generation != _occupancy_generation_for_run(run)
        ):
            raise PermissionError("occupancy result belongs to another Run")
        return resolved


def _analyze_committed_occupancy_resolved(
    resolved: _ResolvedCommittedOccupancy,
    *,
    run_id: str,
    checkpoint: Callable[[], None],
) -> OccupancyAnalysisResult:
    """Stream raw frames once and preserve the complete R/P/site domain."""

    if type(resolved) is not _ResolvedCommittedOccupancy:
        raise TypeError("occupancy analysis requires resolved committed inputs")
    source, calibration, binding = resolved._require_authority()
    if not callable(checkpoint):
        raise TypeError("checkpoint must be callable")
    run = canonical_text(run_id, "run_id")
    schema = binding.counts_schema
    counts_values = np.zeros(schema.physical_shape, dtype="<f8")
    occupied_values = np.zeros(binding.occupied_schema.physical_shape, dtype=bool)
    validity_values = np.zeros(schema.physical_shape, dtype=bool)
    for cell, sample in source.artifact.frame_source.iter_event_order():
        checkpoint()
        result = _apply_readout_model(binding.model, sample.image)
        validity = result.occupied.validity
        if not isinstance(validity, ComponentValidity):
            raise TypeError("readout result requires ComponentValidity")
        location = (cell.repeat_index, cell.point_storage_index)
        counts_values[location] = result.signals.values
        occupied_values[location] = result.occupied.values
        validity_values[location] = validity.mask
    checkpoint()
    validity = ComponentValidity(
        (binding.model.feature.site_axis.axis_id,),
        validity_values,
    )
    del validity_values
    revision = source.artifact.frame_source.revision
    counts = DataBlock(
        OCCUPANCY_COUNTS_BLOCK_ID,
        revision,
        counts_values,
        validity,
        binding.counts_schema,
    )
    del counts_values
    occupied = DataBlock(
        OCCUPANCY_OCCUPIED_BLOCK_ID,
        revision,
        occupied_values,
        validity,
        binding.occupied_schema,
    )
    del occupied_values
    artifact = OccupancyArtifact(
        source.reference,
        calibration.reference,
        binding.readout_event_axis_id,
        binding.model.kind,
        _occupancy_generation_for_run(run),
        counts,
        occupied,
    )
    return OccupancyAnalysisResult._from_analysis(
        _OCCUPANCY_ANALYSIS_TOKEN,
        artifact,
        resolved,
        run,
    )


@dataclass(frozen=True, eq=False)
class OccupancySample:
    """Counts and occupancy atomically derived from one physical camera frame."""

    counts: Value
    occupied: Value
    metadata: CameraFrameMetadata

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.counts, Value) or not isinstance(self.occupied, Value):
            raise TypeError("counts and occupied must be zlc_data.Value")
        if not isinstance(self.metadata, CameraFrameMetadata):
            raise TypeError("metadata must be CameraFrameMetadata")
        _validate_sample_fields(self.counts, self.occupied)


@dataclass(frozen=True)
class OccupancySampleContract:
    """The sole live-payload contract for atomic occupancy samples."""

    counts_schema: ValueSchema
    occupied_schema: ValueSchema
    metadata_contract: CameraFrameMetadataContract

    def __post_init__(self) -> None:
        if not isinstance(self.counts_schema, ValueSchema) or not isinstance(
            self.occupied_schema,
            ValueSchema,
        ):
            raise TypeError("occupancy schemas must be ValueSchema")
        if not isinstance(self.metadata_contract, CameraFrameMetadataContract):
            raise TypeError("metadata_contract must be CameraFrameMetadataContract")
        axes = self.counts_schema.data_axes
        if len(axes) != 1 or axes[0].role != SITE:
            raise ValueError("occupancy output requires exactly one site axis")
        if self.occupied_schema.data_axes != axes:
            raise ValueError("counts and occupied schemas must share the site axis")
        validity = ValidityContract.components(axes[0].axis_id)
        if self.counts_schema.validity_contract != validity or (
            self.occupied_schema.validity_contract != validity
        ):
            raise ValueError("occupancy schemas require site component validity")
        if self.counts_schema.dtype != np.dtype("<f8"):
            raise ValueError("counts must use canonical float64")
        if self.occupied_schema.dtype != np.dtype(bool):
            raise ValueError("occupied must use bool")
        if self.occupied_schema.value_unit != "occupation":
            raise ValueError("occupied must use the canonical occupation unit")

    @property
    def _counts(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.counts_schema)

    @property
    def _occupied(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.occupied_schema)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.OccupancySample",
                "counts": self._counts.fingerprint,
                "occupied": self._occupied.fingerprint,
                "metadata": self.metadata_contract.fingerprint,
            }
        )

    def snapshot(self, payload: OccupancySample) -> OccupancySample:
        self.validate(payload)
        return payload

    def validate(self, payload: OccupancySample) -> None:
        if not isinstance(payload, OccupancySample):
            raise TypeError("payload must be OccupancySample")
        self._counts.validate(payload.counts)
        self._occupied.validate(payload.occupied)
        self.metadata_contract.validate(payload.metadata)

    def digest(self, payload: OccupancySample) -> str:
        self.validate(payload)
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.OccupancySampleContent",
                "counts": self._counts.digest(payload.counts),
                "occupied": self._occupied.digest(payload.occupied),
                "metadata": self.metadata_contract.digest(payload.metadata),
            }
        )

    @staticmethod
    def source_ordinal(payload: OccupancySample) -> int:
        return payload.metadata.source_ordinal

    @staticmethod
    def captured_at(payload: OccupancySample) -> float:
        return payload.metadata.captured_at

    @staticmethod
    def correlation_id(payload: OccupancySample) -> str:
        return payload.metadata.correlation_id


@dataclass(frozen=True, eq=False)
class OccupancyDatasetMetadata:
    """The occupied cell and camera metadata co-sealed beside counts."""

    occupied: Value
    source_metadata: CameraFrameMetadata

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.occupied, Value):
            raise TypeError("occupied must be Value")
        if not isinstance(self.source_metadata, CameraFrameMetadata):
            raise TypeError("source_metadata must be CameraFrameMetadata")
        validity = self.occupied.validity
        if not isinstance(validity, ComponentValidity):
            raise TypeError("occupied metadata requires ComponentValidity")
        if np.any(self.occupied.values[~validity.mask]):
            raise ValueError("invalid occupied metadata requires False fillers")


@dataclass(frozen=True)
class _OccupancyDatasetMetadataContract:
    payload_contract: OccupancySampleContract

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, OccupancySampleContract):
            raise TypeError("payload_contract must be OccupancySampleContract")

    @property
    def _occupied(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.payload_contract.occupied_schema)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.OccupancyDatasetMetadata",
                "occupied": self._occupied.fingerprint,
                "source": self.payload_contract.metadata_contract.fingerprint,
            }
        )

    def snapshot(self, payload: OccupancySample) -> OccupancyDatasetMetadata:
        self.payload_contract.validate(payload)
        return OccupancyDatasetMetadata(payload.occupied, payload.metadata)

    def validate(self, metadata: object | None) -> None:
        if not isinstance(metadata, OccupancyDatasetMetadata):
            raise TypeError("metadata must be OccupancyDatasetMetadata")
        self._occupied.validate(metadata.occupied)
        self.payload_contract.metadata_contract.validate(metadata.source_metadata)

    def digest(self, metadata: object | None) -> str:
        self.validate(metadata)
        assert isinstance(metadata, OccupancyDatasetMetadata)
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.OccupancyDatasetMetadataContent",
                "occupied": self._occupied.digest(metadata.occupied),
                "source": self.payload_contract.metadata_contract.digest(
                    metadata.source_metadata
                ),
            }
        )


@dataclass(frozen=True)
class OccupancyDatasetEventAdapter:
    """Project counts as the DataBlock cell and co-seal occupied metadata."""

    payload_contract: OccupancySampleContract

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, OccupancySampleContract):
            raise TypeError("payload_contract must be OccupancySampleContract")

    @property
    def metadata_contract(self) -> _OccupancyDatasetMetadataContract:
        return _OccupancyDatasetMetadataContract(self.payload_contract)

    @property
    def operator_fingerprint(self) -> str:
        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.readout.OccupancyDatasetEventAdapter",
                "payload": self.payload_contract.fingerprint,
            }
        )

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.counts_schema

    @staticmethod
    def value(payload: OccupancySample) -> Value:
        return payload.counts


@dataclass(frozen=True)
class OccupancyStreamProcessorSpec:
    """Reusable calibration choice; a CaptureSession supplies the input."""

    calibration: ResolvedCalibration
    output_stream_id: StreamId
    output_source_id: str
    model_kind: ReadoutModelKind | None = None
    operator_deadline_seconds: float = 1.0
    terminal_wait_seconds: float = 1.0

    def __post_init__(self) -> None:
        if type(self.calibration) is not ResolvedCalibration:
            raise TypeError("calibration must be an exact ResolvedCalibration")
        self.calibration._require_authority()
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")
        selected = self.calibration.artifact.select_model(self.model_kind)
        object.__setattr__(self, "model_kind", selected.kind)
        if not isinstance(self.output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        canonical_text(self.output_source_id, "output_source_id")
        object.__setattr__(
            self,
            "operator_deadline_seconds",
            positive_real(self.operator_deadline_seconds, "operator_deadline_seconds"),
        )
        object.__setattr__(
            self,
            "terminal_wait_seconds",
            positive_real(self.terminal_wait_seconds, "terminal_wait_seconds"),
        )


@dataclass(frozen=True, eq=False)
class _OccupancyConfig:
    model: ReadoutModel
    input_schema: ValueSchema
    counts_schema: ValueSchema
    occupied_schema: ValueSchema

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ReadoutModel):
            raise TypeError("model must be ReadoutModel")
        if not all(
            isinstance(schema, ValueSchema)
            for schema in (self.input_schema, self.counts_schema, self.occupied_schema)
        ):
            raise TypeError("processor schemas must be ValueSchema")
        if self.counts_schema.data_axes != (self.model.feature.site_axis,) or (
            self.occupied_schema.data_axes != (self.model.feature.site_axis,)
        ):
            raise ValueError("processor output schemas differ from the model site axis")


def _occupancy_operator(payload: object, config: object) -> object:
    """Reviewed synchronous operator used by the generic stream worker."""

    if not isinstance(payload, CameraSample):
        raise TypeError("occupancy processor requires CameraSample")
    if not isinstance(config, _OccupancyConfig):
        raise TypeError("occupancy processor received another config")
    if payload.image.schema is not config.input_schema:
        raise TypeError("camera image must share the bound ValueSchema owner")
    # The complete physical FrameContract was checked once by the binder;
    # repeating it for every frame would create a second invariant owner.
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
class BoundOccupancyStreamProcessor:
    """One validated calibration/camera binding ready for exact execution."""

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
        input_contract = self.capture_input.payload_contract
        if self.processor.input_payload_contract is not input_contract:
            raise ValueError("processor lost the capture payload owner")
        if self.processor.join_key_contract is not self.capture_input.join_key_contract:
            raise ValueError("processor lost the capture key owner")
        if self.output_edge.payload_contract is not self.processor.output_payload_contract:
            raise ValueError("processor and output edge use different payload owners")
        if self.output_edge.cell_schedule is not (
            self.capture_input.capture_contract.cell_schedule
        ):
            raise ValueError("occupancy output schedule differs from camera input")

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
    event_axis_id = provenance.descriptor.readout_event_axis_id
    event_axes = tuple(
        (position, axis)
        for position, axis in enumerate(contract.dataset_schema.point_axes)
        if axis.role == READOUT_EVENT
    )
    if event_axis_id is None:
        if event_axes:
            raise ValueError("camera descriptor and schema disagree on READOUT_EVENT")
        return 0
    if len(event_axes) != 1 or event_axes[0][1].axis_id != event_axis_id:
        raise ValueError("camera descriptor and schema name different READOUT_EVENT axes")
    if event_axes[0][1].size != 1:
        raise ValueError(
            "occupancy input must contain exactly one physical READOUT_EVENT"
        )
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
    if provenance is None:
        raise ValueError("occupancy source lacks camera provenance")
    if provenance.binding != frame_contract.binding:
        raise ValueError("camera and calibration name different readout bindings")
    event_index = _readout_event_index(capture_input)
    frame_contract.assert_compatible(
        frame_contract.binding,
        provenance.descriptor,
        contract.dataset_schema,
        readout_event_index=event_index,
    )
    if payload_contract.value_schema.fingerprint != frame_contract.frame_schema.fingerprint:
        raise ValueError("camera payload schema differs from the calibration FrameContract")
    return payload_contract


def _output_schemas(
    frame_contract: FrameContract,
    site_axis: AxisSpec,
) -> tuple[ValueSchema, ValueSchema]:
    validity = ValidityContract.components(site_axis.axis_id)
    return (
        ValueSchema((site_axis,), validity, np.dtype("<f8"), frame_contract.count_unit),
        ValueSchema((site_axis,), validity, np.dtype(bool), "occupation"),
    )


def _resolve_occupancy_stream_schema_parts(
    calibration: ResolvedCalibration,
    source_schema: DatasetSchema,
    model_kind: ReadoutModelKind,
) -> ResolvedOccupancyStreamSchema:
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an exact ResolvedCalibration")
    calibration._require_authority()
    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(model_kind, ReadoutModelKind):
        raise TypeError("model_kind must be a concrete ReadoutModelKind")

    artifact = calibration.artifact
    frame_contract = artifact.frame_contract
    if (
        source_schema.cell_schema.fingerprint
        != frame_contract.frame_schema.fingerprint
    ):
        raise ValueError("source schema differs from the calibration FrameContract")
    model = artifact.select_model(model_kind)
    if model.kind is not model_kind:
        raise ValueError("selected model differs from the frozen processor spec")
    site_axis = artifact.site_map.site_axis
    if model.feature.site_axis != site_axis:
        raise ValueError("selected model and SiteMap use different site axes")
    outer_axis_ids = {
        source_schema.repeat_axis.axis_id,
        *(axis.axis_id for axis in source_schema.point_axes),
    }
    if site_axis.axis_id in outer_axis_ids:
        raise ValueError("site AxisId collides with a capture repeat/point AxisId")

    counts_value, occupied_value = _output_schemas(frame_contract, site_axis)
    return ResolvedOccupancyStreamSchema(
        model,
        DatasetSchema(
            source_schema.repeat_axis,
            source_schema.point_axes,
            source_schema.point_layout,
            counts_value,
        ),
        DatasetSchema(
            source_schema.repeat_axis,
            source_schema.point_axes,
            source_schema.point_layout,
            occupied_value,
        ),
    )


def resolve_occupancy_stream_schema(
    spec: OccupancyStreamProcessorSpec,
    source_schema: DatasetSchema,
) -> ResolvedOccupancyStreamSchema:
    """Freeze occupancy outputs before any camera session or hardware prepare."""

    if not isinstance(spec, OccupancyStreamProcessorSpec):
        raise TypeError("spec must be OccupancyStreamProcessorSpec")
    model_kind = spec.model_kind
    if not isinstance(model_kind, ReadoutModelKind):
        raise TypeError("processor spec has no concrete model_kind")
    return _resolve_occupancy_stream_schema_parts(
        spec.calibration,
        source_schema,
        model_kind,
    )


def apply_occupancy_snapshot(
    source: OwnedSnapshot,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> tuple[OwnedSnapshot, OwnedSnapshot]:
    """Classify one already-published camera dataset without reacquiring it.

    This is the reactive Processor path: the caller supplies one immutable
    camera revision and an admitted calibration.  The function preserves the
    source repeat/point axes and revision, adds only the calibration's SITE
    axis, and returns counts/occupied as one pair.  It has no camera, pulse,
    timeout, buffering, or scheduling authority.

    Physical camera-role admission remains the composition root's job because
    a materialized :class:`OwnedSnapshot` intentionally carries data lineage,
    not a device capability.  This owner still rejects any structural frame
    mismatch against the calibration's complete frame schema.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("source must be an OwnedSnapshot")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    artifact = calibration.artifact
    model = artifact.select_model(model_kind)
    resolved = _resolve_occupancy_stream_schema_parts(
        calibration,
        source.block.schema,
        model.kind,
    )
    source_block = source.block
    source_schema = source_block.schema
    counts_schema = resolved.counts_schema
    occupied_schema = resolved.occupied_schema
    counts_values = np.zeros(counts_schema.physical_shape, dtype="<f8")
    occupied_values = np.zeros(occupied_schema.physical_shape, dtype=bool)
    validity_values = np.zeros(counts_schema.physical_shape, dtype=bool)

    def source_cell_validity(repeat_index: int, point_index: int):
        validity = source_block.validity
        if isinstance(validity, CellValidity):
            return VALID if validity.mask[repeat_index, point_index] else INVALID
        if isinstance(validity, ComponentValidity):
            return ComponentValidity(
                validity.axis_ids,
                validity.mask[repeat_index, point_index],
            )
        if isinstance(validity, (Valid, Invalid)):
            return validity
        raise TypeError("camera dataset has unsupported validity")

    for repeat_index in range(source_schema.repeat_axis.size):
        for point_index in range(source_schema.point_layout.storage_size):
            frame = Value(
                source_block.values[repeat_index, point_index],
                source_cell_validity(repeat_index, point_index),
                source_schema.cell_schema,
            )
            result = _apply_readout_model(model, frame)
            validity = result.occupied.validity
            if not isinstance(validity, ComponentValidity):
                raise TypeError("readout result requires ComponentValidity")
            location = (repeat_index, point_index)
            counts_values[location] = result.signals.values
            occupied_values[location] = result.occupied.values
            validity_values[location] = validity.mask

    site_axis = resolved.selected_model.feature.site_axis
    validity = ComponentValidity((site_axis.axis_id,), validity_values)
    reference = calibration.reference
    generation_digest = canonical_digest(
        {
            "owner": "zlc_neutral_atom.reactive-occupancy",
            "source_generation": source.ref.stream_generation.value,
            "source_schema": source.ref.schema_fingerprint,
            "calibration_repository": reference.repository_id,
            "calibration_manifest": reference.manifest_digest,
            "model_kind": model.kind.value,
        }
    )
    generation = StreamGenerationId(f"reactive-occupancy-{generation_digest}")
    counts = DataBlock(
        OCCUPANCY_COUNTS_BLOCK_ID,
        source_block.revision,
        counts_values,
        validity,
        counts_schema,
    )
    occupied = DataBlock(
        OCCUPANCY_OCCUPIED_BLOCK_ID,
        source_block.revision,
        occupied_values,
        validity,
        occupied_schema,
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
        raise ValueError("occupancy rate requires exactly one declared SITE axis")
    validity = np.asarray(
        expand_dataset_validity(occupied.block.validity, schema),
        dtype=np.bool_,
    )
    values = np.asarray(occupied.block.values, dtype=np.bool_)
    denominator = np.count_nonzero(validity, axis=2)
    numerator = np.count_nonzero(values & validity, axis=2)
    cell_validity = denominator > 0
    rate_values = np.zeros(cell_validity.shape, dtype="<f8")
    np.divide(
        numerator,
        denominator,
        out=rate_values,
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
        rate_values[..., np.newaxis],
        CellValidity(cell_validity),
        rate_schema,
    )
    return OwnedSnapshot(block.ref(occupied.ref.stream_generation), block)


@dataclass(frozen=True, slots=True, eq=False)
class OccupancyMonitorCellContext:
    """One exact monitor cell and the calibration geometry that judges it."""

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
        if not isinstance(self.background_value, Value):
            raise TypeError("background_value must be Value")
        if not isinstance(self.background_ref, DatasetRevisionRef):
            raise TypeError("background_ref must be DatasetRevisionRef")
        if not isinstance(self.occupied_value, Value):
            raise TypeError("occupied_value must be Value")
        if not isinstance(self.occupied_ref, DatasetRevisionRef):
            raise TypeError("occupied_ref must be DatasetRevisionRef")
        if self.background_ref.revision != self.occupied_ref.revision:
            raise ValueError("monitor background and occupancy revisions differ")
        if not isinstance(self.selection, Selection):
            raise TypeError("selection must be Selection")
        logical_point = tuple(self.logical_point)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in logical_point
        ):
            raise ValueError("logical_point must contain non-negative integers")
        if not isinstance(self.site_map, SiteMap):
            raise TypeError("site_map must be SiteMap")
        if self.occupied_value.schema.data_axes != (self.site_map.site_axis,):
            raise ValueError("occupied monitor cell differs from calibration SITE axis")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        object.__setattr__(self, "logical_point", logical_point)


@dataclass(frozen=True, slots=True, eq=False)
class ReactiveOccupancyMonitorEvaluation:
    """Atomic neutral result for one immutable Camera monitor revision."""

    outputs: Mapping[str, LiveDatasetOutput]
    cell: OccupancyMonitorCellContext

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        outputs = dict(self.outputs)
        if tuple(outputs) != OCCUPANCY_LIVE_OUTPUT_NAMES:
            raise ValueError(
                "reactive occupancy outputs must be counts, occupied, and rate"
            )
        for name, output in outputs.items():
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError(
                    "reactive occupancy outputs must contain LiveDatasetOutput"
                )
            if output.name != name:
                raise ValueError("reactive occupancy output key differs from its name")
        if not isinstance(self.cell, OccupancyMonitorCellContext):
            raise TypeError("cell must be OccupancyMonitorCellContext")
        snapshots = tuple(output.snapshot for output in outputs.values())
        revisions = {
            *(snapshot.ref.revision for snapshot in snapshots),
            self.cell.background_ref.revision,
            self.cell.occupied_ref.revision,
        }
        if len(revisions) != 1:
            raise ValueError("reactive occupancy outputs do not share one revision")
        if len(
            {
                snapshot.ref.stream_generation for snapshot in snapshots
            }
        ) != 1:
            raise ValueError("reactive occupancy outputs do not share one generation")
        if outputs["occupied"].snapshot.ref != self.cell.occupied_ref:
            raise ValueError("reactive occupancy cell differs from occupied output")
        if len({output.join_digest for output in outputs.values()}) != 1:
            raise ValueError("reactive occupancy outputs do not share one join")
        object.__setattr__(self, "outputs", MappingProxyType(outputs))


def evaluate_reactive_occupancy_monitor(
    source: OwnedSnapshot,
    calibration: ResolvedCalibration,
    coverage: MonitorCoverage,
    *,
    model_kind: ReadoutModelKind | None = None,
    source_event_digest: str,
) -> ReactiveOccupancyMonitorEvaluation:
    """Classify and select one current Camera monitor revision atomically.

    This is the sole domain seam for TaskConsole-style reactive occupancy.  It
    performs classification, validity-aware rate reduction, Camera-owned
    current-cell resolution, and same-revision cell extraction.  It returns no
    frontend type and performs no rendering.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("source must be OwnedSnapshot")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    if not isinstance(coverage, MonitorCoverage):
        raise TypeError("coverage must be MonitorCoverage")
    source_digest = sha256_text(source_event_digest, "source_event_digest")
    assert source_digest is not None
    selected_model = calibration.artifact.select_model(model_kind)
    counts, occupied = apply_occupancy_snapshot(
        source,
        calibration,
        model_kind=selected_model.kind,
    )
    rate = occupancy_rate_snapshot(occupied)
    point_index, logical_point, selection = current_camera_monitor_selection(
        source.block.schema,
        coverage,
    )
    reference = calibration.reference
    cell = OccupancyMonitorCellContext(
        dataset_cell_value(source.block, 0, point_index),
        source.ref,
        dataset_cell_value(occupied.block, 0, point_index),
        occupied.ref,
        selection,
        logical_point,
        calibration.artifact.site_map,
        reference,
        selected_model.kind,
    )
    join_digest = canonical_digest(
        {
            "owner": "zlc_neutral_atom.reactive-occupancy-monitor",
            "source_revision": dataset_revision_ref_to_tree(source.ref),
            "source_event": source_digest,
            "calibration": calibration_artifact_ref_to_tree(reference),
            "model_kind": selected_model.kind.value,
        }
    )
    outputs = {
        name: LiveDatasetOutput(name, snapshot, coverage, join_digest)
        for name, snapshot in zip(
            OCCUPANCY_LIVE_OUTPUT_NAMES,
            (counts, occupied, rate),
            strict=True,
        )
    }
    return ReactiveOccupancyMonitorEvaluation(outputs, cell)


def bind_occupancy_stream_processor(
    spec: OccupancyStreamProcessorSpec,
    capture_input: CaptureProcessorInputBinding,
) -> BoundOccupancyStreamProcessor:
    """Validate one complete physical binding and freeze its exact output edge."""

    if not isinstance(spec, OccupancyStreamProcessorSpec):
        raise TypeError("spec must be OccupancyStreamProcessorSpec")
    if not isinstance(capture_input, CaptureProcessorInputBinding):
        raise TypeError("capture_input must be CaptureProcessorInputBinding")
    source = capture_input.capture_contract
    if spec.output_stream_id == source.stream_id:
        raise ValueError("occupancy output stream must differ from camera input")
    if spec.output_source_id == source.source_id:
        raise ValueError("occupancy output source must differ from camera source")

    artifact = spec.calibration.artifact
    input_contract = _validate_camera_binding(
        capture_input,
        artifact.frame_contract,
    )
    resolved_schema = resolve_occupancy_stream_schema(
        spec,
        source.dataset_schema,
    )
    candidate_contract = OccupancySampleContract(
        resolved_schema.counts_schema.cell_schema,
        resolved_schema.occupied_schema.cell_schema,
        input_contract.metadata_contract,
    )
    candidate_adapter = OccupancyDatasetEventAdapter(candidate_contract)
    output_edge = FrozenDatasetEdge(
        resolved_schema.counts_schema,
        candidate_adapter,
        source.cell_schedule,
    )
    output_contract = output_edge.payload_contract
    if not isinstance(output_contract, OccupancySampleContract):
        raise TypeError("occupancy edge reconstructed another payload contract")
    config = _OccupancyConfig(
        resolved_schema.selected_model,
        input_contract.value_schema,
        output_contract.counts_schema,
        output_contract.occupied_schema,
    )
    processor = BoundStreamProcessor(
        OCCUPANCY_STREAM_PROCESSOR_DEFINITION,
        config,
        input_contract,
        output_contract,
        capture_input.join_key_contract,
        spec.output_stream_id,
        spec.output_source_id,
        _occupancy_operator,
        spec.operator_deadline_seconds,
        spec.terminal_wait_seconds,
        (calibration_artifact_input_ref(spec.calibration.reference),),
    )
    return BoundOccupancyStreamProcessor(
        processor,
        capture_input,
        output_edge,
        spec.calibration.reference,
        resolved_schema.model_kind,
    )


__all__ = [
    "apply_occupancy_snapshot",
    "OCCUPANCY_COUNTS_BLOCK_ID",
    "OCCUPANCY_OCCUPIED_BLOCK_ID",
    "OCCUPANCY_RATE_BLOCK_ID",
    "OCCUPANCY_STREAM_PROCESSOR_DEFINITION",
    "OCCUPANCY_STREAM_PROCESSOR_DEFINITIONS",
    "OCCUPANCY_STREAM_PROCESSOR_KEY",
    "OCCUPANCY_LIVE_OUTPUT_NAMES",
    "OCCUPANCY_EXACT_SOURCE_OUTPUT_NAMES",
    "BoundOccupancyStreamProcessor",
    "OccupancyArtifact",
    "OccupancyDatasetEventAdapter",
    "OccupancyDatasetMetadata",
    "OccupancyMonitorCellContext",
    "OccupancySample",
    "OccupancySampleContract",
    "OccupancyStreamProcessorSpec",
    "ReactiveOccupancyMonitorEvaluation",
    "ResolvedOccupancyStreamSchema",
    "ResolvedOccupancy",
    "bind_occupancy_stream_processor",
    "evaluate_reactive_occupancy_monitor",
    "occupancy_rate_snapshot",
    "resolve_occupancy_stream_schema",
]
