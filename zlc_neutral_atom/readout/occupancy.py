"""One bound camera-frame to atomic occupancy processor.

Binding joins an admitted calibration to one broker-owned camera generation and
checks the complete physical frame contract once.  The hot path is then only a
synchronous ``CameraSample -> OccupancySample`` call through the calibration
domain's single readout operator.  Exact ordering, gap handling, and full
``(repeat, point, site)`` materialization remain owned by the generic stream and
dataset machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from zlc_data import (
    READOUT_EVENT,
    SITE,
    AxisSpec,
    ComponentValidity,
    DatasetSchema,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_neutral_atom.acquisition.camera import (
    CameraDatasetEventAdapter,
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.processing.stream import (
    BoundStreamProcessor,
    ExactStreamProcessorWorker,
    StreamProcessorDefinition,
)
from zlc_neutral_atom.runtime.cancellation import CancellationToken
from zlc_neutral_atom.runtime.capture import CaptureProcessorInputBinding
from zlc_neutral_atom.runtime.dataset import DatasetBuilder, FrozenDatasetEdge
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
)

from .calibration import (
    ReadoutModel,
    ReadoutModelKind,
    ResolvedCalibration,
    _apply_readout_model,
)
from .calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_input_ref,
)
from .contracts import FrameContract


OCCUPANCY_STREAM_PROCESSOR_KEY = DefinitionKey(
    "zlc_neutral_atom.readout",
    "occupancy-stream",
)
_OCCUPANCY_CONFIG_FORMAT = "zlc_neutral_atom.occupancy-stream-config"
_RECORD_BYTES = 128


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

    @property
    def max_retained_nbytes(self) -> int:
        return (
            _RECORD_BYTES
            + self._counts.max_retained_nbytes
            + self._occupied.max_retained_nbytes
            + self.metadata_contract.max_retained_nbytes
        )

    @property
    def finalization_scratch_nbytes(self) -> int:
        return 32 * self.counts_schema.data_axes[0].size

    def snapshot(self, payload: OccupancySample) -> OccupancySample:
        self.validate(payload)
        return payload

    def validate(self, payload: OccupancySample) -> None:
        if not isinstance(payload, OccupancySample):
            raise TypeError("payload must be OccupancySample")
        self._counts.validate(payload.counts)
        self._occupied.validate(payload.occupied)
        self.metadata_contract.validate(payload.metadata)

    def retained_nbytes(self, payload: OccupancySample) -> int:
        self.validate(payload)
        return (
            _RECORD_BYTES
            + self._counts.retained_nbytes(payload.counts)
            + self._occupied.retained_nbytes(payload.occupied)
            + self.metadata_contract.retained_nbytes(payload.metadata)
        )

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

    @property
    def max_retained_nbytes(self) -> int:
        return (
            _RECORD_BYTES
            + self._occupied.max_retained_nbytes
            + self.payload_contract.metadata_contract.max_retained_nbytes
        )

    def snapshot(self, payload: OccupancySample) -> OccupancyDatasetMetadata:
        self.payload_contract.validate(payload)
        return OccupancyDatasetMetadata(payload.occupied, payload.metadata)

    def validate(self, metadata: object | None) -> None:
        if not isinstance(metadata, OccupancyDatasetMetadata):
            raise TypeError("metadata must be OccupancyDatasetMetadata")
        self._occupied.validate(metadata.occupied)
        self.payload_contract.metadata_contract.validate(metadata.source_metadata)

    def retained_nbytes(self, metadata: object | None) -> int:
        self.validate(metadata)
        assert isinstance(metadata, OccupancyDatasetMetadata)
        return (
            _RECORD_BYTES
            + self._occupied.retained_nbytes(metadata.occupied)
            + self.payload_contract.metadata_contract.retained_nbytes(
                metadata.source_metadata
            )
        )

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
        if not isinstance(self.calibration, ResolvedCalibration):
            raise TypeError("calibration must be ResolvedCalibration")
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
        if self.output_edge.expected_cells != (
            self.capture_input.capture_contract.expected_cells
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
    position = event_axes[0][0]
    indices = {
        int(contract.dataset_schema.point_layout.multi_index(cell.point_storage_index)[position])
        for cell in contract.expected_cells
    }
    if len(indices) != 1:
        raise ValueError(
            "occupancy input must contain exactly one physical READOUT_EVENT"
        )
    return next(iter(indices))


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
    model = artifact.select_model(spec.model_kind)
    if model.kind is not spec.model_kind:
        raise ValueError("selected model differs from the frozen processor spec")
    site_axis = artifact.site_map.site_axis
    if model.feature.site_axis != site_axis:
        raise ValueError("selected model and SiteMap use different site axes")
    outer_axis_ids = {
        source.dataset_schema.repeat_axis.axis_id,
        *(axis.axis_id for axis in source.dataset_schema.point_axes),
    }
    if site_axis.axis_id in outer_axis_ids:
        raise ValueError("site AxisId collides with a capture repeat/point AxisId")

    counts_schema, occupied_schema = _output_schemas(
        artifact.frame_contract,
        site_axis,
    )
    candidate_contract = OccupancySampleContract(
        counts_schema,
        occupied_schema,
        input_contract.metadata_contract,
    )
    candidate_adapter = OccupancyDatasetEventAdapter(candidate_contract)
    output_schema = DatasetSchema(
        source.dataset_schema.repeat_axis,
        source.dataset_schema.point_axes,
        source.dataset_schema.point_layout,
        candidate_adapter.value_schema,
    )
    output_edge = FrozenDatasetEdge(
        output_schema,
        candidate_adapter,
        source.expected_cells,
    )
    output_contract = output_edge.payload_contract
    if not isinstance(output_contract, OccupancySampleContract):
        raise TypeError("occupancy edge reconstructed another payload contract")
    config = _OccupancyConfig(
        model,
        input_contract.value_schema,
        output_contract.counts_schema,
        output_contract.occupied_schema,
    )
    definition = StreamProcessorDefinition(
        OCCUPANCY_STREAM_PROCESSOR_KEY,
        "Classify one camera frame into atomic site counts and occupancy",
        _OCCUPANCY_CONFIG_FORMAT,
        input_contract.fingerprint,
        output_contract.fingerprint,
        capture_input.join_key_contract.fingerprint,
        spec.operator_deadline_seconds,
        spec.terminal_wait_seconds,
    )
    processor = BoundStreamProcessor(
        definition,
        config,
        input_contract,
        output_contract,
        capture_input.join_key_contract,
        spec.output_stream_id,
        spec.output_source_id,
        _occupancy_operator,
        (calibration_artifact_input_ref(spec.calibration.reference),),
    )
    return BoundOccupancyStreamProcessor(
        processor,
        capture_input,
        output_edge,
        spec.calibration.reference,
        model.kind,
    )


__all__ = [
    "OCCUPANCY_STREAM_PROCESSOR_KEY",
    "BoundOccupancyStreamProcessor",
    "OccupancyDatasetEventAdapter",
    "OccupancyDatasetMetadata",
    "OccupancySample",
    "OccupancySampleContract",
    "OccupancyStreamProcessorSpec",
    "bind_occupancy_stream_processor",
]
