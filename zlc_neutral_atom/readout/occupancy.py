"""One exact camera-sample to occupancy-sample processor.

The composition boundary resolves a source-verified calibration and a broker-
attested camera generation before this processor is bound.  The synchronous hot
path consequently owns no repository, mutable session lookup, device access, or
shape inference: one :class:`CameraSample` becomes one atomic
:class:`OccupancySample` with the same key, trace, and physical frame metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

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
from zlc_storage import canonical_digest

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
    ProcessorExecutionGuard,
    StreamProcessorDefinition,
)
from zlc_neutral_atom.runtime.cancellation import CancellationToken
from zlc_neutral_atom.runtime.capture import CaptureProcessorInputBinding
from zlc_neutral_atom.runtime.dataset import DatasetBuilder, FrozenDatasetEdge
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    ArtifactInputRef,
    ExactReservation,
    StreamId,
)

from .calibration import (
    DEFAULT_CALIBRATION_RESOURCE_POLICY,
    BoxReadoutModel,
    CalibrationResourcePolicy,
    PerSitePsfReadoutModel,
    ReadoutFeatureSpec,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    UniformPsfReadoutModel,
    bind_readout_feature_spec,
    classify_occupancy,
    extract_readout_features,
    validate_readout_feature_spec_model,
    validate_readout_model_resources,
)
from .calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_input_ref,
)
from .calibration_repository import AdmittedCalibration
from .codec import camera_capture_descriptor_to_tree, readout_binding_key_to_tree
from .contracts import FrameContract, ReadoutBindingKey


OCCUPANCY_STREAM_PROCESSOR_KEY = DefinitionKey(
    "zlc_neutral_atom.readout",
    "occupancy-stream",
    1,
)
_OCCUPANCY_CONFIG_SCHEMA = "zlc_neutral_atom.occupancy-stream-config.v1"
_READOUT_MODELS = (
    BoxReadoutModel,
    PerSitePsfReadoutModel,
    UniformPsfReadoutModel,
)
_RECORD_CONTAINER_BYTES = 128
_BOUND_OCCUPANCY_TOKEN = object()
_OCCUPANCY_EXECUTION_GUARD_TOKEN = object()
_OCCUPANCY_EXECUTION_GUARD_SCHEMA = (
    "zlc_neutral_atom.occupancy-exact-execution-guard.v1"
)


def _canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _optional_canonical_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _canonical_text(value, field_name)


def _sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _positive_seconds(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be finite and positive")
    return float(value)


def _model(value: object) -> ReadoutModel:
    if not isinstance(value, _READOUT_MODELS):
        raise TypeError("model must be a member of the closed ReadoutModel union")
    return value


def _validity_equal(left: ComponentValidity, right: ComponentValidity) -> bool:
    return left.axis_ids == right.axis_ids and np.array_equal(left.mask, right.mask)


@dataclass(frozen=True, eq=False)
class OccupancySample:
    """Atomic readout result for one physical camera frame."""

    occupied: Value
    counts: Value
    metadata: CameraFrameMetadata

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.occupied, Value):
            raise TypeError("occupied must be a zlc_data.Value")
        if not isinstance(self.counts, Value):
            raise TypeError("counts must be a zlc_data.Value")
        if not isinstance(self.metadata, CameraFrameMetadata):
            raise TypeError("metadata must be CameraFrameMetadata")
        if self.occupied.schema.data_axes != self.counts.schema.data_axes:
            raise ValueError("occupied and counts must name the same data axes")
        if not isinstance(self.occupied.validity, ComponentValidity) or not isinstance(
            self.counts.validity,
            ComponentValidity,
        ):
            raise TypeError("occupancy fields require explicit component validity")
        if not _validity_equal(self.occupied.validity, self.counts.validity):
            raise ValueError("occupied and counts must have identical component validity")


@dataclass(frozen=True)
class OccupancySampleContract:
    """Single owner of the atomic result schemas, validation, and byte budget."""

    occupied_schema: ValueSchema
    counts_schema: ValueSchema
    metadata_contract: CameraFrameMetadataContract
    _occupied_contract: ValuePayloadContract = field(init=False, repr=False)
    _counts_contract: ValuePayloadContract = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.occupied_schema, ValueSchema):
            raise TypeError("occupied_schema must be ValueSchema")
        if not isinstance(self.counts_schema, ValueSchema):
            raise TypeError("counts_schema must be ValueSchema")
        if not isinstance(self.metadata_contract, CameraFrameMetadataContract):
            raise TypeError("metadata_contract must be CameraFrameMetadataContract")
        axes = self.occupied_schema.data_axes
        if len(axes) != 1 or axes[0].role != SITE:
            raise ValueError("occupancy schemas require exactly one named site axis")
        if self.counts_schema.data_axes != axes:
            raise ValueError("occupied and counts schemas must share the site axis")
        site_contract = ValidityContract.components(axes[0].axis_id)
        if (
            self.occupied_schema.validity_contract != site_contract
            or self.counts_schema.validity_contract != site_contract
        ):
            raise ValueError("occupancy schemas must declare site component validity")
        if self.occupied_schema.dtype != np.dtype(bool):
            raise ValueError("occupied_schema must use bool values")
        if self.occupied_schema.value_unit is not None:
            raise ValueError("occupied_schema must not declare a physical unit")
        if self.counts_schema.dtype != np.dtype("<f8"):
            raise ValueError("counts_schema must use canonical float64 values")
        occupied_contract = ValuePayloadContract(self.occupied_schema)
        counts_contract = ValuePayloadContract(self.counts_schema)
        object.__setattr__(self, "_occupied_contract", occupied_contract)
        object.__setattr__(self, "_counts_contract", counts_contract)
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(
                {
                    "contract": "zlc_neutral_atom.OccupancySample/v2",
                    "occupied": occupied_contract.fingerprint,
                    "counts": counts_contract.fingerprint,
                    "metadata": self.metadata_contract.fingerprint,
                }
            ),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def max_retained_nbytes(self) -> int:
        return (
            _RECORD_CONTAINER_BYTES
            + self._occupied_contract.max_retained_nbytes
            + self._counts_contract.max_retained_nbytes
            + self.metadata_contract.max_retained_nbytes
        )

    def snapshot(self, payload: OccupancySample) -> OccupancySample:
        self.validate(payload)
        return payload

    def validate(self, payload: OccupancySample) -> None:
        if not isinstance(payload, OccupancySample):
            raise TypeError("payload must be OccupancySample")
        self._occupied_contract.validate(payload.occupied)
        self._counts_contract.validate(payload.counts)
        self.metadata_contract.validate(payload.metadata)
        occupied_validity = payload.occupied.validity
        counts_validity = payload.counts.validity
        if not isinstance(occupied_validity, ComponentValidity) or not isinstance(
            counts_validity,
            ComponentValidity,
        ):
            raise TypeError("occupancy payload requires component validity")
        if not _validity_equal(occupied_validity, counts_validity):
            raise ValueError("occupancy fields have different component validity")
        if not np.all(np.isfinite(payload.counts.values)):
            raise ValueError("occupancy counts must be finite")
        invalid = ~occupied_validity.mask
        if np.any(payload.occupied.values[invalid]):
            raise ValueError("invalid occupied components require canonical False fillers")
        invalid_counts = payload.counts.values[invalid]
        if np.any(invalid_counts != 0.0) or np.any(np.signbit(invalid_counts)):
            raise ValueError("invalid count components require canonical zero fillers")

    def retained_nbytes(self, payload: OccupancySample) -> int:
        self.validate(payload)
        return (
            _RECORD_CONTAINER_BYTES
            + self._occupied_contract.retained_nbytes(payload.occupied)
            + self._counts_contract.retained_nbytes(payload.counts)
            + self.metadata_contract.retained_nbytes(payload.metadata)
        )

    def digest(self, payload: OccupancySample) -> str:
        """Canonical identity of both fields and their physical frame metadata."""

        self.validate(payload)
        return canonical_digest(
            {
                "schema": "zlc_neutral_atom.OccupancySampleContent/v2",
                "occupied": self._occupied_contract.digest(payload.occupied),
                "counts": self._counts_contract.digest(payload.counts),
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


class OccupancyDatasetField(str, Enum):
    OCCUPIED = "OCCUPIED"
    COUNTS = "COUNTS"


def _opposite_field(field_value: OccupancyDatasetField) -> OccupancyDatasetField:
    return (
        OccupancyDatasetField.COUNTS
        if field_value is OccupancyDatasetField.OCCUPIED
        else OccupancyDatasetField.OCCUPIED
    )


@dataclass(frozen=True, eq=False)
class OccupancyDatasetMetadata:
    """The non-primary field plus the original physical frame metadata."""

    companion_field: OccupancyDatasetField
    companion_value: Value
    source_metadata: CameraFrameMetadata

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.companion_field, OccupancyDatasetField):
            raise TypeError("companion_field must be OccupancyDatasetField")
        if not isinstance(self.companion_value, Value):
            raise TypeError("companion_value must be Value")
        if not isinstance(self.source_metadata, CameraFrameMetadata):
            raise TypeError("source_metadata must be CameraFrameMetadata")


@dataclass(frozen=True)
class OccupancyDatasetMetadataContract:
    payload_contract: OccupancySampleContract
    selected_field: OccupancyDatasetField
    _companion_contract: ValuePayloadContract = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, OccupancySampleContract):
            raise TypeError("payload_contract must be OccupancySampleContract")
        if not isinstance(self.selected_field, OccupancyDatasetField):
            raise TypeError("selected_field must be OccupancyDatasetField")
        companion_schema = (
            self.payload_contract.counts_schema
            if self.selected_field is OccupancyDatasetField.OCCUPIED
            else self.payload_contract.occupied_schema
        )
        companion_contract = ValuePayloadContract(companion_schema)
        object.__setattr__(self, "_companion_contract", companion_contract)
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(
                {
                    "contract": "zlc_neutral_atom.OccupancyDatasetMetadata/v1",
                    "payload": self.payload_contract.fingerprint,
                    "selected_field": self.selected_field.value,
                    "companion": companion_contract.fingerprint,
                    "source_metadata": self.payload_contract.metadata_contract.fingerprint,
                }
            ),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def max_retained_nbytes(self) -> int:
        return (
            _RECORD_CONTAINER_BYTES
            + self._companion_contract.max_retained_nbytes
            + self.payload_contract.metadata_contract.max_retained_nbytes
        )

    def snapshot(self, payload: OccupancySample) -> OccupancyDatasetMetadata:
        self.payload_contract.validate(payload)
        companion = (
            payload.counts
            if self.selected_field is OccupancyDatasetField.OCCUPIED
            else payload.occupied
        )
        return OccupancyDatasetMetadata(
            _opposite_field(self.selected_field),
            companion,
            payload.metadata,
        )

    def validate(self, metadata: object | None) -> None:
        if not isinstance(metadata, OccupancyDatasetMetadata):
            raise TypeError("metadata must be OccupancyDatasetMetadata")
        if metadata.companion_field is not _opposite_field(self.selected_field):
            raise ValueError("occupancy dataset metadata names the wrong companion field")
        self._companion_contract.validate(metadata.companion_value)
        self.payload_contract.metadata_contract.validate(metadata.source_metadata)

    def retained_nbytes(self, metadata: object | None) -> int:
        self.validate(metadata)
        assert isinstance(metadata, OccupancyDatasetMetadata)
        return (
            _RECORD_CONTAINER_BYTES
            + self._companion_contract.retained_nbytes(metadata.companion_value)
            + self.payload_contract.metadata_contract.retained_nbytes(
                metadata.source_metadata
            )
        )

    def digest(self, metadata: object | None) -> str:
        self.validate(metadata)
        assert isinstance(metadata, OccupancyDatasetMetadata)
        return canonical_digest(
            {
                "schema": "zlc_neutral_atom.OccupancyDatasetMetadataDigest/v1",
                "companion_field": metadata.companion_field.value,
                "companion_value": self._companion_contract.digest(
                    metadata.companion_value
                ),
                "source_metadata": self.payload_contract.metadata_contract.digest(
                    metadata.source_metadata
                ),
            }
        )


@dataclass(frozen=True)
class OccupancyDatasetEventAdapter:
    """Explicit authoritative projection of one atomic record into a DataBlock."""

    payload_contract: OccupancySampleContract
    selected_field: OccupancyDatasetField = OccupancyDatasetField.OCCUPIED
    metadata_contract: OccupancyDatasetMetadataContract = field(
        init=False,
        repr=False,
    )
    operator_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, OccupancySampleContract):
            raise TypeError("payload_contract must be OccupancySampleContract")
        if not isinstance(self.selected_field, OccupancyDatasetField):
            raise TypeError("selected_field must be OccupancyDatasetField")
        object.__setattr__(
            self,
            "metadata_contract",
            OccupancyDatasetMetadataContract(
                self.payload_contract,
                self.selected_field,
            ),
        )
        object.__setattr__(
            self,
            "operator_fingerprint",
            canonical_digest(
                {
                    "owner": "zlc_neutral_atom.readout.OccupancyDatasetEventAdapter",
                    "schema": "v1",
                    "payload": self.payload_contract.fingerprint,
                    "selected_field": self.selected_field.value,
                }
            ),
        )

    @property
    def value_schema(self) -> ValueSchema:
        return (
            self.payload_contract.occupied_schema
            if self.selected_field is OccupancyDatasetField.OCCUPIED
            else self.payload_contract.counts_schema
        )

    def value(self, payload: OccupancySample) -> Value:
        self.payload_contract.validate(payload)
        return (
            payload.occupied
            if self.selected_field is OccupancyDatasetField.OCCUPIED
            else payload.counts
        )


@dataclass(frozen=True)
class OccupancyModelSelection:
    model_id: str
    model_version: str
    model_kind: ReadoutModelKind

    def __post_init__(self) -> None:
        _canonical_text(self.model_id, "model_id")
        _canonical_text(self.model_version, "model_version")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")


def resolve_occupancy_model_selection(
    calibration: AdmittedCalibration,
    *,
    model_id: str | None = None,
    model_kind: ReadoutModelKind | None = None,
) -> OccupancyModelSelection:
    """Resolve explicit/default composition intent into a frozen exact selection."""

    if type(calibration) is not AdmittedCalibration:
        raise TypeError("calibration must be an exact AdmittedCalibration")
    model_id = _optional_canonical_text(model_id, "model_id")
    if model_kind is not None and not isinstance(model_kind, ReadoutModelKind):
        raise TypeError("model_kind must be ReadoutModelKind or None")
    if model_id is not None and model_kind is not None:
        raise ValueError("select by model_id or model_kind, not both")
    artifact = calibration.artifact
    model = artifact.select_model(model_id=model_id, kind=model_kind)
    return OccupancyModelSelection(
        model.header.model_id,
        model.header.model_version,
        model.kind,
    )


@dataclass(frozen=True)
class OccupancyStreamProcessorBindRequest:
    calibration: AdmittedCalibration
    capture_input: CaptureProcessorInputBinding
    readout_binding: ReadoutBindingKey
    model: OccupancyModelSelection
    output_stream_id: StreamId
    output_source_id: str
    output_field: OccupancyDatasetField = OccupancyDatasetField.OCCUPIED
    resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY
    operator_deadline_seconds: float = 1.0
    terminal_wait_seconds: float = 1.0

    def __post_init__(self) -> None:
        if type(self.calibration) is not AdmittedCalibration:
            raise TypeError("calibration must be an exact AdmittedCalibration")
        if type(self.capture_input) is not CaptureProcessorInputBinding:
            raise TypeError("capture_input must be CaptureProcessorInputBinding")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.model, OccupancyModelSelection):
            raise TypeError("model must be OccupancyModelSelection")
        if not isinstance(self.output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        _canonical_text(self.output_source_id, "output_source_id")
        if not isinstance(self.output_field, OccupancyDatasetField):
            raise TypeError("output_field must be OccupancyDatasetField")
        if not isinstance(self.resource_policy, CalibrationResourcePolicy):
            raise TypeError("resource_policy must be CalibrationResourcePolicy")
        object.__setattr__(
            self,
            "operator_deadline_seconds",
            _positive_seconds(
                self.operator_deadline_seconds,
                "operator_deadline_seconds",
            ),
        )
        object.__setattr__(
            self,
            "terminal_wait_seconds",
            _positive_seconds(
                self.terminal_wait_seconds,
                "terminal_wait_seconds",
            ),
        )


@dataclass(frozen=True, eq=False)
class _OccupancyStreamProcessorConfig:
    calibration_ref: CalibrationArtifactRef
    calibration_artifact_fingerprint: str
    calibration_admission_evidence_digest: str
    calibration_commit_id: str
    capture_input_contract_digest: str
    frame_contract: FrameContract
    site_map: SiteMap
    readout_binding: ReadoutBindingKey
    selection: OccupancyModelSelection
    input_schema: ValueSchema
    occupied_schema: ValueSchema
    counts_schema: ValueSchema
    feature_spec: ReadoutFeatureSpec
    model: ReadoutModel
    resource_policy: CalibrationResourcePolicy

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        for name in (
            "calibration_artifact_fingerprint",
            "calibration_admission_evidence_digest",
            "capture_input_contract_digest",
        ):
            _sha256(getattr(self, name), name)
        _canonical_text(self.calibration_commit_id, "calibration_commit_id")
        if not isinstance(self.frame_contract, FrameContract):
            raise TypeError("frame_contract must be FrameContract")
        if not isinstance(self.site_map, SiteMap):
            raise TypeError("site_map must be SiteMap")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.selection, OccupancyModelSelection):
            raise TypeError("selection must be OccupancyModelSelection")
        if not isinstance(self.input_schema, ValueSchema):
            raise TypeError("input_schema must be ValueSchema")
        if not isinstance(self.occupied_schema, ValueSchema):
            raise TypeError("occupied_schema must be ValueSchema")
        if not isinstance(self.counts_schema, ValueSchema):
            raise TypeError("counts_schema must be ValueSchema")
        if not isinstance(self.feature_spec, ReadoutFeatureSpec):
            raise TypeError("feature_spec must be ReadoutFeatureSpec")
        model = _model(self.model)
        if not isinstance(self.resource_policy, CalibrationResourcePolicy):
            raise TypeError("resource_policy must be CalibrationResourcePolicy")
        if (
            model.header.model_id != self.selection.model_id
            or model.header.model_version != self.selection.model_version
            or model.kind is not self.selection.model_kind
        ):
            raise ValueError("frozen model differs from its explicit selection")
        if model.header.frame_contract_fingerprint != self.frame_contract.fingerprint:
            raise ValueError("selected model belongs to another FrameContract")
        if model.header.site_map_fingerprint != self.site_map.fingerprint:
            raise ValueError("selected model belongs to another SiteMap")
        if self.frame_contract.binding != self.readout_binding:
            raise ValueError("processor FrameContract belongs to another readout binding")
        if self.input_schema.fingerprint != self.frame_contract.frame_schema.fingerprint:
            raise ValueError("processor input schema differs from its FrameContract")
        _validate_occupancy_schemas(self.occupied_schema, self.counts_schema)
        site_axis = self.occupied_schema.data_axes[0]
        if (
            site_axis.axis_id != model.header.site_axis_id
            or site_axis.size != model.header.site_count
            or self.feature_spec.site_axis_id != site_axis.axis_id
            or site_axis != self.site_map.site_axis
        ):
            raise ValueError("processor site schema differs from selected model")
        if self.counts_schema.value_unit != self.frame_contract.count_unit:
            raise ValueError("counts schema unit differs from its FrameContract")
        validate_readout_feature_spec_model(self.feature_spec, model)


def _validate_occupancy_schemas(
    occupied_schema: ValueSchema,
    counts_schema: ValueSchema,
) -> None:
    # Reuse the payload owner's complete schema validation without inventing a
    # second set of axis/dtype/validity rules.  Metadata bounds are orthogonal.
    OccupancySampleContract(
        occupied_schema,
        counts_schema,
        CameraFrameMetadataContract(),
    )


def _occupancy_schemas(
    frame_contract: FrameContract,
    site_axis: AxisSpec,
) -> tuple[ValueSchema, ValueSchema]:
    counts = ValueSchema(
        (site_axis,),
        ValidityContract.components(site_axis.axis_id),
        np.dtype("<f8"),
        frame_contract.count_unit,
    )
    occupied = ValueSchema(
        (site_axis,),
        ValidityContract.components(site_axis.axis_id),
        np.dtype(bool),
        None,
    )
    return occupied, counts


class _OccupancyExecutionGuard(ProcessorExecutionGuard):
    """Process-local authority over both source reservation and terminal edge."""

    __slots__ = (
        "_authority",
        "_capture_input",
        "_output_edge",
        "_binding_fingerprint",
        "_bound_processor",
        "_processor_fingerprint",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("_OccupancyExecutionGuard is final")

    def __init__(
        self,
        authority: object,
        *,
        capture_input: CaptureProcessorInputBinding,
        output_edge: FrozenDatasetEdge[OccupancySample],
    ) -> None:
        if authority is not _OCCUPANCY_EXECUTION_GUARD_TOKEN:
            raise PermissionError("occupancy execution guards are binder-owned")
        if type(capture_input) is not CaptureProcessorInputBinding:
            raise TypeError("capture_input must be CaptureProcessorInputBinding")
        if not isinstance(output_edge, FrozenDatasetEdge):
            raise TypeError("output_edge must be FrozenDatasetEdge")
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_capture_input", capture_input)
        object.__setattr__(self, "_output_edge", output_edge)
        object.__setattr__(
            self,
            "_binding_fingerprint",
            self._derive_binding_fingerprint(),
        )
        object.__setattr__(self, "_bound_processor", None)
        object.__setattr__(self, "_processor_fingerprint", None)
        self._validate_guard_identity()

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("_OccupancyExecutionGuard is immutable")

    def __reduce__(self):
        raise TypeError("_OccupancyExecutionGuard is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("_OccupancyExecutionGuard is process-local")

    @property
    def schema_id(self) -> str:
        self._validate_guard_identity()
        return _OCCUPANCY_EXECUTION_GUARD_SCHEMA

    @property
    def binding_fingerprint(self) -> str:
        self._validate_guard_identity()
        return self._binding_fingerprint

    def _derive_binding_fingerprint(self) -> str:
        capture_input = object.__getattribute__(self, "_capture_input")
        output_edge = object.__getattribute__(self, "_output_edge")
        stream = capture_input.stream
        return canonical_digest(
            {
                "schema": _OCCUPANCY_EXECUTION_GUARD_SCHEMA,
                "capture_input": _capture_input_contract_digest(capture_input),
                "stream_id": stream.stream_id.value,
                "stream_generation": stream.generation.value,
                "input_consumer_contract": (
                    capture_input.input_edge.consumer_contract_digest
                ),
                "input_schedule": capture_input.input_edge.schedule_digest,
                "output_consumer_contract": output_edge.consumer_contract_digest,
                "output_schedule": output_edge.schedule_digest,
            }
        )

    def _validate_guard_identity(self) -> None:
        try:
            authority = object.__getattribute__(self, "_authority")
            capture_input = object.__getattribute__(self, "_capture_input")
            output_edge = object.__getattribute__(self, "_output_edge")
            binding_fingerprint = object.__getattribute__(
                self,
                "_binding_fingerprint",
            )
        except AttributeError:
            raise PermissionError("occupancy execution guard authority is absent") from None
        if (
            type(self) is not _OccupancyExecutionGuard
            or authority is not _OCCUPANCY_EXECUTION_GUARD_TOKEN
        ):
            raise PermissionError("occupancy execution guard authority is invalid")
        input_edge = capture_input.input_edge
        if output_edge.expected_cells != input_edge.expected_cells:
            raise ValueError("occupancy execution guard schedules differ")
        if output_edge.key_contract_fingerprint != input_edge.key_contract_fingerprint:
            raise ValueError("occupancy execution guard key domains differ")
        if binding_fingerprint != self._derive_binding_fingerprint():
            raise PermissionError("occupancy execution guard binding changed")

    def _bind_processor(
        self,
        authority: object,
        processor: BoundStreamProcessor,
    ) -> None:
        """Freeze the one semantic processor this guard may ever authorize."""

        if authority is not _OCCUPANCY_EXECUTION_GUARD_TOKEN:
            raise PermissionError("occupancy processor binding is binder-owned")
        if not isinstance(processor, BoundStreamProcessor):
            raise TypeError("processor must be BoundStreamProcessor")
        self._validate_guard_identity()
        if self._bound_processor is not None:
            raise RuntimeError("occupancy execution guard already has a processor")
        if processor.execution_guard is not self:
            raise PermissionError("processor does not own this occupancy guard")
        fingerprint = processor.fingerprint
        object.__setattr__(self, "_bound_processor", processor)
        object.__setattr__(self, "_processor_fingerprint", fingerprint)
        self._validate_bound_processor(processor)

    def _validate_bound_processor(self, processor: BoundStreamProcessor) -> None:
        self._validate_guard_identity()
        bound_processor = object.__getattribute__(self, "_bound_processor")
        fingerprint = object.__getattribute__(self, "_processor_fingerprint")
        if bound_processor is None or fingerprint is None:
            raise PermissionError("occupancy execution guard has no bound processor")
        if processor is not bound_processor:
            raise PermissionError(
                "processor is not the one bound by the occupancy binder"
            )
        if processor.execution_guard is not self:
            raise PermissionError("bound processor lost its occupancy guard")
        if processor.fingerprint != fingerprint:
            raise PermissionError("bound occupancy processor semantics changed")

    def validate_bundle_binding(
        self,
        capture_input: CaptureProcessorInputBinding,
        output_edge: FrozenDatasetEdge[OccupancySample],
    ) -> None:
        self._validate_bound_processor(self._bound_processor)
        if capture_input is not self._capture_input or output_edge is not self._output_edge:
            raise PermissionError("occupancy bundle differs from its execution guard")

    def authorize_exact_worker(
        self,
        *,
        bound: BoundStreamProcessor,
        input_reservation: object,
        input_cursor: object,
        input_edge: object,
        output_producer: object,
        deadline_monotonic: float,
        output_cursor: object | None,
        output_builder: object | None,
        downstream_readiness: object | None,
        cancellation: object | None,
    ) -> None:
        del output_producer, deadline_monotonic, cancellation
        self._validate_bound_processor(bound)
        if bound.execution_guard is not self:
            raise PermissionError("processor lost its occupancy execution guard")
        if not isinstance(input_reservation, ExactReservation):
            raise TypeError("input_reservation must be ExactReservation")
        self._capture_input.require_reservation(input_reservation)
        if input_cursor is not input_reservation._cursor:
            raise PermissionError("input cursor belongs to another reservation")
        if input_edge is not self._capture_input.input_edge:
            raise PermissionError("input edge is not the capture session edge")
        if downstream_readiness is not None:
            raise ValueError(
                "baseline occupancy execution terminates at its frozen dataset edge"
            )
        if output_cursor is None or not isinstance(output_builder, DatasetBuilder):
            raise TypeError("occupancy execution requires a terminal DatasetBuilder")
        if output_builder.edge is not self._output_edge:
            raise PermissionError(
                "output builder is not the occupancy bundle's frozen edge"
            )


class BoundOccupancyStreamProcessor:
    """Sealed authority joining one capture generation to one calibration.

    The generic :class:`BoundStreamProcessor` is deliberately private.  Exact
    execution can therefore only be created through :meth:`create_exact_worker`,
    which revalidates that the reservation belongs to the physically attested
    ``CaptureSession`` stream rather than an equivalent caller-created stream.
    """

    __slots__ = (
        "_authority",
        "_processor",
        "_capture_input",
        "_output_edge",
        "_selected_field",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("BoundOccupancyStreamProcessor is final and cannot be subclassed")

    def __init__(
        self,
        authority: object,
        *,
        processor: BoundStreamProcessor,
        capture_input: CaptureProcessorInputBinding,
        output_edge: FrozenDatasetEdge[OccupancySample],
        selected_field: OccupancyDatasetField,
    ) -> None:
        if authority is not _BOUND_OCCUPANCY_TOKEN:
            raise PermissionError(
                "BoundOccupancyStreamProcessor can only be minted by its binder"
            )
        if not isinstance(processor, BoundStreamProcessor):
            raise TypeError("processor must be BoundStreamProcessor")
        if type(capture_input) is not CaptureProcessorInputBinding:
            raise TypeError("capture_input must be CaptureProcessorInputBinding")
        if not isinstance(output_edge, FrozenDatasetEdge):
            raise TypeError("output_edge must be FrozenDatasetEdge")
        if not isinstance(selected_field, OccupancyDatasetField):
            raise TypeError("selected_field must be OccupancyDatasetField")
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_processor", processor)
        object.__setattr__(self, "_capture_input", capture_input)
        object.__setattr__(self, "_output_edge", output_edge)
        object.__setattr__(self, "_selected_field", selected_field)
        self._validate_identity()

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("BoundOccupancyStreamProcessor is immutable")

    def __reduce__(self):
        raise TypeError("BoundOccupancyStreamProcessor is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("BoundOccupancyStreamProcessor is process-local")

    def _require_authority(self) -> None:
        try:
            authority = object.__getattribute__(self, "_authority")
        except AttributeError:
            raise PermissionError(
                "BoundOccupancyStreamProcessor was not minted by its binder"
            ) from None
        if (
            type(self) is not BoundOccupancyStreamProcessor
            or authority is not _BOUND_OCCUPANCY_TOKEN
        ):
            raise PermissionError(
                "BoundOccupancyStreamProcessor was not minted by its binder"
            )

    def _validate_identity(self) -> None:
        self._require_authority()
        processor = object.__getattribute__(self, "_processor")
        capture_input = object.__getattribute__(self, "_capture_input")
        output_edge = object.__getattribute__(self, "_output_edge")
        selected_field = object.__getattribute__(self, "_selected_field")
        # Reading these properties revalidates the CaptureSession-minted owner
        # graph before every authority-sensitive operation.
        input_contract = capture_input.payload_contract
        join_key_contract = capture_input.join_key_contract
        input_edge = capture_input.input_edge
        adapter = output_edge.event_adapter
        if not isinstance(adapter, OccupancyDatasetEventAdapter):
            raise TypeError("output_edge must use OccupancyDatasetEventAdapter")
        if adapter.selected_field is not selected_field:
            raise ValueError("output edge and bundle select different fields")
        if processor.input_payload_contract is not input_contract:
            raise ValueError("processor and capture input must share PayloadContract owner")
        if processor.join_key_contract is not join_key_contract:
            raise ValueError("processor and capture input must share join-key owner")
        if input_edge.payload_contract is not input_contract:
            raise ValueError("capture input edge lost its PayloadContract owner")
        if output_edge.payload_contract is not processor.output_payload_contract:
            raise ValueError("processor and output edge must share PayloadContract owner")
        if (
            output_edge.key_contract_fingerprint
            != processor.join_key_contract.fingerprint
        ):
            raise ValueError("processor and output edge name different key domains")
        if output_edge.expected_cells is None:
            raise ValueError("occupancy output edge requires a frozen exact schedule")
        if output_edge.expected_cells != capture_input.capture_contract.expected_cells:
            raise ValueError("occupancy output schedule differs from capture schedule")
        guard = processor.execution_guard
        if type(guard) is not _OccupancyExecutionGuard:
            raise PermissionError("occupancy processor lost its execution guard")
        guard.validate_bundle_binding(capture_input, output_edge)
        config = processor.config
        if not isinstance(config, _OccupancyStreamProcessorConfig):
            raise TypeError("processor config must be the bound occupancy config")
        if config.capture_input_contract_digest != _capture_input_contract_digest(
            capture_input
        ):
            raise ValueError("processor config belongs to another capture generation")

    @property
    def definition(self) -> StreamProcessorDefinition:
        self._validate_identity()
        return self._processor.definition

    @property
    def fingerprint(self) -> str:
        self._validate_identity()
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.BoundOccupancyStreamProcessor/v1",
                "processor": self._processor.fingerprint,
                "output_consumer_contract": (
                    self._output_edge.consumer_contract_digest
                ),
                "output_schedule": self._output_edge.schedule_digest,
                "selected_field": self._selected_field.value,
            }
        )

    @property
    def artifact_inputs(self) -> tuple[ArtifactInputRef, ...]:
        self._validate_identity()
        return self._processor.artifact_inputs

    @property
    def output_stream_id(self) -> StreamId:
        self._validate_identity()
        return self._processor.output_stream_id

    @property
    def output_source_id(self) -> str:
        self._validate_identity()
        return self._processor.output_source_id

    @property
    def model_selection(self) -> OccupancyModelSelection:
        self._validate_identity()
        config = self._processor.config
        assert isinstance(config, _OccupancyStreamProcessorConfig)
        return config.selection

    @property
    def calibration_reference(self) -> CalibrationArtifactRef:
        self._validate_identity()
        config = self._processor.config
        assert isinstance(config, _OccupancyStreamProcessorConfig)
        return config.calibration_ref

    @property
    def calibration_admission_evidence_digest(self) -> str:
        self._validate_identity()
        config = self._processor.config
        assert isinstance(config, _OccupancyStreamProcessorConfig)
        return config.calibration_admission_evidence_digest

    @property
    def output_edge(self) -> FrozenDatasetEdge[OccupancySample]:
        self._validate_identity()
        return self._output_edge

    @property
    def output_schema(self) -> DatasetSchema:
        self._validate_identity()
        return self._output_edge.schema

    @property
    def output_adapter(self) -> OccupancyDatasetEventAdapter:
        adapter = self.output_edge.event_adapter
        assert isinstance(adapter, OccupancyDatasetEventAdapter)
        return adapter

    @property
    def output_payload_contract(self) -> OccupancySampleContract:
        contract = self.output_edge.payload_contract
        assert isinstance(contract, OccupancySampleContract)
        return contract

    def evaluate(self, sample: CameraSample) -> OccupancySample:
        """Evaluate the frozen pure operator without minting runtime lineage."""

        self._validate_identity()
        result = _occupancy_stream_operator(sample, self._processor.config)
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
        """Create one worker from the exact CaptureSession reservation only."""

        self._validate_identity()
        self._capture_input.require_reservation(input_reservation)
        return ExactStreamProcessorWorker(
            self._processor,
            input_reservation,
            input_cursor,
            input_edge=self._capture_input.input_edge,
            output_producer=output_producer,
            deadline_monotonic=deadline_monotonic,
            output_cursor=output_cursor,
            output_builder=output_builder,
            cancellation=cancellation,
        )


def _actual_readout_event_indices(capture_input: CaptureProcessorInputBinding) -> tuple[int, ...]:
    contract = capture_input.capture_contract
    schema = contract.dataset_schema
    provenance = contract.camera_provenance
    if provenance is None:
        raise ValueError("occupancy requires broker-attested camera provenance")
    descriptor = provenance.descriptor
    axes = tuple(
        (position, axis)
        for position, axis in enumerate(schema.point_axes)
        if axis.role == READOUT_EVENT
    )
    if descriptor.readout_event_axis_id is None:
        if axes:
            raise ValueError("camera descriptor and source schema disagree on READOUT_EVENT")
        return (0,)
    if len(axes) != 1 or axes[0][1].axis_id != descriptor.readout_event_axis_id:
        raise ValueError("camera descriptor and source schema name different READOUT_EVENT axes")
    position = axes[0][0]
    storage_rows = {
        address.point_storage_index for address in contract.expected_cells
    }
    indices = {
        int(schema.point_layout.multi_index(storage_row)[position])
        for storage_row in storage_rows
    }
    if not indices:
        raise ValueError("camera source has no physical readout-event cells")
    return tuple(sorted(indices))


def _validate_source_frame_contract(
    capture_input: CaptureProcessorInputBinding,
    calibration_frame_contract: FrameContract,
    readout_binding: ReadoutBindingKey,
) -> None:
    contract = capture_input.capture_contract
    if type(contract.event_adapter) is not CameraDatasetEventAdapter:
        raise TypeError("occupancy source must use the camera owner event adapter")
    if type(capture_input.payload_contract) is not CameraSampleContract:
        raise TypeError("occupancy source must publish CameraSample")
    provenance = contract.camera_provenance
    if provenance is None:
        raise ValueError("occupancy source lacks broker-attested camera provenance")
    if provenance.binding != readout_binding:
        raise ValueError("capture provenance belongs to another readout binding")
    if calibration_frame_contract.binding != readout_binding:
        raise ValueError("calibration belongs to another readout binding")
    event_indices = _actual_readout_event_indices(capture_input)
    if len(event_indices) != 1:
        raise ValueError(
            "occupancy input must contain exactly one physical READOUT_EVENT; "
            "mixed reference/readout streams require an explicit upstream contract"
        )
    for event_index in event_indices:
        try:
            calibration_frame_contract.assert_compatible(
                readout_binding,
                provenance.descriptor,
                contract.dataset_schema,
                readout_event_index=event_index,
            )
        except ValueError as error:
            raise ValueError(
                "calibration FrameContract does not apply to physical "
                f"READOUT_EVENT {event_index}: {error}"
            ) from error


def _capture_input_contract_digest(
    capture_input: CaptureProcessorInputBinding,
) -> str:
    contract = capture_input.capture_contract
    provenance = contract.camera_provenance
    if provenance is None:
        raise ValueError("camera provenance is required")
    return canonical_digest(
        {
            "schema": "zlc_neutral_atom.OccupancyCaptureInput/v1",
            "stream_id": contract.stream_id.value,
            "source_id": contract.source_id,
            "dataset_schema": contract.dataset_schema.fingerprint,
            "payload_contract": capture_input.payload_contract.fingerprint,
            "dataset_consumer_contract": contract.dataset_edge.consumer_contract_digest,
            "dataset_schedule": contract.dataset_edge.schedule_digest,
            "camera_descriptor": camera_capture_descriptor_to_tree(
                provenance.descriptor
            ),
            "readout_binding": readout_binding_key_to_tree(provenance.binding),
            "active_settings": provenance.active_settings_fingerprint,
            "camera_arm_spec": provenance.camera_arm_spec_fingerprint,
            "binding_id": provenance.binding_id,
            "connection_generation": provenance.connection_generation,
            "capability": provenance.capability_fingerprint,
        }
    )


def _occupancy_stream_operator(payload: object, config: object) -> object:
    """Reviewed top-level hot-path operator admitted by BoundStreamProcessor."""

    if not isinstance(payload, CameraSample):
        raise TypeError("occupancy processor requires one CameraSample")
    if not isinstance(config, _OccupancyStreamProcessorConfig):
        raise TypeError("config must be the bound occupancy config")
    if payload.image.schema is not config.input_schema:
        raise TypeError("camera image must share the bound input ValueSchema owner")
    signals = extract_readout_features(config.feature_spec, payload.image)
    decision = classify_occupancy(config.model, signals)
    validity = decision.validity
    return OccupancySample(
        Value(decision.occupied, validity, config.occupied_schema),
        Value(signals.values, validity, config.counts_schema),
        payload.metadata,
    )


def bind_occupancy_stream_processor(
    request: OccupancyStreamProcessorBindRequest,
) -> BoundOccupancyStreamProcessor:
    """Bind one admitted calibration to one physically uniform camera generation."""

    if not isinstance(request, OccupancyStreamProcessorBindRequest):
        raise TypeError("request must be OccupancyStreamProcessorBindRequest")
    calibration = request.calibration
    artifact = calibration.artifact
    capture_input = request.capture_input
    contract = capture_input.capture_contract
    if request.output_stream_id == contract.stream_id:
        raise ValueError("occupancy output stream must differ from camera input stream")
    if request.output_source_id == contract.source_id:
        raise ValueError("occupancy output source must differ from camera source")
    _validate_source_frame_contract(
        capture_input,
        artifact.frame_contract,
        request.readout_binding,
    )
    model = _model(artifact.select_model(model_id=request.model.model_id))
    if (
        model.header.model_version != request.model.model_version
        or model.kind is not request.model.model_kind
    ):
        raise ValueError("selected model version/kind differs from the frozen request")
    validate_readout_model_resources(
        model,
        image_shape_yx=artifact.frame_contract.frame_schema.data_shape,
        resource_policy=request.resource_policy,
    )
    feature_spec = bind_readout_feature_spec(model, artifact.site_map)
    input_contract = capture_input.payload_contract
    assert isinstance(input_contract, CameraSampleContract)
    if input_contract.value_schema.fingerprint != artifact.frame_contract.frame_schema.fingerprint:
        raise ValueError("camera payload schema differs from calibration FrameContract")
    occupied_schema, counts_schema = _occupancy_schemas(
        artifact.frame_contract,
        artifact.site_map.site_axis,
    )
    outer_axis_ids = {
        contract.dataset_schema.repeat_axis.axis_id,
        *(axis.axis_id for axis in contract.dataset_schema.point_axes),
    }
    if artifact.site_map.site_axis.axis_id in outer_axis_ids:
        raise ValueError(
            "calibration site AxisId collides with a capture repeat/point AxisId"
        )
    candidate_output_contract = OccupancySampleContract(
        occupied_schema,
        counts_schema,
        input_contract.metadata_contract,
    )
    candidate_adapter = OccupancyDatasetEventAdapter(
        candidate_output_contract,
        request.output_field,
    )
    output_schema = DatasetSchema(
        contract.dataset_schema.repeat_axis,
        contract.dataset_schema.point_axes,
        contract.dataset_schema.point_layout,
        candidate_adapter.value_schema,
    )
    output_edge = FrozenDatasetEdge(
        output_schema,
        candidate_adapter,
        contract.expected_cells,
    )
    # FrozenDatasetEdge reconstructs all immutable owners around the one
    # DatasetSchema cell-schema identity.  Bind the processor to that exact
    # reconstructed payload owner; an equivalent caller-built contract is not
    # sufficient for the worker's generation identity checks.
    output_contract = output_edge.payload_contract
    if not isinstance(output_contract, OccupancySampleContract):
        raise TypeError("occupancy output edge reconstructed another payload type")
    config = _OccupancyStreamProcessorConfig(
        calibration.reference,
        calibration.artifact_fingerprint,
        calibration.evidence_digest,
        calibration.commit_id,
        _capture_input_contract_digest(capture_input),
        artifact.frame_contract,
        artifact.site_map,
        request.readout_binding,
        request.model,
        input_contract.value_schema,
        output_contract.occupied_schema,
        output_contract.counts_schema,
        feature_spec,
        model,
        request.resource_policy,
    )
    definition = StreamProcessorDefinition(
        OCCUPANCY_STREAM_PROCESSOR_KEY,
        "Classify one camera frame into atomic site counts and occupancy",
        _OCCUPANCY_CONFIG_SCHEMA,
        input_contract.fingerprint,
        output_contract.fingerprint,
        capture_input.join_key_contract.fingerprint,
        operator_deadline_seconds=request.operator_deadline_seconds,
        terminal_wait_seconds=request.terminal_wait_seconds,
        execution_guard_schema_id=_OCCUPANCY_EXECUTION_GUARD_SCHEMA,
    )
    execution_guard = _OccupancyExecutionGuard(
        _OCCUPANCY_EXECUTION_GUARD_TOKEN,
        capture_input=capture_input,
        output_edge=output_edge,
    )
    processor = BoundStreamProcessor(
        definition,
        config,
        input_contract,
        output_contract,
        capture_input.join_key_contract,
        request.output_stream_id,
        request.output_source_id,
        _occupancy_stream_operator,
        (calibration_artifact_input_ref(calibration.reference),),
        execution_guard,
    )
    execution_guard._bind_processor(
        _OCCUPANCY_EXECUTION_GUARD_TOKEN,
        processor,
    )
    return BoundOccupancyStreamProcessor(
        _BOUND_OCCUPANCY_TOKEN,
        processor=processor,
        capture_input=capture_input,
        output_edge=output_edge,
        selected_field=request.output_field,
    )


__all__ = [
    "OCCUPANCY_STREAM_PROCESSOR_KEY",
    "BoundOccupancyStreamProcessor",
    "OccupancyDatasetEventAdapter",
    "OccupancyDatasetField",
    "OccupancyDatasetMetadata",
    "OccupancyDatasetMetadataContract",
    "OccupancyModelSelection",
    "OccupancySample",
    "OccupancySampleContract",
    "OccupancyStreamProcessorBindRequest",
    "bind_occupancy_stream_processor",
    "resolve_occupancy_model_selection",
]
