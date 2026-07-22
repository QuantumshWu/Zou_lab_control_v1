"""Typed camera-ROI scalar events for the bounded live monitor path.

The pure reduction owns the physical meaning of one explicit spatial ROI.  The
small runtime projection owns the one fixed processor-ingress monitor tap and
projects a caller-supplied, revisioned binding at a source-shot boundary.  It is
intentionally not a generic live workflow engine: the current product has one
camera input and one optional scalar output branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from zlc_data import (
    INVALID,
    VALID,
    CoordinateRangeSelection,
    ReductionMethod,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
    ValidityPolicy,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
    expand_value_validity,
    resolve_selection_indices,
    selection_to_tree,
)
from zlc_data.numeric import (
    canonical_mean_dtype,
    canonical_sum_dtype,
    checked_numeric_sum,
)
from zlc_storage import canonical_digest, sha256_text

from zlc_neutral_atom.acquisition.camera import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionProducer,
    Envelope,
    EventRef,
    MonitorTap,
    MonitorUpdate,
    TraceContext,
    event_ref_to_tree,
)


@dataclass(frozen=True, init=False)
class RoiScalarBinding:
    """One admitted ROI/reducer and its fixed output contract."""

    input_contract: CameraSampleContract
    selection: Selection
    reduction: ReductionMethod
    validity_policy: ValidityPolicy
    output_schema: ValueSchema = field(init=False, repr=False)
    fingerprint: str = field(init=False)
    _y_range: tuple[int, int] = field(init=False, repr=False, compare=False)
    _x_range: tuple[int, int] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        input_contract: CameraSampleContract,
        selection: Selection,
        reduction: ReductionMethod,
        validity_policy: ValidityPolicy = ValidityPolicy.REQUIRE_ALL,
        output_schema: ValueSchema | None = None,
    ) -> None:
        object.__setattr__(self, "input_contract", input_contract)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "validity_policy", validity_policy)
        self._initialize(output_schema)

    def _initialize(self, admitted_output_schema: ValueSchema | None) -> None:
        if not isinstance(self.input_contract, CameraSampleContract):
            raise TypeError("input_contract must be CameraSampleContract")
        if not isinstance(self.selection, Selection):
            raise TypeError("selection must be zlc_data.Selection")
        if not isinstance(self.reduction, ReductionMethod):
            raise TypeError("reduction must be zlc_data.ReductionMethod")
        if self.reduction not in (
            ReductionMethod.MEAN,
            ReductionMethod.SUM,
            ReductionMethod.MAX,
        ):
            raise ValueError("ROI scalar reduction must be MEAN, SUM, or MAX")
        if not isinstance(self.validity_policy, ValidityPolicy):
            raise TypeError("validity_policy must be zlc_data.ValidityPolicy")
        if self.validity_policy is ValidityPolicy.MIN_COUNT:
            raise ValueError("ROI scalar MIN_COUNT is not implemented")
        schema = self.input_contract.value_schema
        axes = schema.data_axes
        x_axes = tuple(axis for axis in axes if axis.role == SPATIAL_X)
        y_axes = tuple(axis for axis in axes if axis.role == SPATIAL_Y)
        if len(axes) != 2 or len(x_axes) != 1 or len(y_axes) != 1:
            raise ValueError(
                "ROI scalar input requires exactly one SPATIAL_Y and SPATIAL_X axis"
            )
        terms = {term.axis_id: term for term in self.selection.terms}
        if (
            len(terms) != 2
            or set(terms) != {x_axes[0].axis_id, y_axes[0].axis_id}
            or any(not isinstance(term, CoordinateRangeSelection) for term in terms.values())
        ):
            raise ValueError("ROI scalar selection must be one typed spatial rectangle")
        x_indices, x_drop = resolve_selection_indices(x_axes[0], terms[x_axes[0].axis_id])
        y_indices, y_drop = resolve_selection_indices(y_axes[0], terms[y_axes[0].axis_id])
        if x_drop or y_drop or not isinstance(x_indices, range) or not isinstance(y_indices, range):
            raise ValueError("ROI scalar requires contiguous non-dropping spatial ranges")
        if x_indices.step != 1 or y_indices.step != 1:
            raise ValueError("ROI scalar ranges must use unit stride")
        object.__setattr__(self, "_x_range", (x_indices.start, x_indices.stop))
        object.__setattr__(self, "_y_range", (y_indices.start, y_indices.stop))
        expected_output_schema = roi_scalar_output_schema(
            self.input_contract,
            self.reduction,
        )
        output_schema = admitted_output_schema
        if output_schema is None:
            output_schema = expected_output_schema
        elif not isinstance(output_schema, ValueSchema):
            raise TypeError("output_schema must be ValueSchema or None")
        elif output_schema.fingerprint != expected_output_schema.fingerprint:
            raise ValueError("ROI scalar output_schema differs from the canonical reduction schema")
        object.__setattr__(self, "output_schema", output_schema)
        contributors = len(x_indices) * len(y_indices)
        if self.reduction is ReductionMethod.SUM and schema.dtype.kind in "iu":
            source_limits = np.iinfo(schema.dtype)
            output_limits = np.iinfo(output_schema.dtype)
            if (
                source_limits.max * contributors > output_limits.max
                or source_limits.min * contributors < output_limits.min
            ):
                raise OverflowError(
                    "ROI SUM contributor bound exceeds its canonical output dtype"
                )
        object.__setattr__(
            self,
            "fingerprint",
            canonical_digest(
                {
                    "binding": "zlc_neutral_atom.camera-roi-scalar",
                    "input_contract": self.input_contract.fingerprint,
                    "selection": selection_to_tree(self.selection),
                    "reduction": self.reduction.value,
                    "validity_policy": self.validity_policy.value,
                    "output_schema": output_schema.fingerprint,
                }
            ),
        )


def roi_scalar_output_schema(
    input_contract: CameraSampleContract,
    reduction: ReductionMethod,
) -> ValueSchema:
    """Canonical scalar ValueSchema for one camera reduction family."""

    if not isinstance(input_contract, CameraSampleContract):
        raise TypeError("input_contract must be CameraSampleContract")
    if not isinstance(reduction, ReductionMethod):
        raise TypeError("reduction must be zlc_data.ReductionMethod")
    schema = input_contract.value_schema
    if reduction is ReductionMethod.MEAN:
        output_dtype = canonical_mean_dtype(schema.dtype)
    elif reduction is ReductionMethod.SUM:
        output_dtype = canonical_sum_dtype(schema.dtype)
    elif reduction is ReductionMethod.MAX:
        if schema.dtype.kind == "c":
            raise TypeError("ROI MAX is undefined for complex camera values")
        output_dtype = schema.dtype
    else:
        raise ValueError("ROI scalar reduction must be MEAN, SUM, or MAX")
    return ValueSchema(
        (),
        ValidityContract.value(),
        output_dtype,
        schema.value_unit,
    )


def reduce_camera_roi(sample: CameraSample, binding: RoiScalarBinding) -> Value:
    """Reduce one admitted rectangle without flattening or implicit axis choice."""

    if not isinstance(binding, RoiScalarBinding):
        raise TypeError("binding must be RoiScalarBinding")
    binding.input_contract.validate(sample)
    image = sample.image
    schema = image.schema
    y_position = next(
        index for index, axis in enumerate(schema.data_axes) if axis.role == SPATIAL_Y
    )
    x_position = next(
        index for index, axis in enumerate(schema.data_axes) if axis.role == SPATIAL_X
    )
    index = [slice(None)] * len(schema.data_axes)
    index[y_position] = slice(*binding._y_range)
    index[x_position] = slice(*binding._x_range)
    selected = image.values[tuple(index)]
    validity = expand_value_validity(image.validity, schema)[tuple(index)]
    valid_count = int(np.count_nonzero(validity))
    output_valid = (
        valid_count == selected.size
        if binding.validity_policy is ValidityPolicy.REQUIRE_ALL
        else valid_count > 0
    )
    if valid_count == 0:
        return Value(
            np.zeros((), dtype=binding.output_schema.dtype),
            INVALID,
            binding.output_schema,
        )
    axes = tuple(range(selected.ndim))
    if binding.reduction in (ReductionMethod.SUM, ReductionMethod.MEAN):
        safe = (
            selected
            if valid_count == selected.size
            else np.where(validity, selected, 0)
        )
        if safe.dtype.kind in "fc" and not np.all(np.isfinite(safe)):
            raise ValueError("ROI reduction received a valid non-finite camera value")
        if binding.reduction is ReductionMethod.SUM:
            reduced = checked_numeric_sum(
                safe,
                axes,
                output_dtype=binding.output_schema.dtype,
            )
        else:
            widened = safe.astype(binding.output_schema.dtype, copy=False)
            summed = checked_numeric_sum(
                widened,
                axes,
                output_dtype=binding.output_schema.dtype,
            )
            reduced = np.asarray(summed, dtype=binding.output_schema.dtype) / valid_count
    else:
        valid_values = selected if valid_count == selected.size else selected[validity]
        if valid_values.dtype.kind in "fc" and not np.all(np.isfinite(valid_values)):
            raise ValueError("ROI MAX received a valid non-finite camera value")
        reduced = np.max(valid_values)
    return Value(
        np.asarray(reduced, dtype=binding.output_schema.dtype).reshape(()),
        VALID if output_valid else INVALID,
        binding.output_schema,
    )


@dataclass(frozen=True)
class RoiScalarMetadata:
    source_event_ref: EventRef
    source_metadata: CameraFrameMetadata
    binding_fingerprint: str
    control_revision: int
    source_missed: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_event_ref, EventRef):
            raise TypeError("source_event_ref must be EventRef")
        if not isinstance(self.source_metadata, CameraFrameMetadata):
            raise TypeError("source_metadata must be CameraFrameMetadata")
        sha256_text(self.binding_fingerprint, "binding_fingerprint")
        if (
            isinstance(self.control_revision, bool)
            or not isinstance(self.control_revision, Integral)
            or self.control_revision <= 0
        ):
            raise ValueError("control_revision must be a positive integer")
        object.__setattr__(self, "control_revision", int(self.control_revision))
        if (
            isinstance(self.source_missed, bool)
            or not isinstance(self.source_missed, Integral)
            or self.source_missed < 0
        ):
            raise ValueError("source_missed must be a non-negative integer")
        object.__setattr__(
            self,
            "source_missed",
            int(self.source_missed),
        )


@dataclass(frozen=True, eq=False)
class RoiScalarSample:
    value: Value
    metadata: RoiScalarMetadata

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, Value):
            raise TypeError("value must be zlc_data.Value")
        if not isinstance(self.metadata, RoiScalarMetadata):
            raise TypeError("metadata must be RoiScalarMetadata")


@dataclass(frozen=True)
class RoiScalarMetadataContract:
    source_metadata_contract: CameraFrameMetadataContract

    def __post_init__(self) -> None:
        if not isinstance(self.source_metadata_contract, CameraFrameMetadataContract):
            raise TypeError("source_metadata_contract must be CameraFrameMetadataContract")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.RoiScalarRevisionMetadata",
                "source_metadata": self.source_metadata_contract.fingerprint,
            }
        )

    def snapshot(self, payload: RoiScalarSample) -> RoiScalarMetadata:
        if not isinstance(payload, RoiScalarSample):
            raise TypeError("metadata snapshot requires RoiScalarSample")
        metadata = payload.metadata
        self.validate(metadata)
        return metadata

    def validate(self, metadata: object | None) -> None:
        if not isinstance(metadata, RoiScalarMetadata):
            raise TypeError("metadata must be RoiScalarMetadata")
        self.source_metadata_contract.validate(metadata.source_metadata)
        sha256_text(metadata.binding_fingerprint, "binding_fingerprint")
        if metadata.control_revision <= 0:
            raise ValueError("control_revision must be a positive integer")

    def digest(self, metadata: object | None) -> str:
        self.validate(metadata)
        assert isinstance(metadata, RoiScalarMetadata)
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.RoiScalarBindingMetadataContent",
                "source_event_ref": event_ref_to_tree(metadata.source_event_ref),
                "source_metadata": self.source_metadata_contract.digest(
                    metadata.source_metadata
                ),
                "binding_fingerprint": metadata.binding_fingerprint,
                "control_revision": metadata.control_revision,
                "source_missed": metadata.source_missed,
            }
        )


@dataclass(frozen=True)
class RoiScalarSampleContract:
    output_schema: ValueSchema
    metadata_contract: RoiScalarMetadataContract

    def __post_init__(self) -> None:
        if not isinstance(self.output_schema, ValueSchema):
            raise TypeError("output_schema must be ValueSchema")
        if not isinstance(self.metadata_contract, RoiScalarMetadataContract):
            raise TypeError("metadata_contract must be RoiScalarMetadataContract")

    @property
    def value_contract(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.output_schema)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.RoiScalarSample",
                "value": self.value_contract.fingerprint,
                "metadata": self.metadata_contract.fingerprint,
            }
        )

    def snapshot(self, payload: RoiScalarSample) -> RoiScalarSample:
        self.validate(payload)
        return payload

    def validate(self, payload: RoiScalarSample) -> None:
        if not isinstance(payload, RoiScalarSample):
            raise TypeError("payload must be RoiScalarSample")
        self.value_contract.validate(payload.value)
        self.metadata_contract.validate(payload.metadata)

    def digest(self, payload: RoiScalarSample) -> str:
        self.validate(payload)
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.RoiScalarSampleContent",
                "value": self.value_contract.digest(payload.value),
                "metadata": self.metadata_contract.digest(payload.metadata),
            }
        )


@dataclass(frozen=True)
class RoiScalarDatasetEventAdapter:
    payload_contract: RoiScalarSampleContract

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, RoiScalarSampleContract):
            raise TypeError("payload_contract must be RoiScalarSampleContract")

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.output_schema

    @property
    def metadata_contract(self) -> RoiScalarMetadataContract:
        return self.payload_contract.metadata_contract

    @property
    def operator_fingerprint(self) -> str:
        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.processing.RoiScalarDatasetEventAdapter",
                "payload": self.payload_contract.fingerprint,
            }
        )

    @staticmethod
    def value(payload: RoiScalarSample) -> Value:
        return payload.value


class RoiScalarStreamProjection:
    """Stable owner of the fixed processor-ingress camera monitor tap."""

    def __init__(
        self,
        source: MonitorTap[CameraSample],
    ) -> None:
        if not isinstance(source, MonitorTap):
            raise TypeError("source must be MonitorTap")
        self._source = source
        self._closed = False
        source._claim_consumer(self)

    def take_next(self, timeout: float | None = None) -> MonitorUpdate[CameraSample]:
        if self._closed:
            raise RuntimeError("ROI scalar projection is closed")
        return self._source._next_for(self, timeout)

    def project(
        self,
        update: MonitorUpdate[CameraSample],
        binding: RoiScalarBinding,
        control_revision: int,
    ) -> RoiScalarSample:
        if self._closed:
            raise RuntimeError("ROI scalar projection is closed")
        if not isinstance(update, MonitorUpdate):
            raise TypeError("update must be MonitorUpdate")
        if not isinstance(binding, RoiScalarBinding):
            raise TypeError("binding must be RoiScalarBinding")
        source = update.envelope
        value = reduce_camera_roi(source.payload, binding)
        return RoiScalarSample(
            value,
            RoiScalarMetadata(
                source.ref,
                source.payload.metadata,
                binding.fingerprint,
                control_revision,
                update.missed,
            ),
        )

    def publish(
        self,
        update: MonitorUpdate[CameraSample],
        payload: RoiScalarSample,
        output: AcquisitionProducer[RoiScalarSample],
    ) -> Envelope[RoiScalarSample]:
        if self._closed:
            raise RuntimeError("ROI scalar projection is closed")
        if not isinstance(update, MonitorUpdate):
            raise TypeError("update must be MonitorUpdate")
        if not isinstance(payload, RoiScalarSample):
            raise TypeError("payload must be RoiScalarSample")
        if not isinstance(output, AcquisitionProducer):
            raise TypeError("output must be AcquisitionProducer")
        source = update.envelope
        if payload.metadata.source_event_ref != source.ref:
            raise ValueError("ROI scalar payload belongs to another source update")
        return output.emit(
            payload,
            captured_at=source.captured_at,
            trace=TraceContext(
                source.trace.run_id,
                f"{source.trace.source_id}.roi-scalar",
                source.trace.correlation_id,
                (source.ref,),
                control_revision=payload.metadata.control_revision,
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._source.close()


__all__ = [
    "RoiScalarBinding",
    "RoiScalarDatasetEventAdapter",
    "RoiScalarMetadata",
    "RoiScalarMetadataContract",
    "RoiScalarSample",
    "RoiScalarSampleContract",
    "RoiScalarStreamProjection",
    "reduce_camera_roi",
    "roi_scalar_output_schema",
]
