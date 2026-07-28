"""Immutable, renderer-free figure and view values.

This module deliberately contains no Matplotlib, Qt, runtime reference, or
authoritative transform type.  A :class:`ViewSpec` says only how a named
dataset is presented; it is not an analysis program and cannot be committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Any

import numpy as np

from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSourceRef,
    CoordinateFrameId,
    DatasetRevisionRef,
    Selection,
    immutable_array,
)
from zlc_storage.canonical import (
    canonical_text as _text,
    nonnegative_integer as _nonnegative,
    sha256_text as _digest,
)


_POINT_ORDINAL_INTERACTION_AXIS_ID = AxisId(
    "zlc_frontend.point-ordinal-interaction"
)


def _immutable_evaluated_array(
    value: object,
    *,
    ndim: int | None = None,
    boolean: bool = False,
) -> np.ndarray:
    """Freeze one evaluator DTO array on the shared immutable-bytes primitive."""

    source = np.asarray(value)
    if ndim is not None and source.ndim != ndim:
        raise ValueError(f"array must have rank {ndim}, got {source.ndim}")
    if source.dtype.kind not in "biufc":
        raise TypeError("evaluated arrays must have a numeric or boolean dtype")
    if boolean and source.dtype != np.dtype(bool):
        raise TypeError(f"evaluated validity dtype must be bool, got {source.dtype}")
    return immutable_array(
        source,
        dtype=source.dtype.newbyteorder("<"),
        shape=source.shape,
    )


class ViewIntent(str, Enum):
    IMAGE = "IMAGE"
    CURVE = "CURVE"
    HISTOGRAM = "HISTOGRAM"
    METER = "METER"
    PULSE = "PULSE"


DATASET_VIEW_INTENTS = frozenset(
    {
        ViewIntent.IMAGE,
        ViewIntent.CURVE,
        ViewIntent.HISTOGRAM,
        ViewIntent.METER,
    }
)


class AxisViewRole(str, Enum):
    X = "X"
    IMAGE_X = "IMAGE_X"
    IMAGE_Y = "IMAGE_Y"
    SAMPLE = "SAMPLE"
    BATCH = "BATCH"
    FACET = "FACET"
    SELECTED = "SELECTED"
    REDUCED = "REDUCED"


class DisplayReductionMethod(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"


@dataclass(frozen=True)
class FixedIndex:
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _nonnegative(self.index, "selector index"))


@dataclass(frozen=True)
class LatestNonempty:
    """Resolve the greatest allowed non-empty logical index, not publication time."""


DisplaySelector = FixedIndex | LatestNonempty


@dataclass(frozen=True)
class DisplayReduction:
    method: DisplayReductionMethod

    def __post_init__(self) -> None:
        if not isinstance(self.method, DisplayReductionMethod):
            raise TypeError("display reduction method must be DisplayReductionMethod")


@dataclass(frozen=True)
class SourceViewBinding:
    source: AxisSourceRef
    role: AxisViewRole
    selector: DisplaySelector | None = None
    reduction: DisplayReduction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, AxisSourceRef):
            raise TypeError("source must be AxisSourceRef")
        if not isinstance(self.role, AxisViewRole):
            raise TypeError("role must be AxisViewRole")
        if self.role is AxisViewRole.SELECTED:
            if not isinstance(self.selector, (FixedIndex, LatestNonempty)):
                raise ValueError("SELECTED binding requires a display selector")
            if self.reduction is not None:
                raise ValueError("selected binding cannot carry a reduction")
        elif self.role is AxisViewRole.REDUCED:
            if not isinstance(self.reduction, DisplayReduction):
                raise ValueError("REDUCED binding requires a display reduction")
            if self.selector is not None:
                raise ValueError("reduced binding cannot carry a selector")
        elif self.selector is not None or self.reduction is not None:
            raise ValueError(f"{self.role.value} binding cannot carry an operation")


@dataclass(frozen=True)
class ViewSpec:
    """The sole persistent presentation transform for one dataset layer."""

    schema_fingerprint: str
    intent: ViewIntent
    source_bindings: tuple[SourceViewBinding, ...]
    point_ordinals: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_fingerprint",
            _digest(self.schema_fingerprint, "schema_fingerprint"),
        )
        if not isinstance(self.intent, ViewIntent):
            raise TypeError("intent must be ViewIntent")
        if self.intent not in DATASET_VIEW_INTENTS:
            raise ValueError(
                f"{self.intent.value} is document-fed and cannot be stored in ViewSpec"
            )
        bindings = tuple(self.source_bindings)
        if any(not isinstance(binding, SourceViewBinding) for binding in bindings):
            raise TypeError("source_bindings must contain SourceViewBinding values")
        sources = tuple(binding.source for binding in bindings)
        if len(set(sources)) != len(sources):
            raise ValueError("ViewSpec may bind each AxisSourceRef only once")
        object.__setattr__(
            self,
            "source_bindings",
            tuple(
                sorted(
                    bindings,
                    key=lambda binding: (
                        binding.source.kind,
                        "" if binding.source.axis_id is None else binding.source.axis_id.value,
                    ),
                )
            ),
        )
        if self.point_ordinals is not None:
            ordinals = tuple(
                _nonnegative(item, "point ordinal") for item in self.point_ordinals
            )
            if not ordinals or tuple(sorted(set(ordinals))) != ordinals:
                raise ValueError(
                    "point_ordinals must be non-empty, unique, and increasing"
                )
            object.__setattr__(self, "point_ordinals", ordinals)

    def binding(self, source: AxisSourceRef) -> SourceViewBinding:
        if not isinstance(source, AxisSourceRef):
            raise TypeError("source must be AxisSourceRef")
        for binding in self.source_bindings:
            if binding.source == source:
                return binding
        raise KeyError(source)


@dataclass(frozen=True)
class ViewPreferences:
    """Explicit choices constrained by a :class:`ViewContract`.

    Preferences select among already permitted presentation choices; they do
    not register reducers or create new axis capabilities.
    """

    repeat_binding: SourceViewBinding | None = None
    x_source: AxisSourceRef | None = None
    image_x_source: AxisSourceRef | None = None
    image_y_source: AxisSourceRef | None = None
    batch_sources: tuple[AxisSourceRef, ...] = ()
    facet_sources: tuple[AxisSourceRef, ...] = ()
    sample_sources: tuple[AxisSourceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.repeat_binding is not None and not isinstance(
            self.repeat_binding, SourceViewBinding
        ):
            raise TypeError("repeat_binding must be SourceViewBinding or None")
        for field in ("x_source", "image_x_source", "image_y_source"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, AxisSourceRef):
                raise TypeError(f"{field} must be AxisSourceRef or None")
        all_sources: list[AxisSourceRef] = [
            source
            for source in (self.x_source, self.image_x_source, self.image_y_source)
            if source is not None
        ]
        if self.repeat_binding is not None:
            all_sources.append(self.repeat_binding.source)
        for field in ("batch_sources", "facet_sources", "sample_sources"):
            values = tuple(getattr(self, field))
            if any(not isinstance(source, AxisSourceRef) for source in values):
                raise TypeError(f"{field} must contain AxisSourceRef values")
            if len(set(values)) != len(values):
                raise ValueError(f"{field} cannot contain duplicate sources")
            values = tuple(
                sorted(
                    values,
                    key=lambda source: (
                        source.kind,
                        "" if source.axis_id is None else source.axis_id.value,
                    ),
                )
            )
            object.__setattr__(self, field, values)
            all_sources.extend(values)
        if len(set(all_sources)) != len(all_sources):
            raise ValueError("a source cannot have multiple explicit preference roles")


@dataclass(frozen=True)
class DisplaySlot:
    binding_role: AxisViewRole
    preferred_axis_roles: tuple[AxisRoleId, ...]

    def __post_init__(self) -> None:
        if self.binding_role not in (
            AxisViewRole.X,
            AxisViewRole.IMAGE_X,
            AxisViewRole.IMAGE_Y,
        ):
            raise ValueError("display slot must be X, IMAGE_X, or IMAGE_Y")
        roles = tuple(self.preferred_axis_roles)
        if not roles or any(not isinstance(role, AxisRoleId) for role in roles):
            raise ValueError("display slot requires AxisRoleId preferences")
        object.__setattr__(self, "preferred_axis_roles", roles)


@dataclass(frozen=True)
class AxisRolePolicy:
    axis_role: AxisRoleId
    automatic_roles: tuple[AxisViewRole, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axis_role, AxisRoleId):
            raise TypeError("axis_role must be AxisRoleId")
        roles = tuple(self.automatic_roles)
        allowed = {
            AxisViewRole.SAMPLE,
            AxisViewRole.BATCH,
            AxisViewRole.FACET,
            AxisViewRole.SELECTED,
        }
        if any(role not in allowed for role in roles):
            raise ValueError("automatic role policy contains an unsupported binding")
        object.__setattr__(self, "automatic_roles", roles)


@dataclass(frozen=True)
class ViewContract:
    intent: ViewIntent
    display_slots: tuple[DisplaySlot, ...]
    role_policies: tuple[AxisRolePolicy, ...]
    reducible_axis_roles: tuple[AxisRoleId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.intent, ViewIntent):
            raise TypeError("intent must be ViewIntent")
        if self.intent not in DATASET_VIEW_INTENTS:
            raise ValueError("document-fed intents cannot use dataset ViewContract")
        slots = tuple(self.display_slots)
        policies = tuple(self.role_policies)
        reducible = tuple(self.reducible_axis_roles)
        if any(not isinstance(slot, DisplaySlot) for slot in slots):
            raise TypeError("display_slots must contain DisplaySlot values")
        if any(not isinstance(policy, AxisRolePolicy) for policy in policies):
            raise TypeError("role_policies must contain AxisRolePolicy values")
        if len({policy.axis_role for policy in policies}) != len(policies):
            raise ValueError("ViewContract may define each axis role only once")
        if any(not isinstance(role, AxisRoleId) for role in reducible):
            raise TypeError("reducible_axis_roles must contain AxisRoleId values")
        object.__setattr__(self, "display_slots", slots)
        object.__setattr__(self, "role_policies", policies)
        object.__setattr__(self, "reducible_axis_roles", reducible)

    def policy_for(self, role: AxisRoleId) -> AxisRolePolicy | None:
        for policy in self.role_policies:
            if policy.axis_role == role:
                return policy
        return None


class SuggestionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NEEDS_INPUT = "NEEDS_INPUT"


@dataclass(frozen=True)
class DecisionReason:
    code: str
    message: str
    source: AxisSourceRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "reason code"))
        object.__setattr__(self, "message", _text(self.message, "reason message"))
        if self.source is not None and not isinstance(self.source, AxisSourceRef):
            raise TypeError("reason source must be AxisSourceRef or None")


@dataclass(frozen=True)
class ViewAlternative:
    source: AxisSourceRef
    binding_role: AxisViewRole
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, AxisSourceRef):
            raise TypeError("alternative source must be AxisSourceRef")
        if not isinstance(self.binding_role, AxisViewRole):
            raise TypeError("alternative binding_role must be AxisViewRole")
        object.__setattr__(self, "label", _text(self.label, "alternative label"))


@dataclass(frozen=True)
class ViewSuggestion:
    spec: ViewSpec | None
    status: SuggestionStatus
    reasons: tuple[DecisionReason, ...] = ()
    alternatives: tuple[ViewAlternative, ...] = ()

    def __post_init__(self) -> None:
        if self.spec is not None and not isinstance(self.spec, ViewSpec):
            raise TypeError("spec must be ViewSpec or None")
        if not isinstance(self.status, SuggestionStatus):
            raise TypeError("status must be SuggestionStatus")
        reasons = tuple(self.reasons)
        alternatives = tuple(self.alternatives)
        if any(not isinstance(reason, DecisionReason) for reason in reasons):
            raise TypeError("reasons must contain DecisionReason values")
        if any(not isinstance(alternative, ViewAlternative) for alternative in alternatives):
            raise TypeError("alternatives must contain ViewAlternative values")
        if self.status is SuggestionStatus.NEEDS_INPUT and self.spec is not None:
            raise ValueError("NEEDS_INPUT suggestion cannot carry a resolved ViewSpec")
        if self.status is not SuggestionStatus.NEEDS_INPUT and self.spec is None:
            raise ValueError("resolved suggestion must carry a ViewSpec")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "alternatives", alternatives)


@dataclass(frozen=True, order=True)
class DatasetId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _text(self.value, "DatasetId"))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_id: DatasetId
    label: str
    schema_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        object.__setattr__(self, "label", _text(self.label, "dataset label"))
        object.__setattr__(
            self,
            "schema_fingerprint",
            _digest(self.schema_fingerprint, "schema_fingerprint"),
        )


@dataclass(frozen=True)
class FigureLayer:
    layer_id: str
    dataset_id: DatasetId
    view: ViewSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_id", _text(self.layer_id, "layer_id"))
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if not isinstance(self.view, ViewSpec):
            raise TypeError("view must be ViewSpec")


@dataclass(frozen=True)
class FigureSelection:
    """A named selection-library/controller input.

    Merely storing this value in :class:`FigureDocument` never applies it to a
    layer.  Point-row filtering is carried only by the layer's exact
    ``ViewSpec.point_ordinals``; tensor display selection remains surface-local
    state rather than a second ViewSpec authority.
    """

    selection_id: str
    dataset_id: DatasetId
    selection: Selection

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_id", _text(self.selection_id, "selection_id"))
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if not isinstance(self.selection, Selection):
            raise TypeError("selection must be zlc_data.Selection")


@dataclass(frozen=True)
class FigureDocument:
    document_id: str
    revision: int
    datasets: tuple[DatasetDescriptor, ...]
    layers: tuple[FigureLayer, ...]
    selections: tuple[FigureSelection, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _text(self.document_id, "document_id"))
        object.__setattr__(self, "revision", _nonnegative(self.revision, "document revision"))
        datasets = tuple(self.datasets)
        layers = tuple(self.layers)
        selections = tuple(self.selections)
        if not datasets or any(not isinstance(item, DatasetDescriptor) for item in datasets):
            raise ValueError("FigureDocument requires DatasetDescriptor values")
        if not layers or any(not isinstance(item, FigureLayer) for item in layers):
            raise ValueError("FigureDocument requires FigureLayer values")
        if any(not isinstance(item, FigureSelection) for item in selections):
            raise TypeError("selections must contain FigureSelection values")
        dataset_ids = tuple(item.dataset_id for item in datasets)
        layer_ids = tuple(item.layer_id for item in layers)
        selection_ids = tuple(item.selection_id for item in selections)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("FigureDocument dataset ids must be unique")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("FigureDocument layer ids must be unique")
        if len(set(selection_ids)) != len(selection_ids):
            raise ValueError("FigureDocument selection ids must be unique")
        descriptors = {item.dataset_id: item for item in datasets}
        for layer in layers:
            if layer.dataset_id not in descriptors:
                raise ValueError(f"layer {layer.layer_id!r} references an unknown dataset")
            if layer.view.schema_fingerprint != descriptors[layer.dataset_id].schema_fingerprint:
                raise ValueError(f"layer {layer.layer_id!r} view/schema fingerprint mismatch")
        if any(item.dataset_id not in descriptors for item in selections):
            raise ValueError("FigureSelection references an unknown dataset")
        object.__setattr__(
            self,
            "datasets",
            tuple(sorted(datasets, key=lambda item: item.dataset_id.value)),
        )
        object.__setattr__(self, "layers", layers)
        object.__setattr__(
            self,
            "selections",
            tuple(sorted(selections, key=lambda item: item.selection_id)),
        )

    def descriptor(self, dataset_id: DatasetId) -> DatasetDescriptor:
        for descriptor in self.datasets:
            if descriptor.dataset_id == dataset_id:
                return descriptor
        raise KeyError(dataset_id)


@dataclass(frozen=True)
class AxisAddress:
    source: AxisSourceRef
    axis_name: str
    axis_role: AxisRoleId
    index: int
    coordinate: Any

    def __post_init__(self) -> None:
        if not isinstance(self.source, AxisSourceRef):
            raise TypeError("address source must be AxisSourceRef")
        object.__setattr__(self, "axis_name", _text(self.axis_name, "address axis name"))
        if not isinstance(self.axis_role, AxisRoleId):
            raise TypeError("address axis_role must be AxisRoleId")
        object.__setattr__(self, "index", _nonnegative(self.index, "address index"))


@dataclass(frozen=True)
class EvaluatedAxis:
    source: AxisSourceRef
    name: str
    role: AxisRoleId
    unit: str | None
    indices: tuple[int, ...]
    coordinates: tuple[Any, ...]
    coordinate_frame: CoordinateFrameId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, AxisSourceRef):
            raise TypeError("evaluated source must be AxisSourceRef")
        object.__setattr__(self, "name", _text(self.name, "evaluated axis name"))
        if not isinstance(self.role, AxisRoleId):
            raise TypeError("evaluated axis role must be AxisRoleId")
        indices = tuple(_nonnegative(index, "axis index") for index in self.indices)
        coordinates = tuple(self.coordinates)
        if len(indices) != len(coordinates):
            raise ValueError("evaluated axis indices and coordinates must align")
        if self.unit is not None:
            _text(self.unit, "evaluated axis unit")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame,
            CoordinateFrameId,
        ):
            raise TypeError(
                "evaluated axis coordinate_frame must be CoordinateFrameId or None"
            )
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def axis_id(self) -> AxisId:
        """Return the transient axis identity used by gesture geometry.

        Persisted authority remains :attr:`source`. Tensor, point-coordinate,
        and grid sources reuse their declared ``AxisId``. The synthetic point
        ordinal has no persisted axis id, so it receives one private frontend
        id solely for numeric viewport queries; gesture binding resolves that
        query back to exact point ordinals before commit.
        """

        if self.source.axis_id is not None:
            return self.source.axis_id
        if self.source.kind == AxisSourceRef.POINT_ORDINAL:
            return _POINT_ORDINAL_INTERACTION_AXIS_ID
        raise AttributeError(f"{self.source.kind} is not an evaluated numeric axis")


@dataclass(frozen=True, eq=False)
class EvaluatedImage:
    x_axis: EvaluatedAxis
    y_axis: EvaluatedAxis
    values: np.ndarray
    validity: np.ndarray
    value_unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.x_axis, EvaluatedAxis) or not isinstance(self.y_axis, EvaluatedAxis):
            raise TypeError("image axes must be EvaluatedAxis values")
        values = _immutable_evaluated_array(self.values, ndim=2)
        validity = _immutable_evaluated_array(
            self.validity,
            ndim=2,
            boolean=True,
        )
        expected = (len(self.y_axis.indices), len(self.x_axis.indices))
        if values.shape != expected or validity.shape != expected:
            raise ValueError("image arrays do not match y/x axes")
        if self.value_unit is not None:
            object.__setattr__(
                self,
                "value_unit",
                _text(self.value_unit, "image value unit"),
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True, eq=False)
class EvaluatedCurve:
    x_axis: EvaluatedAxis
    value_unit: str | None
    values: np.ndarray
    validity: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.x_axis, EvaluatedAxis):
            raise TypeError("curve x_axis must be EvaluatedAxis")
        if self.value_unit is not None:
            object.__setattr__(
                self,
                "value_unit",
                _text(self.value_unit, "curve value unit"),
            )
        values = _immutable_evaluated_array(self.values, ndim=1)
        validity = _immutable_evaluated_array(
            self.validity,
            ndim=1,
            boolean=True,
        )
        if values.shape != (len(self.x_axis.indices),) or validity.shape != values.shape:
            raise ValueError("curve arrays do not match x axis")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True)
class SampleCoordinates:
    source: AxisSourceRef
    coordinates: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, AxisSourceRef):
            raise TypeError("sample source must be AxisSourceRef")
        object.__setattr__(self, "coordinates", tuple(self.coordinates))


@dataclass(frozen=True, eq=False)
class EvaluatedHistogram:
    samples: np.ndarray
    sample_coordinates: tuple[SampleCoordinates, ...]
    dropped_count: int
    value_unit: str | None = None

    def __post_init__(self) -> None:
        samples = _immutable_evaluated_array(self.samples, ndim=1)
        coordinates = tuple(self.sample_coordinates)
        if any(not isinstance(item, SampleCoordinates) for item in coordinates):
            raise TypeError("sample_coordinates must contain SampleCoordinates values")
        if any(len(item.coordinates) != len(samples) for item in coordinates):
            raise ValueError("sample coordinates must align with samples")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_coordinates", coordinates)
        object.__setattr__(self, "dropped_count", _nonnegative(self.dropped_count, "dropped_count"))
        if self.value_unit is not None:
            object.__setattr__(
                self,
                "value_unit",
                _text(self.value_unit, "histogram value unit"),
            )


@dataclass(frozen=True)
class EvaluatedMeter:
    value: bool | Number
    valid: bool
    value_unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, (bool, np.bool_)):
            raise TypeError("meter valid must be boolean")
        object.__setattr__(self, "valid", bool(self.valid))
        value = self.value.item() if isinstance(self.value, np.generic) else self.value
        if not isinstance(value, (bool, Number)):
            raise TypeError("meter value must be a numeric or boolean scalar")
        object.__setattr__(self, "value", value)
        if self.value_unit is not None:
            object.__setattr__(
                self,
                "value_unit",
                _text(self.value_unit, "meter value unit"),
            )


EvaluatedLayerData = EvaluatedImage | EvaluatedCurve | EvaluatedHistogram | EvaluatedMeter


@dataclass(frozen=True)
class ReductionResolution:
    sources: tuple[AxisSourceRef, ...]
    method: DisplayReductionMethod
    minimum_contributors: int
    maximum_contributors: int

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if not sources or any(
            not isinstance(source, AxisSourceRef) for source in sources
        ):
            raise ValueError("reduction resolution requires AxisSourceRef values")
        if not isinstance(self.method, DisplayReductionMethod):
            raise TypeError("reduction resolution method is invalid")
        low = _nonnegative(self.minimum_contributors, "minimum contributors")
        high = _nonnegative(self.maximum_contributors, "maximum contributors")
        if high < low:
            raise ValueError("maximum contributors cannot be below minimum")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "minimum_contributors", low)
        object.__setattr__(self, "maximum_contributors", high)


@dataclass(frozen=True)
class EvaluatedSeries:
    batch_address: tuple[AxisAddress, ...]
    data: EvaluatedLayerData
    reductions: tuple[ReductionResolution, ...] = ()

    def __post_init__(self) -> None:
        addresses = tuple(self.batch_address)
        reductions = tuple(self.reductions)
        if any(not isinstance(item, AxisAddress) for item in addresses):
            raise TypeError("batch_address must contain AxisAddress values")
        if not isinstance(self.data, (EvaluatedImage, EvaluatedCurve, EvaluatedHistogram, EvaluatedMeter)):
            raise TypeError("data must be an evaluated layer DTO")
        if any(not isinstance(item, ReductionResolution) for item in reductions):
            raise TypeError("reductions must contain ReductionResolution values")
        object.__setattr__(self, "batch_address", addresses)
        object.__setattr__(self, "reductions", reductions)


@dataclass(frozen=True)
class EvaluatedCell:
    facet_address: tuple[AxisAddress, ...]
    series: tuple[EvaluatedSeries, ...]

    def __post_init__(self) -> None:
        addresses = tuple(self.facet_address)
        series = tuple(self.series)
        if any(not isinstance(item, AxisAddress) for item in addresses):
            raise TypeError("facet_address must contain AxisAddress values")
        if not series or any(not isinstance(item, EvaluatedSeries) for item in series):
            raise ValueError("EvaluatedCell requires EvaluatedSeries values")
        object.__setattr__(self, "facet_address", addresses)
        object.__setattr__(self, "series", series)


@dataclass(frozen=True)
class AxisResolution:
    source: AxisSourceRef
    selector: str
    index: int
    coordinate: Any

    def __post_init__(self) -> None:
        if not isinstance(self.source, AxisSourceRef):
            raise TypeError("resolution source must be AxisSourceRef")
        object.__setattr__(self, "selector", _text(self.selector, "selector"))
        object.__setattr__(self, "index", _nonnegative(self.index, "resolution index"))


@dataclass(frozen=True)
class EvaluatedInput:
    dataset_id: DatasetId
    ref: DatasetRevisionRef

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if not isinstance(self.ref, DatasetRevisionRef):
            raise TypeError("ref must be DatasetRevisionRef")


@dataclass(frozen=True)
class EvaluatedLayer:
    layer_id: str
    dataset_id: DatasetId
    cells: tuple[EvaluatedCell, ...]
    resolutions: tuple[AxisResolution, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_id", _text(self.layer_id, "layer_id"))
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        cells = tuple(self.cells)
        resolutions = tuple(self.resolutions)
        if not cells or any(not isinstance(item, EvaluatedCell) for item in cells):
            raise ValueError("EvaluatedLayer requires EvaluatedCell values")
        if any(not isinstance(item, AxisResolution) for item in resolutions):
            raise TypeError("resolutions must contain AxisResolution values")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "resolutions", resolutions)


@dataclass(frozen=True)
class EvaluatedFigureData:
    document_id: str
    document_revision: int
    inputs: tuple[EvaluatedInput, ...]
    layers: tuple[EvaluatedLayer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _text(self.document_id, "document_id"))
        object.__setattr__(
            self, "document_revision", _nonnegative(self.document_revision, "document revision")
        )
        inputs = tuple(self.inputs)
        layers = tuple(self.layers)
        if not inputs or any(not isinstance(item, EvaluatedInput) for item in inputs):
            raise ValueError("EvaluatedFigureData requires EvaluatedInput values")
        if not layers or any(not isinstance(item, EvaluatedLayer) for item in layers):
            raise ValueError("EvaluatedFigureData requires EvaluatedLayer values")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True, slots=True)
class EvaluatedProjectionIdentity:
    """Exact identity of one evaluated series projection.

    A dataset revision identifies the frozen input, not one of the potentially
    many facet cells and series evaluated from it.  Values, validity, and
    derived display histograms may therefore cache only against this complete
    typed identity.  The evaluated data DTOs are immutable and identity-equal,
    so a new evaluation is deliberately a cache miss even when its numbers
    happen to compare equal.
    """

    document_id: str
    document_revision: int
    evaluated_input: EvaluatedInput
    layer_id: str
    resolutions: tuple[AxisResolution, ...]
    facet_address: tuple[AxisAddress, ...]
    batch_address: tuple[AxisAddress, ...]
    data: EvaluatedLayerData

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _text(self.document_id, "projection document_id"),
        )
        object.__setattr__(
            self,
            "document_revision",
            _nonnegative(self.document_revision, "projection document revision"),
        )
        if not isinstance(self.evaluated_input, EvaluatedInput):
            raise TypeError("projection evaluated_input must be EvaluatedInput")
        object.__setattr__(
            self,
            "layer_id",
            _text(self.layer_id, "projection layer_id"),
        )
        resolutions = tuple(self.resolutions)
        facet_address = tuple(self.facet_address)
        batch_address = tuple(self.batch_address)
        if any(not isinstance(item, AxisResolution) for item in resolutions):
            raise TypeError("projection resolutions must contain AxisResolution values")
        if any(not isinstance(item, AxisAddress) for item in facet_address):
            raise TypeError("projection facet_address must contain AxisAddress values")
        if any(not isinstance(item, AxisAddress) for item in batch_address):
            raise TypeError("projection batch_address must contain AxisAddress values")
        if not isinstance(
            self.data,
            (EvaluatedImage, EvaluatedCurve, EvaluatedHistogram, EvaluatedMeter),
        ):
            raise TypeError("projection data must be an evaluated layer DTO")
        object.__setattr__(self, "resolutions", resolutions)
        object.__setattr__(self, "facet_address", facet_address)
        object.__setattr__(self, "batch_address", batch_address)


__all__ = [
    "DATASET_VIEW_INTENTS",
    "AxisAddress",
    "AxisResolution",
    "AxisRolePolicy",
    "SourceViewBinding",
    "AxisViewRole",
    "DatasetDescriptor",
    "DatasetId",
    "DecisionReason",
    "DisplayReduction",
    "DisplayReductionMethod",
    "DisplaySelector",
    "DisplaySlot",
    "EvaluatedAxis",
    "EvaluatedCell",
    "EvaluatedCurve",
    "EvaluatedFigureData",
    "EvaluatedHistogram",
    "EvaluatedImage",
    "EvaluatedInput",
    "EvaluatedLayer",
    "EvaluatedLayerData",
    "EvaluatedMeter",
    "EvaluatedProjectionIdentity",
    "EvaluatedSeries",
    "FigureDocument",
    "FigureLayer",
    "FigureSelection",
    "FixedIndex",
    "LatestNonempty",
    "ReductionResolution",
    "SampleCoordinates",
    "SuggestionStatus",
    "ViewAlternative",
    "ViewContract",
    "ViewIntent",
    "ViewPreferences",
    "ViewSpec",
]
