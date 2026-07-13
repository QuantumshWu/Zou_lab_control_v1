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

from zlc_data import AxisId, AxisRoleId, DatasetRevisionRef, Selection
from zlc_storage.canonical import (
    canonical_text as _text,
    nonnegative_integer as _nonnegative,
    sha256_text as _digest,
)


def readonly_array(value: object, *, ndim: int | None = None) -> np.ndarray:
    """Return an owned read-only array for transient evaluator DTOs."""

    result = np.array(value, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"array must have rank {ndim}, got {result.ndim}")
    if result.dtype.kind not in "biufc":
        raise TypeError("evaluated arrays must have a numeric or boolean dtype")
    result.setflags(write=False)
    return result


class ViewIntent(str, Enum):
    IMAGE = "IMAGE"
    CURVE = "CURVE"
    HISTOGRAM = "HISTOGRAM"
    METER = "METER"


class AxisViewRole(str, Enum):
    X = "X"
    IMAGE_X = "IMAGE_X"
    IMAGE_Y = "IMAGE_Y"
    SAMPLE = "SAMPLE"
    BATCH = "BATCH"
    FACET = "FACET"
    SLIDER = "SLIDER"
    SELECTED = "SELECTED"
    REDUCED = "REDUCED"


class DisplayReductionMethod(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"


class RepeatViewMode(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"
    LATEST = "LATEST"
    BATCH = "BATCH"
    SAMPLE = "SAMPLE"


@dataclass(frozen=True)
class FixedIndex:
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _nonnegative(self.index, "selector index"))


@dataclass(frozen=True)
class LatestNonempty:
    """Resolve one global latest non-empty index for the complete layer."""


DisplaySelector = FixedIndex | LatestNonempty


@dataclass(frozen=True)
class DisplayReduction:
    method: DisplayReductionMethod

    def __post_init__(self) -> None:
        if not isinstance(self.method, DisplayReductionMethod):
            raise TypeError("display reduction method must be DisplayReductionMethod")


@dataclass(frozen=True)
class AxisViewBinding:
    axis_id: AxisId
    role: AxisViewRole
    selector: DisplaySelector | None = None
    reduction: DisplayReduction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        if not isinstance(self.role, AxisViewRole):
            raise TypeError("role must be AxisViewRole")
        if self.role in (AxisViewRole.SELECTED, AxisViewRole.SLIDER):
            if not isinstance(self.selector, (FixedIndex, LatestNonempty)):
                raise ValueError(f"{self.role.value} binding requires a display selector")
            if self.reduction is not None:
                raise ValueError("selected/slider binding cannot carry a reduction")
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
    axis_bindings: tuple[AxisViewBinding, ...]
    display_selections: tuple[Selection, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_fingerprint",
            _digest(self.schema_fingerprint, "schema_fingerprint"),
        )
        if not isinstance(self.intent, ViewIntent):
            raise TypeError("intent must be ViewIntent")
        bindings = tuple(self.axis_bindings)
        if any(not isinstance(binding, AxisViewBinding) for binding in bindings):
            raise TypeError("axis_bindings must contain AxisViewBinding values")
        ids = tuple(binding.axis_id for binding in bindings)
        if len(set(ids)) != len(ids):
            raise ValueError("ViewSpec may bind each AxisId only once")
        object.__setattr__(
            self,
            "axis_bindings",
            tuple(sorted(bindings, key=lambda binding: binding.axis_id.value)),
        )
        selections = tuple(self.display_selections)
        if any(not isinstance(selection, Selection) for selection in selections):
            raise TypeError("display_selections must contain zlc_data.Selection values")
        selected_ids = tuple(
            term.axis_id for selection in selections for term in selection.terms
        )
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("display selections may constrain each AxisId only once")
        object.__setattr__(
            self,
            "display_selections",
            tuple(
                sorted(
                    selections,
                    key=lambda selection: tuple(term.axis_id.value for term in selection.terms),
                )
            ),
        )

    def binding(self, axis_id: AxisId) -> AxisViewBinding:
        for binding in self.axis_bindings:
            if binding.axis_id == axis_id:
                return binding
        raise KeyError(axis_id)


@dataclass(frozen=True)
class ViewPreferences:
    """Explicit choices constrained by a :class:`ViewContract`.

    Preferences select among already permitted presentation choices; they do
    not register reducers or create new axis capabilities.
    """

    repeat_mode: RepeatViewMode | None = None
    x_axis_id: AxisId | None = None
    image_x_axis_id: AxisId | None = None
    image_y_axis_id: AxisId | None = None
    batch_axis_ids: tuple[AxisId, ...] = ()
    facet_axis_ids: tuple[AxisId, ...] = ()
    sample_axis_ids: tuple[AxisId, ...] = ()

    def __post_init__(self) -> None:
        if self.repeat_mode is not None and not isinstance(self.repeat_mode, RepeatViewMode):
            raise TypeError("repeat_mode must be RepeatViewMode or None")
        for field in ("x_axis_id", "image_x_axis_id", "image_y_axis_id"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, AxisId):
                raise TypeError(f"{field} must be AxisId or None")
        all_ids: list[AxisId] = []
        for field in ("batch_axis_ids", "facet_axis_ids", "sample_axis_ids"):
            values = tuple(getattr(self, field))
            if any(not isinstance(axis_id, AxisId) for axis_id in values):
                raise TypeError(f"{field} must contain AxisId values")
            if len(set(values)) != len(values):
                raise ValueError(f"{field} cannot contain duplicate AxisId values")
            values = tuple(sorted(values, key=lambda axis_id: axis_id.value))
            object.__setattr__(self, field, values)
            all_ids.extend(values)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("an axis cannot have multiple explicit preference roles")


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
            AxisViewRole.SLIDER,
        }
        if any(role not in allowed for role in roles):
            raise ValueError("automatic role policy contains an unsupported binding")
        object.__setattr__(self, "automatic_roles", roles)


@dataclass(frozen=True)
class ViewContract:
    intent: ViewIntent
    display_slots: tuple[DisplaySlot, ...]
    role_policies: tuple[AxisRolePolicy, ...]
    repeat_modes: tuple[RepeatViewMode, ...]
    default_repeat_mode: RepeatViewMode
    reducible_axis_roles: tuple[AxisRoleId, ...]
    maximum_batch_series: int
    maximum_facet_cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.intent, ViewIntent):
            raise TypeError("intent must be ViewIntent")
        slots = tuple(self.display_slots)
        policies = tuple(self.role_policies)
        repeat_modes = tuple(self.repeat_modes)
        reducible = tuple(self.reducible_axis_roles)
        if any(not isinstance(slot, DisplaySlot) for slot in slots):
            raise TypeError("display_slots must contain DisplaySlot values")
        if any(not isinstance(policy, AxisRolePolicy) for policy in policies):
            raise TypeError("role_policies must contain AxisRolePolicy values")
        if len({policy.axis_role for policy in policies}) != len(policies):
            raise ValueError("ViewContract may define each axis role only once")
        if not repeat_modes or any(not isinstance(mode, RepeatViewMode) for mode in repeat_modes):
            raise ValueError("repeat_modes must contain RepeatViewMode values")
        if self.default_repeat_mode not in repeat_modes:
            raise ValueError("default repeat mode must be allowed by the contract")
        if any(not isinstance(role, AxisRoleId) for role in reducible):
            raise TypeError("reducible_axis_roles must contain AxisRoleId values")
        object.__setattr__(
            self, "maximum_batch_series", _nonnegative(self.maximum_batch_series, "batch limit")
        )
        object.__setattr__(
            self, "maximum_facet_cells", _nonnegative(self.maximum_facet_cells, "facet limit")
        )
        object.__setattr__(self, "display_slots", slots)
        object.__setattr__(self, "role_policies", policies)
        object.__setattr__(self, "repeat_modes", repeat_modes)
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
    axis_id: AxisId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "reason code"))
        object.__setattr__(self, "message", _text(self.message, "reason message"))
        if self.axis_id is not None and not isinstance(self.axis_id, AxisId):
            raise TypeError("reason axis_id must be AxisId or None")


@dataclass(frozen=True)
class ViewAlternative:
    axis_id: AxisId
    binding_role: AxisViewRole
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("alternative axis_id must be AxisId")
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
    layer.  Presentation changes only when a controller copies the immutable
    ``Selection`` into that layer's ``ViewSpec.display_selections``.
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
    axis_id: AxisId
    index: int
    coordinate: Any

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("address axis_id must be AxisId")
        object.__setattr__(self, "index", _nonnegative(self.index, "address index"))


@dataclass(frozen=True)
class EvaluatedAxis:
    axis_id: AxisId
    name: str
    unit: str | None
    indices: tuple[int, ...]
    coordinates: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("evaluated axis_id must be AxisId")
        object.__setattr__(self, "name", _text(self.name, "evaluated axis name"))
        indices = tuple(_nonnegative(index, "axis index") for index in self.indices)
        coordinates = tuple(self.coordinates)
        if len(indices) != len(coordinates):
            raise ValueError("evaluated axis indices and coordinates must align")
        if self.unit is not None:
            _text(self.unit, "evaluated axis unit")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, eq=False)
class EvaluatedImage:
    x_axis: EvaluatedAxis
    y_axis: EvaluatedAxis
    values: np.ndarray
    validity: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.x_axis, EvaluatedAxis) or not isinstance(self.y_axis, EvaluatedAxis):
            raise TypeError("image axes must be EvaluatedAxis values")
        values = readonly_array(self.values, ndim=2)
        validity = readonly_array(self.validity, ndim=2).astype(bool, copy=False)
        validity.setflags(write=False)
        expected = (len(self.y_axis.indices), len(self.x_axis.indices))
        if values.shape != expected or validity.shape != expected:
            raise ValueError("image arrays do not match y/x axes")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True, eq=False)
class EvaluatedCurve:
    x_axis: EvaluatedAxis
    values: np.ndarray
    validity: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.x_axis, EvaluatedAxis):
            raise TypeError("curve x_axis must be EvaluatedAxis")
        values = readonly_array(self.values, ndim=1)
        validity = readonly_array(self.validity, ndim=1).astype(bool, copy=False)
        validity.setflags(write=False)
        if values.shape != (len(self.x_axis.indices),) or validity.shape != values.shape:
            raise ValueError("curve arrays do not match x axis")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True)
class SampleCoordinates:
    axis_id: AxisId
    coordinates: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("sample axis_id must be AxisId")
        object.__setattr__(self, "coordinates", tuple(self.coordinates))


@dataclass(frozen=True, eq=False)
class EvaluatedHistogram:
    samples: np.ndarray
    sample_coordinates: tuple[SampleCoordinates, ...]
    dropped_count: int

    def __post_init__(self) -> None:
        samples = readonly_array(self.samples, ndim=1)
        coordinates = tuple(self.sample_coordinates)
        if any(not isinstance(item, SampleCoordinates) for item in coordinates):
            raise TypeError("sample_coordinates must contain SampleCoordinates values")
        if any(len(item.coordinates) != len(samples) for item in coordinates):
            raise ValueError("sample coordinates must align with samples")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_coordinates", coordinates)
        object.__setattr__(self, "dropped_count", _nonnegative(self.dropped_count, "dropped_count"))


@dataclass(frozen=True)
class EvaluatedMeter:
    value: bool | Number
    valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.valid, (bool, np.bool_)):
            raise TypeError("meter valid must be boolean")
        object.__setattr__(self, "valid", bool(self.valid))
        value = self.value.item() if isinstance(self.value, np.generic) else self.value
        if not isinstance(value, (bool, Number)):
            raise TypeError("meter value must be a numeric or boolean scalar")
        object.__setattr__(self, "value", value)


EvaluatedLayerData = EvaluatedImage | EvaluatedCurve | EvaluatedHistogram | EvaluatedMeter


@dataclass(frozen=True)
class ReductionResolution:
    axis_ids: tuple[AxisId, ...]
    method: DisplayReductionMethod
    minimum_contributors: int
    maximum_contributors: int

    def __post_init__(self) -> None:
        ids = tuple(self.axis_ids)
        if not ids or any(not isinstance(axis_id, AxisId) for axis_id in ids):
            raise ValueError("reduction resolution requires AxisId values")
        if not isinstance(self.method, DisplayReductionMethod):
            raise TypeError("reduction resolution method is invalid")
        low = _nonnegative(self.minimum_contributors, "minimum contributors")
        high = _nonnegative(self.maximum_contributors, "maximum contributors")
        if high < low:
            raise ValueError("maximum contributors cannot be below minimum")
        object.__setattr__(self, "axis_ids", ids)
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
    axis_id: AxisId
    selector: str
    index: int
    coordinate: Any

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("resolution axis_id must be AxisId")
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


__all__ = [
    "AxisAddress",
    "AxisResolution",
    "AxisRolePolicy",
    "AxisViewBinding",
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
    "EvaluatedSeries",
    "FigureDocument",
    "FigureLayer",
    "FigureSelection",
    "FixedIndex",
    "LatestNonempty",
    "ReductionResolution",
    "RepeatViewMode",
    "SampleCoordinates",
    "SuggestionStatus",
    "ViewAlternative",
    "ViewContract",
    "ViewIntent",
    "ViewPreferences",
    "ViewSpec",
]
